"""WhaleScout agent for institutional accumulation detection."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import logging
from datetime import UTC, datetime
import os
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import yfinance as yf

from agents.base_agent import BaseAgent
from shared.database import Database, MacroIndicatorRecord

LOGGER = logging.getLogger(__name__)


class WhaleScoutAgent(BaseAgent):
    """Detects unusual open-interest growth under price compression."""

    ANALYSIS_INTERVAL_SECONDS = 300
    TIMEFRAME = "1d"
    LOOKBACK_DAYS = 40
    TARGETS = ("HG=F", "GC=F")

    def __init__(self) -> None:
        super().__init__(name="whale_scout")
        self._analysis_task: asyncio.Task[None] | None = None
        self._alpha_vantage_api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        self._database = Database()

    @property
    def topics(self) -> list[str]:
        return ["system.heartbeat"]

    async def handle_message(
        self,
        channel: str,
        payload: dict[str, object],
    ) -> None:
        """Run out-of-schedule scan on heartbeat events."""
        if channel != "system.heartbeat":
            return
        del payload
        await self._analyze_and_publish(trigger="heartbeat")

    async def run(self) -> None:
        """Run periodic and event-driven loops concurrently."""
        self._analysis_task = asyncio.create_task(self._periodic_loop())
        try:
            await super().run()
        finally:
            if self._analysis_task is not None:
                self._analysis_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._analysis_task
            await self._database.close()

    async def _periodic_loop(self) -> None:
        """Run accumulation detection every configured interval."""
        self.logger.info(
            "WhaleScout periodic loop started (interval=%ss)",
            self.ANALYSIS_INTERVAL_SECONDS,
        )
        try:
            while True:
                await self._analyze_and_publish(trigger="scheduler")
                await asyncio.sleep(self.ANALYSIS_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            self.logger.info("WhaleScout periodic loop cancelled")
            raise

    async def _analyze_and_publish(self, trigger: str) -> None:
        """Detect OI/price anomalies and publish institutional signal."""
        try:
            detections: list[dict[str, object]] = []
            for symbol in self.TARGETS:
                frame = await self._load_series(symbol)
                if frame.empty or len(frame) < 6:
                    continue
                await self._persist_whale_oi_history(symbol, frame)
                detection = self._evaluate_symbol(symbol, frame)
                if detection is not None:
                    detections.append(detection)

            if not detections:
                payload = {
                    "agent": self.name,
                    "signal": "INFO",
                    "confidence": 0.35,
                    "summary": (
                        "Sin acumulacion institucional clara en cobre/oro "
                        "segun OI y lateralidad de precio."
                    ),
                    "trigger": trigger,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
                await self.bus.publish(self.REPORT_CHANNEL, payload)
                return

            strongest = max(
                detections,
                key=lambda item: float(item.get("confidence", 0.0)),
            )
            strongest.update(
                {
                    "agent": self.name,
                    "signal": "INSTITUTIONAL_ACCUMULATION",
                    "trigger": trigger,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )
            await self.bus.publish(self.REPORT_CHANNEL, strongest)
            self.logger.info(
                "WhaleScout accumulation signal published: %s",
                strongest,
            )
        except Exception:
            self.logger.exception("WhaleScout analysis cycle failed")

    async def _persist_whale_oi_history(
        self,
        symbol: str,
        frame: pd.DataFrame,
    ) -> None:
        """Persist daily open-interest proxy for War Room dual-axis charts."""
        if frame.empty or "open_interest" not in frame.columns:
            return
        records: list[MacroIndicatorRecord] = []
        for ts, row in frame.iterrows():
            pd_ts = pd.Timestamp(ts)
            if pd_ts.tzinfo is None:
                pd_ts = pd_ts.tz_localize("UTC")
            else:
                pd_ts = pd_ts.tz_convert("UTC")
            day_start = pd_ts.normalize().to_pydatetime()
            try:
                oi_val = float(row["open_interest"])
            except (TypeError, ValueError):
                continue
            records.append(
                MacroIndicatorRecord(
                    ts=day_start,
                    indicator=f"WHALE_OI::{symbol}",
                    country="global",
                    value=oi_val,
                    unit="contracts_proxy",
                    source="whale_scout",
                ),
            )
        if records:
            await self._database.upsert_macro_indicators(records)

    async def _load_series(self, symbol: str) -> pd.DataFrame:
        """Load price + synthetic/real OI series."""
        frame = await asyncio.to_thread(
            yf.download,
            symbol,
            period="60d",
            interval=self.TIMEFRAME,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
        if frame.empty:
            return pd.DataFrame()

        normalized = frame.copy()
        if isinstance(normalized.columns, pd.MultiIndex):
            normalized.columns = normalized.columns.get_level_values(-1)
        if "Close" not in normalized.columns:
            return pd.DataFrame()
        normalized["Close"] = pd.to_numeric(
            normalized["Close"],
            errors="coerce",
        )
        normalized["Volume"] = pd.to_numeric(
            normalized.get("Volume", 0.0),
            errors="coerce",
        ).fillna(0.0)
        normalized = normalized.dropna(subset=["Close"]).tail(self.LOOKBACK_DAYS)
        normalized["open_interest"] = await self._load_open_interest(
            symbol,
            normalized,
        )
        return normalized.dropna(subset=["open_interest"])

    async def _load_open_interest(
        self,
        symbol: str,
        frame: pd.DataFrame,
    ) -> pd.Series:
        """Fetch or synthesize open-interest series."""
        if self._alpha_vantage_api_key:
            series = await self._fetch_alpha_open_interest(symbol)
            if series is not None and not series.empty:
                return series.reindex(frame.index).ffill().bfill()

        # Mock robusto: aproxima OI acumulando volumen suavizado.
        rolling = frame["Volume"].rolling(3, min_periods=1).mean()
        base = max(float(rolling.iloc[0]), 1.0)
        synthetic = 100000.0 + (rolling / base).cumsum() * 2000.0
        return synthetic.astype(float)

    async def _fetch_alpha_open_interest(
        self,
        symbol: str,
    ) -> pd.Series | None:
        """Attempt AlphaVantage call for OI-like proxy series."""
        params = urlencode(
            {
                "function": "OPEN_INTEREST",
                "symbol": symbol,
                "apikey": self._alpha_vantage_api_key,
            },
        )
        url = f"https://www.alphavantage.co/query?{params}"
        payload = await asyncio.to_thread(self._http_get_json, url)
        raw = payload.get("data")
        if not isinstance(raw, list):
            return None
        points: list[tuple[pd.Timestamp, float]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                ts = pd.to_datetime(item["date"], utc=True)
                oi = float(item["open_interest"])
            except (KeyError, TypeError, ValueError):
                continue
            points.append((ts, oi))
        if not points:
            return None
        idx = [point[0] for point in points]
        vals = [point[1] for point in points]
        return pd.Series(vals, index=idx).sort_index()

    def _evaluate_symbol(
        self,
        symbol: str,
        frame: pd.DataFrame,
    ) -> dict[str, object] | None:
        """Return anomaly payload when OI rises and price stays lateral."""
        close = frame["Close"].to_numpy(dtype=float)
        oi = frame["open_interest"].to_numpy(dtype=float)
        if close.size < 6 or oi.size < 6:
            return None
        oi_change = ((oi[-1] / max(oi[-2], 1e-8)) - 1.0) * 100.0
        price_window = close[-5:]
        lateral_range = (np.max(price_window) - np.min(price_window)) / max(
            np.mean(price_window),
            1e-8,
        )
        is_lateral = lateral_range <= 0.012
        if oi_change <= 5.0 or not is_lateral:
            return None

        confidence = min(0.95, 0.55 + (oi_change / 20.0))
        return {
            "asset": symbol,
            "confidence": round(confidence, 4),
            "open_interest_change_pct": round(oi_change, 2),
            "price_lateral_range_pct": round(lateral_range * 100.0, 2),
            "summary": (
                f"Open Interest +{oi_change:.2f}% en 1 dia con precio lateral "
                f"(rango {lateral_range * 100:.2f}%)."
            ),
            "reason": (
                "Flujo institucional en acumulacion "
                "bajo compresion de precio"
            ),
        }

    def _http_get_json(self, url: str) -> dict[str, object]:
        request = Request(url, method="GET")
        with urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


async def main() -> None:
    configure_logging()
    agent = WhaleScoutAgent()
    await agent.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        LOGGER.info("WhaleScout interrupted by user")
