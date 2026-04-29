"""ValueHunter specialist agent for commodity-energy dislocation signals."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
import os
from typing import Any, Literal
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import yfinance as yf

from agents.base_agent import BaseAgent
from shared.database import CandleRecord, Database

LOGGER = logging.getLogger(__name__)
Signal = Literal["BUY", "HOLD", "SELL"]


@dataclass(frozen=True)
class InstrumentSnapshot:
    """Holds current and rolling history for a market instrument."""

    symbol: str
    closes: np.ndarray

    @property
    def current(self) -> float:
        latest = np.asarray(self.closes).reshape(-1)[-1]
        return float(latest)

    @property
    def mean_20d(self) -> float:
        return float(np.mean(self.closes))

    @property
    def std_20d(self) -> float:
        return float(np.std(self.closes))

    @property
    def z_score(self) -> float:
        std = self.std_20d
        if std <= 1e-12:
            return 0.0
        return float((self.current - self.mean_20d) / std)


class ValueHunterAgent(BaseAgent):
    """Detects undervaluation opportunities under geopolitical stress."""

    ANALYSIS_INTERVAL_SECONDS = 300
    TIMEFRAME = "1d"
    LOOKBACK_WINDOW_DAYS = 20
    COPPER_SYMBOL = "HG=F"
    GOLD_SYMBOL = "GC=F"
    ENERGY_SYMBOL = os.getenv("VALUE_HUNTER_ENERGY_ETF", "NLR")
    MARKET_SYMBOLS = (COPPER_SYMBOL, GOLD_SYMBOL, ENERGY_SYMBOL)

    def __init__(self) -> None:
        super().__init__(name="value_hunter")
        self._database = Database()
        self._alpha_vantage_api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        self._analysis_task: asyncio.Task[None] | None = None
        self._latest_geo_sentiment: float | None = None

    @property
    def topics(self) -> list[str]:
        return ["agents.reports", "system.heartbeat"]

    async def handle_message(
        self,
        channel: str,
        payload: dict[str, Any],
    ) -> None:
        """Handle inter-agent reports and heartbeat triggers."""
        if channel == "agents.reports":
            await self._ingest_agent_report(payload)
            return
        if channel == "system.heartbeat":
            self.logger.info(
                "Heartbeat received, triggering immediate valuation run",
            )
            await self._analyze_and_publish(trigger="heartbeat")
            return
        self.logger.debug(
            "Ignored message channel=%s payload=%s",
            channel,
            payload,
        )

    async def run(self) -> None:
        """Run scheduler and message-listener loops concurrently."""
        self._analysis_task = asyncio.create_task(
            self._periodic_analysis_loop(),
        )
        try:
            await super().run()
        finally:
            if self._analysis_task is not None:
                self._analysis_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._analysis_task
            await self._database.close()

    async def _ingest_agent_report(self, payload: dict[str, Any]) -> None:
        """Track GeoOracle sentiment from shared agent reports channel."""
        agent_name = str(payload.get("agent", "")).lower()
        if agent_name != "geo_oracle":
            return

        raw_sentiment = payload.get("sentiment_score")
        try:
            self._latest_geo_sentiment = float(raw_sentiment)
            self.logger.info(
                "Updated GeoOracle sentiment context: %.4f",
                self._latest_geo_sentiment,
            )
        except (TypeError, ValueError):
            self.logger.warning(
                "GeoOracle report missing numeric sentiment_score: %s",
                raw_sentiment,
            )

    async def _periodic_analysis_loop(self) -> None:
        """Execute valuation analysis every 5 minutes."""
        self.logger.info(
            "ValueHunter periodic loop started (interval=%ss)",
            self.ANALYSIS_INTERVAL_SECONDS,
        )
        try:
            while True:
                await self._analyze_and_publish(trigger="scheduler")
                await asyncio.sleep(self.ANALYSIS_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            self.logger.info("ValueHunter periodic loop cancelled")
            raise
        except Exception:
            self.logger.exception("ValueHunter periodic loop failed")
            raise

    async def _analyze_and_publish(self, trigger: str) -> None:
        """Compute dislocation signal and publish report."""
        try:
            snapshots = await self._fetch_snapshots()
            if snapshots is None:
                self.logger.warning(
                    "Skipping publish because no market data is available",
                )
                return

            copper = snapshots[self.COPPER_SYMBOL]
            gold = snapshots[self.GOLD_SYMBOL]
            energy = snapshots[self.ENERGY_SYMBOL]
            ratio = copper.current / gold.current
            ratio_series = copper.closes / gold.closes
            ratio_mean = float(np.mean(ratio_series))
            ratio_std = float(np.std(ratio_series))
            ratio_z = 0.0
            if ratio_std > 1e-12:
                ratio_z = float((ratio - ratio_mean) / ratio_std)

            signal, asset, reason, confidence = self._build_signal(
                copper=copper,
                gold=gold,
                energy=energy,
                ratio_z=ratio_z,
            )

            report = {
                "agent": self.name,
                "signal": signal,
                "asset": asset,
                "reason": reason,
                "confidence": confidence,
                "sentiment_score": ratio_z,
                "impacted_sectors": ["Commodities", "Energy"],
                "summary": (
                    f"Cu/Au z-score={ratio_z:.2f}, "
                    f"HG z-score={copper.z_score:.2f}, "
                    f"Energy z-score={energy.z_score:.2f}."
                ),
                "trigger": trigger,
                "timestamp": datetime.now(UTC).isoformat(),
            }
            await self.bus.publish(self.REPORT_CHANNEL, report)
            self.logger.info(
                "ValueHunter report published signal=%s asset=%s confidence=%.4f",
                signal,
                asset,
                confidence,
            )
        except Exception:
            self.logger.exception("ValueHunter analysis cycle failed")

    async def _fetch_snapshots(
        self,
    ) -> dict[str, InstrumentSnapshot] | None:
        """Fetch windows and persist candles for each configured symbol."""
        if self._alpha_vantage_api_key:
            self.logger.info(
                (
                    "ALPHAVANTAGE_API_KEY detected; "
                    "using yfinance feed fallback currently"
                ),
            )

        snapshots: dict[str, InstrumentSnapshot] = {}
        for symbol in self.MARKET_SYMBOLS:
            frame, source = await self._download_symbol_frame(symbol)
            frame = self._normalize_ohlcv_frame(frame)
            if frame.empty:
                self.logger.warning(
                    "No data for %s (holiday/market closed/provider issue)",
                    symbol,
                )
                continue

            last_window = frame.tail(self.LOOKBACK_WINDOW_DAYS).copy()
            closes = last_window["Close"].dropna().to_numpy(dtype=float)
            closes = np.asarray(closes).reshape(-1)
            if closes.size < self.LOOKBACK_WINDOW_DAYS:
                self.logger.warning(
                    "Insufficient bars for %s (got=%s need=%s)",
                    symbol,
                    closes.size,
                    self.LOOKBACK_WINDOW_DAYS,
                )
                continue

            snapshots[symbol] = InstrumentSnapshot(
                symbol=symbol,
                closes=closes,
            )
            await self._persist_candles(symbol, last_window, source)

        required = {self.COPPER_SYMBOL, self.GOLD_SYMBOL, self.ENERGY_SYMBOL}
        if not required.issubset(snapshots.keys()):
            self.logger.warning(
                "Missing required instruments; expected=%s got=%s",
                sorted(required),
                sorted(snapshots.keys()),
            )
            return None
        return snapshots

    async def _download_symbol_frame(
        self,
        symbol: str,
    ) -> tuple[pd.DataFrame, str]:
        """Download market data with selective provider routing."""
        if symbol == self.COPPER_SYMBOL and self._alpha_vantage_api_key:
            frame = await self._download_copper_alpha_vantage()
            if not frame.empty:
                return frame, "alphavantage"
            self.logger.warning(
                (
                    "AlphaVantage copper feed returned empty data, "
                    "fallback to yfinance"
                ),
            )

        frame = await asyncio.to_thread(
            yf.download,
            symbol,
            period="40d",
            interval=self.TIMEFRAME,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
        return frame, "yfinance"

    async def _download_copper_alpha_vantage(self) -> pd.DataFrame:
        """Fetch copper series from AlphaVantage daily time-series API."""
        params = urlencode(
            {
                "function": "TIME_SERIES_DAILY",
                "symbol": self.COPPER_SYMBOL,
                "outputsize": "compact",
                "apikey": self._alpha_vantage_api_key,
            }
        )
        url = f"https://www.alphavantage.co/query?{params}"
        payload = await asyncio.to_thread(self._http_get_json, url)
        series = payload.get("Time Series (Daily)", {})
        if not isinstance(series, dict) or not series:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for dt_str, row in series.items():
            if not isinstance(row, dict):
                continue
            try:
                rows.append(
                    {
                        "Date": pd.to_datetime(dt_str, utc=True),
                        "Open": float(row["1. open"]),
                        "High": float(row["2. high"]),
                        "Low": float(row["3. low"]),
                        "Close": float(row["4. close"]),
                        "Volume": float(row.get("5. volume", 0.0)),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue

        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows).set_index("Date").sort_index()
        return frame

    def _normalize_ohlcv_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Normalize provider output to flat OHLCV columns."""
        if frame.empty:
            return frame

        normalized = frame.copy()
        if isinstance(normalized.columns, pd.MultiIndex):
            if normalized.columns.nlevels >= 2:
                last_level = normalized.columns.get_level_values(-1)
                if {"Open", "High", "Low", "Close"}.issubset(set(last_level)):
                    normalized.columns = last_level
                else:
                    normalized.columns = normalized.columns.get_level_values(0)
            else:
                normalized.columns = normalized.columns.get_level_values(0)

        normalized.columns = [str(column) for column in normalized.columns]
        required = ["Open", "High", "Low", "Close"]
        if not all(column in normalized.columns for column in required):
            return pd.DataFrame()
        if "Volume" not in normalized.columns:
            normalized["Volume"] = np.nan
        normalized = normalized[["Open", "High", "Low", "Close", "Volume"]]
        for column in ["Open", "High", "Low", "Close", "Volume"]:
            normalized[column] = pd.to_numeric(
                normalized[column],
                errors="coerce",
            )
        normalized = normalized.dropna(subset=["Open", "High", "Low", "Close"])
        return normalized

    async def _persist_candles(
        self,
        symbol: str,
        frame: Any,
        source: str,
    ) -> None:
        """Store fetched OHLCV in TimescaleDB candles hypertable."""
        candles: list[CandleRecord] = []
        for ts, row in frame.iterrows():
            try:
                volume_value = row.get("Volume")
                volume = None
                if volume_value is not None and not np.isnan(volume_value):
                    volume = float(volume_value)
                candles.append(
                    CandleRecord(
                        ts=ts.to_pydatetime(),
                        symbol=symbol,
                        timeframe=self.TIMEFRAME,
                        open=float(row["Open"]),
                        high=float(row["High"]),
                        low=float(row["Low"]),
                        close=float(row["Close"]),
                        volume=volume,
                        source=source,
                    )
                )
            except (TypeError, ValueError, KeyError):
                self.logger.warning(
                    "Skipping malformed candle for %s ts=%s",
                    symbol,
                    ts,
                )

        await self._database.upsert_candles(candles)

    def _build_signal(
        self,
        copper: InstrumentSnapshot,
        gold: InstrumentSnapshot,
        energy: InstrumentSnapshot,
        ratio_z: float,
    ) -> tuple[Signal, str, str, float]:
        """Build actionable valuation signal from dislocation metrics."""
        geo_negative = (self._latest_geo_sentiment or 0.0) < 0.0
        copper_undervalued = ratio_z <= -2.0 or copper.z_score <= -2.0
        gold_overbought = gold.z_score >= 2.0
        energy_undervalued = energy.z_score <= -2.0

        if geo_negative and copper_undervalued:
            divergence_strength = max(abs(ratio_z), abs(copper.z_score))
            confidence = min(1.0, 0.55 + (divergence_strength / 6.0))
            return (
                "BUY",
                self.COPPER_SYMBOL,
                "Ratio Cu/Au en piso + Alerta Geopolítica",
                confidence,
            )

        if gold_overbought and not geo_negative:
            confidence = min(0.9, 0.45 + (abs(gold.z_score) / 8.0))
            return (
                "SELL",
                self.GOLD_SYMBOL,
                "Oro sobreextendido sin stress geopolítico activo",
                confidence,
            )

        if energy_undervalued:
            confidence = min(0.85, 0.40 + (abs(energy.z_score) / 8.0))
            return (
                "BUY",
                self.ENERGY_SYMBOL,
                "ETF de energía >2σ bajo media de 20d",
                confidence,
            )

        return (
            "HOLD",
            self.COPPER_SYMBOL,
            "Sin divergencia extrema confirmada por contexto geopolítico",
            0.35,
        )

    def _http_get_json(self, url: str) -> dict[str, Any]:
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
    agent = ValueHunterAgent()
    await agent.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        LOGGER.info("ValueHunter interrupted by user")
