"""MacroScout specialist agent for dollar-rate regime detection."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
import pandas as pd
import yfinance as yf

from agents.base_agent import BaseAgent
from shared.database import Database, MacroIndicatorRecord

LOGGER = logging.getLogger(__name__)
Trend = Literal["UP", "DOWN", "FLAT"]
MacroRegime = Literal["RISK_ON", "RISK_OFF"]


@dataclass(frozen=True)
class MacroSeriesSnapshot:
    """Current macro series values and trend context."""

    symbol: str
    closes: np.ndarray

    @property
    def current(self) -> float:
        latest = np.asarray(self.closes).reshape(-1)[-1]
        return float(latest)

    @property
    def prev(self) -> float:
        previous = np.asarray(self.closes).reshape(-1)[-2]
        return float(previous)

    @property
    def sma_5(self) -> float:
        return float(np.mean(self.closes[-5:]))

    @property
    def sma_20(self) -> float:
        return float(np.mean(self.closes[-20:]))

    @property
    def slope(self) -> float:
        return self.current - self.prev


class MacroScoutAgent(BaseAgent):
    """Monitors DXY and rates as macro gravity for risk assets."""

    ANALYSIS_INTERVAL_SECONDS = 300
    TIMEFRAME = "1d"
    LOOKBACK_WINDOW_DAYS = 30
    DXY_SYMBOL = "DX-Y.NYB"
    TNX_SYMBOL = "^TNX"
    REQUIRED_SYMBOLS = (DXY_SYMBOL, TNX_SYMBOL)
    COUNTRY = "us"

    def __init__(self) -> None:
        super().__init__(name="macro_scout")
        self._database = Database()
        self._analysis_task: asyncio.Task[None] | None = None
        self._pending_value_hunter_buy: bool = False
        self._last_warning_at: datetime | None = None

    @property
    def topics(self) -> list[str]:
        return ["agents.reports", "system.heartbeat"]

    async def handle_message(
        self,
        channel: str,
        payload: dict[str, Any],
    ) -> None:
        """Handle inter-agent updates and heartbeat trigger."""
        if channel == "agents.reports":
            await self._ingest_agent_report(payload)
            return
        if channel == "system.heartbeat":
            self.logger.info("Heartbeat received, triggering macro scan")
            await self._analyze_and_publish(trigger="heartbeat")
            return
        self.logger.debug("Ignored channel=%s payload=%s", channel, payload)

    async def run(self) -> None:
        """Run periodic scanner and shared-message listener."""
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
        """Track ValueHunter BUY alerts for divergence warning logic."""
        agent_name = str(payload.get("agent", "")).lower()
        signal = str(payload.get("signal", "")).upper()
        asset = str(payload.get("asset", ""))
        if agent_name != "value_hunter":
            return

        if signal == "BUY" and asset == "HG=F":
            self._pending_value_hunter_buy = True
            self.logger.info("Detected ValueHunter BUY on copper")

    async def _periodic_analysis_loop(self) -> None:
        """Run macro analysis every 5 minutes."""
        self.logger.info(
            "MacroScout periodic loop started (interval=%ss)",
            self.ANALYSIS_INTERVAL_SECONDS,
        )
        try:
            while True:
                await self._analyze_and_publish(trigger="scheduler")
                await asyncio.sleep(self.ANALYSIS_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            self.logger.info("MacroScout periodic loop cancelled")
            raise
        except Exception:
            self.logger.exception("MacroScout periodic loop failed")
            raise

    async def _analyze_and_publish(self, trigger: str) -> None:
        """Build macro regime report and optional ValueHunter warning."""
        try:
            snapshots = await self._fetch_macro_snapshots()
            if snapshots is None:
                self.logger.warning(
                    "Skipping macro report; missing DXY/TNX data",
                )
                return

            dxy = snapshots[self.DXY_SYMBOL]
            tnx = snapshots[self.TNX_SYMBOL]
            dxy_trend = self._compute_trend(dxy)
            yields_trend = self._compute_trend(tnx)
            macro_regime = self._compute_regime(dxy_trend, yields_trend)
            confidence = self._compute_confidence(dxy, tnx, macro_regime)

            await self._persist_macro_indicators(dxy, tnx)

            report: dict[str, Any] = {
                "agent": self.name,
                "signal": "INFO",
                "macro_regime": macro_regime,
                "dxy_trend": dxy_trend,
                "yields_trend": yields_trend,
                "confidence": confidence,
                "summary": (
                    f"DXY={dxy.current:.2f} (SMA20={dxy.sma_20:.2f}) "
                    f"TNX={tnx.current:.2f} (SMA20={tnx.sma_20:.2f})"
                ),
                "trigger": trigger,
                "timestamp": datetime.now(UTC).isoformat(),
            }
            await self.bus.publish(self.REPORT_CHANNEL, report)
            self.logger.info(
                "Macro report published regime=%s dxy=%s tnx=%s",
                macro_regime,
                dxy_trend,
                yields_trend,
            )

            if self._should_emit_warning(
                macro_regime=macro_regime,
                dxy_trend=dxy_trend,
            ):
                await self._publish_divergence_warning(
                    dxy=dxy,
                    tnx=tnx,
                    confidence=confidence,
                )
                self._pending_value_hunter_buy = False
        except Exception:
            self.logger.exception("MacroScout analysis cycle failed")

    async def _fetch_macro_snapshots(
        self,
    ) -> dict[str, MacroSeriesSnapshot] | None:
        """Fetch DXY and TNX market data with resilient error handling."""
        snapshots: dict[str, MacroSeriesSnapshot] = {}
        for symbol in self.REQUIRED_SYMBOLS:
            try:
                frame = await asyncio.to_thread(
                    yf.download,
                    symbol,
                    period="60d",
                    interval=self.TIMEFRAME,
                    progress=False,
                    auto_adjust=False,
                    threads=False,
                )
            except Exception:
                self.logger.exception(
                    "Download failed for macro symbol %s",
                    symbol,
                )
                continue

            if frame.empty:
                self.logger.warning(
                    "No data for %s (market closed/holiday/provider lag)",
                    symbol,
                )
                continue

            frame = self._normalize_ohlcv_frame(frame)
            if frame.empty:
                self.logger.warning(
                    "Invalid OHLC structure for %s after normalization",
                    symbol,
                )
                continue

            closes = frame["Close"].dropna().tail(self.LOOKBACK_WINDOW_DAYS)
            close_values = np.asarray(closes.to_numpy(dtype=float)).reshape(-1)
            if close_values.size < 20:
                self.logger.warning(
                    "Not enough bars for %s (got=%s need=20)",
                    symbol,
                    close_values.size,
                )
                continue

            snapshots[symbol] = MacroSeriesSnapshot(
                symbol=symbol,
                closes=close_values,
            )

        if not set(self.REQUIRED_SYMBOLS).issubset(snapshots.keys()):
            return None
        return snapshots

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

    async def _persist_macro_indicators(
        self,
        dxy: MacroSeriesSnapshot,
        tnx: MacroSeriesSnapshot,
    ) -> None:
        """Store latest macro indicators into TimescaleDB hypertable."""
        now = datetime.now(UTC)
        indicators = [
            MacroIndicatorRecord(
                ts=now,
                indicator="DXY_SPOT",
                country=self.COUNTRY,
                value=dxy.current,
                unit="index_points",
                source="yfinance",
            ),
            MacroIndicatorRecord(
                ts=now,
                indicator="DXY_SMA5",
                country=self.COUNTRY,
                value=dxy.sma_5,
                unit="index_points",
                source="yfinance",
            ),
            MacroIndicatorRecord(
                ts=now,
                indicator="DXY_SMA20",
                country=self.COUNTRY,
                value=dxy.sma_20,
                unit="index_points",
                source="yfinance",
            ),
            MacroIndicatorRecord(
                ts=now,
                indicator="TNX_SPOT",
                country=self.COUNTRY,
                value=tnx.current,
                unit="yield_percent",
                source="yfinance",
            ),
            MacroIndicatorRecord(
                ts=now,
                indicator="TNX_SMA5",
                country=self.COUNTRY,
                value=tnx.sma_5,
                unit="yield_percent",
                source="yfinance",
            ),
            MacroIndicatorRecord(
                ts=now,
                indicator="TNX_SMA20",
                country=self.COUNTRY,
                value=tnx.sma_20,
                unit="yield_percent",
                source="yfinance",
            ),
        ]
        await self._database.upsert_macro_indicators(indicators)

    def _compute_trend(self, snapshot: MacroSeriesSnapshot) -> Trend:
        """Map price action against SMA20 and short-term slope."""
        if snapshot.current > snapshot.sma_20 and snapshot.slope > 0:
            return "UP"
        if snapshot.current < snapshot.sma_20 and snapshot.slope < 0:
            return "DOWN"
        return "FLAT"

    def _compute_regime(
        self,
        dxy_trend: Trend,
        yields_trend: Trend,
    ) -> MacroRegime:
        """Infer broad macro regime from dollar and yield direction."""
        if dxy_trend == "UP" or yields_trend == "UP":
            return "RISK_OFF"
        return "RISK_ON"

    def _compute_confidence(
        self,
        dxy: MacroSeriesSnapshot,
        tnx: MacroSeriesSnapshot,
        macro_regime: MacroRegime,
    ) -> float:
        """Estimate confidence from normalized distance to SMA20."""
        dxy_gap = abs(dxy.current - dxy.sma_20) / max(dxy.sma_20, 1e-8)
        tnx_gap = abs(tnx.current - tnx.sma_20) / max(abs(tnx.sma_20), 1e-8)
        raw = min(1.0, (dxy_gap + tnx_gap) * 8.0)
        if macro_regime == "RISK_OFF":
            return max(0.45, raw)
        return max(0.35, raw * 0.9)

    def _should_emit_warning(
        self,
        macro_regime: MacroRegime,
        dxy_trend: Trend,
    ) -> bool:
        """Emit warning when ValueHunter BUY conflicts with macro stress."""
        if not self._pending_value_hunter_buy:
            return False
        if macro_regime != "RISK_OFF":
            return False
        if dxy_trend != "UP":
            return False

        now = datetime.now(UTC)
        if self._last_warning_at is not None:
            delta_seconds = (now - self._last_warning_at).total_seconds()
            if delta_seconds < self.ANALYSIS_INTERVAL_SECONDS:
                return False
        self._last_warning_at = now
        return True

    async def _publish_divergence_warning(
        self,
        dxy: MacroSeriesSnapshot,
        tnx: MacroSeriesSnapshot,
        confidence: float,
    ) -> None:
        """Publish explicit warning against pro-cyclical commodity buy."""
        warning_payload = {
            "agent": self.name,
            "signal": "WARNING",
            "macro_regime": "RISK_OFF",
            "dxy_trend": "UP",
            "yields_trend": self._compute_trend(tnx),
            "confidence": min(1.0, max(0.5, confidence)),
            "reason": (
                "ValueHunter BUY en cobre bajo presión macro: "
                "DXY rompiendo al alza y tasas restrictivas."
            ),
            "summary": (
                f"Riesgo de divergencia: DXY={dxy.current:.2f} > "
                f"SMA20={dxy.sma_20:.2f}, TNX={tnx.current:.2f}."
            ),
            "timestamp": datetime.now(UTC).isoformat(),
        }
        await self.bus.publish(self.REPORT_CHANNEL, warning_payload)
        self.logger.warning(
            "Published macro divergence WARNING against copper BUY",
        )


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


async def main() -> None:
    configure_logging()
    agent = MacroScoutAgent()
    await agent.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        LOGGER.info("MacroScout interrupted by user")
