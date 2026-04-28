"""BlackSwan agent for tail-risk and market stress detection."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
import yfinance as yf

from agents.base_agent import BaseAgent
from shared.database import Database, MacroIndicatorRecord

LOGGER = logging.getLogger(__name__)
RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


@dataclass(frozen=True)
class SeriesSnapshot:
    """Close-price series holder with convenience metrics."""

    symbol: str
    closes: np.ndarray

    @property
    def current(self) -> float:
        return float(self.closes[-1])

    @property
    def previous(self) -> float:
        return float(self.closes[-2])

    @property
    def daily_return(self) -> float:
        prev = max(self.previous, 1e-10)
        return (self.current / prev) - 1.0


class BlackSwanAgent(BaseAgent):
    """Monitors tail-risk regime and emits defensive commands."""

    ANALYSIS_INTERVAL_SECONDS = 300
    TIMEFRAME = "1d"
    STOP_LOSS_CHANNEL = "risk.commands"
    REPORT_CHANNEL = "agents.reports"
    COUNTRY = "us"

    VIX_SYMBOL = "^VIX"
    NASDAQ_SYMBOL = "^IXIC"
    AI_ETF_SYMBOL = "BOTZ"
    COPPER_SYMBOL = "HG=F"
    GOLD_SYMBOL = "GC=F"
    SPY_SYMBOL = "SPY"
    REQUIRED_SYMBOLS = (
        VIX_SYMBOL,
        NASDAQ_SYMBOL,
        AI_ETF_SYMBOL,
        COPPER_SYMBOL,
        GOLD_SYMBOL,
        SPY_SYMBOL,
    )

    def __init__(self) -> None:
        super().__init__(name="black_swan")
        self._database = Database()
        self._analysis_task: asyncio.Task[None] | None = None
        self._last_stop_loss_at: datetime | None = None

    @property
    def topics(self) -> list[str]:
        return ["system.heartbeat", "agents.reports"]

    async def handle_message(
        self,
        channel: str,
        payload: dict[str, Any],
    ) -> None:
        """Handle control messages and allow immediate scans."""
        if channel == "system.heartbeat":
            self.logger.info("Heartbeat received, triggering black-swan scan")
            await self._analyze_and_publish(trigger="heartbeat")
            return
        if channel == "agents.reports":
            self.logger.debug(
                "Observed inter-agent report: %s",
                payload.get("agent"),
            )
            return
        self.logger.debug("Ignored channel=%s", channel)

    async def run(self) -> None:
        """Run scheduler and bus listener concurrently."""
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

    async def _periodic_analysis_loop(self) -> None:
        """Execute tail-risk analysis every 5 minutes."""
        self.logger.info(
            "BlackSwan periodic loop started (interval=%ss)",
            self.ANALYSIS_INTERVAL_SECONDS,
        )
        try:
            while True:
                await self._analyze_and_publish(trigger="scheduler")
                await asyncio.sleep(self.ANALYSIS_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            self.logger.info("BlackSwan periodic loop cancelled")
            raise
        except Exception:
            self.logger.exception("BlackSwan periodic loop failed")
            raise

    async def _analyze_and_publish(self, trigger: str) -> None:
        """Fetch stress metrics and publish risk posture."""
        try:
            snapshots = await self._fetch_market_snapshots()
            if snapshots is None:
                self.logger.warning(
                    "Skipping BlackSwan report due to missing data",
                )
                return

            vix = snapshots[self.VIX_SYMBOL]
            nasdaq = snapshots[self.NASDAQ_SYMBOL]
            ai_etf = snapshots[self.AI_ETF_SYMBOL]
            copper = snapshots[self.COPPER_SYMBOL]
            gold = snapshots[self.GOLD_SYMBOL]
            spy = snapshots[self.SPY_SYMBOL]

            iv_hv_ratio = self._compute_iv_hv_ratio(vix=vix, spy=spy)
            bubble_detected = self._is_bubble_detected(
                nasdaq=nasdaq,
                ai_etf=ai_etf,
            )
            liquidity_crunch = self._is_liquidity_crunch(
                gold=gold,
                copper=copper,
                spy=spy,
            )
            risk_level, event = self._classify_risk(
                vix=vix,
                iv_hv_ratio=iv_hv_ratio,
                bubble_detected=bubble_detected,
                liquidity_crunch=liquidity_crunch,
            )

            await self._persist_vix(vix=vix, iv_hv_ratio=iv_hv_ratio)

            payload: dict[str, Any] = {
                "agent": self.name,
                "signal": "ALERT",
                "risk_level": risk_level,
                "is_bubble_detected": bubble_detected,
                "black_swan_event": event,
                "summary": (
                    f"VIX={vix.current:.2f}, IV/HV={iv_hv_ratio:.2f}, "
                    f"liq_crunch={liquidity_crunch}"
                ),
                "confidence": self._risk_confidence(
                    vix=vix.current,
                    iv_hv_ratio=iv_hv_ratio,
                    risk_level=risk_level,
                ),
                "trigger": trigger,
                "timestamp": datetime.now(UTC).isoformat(),
            }
            await self.bus.publish(self.REPORT_CHANNEL, payload)
            self.logger.info(
                "BlackSwan report published risk=%s bubble=%s",
                risk_level,
                bubble_detected,
            )

            if risk_level == "CRITICAL":
                await self._send_stop_loss_all(event)
        except Exception:
            self.logger.exception("BlackSwan analysis cycle failed")

    async def _fetch_market_snapshots(
        self,
    ) -> dict[str, SeriesSnapshot] | None:
        """Download required market series with robust fallback checks."""
        snapshots: dict[str, SeriesSnapshot] = {}
        for symbol in self.REQUIRED_SYMBOLS:
            try:
                frame = await asyncio.to_thread(
                    yf.download,
                    symbol,
                    period="3y",
                    interval=self.TIMEFRAME,
                    progress=False,
                    auto_adjust=False,
                    threads=False,
                )
            except Exception:
                self.logger.exception("Failed downloading symbol %s", symbol)
                continue

            if frame.empty:
                self.logger.warning(
                    "No data for %s (market closed/holiday/provider lag)",
                    symbol,
                )
                continue

            closes = frame["Close"].dropna().to_numpy(dtype=float)
            if closes.size < 30:
                self.logger.warning(
                    "Insufficient bars for %s (got=%s need=30)",
                    symbol,
                    closes.size,
                )
                continue

            snapshots[symbol] = SeriesSnapshot(symbol=symbol, closes=closes)

        if not set(self.REQUIRED_SYMBOLS).issubset(snapshots.keys()):
            return None
        return snapshots

    def _compute_iv_hv_ratio(
        self,
        vix: SeriesSnapshot,
        spy: SeriesSnapshot,
    ) -> float:
        """Compute implied-to-historical volatility ratio."""
        spy_returns = np.diff(np.log(spy.closes[-21:]))
        hv_annualized = float(np.std(spy_returns) * np.sqrt(252))
        implied_vol = max(vix.current, 0.0) / 100.0
        if hv_annualized <= 1e-8:
            return 1.0
        return implied_vol / hv_annualized

    def _is_bubble_detected(
        self,
        nasdaq: SeriesSnapshot,
        ai_etf: SeriesSnapshot,
    ) -> bool:
        """Detect bubble-like acceleration above historical z-threshold."""
        nasdaq_3m_ret = self._rolling_return(nasdaq.closes, days=63)
        ai_3m_ret = self._rolling_return(ai_etf.closes, days=63)
        nasdaq_z = self._return_zscore(nasdaq.closes, days=63)
        ai_z = self._return_zscore(ai_etf.closes, days=63)
        return (
            nasdaq_z > 3.0
            or ai_z > 3.0
            or (nasdaq_3m_ret > 0.25 and ai_3m_ret > 0.25 and ai_z > 2.0)
        )

    def _is_liquidity_crunch(
        self,
        gold: SeriesSnapshot,
        copper: SeriesSnapshot,
        spy: SeriesSnapshot,
    ) -> bool:
        """Detect synchronous downside across defensive and cyclical assets."""
        all_down_today = (
            gold.daily_return < -0.01
            and copper.daily_return < -0.01
            and spy.daily_return < -0.015
        )
        gold_5d = self._rolling_return(gold.closes, days=5)
        copper_5d = self._rolling_return(copper.closes, days=5)
        spy_5d = self._rolling_return(spy.closes, days=5)
        all_down_5d = gold_5d < -0.02 and copper_5d < -0.03 and spy_5d < -0.03
        return all_down_today or all_down_5d

    def _classify_risk(
        self,
        vix: SeriesSnapshot,
        iv_hv_ratio: float,
        bubble_detected: bool,
        liquidity_crunch: bool,
    ) -> tuple[RiskLevel, str]:
        """Classify stress level and produce technical threat description."""
        if liquidity_crunch and vix.current >= 30.0:
            return "CRITICAL", "Liquidity Crunch Multi-Asset Selloff"
        if bubble_detected and vix.current >= 24.0:
            return "HIGH", "VIX Spike + AI Overextension"
        if bubble_detected:
            return "MEDIUM", "BUBBLE_RISK: Overextended growth beta"
        if iv_hv_ratio >= 1.35 or vix.current >= 22.0:
            return "MEDIUM", "Elevated volatility premium"
        return "LOW", "No acute black swan signature detected"

    async def _persist_vix(
        self,
        vix: SeriesSnapshot,
        iv_hv_ratio: float,
    ) -> None:
        """Persist VIX and IV/HV ratio into macro_indicators hypertable."""
        now = datetime.now(UTC)
        records = [
            MacroIndicatorRecord(
                ts=now,
                indicator="VIX_SPOT",
                country=self.COUNTRY,
                value=vix.current,
                unit="index_points",
                source="yfinance",
            ),
            MacroIndicatorRecord(
                ts=now,
                indicator="IV_HV_RATIO",
                country=self.COUNTRY,
                value=iv_hv_ratio,
                unit="ratio",
                source="derived",
            ),
        ]
        await self._database.upsert_macro_indicators(records)

    async def _send_stop_loss_all(self, event: str) -> None:
        """Publish defensive command when risk is critical."""
        now = datetime.now(UTC)
        if self._last_stop_loss_at is not None:
            elapsed = (now - self._last_stop_loss_at).total_seconds()
            if elapsed < self.ANALYSIS_INTERVAL_SECONDS:
                return
        self._last_stop_loss_at = now

        command = {
            "action": "STOP_LOSS_ALL",
            "source_agent": self.name,
            "reason": event,
            "timestamp": now.isoformat(),
        }
        await self.bus.publish(self.STOP_LOSS_CHANNEL, command)
        self.logger.critical("STOP_LOSS_ALL command sent to risk bus")

    def _risk_confidence(
        self,
        vix: float,
        iv_hv_ratio: float,
        risk_level: RiskLevel,
    ) -> float:
        """Estimate confidence based on volatility stress intensity."""
        base = min(
            1.0,
            (max(vix - 15.0, 0.0) / 20.0) + ((iv_hv_ratio - 1.0) * 0.6),
        )
        if risk_level == "CRITICAL":
            return max(0.85, base)
        if risk_level == "HIGH":
            return max(0.70, base)
        if risk_level == "MEDIUM":
            return max(0.45, base * 0.9)
        return max(0.30, base * 0.6)

    def _rolling_return(self, closes: np.ndarray, days: int) -> float:
        """Compute discrete rolling return."""
        if closes.size <= days:
            return 0.0
        ref = max(float(closes[-days - 1]), 1e-10)
        return (float(closes[-1]) / ref) - 1.0

    def _return_zscore(self, closes: np.ndarray, days: int) -> float:
        """Compute z-score of the latest rolling return versus history."""
        if closes.size <= (days + 40):
            return 0.0

        rolling_returns: list[float] = []
        for idx in range(days, closes.size - 1):
            start = max(float(closes[idx - days]), 1e-10)
            end = float(closes[idx])
            rolling_returns.append((end / start) - 1.0)

        if len(rolling_returns) < 30:
            return 0.0

        arr = np.asarray(rolling_returns, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        if std <= 1e-10:
            return 0.0

        latest = self._rolling_return(closes, days=days)
        return (latest - mean) / std
