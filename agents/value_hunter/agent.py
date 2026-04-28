"""ValueHunter specialist agent for commodity-energy dislocation signals."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
import os
from typing import Any, Literal

import numpy as np
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
        return float(self.closes[-1])

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
            frame = await asyncio.to_thread(
                yf.download,
                symbol,
                period="40d",
                interval=self.TIMEFRAME,
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            if frame.empty:
                self.logger.warning(
                    "No data for %s (holiday/market closed/provider issue)",
                    symbol,
                )
                continue

            last_window = frame.tail(self.LOOKBACK_WINDOW_DAYS).copy()
            closes = last_window["Close"].dropna().to_numpy(dtype=float)
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
            await self._persist_candles(symbol, last_window)

        required = {self.COPPER_SYMBOL, self.GOLD_SYMBOL, self.ENERGY_SYMBOL}
        if not required.issubset(snapshots.keys()):
            self.logger.warning(
                "Missing required instruments; expected=%s got=%s",
                sorted(required),
                sorted(snapshots.keys()),
            )
            return None
        return snapshots

    async def _persist_candles(self, symbol: str, frame: Any) -> None:
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
                        source="yfinance",
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
