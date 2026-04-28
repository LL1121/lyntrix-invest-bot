"""GeoOracle specialist agent for geopolitical market impact analysis."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from agents.base_agent import BaseAgent

LOGGER = logging.getLogger(__name__)
Sector = Literal["Energy", "Tech", "Commodities"]


@dataclass(frozen=True)
class NewsArticle:
    """Minimal financial-news entity used by GeoOracle."""

    title: str
    summary: str
    source: str
    published_at: datetime


@dataclass(frozen=True)
class GeopoliticalAnalysis:
    """Structured LLM analysis output for downstream consensus."""

    sentiment_score: float
    impacted_sectors: list[Sector]
    summary: str


class GeoOracleAgent(BaseAgent):
    """Agent focused on geopolitical and supply-risk sentiment."""

    ANALYSIS_INTERVAL_SECONDS = 300

    def __init__(self) -> None:
        super().__init__(name="geo_oracle")
        self._llm_provider = os.getenv("LLM_PROVIDER", "mock").lower()
        self._openai_api_key = os.getenv("OPENAI_API_KEY")
        self._anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self._news_api_key = os.getenv("NEWS_API_KEY")
        self._analysis_task: asyncio.Task[None] | None = None

    @property
    def topics(self) -> list[str]:
        return ["system.heartbeat"]

    async def handle_message(
        self,
        channel: str,
        payload: dict[str, Any],
    ) -> None:
        """React to inbound control messages."""
        if channel != "system.heartbeat":
            self.logger.debug("Ignored channel %s payload=%s", channel, payload)
            return

        self.logger.info("Heartbeat received, triggering immediate analysis")
        await self._analyze_and_publish(trigger="heartbeat")

    async def run(self) -> None:
        """Run both periodic analysis loop and heartbeat listener."""
        self._analysis_task = asyncio.create_task(self._periodic_analysis_loop())
        try:
            await super().run()
        finally:
            if self._analysis_task is not None:
                self._analysis_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._analysis_task

    async def _periodic_analysis_loop(self) -> None:
        """Continuously run analysis at fixed intervals."""
        self.logger.info(
            "GeoOracle periodic loop started (interval=%ss)",
            self.ANALYSIS_INTERVAL_SECONDS,
        )
        try:
            while True:
                await self._analyze_and_publish(trigger="scheduler")
                await asyncio.sleep(self.ANALYSIS_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            self.logger.info("GeoOracle periodic loop cancelled")
            raise
        except Exception:
            self.logger.exception("GeoOracle periodic loop failed")
            raise

    async def _analyze_and_publish(self, trigger: str) -> None:
        """Fetch context, run LLM analysis, and publish standardized report."""
        try:
            articles = await self._fetch_financial_news()
            analysis = await self._analyze_with_llm(articles)
            confidence = min(1.0, max(0.0, abs(analysis.sentiment_score)))
            signal = "risk_on" if analysis.sentiment_score >= 0 else "risk_off"
            report_payload: dict[str, Any] = {
                "agent": self.name,
                "signal": signal,
                "confidence": confidence,
                "sentiment_score": analysis.sentiment_score,
                "impacted_sectors": analysis.impacted_sectors,
                "summary": analysis.summary,
                "trigger": trigger,
                "analysis_date": "2026-04-28",
                "timestamp": datetime.now(UTC).isoformat(),
            }
            await self.bus.publish(self.REPORT_CHANNEL, report_payload)
            self.logger.info(
                "GeoOracle report published: sentiment=%.4f sectors=%s",
                analysis.sentiment_score,
                analysis.impacted_sectors,
            )
        except Exception:
            self.logger.exception("GeoOracle analysis cycle failed")

    async def _fetch_financial_news(self) -> list[NewsArticle]:
        """Mock async ingestion of financial and geopolitical headlines."""
        if self._news_api_key:
            self.logger.info(
                "NEWS_API_KEY detected, using mocked provider integration path",
            )
        else:
            self.logger.warning("NEWS_API_KEY is not set, using local mock headlines")

        await asyncio.sleep(0.05)
        now = datetime.now(UTC)
        return [
            NewsArticle(
                title="Freight bottlenecks pressure uranium routes in key chokepoints",
                summary=(
                    "Shipping insurers reprice risk amid regional tensions, "
                    "raising short-term nuclear fuel transport costs."
                ),
                source="mock-newswire",
                published_at=now,
            ),
            NewsArticle(
                title="Copper smelter maintenance narrows refined output guidance",
                summary=(
                    "Unexpected downtime in multiple facilities tightens global "
                    "inventories while demand from grid projects remains firm."
                ),
                source="mock-markets",
                published_at=now,
            ),
            NewsArticle(
                title="Policy split on AI capex rotates flows toward hard assets",
                summary=(
                    "Institutional desks report lower overweight in mega-cap tech "
                    "and increased allocation to commodities-linked producers."
                ),
                source="mock-macro",
                published_at=now,
            ),
        ]

    async def _analyze_with_llm(
        self,
        articles: list[NewsArticle],
    ) -> GeopoliticalAnalysis:
        """Asynchronous LLM placeholder for geopolitical impact scoring."""
        prompt = self._build_prompt(articles)
        self.logger.debug("GeoOracle prompt prepared: %s", prompt)

        if self._llm_provider == "openai" and self._openai_api_key:
            return await self._call_openai_placeholder(prompt, articles)
        if self._llm_provider == "anthropic" and self._anthropic_api_key:
            return await self._call_anthropic_placeholder(prompt, articles)

        self.logger.warning(
            "No valid LLM provider credentials configured, using heuristic fallback",
        )
        return await self._heuristic_analysis(articles)

    def _build_prompt(self, articles: list[NewsArticle]) -> str:
        """Build prompt focused on 2026-04-28 geopolitical context."""
        headlines = "\n".join(
            f"- {article.title}: {article.summary}" for article in articles
        )
        return (
            "Date: 2026-04-28.\n"
            "You are a geopolitical market analyst. Assess supply-chain risks and "
            "global tensions affecting Energy, Tech, and Commodities. "
            "Avoid AI bubble overexposure bias, account for oil upside.\n"
            "Return JSON with keys: sentiment_score (-1.0 to 1.0), "
            "impacted_sectors (subset of [Energy, Tech, Commodities]), summary.\n"
            f"News:\n{headlines}"
        )

    async def _call_openai_placeholder(
        self,
        prompt: str,
        articles: list[NewsArticle],
    ) -> GeopoliticalAnalysis:
        """Placeholder async call for future OpenAI integration."""
        _ = prompt
        await asyncio.sleep(0.05)
        return await self._heuristic_analysis(articles)

    async def _call_anthropic_placeholder(
        self,
        prompt: str,
        articles: list[NewsArticle],
    ) -> GeopoliticalAnalysis:
        """Placeholder async call for future Anthropic integration."""
        _ = prompt
        await asyncio.sleep(0.05)
        return await self._heuristic_analysis(articles)

    async def _heuristic_analysis(
        self,
        articles: list[NewsArticle],
    ) -> GeopoliticalAnalysis:
        """Fallback deterministic analyzer preserving async behavior."""
        await asyncio.sleep(0.01)
        merged_text = " ".join(
            f"{article.title} {article.summary}".lower() for article in articles
        )
        score = 0.0
        sectors: set[Sector] = set()

        if "uranium" in merged_text or "nuclear" in merged_text:
            score -= 0.25
            sectors.add("Energy")
        if "copper" in merged_text or "commodit" in merged_text:
            score += 0.20
            sectors.add("Commodities")
        if "ai" in merged_text or "tech" in merged_text:
            score -= 0.10
            sectors.add("Tech")
        if "oil" in merged_text:
            score += 0.15
            sectors.add("Energy")

        sentiment_score = max(-1.0, min(1.0, score))
        impacted_sectors = sorted(sectors) or ["Commodities"]
        summary = (
            "Mixed geopolitical regime: supply-chain friction increases "
            "Energy volatility, while constrained metals flow sustains "
            "Commodities resilience and keeps Tech risk premium elevated."
        )
        return GeopoliticalAnalysis(
            sentiment_score=sentiment_score,
            impacted_sectors=impacted_sectors,
            summary=summary,
        )
