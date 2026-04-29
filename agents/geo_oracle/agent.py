"""GeoOracle specialist agent for geopolitical market impact analysis."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from agents.base_agent import BaseAgent
import groq
from groq import AsyncGroq

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
        self._llm_provider = os.getenv("LLM_PROVIDER", "groq").lower()
        self._groq_api_key = os.getenv("GROQ_API_KEY")
        self._groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self._openai_api_key = os.getenv("OPENAI_API_KEY")
        self._anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self._news_api_key = os.getenv("NEWS_API_KEY")
        self._analysis_task: asyncio.Task[None] | None = None
        self.client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

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
            self.logger.debug(
                "Ignored channel %s payload=%s",
                channel,
                payload,
            )
            return

        self.logger.info("Heartbeat received, triggering immediate analysis")
        await self._analyze_and_publish(trigger="heartbeat")

    async def run(self) -> None:
        """Run both periodic analysis loop and heartbeat listener."""
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
        """Fetch latest geopolitical-financial news from NewsAPI."""
        if self._news_api_key:
            try:
                return await self._fetch_newsapi_articles()
            except Exception:
                self.logger.exception(
                    "NewsAPI request failed, using local fallback",
                )
        else:
            self.logger.warning(
                "NEWS_API_KEY is not set, using local mock headlines",
            )

        await asyncio.sleep(0.02)
        now = datetime.now(UTC)
        return [
            NewsArticle(
                title=(
                    "Freight bottlenecks pressure uranium routes "
                    "in key chokepoints"
                ),
                summary=(
                    "Shipping insurers reprice risk amid regional tensions, "
                    "raising short-term nuclear fuel transport costs."
                ),
                source="mock-newswire",
                published_at=now,
            ),
            NewsArticle(
                title=(
                    "Copper smelter maintenance narrows "
                    "refined output guidance"
                ),
                summary=(
                    "Unexpected downtime in multiple facilities tightens global "
                    "inventories while demand from grid projects remains firm."
                ),
                source="mock-markets",
                published_at=now,
            ),
            NewsArticle(
                title=(
                    "Policy split on AI capex rotates "
                    "flows toward hard assets"
                ),
                summary=(
                    "Institutional desks report lower overweight in "
                    "mega-cap tech and increased allocation to "
                    "commodities-linked producers."
                ),
                source="mock-macro",
                published_at=now,
            ),
        ]

    async def _fetch_newsapi_articles(self) -> list[NewsArticle]:
        """Pull macro/geopolitical commodity headlines from NewsAPI."""
        query = (
            "copper OR uranium OR oil OR nuclear OR hydroelectric OR "
            "geopolitics OR supply chain"
        )
        params = urlencode(
            {
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": "20",
                "apiKey": self._news_api_key,
            }
        )
        url = f"https://newsapi.org/v2/everything?{params}"
        response_data = await asyncio.to_thread(self._http_get_json, url)
        raw_articles = response_data.get("articles", [])
        if not isinstance(raw_articles, list) or not raw_articles:
            raise RuntimeError("NewsAPI returned no articles")

        articles: list[NewsArticle] = []
        for item in raw_articles:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            summary = str(item.get("description", "")).strip()
            if not title:
                continue
            source_obj = item.get("source", {})
            source = (
                str(source_obj.get("name", "newsapi"))
                if isinstance(source_obj, dict)
                else "newsapi"
            )
            published_raw = str(item.get("publishedAt", ""))
            published_at = datetime.now(UTC)
            if published_raw:
                try:
                    published_at = datetime.fromisoformat(
                        published_raw.replace("Z", "+00:00")
                    )
                except ValueError:
                    pass
            articles.append(
                NewsArticle(
                    title=title,
                    summary=summary or "No description provided.",
                    source=source,
                    published_at=published_at,
                )
            )
            if len(articles) >= 8:
                break

        if not articles:
            raise RuntimeError("NewsAPI returned no parsable articles")
        return articles

    async def _analyze_with_llm(
        self,
        articles: list[NewsArticle],
    ) -> GeopoliticalAnalysis:
        """Asynchronous LLM analysis with Groq as primary provider."""
        prompt = self._build_prompt(articles)
        self.logger.debug("GeoOracle prompt prepared: %s", prompt)

        if self._llm_provider == "groq" and self._groq_api_key:
            return await self._call_groq_llm(prompt, articles)
        if self._llm_provider == "openai" and self._openai_api_key:
            return await self._call_openai_placeholder(prompt, articles)
        if self._llm_provider == "anthropic" and self._anthropic_api_key:
            return await self._call_anthropic_placeholder(prompt, articles)

        self.logger.warning(
            (
                "No valid LLM provider credentials configured, "
                "using heuristic fallback"
            ),
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
            "Use current context: Brent crude near $111 and capital rotation "
            "toward commodities producers and hard assets.\n"
            "Return STRICT JSON with keys: "
            "sentiment_score (-1.0 to 1.0), "
            "impacted_sectors (subset of [Energy, Tech, Commodities]), summary.\n"
            f"News:\n{headlines}"
        )

    async def _call_groq_llm(
        self,
        prompt: str,
        articles: list[NewsArticle],
    ) -> GeopoliticalAnalysis:
        """Run Groq via official async SDK with retry on rate limits."""
        assert self._groq_api_key is not None
        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            try:
                completion = await self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    temperature=0.2,
                    max_tokens=1024,
                    stream=False,
                    response_format={"type": "json_object"},
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are GeoOracle for an investment swarm. "
                                "Date context: 2026-04-28. Brent crude is near "
                                "$111 and capital rotates toward commodities. "
                                "Output ONLY strict JSON with keys "
                                "sentiment_score, impacted_sectors, summary."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                content = completion.choices[0].message.content
                if content is None:
                    raise RuntimeError("Groq returned empty content")
                parsed = self._parse_llm_json(content)
                sentiment = float(parsed["sentiment_score"])
                sectors = parsed["impacted_sectors"]
                if not isinstance(sectors, list):
                    raise TypeError("impacted_sectors must be list")
                clean_sectors = [str(sector) for sector in sectors][:3]
                summary = str(parsed["summary"])
                return GeopoliticalAnalysis(
                    sentiment_score=max(-1.0, min(1.0, sentiment)),
                    impacted_sectors=clean_sectors or ["Commodities"],
                    summary=summary,
                )
            except groq.RateLimitError:
                if attempt >= max_attempts:
                    raise
                sleep_s = 2 ** attempt
                self.logger.warning(
                    "Groq rate limit hit (attempt %s/%s), retrying in %ss",
                    attempt,
                    max_attempts,
                    sleep_s,
                )
                await asyncio.sleep(sleep_s)
            except Exception:
                self.logger.exception("Groq request failed without fallback")
                raise

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
            f"{article.title} {article.summary}".lower()
            for article in articles
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

    def _http_get_json(self, url: str) -> dict[str, Any]:
        request = Request(url, method="GET")
        with urlopen(request, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))

    def _parse_llm_json(self, content: str) -> dict[str, Any]:
        """Parse JSON object from LLM content with minor normalization."""
        raw = content.strip()
        if not raw:
            raise ValueError("Groq returned empty text content")
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"\s*```$", "", raw)
            raw = raw.strip()
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            candidate = match.group(0)
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed

        preview = raw[:300].replace("\n", " ")
        raise ValueError(f"Invalid JSON response from Groq: {preview}")

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


async def main() -> None:
    configure_logging()
    agent = GeoOracleAgent()
    await agent.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        LOGGER.info("GeoOracle interrupted by user")
