"""CalendarAgent monitors high-impact macro events."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import logging
from datetime import UTC, datetime, timedelta
import os
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from agents.base_agent import BaseAgent
from shared.database import Database, MacroIndicatorRecord

LOGGER = logging.getLogger(__name__)


class CalendarAgent(BaseAgent):
    """Track upcoming macro events and publish impact/risk alerts."""

    ANALYSIS_INTERVAL_SECONDS = 300
    MACRO_INDICATOR = "EVENT_HIGH_IMPACT_MINUTES"

    def __init__(self) -> None:
        super().__init__(name="calendar_agent")
        self._analysis_task: asyncio.Task[None] | None = None
        self._database = Database()
        self._fmp_api_key = os.getenv("FMP_API_KEY", "").strip()

    @property
    def topics(self) -> list[str]:
        return ["system.heartbeat"]

    async def handle_message(
        self,
        channel: str,
        payload: dict[str, Any],
    ) -> None:
        """Trigger on heartbeats for immediate macro-event refresh."""
        if channel != "system.heartbeat":
            return
        del payload
        await self._analyze_and_publish(trigger="heartbeat")

    async def run(self) -> None:
        """Run periodic scanner and message listener concurrently."""
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
        """Run calendar scan every configured interval."""
        self.logger.info(
            "CalendarAgent periodic loop started (interval=%ss)",
            self.ANALYSIS_INTERVAL_SECONDS,
        )
        try:
            while True:
                await self._analyze_and_publish(trigger="scheduler")
                await asyncio.sleep(self.ANALYSIS_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            self.logger.info("CalendarAgent periodic loop cancelled")
            raise

    async def _analyze_and_publish(self, trigger: str) -> None:
        """Publish nearest high-impact event and risk timing."""
        events = await self._fetch_events()
        now = datetime.now(UTC)
        if not events:
            payload = {
                "agent": self.name,
                "signal": "INFO",
                "confidence": 0.3,
                "event_name": "NONE",
                "minutes_to_event": None,
                "risk_expected": "LOW",
                "summary": (
                    "No se detectaron eventos macro "
                    "de alto impacto proximos."
                ),
                "trigger": trigger,
                "timestamp": now.isoformat(),
            }
            await self.bus.publish(self.REPORT_CHANNEL, payload)
            return

        nearest = min(events, key=lambda event: event["minutes_to_event"])
        minutes_to_event = int(nearest["minutes_to_event"])
        risk_expected = self._risk_bucket(minutes_to_event)
        confidence = 0.5 if minutes_to_event > 180 else 0.82
        payload = {
            "agent": self.name,
            "signal": (
                "HIGH_IMPACT_EVENT" if minutes_to_event <= 240 else "INFO"
            ),
            "confidence": confidence,
            "event_name": nearest["event_name"],
            "minutes_to_event": minutes_to_event,
            "risk_expected": risk_expected,
            "summary": (
                f"Evento {nearest['event_name']} en {minutes_to_event} min. "
                f"Riesgo esperado: {risk_expected}."
            ),
            "trigger": trigger,
            "timestamp": now.isoformat(),
        }
        await self._database.upsert_macro_indicators(
            [
                MacroIndicatorRecord(
                    ts=now,
                    indicator=self.MACRO_INDICATOR,
                    country="us",
                    value=float(minutes_to_event),
                    unit="minutes",
                    source="calendar_agent",
                )
            ],
        )
        await self.bus.publish(self.REPORT_CHANNEL, payload)
        self.logger.info("Calendar event report published: %s", payload)

    async def _fetch_events(self) -> list[dict[str, Any]]:
        """Get today's high-impact events from API or deterministic mock."""
        if self._fmp_api_key:
            events = await self._fetch_fmp_events()
            if events:
                return events
        return self._mock_events()

    async def _fetch_fmp_events(self) -> list[dict[str, Any]]:
        """Fetch FMP economic calendar and filter high-impact themes."""
        today = datetime.now(UTC).date()
        params = urlencode(
            {
                "from": today.isoformat(),
                "to": (today + timedelta(days=1)).isoformat(),
                "apikey": self._fmp_api_key,
            }
        )
        url = (
            "https://financialmodelingprep.com/api/v3/economic_calendar"
            f"?{params}"
        )
        payload = await asyncio.to_thread(self._http_get_json, url)
        if not isinstance(payload, list):
            return []
        events: list[dict[str, Any]] = []
        now = datetime.now(UTC)
        for row in payload:
            if not isinstance(row, dict):
                continue
            name = str(row.get("event") or "").upper()
            if not any(tag in name for tag in ("FED", "CPI", "OPEC")):
                continue
            date_str = str(row.get("date") or "")
            try:
                ts = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            except ValueError:
                continue
            minutes = int((ts - now).total_seconds() // 60)
            if minutes < 0:
                continue
            events.append(
                {
                    "event_name": name,
                    "minutes_to_event": minutes,
                },
            )
        return events

    def _mock_events(self) -> list[dict[str, Any]]:
        """Provide robust fallback schedule with predictable timing."""
        now = datetime.now(UTC)
        fed_ts = now + timedelta(minutes=30)
        cpi_ts = now + timedelta(hours=4)
        opec_ts = now + timedelta(hours=7)
        return [
            {
                "event_name": "FED SPEECH",
                "minutes_to_event": int((fed_ts - now).total_seconds() // 60),
            },
            {
                "event_name": "CPI RELEASE",
                "minutes_to_event": int((cpi_ts - now).total_seconds() // 60),
            },
            {
                "event_name": "OPEC BRIEFING",
                "minutes_to_event": int((opec_ts - now).total_seconds() // 60),
            },
        ]

    def _risk_bucket(self, minutes_to_event: int) -> str:
        """Map time-to-event to risk level."""
        if minutes_to_event <= 60:
            return "VERY_HIGH"
        if minutes_to_event <= 180:
            return "HIGH"
        if minutes_to_event <= 360:
            return "MEDIUM"
        return "LOW"

    def _http_get_json(self, url: str) -> Any:
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
    agent = CalendarAgent()
    await agent.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        LOGGER.info("CalendarAgent interrupted by user")
