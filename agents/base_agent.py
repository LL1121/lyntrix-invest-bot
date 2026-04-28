"""Base class contract for all swarm specialist agents."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

from shared.messaging import MessageBus

LOGGER = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for asynchronous event-driven agents."""

    REPORT_CHANNEL = "agents.reports"

    def __init__(self, name: str, bus: MessageBus | None = None) -> None:
        self.name = name
        self.bus = bus or MessageBus()
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        self._running = False

    @property
    @abstractmethod
    def topics(self) -> list[str]:
        """List of Redis channels this agent listens to."""

    @abstractmethod
    async def handle_message(
        self,
        channel: str,
        payload: dict[str, Any],
    ) -> None:
        """Process a single incoming event."""

    async def publish_report(
        self,
        signal: str,
        confidence: float,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Publish a normalized report consumed by the orchestrator."""
        report_payload: dict[str, Any] = {
            "agent": self.name,
            "signal": signal,
            "confidence": confidence,
            "details": details or {},
            "timestamp": datetime.now(UTC).isoformat(),
        }
        try:
            await self.bus.publish(self.REPORT_CHANNEL, report_payload)
            self.logger.info(
                "Report published: signal=%s confidence=%.4f",
                signal,
                confidence,
            )
        except Exception:
            self.logger.exception("Failed to publish report")
            raise

    async def run(self) -> None:
        """Run agent loop listening and dispatching subscribed topics."""
        if self._running:
            self.logger.warning("Agent already running")
            return

        self._running = True
        await self.bus.connect()
        pubsub = await self.bus.subscribe(self.topics)
        self.logger.info(
            "Agent started. Listening to topics: %s",
            ", ".join(self.topics),
        )

        try:
            async for message in self.bus.iter_messages(pubsub):
                await self.handle_message(
                    channel=message["channel"],
                    payload=message["payload"],
                )
        except asyncio.CancelledError:
            self.logger.info("Agent task cancelled")
            raise
        except Exception:
            self.logger.exception("Unhandled error in agent run loop")
            raise
        finally:
            self._running = False
            await pubsub.aclose()
            self.logger.info("Agent stopped")
