"""Asynchronous Redis Pub/Sub messaging bus for swarm agents."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from redis.asyncio import Redis
from redis.asyncio.client import PubSub

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BusConfig:
    """Configuration for the Redis message bus."""

    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_db: int = int(os.getenv("REDIS_DB", "0"))
    redis_password: str | None = os.getenv("REDIS_PASSWORD")


class MessageBus:
    """Thin wrapper around Redis Pub/Sub with JSON payload support."""

    def __init__(self, config: BusConfig | None = None) -> None:
        self._config = config or BusConfig()
        self._redis: Redis | None = None

    async def connect(self) -> None:
        """Connect to Redis if no active client exists."""
        if self._redis is not None:
            return

        self._redis = Redis(
            host=self._config.redis_host,
            port=self._config.redis_port,
            db=self._config.redis_db,
            password=self._config.redis_password,
            decode_responses=True,
        )
        await self._redis.ping()
        LOGGER.info(
            "Connected to Redis bus at %s:%s/%s",
            self._config.redis_host,
            self._config.redis_port,
            self._config.redis_db,
        )

    async def close(self) -> None:
        """Close active Redis client connection."""
        if self._redis is None:
            return

        await self._redis.aclose()
        self._redis = None
        LOGGER.info("Redis bus connection closed")

    async def publish(self, channel: str, payload: dict[str, Any]) -> int:
        """Publish a JSON payload to a channel."""
        if self._redis is None:
            await self.connect()
        assert self._redis is not None

        serialized = json.dumps(payload, default=str)
        subscribers = await self._redis.publish(channel, serialized)
        return subscribers

    async def subscribe(self, channels: list[str]) -> PubSub:
        """Subscribe to channels and return an active PubSub instance."""
        if self._redis is None:
            await self.connect()
        assert self._redis is not None

        pubsub = self._redis.pubsub()
        await pubsub.subscribe(*channels)
        LOGGER.info("Subscribed to channels: %s", ", ".join(channels))
        return pubsub

    async def iter_messages(
        self,
        pubsub: PubSub,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield parsed JSON messages from a PubSub subscription."""
        async for raw_message in pubsub.listen():
            if raw_message.get("type") != "message":
                continue

            channel = str(raw_message.get("channel"))
            data = raw_message.get("data")
            if not isinstance(data, str):
                LOGGER.warning("Skipping non-string payload on %s", channel)
                continue

            try:
                parsed = json.loads(data)
            except json.JSONDecodeError as exc:
                LOGGER.exception(
                    "Invalid JSON payload on channel %s: %s", channel, exc
                )
                continue

            yield {"channel": channel, "payload": parsed}
