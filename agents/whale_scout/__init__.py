"""WhaleScout agent package."""

from typing import TYPE_CHECKING, Any

__all__ = ["WhaleScoutAgent"]

if TYPE_CHECKING:
    from agents.whale_scout.agent import WhaleScoutAgent


def __getattr__(name: str) -> Any:
    if name == "WhaleScoutAgent":
        from agents.whale_scout.agent import WhaleScoutAgent

        return WhaleScoutAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
