"""ValueHunter agent package."""

from typing import TYPE_CHECKING, Any

__all__ = ["ValueHunterAgent"]

if TYPE_CHECKING:
    from agents.value_hunter.agent import ValueHunterAgent


def __getattr__(name: str) -> Any:
    if name == "ValueHunterAgent":
        from agents.value_hunter.agent import ValueHunterAgent

        return ValueHunterAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
