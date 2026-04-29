"""MacroScout agent package."""

from typing import TYPE_CHECKING, Any

__all__ = ["MacroScoutAgent"]

if TYPE_CHECKING:
    from agents.macro_scout.agent import MacroScoutAgent


def __getattr__(name: str) -> Any:
    if name == "MacroScoutAgent":
        from agents.macro_scout.agent import MacroScoutAgent

        return MacroScoutAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
