"""BlackSwan agent package."""

from typing import TYPE_CHECKING, Any

__all__ = ["BlackSwanAgent"]

if TYPE_CHECKING:
    from agents.black_swan.agent import BlackSwanAgent


def __getattr__(name: str) -> Any:
    if name == "BlackSwanAgent":
        from agents.black_swan.agent import BlackSwanAgent

        return BlackSwanAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
