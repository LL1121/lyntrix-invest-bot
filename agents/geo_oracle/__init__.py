"""GeoOracle agent package."""

from typing import TYPE_CHECKING, Any

__all__ = ["GeoOracleAgent"]

if TYPE_CHECKING:
    from agents.geo_oracle.agent import GeoOracleAgent


def __getattr__(name: str) -> Any:
    if name == "GeoOracleAgent":
        from agents.geo_oracle.agent import GeoOracleAgent

        return GeoOracleAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
