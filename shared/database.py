"""Database layer for TimescaleDB access and hypertable bootstrap."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatabaseConfig:
    """Connection settings for TimescaleDB."""

    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "5432"))
    user: str = os.getenv("DB_USER", "lyntrix_admin")
    password: str = os.getenv("DB_PASSWORD", "lyntrix_pass")
    database: str = os.getenv("DB_NAME", "lyntrix_invest")

    @property
    def dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


@dataclass(frozen=True)
class CandleRecord:
    """Normalized candle record for persistence."""

    ts: datetime
    symbol: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float | None
    source: str


@dataclass(frozen=True)
class MacroIndicatorRecord:
    """Normalized macro indicator record for persistence."""

    ts: datetime
    indicator: str
    country: str
    value: float
    unit: str | None
    source: str


@dataclass(frozen=True)
class FinalDecisionRecord:
    """Consensus decision snapshot persisted by orchestrator."""

    ts: datetime
    action: str
    consensus_score: float
    votes_for: list[str]
    latest_signals: dict[str, object]
    rationale: str
    source: str = "orchestrator"


@dataclass(frozen=True)
class ExecutedTradeRecord:
    """Executed paper-trade entry persisted by mock executor."""

    ts: datetime
    side: str
    symbol: str
    quantity: float
    price: float
    fee: float
    slippage: float
    notional: float
    realized_pnl: float
    source_signal: str


@dataclass(frozen=True)
class PortfolioHistoryRecord:
    """Point-in-time portfolio valuation persisted by mock executor."""

    ts: datetime
    cash_balance: float
    positions_value: float
    total_equity: float
    realized_pnl: float
    unrealized_pnl: float
    source: str = "mock_executor"


@dataclass(frozen=True)
class AgentReportRecord:
    """Raw report event persisted from Redis agent channel."""

    ts: datetime
    agent: str
    payload: dict[str, Any]
    channel: str = "agents.reports"


class Database:
    """Async SQLAlchemy manager for TimescaleDB."""

    def __init__(self, config: DatabaseConfig | None = None) -> None:
        self._config = config or DatabaseConfig()
        self._engine: AsyncEngine = create_async_engine(
            self._config.dsn,
            echo=False,
            future=True,
            pool_pre_ping=True,
        )
        self._session_factory: async_sessionmaker[
            AsyncSession
        ] = async_sessionmaker(self._engine, expire_on_commit=False)

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Session maker for repositories/services."""
        return self._session_factory

    async def close(self) -> None:
        """Dispose database engine resources."""
        await self._engine.dispose()
        LOGGER.info("TimescaleDB engine disposed")

    async def init_hypertables(self) -> None:
        """Create base tables and convert them to Timescale hypertables."""
        statements = [
            text("CREATE EXTENSION IF NOT EXISTS timescaledb;"),
            text(
                """
                CREATE TABLE IF NOT EXISTS candles (
                    ts TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    open DOUBLE PRECISION NOT NULL,
                    high DOUBLE PRECISION NOT NULL,
                    low DOUBLE PRECISION NOT NULL,
                    close DOUBLE PRECISION NOT NULL,
                    volume DOUBLE PRECISION,
                    source TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (ts, symbol, timeframe)
                );
                """
            ),
            text(
                """
                SELECT create_hypertable(
                    'candles',
                    'ts',
                    if_not_exists => TRUE,
                    migrate_data => TRUE
                );
                """
            ),
            text(
                """
                CREATE TABLE IF NOT EXISTS macro_indicators (
                    ts TIMESTAMPTZ NOT NULL,
                    indicator TEXT NOT NULL,
                    country TEXT NOT NULL DEFAULT 'global',
                    value DOUBLE PRECISION NOT NULL,
                    unit TEXT,
                    source TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (ts, indicator, country)
                );
                """
            ),
            text(
                """
                SELECT create_hypertable(
                    'macro_indicators',
                    'ts',
                    if_not_exists => TRUE,
                    migrate_data => TRUE
                );
                """
            ),
            text(
                """
                CREATE TABLE IF NOT EXISTS final_decisions (
                    ts TIMESTAMPTZ NOT NULL,
                    action TEXT NOT NULL,
                    consensus_score DOUBLE PRECISION NOT NULL,
                    votes_for JSONB NOT NULL,
                    latest_signals JSONB NOT NULL,
                    rationale TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'orchestrator',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (ts)
                );
                """
            ),
            text(
                """
                CREATE TABLE IF NOT EXISTS executed_trades (
                    id BIGSERIAL PRIMARY KEY,
                    ts TIMESTAMPTZ NOT NULL,
                    side TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    quantity DOUBLE PRECISION NOT NULL,
                    price DOUBLE PRECISION NOT NULL,
                    fee DOUBLE PRECISION NOT NULL,
                    slippage DOUBLE PRECISION NOT NULL,
                    notional DOUBLE PRECISION NOT NULL,
                    realized_pnl DOUBLE PRECISION NOT NULL,
                    source_signal TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            ),
            text(
                """
                CREATE TABLE IF NOT EXISTS portfolio_history (
                    ts TIMESTAMPTZ PRIMARY KEY,
                    cash_balance DOUBLE PRECISION NOT NULL,
                    positions_value DOUBLE PRECISION NOT NULL,
                    total_equity DOUBLE PRECISION NOT NULL,
                    realized_pnl DOUBLE PRECISION NOT NULL,
                    unrealized_pnl DOUBLE PRECISION NOT NULL,
                    source TEXT NOT NULL DEFAULT 'mock_executor',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            ),
            text(
                """
                CREATE TABLE IF NOT EXISTS agent_reports (
                    id BIGSERIAL PRIMARY KEY,
                    ts TIMESTAMPTZ NOT NULL,
                    agent TEXT NOT NULL,
                    channel TEXT NOT NULL DEFAULT 'agents.reports',
                    payload JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            ),
        ]

        try:
            async with self._engine.begin() as conn:
                for statement in statements:
                    await conn.execute(statement)
            LOGGER.info("TimescaleDB hypertables initialized successfully")
        except Exception:
            LOGGER.exception("Failed to initialize TimescaleDB hypertables")
            raise

    async def upsert_candles(self, candles: Sequence[CandleRecord]) -> None:
        """Persist candle rows in TimescaleDB with conflict-safe upsert."""
        if not candles:
            return

        statement = text(
            """
            INSERT INTO candles (
                ts,
                symbol,
                timeframe,
                open,
                high,
                low,
                close,
                volume,
                source
            )
            VALUES (
                :ts,
                :symbol,
                :timeframe,
                :open,
                :high,
                :low,
                :close,
                :volume,
                :source
            )
            ON CONFLICT (ts, symbol, timeframe) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                source = EXCLUDED.source;
            """
        )
        payload = [
            {
                "ts": row.ts,
                "symbol": row.symbol,
                "timeframe": row.timeframe,
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "close": row.close,
                "volume": row.volume,
                "source": row.source,
            }
            for row in candles
        ]

        try:
            async with self._engine.begin() as conn:
                await conn.execute(statement, payload)
            LOGGER.info("Upserted %s candle rows", len(candles))
        except Exception:
            LOGGER.exception("Failed to upsert candle rows")
            raise

    async def upsert_macro_indicators(
        self,
        indicators: Sequence[MacroIndicatorRecord],
    ) -> None:
        """Persist macro indicator rows with conflict-safe upsert."""
        if not indicators:
            return

        statement = text(
            """
            INSERT INTO macro_indicators (
                ts,
                indicator,
                country,
                value,
                unit,
                source
            )
            VALUES (
                :ts,
                :indicator,
                :country,
                :value,
                :unit,
                :source
            )
            ON CONFLICT (ts, indicator, country) DO UPDATE SET
                value = EXCLUDED.value,
                unit = EXCLUDED.unit,
                source = EXCLUDED.source;
            """
        )
        payload = [
            {
                "ts": row.ts,
                "indicator": row.indicator,
                "country": row.country,
                "value": row.value,
                "unit": row.unit,
                "source": row.source,
            }
            for row in indicators
        ]

        try:
            async with self._engine.begin() as conn:
                await conn.execute(statement, payload)
            LOGGER.info("Upserted %s macro indicator rows", len(indicators))
        except Exception:
            LOGGER.exception("Failed to upsert macro indicator rows")
            raise

    async def insert_final_decision(self, decision: FinalDecisionRecord) -> None:
        """Persist consensus output from orchestrator."""
        statement = text(
            """
            INSERT INTO final_decisions (
                ts,
                action,
                consensus_score,
                votes_for,
                latest_signals,
                rationale,
                source
            )
            VALUES (
                :ts,
                :action,
                :consensus_score,
                CAST(:votes_for AS JSONB),
                CAST(:latest_signals AS JSONB),
                :rationale,
                :source
            );
            """
        )
        payload = {
            "ts": decision.ts,
            "action": decision.action,
            "consensus_score": decision.consensus_score,
            "votes_for": json.dumps(decision.votes_for),
            "latest_signals": json.dumps(decision.latest_signals),
            "rationale": decision.rationale,
            "source": decision.source,
        }

        try:
            async with self._engine.begin() as conn:
                await conn.execute(statement, payload)
            LOGGER.info(
                "Persisted final decision action=%s score=%.4f",
                decision.action,
                decision.consensus_score,
            )
        except Exception:
            LOGGER.exception("Failed to persist final decision")
            raise

    async def get_latest_candle_close(
        self,
        symbol: str,
        timeframe: str = "1d",
    ) -> float | None:
        """Fetch the latest close price for symbol/timeframe from candles."""
        statement = text(
            """
            SELECT close
            FROM candles
            WHERE symbol = :symbol
              AND timeframe = :timeframe
            ORDER BY ts DESC
            LIMIT 1;
            """
        )
        try:
            async with self._engine.begin() as conn:
                result = await conn.execute(
                    statement,
                    {"symbol": symbol, "timeframe": timeframe},
                )
                row = result.first()
            if row is None:
                return None
            return float(row[0])
        except Exception:
            LOGGER.exception("Failed fetching latest candle close for %s", symbol)
            raise

    async def insert_executed_trade(self, trade: ExecutedTradeRecord) -> None:
        """Persist one executed trade row."""
        statement = text(
            """
            INSERT INTO executed_trades (
                ts,
                side,
                symbol,
                quantity,
                price,
                fee,
                slippage,
                notional,
                realized_pnl,
                source_signal
            )
            VALUES (
                :ts,
                :side,
                :symbol,
                :quantity,
                :price,
                :fee,
                :slippage,
                :notional,
                :realized_pnl,
                :source_signal
            );
            """
        )
        payload = {
            "ts": trade.ts,
            "side": trade.side,
            "symbol": trade.symbol,
            "quantity": trade.quantity,
            "price": trade.price,
            "fee": trade.fee,
            "slippage": trade.slippage,
            "notional": trade.notional,
            "realized_pnl": trade.realized_pnl,
            "source_signal": trade.source_signal,
        }
        try:
            async with self._engine.begin() as conn:
                await conn.execute(statement, payload)
            LOGGER.info(
                "Trade persisted side=%s symbol=%s qty=%.6f",
                trade.side,
                trade.symbol,
                trade.quantity,
            )
        except Exception:
            LOGGER.exception("Failed to persist executed trade")
            raise

    async def insert_portfolio_history(
        self,
        snapshot: PortfolioHistoryRecord,
    ) -> None:
        """Persist one equity snapshot row."""
        statement = text(
            """
            INSERT INTO portfolio_history (
                ts,
                cash_balance,
                positions_value,
                total_equity,
                realized_pnl,
                unrealized_pnl,
                source
            )
            VALUES (
                :ts,
                :cash_balance,
                :positions_value,
                :total_equity,
                :realized_pnl,
                :unrealized_pnl,
                :source
            )
            ON CONFLICT (ts) DO UPDATE SET
                cash_balance = EXCLUDED.cash_balance,
                positions_value = EXCLUDED.positions_value,
                total_equity = EXCLUDED.total_equity,
                realized_pnl = EXCLUDED.realized_pnl,
                unrealized_pnl = EXCLUDED.unrealized_pnl,
                source = EXCLUDED.source;
            """
        )
        payload = {
            "ts": snapshot.ts,
            "cash_balance": snapshot.cash_balance,
            "positions_value": snapshot.positions_value,
            "total_equity": snapshot.total_equity,
            "realized_pnl": snapshot.realized_pnl,
            "unrealized_pnl": snapshot.unrealized_pnl,
            "source": snapshot.source,
        }
        try:
            async with self._engine.begin() as conn:
                await conn.execute(statement, payload)
            LOGGER.info(
                "Portfolio snapshot persisted equity=%.2f",
                snapshot.total_equity,
            )
        except Exception:
            LOGGER.exception("Failed to persist portfolio history")
            raise

    async def insert_agent_report(self, report: AgentReportRecord) -> None:
        """Persist one agent report snapshot from the message bus."""
        statement = text(
            """
            INSERT INTO agent_reports (
                ts,
                agent,
                channel,
                payload
            )
            VALUES (
                :ts,
                :agent,
                :channel,
                CAST(:payload AS JSONB)
            );
            """
        )
        payload = {
            "ts": report.ts,
            "agent": report.agent,
            "channel": report.channel,
            "payload": json.dumps(report.payload),
        }
        try:
            async with self._engine.begin() as conn:
                await conn.execute(statement, payload)
        except Exception:
            LOGGER.exception("Failed to persist agent report")
            raise

    async def fetch_latest_portfolio_snapshot(self) -> dict[str, Any] | None:
        """Return most recent portfolio snapshot."""
        statement = text(
            """
            SELECT ts, cash_balance, positions_value, total_equity,
                   realized_pnl, unrealized_pnl
            FROM portfolio_history
            ORDER BY ts DESC
            LIMIT 1;
            """
        )
        async with self._engine.begin() as conn:
            row = (await conn.execute(statement)).mappings().first()
        return dict(row) if row else None

    async def fetch_portfolio_history(
        self,
        start_date: date | None,
        end_date: date | None,
    ) -> list[dict[str, Any]]:
        """Return portfolio timeline filtered by date range."""
        statement = text(
            """
            SELECT ts, total_equity, cash_balance, positions_value
            FROM portfolio_history
            WHERE (:start_ts IS NULL OR ts >= :start_ts)
              AND (:end_ts IS NULL OR ts < :end_ts)
            ORDER BY ts ASC;
            """
        )
        params = {
            "start_ts": (
                datetime.combine(start_date, datetime.min.time(), tzinfo=None)
                if start_date
                else None
            ),
            "end_ts": (
                datetime.combine(end_date, datetime.min.time(), tzinfo=None)
                if end_date
                else None
            ),
        }
        async with self._engine.begin() as conn:
            rows = (await conn.execute(statement, params)).mappings().all()
        return [dict(row) for row in rows]

    async def fetch_latest_agent_reports(self) -> list[dict[str, Any]]:
        """Return latest report per agent from persisted bus events."""
        statement = text(
            """
            WITH ranked AS (
                SELECT ts, agent, payload,
                       ROW_NUMBER() OVER (PARTITION BY agent ORDER BY ts DESC) AS rn
                FROM agent_reports
            )
            SELECT ts, agent, payload
            FROM ranked
            WHERE rn = 1
            ORDER BY ts DESC;
            """
        )
        async with self._engine.begin() as conn:
            rows = (await conn.execute(statement)).mappings().all()
        return [dict(row) for row in rows]

    async def fetch_latest_macro_regime(self) -> str | None:
        """Return latest macro regime from MacroScout payloads."""
        statement = text(
            """
            SELECT payload ->> 'macro_regime' AS macro_regime
            FROM agent_reports
            WHERE agent = 'macro_scout'
              AND payload ? 'macro_regime'
            ORDER BY ts DESC
            LIMIT 1;
            """
        )
        async with self._engine.begin() as conn:
            row = (await conn.execute(statement)).first()
        if row is None:
            return None
        return str(row[0]) if row[0] is not None else None

    async def fetch_latest_black_swan_risk(self) -> str | None:
        """Return latest BlackSwan risk level."""
        statement = text(
            """
            SELECT payload ->> 'risk_level' AS risk_level
            FROM agent_reports
            WHERE agent = 'black_swan'
              AND payload ? 'risk_level'
            ORDER BY ts DESC
            LIMIT 1;
            """
        )
        async with self._engine.begin() as conn:
            row = (await conn.execute(statement)).first()
        if row is None:
            return None
        return str(row[0]) if row[0] is not None else None

    async def fetch_final_decisions(
        self,
        start_date: date | None,
        end_date: date | None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return consensus decisions timeline."""
        statement = text(
            """
            SELECT ts, action, consensus_score, votes_for, rationale
            FROM final_decisions
            WHERE (:start_ts IS NULL OR ts >= :start_ts)
              AND (:end_ts IS NULL OR ts < :end_ts)
            ORDER BY ts DESC
            LIMIT :limit_n;
            """
        )
        params = {
            "start_ts": (
                datetime.combine(start_date, datetime.min.time(), tzinfo=None)
                if start_date
                else None
            ),
            "end_ts": (
                datetime.combine(end_date, datetime.min.time(), tzinfo=None)
                if end_date
                else None
            ),
            "limit_n": limit,
        }
        async with self._engine.begin() as conn:
            rows = (await conn.execute(statement, params)).mappings().all()
        return [dict(row) for row in rows]

    async def fetch_open_positions(self) -> list[dict[str, Any]]:
        """Compute current open positions from executed trades ledger."""
        statement = text(
            """
            WITH trade_rollup AS (
                SELECT
                    symbol,
                    SUM(CASE WHEN side = 'BUY' THEN quantity ELSE -quantity END) AS net_qty,
                    SUM(CASE WHEN side = 'BUY' THEN notional + fee ELSE 0 END) AS buy_cost
                FROM executed_trades
                GROUP BY symbol
            ),
            latest_prices AS (
                SELECT DISTINCT ON (symbol)
                    symbol,
                    close AS mark_price
                FROM candles
                WHERE timeframe = '1d'
                ORDER BY symbol, ts DESC
            )
            SELECT
                t.symbol,
                t.net_qty,
                CASE
                    WHEN t.net_qty > 0 THEN t.buy_cost / NULLIF(
                        (
                            SELECT SUM(quantity)
                            FROM executed_trades et
                            WHERE et.symbol = t.symbol
                              AND et.side = 'BUY'
                        ),
                        0
                    )
                    ELSE 0
                END AS avg_entry_price,
                COALESCE(lp.mark_price, 0) AS mark_price,
                t.net_qty * COALESCE(lp.mark_price, 0) AS market_value
            FROM trade_rollup t
            LEFT JOIN latest_prices lp ON lp.symbol = t.symbol
            WHERE t.net_qty > 1e-10
            ORDER BY market_value DESC;
            """
        )
        async with self._engine.begin() as conn:
            rows = (await conn.execute(statement)).mappings().all()
        return [dict(row) for row in rows]

    async def fetch_executed_trades(
        self,
        start_date: date | None,
        end_date: date | None,
        symbol: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Return executed trades ledger filtered by date and symbol."""
        statement = text(
            """
            SELECT
                ts,
                side,
                symbol,
                quantity,
                price,
                fee,
                slippage,
                notional,
                realized_pnl,
                source_signal
            FROM executed_trades
            WHERE (:start_ts IS NULL OR ts >= :start_ts)
              AND (:end_ts IS NULL OR ts < :end_ts)
              AND (:symbol_filter IS NULL OR symbol = :symbol_filter)
            ORDER BY ts DESC
            LIMIT :limit_n;
            """
        )
        params = {
            "start_ts": (
                datetime.combine(start_date, datetime.min.time(), tzinfo=None)
                if start_date
                else None
            ),
            "end_ts": (
                datetime.combine(end_date, datetime.min.time(), tzinfo=None)
                if end_date
                else None
            ),
            "symbol_filter": symbol,
            "limit_n": limit,
        }
        async with self._engine.begin() as conn:
            rows = (await conn.execute(statement, params)).mappings().all()
        return [dict(row) for row in rows]
