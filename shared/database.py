"""Database layer for TimescaleDB access and hypertable bootstrap."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

LOGGER = logging.getLogger(__name__)
SCHEMA_INIT_LOCK_KEY = 872341


@dataclass(frozen=True)
class DatabaseConfig:
    """Connection settings for TimescaleDB."""

    host: str = os.getenv("DB_HOST", "127.0.0.1")
    port: int = int(os.getenv("DB_PORT", "5435"))
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
    consensus_logic: str
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
    human_rationale: str | None = None
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
                    consensus_logic TEXT,
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
                    human_rationale TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            ),
            text(
                """
                ALTER TABLE agent_reports
                ADD COLUMN IF NOT EXISTS human_rationale TEXT;
                """
            ),
            text(
                """
                ALTER TABLE final_decisions
                ADD COLUMN IF NOT EXISTS consensus_logic TEXT;
                """
            ),
        ]

        try:
            async with self._engine.begin() as conn:
                await conn.execute(
                    text("SELECT pg_advisory_lock(:lock_key);"),
                    {"lock_key": SCHEMA_INIT_LOCK_KEY},
                )
                try:
                    for statement in statements:
                        await conn.execute(statement)
                finally:
                    await conn.execute(
                        text("SELECT pg_advisory_unlock(:lock_key);"),
                        {"lock_key": SCHEMA_INIT_LOCK_KEY},
                    )
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

    async def insert_final_decision(
        self,
        decision: FinalDecisionRecord,
    ) -> None:
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
                consensus_logic,
                source
            )
            VALUES (
                :ts,
                :action,
                :consensus_score,
                CAST(:votes_for AS JSONB),
                CAST(:latest_signals AS JSONB),
                :rationale,
                :consensus_logic,
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
            "consensus_logic": decision.consensus_logic,
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
            LOGGER.exception(
                "Failed fetching latest candle close for %s",
                symbol,
            )
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
                payload,
                human_rationale
            )
            VALUES (
                :ts,
                :agent,
                :channel,
                CAST(:payload AS JSONB),
                :human_rationale
            );
            """
        )
        payload = {
            "ts": report.ts,
            "agent": report.agent,
            "channel": report.channel,
            "payload": json.dumps(report.payload),
            "human_rationale": report.human_rationale,
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
            WHERE (
                CAST(:start_ts AS TIMESTAMPTZ) IS NULL
                OR ts >= CAST(:start_ts AS TIMESTAMPTZ)
            )
              AND (
                CAST(:end_ts AS TIMESTAMPTZ) IS NULL
                OR ts < CAST(:end_ts AS TIMESTAMPTZ)
              )
            ORDER BY ts ASC;
            """
        )
        params = {
            "start_ts": (
                datetime.combine(
                    start_date,
                    datetime.min.time(),
                    tzinfo=timezone.utc,
                )
                if start_date
                else None
            ),
            "end_ts": (
                datetime.combine(
                    end_date,
                    datetime.min.time(),
                    tzinfo=timezone.utc,
                )
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
                SELECT
                       ts,
                       agent,
                       payload,
                       human_rationale,
                       ROW_NUMBER() OVER (PARTITION BY agent ORDER BY ts DESC) AS rn
                FROM agent_reports
            )
            SELECT ts, agent, payload, human_rationale
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
            SELECT
                ts,
                action,
                consensus_score,
                votes_for,
                rationale,
                consensus_logic
            FROM final_decisions
            WHERE (
                CAST(:start_ts AS TIMESTAMPTZ) IS NULL
                OR ts >= CAST(:start_ts AS TIMESTAMPTZ)
            )
              AND (
                CAST(:end_ts AS TIMESTAMPTZ) IS NULL
                OR ts < CAST(:end_ts AS TIMESTAMPTZ)
              )
            ORDER BY ts DESC
            LIMIT :limit_n;
            """
        )
        params = {
            "start_ts": (
                datetime.combine(
                    start_date,
                    datetime.min.time(),
                    tzinfo=timezone.utc,
                )
                if start_date
                else None
            ),
            "end_ts": (
                datetime.combine(
                    end_date,
                    datetime.min.time(),
                    tzinfo=timezone.utc,
                )
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
                    SUM(
                        CASE WHEN side = 'BUY'
                        THEN quantity ELSE -quantity END
                    ) AS net_qty,
                    SUM(
                        CASE WHEN side = 'BUY'
                        THEN notional + fee ELSE 0 END
                    ) AS buy_cost
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
            WHERE (
                CAST(:start_ts AS TIMESTAMPTZ) IS NULL
                OR ts >= CAST(:start_ts AS TIMESTAMPTZ)
            )
              AND (
                CAST(:end_ts AS TIMESTAMPTZ) IS NULL
                OR ts < CAST(:end_ts AS TIMESTAMPTZ)
              )
              AND (
                CAST(:symbol_filter AS TEXT) IS NULL
                OR symbol = CAST(:symbol_filter AS TEXT)
              )
            ORDER BY ts DESC
            LIMIT :limit_n;
            """
        )
        params = {
            "start_ts": (
                datetime.combine(
                    start_date,
                    datetime.min.time(),
                    tzinfo=timezone.utc,
                )
                if start_date
                else None
            ),
            "end_ts": (
                datetime.combine(
                    end_date,
                    datetime.min.time(),
                    tzinfo=timezone.utc,
                )
                if end_date
                else None
            ),
            "symbol_filter": symbol,
            "limit_n": limit,
        }
        async with self._engine.begin() as conn:
            rows = (await conn.execute(statement, params)).mappings().all()
        return [dict(row) for row in rows]

    async def fetch_candles_history(
        self,
        symbols: list[str],
        start_date: date | None,
        end_date: date | None,
        timeframe: str = "1d",
    ) -> list[dict[str, Any]]:
        """Return candle close history for selected symbols/date range."""
        if not symbols:
            return []
        statement = text(
            """
            SELECT ts, symbol, close, volume
            FROM candles
            WHERE symbol = ANY(CAST(:symbols AS TEXT[]))
              AND timeframe = :timeframe
              AND (
                CAST(:start_ts AS TIMESTAMPTZ) IS NULL
                OR ts >= CAST(:start_ts AS TIMESTAMPTZ)
              )
              AND (
                CAST(:end_ts AS TIMESTAMPTZ) IS NULL
                OR ts < CAST(:end_ts AS TIMESTAMPTZ)
              )
            ORDER BY ts ASC;
            """
        )
        params = {
            "symbols": symbols,
            "timeframe": timeframe,
            "start_ts": (
                datetime.combine(
                    start_date,
                    datetime.min.time(),
                    tzinfo=timezone.utc,
                )
                if start_date
                else None
            ),
            "end_ts": (
                datetime.combine(
                    end_date,
                    datetime.min.time(),
                    tzinfo=timezone.utc,
                )
                if end_date
                else None
            ),
        }
        async with self._engine.begin() as conn:
            rows = (await conn.execute(statement, params)).mappings().all()
        return [dict(row) for row in rows]

    async def fetch_reports_for_query(
        self,
        keywords: list[str],
        asset_symbols: list[str],
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Return recent reports filtered by asset or keyword matches."""
        pattern_values = [f"%{term}%" for term in keywords if term]
        statement = text(
            """
            SELECT ts, agent, payload, human_rationale
            FROM agent_reports
            WHERE (
                CARDINALITY(CAST(:asset_symbols AS TEXT[])) = 0
                OR (payload ->> 'asset') = ANY(CAST(:asset_symbols AS TEXT[]))
            )
               OR (
                CARDINALITY(CAST(:patterns AS TEXT[])) > 0
                AND (
                    COALESCE(human_rationale, '') ILIKE ANY(CAST(:patterns AS TEXT[]))
                    OR CAST(payload AS TEXT) ILIKE ANY(CAST(:patterns AS TEXT[]))
                )
            )
            ORDER BY ts DESC
            LIMIT :limit_n;
            """
        )
        params = {
            "asset_symbols": asset_symbols,
            "patterns": pattern_values,
            "limit_n": limit,
        }
        async with self._engine.begin() as conn:
            rows = (await conn.execute(statement, params)).mappings().all()
        return [dict(row) for row in rows]

    async def fetch_recent_final_decisions(
        self,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Return most recent consensus decisions for conversational context."""
        statement = text(
            """
            SELECT
                ts,
                action,
                consensus_score,
                rationale,
                consensus_logic
            FROM final_decisions
            ORDER BY ts DESC
            LIMIT :limit_n;
            """
        )
        async with self._engine.begin() as conn:
            rows = (
                await conn.execute(statement, {"limit_n": limit})
            ).mappings().all()
        return [dict(row) for row in rows]

    async def fetch_whale_open_interest_series(
        self,
        symbols: list[str],
        start_date: date | None,
        end_date: date | None,
    ) -> list[dict[str, Any]]:
        """Return WhaleScout OI proxy series from macro_indicators."""
        if not symbols:
            return []
        indicators = [f"WHALE_OI::{sym}" for sym in symbols]
        statement = text(
            """
            SELECT ts, indicator, value
            FROM macro_indicators
            WHERE indicator = ANY(CAST(:indicators AS TEXT[]))
              AND country = 'global'
              AND (
                CAST(:start_ts AS TIMESTAMPTZ) IS NULL
                OR ts >= CAST(:start_ts AS TIMESTAMPTZ)
              )
              AND (
                CAST(:end_ts AS TIMESTAMPTZ) IS NULL
                OR ts < CAST(:end_ts AS TIMESTAMPTZ)
              )
            ORDER BY ts ASC;
            """
        )
        params = {
            "indicators": indicators,
            "start_ts": (
                datetime.combine(
                    start_date,
                    datetime.min.time(),
                    tzinfo=timezone.utc,
                )
                if start_date
                else None
            ),
            "end_ts": (
                datetime.combine(
                    end_date,
                    datetime.min.time(),
                    tzinfo=timezone.utc,
                )
                if end_date
                else None
            ),
        }
        async with self._engine.begin() as conn:
            rows = (await conn.execute(statement, params)).mappings().all()
        return [dict(row) for row in rows]

    async def fetch_calendar_agent_events(
        self,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Latest calendar_agent rows for macro countdown UI."""
        statement = text(
            """
            SELECT ts, payload
            FROM agent_reports
            WHERE agent = 'calendar_agent'
            ORDER BY ts DESC
            LIMIT :limit_n;
            """
        )
        async with self._engine.begin() as conn:
            rows = (
                await conn.execute(statement, {"limit_n": limit})
            ).mappings().all()
        return [dict(row) for row in rows]

    async def fetch_closes_for_zscore(
        self,
        symbols: list[str],
        bars: int = 22,
        timeframe: str = "1d",
    ) -> list[dict[str, Any]]:
        """Last N daily closes per symbol (window over hypertable)."""
        if not symbols:
            return []
        statement = text(
            """
            WITH ranked AS (
                SELECT
                    ts,
                    symbol,
                    close,
                    ROW_NUMBER() OVER (
                        PARTITION BY symbol ORDER BY ts DESC
                    ) AS rn
                FROM candles
                WHERE timeframe = :timeframe
                  AND symbol = ANY(CAST(:symbols AS TEXT[]))
            )
            SELECT ts, symbol, close
            FROM ranked
            WHERE rn <= :bars
            ORDER BY symbol, ts ASC;
            """
        )
        async with self._engine.begin() as conn:
            rows = (
                await conn.execute(
                    statement,
                    {"symbols": symbols, "bars": bars, "timeframe": timeframe},
                )
            ).mappings().all()
        return [dict(row) for row in rows]

    async def fetch_dashboard_bundle(
        self,
        start_date: date,
        end_date: date,
        trades_symbol: str | None,
        trades_limit: int,
    ) -> dict[str, Any]:
        """Load dashboard datasets in one DB session (fewer pool round-trips)."""
        start_ts = datetime.combine(
            start_date,
            datetime.min.time(),
            tzinfo=timezone.utc,
        )
        end_ts = datetime.combine(
            end_date,
            datetime.min.time(),
            tzinfo=timezone.utc,
        )
        sym_candles = ["HG=F", "GC=F", "NLR"]
        whale_inds = ["WHALE_OI::HG=F", "WHALE_OI::GC=F"]

        snap_sql = text(
            """
            SELECT ts, cash_balance, positions_value, total_equity,
                   realized_pnl, unrealized_pnl
            FROM portfolio_history
            ORDER BY ts DESC
            LIMIT 1;
            """
        )
        hist_sql = text(
            """
            SELECT ts, total_equity, cash_balance, positions_value
            FROM portfolio_history
            WHERE ts >= :start_ts AND ts < :end_ts
            ORDER BY ts ASC;
            """
        )
        reports_sql = text(
            """
            WITH ranked AS (
                SELECT
                       ts,
                       agent,
                       payload,
                       human_rationale,
                       ROW_NUMBER() OVER (
                           PARTITION BY agent ORDER BY ts DESC
                       ) AS rn
                FROM agent_reports
            )
            SELECT ts, agent, payload, human_rationale
            FROM ranked
            WHERE rn = 1
            ORDER BY ts DESC;
            """
        )
        decisions_sql = text(
            """
            SELECT ts, action, consensus_score, votes_for, rationale,
                   consensus_logic
            FROM final_decisions
            WHERE ts >= :start_ts AND ts < :end_ts
            ORDER BY ts DESC
            LIMIT 50;
            """
        )
        positions_sql = text(
            """
            WITH trade_rollup AS (
                SELECT
                    symbol,
                    SUM(
                        CASE WHEN side = 'BUY'
                        THEN quantity ELSE -quantity END
                    ) AS net_qty,
                    SUM(
                        CASE WHEN side = 'BUY'
                        THEN notional + fee ELSE 0 END
                    ) AS buy_cost
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
        trades_sql = text(
            """
            SELECT ts, side, symbol, quantity, price, fee, slippage,
                   notional, realized_pnl, source_signal
            FROM executed_trades
            WHERE ts >= :start_ts AND ts < :end_ts
              AND (
                CAST(:symbol_filter AS TEXT) IS NULL
                OR symbol = CAST(:symbol_filter AS TEXT)
              )
            ORDER BY ts DESC
            LIMIT :limit_n;
            """
        )
        macro_sql = text(
            """
            SELECT payload ->> 'macro_regime' AS macro_regime
            FROM agent_reports
            WHERE agent = 'macro_scout'
              AND payload ? 'macro_regime'
            ORDER BY ts DESC
            LIMIT 1;
            """
        )
        bs_sql = text(
            """
            SELECT payload ->> 'risk_level' AS risk_level
            FROM agent_reports
            WHERE agent = 'black_swan'
              AND payload ? 'risk_level'
            ORDER BY ts DESC
            LIMIT 1;
            """
        )
        candles_sql = text(
            """
            SELECT ts, symbol, close, volume
            FROM candles
            WHERE symbol = ANY(CAST(:symbols AS TEXT[]))
              AND timeframe = '1d'
              AND ts >= :start_ts AND ts < :end_ts
            ORDER BY ts ASC;
            """
        )
        whale_sql = text(
            """
            SELECT ts, indicator, value
            FROM macro_indicators
            WHERE indicator = ANY(CAST(:indicators AS TEXT[]))
              AND country = 'global'
              AND ts >= :start_ts AND ts < :end_ts
            ORDER BY ts ASC;
            """
        )
        cal_sql = text(
            """
            SELECT ts, payload
            FROM agent_reports
            WHERE agent = 'calendar_agent'
            ORDER BY ts DESC
            LIMIT 5;
            """
        )
        z_sql = text(
            """
            WITH ranked AS (
                SELECT ts, symbol, close,
                    ROW_NUMBER() OVER (
                        PARTITION BY symbol ORDER BY ts DESC
                    ) AS rn
                FROM candles
                WHERE timeframe = '1d'
                  AND symbol = ANY(CAST(:symbols AS TEXT[]))
            )
            SELECT ts, symbol, close
            FROM ranked
            WHERE rn <= 22
            ORDER BY symbol, ts ASC;
            """
        )

        async with self._engine.connect() as conn:
            snap = (await conn.execute(snap_sql)).mappings().first()
            hist = (
                await conn.execute(hist_sql, {"start_ts": start_ts, "end_ts": end_ts})
            ).mappings().all()
            reps = (await conn.execute(reports_sql)).mappings().all()
            decs = (
                await conn.execute(
                    decisions_sql,
                    {"start_ts": start_ts, "end_ts": end_ts},
                )
            ).mappings().all()
            pos = (await conn.execute(positions_sql)).mappings().all()
            trd = (
                await conn.execute(
                    trades_sql,
                    {
                        "start_ts": start_ts,
                        "end_ts": end_ts,
                        "symbol_filter": trades_symbol,
                        "limit_n": trades_limit,
                    },
                )
            ).mappings().all()
            macro_row = (await conn.execute(macro_sql)).first()
            bs_row = (await conn.execute(bs_sql)).first()
            cndl = (
                await conn.execute(
                    candles_sql,
                    {
                        "symbols": sym_candles,
                        "start_ts": start_ts,
                        "end_ts": end_ts,
                    },
                )
            ).mappings().all()
            whale = (
                await conn.execute(
                    whale_sql,
                    {
                        "indicators": whale_inds,
                        "start_ts": start_ts,
                        "end_ts": end_ts,
                    },
                )
            ).mappings().all()
            cal = (await conn.execute(cal_sql)).mappings().all()
            zrows = (
                await conn.execute(z_sql, {"symbols": sym_candles})
            ).mappings().all()

        return {
            "snapshot": dict(snap) if snap else None,
            "history": [dict(r) for r in hist],
            "reports": [dict(r) for r in reps],
            "decisions": [dict(r) for r in decs],
            "positions": [dict(r) for r in pos],
            "trades": [dict(r) for r in trd],
            "macro_regime": (
                str(macro_row[0]) if macro_row and macro_row[0] else None
            ),
            "black_swan_risk": (
                str(bs_row[0]) if bs_row and bs_row[0] else None
            ),
            "candles": [dict(r) for r in cndl],
            "whale_oi": [dict(r) for r in whale],
            "calendar_events": [dict(r) for r in cal],
            "zscore_closes": [dict(r) for r in zrows],
        }

    async def fetch_recent_portfolio_snapshots(
        self,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Return latest portfolio snapshots for conversational context."""
        statement = text(
            """
            SELECT
                ts,
                cash_balance,
                positions_value,
                total_equity,
                realized_pnl,
                unrealized_pnl
            FROM portfolio_history
            ORDER BY ts DESC
            LIMIT :limit_n;
            """
        )
        async with self._engine.begin() as conn:
            rows = (
                await conn.execute(statement, {"limit_n": limit})
            ).mappings().all()
        return [dict(row) for row in rows]
