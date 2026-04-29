"""Mock paper-trading executor consuming orchestrator execution signals."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
import logging
import os
from typing import Any

from shared.database import (
    Database,
    ExecutedTradeRecord,
    PortfolioHistoryRecord,
)
from shared.messaging import MessageBus

LOGGER = logging.getLogger(__name__)


@dataclass
class Position:
    """Open position tracked in memory by mock executor."""

    symbol: str
    quantity: float
    avg_entry_price: float
    accumulated_fees: float = 0.0


class MockExecutor:
    """Execute simulated orders with slippage and fees."""

    EXECUTION_CHANNEL = "execution.signals"
    DEFAULT_TIMEFRAME = "1d"

    def __init__(
        self,
        bus: MessageBus | None = None,
        database: Database | None = None,
    ) -> None:
        self.bus = bus or MessageBus()
        self.db = database or Database()
        self.logger = logging.getLogger(__name__)

        self.starting_balance = float(
            os.getenv("PAPER_START_BALANCE", "10000"),
        )
        self.cash_balance = self.starting_balance
        self.allocation_fraction = float(
            os.getenv("PAPER_ALLOCATION_FRACTION", "0.10"),
        )
        self.fee_rate = float(os.getenv("PAPER_FEE_RATE", "0.001"))
        self.slippage_rate = float(os.getenv("PAPER_SLIPPAGE_RATE", "0.0005"))
        self.default_symbol = os.getenv("EXECUTOR_DEFAULT_SYMBOL", "HG=F")

        self.positions: dict[str, Position] = {}
        self.realized_pnl = 0.0

    async def run(self) -> None:
        """Listen to execution signals and simulate trade fills."""
        await self.bus.connect()
        await self.db.init_hypertables()
        pubsub = await self.bus.subscribe([self.EXECUTION_CHANNEL])
        self.logger.info(
            "MockExecutor listening on %s",
            self.EXECUTION_CHANNEL,
        )

        try:
            async for message in self.bus.iter_messages(pubsub):
                payload = message["payload"]
                await self._handle_execution_signal(payload)
        except asyncio.CancelledError:
            self.logger.info("MockExecutor cancelled")
            raise
        except Exception:
            self.logger.exception(
                "MockExecutor failed while consuming signals",
            )
            raise
        finally:
            await pubsub.aclose()
            await self.db.close()
            await self.bus.close()

    async def _handle_execution_signal(self, payload: dict[str, Any]) -> None:
        """Route action to position-management logic."""
        action = str(payload.get("action", payload.get("signal", ""))).upper()
        symbol = str(payload.get("asset", self.default_symbol))
        self.logger.info(
            "Execution signal received action=%s symbol=%s",
            action,
            symbol,
        )

        if action in {"STRONG_BUY", "BUY"}:
            allocation = self.allocation_fraction
            if action == "STRONG_BUY":
                allocation = min(0.25, allocation * 1.5)
            await self._open_or_add_position(
                symbol=symbol,
                source_signal=action,
                allocation_fraction=allocation,
            )
            return

        if action == "SELL":
            await self._close_position(symbol=symbol, source_signal=action)
            return

        if action == "ABORT/STAY_OUT":
            await self._panic_sell_all(source_signal=action)
            return

        if action in {"HOLD", "INFO", "NEUTRAL"}:
            self.logger.info(
                "No-op execution action received: %s",
                action,
            )
            await self._persist_and_log_balance()
            return

        self.logger.warning("Unsupported execution action: %s", action)

    async def _open_or_add_position(
        self,
        symbol: str,
        source_signal: str,
        allocation_fraction: float,
    ) -> None:
        """Buy with fraction of available cash."""
        market_price = await self.db.get_latest_candle_close(
            symbol=symbol,
            timeframe=self.DEFAULT_TIMEFRAME,
        )
        if market_price is None:
            self.logger.warning(
                "No market price in candles for %s; buy skipped",
                symbol,
            )
            return

        budget = self.cash_balance * allocation_fraction
        if budget <= 0.0:
            self.logger.warning("No available cash to open new positions")
            return

        execution_price = market_price * (1.0 + self.slippage_rate)
        quantity = budget / execution_price
        notional = quantity * execution_price
        fee = notional * self.fee_rate
        total_cost = notional + fee
        if total_cost > self.cash_balance:
            quantity = max(
                0.0,
                (self.cash_balance / (1.0 + self.fee_rate))
                / execution_price,
            )
            notional = quantity * execution_price
            fee = notional * self.fee_rate
            total_cost = notional + fee

        if quantity <= 1e-10:
            self.logger.warning(
                "Computed buy quantity is too small for %s",
                symbol,
            )
            return

        position = self.positions.get(symbol)
        if position is None:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_entry_price=execution_price,
                accumulated_fees=fee,
            )
        else:
            new_qty = position.quantity + quantity
            position.avg_entry_price = (
                (position.avg_entry_price * position.quantity)
                + (execution_price * quantity)
            ) / new_qty
            position.quantity = new_qty
            position.accumulated_fees += fee

        self.cash_balance -= total_cost
        await self.db.insert_executed_trade(
            ExecutedTradeRecord(
                ts=datetime.now(UTC),
                side="BUY",
                symbol=symbol,
                quantity=quantity,
                price=execution_price,
                fee=fee,
                slippage=self.slippage_rate,
                notional=notional,
                realized_pnl=0.0,
                source_signal=source_signal,
            )
        )
        await self._persist_and_log_balance()

    async def _close_position(self, symbol: str, source_signal: str) -> None:
        """Close one open position at current market price."""
        position = self.positions.get(symbol)
        if position is None:
            self.logger.info("No open position for %s", symbol)
            return

        market_price = await self.db.get_latest_candle_close(
            symbol=symbol,
            timeframe=self.DEFAULT_TIMEFRAME,
        )
        if market_price is None:
            self.logger.warning(
                "No market price in candles for %s; sell skipped",
                symbol,
            )
            return

        execution_price = market_price * (1.0 - self.slippage_rate)
        notional = position.quantity * execution_price
        fee = notional * self.fee_rate
        proceeds = notional - fee
        entry_cost = position.quantity * position.avg_entry_price
        realized = proceeds - entry_cost - position.accumulated_fees

        self.cash_balance += proceeds
        self.realized_pnl += realized
        del self.positions[symbol]

        await self.db.insert_executed_trade(
            ExecutedTradeRecord(
                ts=datetime.now(UTC),
                side="SELL",
                symbol=symbol,
                quantity=position.quantity,
                price=execution_price,
                fee=fee,
                slippage=self.slippage_rate,
                notional=notional,
                realized_pnl=realized,
                source_signal=source_signal,
            )
        )
        await self._persist_and_log_balance()

    async def _panic_sell_all(self, source_signal: str) -> None:
        """Close all open positions immediately under risk veto."""
        if not self.positions:
            self.logger.warning(
                "Panic sell requested but no open positions exist",
            )
            await self._persist_and_log_balance()
            return

        symbols = list(self.positions.keys())
        for symbol in symbols:
            await self._close_position(
                symbol=symbol,
                source_signal=source_signal,
            )

    async def _persist_and_log_balance(self) -> None:
        """Persist portfolio equity and log current paper performance."""
        positions_value = 0.0
        unrealized = 0.0
        for symbol, position in self.positions.items():
            last_price = await self.db.get_latest_candle_close(
                symbol=symbol,
                timeframe=self.DEFAULT_TIMEFRAME,
            )
            if last_price is None:
                continue
            mark_price = last_price * (1.0 - self.slippage_rate)
            market_value = position.quantity * mark_price
            positions_value += market_value
            unrealized += (
                market_value
                - (position.quantity * position.avg_entry_price)
                - position.accumulated_fees
            )

        total_equity = self.cash_balance + positions_value
        await self.db.insert_portfolio_history(
            PortfolioHistoryRecord(
                ts=datetime.now(UTC),
                cash_balance=self.cash_balance,
                positions_value=positions_value,
                total_equity=total_equity,
                realized_pnl=self.realized_pnl,
                unrealized_pnl=unrealized,
            )
        )

        pct = ((total_equity / self.starting_balance) - 1.0) * 100.0
        self.logger.info(
            "Balance Actual: $%.2f [Profit: %+0.2f%%]",
            total_equity,
            pct,
        )


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


async def main() -> None:
    configure_logging()
    executor = MockExecutor()
    await executor.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        LOGGER.info("MockExecutor interrupted by user")
