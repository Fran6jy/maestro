"""Network-free broker simulator for forward paper trading."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

from maestro.config.instruments import InstrumentSpec, get_instrument_spec
from maestro.trader.order_planner import (
    BrokerPosition,
    BrokerSnapshot,
    OrderPlan,
    PlannedOrder,
)
from maestro.trader.portfolio_allocator import AllocationResult, MarketQuote


@dataclass
class _PaperPosition:
    units: int
    average_price: float
    stop_loss: float | None = None


@dataclass(frozen=True)
class PaperFill:
    transaction_id: str
    client_order_id: str
    instrument: str
    requested_units: int
    filled_units: int
    fill_price: float | None
    realized_pnl_usd: float
    status: str
    reason: str
    timestamp: datetime


@dataclass(frozen=True)
class PositionDifference:
    instrument: str
    expected_units: int
    actual_units: int


@dataclass(frozen=True)
class ReconciliationReport:
    as_of: datetime
    in_sync: bool
    differences: tuple[PositionDifference, ...]
    missing_positions: tuple[str, ...]
    orphan_positions: tuple[str, ...]
    equity: float


class PaperBroker:
    """Execute planned orders against quotes without any network capability."""

    def __init__(
        self,
        initial_cash: float = 10_000.0,
        slippage_bps: float = 0.25,
        max_quote_age_seconds: float = 15.0,
    ) -> None:
        if initial_cash <= 0 or slippage_bps < 0 or max_quote_age_seconds <= 0:
            raise ValueError("paper broker configuration is invalid")
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.slippage_bps = float(slippage_bps)
        self.max_quote_age_seconds = float(max_quote_age_seconds)
        self._positions: dict[str, _PaperPosition] = {}
        self._processed: dict[str, PaperFill] = {}
        self._transactions: list[PaperFill] = []
        self._sequence = 0

    def execute_plan(
        self,
        plan: OrderPlan,
        quotes: Mapping[str, MarketQuote],
    ) -> tuple[PaperFill, ...]:
        return tuple(self.submit(order, quotes.get(order.instrument), plan.as_of) for order in plan.orders)

    def submit(
        self,
        order: PlannedOrder,
        quote: MarketQuote | None,
        as_of: datetime,
    ) -> PaperFill:
        if order.client_order_id in self._processed:
            original = self._processed[order.client_order_id]
            return PaperFill(
                transaction_id=original.transaction_id,
                client_order_id=original.client_order_id,
                instrument=original.instrument,
                requested_units=order.units,
                filled_units=0,
                fill_price=original.fill_price,
                realized_pnl_usd=0.0,
                status="DUPLICATE",
                reason="already_filled",
                timestamp=as_of,
            )
        if quote is None or quote.instrument != order.instrument:
            return self._rejection(order, as_of, "missing_or_mismatched_quote")
        if not quote.tradeable:
            return self._rejection(order, as_of, "instrument_not_tradeable")
        quote_age = (as_of - quote.timestamp).total_seconds()
        if quote_age < 0 or quote_age > self.max_quote_age_seconds:
            return self._rejection(order, as_of, "stale_quote")

        current = self._positions.get(order.instrument)
        actual_units = current.units if current is not None else 0
        if actual_units != order.current_units:
            return self._rejection(order, as_of, "position_changed_since_planning")
        if actual_units + order.units != order.resulting_units:
            return self._rejection(order, as_of, "invalid_resulting_position")

        fill_price = self._fill_price(order.units, quote)
        if order.units > 0 and fill_price > order.price_bound:
            return self._rejection(order, as_of, "buy_price_bound_exceeded")
        if order.units < 0 and fill_price < order.price_bound:
            return self._rejection(order, as_of, "sell_price_bound_exceeded")

        spec = get_instrument_spec(order.instrument)
        realized = self._apply_fill(spec, order, fill_price)
        self.cash += realized
        self._sequence += 1
        fill = PaperFill(
            transaction_id=f"PAPER-{self._sequence:08d}",
            client_order_id=order.client_order_id,
            instrument=order.instrument,
            requested_units=order.units,
            filled_units=order.units,
            fill_price=fill_price,
            realized_pnl_usd=realized,
            status="FILLED",
            reason=order.reason,
            timestamp=as_of,
        )
        self._processed[order.client_order_id] = fill
        self._transactions.append(fill)
        return fill

    def snapshot(self, quotes: Mapping[str, MarketQuote], as_of: datetime) -> BrokerSnapshot:
        positions = {
            instrument: BrokerPosition(instrument, state.units, state.average_price)
            for instrument, state in self._positions.items()
        }
        return BrokerSnapshot(as_of, positions, frozenset(self._processed))

    def equity(self, quotes: Mapping[str, MarketQuote]) -> float:
        unrealized = 0.0
        for instrument, position in self._positions.items():
            quote = quotes.get(instrument)
            if quote is None:
                raise ValueError(f"missing quote for open paper position {instrument}")
            spec = get_instrument_spec(instrument)
            quote_pnl = position.units * (quote.mid - position.average_price)
            unrealized += self._home_pnl(spec, quote_pnl, quote.mid)
        return self.cash + unrealized

    def reconcile(
        self,
        allocation: AllocationResult,
        quotes: Mapping[str, MarketQuote],
        as_of: datetime,
    ) -> ReconciliationReport:
        expected = {target.instrument: target.units for target in allocation.targets}
        actual = {instrument: position.units for instrument, position in self._positions.items()}
        instruments = sorted(set(expected) | set(actual))
        differences = tuple(
            PositionDifference(instrument, expected.get(instrument, 0), actual.get(instrument, 0))
            for instrument in instruments
            if expected.get(instrument, 0) != actual.get(instrument, 0)
        )
        missing = tuple(sorted(instrument for instrument in expected if expected[instrument] and not actual.get(instrument)))
        orphan = tuple(sorted(instrument for instrument in actual if actual[instrument] and not expected.get(instrument)))
        return ReconciliationReport(
            as_of=as_of,
            in_sync=not differences,
            differences=differences,
            missing_positions=missing,
            orphan_positions=orphan,
            equity=self.equity(quotes),
        )

    @property
    def transactions(self) -> tuple[PaperFill, ...]:
        return tuple(self._transactions)

    def _fill_price(self, units: int, quote: MarketQuote) -> float:
        multiplier = self.slippage_bps / 10_000.0
        return quote.ask * (1.0 + multiplier) if units > 0 else quote.bid * (1.0 - multiplier)

    def _apply_fill(self, spec: InstrumentSpec, order: PlannedOrder, fill_price: float) -> float:
        old = self._positions.get(order.instrument)
        old_units = old.units if old is not None else 0
        old_average = old.average_price if old is not None else fill_price
        delta = order.units
        new_units = old_units + delta
        realized = 0.0

        if old_units and (old_units > 0) != (delta > 0):
            closed_units = min(abs(old_units), abs(delta))
            quote_pnl = closed_units * (fill_price - old_average) * (1 if old_units > 0 else -1)
            realized = self._home_pnl(spec, quote_pnl, fill_price)

        if new_units == 0:
            self._positions.pop(order.instrument, None)
        elif old_units == 0 or (old_units > 0) == (delta > 0):
            total_abs = abs(old_units) + abs(delta)
            average = (abs(old_units) * old_average + abs(delta) * fill_price) / total_abs
            self._positions[order.instrument] = _PaperPosition(new_units, average, order.stop_loss)
        elif (new_units > 0) == (old_units > 0):
            assert old is not None
            old.units = new_units
            if order.stop_loss is not None:
                old.stop_loss = order.stop_loss
        else:
            self._positions[order.instrument] = _PaperPosition(new_units, fill_price, order.stop_loss)
        return realized

    @staticmethod
    def _home_pnl(spec: InstrumentSpec, quote_pnl: float, current_price: float) -> float:
        if spec.quote_currency == "USD":
            return quote_pnl
        if spec.base_currency == "USD":
            return quote_pnl / current_price
        raise ValueError(f"{spec.instrument} needs a point-in-time home conversion")

    @staticmethod
    def _rejection(order: PlannedOrder, as_of: datetime, reason: str) -> PaperFill:
        return PaperFill(
            transaction_id="",
            client_order_id=order.client_order_id,
            instrument=order.instrument,
            requested_units=order.units,
            filled_units=0,
            fill_price=None,
            realized_pnl_usd=0.0,
            status="REJECTED",
            reason=reason,
            timestamp=as_of,
        )
