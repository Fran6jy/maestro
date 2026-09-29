"""Convert target positions into deterministic, idempotent order intents."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
from typing import Mapping

from maestro.config.instruments import get_instrument_spec
from maestro.trader.portfolio_allocator import AllocationResult, MarketQuote, TargetPosition


@dataclass(frozen=True)
class BrokerPosition:
    instrument: str
    units: int
    average_price: float

    def __post_init__(self) -> None:
        get_instrument_spec(self.instrument)
        if self.average_price <= 0:
            raise ValueError("average_price must be positive")


@dataclass(frozen=True)
class BrokerSnapshot:
    as_of: datetime
    positions: Mapping[str, BrokerPosition]
    processed_client_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class PlannedOrder:
    client_order_id: str
    instrument: str
    units: int
    current_units: int
    resulting_units: int
    order_type: str
    reason: str
    reference_price: float
    price_bound: float
    stop_loss: float | None
    created_at: datetime


@dataclass(frozen=True)
class PlanningRejection:
    instrument: str
    reason: str
    detail: str = ""


@dataclass(frozen=True)
class OrderPlan:
    as_of: datetime
    orders: tuple[PlannedOrder, ...]
    rejected: tuple[PlanningRejection, ...]


class OrderPlanner:
    """Plan only the delta between managed broker positions and targets."""

    def __init__(self, max_slippage_bps: float = 2.0, max_quote_age_seconds: float = 15.0) -> None:
        if max_slippage_bps <= 0 or max_quote_age_seconds <= 0:
            raise ValueError("planner limits must be positive")
        self.max_slippage_bps = max_slippage_bps
        self.max_quote_age_seconds = max_quote_age_seconds

    def plan(
        self,
        allocation: AllocationResult,
        broker: BrokerSnapshot,
        quotes: Mapping[str, MarketQuote],
    ) -> OrderPlan:
        if broker.as_of.tzinfo is None or broker.as_of.utcoffset() is None:
            raise ValueError("broker snapshot time must be timezone-aware")

        targets = {target.instrument: target for target in allocation.targets}
        instruments = sorted(set(targets) | set(broker.positions))
        orders: list[PlannedOrder] = []
        rejected: list[PlanningRejection] = []

        for instrument in instruments:
            target = targets.get(instrument)
            current = broker.positions.get(instrument)
            current_units = current.units if current is not None else 0
            target_units = target.units if target is not None else 0
            if current_units == target_units:
                continue

            quote = quotes.get(instrument)
            rejection = self._validate_quote(instrument, quote, allocation.as_of)
            if rejection is not None:
                rejected.append(rejection)
                continue
            assert quote is not None

            # A reversal is deliberately two-stage. Closing is safe and
            # idempotent; opening the opposite side requires a fresh cycle and
            # a reconciled zero position.
            reversing = current_units != 0 and target_units != 0 and (current_units > 0) != (target_units > 0)
            if reversing:
                delta = -current_units
                resulting = 0
                reason = "close_for_reversal"
            else:
                delta = target_units - current_units
                resulting = target_units
                reason = self._reason(current_units, target_units)

            spec = get_instrument_spec(instrument)
            if abs(delta) < spec.min_units:
                rejected.append(PlanningRejection(instrument, "below_minimum_order_size"))
                continue
            client_id = self._client_id(allocation.as_of, instrument, current_units, resulting)
            if client_id in broker.processed_client_ids:
                rejected.append(PlanningRejection(instrument, "duplicate_client_order_id", client_id))
                continue

            buying = delta > 0
            reference = quote.ask if buying else quote.bid
            bound_multiplier = 1.0 + self.max_slippage_bps / 10_000.0 if buying else 1.0 - self.max_slippage_bps / 10_000.0
            stop_loss = self._stop_loss(target, quote.mid) if resulting != 0 else None
            orders.append(PlannedOrder(
                client_order_id=client_id,
                instrument=instrument,
                units=delta,
                current_units=current_units,
                resulting_units=resulting,
                order_type="MARKET",
                reason=reason,
                reference_price=reference,
                price_bound=reference * bound_multiplier,
                stop_loss=stop_loss,
                created_at=allocation.as_of,
            ))

        return OrderPlan(allocation.as_of, tuple(orders), tuple(rejected))

    def _validate_quote(
        self,
        instrument: str,
        quote: MarketQuote | None,
        as_of: datetime,
    ) -> PlanningRejection | None:
        if quote is None:
            return PlanningRejection(instrument, "missing_quote")
        if quote.instrument != instrument:
            return PlanningRejection(instrument, "quote_instrument_mismatch")
        if not quote.tradeable:
            return PlanningRejection(instrument, "instrument_not_tradeable")
        age = (as_of - quote.timestamp).total_seconds()
        if age < 0 or age > self.max_quote_age_seconds:
            return PlanningRejection(instrument, "stale_quote", f"age={age:.1f}s")
        return None

    @staticmethod
    def _reason(current: int, target: int) -> str:
        if current == 0 and target != 0:
            return "open"
        if target == 0:
            return "close"
        if abs(target) > abs(current):
            return "increase"
        return "reduce"

    @staticmethod
    def _stop_loss(target: TargetPosition | None, mid: float) -> float | None:
        if target is None or target.units == 0:
            return None
        direction = 1 if target.units > 0 else -1
        return mid * (1.0 - direction * target.stop_distance_pct)

    @staticmethod
    def _client_id(as_of: datetime, instrument: str, current: int, target: int) -> str:
        raw = f"{as_of.isoformat()}|{instrument}|{current}|{target}".encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()[:16]
        return f"MAESTRO-{digest}"

