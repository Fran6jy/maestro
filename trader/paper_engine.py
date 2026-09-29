"""Cycle-level orchestration for the network-free paper trading system."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping

from maestro.shared.contracts import PredictionSignal
from maestro.trader.order_planner import OrderPlan, OrderPlanner
from maestro.trader.paper_broker import PaperBroker, PaperFill, ReconciliationReport
from maestro.trader.portfolio_allocator import (
    AllocationResult,
    MarketQuote,
    PortfolioAllocator,
)


@dataclass(frozen=True)
class PaperCycleResult:
    as_of: datetime
    starting_equity: float
    ending_equity: float
    allocation: AllocationResult
    order_plan: OrderPlan
    fills: tuple[PaperFill, ...]
    reconciliation: ReconciliationReport


class PaperTradingEngine:
    """Own allocator, planner, and paper broker state for repeated cycles."""

    def __init__(
        self,
        allocator: PortfolioAllocator | None = None,
        planner: OrderPlanner | None = None,
        broker: PaperBroker | None = None,
    ) -> None:
        self.allocator = allocator or PortfolioAllocator()
        self.planner = planner or OrderPlanner()
        self.broker = broker or PaperBroker()

    def run_cycle(
        self,
        predictions: Iterable[PredictionSignal],
        quotes: Mapping[str, MarketQuote],
        as_of: datetime,
    ) -> PaperCycleResult:
        starting_equity = self.broker.equity(quotes)
        allocation = self.allocator.allocate(predictions, quotes, starting_equity, as_of)
        snapshot = self.broker.snapshot(quotes, as_of)
        plan = self.planner.plan(allocation, snapshot, quotes)
        fills = self.broker.execute_plan(plan, quotes)
        reconciliation = self.broker.reconcile(allocation, quotes, as_of)
        ending_equity = self.broker.equity(quotes)
        return PaperCycleResult(
            as_of=as_of,
            starting_equity=starting_equity,
            ending_equity=ending_equity,
            allocation=allocation,
            order_plan=plan,
            fills=fills,
            reconciliation=reconciliation,
        )

