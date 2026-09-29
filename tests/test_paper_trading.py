from __future__ import annotations

from datetime import timedelta
import unittest

from maestro.trader.order_planner import OrderPlanner
from maestro.trader.paper_broker import PaperBroker
from maestro.trader.paper_engine import PaperTradingEngine
from maestro.trader.portfolio_allocator import PortfolioAllocator
from maestro.tests.test_portfolio_allocator import NOW, quote, signal


class PaperTradingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.quotes = {
            "EUR_USD": quote("EUR_USD", 1.10, 0.0002),
            "XAU_USD": quote("XAU_USD", 2_000.0, 0.20, stop_distance_pct=0.01),
        }
        self.allocator = PortfolioAllocator()
        self.planner = OrderPlanner()
        self.broker = PaperBroker(initial_cash=10_000, slippage_bps=0.25)

    def _allocation(self):
        return self.allocator.allocate(
            [
                signal("EUR_USD", direction=1, expected_return=0.003),
                signal("XAU_USD", direction=-1, expected_return=-0.004),
            ],
            self.quotes,
            10_000,
            NOW,
        )

    def test_allocate_plan_fill_and_reconcile(self) -> None:
        allocation = self._allocation()
        snapshot = self.broker.snapshot(self.quotes, NOW)
        plan = self.planner.plan(allocation, snapshot, self.quotes)
        fills = self.broker.execute_plan(plan, self.quotes)
        report = self.broker.reconcile(allocation, self.quotes, NOW)

        self.assertEqual(len(plan.orders), 2)
        self.assertTrue(all(fill.status == "FILLED" for fill in fills))
        self.assertTrue(report.in_sync)
        self.assertLess(report.equity, 10_000)  # spread/slippage mark-to-market

    def test_replanning_an_in_sync_portfolio_creates_no_orders(self) -> None:
        allocation = self._allocation()
        first = self.planner.plan(allocation, self.broker.snapshot(self.quotes, NOW), self.quotes)
        self.broker.execute_plan(first, self.quotes)
        second = self.planner.plan(allocation, self.broker.snapshot(self.quotes, NOW), self.quotes)
        self.assertEqual(second.orders, ())

    def test_duplicate_client_id_does_not_fill_twice(self) -> None:
        allocation = self._allocation()
        plan = self.planner.plan(allocation, self.broker.snapshot(self.quotes, NOW), self.quotes)
        order = plan.orders[0]
        first = self.broker.submit(order, self.quotes[order.instrument], NOW)
        duplicate = self.broker.submit(order, self.quotes[order.instrument], NOW)

        self.assertEqual(first.status, "FILLED")
        self.assertEqual(duplicate.status, "DUPLICATE")
        self.assertEqual(len(self.broker.transactions), 1)

    def test_disappearing_target_is_closed(self) -> None:
        allocation = self._allocation()
        opening = self.planner.plan(allocation, self.broker.snapshot(self.quotes, NOW), self.quotes)
        self.broker.execute_plan(opening, self.quotes)

        empty = self.allocator.allocate([], self.quotes, self.broker.equity(self.quotes), NOW)
        closing = self.planner.plan(empty, self.broker.snapshot(self.quotes, NOW), self.quotes)
        fills = self.broker.execute_plan(closing, self.quotes)

        self.assertEqual({order.reason for order in closing.orders}, {"close"})
        self.assertTrue(all(fill.status == "FILLED" for fill in fills))
        self.assertTrue(self.broker.reconcile(empty, self.quotes, NOW).in_sync)

    def test_reversal_is_closed_then_opened_on_a_fresh_cycle(self) -> None:
        long_allocation = self.allocator.allocate(
            [signal("EUR_USD", 1, 0.003)], self.quotes, 10_000, NOW
        )
        opening = self.planner.plan(long_allocation, self.broker.snapshot(self.quotes, NOW), self.quotes)
        self.broker.execute_plan(opening, self.quotes)

        short_allocation = self.allocator.allocate(
            [signal("EUR_USD", -1, -0.003)], self.quotes, 10_000, NOW
        )
        close_cycle = self.planner.plan(
            short_allocation, self.broker.snapshot(self.quotes, NOW), self.quotes
        )
        self.assertEqual(close_cycle.orders[0].reason, "close_for_reversal")
        self.assertEqual(close_cycle.orders[0].resulting_units, 0)
        self.broker.execute_plan(close_cycle, self.quotes)
        self.assertFalse(self.broker.reconcile(short_allocation, self.quotes, NOW).in_sync)

        open_cycle = self.planner.plan(
            short_allocation, self.broker.snapshot(self.quotes, NOW), self.quotes
        )
        self.assertEqual(open_cycle.orders[0].reason, "open")
        self.broker.execute_plan(open_cycle, self.quotes)
        self.assertTrue(self.broker.reconcile(short_allocation, self.quotes, NOW).in_sync)

    def test_position_change_after_planning_rejects_stale_order(self) -> None:
        allocation = self._allocation()
        stale_plan = self.planner.plan(allocation, self.broker.snapshot(self.quotes, NOW), self.quotes)
        first_order = stale_plan.orders[0]
        self.broker.submit(first_order, self.quotes[first_order.instrument], NOW)

        # Re-submit a different stale intent that still claims the position was zero.
        other = next(order for order in stale_plan.orders if order.instrument != first_order.instrument)
        self.broker.submit(other, self.quotes[other.instrument], NOW)
        duplicate_state_order = first_order.__class__(
            client_order_id=first_order.client_order_id + "-stale",
            instrument=first_order.instrument,
            units=first_order.units,
            current_units=0,
            resulting_units=first_order.units,
            order_type=first_order.order_type,
            reason="open",
            reference_price=first_order.reference_price,
            price_bound=first_order.price_bound,
            stop_loss=first_order.stop_loss,
            created_at=first_order.created_at,
        )
        rejected = self.broker.submit(
            duplicate_state_order, self.quotes[first_order.instrument], NOW
        )
        self.assertEqual(rejected.status, "REJECTED")
        self.assertEqual(rejected.reason, "position_changed_since_planning")

    def test_price_bound_rejects_excessive_slippage(self) -> None:
        strict_planner = OrderPlanner(max_slippage_bps=0.1)
        costly_broker = PaperBroker(initial_cash=10_000, slippage_bps=1.0)
        allocation = self.allocator.allocate(
            [signal("EUR_USD", 1, 0.003)], self.quotes, 10_000, NOW
        )
        plan = strict_planner.plan(
            allocation, costly_broker.snapshot(self.quotes, NOW), self.quotes
        )
        fill = costly_broker.execute_plan(plan, self.quotes)[0]
        self.assertEqual(fill.status, "REJECTED")
        self.assertEqual(fill.reason, "buy_price_bound_exceeded")

    def test_broker_rechecks_quote_freshness(self) -> None:
        allocation = self._allocation()
        plan = self.planner.plan(allocation, self.broker.snapshot(self.quotes, NOW), self.quotes)
        order = plan.orders[0]
        fill = self.broker.submit(
            order,
            self.quotes[order.instrument],
            NOW + timedelta(minutes=1),
        )
        self.assertEqual(fill.status, "REJECTED")
        self.assertEqual(fill.reason, "stale_quote")

    def test_profitable_round_trip_increases_realized_cash(self) -> None:
        allocation = self.allocator.allocate(
            [signal("EUR_USD", 1, 0.003)], self.quotes, 10_000, NOW
        )
        self.broker.execute_plan(
            self.planner.plan(allocation, self.broker.snapshot(self.quotes, NOW), self.quotes),
            self.quotes,
        )
        higher_quotes = {
            **self.quotes,
            "EUR_USD": quote(
                "EUR_USD", 1.12, 0.0002, timestamp=NOW + timedelta(minutes=5)
            ),
        }
        later = NOW + timedelta(minutes=5, seconds=1)
        empty = self.allocator.allocate([], higher_quotes, self.broker.equity(higher_quotes), later)
        close_plan = self.planner.plan(
            empty, self.broker.snapshot(higher_quotes, later), higher_quotes
        )
        fills = self.broker.execute_plan(close_plan, higher_quotes)

        self.assertGreater(sum(fill.realized_pnl_usd for fill in fills), 0)
        self.assertGreater(self.broker.cash, 10_000)

    def test_cycle_engine_owns_state_across_repeated_cycles(self) -> None:
        engine = PaperTradingEngine(
            allocator=self.allocator,
            planner=self.planner,
            broker=self.broker,
        )
        predictions = [
            signal("EUR_USD", 1, 0.003),
            signal("XAU_USD", -1, -0.004),
        ]
        first = engine.run_cycle(predictions, self.quotes, NOW)
        second = engine.run_cycle(predictions, self.quotes, NOW)

        self.assertEqual(len(first.fills), 2)
        self.assertTrue(first.reconciliation.in_sync)
        self.assertEqual(second.order_plan.orders, ())
        self.assertTrue(second.reconciliation.in_sync)


if __name__ == "__main__":
    unittest.main()
