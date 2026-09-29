from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from maestro.shared.contracts import PredictionSignal
from maestro.trader.portfolio_allocator import MarketQuote, PortfolioAllocator, PortfolioLimits


NOW = datetime(2025, 1, 6, 12, 0, tzinfo=timezone.utc)


def signal(
    instrument: str,
    direction: int = 1,
    expected_return: float = 0.003,
    confidence: float = 0.70,
    uncertainty: float = 0.25,
    timestamp: datetime = NOW,
) -> PredictionSignal:
    return PredictionSignal(
        instrument=instrument,
        timestamp=timestamp,
        data_as_of=timestamp,
        expires_at=timestamp + timedelta(minutes=30),
        horizon_bars=6,
        direction=direction,
        expected_return=expected_return,
        confidence=confidence,
        uncertainty=uncertainty,
        model_version=f"{instrument.lower()}-v1",
    )


def quote(
    instrument: str,
    mid: float,
    spread: float,
    stop_distance_pct: float = 0.005,
    timestamp: datetime = NOW - timedelta(seconds=1),
    tradeable: bool = True,
) -> MarketQuote:
    return MarketQuote(
        instrument=instrument,
        timestamp=timestamp,
        bid=mid - spread / 2,
        ask=mid + spread / 2,
        stop_distance_pct=stop_distance_pct,
        tradeable=tradeable,
    )


class PortfolioAllocatorTests(unittest.TestCase):
    def test_gold_receives_its_separate_risk_budget(self) -> None:
        result = PortfolioAllocator().allocate(
            [signal("XAU_USD")],
            {"XAU_USD": quote("XAU_USD", 2_000.0, 0.20, stop_distance_pct=0.01)},
            equity=10_000,
            as_of=NOW,
        )

        self.assertEqual(len(result.targets), 1)
        self.assertEqual(result.targets[0].instrument, "XAU_USD")
        self.assertLessEqual(result.metal_risk_usd, 25.0)
        self.assertEqual(result.fx_risk_usd, 0.0)

    def test_same_direction_usd_bets_hit_currency_cap(self) -> None:
        predictions = [signal("EUR_USD"), signal("GBP_USD")]
        quotes = {
            "EUR_USD": quote("EUR_USD", 1.10, 0.0002),
            "GBP_USD": quote("GBP_USD", 1.25, 0.0002),
        }
        result = PortfolioAllocator().allocate(predictions, quotes, 10_000, NOW)

        self.assertEqual(len(result.targets), 1)
        self.assertLessEqual(abs(result.net_exposure_usd["USD"]), 5_000.0)
        self.assertIn("currency_exposure_limit", {item.reason for item in result.rejected})

    def test_opposing_usd_exposure_allows_fx_and_gold(self) -> None:
        predictions = [
            signal("EUR_USD", direction=1, expected_return=0.003),
            signal("XAU_USD", direction=-1, expected_return=-0.004),
        ]
        quotes = {
            "EUR_USD": quote("EUR_USD", 1.10, 0.0002),
            "XAU_USD": quote("XAU_USD", 2_000.0, 0.20, stop_distance_pct=0.01),
        }
        result = PortfolioAllocator().allocate(predictions, quotes, 10_000, NOW)

        self.assertEqual({target.instrument for target in result.targets}, {"EUR_USD", "XAU_USD"})
        self.assertLessEqual(result.total_risk_usd, 100.0)

    def test_predictions_are_ranked_and_position_count_is_limited(self) -> None:
        limits = PortfolioLimits(
            max_open_positions=2,
            max_net_currency_exposure_pct=5.0,
            max_gross_currency_exposure_pct=10.0,
        )
        predictions = [
            signal("EUR_USD", expected_return=0.005),
            signal("GBP_USD", expected_return=0.004),
            signal("AUD_USD", expected_return=0.003),
        ]
        quotes = {
            "EUR_USD": quote("EUR_USD", 1.10, 0.0001),
            "GBP_USD": quote("GBP_USD", 1.25, 0.0001),
            "AUD_USD": quote("AUD_USD", 0.70, 0.0001),
        }
        result = PortfolioAllocator(limits).allocate(predictions, quotes, 10_000, NOW)

        self.assertEqual([target.instrument for target in result.targets], ["EUR_USD", "GBP_USD"])
        self.assertIn("position_limit", {item.reason for item in result.rejected})

    def test_stale_prediction_and_quote_are_rejected(self) -> None:
        old = NOW - timedelta(hours=1)
        expired = PredictionSignal(
            instrument="EUR_USD",
            timestamp=old,
            data_as_of=old,
            expires_at=old + timedelta(minutes=30),
            horizon_bars=6,
            direction=1,
            expected_return=0.003,
            confidence=0.7,
            uncertainty=0.2,
            model_version="old-v1",
        )
        predictions = [expired, signal("GBP_USD")]
        quotes = {
            "EUR_USD": quote("EUR_USD", 1.10, 0.0001),
            "GBP_USD": quote(
                "GBP_USD", 1.25, 0.0001, timestamp=NOW - timedelta(minutes=1)
            ),
        }
        result = PortfolioAllocator().allocate(predictions, quotes, 10_000, NOW)

        reasons = {item.instrument: item.reason for item in result.rejected}
        self.assertEqual(reasons["EUR_USD"], "stale_prediction")
        self.assertEqual(reasons["GBP_USD"], "stale_quote")
        self.assertEqual(result.targets, ())

    def test_spread_and_slippage_can_remove_the_edge(self) -> None:
        result = PortfolioAllocator().allocate(
            [signal("EUR_USD", expected_return=0.0002)],
            {"EUR_USD": quote("EUR_USD", 1.10, 0.0003)},
            10_000,
            NOW,
        )
        self.assertEqual(result.targets, ())
        self.assertEqual(result.rejected[0].reason, "insufficient_net_edge")

    def test_mismatched_quote_fails_closed(self) -> None:
        result = PortfolioAllocator().allocate(
            [signal("EUR_USD")],
            {"EUR_USD": quote("GBP_USD", 1.25, 0.0001)},
            10_000,
            NOW,
        )
        self.assertEqual(result.targets, ())
        self.assertEqual(result.rejected[0].reason, "quote_instrument_mismatch")


if __name__ == "__main__":
    unittest.main()
