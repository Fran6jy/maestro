from __future__ import annotations

from datetime import datetime, timezone
import unittest

from maestro.trader.oanda_readonly import HedgedPositionError, OANDAReadOnlyAdapter


class FakeOANDAClient:
    def __init__(self, positions=None, nav=12_500.0) -> None:
        self.positions = positions or []
        self.nav = nav

    def get_open_positions(self):
        return self.positions

    def get_account_summary(self):
        return {"nav": self.nav}


class OANDAReadOnlyAdapterTests(unittest.TestCase):
    def test_long_and_short_payloads_become_net_positions(self) -> None:
        raw = [
            {
                "instrument": "EUR_USD",
                "long": {"units": "350", "averagePrice": "1.13032"},
                "short": {"units": "0"},
            },
            {
                "instrument": "USD_CAD",
                "long": {"units": "0"},
                "short": {"units": "-600", "averagePrice": "1.28241"},
            },
        ]
        adapter = OANDAReadOnlyAdapter(FakeOANDAClient(raw))
        snapshot = adapter.snapshot(datetime(2025, 1, 6, tzinfo=timezone.utc))

        self.assertEqual(snapshot.positions["EUR_USD"].units, 350)
        self.assertEqual(snapshot.positions["USD_CAD"].units, -600)
        self.assertAlmostEqual(snapshot.positions["USD_CAD"].average_price, 1.28241)
        self.assertEqual(adapter.account_equity(), 12_500.0)

    def test_hedged_position_fails_closed(self) -> None:
        raw = [{
            "instrument": "EUR_USD",
            "long": {"units": "100", "averagePrice": "1.10"},
            "short": {"units": "-50", "averagePrice": "1.11"},
        }]
        with self.assertRaises(HedgedPositionError):
            OANDAReadOnlyAdapter.parse_positions(raw)

    def test_price_tick_becomes_market_quote(self) -> None:
        tick = {
            "instrument": "XAU_USD",
            "time": "2025-01-06T12:00:00Z",
            "bid": 1999.90,
            "ask": 2000.10,
            "tradeable": True,
        }
        result = OANDAReadOnlyAdapter.quote_from_tick(tick, stop_distance_pct=0.01)
        self.assertEqual(result.instrument, "XAU_USD")
        self.assertAlmostEqual(result.mid, 2000.0)
        self.assertEqual(result.timestamp.tzinfo, timezone.utc)

    def test_adapter_exposes_no_order_submission_method(self) -> None:
        adapter = OANDAReadOnlyAdapter(FakeOANDAClient())
        self.assertFalse(hasattr(adapter, "place_market_order"))
        self.assertFalse(hasattr(adapter, "execute"))


if __name__ == "__main__":
    unittest.main()

