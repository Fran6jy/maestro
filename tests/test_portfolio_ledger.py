from __future__ import annotations

import pandas as pd
import unittest
from datetime import datetime, timedelta, timezone
import os
import tempfile

from maestro.backtesting.portfolio_ledger import CausalPortfolioLedger, LedgerConfig
from maestro.config.instruments import get_instrument_spec
from maestro.shared.contracts import PredictionSignal


def _bars(rows: list[tuple[float, float]]) -> pd.DataFrame:
    index = pd.date_range("2025-01-06", periods=len(rows), freq="5min", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "close"], index=index)


def _decisions(index: pd.Index, units: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "action": ["trade" if value else "flat" for value in units],
            "units": units,
            "compliant": True,
            "final_signal": [1 if value > 0 else -1 if value < 0 else 0 for value in units],
            "regime": 0,
        },
        index=index,
    )


class PortfolioLedgerTests(unittest.TestCase):
    def test_decision_cannot_earn_its_own_bar_return(self) -> None:
        bars = _bars([(1.00, 1.10), (1.11, 1.12), (1.12, 1.13)])
        decisions = _decisions(bars.index, [1_000, 1_000, 0])
        ledger = CausalPortfolioLedger(
            "EUR_USD", LedgerConfig(initial_equity=10_000, force_close=False)
        ).run(decisions, bars)

        self.assertEqual(ledger.iloc[0]["gross_pnl"], 0)
        self.assertEqual(ledger.iloc[1]["decision_timestamp"], bars.index[0])
        self.assertAlmostEqual(ledger.iloc[1]["gross_pnl"], 10.0)
        self.assertNotAlmostEqual(ledger.iloc[1]["gross_pnl"], 100.0)

    def test_unchanged_target_does_not_pay_repeated_entry_cost(self) -> None:
        bars = _bars([(1.00, 1.00), (1.00, 1.00), (1.00, 1.00), (1.00, 1.00)])
        decisions = _decisions(bars.index, [1_000, 1_000, 1_000, 0])
        ledger = CausalPortfolioLedger(
            "EUR_USD", LedgerConfig(initial_equity=10_000, force_close=False)
        ).run(decisions, bars)

        self.assertEqual(int(ledger["executed"].sum()), 1)
        self.assertGreater(ledger.iloc[1]["transaction_cost"], 0)
        self.assertEqual(ledger.iloc[2]["transaction_cost"], 0)

    def test_gold_is_a_first_class_instrument(self) -> None:
        spec = get_instrument_spec("XAU_USD")
        self.assertEqual(spec.asset_class, "metal")
        self.assertEqual(spec.pip_size, 0.01)

        bars = _bars([(2_000.0, 2_000.0), (2_001.0, 2_002.0)])
        decisions = _decisions(bars.index, [2, 2])
        ledger = CausalPortfolioLedger(
            "XAU_USD", LedgerConfig(initial_equity=10_000, force_close=False)
        ).run(decisions, bars)
        self.assertAlmostEqual(ledger.iloc[1]["gross_pnl"], 2.0)

    def test_jpy_uses_jpy_pip_size(self) -> None:
        self.assertEqual(get_instrument_spec("USD_JPY").pip_size, 0.01)

    def test_prediction_contract_cannot_contain_an_expired_signal(self) -> None:
        timestamp = datetime(2025, 1, 6, 12, 0, tzinfo=timezone.utc)
        signal = PredictionSignal(
            instrument="XAU_USD",
            timestamp=timestamp,
            data_as_of=timestamp,
            expires_at=timestamp + timedelta(minutes=30),
            horizon_bars=6,
            direction=1,
            expected_return=0.001,
            confidence=0.63,
            uncertainty=0.37,
            model_version="gold-v1",
        )
        self.assertTrue(signal.is_fresh(timestamp + timedelta(minutes=5)))
        self.assertFalse(signal.is_fresh(timestamp + timedelta(minutes=31)))
        self.assertNotIn("units", signal.to_dict())

    def test_backtest_metrics_count_fills_not_active_bars(self) -> None:
        test_root = os.path.join(tempfile.gettempdir(), "maestro-ledger-tests")
        os.environ["MAESTRO_MODEL_DIR"] = os.path.join(test_root, "models")
        os.environ["MAESTRO_OUTPUT_DIR"] = os.path.join(test_root, "outputs")
        from maestro.backtesting.backtest_engine import BacktestConfig, BacktestEngine

        bars = _bars([(1.0, 1.0)] * 12)
        bars["log_return_1"] = 0.0
        decisions = _decisions(bars.index, [1_000] * len(bars))
        ledger = CausalPortfolioLedger(
            "EUR_USD", LedgerConfig(initial_equity=10_000, force_close=False)
        ).run(decisions, bars)
        engine = BacktestEngine(BacktestConfig(account_equity=10_000))
        result = engine._compute_split_metrics(
            ledger, ledger, bars, pd.DataFrame(index=bars.index), 0, "EUR_USD"
        )

        self.assertEqual(result.n_trades, 1)
        self.assertLess(result.cum_return_net, 0.0)  # entry cost, no price edge


if __name__ == "__main__":
    unittest.main()
