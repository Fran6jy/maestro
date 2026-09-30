"""
The live paper ledger (maestro/live/ledger.py) must score positions exactly as
the shared evaluator does, so paper results and backtest results are comparable.
"""
import numpy as np
import pandas as pd
import pytest

from maestro.backtesting.baselines import simulate, summarise
from maestro.live.ledger import Journal

PIP = 1e-4


def test_paper_ledger_matches_the_evaluator(tmp_path):
    rng = np.random.default_rng(2)
    n = 400
    idx = pd.date_range("2024-03-04", periods=n, freq="5min", tz="UTC")
    close = pd.Series(1.08 + np.cumsum(rng.normal(0, 1e-4, n)), index=idx)
    target = pd.Series(rng.choice([-1.0, 0.0, 1.0, 0.5], n), index=idx)
    target.iloc[-1] = 0.0                                            # end flat, as a test window does
    j = Journal(tmp_path / "j.db")
    for t in idx:
        c = float(close[t])
        state = j.step(t, "s", float(target[t]), mid=c, bid=c - 0.4 * PIP, ask=c + 0.4 * PIP)
    res = simulate(target, close, PIP)
    for cost in (0.0, 0.8):
        expected = summarise([res], cost_pips=cost, pip=PIP)["net_pips_total"]
        got = state["gross_pips"] - (state["cost_ref_pips"] if cost else 0.0)
        assert got == pytest.approx(expected)
    assert state["cost_quoted_pips"] == pytest.approx(state["cost_ref_pips"])   # 0.8-pip quotes here
    assert state["trades"] == len(res.trades)


def test_confidence_history_is_in_time_order(tmp_path):
    j = Journal(tmp_path / "j.db")
    for k, t in enumerate(pd.date_range("2024-03-04", periods=5, freq="5min", tz="UTC")[::-1]):
        j.record_forecast(t, {"confidence": float(k)})
    hist = j.confidence_history(10)
    assert hist.index.is_monotonic_increasing and list(hist) == [4.0, 3.0, 2.0, 1.0, 0.0]
