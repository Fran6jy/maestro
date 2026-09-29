"""
Tests for the shared evaluator in maestro/backtesting/baselines.py.

These pin down the timing convention (a position decided at bar t earns the
return of bar t+1) and the cost bookkeeping, so no strategy — MSc or MAESTRO —
can score well by accidentally seeing the future.
"""
import numpy as np
import pandas as pd
import pytest

from maestro.backtesting.baselines import STRATEGIES, simulate, summarise

PIP = 1e-4


def _random_walk(n: int = 3000, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.Series(1.10 + np.cumsum(rng.normal(0, 1e-4, n)), index=idx)


def test_perfect_foresight_scores_100pct():
    close = _random_walk()
    r = np.log(close / close.shift(1))
    pos = np.sign(r.shift(-1)).fillna(0.0)          # deliberately peeks one bar ahead
    s = summarise([simulate(pos, close, PIP)], cost_pips=0.0, pip=PIP)
    assert s["hit_directional"] == pytest.approx(1.0)


def test_same_bar_signal_is_not_rewarded():
    # A signal built from the bar's OWN return must be useless on a random walk.
    close = _random_walk()
    r = np.log(close / close.shift(1))
    pos = np.sign(r).fillna(0.0)
    s = summarise([simulate(pos, close, PIP)], cost_pips=0.0, pip=PIP)
    assert 0.45 < s["hit_directional"] < 0.55


def test_buy_and_hold_pnl_equals_price_change():
    close = _random_walk()
    res = simulate(pd.Series(1.0, index=close.index), close, PIP)
    assert res.bars["pnl_pips"].sum() == pytest.approx((close.iloc[-1] - close.iloc[0]) / PIP)
    assert len(res.trades) == 1


def test_one_round_trip_cost_per_trade():
    close = _random_walk(n=500)
    pos = pd.Series(np.tile([1.0, 1.0, -1.0, -1.0, 0.0], 100), index=close.index)
    res = simulate(pos, close, PIP)
    gross = summarise([res], cost_pips=0.0, pip=PIP)["net_pips_total"]
    net = summarise([res], cost_pips=1.0, pip=PIP)["net_pips_total"]
    assert gross - net == pytest.approx(len(res.trades) * 1.0)
    assert res.bars["turnover"].sum() == pytest.approx(2 * len(res.trades))


def test_msc_style_hit_counts_flat_bars_as_misses():
    idx = pd.date_range("2024-01-01", periods=5, freq="5min", tz="UTC")
    close = pd.Series([1.0, 1.0001, 1.0001, 1.0002, 1.0002], index=idx)
    pos = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], index=idx)
    s = summarise([simulate(pos, close, PIP)], cost_pips=0.0, pip=PIP)
    # held during bars 1..4 (closed at bar 4's close): moves +, 0, +, 0
    assert s["hit_directional"] == pytest.approx(1.0)       # 2/2 bars that moved
    assert s["hit_msc_style"] == pytest.approx(2 / 4)       # flat bars counted as misses


@pytest.mark.parametrize("name", ["linreg_lag1", "linreg_lag5", "logreg_lag5"])
def test_ml_baselines_only_learn_from_train(name):
    # Train on a mean-reverting segment, test on a trending one: if the model
    # were fitted on test data its hit rate on the trend would be near 100%.
    rng = np.random.default_rng(3)
    n_train, n_test = 2000, 500
    idx = pd.date_range("2024-01-01", periods=n_train + n_test, freq="5min", tz="UTC")
    r_train = rng.normal(0, 1e-4, n_train)
    r_train[1:] -= 0.6 * r_train[:-1]                       # negative autocorrelation
    r_test = np.abs(rng.normal(0, 1e-4, n_test)) + 5e-5    # always up
    close = pd.Series(1.1 * np.exp(np.cumsum(np.r_[r_train, r_test])), index=idx)
    returns = np.log(close / close.shift(1))
    pos = STRATEGIES[name](close, returns, idx[:n_train], idx[n_train:], 0)
    assert pos.index.equals(idx[n_train:])
    s = summarise([simulate(pos, close, PIP)], cost_pips=0.0, pip=PIP)
    assert s["hit_directional"] < 0.9
