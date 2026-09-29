"""
Tests for the power test's planted edge (maestro/backtesting/power_test.py).
"""
import numpy as np
import pandas as pd
import pytest

from maestro.backtesting.power_test import calibrate, oracle_hit, plant, planted_candles


def _returns(n=60_000, seed=5):
    return np.random.default_rng(seed).standard_t(4, n) * 1e-4   # fat tails, like FX


def test_zero_beta_plants_nothing():
    r = _returns()
    assert np.allclose(plant(r, 0.0), r)
    assert 0.49 < oracle_hit(r, 0.0) < 0.51


@pytest.mark.parametrize("target", [0.52, 0.55])
def test_calibration_hits_the_target(target):
    r = _returns()
    assert oracle_hit(r, calibrate(r, target)) == pytest.approx(target, abs=0.003)


def test_planted_candles_follow_the_planted_returns():
    r = _returns(2000)
    close = 1.1 * np.exp(np.cumsum(r))
    idx = pd.date_range("2024-01-01", periods=len(r), freq="5min", tz="UTC")
    raw = pd.DataFrame({"open": np.r_[1.1, close[:-1]], "close": close}, index=idx)
    raw["high"] = raw[["open", "close"]].max(axis=1) + 2e-5
    raw["low"] = raw[["open", "close"]].min(axis=1) - 2e-5
    out = planted_candles(raw, 0.3)
    got = np.diff(np.log(out["close"].to_numpy()))
    want = plant(np.r_[0.0, np.diff(np.log(close))], 0.3)[1:]
    assert np.allclose(got, want)
    assert (out["high"] >= out[["open", "close"]].max(axis=1)).all()
    assert (out["low"] <= out[["open", "close"]].min(axis=1)).all()
