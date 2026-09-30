"""
The live trial's decisions (maestro/live/strategies.py) must be exactly the
backtest's decisions at the same bar, or the trial measures a different system.
"""
import numpy as np
import pandas as pd

from maestro.backtesting.maestro_runner import positions
from maestro.backtesting.risk_layer import risk_positions
from maestro.live.strategies import maestro_targets


def test_live_maestro_rules_match_the_backtest_bar_for_bar():
    rng = np.random.default_rng(4)
    n = 2500
    idx = pd.date_range("2024-03-04", periods=n, freq="5min", tz="UTC")
    close = pd.Series(1.08 + np.cumsum(rng.normal(0, 1e-4, n)), index=idx)
    signals = pd.DataFrame({
        "signal": rng.choice([-1, 0, 1], n),
        "confidence": rng.uniform(0, 0.7, n),
        "regime": rng.choice([0, 1, 2, 3], n),
        "pred_p50": rng.normal(0, 2e-4, n),
    }, index=idx)
    back = {**positions(signals, close.index), **risk_positions(signals, close)}
    for i in range(300, n, 37):
        t = idx[i]
        live = maestro_targets(signals.iloc[i], signals["confidence"].iloc[:i], float(close.iloc[i]))
        for name, value in live.items():
            assert value == back[name].loc[t], (name, t)
