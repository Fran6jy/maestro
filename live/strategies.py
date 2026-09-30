"""
maestro/live/strategies.py
==========================
Each strategy's target position at the latest bar, decided with the backtest's own rules.

MAESTRO's variants use the same rules as backtesting/maestro_runner.positions and
backtesting/risk_layer.risk_positions; the baselines call the very functions in
baselines.STRATEGIES, fitted on the same training window the live model was trained
on. A strategy's live position is the decision those rules give at the latest bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from maestro.backtesting.baselines import PIP_SIZE, STRATEGIES, default_cost_scenarios
from maestro.backtesting.maestro_runner import TOP_SHARE, position_windows
from maestro.backtesting.risk_layer import MIN_EDGE_PIPS

MAESTRO_VARIANTS = ("maestro_gated", "maestro_top10", "maestro_ungated", "risk_cost_filter")
BASELINES = ("logreg_lag5", "bollinger_20_2", "sma_20_200", "contrarian_3", "buy_hold")


def maestro_targets(forecast: pd.Series, confidence_history: pd.Series, close: float,
                    instrument: str = "EUR_USD") -> dict[str, float]:
    """MAESTRO's variants at one bar, from its forecast and the confidence of earlier bars."""
    from maestro.agents.signal.signal_agent import CONFIDENCE_THRESHOLDS
    sig = float(forecast["signal"])
    conf = float(forecast["confidence"])
    threshold = CONFIDENCE_THRESHOLDS.get(int(forecast["regime"]), 0.55)
    lookback, min_history, _ = position_windows("M5")
    past = confidence_history.tail(lookback)
    cutoff = past.quantile(1 - TOP_SHARE) if len(past) >= min_history else np.inf
    cost = default_cost_scenarios(instrument)["spread"]
    move_pips = float(forecast["pred_p50"]) * close / PIP_SIZE[instrument]
    return {
        "maestro_gated": sig if conf >= threshold else 0.0,
        "maestro_top10": sig if conf >= cutoff else 0.0,
        "maestro_ungated": sig,
        "risk_cost_filter": sig if sig * move_pips >= cost + MIN_EDGE_PIPS else 0.0,
    }


def baseline_targets(close: pd.Series, train_idx: pd.Index, recent: int = 600) -> dict[str, float]:
    """Each baseline's position at the last bar of `close`, fitted on train_idx as in the backtest."""
    returns = np.log(close / close.shift(1))
    test_idx = close.index[close.index > train_idx[-1]][-recent:]
    out = {}
    for name in BASELINES:
        pos = STRATEGIES[name](close, returns, train_idx, test_idx, 0)
        out[name] = float(pos.iloc[-1])
    return out
