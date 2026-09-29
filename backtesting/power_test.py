"""
maestro/backtesting/power_test.py
=================================
Could the pipeline have found an edge if there was one?

"MAESTRO has no edge" only means something if the same pipeline says "yes"
when an edge exists. This plants a known momentum effect in real EUR/USD
prices, rebuilds every feature from the altered prices, and runs MAESTRO and
the baselines through exactly the same retraining plan and evaluator:

    r'_t = r_t + beta * mean(r'_{t-6} ... r'_{t-1})

The effect lives on the 30-minute scale MAESTRO forecasts at, and everything
needed to see it (recent returns, momentum indicators) is among its inputs.
beta is set so that a trader who knew the rule would call the next bar's
direction right `oracle_hit` of the time; that trader is scored as "oracle".
Level 0 plants nothing, so the same window is also the null comparison.

MAESTRO passes if its hit rate and pre-cost Sharpe rise with the planted edge.
Logistic regression on the last five moves can express the rule exactly, so it
should recover the edge too: if it can't, the plant is too weak to judge.

    python -m maestro.backtesting.power_test                       # levels 0, 0.52, 0.55
    python -m maestro.backtesting.power_test --levels 0.55 --blocks 1 --epochs 1   # quick check
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from maestro.backtesting.baselines import OUTPUT_DIR
from maestro.backtesting.maestro_runner import EPOCHS, print_summary, run_design

logger = logging.getLogger(__name__)

WINDOW = ("2024-01-01", "2025-12-31")   # before the sealed holdout; tests 2025
LOOKBACK = 6                            # bars in the planted momentum (30 minutes)


def plant(r: np.ndarray, beta: float) -> np.ndarray:
    """r'_t = r_t + beta * mean of the previous LOOKBACK planted returns (a linear filter)."""
    return lfilter([1.0], np.r_[1.0, -np.full(LOOKBACK, beta / LOOKBACK)], r)


def momentum(r_planted: np.ndarray) -> np.ndarray:
    """What the rule knows at bar t: the mean of the last LOOKBACK returns, including t."""
    return pd.Series(r_planted).rolling(LOOKBACK).mean().to_numpy()


def oracle_hit(r: np.ndarray, beta: float) -> float:
    rp = plant(r, beta)
    m = momentum(rp)[:-1]
    nxt = rp[1:]
    ok = (m != 0) & (nxt != 0) & np.isfinite(m)
    return float((np.sign(m[ok]) == np.sign(nxt[ok])).mean())


def calibrate(r: np.ndarray, target: float) -> float:
    """Smallest beta whose oracle hit reaches target (bisection; hit rises with beta)."""
    if target <= 0.5:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if oracle_hit(r, mid) < target else (lo, mid)
    return hi


def planted_candles(raw: pd.DataFrame, beta: float) -> pd.DataFrame:
    """Candles whose closes follow the planted returns; bars keep their own shape."""
    logc = np.log(raw["close"].to_numpy())
    r = np.r_[0.0, np.diff(logc)]
    shift = np.cumsum(plant(r, beta) - r)                  # log distance from the real path
    out = raw.copy()
    prev = np.r_[0.0, shift[:-1]]
    out["close"] = raw["close"] * np.exp(shift)
    out["open"] = raw["open"] * np.exp(prev)
    out["high"] = np.maximum(raw["high"] * np.exp(shift), out[["open", "close"]].max(axis=1))
    out["low"] = np.minimum(raw["low"] * np.exp(shift), out[["open", "close"]].min(axis=1))
    for col in ("bid_close", "ask_close"):
        if col in out:
            out[col] = raw[col] * np.exp(shift)
    return out


def run_level(level: float, raw: pd.DataFrame, blocks: int | None, epochs: int) -> pd.DataFrame:
    from maestro.data.pipeline.store import build_features

    r = np.r_[0.0, np.diff(np.log(raw["close"].to_numpy()))]
    beta = calibrate(r, level)
    candles = planted_candles(raw, beta)
    logger.info("level %.3f: beta %.4f, oracle hit %.4f", level, beta, oracle_hit(r, beta))

    feats = build_features(candles)
    close = feats["close"]
    rp = np.log(close / close.shift(1)).fillna(0.0).to_numpy()
    oracle = pd.Series(np.sign(np.nan_to_num(momentum(rp))), index=close.index)

    out_dir = OUTPUT_DIR / "power" / f"level_{level:.3f}"
    pooled = run_design(close, out_dir, "EUR_USD", lambda: feats, epochs=epochs,
                        max_blocks=blocks, train=True)
    if pooled is None:
        return pd.DataFrame()
    # Score the oracle with the same code, alongside MAESTRO and the baselines.
    from maestro.backtesting.baselines import run as score
    maestro = pooled[pooled["strategy"].str.startswith("maestro")]
    oracle_pooled, _ = score("EUR_USD", strategies=["logreg_lag5", "buy_hold", "random"],
                             max_splits=int(pd.read_csv(out_dir / "scores" / "EUR_USD_baselines_per_split.csv")
                                            ["split_id"].max()) + 1,
                             external={"oracle": oracle}, out_dir=out_dir / "scores_oracle", close=close)
    res = pd.concat([maestro, oracle_pooled], ignore_index=True)
    res.insert(0, "level", level)
    res.insert(1, "beta", beta)
    return res


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description="Power test: plant a known edge, see if MAESTRO finds it")
    p.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.52, 0.55],
                   help="oracle hit rates to plant (0 = no edge)")
    p.add_argument("--blocks", type=int, default=None, help="limit retraining blocks per level")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    args = p.parse_args()

    from maestro.data.pipeline.store import TARGET, load_candles
    raw = load_candles(TARGET).loc[WINDOW[0]:WINDOW[1]]
    results = [run_level(level, raw, args.blocks, args.epochs) for level in args.levels]
    table = pd.concat(results, ignore_index=True)
    path = OUTPUT_DIR / "power" / "summary.csv"
    table.to_csv(path, index=False)
    for level, g in table.groupby("level"):
        print(f"\n=== planted oracle hit {level:.3f} ===")
        print_summary(g)
        o = g[(g["strategy"] == "oracle") & (g["cost"] == "gross")].iloc[0]
        print(f"oracle            {o['n_trades']:>9.0f}{o['hit_directional'] * 100:>7.1f}%{o['sharpe']:>10.2f}")
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
