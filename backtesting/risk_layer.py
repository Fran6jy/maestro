"""
maestro/backtesting/risk_layer.py
=================================
Step 3: does MAESTRO's risk agent turn its small edge into a profit?

Applied to MAESTRO's saved forecasts (no retraining), scored with the shared
evaluator like every other strategy. The variants below were fixed before any
of them was run:

  risk_as_designed   the risk agent's own gates in front of every signal: no trade
                     below 0.52 confidence, or below 0.62 in the crisis regime.
                     MAESTRO's calibrated confidence never reaches 0.52, so this is
                     expected to make no trades, like maestro_gated.
  risk_cost_filter   every signal, but only when the forecast move in the trade's
                     direction (the TFT's median 6-bar forecast, in pips) is at least
                     the round-trip cost plus the agent's own 0.5-pip minimum edge
                     (0.8 + 0.5 = 1.3 pips at the reference cost).

Kelly sizing (the agent's layer C) depends on the strategy's own recent profits,
so it needs a bar-by-bar simulation and is added separately.

    python -m maestro.backtesting.risk_layer --refit-months 3 --train-months 12 --fast
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from maestro.backtesting.baselines import (
    OUTPUT_DIR, PIP_SIZE, default_cost_scenarios, design_tag, load_close, refit_plan, run as score,
)
from maestro.backtesting.maestro_runner import MIN_TRAIN_BARS, position_windows

logger = logging.getLogger(__name__)

MIN_EDGE_PIPS = 0.5                  # risk_agent.HARD_LIMITS["min_edge_pips"]
MIN_CONFIDENCE = 0.52                # risk_agent.HARD_LIMITS["min_confidence"]
CRISIS, CRISIS_CONFIDENCE = 3, 0.62  # risk_agent._check_hard_gates


def risk_positions(signals: pd.DataFrame, close: pd.Series, instrument: str = "EUR_USD",
                   granularity: str = "M5") -> dict[str, pd.Series]:
    sig = signals["signal"].astype(float)
    conf = signals["confidence"]
    regime = signals["regime"].astype(int)
    gate = (conf >= MIN_CONFIDENCE) & ~((regime == CRISIS) & (conf < CRISIS_CONFIDENCE))

    cost = default_cost_scenarios(instrument)["spread"]
    move_pips = signals["pred_p50"] * close.reindex(signals.index) / PIP_SIZE[instrument]
    worth_it = sig * move_pips >= cost + MIN_EDGE_PIPS

    bars = close.index[(close.index >= signals.index[0]) & (close.index <= signals.index[-1])]
    max_carry = position_windows(granularity)[2]
    carry = lambda s: s.reindex(bars).ffill(limit=max_carry).fillna(0.0)
    return {"risk_as_designed": carry(sig.where(gate, 0.0)),
            "risk_cost_filter": carry(sig.where(worth_it, 0.0))}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description="Score MAESTRO's risk layers on saved forecasts")
    p.add_argument("--instrument", default="EUR_USD")
    p.add_argument("--granularity", default="M5")
    p.add_argument("--refit-months", type=int, default=3)
    p.add_argument("--train-months", type=int, default=12)
    p.add_argument("--expanding", action="store_true")
    p.add_argument("--fast", action="store_true", help="the design was trained with --fast")
    args = p.parse_args()

    train_months = None if args.expanding else args.train_months
    prefix = "" if args.granularity == "M5" else f"{args.granularity}_"
    tag = prefix + design_tag(args.refit_months, train_months) + ("_fast" if args.fast else "")
    design = OUTPUT_DIR / "maestro" / tag
    close = load_close(args.instrument, args.granularity)
    min_bars = MIN_TRAIN_BARS[args.granularity]
    blocks = refit_plan(close, args.refit_months, train_months, min_train_bars=min_bars)
    files = [design / f"block_{b.block_id:02d}.parquet" for b in blocks]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"{tag}: {len(missing)} blocks missing, e.g. {missing[:3]}")
    signals = pd.concat(pd.read_parquet(f) for f in files).sort_index()

    pooled, _ = score(args.instrument, strategies=["logreg_lag5", "bollinger_20_2", "buy_hold"],
                      refit_months=args.refit_months, train_months=train_months,
                      external=risk_positions(signals, close, args.instrument, args.granularity),
                      out_dir=design / "scores_risk", close=close, min_train_bars=min_bars)
    g = pooled[pooled["cost"] == "gross"].set_index("strategy")
    n = pooled[pooled["cost"] == "spread"].set_index("strategy")
    print(f"\n{'strategy':<18}{'trades':>9}{'hit':>8}{'pips/tr gross':>15}{'gross Sh':>10}{'net Sh':>9}")
    for s in g.index:
        hit = n.loc[s, "hit_directional"]
        print(f"{s:<18}{n.loc[s, 'n_trades']:>9.0f}{hit * 100 if hit == hit else float('nan'):>7.1f}%"
              f"{g.loc[s, 'net_pips_per_trade']:>15.3f}{g.loc[s, 'sharpe']:>10.2f}{n.loc[s, 'sharpe']:>9.2f}")


if __name__ == "__main__":
    main()
