"""
maestro/live/expectations.py
============================
What the 20-year backtest says each strategy should do in the live trial, written
down before the trial starts.

For every strategy and every trial length from 1 to HORIZON trading days, this takes
every run of that many consecutive days in the backtest and records the spread of
outcomes (5th, 25th, 50th, 75th and 95th percentiles) at the reference cost. The
live page draws these as the "usual range" a live result should land in. Committing
the file before the trial starts is what makes it a prediction rather than a
rationalisation: git history shows it was there first.

Days are UTC calendar days with at least one bar, as in the backtest's daily series
and in live/snapshot, so a live day and a backtest day are the same unit.

    python -m maestro.live.expectations                       # frozen design's scores
    python -m maestro.live.expectations --results <scores dir> --risk <scores_risk dir>
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from maestro.backtesting.baselines import OUTPUT_DIR
from maestro.demo.export_web_data import MAESTRO_TEXT, STRATEGY_TEXT

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "src" / "data" / "live_expectations.json"
DESIGN = OUTPUT_DIR / "maestro" / "refit3_roll12_fast"
HORIZON = 65                       # about three months of trading days
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
START = 10_000                     # the site's starting balance


def ranges(daily: pd.Series) -> list[list[float]]:
    """Percentiles of the balance change over every run of n consecutive days, n = 1..HORIZON."""
    x = daily.to_numpy()
    csum = np.concatenate([[0.0], np.cumsum(x)])
    out = []
    for n in range(1, HORIZON + 1):
        sums = csum[n:] - csum[:-n]
        q = np.quantile(START * (np.exp(sums) - 1.0), QUANTILES)
        out.append([round(float(v), 2) for v in q])
    return out


def build(results: Path, risk: Path, instrument: str = "EUR_USD") -> dict:
    daily = pd.read_csv(results / f"{instrument}_baselines_daily.csv", index_col=0).fillna(0.0)
    pooled = pd.read_csv(results / f"{instrument}_baselines_pooled.csv")
    if (risk / f"{instrument}_baselines_daily.csv").exists():
        rd = pd.read_csv(risk / f"{instrument}_baselines_daily.csv", index_col=0).fillna(0.0)
        daily = daily.join(rd[[c for c in rd.columns if c.startswith("risk_")]], how="left").fillna(0.0)
        rp = pd.read_csv(risk / f"{instrument}_baselines_pooled.csv")
        pooled = pd.concat([pooled, rp[rp["strategy"].str.startswith("risk_")]])
    net = pooled[pooled["cost"] == "spread"].set_index("strategy")
    gross = pooled[pooled["cost"] == "gross"].set_index("strategy")

    text = {**MAESTRO_TEXT, **STRATEGY_TEXT}
    strategies = {}
    for key, (label, group, desc) in text.items():
        if f"{key}|spread" not in daily.columns:
            continue
        n_days = len(daily)
        strategies[key] = {
            "label": label, "group": group, "desc": desc,
            "hit": round(float(net.loc[key, "hit_directional"]), 4) if pd.notna(net.loc[key, "hit_directional"]) else None,
            "trades_per_day": round(float(net.loc[key, "n_trades"]) / n_days, 3),
            "gross_pips_per_trade": round(float(gross.loc[key, "net_pips_per_trade"]), 3)
            if pd.notna(gross.loc[key, "net_pips_per_trade"]) else None,
            "ranges": ranges(daily[f"{key}|spread"]),
        }
    design = json.loads((results / "design.json").read_text()) if (results / "design.json").exists() else {}
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or None
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_commit": commit,
        "design": design,
        "backtest": {"first_day": str(daily.index[0])[:10], "last_day": str(daily.index[-1])[:10],
                     "days": len(daily)},
        "start_balance": START,
        "quantiles": list(QUANTILES),
        "horizon_days": HORIZON,
        "strategies": strategies,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Write the backtest's expectations for the live trial")
    p.add_argument("--results", type=Path, default=DESIGN / "scores")
    p.add_argument("--risk", type=Path, default=DESIGN / "scores_risk")
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    data = build(args.results, args.risk)
    args.out.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {args.out}: {len(data['strategies'])} strategies, "
          f"{data['backtest']['days']} backtest days, ranges for 1-{HORIZON} days")


if __name__ == "__main__":
    main()
