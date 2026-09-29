"""
maestro/demo/export_web_data.py
===============================
Exports the tested results into JSON for the MAESTRO website (web/).
The site never computes results of its own: it only re-scales these numbers
(e.g. the cost slider), so every figure on it traces back to
backtesting/baselines.py.

    python -m maestro.backtesting.baselines          # produce results first
    python -m maestro.demo.export_web_data           # writes web/src/data/*.json

Cost slider maths
-----------------
Costs are linear in the round-trip cost c. For each strategy we export daily
log returns before costs (gross) and at the model's spread (0.8 pips). The site
recovers any cost exactly as:
    net_c = gross - (c / 0.8) * (gross - net_0.8)
and monthly pips as gross_pips - trades * c.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from maestro.backtesting.baselines import OUTPUT_DIR, load_close
from maestro.demo.findings.build_page import STRATEGY_TEXT, msc_replication

ROOT = Path(__file__).resolve().parent.parent
WEB_DATA = ROOT / "web" / "src" / "data"
SCALE = 1e7          # daily log returns stored as integers (1e-7 precision)
RIDGE_DAYS = 150     # intraday paths used by the hero visual
RIDGE_POINTS = 72    # 20-minute resolution per day


def export_results(results: Path, instrument: str = "EUR_USD") -> dict:
    pooled = pd.read_csv(results / f"{instrument}_baselines_pooled.csv")
    per_split = pd.read_csv(results / f"{instrument}_baselines_per_split.csv")
    daily = pd.read_csv(results / f"{instrument}_baselines_daily.csv", index_col=0).fillna(0.0)
    daily.index = pd.to_datetime(daily.index, utc=True)

    gross_row = pooled[pooled["cost"] == "gross"].set_index("strategy")
    net_row = pooled[pooled["cost"] == "spread"].set_index("strategy")
    ref_cost = float(net_row["cost_pips"].iloc[0])

    months = (per_split[per_split["strategy"] == "buy_hold"].sort_values("split_id"))
    month_list = [{"start": pd.Timestamp(r.test_start).strftime("%Y-%m-%d"),
                   "end": pd.Timestamp(r.test_end).strftime("%Y-%m-%d"),
                   "label": pd.Timestamp(r.test_start).strftime("%b %Y")}
                  for r in months.itertuples()]

    strategies = []
    for key, (label, group, desc) in STRATEGY_TEXT.items():
        g, n = gross_row.loc[key], net_row.loc[key]
        sp = per_split[per_split["strategy"] == key].sort_values("split_id")
        strategies.append({
            "key": key, "label": label, "group": group, "desc": desc,
            "hit": round(float(n["hit_directional"]), 4),
            "hitMsc": round(float(n["hit_msc_style"]), 4),
            "trades": int(n["n_trades"]),
            "exposure": round(float(n["exposure"]), 4),
            "grossPerTrade": round(float(g["net_pips_per_trade"]), 4),
            "daily": {
                "gross": [int(round(v * SCALE)) for v in daily[f"{key}|gross"]],
                "ref":   [int(round(v * SCALE)) for v in daily[f"{key}|spread"]],
            },
            "monthly": {
                "grossPips": [round(float(v), 1) for v in sp["net_pips_total"] + sp["n_trades"] * ref_cost],
                "trades": [int(v) for v in sp["n_trades"]],
            },
        })

    return {
        "instrument": instrument,
        "refCostPips": ref_cost,
        "scale": SCALE,
        "dates": [d.strftime("%Y-%m-%d") for d in daily.index],
        "months": month_list,
        "strategies": strategies,
    }


def export_ridge(close: pd.Series, start: str, end: str) -> dict:
    """Real intraday EUR/USD paths (pips from each day's first price) for the hero."""
    c = close.loc[start:end]
    days = []
    for day, s in c.groupby(c.index.normalize()):
        if len(s) < 270:                      # skip partial / holiday sessions
            continue
        pips = (s.values - s.values[0]) / 1e-4
        idx = np.linspace(0, len(pips) - 1, RIDGE_POINTS).round().astype(int)
        days.append((day.strftime("%Y-%m-%d"), [int(round(v)) for v in pips[idx]]))
    pick = np.linspace(0, len(days) - 1, min(RIDGE_DAYS, len(days))).round().astype(int)
    chosen = [days[i] for i in sorted(set(pick))]
    return {"days": [d for d, _ in chosen], "paths": [p for _, p in chosen]}


def main() -> None:
    results = OUTPUT_DIR / "baselines"
    close = load_close("EUR_USD")
    data = export_results(results)
    ridge = export_ridge(close, data["dates"][0], data["dates"][-1])
    msc = msc_replication(close)

    WEB_DATA.mkdir(parents=True, exist_ok=True)
    for name, obj in (("results", data), ("ridge", ridge), ("msc", msc)):
        path = WEB_DATA / f"{name}.json"
        path.write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")
        print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")
    print(f"  {len(data['strategies'])} strategies, {len(data['dates'])} days, "
          f"{len(data['months'])} months, {len(ridge['paths'])} ridge days")


if __name__ == "__main__":
    main()
