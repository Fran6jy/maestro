"""
maestro/demo/findings/build_page.py
===================================
Builds "The Cost of Being Right", a plain-language findings page for readers
without a technical background, from the outputs of backtesting/baselines.py.

Every figure on the page comes from those outputs or is recomputed here from
the price data (the MSc replication), so the page can always be regenerated
after the pipeline changes.

    python -m maestro.backtesting.baselines                 # produce results first
    python -m maestro.demo.findings.build_page --out findings.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from maestro.backtesting.baselines import OUTPUT_DIR, load_close

HERE = Path(__file__).resolve().parent
PLACEHOLDER = "/*__DATA__*/null"

# (label, group, one-line description for a non-technical reader)
STRATEGY_TEXT = {
    "logreg_lag5":    ("Logistic regression", "msc",
                       "Looks at the last five 5-minute moves and predicts whether the next one is up or down."),
    "bollinger_20_2": ("Bollinger bands", "msc",
                       "Buys when the price is unusually low for the last 100 minutes and sells when it is unusually high."),
    "contrarian_3":   ("Contrarian", "msc",
                       "Bets that the last 15 minutes of movement will reverse."),
    "linreg_lag5":    ("Linear regression, 5 moves", "msc",
                       "Fits a straight line through the last five moves to predict the next one."),
    "linreg_lag1":    ("Linear regression, 1 move", "msc",
                       "Predicts the next move from the last one alone. This is the model behind the MSc's 37.6% figure."),
    "sma_20_200":     ("Moving-average crossover", "msc",
                       "Follows the trend when the 20-bar average crosses the 200-bar average."),
    "random":         ("Coin flip", "ref",
                       "Picks up or down at random every 5 minutes. It should lose roughly what its trades cost, and it does."),
    "buy_hold":       ("Buy and hold", "ref",
                       "Buys EUR/USD at the start of each test month and holds it to the end."),
}

MSC_REPORTED = {"hit": 37.56, "hit_5lag": 36.875, "sharpe": 0.599}
MSC_WINDOW = ("2023-06-29", "2023-07-31")
SAMPLE_BARS = 96   # eight hours of 5-minute bars


def msc_replication(close: pd.Series) -> dict:
    """Re-run the MSc's one-lag linear regression on its own window, scored its way."""
    w = close.loc[MSC_WINDOW[0]:MSC_WINDOW[1]]
    rerun, sample = {}, {}
    for key, series in (("5dp", w), ("4dp", w.round(4))):
        r = np.log(series / series.shift(1))
        d = pd.DataFrame({"r": r, "lag": r.shift(1)}).dropna()
        coef = np.linalg.lstsq(d[["lag"]].values, d["r"].values, rcond=None)[0]
        pred = np.sign(d[["lag"]].values @ coef).ravel()
        hits = np.sign(d["r"].values * pred) == 1
        rerun[key] = {"hit": float(hits.mean()), "flat_share": float((d["r"] == 0).mean())}
        sample[key] = [int(v) for v in np.sign(r.dropna().iloc[:SAMPLE_BARS].values)]
    return {"reported_hit": MSC_REPORTED["hit"], "reported_hit_5lag": MSC_REPORTED["hit_5lag"],
            "reported_sharpe": MSC_REPORTED["sharpe"], "rerun": rerun, "sample": sample}


def build(results: Path, instrument: str = "EUR_USD") -> dict:
    pooled = pd.read_csv(results / f"{instrument}_baselines_pooled.csv")
    per_split = pd.read_csv(results / f"{instrument}_baselines_per_split.csv")
    daily = pd.read_csv(results / f"{instrument}_baselines_daily.csv", index_col=0).fillna(0.0)
    daily.index = pd.to_datetime(daily.index, utc=True)

    gross = pooled[pooled["cost"] == "gross"].set_index("strategy")
    net = pooled[pooled["cost"] == "spread"].set_index("strategy")
    cost_pips = float(net["cost_pips"].iloc[0])

    months_df = (per_split[per_split["strategy"] == "buy_hold"]
                 .sort_values("split_id")[["split_id", "test_start", "test_end"]])
    months = [{"start": pd.Timestamp(r.test_start).strftime("%Y-%m-%d"),
               "end": pd.Timestamp(r.test_end).strftime("%Y-%m-%d"),
               "label": pd.Timestamp(r.test_start).strftime("%b %Y")}
              for r in months_df.itertuples()]

    strategies, equity, month_pnl = [], {}, {}
    for key, (label, group, desc) in STRATEGY_TEXT.items():
        g, n = gross.loc[key], net.loc[key]
        split_net = per_split[per_split["strategy"] == key].sort_values("split_id")["net_pips_total"]
        strategies.append({
            "key": key, "label": label, "group": group, "desc": desc,
            "hit": round(float(n["hit_directional"]), 4),
            "hit_msc": round(float(n["hit_msc_style"]), 4),
            "trades": int(n["n_trades"]),
            "gross_per_trade": round(float(g["net_pips_per_trade"]), 3),
            "net_per_trade": round(float(n["net_pips_per_trade"]), 3),
            "net_pips_total": round(float(n["net_pips_total"]), 1),
            "sharpe_gross": round(float(g["sharpe"]), 2),
            "sharpe_net": round(float(n["sharpe"]), 2),
            "months_profitable": int((split_net > 0).sum()),
            "months_total": int(len(split_net)),
        })
        equity[key] = {
            "before": [round(float(v), 2) for v in 10000 * np.exp(daily[f"{key}|gross"].cumsum())],
            "after":  [round(float(v), 2) for v in 10000 * np.exp(daily[f"{key}|spread"].cumsum())],
        }
        month_pnl[key] = [round(float(v), 1) for v in split_net]

    close = load_close(instrument)
    return {
        "period": {"data_start": close.index[0].strftime("%Y-%m-%d"),
                   "start": daily.index[0].strftime("%Y-%m-%d"),
                   "end": daily.index[-1].strftime("%Y-%m-%d")},
        "cost_pips": cost_pips,
        "strategies": strategies,
        "dates": [d.strftime("%Y-%m-%d") for d in daily.index],
        "equity": equity,
        "months": months,
        "month_pnl": month_pnl,
        "msc": msc_replication(close),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Build the findings page")
    p.add_argument("--results", default=str(OUTPUT_DIR / "baselines"))
    p.add_argument("--out", default=str(HERE / "findings.html"))
    args = p.parse_args()

    data = build(Path(args.results))
    template = (HERE / "template.html").read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise SystemExit("Data placeholder missing from template.html")
    html = template.replace(PLACEHOLDER, json.dumps(data, separators=(",", ":")))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")

    lr = next(s for s in data["strategies"] if s["key"] == "logreg_lag5")
    rr = data["msc"]["rerun"]
    print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    print(f"  logreg: hit {lr['hit']:.3f}, gross/trade {lr['gross_per_trade']:+.3f} pips, "
          f"£10k -> {data['equity']['logreg_lag5']['before'][-1]:,.0f} / {data['equity']['logreg_lag5']['after'][-1]:,.2f}")
    print(f"  MSc re-run: 5dp {rr['5dp']['hit']:.3f} ({rr['5dp']['flat_share']:.3f} flat), "
          f"4dp {rr['4dp']['hit']:.3f} ({rr['4dp']['flat_share']:.3f} flat)")


if __name__ == "__main__":
    main()
