"""
maestro/demo/export_web_data.py
===============================
Exports the tested results into JSON for the MAESTRO website (web/).
The site never computes results of its own: it only re-scales these numbers
(e.g. the cost slider), so every figure on it traces back to
backtesting/baselines.py.

    python -m maestro.backtesting.maestro_runner     # produce results first (baselines + MAESTRO)
    python -m maestro.demo.export_web_data           # writes web/src/data/*.json

By default it reads the MAESTRO run's scores folder, which holds every
baseline and both MAESTRO variants scored together; if that run has not
finished it falls back to the baselines alone. `--results DIR` overrides both.

Cost slider maths
-----------------
Costs are linear in the round-trip cost c. For each strategy we export weekly
log returns before costs (gross) and at the model's spread (0.8 pips). The site
recovers any cost exactly as:
    net_c = gross - (c / 0.8) * (gross - net_0.8)
and monthly pips as gross_pips - trades * c. Weekly sums keep balances exact at
every week's end while shipping a fifth of the daily series. The Sharpe ratio
needs daily returns, so the sums and cross-products of the daily gross return g
and cost return d = gross - net_0.8 are exported instead; with r = g - k*d,
    mean = (Sg - k Sd) / n,  var = (Sgg - 2k Sgd + k^2 Sdd - n mean^2) / (n - 1),
which gives the daily Sharpe ratio at any cost exactly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from maestro.backtesting.baselines import OUTPUT_DIR, design_tag, load_close

ROOT = Path(__file__).resolve().parent.parent
WEB_DATA = ROOT / "web" / "src" / "data"
SCALE = 1e7          # daily log returns stored as integers (1e-7 precision)
RIDGE_DAYS = 150     # intraday paths used by the hero visual
RIDGE_POINTS = 72    # 20-minute resolution per day

# Plain-language labels for the site: (label, group, description).
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


DESIGN = design_tag(refit_months=3, train_months=12)   # the design the site reports

MAESTRO_TEXT = {
    "maestro_top10":   ("MAESTRO, most confident 10%", "maestro",
                        "The multi-agent system: it reads the market regime, forecasts the next 30 minutes "
                        "with two deep-learning models, and trades only its 10% most confident calls."),
    "maestro_ungated": ("MAESTRO, every signal", "maestro",
                        "The same forecasts with no confidence filter, so it acts on every up or down call."),
    "risk_cost_filter": ("MAESTRO with a cost check", "maestro",
                         "MAESTRO's forecasts, but it trades only when the move it expects is worth "
                         "more than the cost of trading plus half a pip."),
    "maestro_gated":   ("MAESTRO as designed", "maestro",
                        "Trades only when confidence clears the thresholds it was designed with. Over 20 "
                        "years that happened once."),
}


def _num(v: float, digits: int = 4) -> float | None:
    """JSON has no NaN: a strategy that never traded has no hit rate."""
    return None if pd.isna(v) else round(float(v), digits)


def _read(folder: Path, instrument: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pooled = pd.read_csv(folder / f"{instrument}_baselines_pooled.csv")
    per_split = pd.read_csv(folder / f"{instrument}_baselines_per_split.csv")
    daily = pd.read_csv(folder / f"{instrument}_baselines_daily.csv", index_col=0).fillna(0.0)
    daily.index = pd.to_datetime(daily.index, utc=True)
    return pooled, per_split, daily


def _moments(g: pd.Series, d: pd.Series) -> dict:
    """Sums the site needs to compute the daily Sharpe ratio exactly at any cost."""
    g, d = g.to_numpy(float), d.to_numpy(float)
    f = lambda v: float(f"{v:.10g}")  # noqa: E731
    return {"n": int(len(g)), "g": f(g.sum()), "d": f(d.sum()), "gg": f((g * g).sum()),
            "dd": f((d * d).sum()), "gd": f((g * d).sum())}


def export_results(results: Path, instrument: str = "EUR_USD") -> dict:
    pooled, per_split, daily = _read(results, instrument)
    risk = results.parent / "scores_risk"
    if (risk / f"{instrument}_baselines_pooled.csv").exists():
        # The risk layer's variants (backtesting/risk_layer.py) were scored on the same bars.
        rp, rs, rd = _read(risk, instrument)
        keep = lambda df: df[df["strategy"].str.startswith("risk_")]  # noqa: E731
        pooled = pd.concat([pooled, keep(rp)])
        per_split = pd.concat([per_split, keep(rs)])
        daily = daily.join(rd[[c for c in rd.columns if c.startswith("risk_")]], how="left").fillna(0.0)

    gross_row = pooled[pooled["cost"] == "gross"].set_index("strategy")
    net_row = pooled[pooled["cost"] == "spread"].set_index("strategy")
    ref_cost = float(net_row["cost_pips"].iloc[0])

    months = (per_split[per_split["strategy"] == "buy_hold"].sort_values("split_id"))
    month_list = [{"start": pd.Timestamp(r.test_start).strftime("%Y-%m-%d"),
                   "end": pd.Timestamp(r.test_end).strftime("%Y-%m-%d"),
                   "label": pd.Timestamp(r.test_start).strftime("%b %Y")}
                  for r in months.itertuples()]

    # Weeks end on Friday; Sunday's first bars belong to the week they open.
    week = daily.index.tz_convert(None).to_period("W-FRI")
    weekly = daily.groupby(week).sum()
    week_ends = daily.index.to_series().groupby(week).max()

    strategies = []
    for key, (label, group, desc) in {**MAESTRO_TEXT, **STRATEGY_TEXT}.items():
        if key not in net_row.index:
            continue
        g, n = gross_row.loc[key], net_row.loc[key]
        sp = per_split[per_split["strategy"] == key].sort_values("split_id")
        strategies.append({
            "key": key, "label": label, "group": group, "desc": desc,
            "hit": _num(n["hit_directional"]),
            "hitMsc": _num(n["hit_msc_style"]),
            "trades": int(n["n_trades"]),
            "exposure": _num(n["exposure"]) or 0.0,
            "grossPerTrade": _num(g["net_pips_per_trade"]) or 0.0,
            "weekly": {
                "gross": [int(round(v * SCALE)) for v in weekly[f"{key}|gross"]],
                "ref":   [int(round(v * SCALE)) for v in weekly[f"{key}|spread"]],
            },
            "moments": _moments(daily[f"{key}|gross"], daily[f"{key}|gross"] - daily[f"{key}|spread"]),
            "monthly": {
                "grossPips": [round(float(v), 1) for v in sp["net_pips_total"] + sp["n_trades"] * ref_cost],
                "trades": [int(v) for v in sp["n_trades"]],
            },
        })

    return {
        "instrument": instrument,
        "design": results.name if results.name != "scores" else results.parent.name,
        "refCostPips": ref_cost,
        "scale": SCALE,
        "dates": [d.strftime("%Y-%m-%d") for d in week_ends],
        "tradingDays": len(daily),
        "firstDay": daily.index[0].strftime("%Y-%m-%d"),
        "lastDay": daily.index[-1].strftime("%Y-%m-%d"),
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


def default_results() -> Path:
    """MAESTRO's scored design (it includes every baseline), else the baselines alone."""
    for tag in (DESIGN, f"{DESIGN}_fast"):
        maestro = OUTPUT_DIR / "maestro" / tag / "scores"
        if (maestro / "EUR_USD_baselines_pooled.csv").exists():
            return maestro
    return OUTPUT_DIR / "baselines" / DESIGN


def main() -> None:
    p = argparse.ArgumentParser(description="Export tested results for the website")
    p.add_argument("--results", type=Path, default=None, help="folder written by baselines.run")
    results = p.parse_args().results or default_results()
    print(f"reading {results}")
    close = load_close("EUR_USD")
    data = export_results(results)
    ridge = export_ridge(close, data["dates"][0], data["dates"][-1])
    msc = msc_replication(close)

    WEB_DATA.mkdir(parents=True, exist_ok=True)
    for name, obj in (("results", data), ("ridge", ridge), ("msc", msc)):
        path = WEB_DATA / f"{name}.json"
        path.write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False), encoding="utf-8")
        print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")
    print(f"  {len(data['strategies'])} strategies, {data['tradingDays']} days in {len(data['dates'])} weeks, "
          f"{len(data['months'])} months, {len(ridge['paths'])} ridge days")


if __name__ == "__main__":
    main()
