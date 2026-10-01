"""
maestro/live/snapshot.py
========================
The live trial's public snapshot, built from the journal once a night.

The website's live page and the operator report both read this. It holds returns,
pips, counts and differences only, never OANDA's prices (which may not be
redistributed), so it can be published as it is.

Returns follow the evaluator exactly (backtesting/baselines.simulate): the position
decided at a bar earns the next bar's log move, and costs are booked on turnover,
half the round-trip cost per unit of position change. Days are UTC calendar days,
as in the backtest's daily series, so live days and backtest days compare directly.

    python -m maestro.live.snapshot --state /state --out snapshot.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from maestro.data.pipeline.store import market_closed

PIP = 1e-4
REF_COST_PIPS = 0.8
BAR = pd.Timedelta(minutes=5)
TRADE_DAYS = 2           # days of individual trades kept for the "what it did" list
ORDER_ROWS = 200         # most recent practice orders kept in full
START_GBP = 10_000       # the website's starting balance, for pound figures


def _connect(path: Path) -> sqlite3.Connection:
    # Read-only, so the publisher can never disturb the loop writing the same file.
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def load_ledger(db: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM ledger", db)
    df["bar"] = pd.to_datetime(df["bar"], utc=True)
    return df.sort_values(["strategy", "bar"])


def strategy_bars(rows: pd.DataFrame) -> pd.DataFrame:
    """One strategy's ledger rows as the evaluator's per-bar frame."""
    rows = rows.set_index("bar").sort_index()
    held = rows["position"].shift(1).fillna(0.0)          # exposure during bar t
    r_log = np.log(rows["mid"] / rows["mid"].shift(1)).fillna(0.0)
    cost_ref = rows["cost_ref_pips"].diff().fillna(rows["cost_ref_pips"])
    cost_quoted = rows["cost_quoted_pips"].diff().fillna(rows["cost_quoted_pips"])
    return pd.DataFrame({
        "held": held,
        "r_log": r_log,
        "pnl_pips": held * rows["mid"].diff().fillna(0.0) / PIP,
        "cost_ref_pips": cost_ref,
        "cost_quoted_pips": cost_quoted,
        "gross_log": held * r_log,
        "ref_log": held * r_log - cost_ref * PIP / rows["mid"],
        "quoted_log": held * r_log - cost_quoted * PIP / rows["mid"],
        "pip_log": PIP / rows["mid"],                      # one pip as a log return, for pound figures
        "position": rows["position"],
    })


def trades_of(bars: pd.DataFrame) -> pd.DataFrame:
    """Runs in one direction, as simulate() counts them, with entry and exit times."""
    direction = np.sign(bars["held"])
    run_id = (direction != direction.shift(1)).cumsum()
    in_mkt = bars["held"] != 0
    if not in_mkt.any():
        return pd.DataFrame(columns=["opened", "closed", "direction", "gross_pips", "cost_pips",
                                     "gross_gbp", "cost_gbp", "open"])
    held = bars[in_mkt].assign(run=run_id[in_mkt], t=bars.index[in_mkt])
    out = held.groupby("run").agg(first=("t", "first"), last=("t", "last"),
                                  direction=("held", lambda h: float(np.sign(h.iloc[0]))),
                                  gross_pips=("pnl_pips", "sum"), gross_log=("gross_log", "sum"),
                                  pip_log=("pip_log", "mean"), size=("held", lambda h: float(h.abs().mean())))
    last_bar = bars.index[-1]
    return pd.DataFrame({
        # Clock times. Bars are labelled by their open (OANDA's convention), and a position
        # taken at one bar's close is held from the next bar's open, so a trade opens at its
        # first held bar's label and closes when its last held bar closes.
        "opened": out["first"],
        "closed": out["last"] + BAR,
        "direction": out["direction"],
        "gross_pips": out["gross_pips"],
        "cost_pips": out["size"] * REF_COST_PIPS,
        "gross_gbp": START_GBP * (np.exp(out["gross_log"]) - 1.0),
        "cost_gbp": START_GBP * out["size"] * REF_COST_PIPS * out["pip_log"],
        "open": (out["last"] == last_bar) & (bars["position"].iloc[-1] != 0),
    }).reset_index(drop=True)


def _hit(bars: pd.DataFrame) -> float | None:
    both = (bars["held"] != 0) & (bars["r_log"] != 0)
    return float((np.sign(bars["held"]) == np.sign(bars["r_log"]))[both].mean()) if both.any() else None


def _round(v, digits: int = 4):
    return None if v is None or not np.isfinite(v) else round(float(v), digits)


def strategy_summary(bars: pd.DataFrame, days: pd.DatetimeIndex) -> dict:
    trades = trades_of(bars)
    daily = bars[["gross_log", "ref_log", "quoted_log"]].groupby(bars.index.normalize()).sum()
    daily = daily.reindex(days, fill_value=0.0)
    return {
        "daily": {k: [round(float(v), 7) for v in daily[f"{k}_log"]] for k in ("gross", "ref", "quoted")},
        "trades": int(len(trades)),
        "hit": _round(_hit(bars)),
        "bars_in_market": int((bars["held"] != 0).sum()),
        "gross_pips": _round(bars["pnl_pips"].sum(), 2),
        "cost_ref_pips": _round(bars["cost_ref_pips"].sum(), 2),
        "cost_quoted_pips": _round(bars["cost_quoted_pips"].sum(), 2),
        "position": float(bars["position"].iloc[-1]),
    }


def recent_trades(bars: pd.DataFrame) -> list[dict]:
    trades = trades_of(bars)
    if trades.empty:
        return []
    days = sorted(trades["closed"].dt.normalize().unique())[-TRADE_DAYS:]
    keep = trades[trades["closed"].dt.normalize().isin(days)]
    return [{"opened": t.opened.isoformat(), "closed": t.closed.isoformat(), "direction": int(t.direction),
             "gross_pips": round(t.gross_pips, 2), "cost_pips": round(t.cost_pips, 2),
             "gross_gbp": round(t.gross_gbp, 2), "cost_gbp": round(t.cost_gbp, 2), "open": bool(t.open)}
            for t in keep.itertuples()]


def orders_summary(db: sqlite3.Connection, ledger: pd.DataFrame, order_units: int) -> dict:
    """Practice fills against the paper ledger: how much worse (or better) reality was."""
    orders = pd.read_sql("SELECT * FROM orders", db)
    if orders.empty:
        return {"filled": 0, "failed": 0, "recent": []}
    orders["bar"] = pd.to_datetime(orders["bar"], utc=True)
    mids = ledger.drop_duplicates("bar").set_index("bar")["mid"]
    filled = orders[orders["status"] == "filled"].copy()
    side = np.sign(filled["units"])
    mid = filled["bar"].map(mids)
    # Paper fills at mid and books half the reference cost per unit traded; the practice
    # fill pays whatever the market gave. Positive = reality cost more than paper assumed.
    filled["vs_mid_pips"] = (filled["price"] - mid) * side / PIP
    filled["extra_pips"] = filled["vs_mid_pips"] - REF_COST_PIPS / 2
    # In pounds on the site's £10,000 at one unit of position = order_units practice units.
    filled["extra_gbp"] = START_GBP * filled["extra_pips"] * PIP / mid * filled["units"].abs() / order_units
    filled["spread_pips"] = (filled["ask"] - filled["bid"]) / PIP
    decided = filled["bar"] + BAR
    filled["delay_s"] = (pd.to_datetime(filled["filled_at"], utc=True, errors="coerce") - decided).dt.total_seconds()
    recent = orders.sort_values("bar").tail(ORDER_ROWS).merge(
        filled[["tag", "vs_mid_pips", "extra_pips", "spread_pips", "delay_s"]], on="tag", how="left")
    return {
        "filled": int(len(filled)),
        "failed": int((orders["status"] == "failed").sum()),
        "units_traded": int(filled["units"].abs().sum()),
        "mean_vs_mid_pips": _round(filled["vs_mid_pips"].mean(), 3),
        "mean_extra_pips": _round(filled["extra_pips"].mean(), 3),
        "total_extra_pips": _round(filled["extra_pips"].sum(), 2),
        "total_extra_gbp": _round(filled["extra_gbp"].sum(), 2),
        "mean_spread_pips": _round(filled["spread_pips"].mean(), 3),
        "median_delay_s": _round(filled["delay_s"].median(), 1),
        "recent": [{"bar": r.bar.isoformat(), "units": int(r.units), "status": r.status,
                    "vs_mid_pips": _round(r.vs_mid_pips, 2), "extra_pips": _round(r.extra_pips, 2),
                    "spread_pips": _round(r.spread_pips, 2), "delay_s": _round(r.delay_s, 1)}
                   for r in recent.itertuples()],
    }


def health(db: sqlite3.Connection, bars: pd.DatetimeIndex, start: pd.Timestamp,
           until: pd.Timestamp, now: pd.Timestamp) -> dict:
    """Bars the loop recorded against the bars the market was open for, before `until`.

    A bar only counts once the loop has had time to handle it (its close plus a minute
    by `now`), so a snapshot taken mid-cycle doesn't report the bar in hand as missed.
    """
    expected = pd.date_range(start, until, freq="5min", inclusive="left")
    expected = expected[expected + BAR + pd.Timedelta(minutes=1) <= now]
    # A bar is open if the market was open when it closed (the moment the loop acts on it).
    expected = expected[[not market_closed(t + BAR) for t in expected]]
    missed = expected.difference(bars)
    rows = db.execute("SELECT data FROM forecasts WHERE bar >= ?", (str(start),)).fetchall()
    unscorable = sum(1 for (d,) in rows if json.loads(d).get("unscorable"))
    return {"bars_expected": int(len(expected)), "bars_recorded": int(len(bars.intersection(expected))),
            "bars_missed": int(len(missed)), "bars_unscorable": int(unscorable),
            "last_missed": [t.isoformat() for t in missed[-20:]]}


def _code_commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def build(state: Path, phase: str, now: pd.Timestamp | None = None, cutoff: pd.Timestamp | None = None) -> dict:
    """The snapshot as of `now`, counting bars before `cutoff` only (default: all of them)."""
    now = now or pd.Timestamp.now(tz="UTC")
    db = _connect(state / "journal.db")
    ledger = load_ledger(db)
    if cutoff is not None:
        ledger = ledger[ledger["bar"] < cutoff]
    status = json.loads((state / "status.json").read_text()) if (state / "status.json").exists() else {}
    meta_path = state / "models" / "current" / "meta.json"
    model = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    out = {
        "generated_at": now.isoformat(timespec="seconds"),
        "phase": phase,
        "ref_cost_pips": REF_COST_PIPS,
        "start_gbp": START_GBP,
        "cutoff": None if cutoff is None else cutoff.isoformat(),
        "code_commit": _code_commit(),
        "model": {k: model.get(k) for k in ("deployed_at", "train_start", "train_end", "commit")},
        "order_strategy": status.get("order_strategy"),
        "heartbeat": {"at": status.get("at"), "error": status.get("error")},
    }
    if ledger.empty:
        return {**out, "start": None, "days": [], "strategies": {}, "trades": {}, "orders": {}, "health": {}}

    start = ledger["bar"].min()
    days = pd.DatetimeIndex(sorted(ledger["bar"].dt.normalize().unique()))
    strategies, trades = {}, {}
    for name, rows in ledger.groupby("strategy"):
        bars = strategy_bars(rows)
        strategies[name] = strategy_summary(bars, days)
        trades[name] = recent_trades(bars)
    recorded = pd.DatetimeIndex(ledger["bar"].unique())
    return {
        **out,
        "start": start.isoformat(),
        "last_bar": ledger["bar"].max().isoformat(),
        "days": [d.strftime("%Y-%m-%d") for d in days],
        "strategies": strategies,
        "trades": trades,
        "orders": orders_summary(db, ledger, int(status.get("order_units") or 10_000)),
        "health": health(db, recorded, start, cutoff or now, now),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Build the live trial's public snapshot")
    p.add_argument("--state", type=Path, default=Path("/state"))
    p.add_argument("--phase", default="shakedown", choices=["shakedown", "trial"])
    p.add_argument("--out", type=Path, default=None, help="default: print to stdout")
    args = p.parse_args()
    snap = build(args.state, args.phase)
    text = json.dumps(snap, separators=(",", ":"))
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out} ({len(text) / 1024:.0f} KB)")
    else:
        print(text)


if __name__ == "__main__":
    main()
