"""
maestro/live/ledger.py
======================
The live trial's journal and paper ledger, in one SQLite file.

Every bar the loop records MAESTRO's forecast, each strategy's decision, each
strategy's paper position and profit, and any practice order. Paper profit
follows the evaluator's timing (a position decided at a bar's close earns the
next bar's move, marked at mid prices) and books costs twice: at the reference
cost the backtests use, and at the spread OANDA actually quoted when the
position changed. The gap between the two is part of what the trial measures.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS forecasts (bar TEXT PRIMARY KEY, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS decisions (bar TEXT, strategy TEXT, target REAL, PRIMARY KEY (bar, strategy));
CREATE TABLE IF NOT EXISTS ledger (
    bar TEXT, strategy TEXT, position REAL, mid REAL, gross_pips REAL,
    cost_ref_pips REAL, cost_quoted_pips REAL, trades INTEGER, PRIMARY KEY (bar, strategy));
CREATE TABLE IF NOT EXISTS orders (
    tag TEXT PRIMARY KEY, bar TEXT, strategy TEXT, units INTEGER, status TEXT,
    price REAL, bid REAL, ask REAL, filled_at TEXT, error TEXT);
"""


class Journal:
    def __init__(self, path: Path, pip: float = 1e-4, ref_cost_pips: float = 0.8) -> None:
        self.db = sqlite3.connect(str(path))
        self.db.executescript(SCHEMA)
        self.pip, self.ref_cost = pip, ref_cost_pips

    # ── Recording ────────────────────────────────────────────────────────────
    def seen(self, bar: pd.Timestamp) -> bool:
        return self.db.execute("SELECT 1 FROM forecasts WHERE bar = ?", (str(bar),)).fetchone() is not None

    def record_forecast(self, bar: pd.Timestamp, forecast: dict) -> None:
        self.db.execute("INSERT OR REPLACE INTO forecasts VALUES (?, ?)",
                        (str(bar), json.dumps(forecast, default=float)))

    def confidence_history(self, n: int) -> pd.Series:
        rows = self.db.execute("SELECT bar, data FROM forecasts ORDER BY bar DESC LIMIT ?", (n,)).fetchall()
        s = pd.Series({pd.Timestamp(b): json.loads(d).get("confidence") for b, d in rows}, dtype=float)
        return s.sort_index().dropna()

    def last(self, strategy: str) -> dict | None:
        row = self.db.execute(
            "SELECT position, mid, gross_pips, cost_ref_pips, cost_quoted_pips, trades FROM ledger "
            "WHERE strategy = ? ORDER BY bar DESC LIMIT 1", (strategy,)).fetchone()
        keys = ("position", "mid", "gross_pips", "cost_ref_pips", "cost_quoted_pips", "trades")
        return dict(zip(keys, row)) if row else None

    def step(self, bar: pd.Timestamp, strategy: str, target: float,
             mid: float, bid: float, ask: float) -> dict:
        """Mark the position held over the bar that just closed, then move to `target`."""
        prev = self.last(strategy) or {"position": 0.0, "mid": mid, "gross_pips": 0.0,
                                       "cost_ref_pips": 0.0, "cost_quoted_pips": 0.0, "trades": 0}
        gross = prev["gross_pips"] + prev["position"] * (mid - prev["mid"]) / self.pip
        turnover = abs(target - prev["position"])
        spread_pips = (ask - bid) / self.pip
        new_trade = target != 0 and (prev["position"] == 0 or (target > 0) != (prev["position"] > 0))
        state = {"position": target, "mid": mid, "gross_pips": gross,
                 "cost_ref_pips": prev["cost_ref_pips"] + turnover * self.ref_cost / 2,
                 "cost_quoted_pips": prev["cost_quoted_pips"] + turnover * spread_pips / 2,
                 "trades": prev["trades"] + int(new_trade)}
        self.db.execute("INSERT OR REPLACE INTO decisions VALUES (?, ?, ?)", (str(bar), strategy, target))
        self.db.execute("INSERT OR REPLACE INTO ledger VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (str(bar), strategy, *[state[k] for k in
                         ("position", "mid", "gross_pips", "cost_ref_pips", "cost_quoted_pips", "trades")]))
        return state

    def record_order(self, tag: str, bar: pd.Timestamp, strategy: str, units: int, status: str,
                     fill=None, error: str | None = None) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (tag, str(bar), strategy, units, status,
             getattr(fill, "price", None), getattr(fill, "bid", None), getattr(fill, "ask", None),
             str(getattr(fill, "time", "")) or None, error))

    def commit(self) -> None:
        self.db.commit()

    # ── Reading ──────────────────────────────────────────────────────────────
    def summary(self) -> pd.DataFrame:
        rows = self.db.execute(
            "SELECT l.strategy, l.position, l.gross_pips, l.cost_ref_pips, l.cost_quoted_pips, l.trades "
            "FROM ledger l JOIN (SELECT strategy, MAX(bar) AS bar FROM ledger GROUP BY strategy) m "
            "ON l.strategy = m.strategy AND l.bar = m.bar").fetchall()
        df = pd.DataFrame(rows, columns=["strategy", "position", "gross_pips", "cost_ref_pips",
                                         "cost_quoted_pips", "trades"]).set_index("strategy")
        df["net_ref_pips"] = df["gross_pips"] - df["cost_ref_pips"]
        df["net_quoted_pips"] = df["gross_pips"] - df["cost_quoted_pips"]
        return df
