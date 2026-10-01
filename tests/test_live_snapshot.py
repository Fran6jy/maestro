"""
The live trial's public snapshot (maestro/live/snapshot.py) must report the paper
ledger exactly as the shared evaluator would score it, day by day, so a live day
and a backtest day are the same thing. And the nightly check (live/check.py) must
fail when the trial is unwell.
"""
import json

import numpy as np
import pandas as pd
import pytest

from maestro.backtesting.baselines import _net_log, simulate, summarise
from maestro.live.check import problems
from maestro.live.ledger import Journal
from maestro.live.snapshot import build

PIP = 1e-4


@pytest.fixture
def state(tmp_path):
    rng = np.random.default_rng(5)
    n = 700                                                   # spans three UTC days
    idx = pd.date_range("2024-03-05 20:00", periods=n, freq="5min", tz="UTC")
    close = pd.Series(1.08 + np.cumsum(rng.normal(0, 1e-4, n)), index=idx)
    target = pd.Series(rng.choice([-1.0, 0.0, 1.0], n, p=[0.1, 0.8, 0.1]), index=idx)
    target.iloc[-1] = 0.0                                     # end flat, as a test window does
    j = Journal(tmp_path / "journal.db")
    for t in idx:
        c = float(close[t])
        j.record_forecast(t, {"confidence": 0.5})
        j.step(t, "s", float(target[t]), mid=c, bid=c - 0.4 * PIP, ask=c + 0.4 * PIP)
    j.commit()
    (tmp_path / "status.json").write_text(json.dumps({"at": str(idx[-1] + pd.Timedelta(minutes=5)),
                                                      "error": None, "order_strategy": None}))
    return tmp_path, close, target


def test_snapshot_matches_the_evaluator(state):
    path, close, target = state
    snap = build(path, "trial", now=close.index[-1] + pd.Timedelta(minutes=6))
    s = snap["strategies"]["s"]
    res = simulate(target, close, PIP)          # target ends flat, so nothing differs at the end
    for kind, cost in (("gross", 0.0), ("ref", 0.8)):
        daily = _net_log(res.bars, cost, PIP).groupby(res.bars.index.normalize()).sum()
        assert np.allclose(s["daily"][kind], daily.to_numpy(), atol=1e-7)
    assert snap["days"] == [d.strftime("%Y-%m-%d") for d in daily.index]
    pooled = summarise([res], 0.8, PIP)
    assert s["gross_pips"] - s["cost_ref_pips"] == pytest.approx(pooled["net_pips_total"], abs=0.01)
    assert s["hit"] == pytest.approx(pooled["hit_directional"], abs=1e-4)
    assert s["trades"] == len(res.trades)
    assert snap["health"]["bars_missed"] == 0


def test_cutoff_keeps_whole_days_only(state):
    path, close, _ = state
    cut = close.index[-1].normalize()
    snap = build(path, "trial", now=close.index[-1] + pd.Timedelta(minutes=6), cutoff=cut)
    assert pd.Timestamp(snap["last_bar"]) < cut
    assert snap["days"][-1] < cut.strftime("%Y-%m-%d")


def test_snapshot_never_carries_prices(state):
    path, close, _ = state
    text = json.dumps(build(path, "trial", now=close.index[-1]))
    assert '"mid"' not in text and '"price"' not in text and '"bid"' not in text


def test_check_flags_a_stopped_loop_and_a_stale_snapshot(state):
    path, close, _ = state
    now = close.index[-1] + pd.Timedelta(minutes=6)
    snap = build(path, "trial", now=now)
    assert problems(snap, now) == []
    assert any("stopped" in p for p in problems(snap, now + pd.Timedelta(hours=30)))
    snap["heartbeat"]["at"] = str(now - pd.Timedelta(hours=2))
    assert any("loop has stopped" in p for p in problems(snap, now))
    snap["heartbeat"] = {"at": str(now), "error": "ValueError: boom"}
    assert any("boom" in p for p in problems(snap, now))
