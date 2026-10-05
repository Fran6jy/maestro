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
from maestro.live import publish as publisher
from maestro.live.check import problems
from maestro.live.ledger import Journal
from maestro.live.publish import current_edition, next_run
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


def test_foundation_schedule():
    dawn = pd.Timestamp("2026-10-05 00:20", tz="UTC")
    assert next_run(pd.Timestamp("2026-10-05 00:19", tz="UTC")) == (dawn, "dawn")
    assert next_run(dawn) == (pd.Timestamp("2026-10-05 12:20", tz="UTC"), "day")
    assert next_run(pd.Timestamp("2026-10-05 21:00", tz="UTC")) == (
        pd.Timestamp("2026-10-06 00:20", tz="UTC"), "dawn")
    assert current_edition(pd.Timestamp("2026-10-05 08:00", tz="UTC")) == "dawn"
    assert current_edition(pd.Timestamp("2026-10-05 15:00", tz="UTC")) == "day"
    assert current_edition(pd.Timestamp("2026-10-05 23:00", tz="UTC")) == "dusk"


def test_foundation_metadata_is_explicit(state):
    path, close, _ = state
    snap = build(path, "trial", now=close.index[-1], edition="day", official=False)
    assert snap["edition"] == "day"
    assert snap["official"] is False


def test_foundation_publication_keeps_official_and_provisional_separate(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(publisher, "checkout", lambda *args: None)
    monkeypatch.setattr(publisher, "build", lambda state, phase, now, cutoff, edition, official: {
        "generated_at": now.isoformat(), "cutoff": cutoff.isoformat() if cutoff is not None else None,
        "edition": edition, "official": official,
    })

    def fake_git(work, *args, key):
        calls.append(args)
        return "changed" if args[0] == "status" else ""

    monkeypatch.setattr(publisher, "git", fake_git)
    common = (tmp_path, "trial", "repo", tmp_path / "public", tmp_path / "key")
    common[3].mkdir()
    dawn = pd.Timestamp("2026-10-06 00:20", tz="UTC")
    publisher.publish(*common, edition="dawn", now=dawn)
    public = tmp_path / "public"
    assert (public / "official.json").exists()
    assert (public / "history" / "2026-10-05.json").exists()
    assert (public / "history" / "foundation" / "2026-10-05-dawn.json").exists()

    day = pd.Timestamp("2026-10-06 12:20", tz="UTC")
    publisher.publish(*common, edition="day", now=day)
    assert json.loads((public / "snapshot.json").read_text())["edition"] == "day"
    assert json.loads((public / "official.json").read_text())["edition"] == "dawn"
    assert (public / "history" / "foundation" / "2026-10-06-day.json").exists()
    assert publisher.publish(*common, edition="day", now=day).endswith("already published")
    assert any(args[0] == "push" for args in calls)


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
