"""
Tests that data only reaches a bar once it was public, and that the sealed
holdout stays sealed. The old macro pipeline let every 5-minute bar see that
day's closing VIX and CPI six weeks before release; these pin the fix down.
"""
import numpy as np
import pandas as pd
import pytest

from maestro.data.features.macro import available_at, macro_frame
from maestro.data.holdout import HOLDOUT_START, seal, unlock


def _ts(s):
    return pd.Timestamp(s, tz="UTC")


def test_daily_series_are_used_from_the_next_afternoon():
    t = available_at("VIXCLS", pd.Series([_ts("2024-08-05")]))
    assert t.iloc[0] == _ts("2024-08-06 14:00")


@pytest.mark.parametrize("day", ["2024-08-05", "2024-08-07", "2024-08-09"])  # Mon, Wed, Fri
def test_h10_rates_wait_for_the_following_monday(day):
    t = available_at("DEXUSEU", pd.Series([_ts(day)]))
    assert t.iloc[0] == _ts("2024-08-12 21:00")


def test_monthly_series_use_the_first_release_date():
    t = available_at("CPIAUCSL", pd.Series([_ts("2024-07-01")]), pd.Series(["2024-08-14"]))
    assert t.iloc[0] == _ts("2024-08-14 14:00")
    fallback = available_at("CPIAUCSL", pd.Series([_ts("2024-07-01")]), pd.Series([None]))
    assert fallback.iloc[0] > _ts("2024-08-01")                 # never before the month has ended


def _raw_vix(days):
    dates = pd.Series(pd.to_datetime(days, utc=True))
    return pd.DataFrame({"series": "VIXCLS", "date": dates,
                         "value": np.arange(len(days), dtype=float) + 10,
                         "available_at": available_at("VIXCLS", dates)})


def test_no_bar_sees_a_value_before_it_was_public():
    days = pd.bdate_range("2024-01-01", "2024-03-29", tz="UTC")
    raw = _raw_vix(days)
    bars = pd.date_range("2024-01-02", "2024-03-29", freq="5min", tz="UTC")
    seen = macro_frame(raw, bars)["VIXCLS"]
    public = raw.set_index("value")["available_at"]
    ok = seen.dropna().map(public) <= seen.dropna().index.to_series()
    assert ok.all()
    # The first bar of a day must not already know that day's close.
    same_day = raw.set_index("date").loc[_ts("2024-02-05"), "value"]
    assert seen[_ts("2024-02-05 00:00")] != same_day
    assert seen[_ts("2024-02-06 14:00")] == same_day


def test_holdout_is_sealed_unless_unlocked(monkeypatch):
    monkeypatch.delenv("MAESTRO_HOLDOUT", raising=False)
    s = pd.Series(1.0, index=pd.date_range(HOLDOUT_START - pd.Timedelta(days=2), periods=5, freq="D", tz="UTC"))
    assert seal(s).index.max() < HOLDOUT_START
    unlock()
    assert seal(s).index.max() >= HOLDOUT_START
    monkeypatch.delenv("MAESTRO_HOLDOUT")
