"""
Tests for the raw store layout (maestro/data/pipeline/store.py): finished months
in one file, the month in progress by day, folded when it ends, nothing lost.
"""
import numpy as np
import pandas as pd
import pytest

import maestro.data.pipeline.store as st


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "RAW", tmp_path)
    return tmp_path


def _candles(start, end):
    idx = pd.date_range(start, end, freq="5min", tz="UTC", inclusive="left").rename("time")
    c = 1.1 + np.cumsum(np.random.default_rng(0).normal(0, 1e-4, len(idx)))
    return pd.DataFrame({"open": c, "high": c + 1e-4, "low": c - 1e-4, "close": c,
                         "bid_close": c - 5e-5, "ask_close": c + 5e-5, "volume": 1}, index=idx)


def test_finished_months_get_one_file_and_the_current_month_one_per_day(store):
    now = pd.Timestamp("2026-09-03 12:00", tz="UTC")
    st.write_candles("EUR_USD", _candles("2026-08-30", "2026-09-03"), now)
    names = sorted(f.name for f in st._files("EUR_USD"))
    assert names == ["2026-08.parquet", "2026-09-01.parquet", "2026-09-02.parquet"]


def test_month_end_folds_day_files_without_losing_candles(store):
    df = _candles("2026-09-28", "2026-10-02")
    st.write_candles("EUR_USD", df, pd.Timestamp("2026-09-30 23:00", tz="UTC"))
    st.write_candles("EUR_USD", df.loc["2026-10-01":], pd.Timestamp("2026-10-01 22:30", tz="UTC"))
    st.compact("EUR_USD", pd.Timestamp("2026-10-01 22:30", tz="UTC"))
    names = sorted(f.name for f in st._files("EUR_USD"))
    assert names == ["2026-09.parquet", "2026-10-01.parquet"]
    assert st.load_candles("EUR_USD").equals(df)
    assert st.last_candle("EUR_USD") == df.index[-1]


def test_rewriting_the_same_candles_changes_nothing(store):
    now = pd.Timestamp("2026-09-10", tz="UTC")
    df = _candles("2026-08-20", "2026-09-09")
    st.write_candles("EUR_USD", df, now)
    st.write_candles("EUR_USD", df.iloc[-500:], now)
    assert st.load_candles("EUR_USD").equals(df)
