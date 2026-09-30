"""
Tests for how maestro_runner splits and schedules its retraining blocks. Training
itself is replaced by a stub, so these run in seconds.
"""
import time

import numpy as np
import pandas as pd

import maestro.backtesting.maestro_runner as runner
from maestro.backtesting.baselines import refit_plan


def _close():
    rng = np.random.default_rng(7)
    idx = pd.date_range("2022-01-03", periods=3 * 365 * 24, freq="h", tz="UTC")
    return pd.Series(1.1 + np.cumsum(rng.normal(0, 1e-4, len(idx))), index=idx)


def _stub(calls):
    def train_and_predict(df, train_idx, test_idx, instrument, epochs, fast=False, granularity="M5"):
        calls.append(test_idx[0])
        sig = pd.DataFrame({"signal": 1.0, "confidence": 0.9, "regime": 2}, index=test_idx)
        return sig, {"fit_bars": len(train_idx)}
    return train_and_predict


def test_shards_cover_every_block_exactly_once(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "train_and_predict", _stub(calls))
    close = _close()
    n_blocks = len(refit_plan(close, 3, 12))
    first = runner.run_design(close, tmp_path, "EUR_USD", lambda: None, shard=(0, 2))
    assert first is None                                   # half the blocks missing: no scoring yet
    pooled = runner.run_design(close, tmp_path, "EUR_USD", lambda: None, shard=(1, 2), reverse=True)
    assert len(calls) == n_blocks == len(set(calls))       # each block trained once, by one shard
    assert pooled is not None and "maestro_ungated" in set(pooled["strategy"])


def test_deadline_stops_new_blocks(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "train_and_predict", _stub(calls))
    assert runner.run_design(_close(), tmp_path, "EUR_USD", lambda: None,
                             deadline=time.time() + 60) is None
    assert calls == []                                     # a 20-minute block can't fit in a minute


def test_block_id_lists_round_trip():
    ids = {0, 1, 2, 3, 7, 9, 10}
    assert runner.format_ids(ids) == "0-3,7,9-10"
    assert runner.parse_ids(runner.format_ids(ids)) == ids
    assert runner.parse_ids("") == set()


def test_skipped_blocks_are_not_trained(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "train_and_predict", _stub(calls))
    close = _close()
    blocks = refit_plan(close, 3, 12)
    runner.run_design(close, tmp_path, "EUR_USD", lambda: None, skip={0, 1})
    assert len(calls) == len(blocks) - 2
    assert blocks[0].test_idx[0] not in calls and blocks[1].test_idx[0] not in calls


def test_limit_trains_at_most_n_blocks_per_call(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "train_and_predict", _stub(calls))
    close = _close()
    runner.run_design(close, tmp_path, "EUR_USD", lambda: None, limit=1)
    runner.run_design(close, tmp_path, "EUR_USD", lambda: None, limit=1)
    assert len(calls) == 2 == len(set(calls))              # a second call picks up the next block


def test_five_minute_position_rules_are_unchanged():
    assert runner.position_windows("M5") == (1440, 288, 12)      # as fixed for Chapter 7
    assert runner.position_windows("D") == (60, 20, 1)            # floors for long bars


def test_longer_bars_never_fall_back_to_five_minute_features(tmp_path, monkeypatch):
    import pytest
    import maestro.backtesting.baselines as b
    monkeypatch.setattr(b, "DATA_DIR", tmp_path)
    (tmp_path / "EUR_USD_features.parquet").write_bytes(b"")     # only the 5-minute file exists
    assert b.feature_file("EUR_USD", "M5").name == "EUR_USD_features.parquet"
    with pytest.raises(FileNotFoundError):
        b.feature_file("EUR_USD", "H1")
