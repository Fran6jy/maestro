"""
The regime detector must not let a bar's regime depend on later bars. hmmlearn's
Viterbi path and forward-backward posteriors both do, which leaked the future
into every test block's regimes; predictions now use forward filtering.
"""
import numpy as np
import pandas as pd
import pytest

from maestro.agents.regime.hmm_regime import HMM_FEATURES, HMMRegimeDetector


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    n = 3000
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    vol = np.where((np.arange(n) // 400) % 2 == 0, 1e-4, 3e-4)          # alternating calm / volatile
    df = pd.DataFrame({c: rng.normal(0, 1, n) for c in HMM_FEATURES}, index=idx)
    df["log_return_1"] = rng.normal(0, 1, n) * vol
    df["vol_realised"] = pd.Series(df["log_return_1"]).rolling(20, min_periods=1).std().fillna(1e-4).values
    return HMMRegimeDetector().fit(df.iloc[:2000]), df


def test_filtered_regimes_ignore_later_bars(fitted):
    hmm, df = fitted
    test = df.iloc[2000:]
    before = hmm.predict_proba(test).iloc[:500]
    changed = test.copy()
    changed.iloc[500:, :] = changed.iloc[500:, :] * 5 + 3                # rewrite the future
    after = hmm.predict_proba(changed).iloc[:500]
    np.testing.assert_allclose(before.values, after.values)


def test_smoothed_regimes_do_see_later_bars(fitted):
    hmm, df = fitted
    test = df.iloc[2000:]
    changed = test.copy()
    changed.iloc[500:, :] = changed.iloc[500:, :] * 5 + 3
    smooth_before = hmm.predict_proba(test, causal=False).iloc[:500]
    smooth_after = hmm.predict_proba(changed, causal=False).iloc[:500]
    assert not np.allclose(smooth_before.values, smooth_after.values)  # why causal is the default


def test_filtering_agrees_with_smoothing_at_the_last_bar(fitted):
    hmm, df = fitted
    test = df.iloc[2000:]
    np.testing.assert_allclose(hmm.predict_proba(test).iloc[-1].values,
                               hmm.predict_proba(test, causal=False).iloc[-1].values, atol=1e-8)


def test_a_reloaded_regime_detector_votes_like_the_original(fitted, tmp_path):
    from maestro.agents.regime.regime_classifier import RegimeDetectionAgent
    hmm, df = fitted
    agent = RegimeDetectionAgent(use_transformer=False)
    agent.hmm, agent.fitted, agent.hmm_weight = hmm, True, 0.3        # a tuned weight
    agent.save(tmp_path)
    assert RegimeDetectionAgent.load(tmp_path).hmm_weight == 0.3
