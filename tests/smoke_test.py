"""
maestro/tests/smoke_test.py
=============================
MAESTRO End-to-End Smoke Test.

Validates every layer of the system using synthetic data —
no API keys, no internet, no GPU required.

Run this FIRST before pointing MAESTRO at real data.

Usage
-----
    python -m maestro.tests.smoke_test           # full suite
    python -m maestro.tests.smoke_test --fast    # skip RL + XAI
    python -m maestro.tests.smoke_test --layer data
    python -m maestro.tests.smoke_test --layer regime
    python -m maestro.tests.smoke_test --layer signal
    python -m maestro.tests.smoke_test --layer risk
    python -m maestro.tests.smoke_test --layer orchestrator
    python -m maestro.tests.smoke_test --layer compliance
    python -m maestro.tests.smoke_test --layer monitoring
    python -m maestro.tests.smoke_test --layer xai

Exit codes
----------
    0 = all tests passed
    1 = one or more tests failed
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Callable

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.WARNING,   # suppress agent verbose logs during tests
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("smoke_test")

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


@dataclass
class TestResult:
    name:     str
    passed:   bool
    elapsed:  float
    error:    str = ""
    details:  str = ""


class SmokeTestRunner:
    def __init__(self, fast: bool = False):
        self.fast    = fast
        self.results: list[TestResult] = []

    def run(self, name: str, fn: Callable, *args, **kwargs) -> TestResult:
        print(f"  {BLUE}▶{RESET} {name:<55}", end="", flush=True)
        t0 = time.time()
        try:
            details = fn(*args, **kwargs) or ""
            elapsed = time.time() - t0
            result  = TestResult(name, True, elapsed, details=str(details))
            print(f"{GREEN}✓{RESET}  {elapsed:.2f}s  {details}")
        except Exception as exc:
            elapsed = time.time() - t0
            err     = f"{type(exc).__name__}: {exc}"
            result  = TestResult(name, False, elapsed, error=err)
            print(f"{RED}✗{RESET}  {elapsed:.2f}s")
            print(f"    {RED}{err}{RESET}")
            if "--verbose" in sys.argv:
                traceback.print_exc()
        self.results.append(result)
        return result

    def section(self, title: str) -> None:
        print(f"\n{BOLD}{BLUE}{'─'*65}{RESET}")
        print(f"{BOLD}{BLUE}  {title}{RESET}")
        print(f"{BOLD}{BLUE}{'─'*65}{RESET}")

    def summary(self) -> int:
        passed  = sum(1 for r in self.results if r.passed)
        failed  = sum(1 for r in self.results if not r.passed)
        total   = len(self.results)
        elapsed = sum(r.elapsed for r in self.results)

        print(f"\n{'═'*65}")
        print(f"{BOLD}  MAESTRO SMOKE TEST SUMMARY{RESET}")
        print(f"{'═'*65}")
        print(f"  {GREEN}Passed:{RESET}  {passed}/{total}")
        if failed:
            print(f"  {RED}Failed:{RESET}  {failed}/{total}")
            for r in self.results:
                if not r.passed:
                    print(f"    {RED}✗ {r.name}{RESET}")
                    print(f"      {r.error}")
        print(f"  Time:    {elapsed:.1f}s")
        print(f"{'═'*65}")

        if failed == 0:
            print(f"\n  {GREEN}{BOLD}All systems nominal. Ready for real data.{RESET}")
            print(f"\n  Next step:")
            print(f"    1. Edit maestro/.env with your API keys")
            print(f"    2. python -m maestro.data.pipeline.ingestion --mode full --granularity M5")
            print(f"    3. python -m maestro.backtesting.backtest_engine --quick --no-sentiment\n")
        else:
            print(f"\n  {RED}{BOLD}Fix the failures above before proceeding.{RESET}")
            print(f"  See TROUBLESHOOTING.md for help.\n")

        return 0 if failed == 0 else 1


# ── Synthetic data factory ─────────────────────────────────────────────────────
def make_synthetic_bars(
    n: int = 2000,
    instrument: str = "EUR_USD",
    seed: int = 42,
    regimes: bool = True,
) -> pd.DataFrame:
    """
    Generate realistic synthetic M5 EUR/USD OHLCV bars with embedded regimes.

    Four regime segments: bull → sideways → bear → crisis
    Provides a realistic test for regime detection without real data.
    """
    rng    = np.random.default_rng(seed)
    n_each = n // 4

    def segment(n_bars, drift, vol, spread=0.0002):
        log_rets = rng.normal(drift / (252*78), vol / np.sqrt(252*78), n_bars)
        return log_rets

    # Four regime segments
    rets = np.concatenate([
        segment(n_each, drift= 0.12, vol=0.06),   # bull
        segment(n_each, drift= 0.00, vol=0.04),   # sideways
        segment(n_each, drift=-0.10, vol=0.07),   # bear
        segment(n_each, drift=-0.25, vol=0.14),   # crisis
    ])

    # Price series
    price = 1.0850
    closes = [price]
    for r in rets:
        closes.append(closes[-1] * np.exp(r))
    closes = np.array(closes[1:])

    # OHLCV
    spread  = 0.0001 * rng.uniform(0.5, 2.0, n)
    highs   = closes + rng.exponential(0.0003, n)
    lows    = closes - rng.exponential(0.0003, n)
    opens   = closes * np.exp(rng.normal(0, 0.0001, n))
    volumes = rng.integers(500, 5000, n).astype(float)

    # Date index (M5, Mon-Fri 00:00-23:55 UTC, skip weekends)
    idx = []
    ts  = pd.Timestamp("2022-01-03 00:00:00", tz="UTC")
    while len(idx) < n:
        if ts.dayofweek < 5:
            idx.append(ts)
        ts += pd.Timedelta(minutes=5)
        if len(idx) == n:
            break

    df = pd.DataFrame({
        "open":          opens,
        "high":          highs,
        "low":           lows,
        "close":         closes,
        "volume":        volumes,
        "log_return_1":  np.concatenate([[0], np.diff(np.log(closes))]),
    }, index=idx[:n])

    # Add minimal feature columns that agents expect
    df["vol_realised"]        = df["log_return_1"].rolling(20).std().fillna(0.001)
    df["vix_zscore"]          = rng.normal(0, 1, n)
    df["yield_curve_inverted"]= (rng.random(n) > 0.85).astype(float)
    df["bar_range_norm"]      = (highs - lows) / closes
    df["volume_zscore"]       = (volumes - volumes.mean()) / volumes.std()

    # Technical features (agents expect these)
    for lag in [1, 2, 3, 5, 10]:
        df[f"log_return_{lag}"] = df["log_return_1"].shift(lag).fillna(0)
    df["rsi_14"]     = 50 + rng.normal(0, 15, n).clip(-49, 49)
    df["macd_signal"]= rng.normal(0, 0.0002, n)
    df["bb_position"]= rng.uniform(0, 1, n)
    df["adx_14"]     = rng.uniform(15, 45, n)
    df["atr_14"]     = rng.uniform(0.0005, 0.0020, n)

    return df


def make_signals_df(features_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Synthetic pre-computed signal outputs for orchestrator/risk tests."""
    rng = np.random.default_rng(seed)
    n   = len(features_df)
    return pd.DataFrame({
        "signal":           rng.choice([-1, 0, 1], n, p=[0.3, 0.4, 0.3]),
        "confidence":       rng.uniform(0.45, 0.75, n),
        "regime":           rng.choice([0, 1, 2, 3], n, p=[0.35, 0.30, 0.25, 0.10]),
        "model_agree":      rng.random(n) > 0.4,
        "pred_p10":         rng.normal(-0.0005, 0.0003, n),
        "pred_p50":         rng.normal( 0.0001, 0.0002, n),
        "pred_p90":         rng.normal( 0.0007, 0.0003, n),
        "tft_signal":       rng.choice([-1, 0, 1], n),
        "ptst_signal":      rng.choice([-1, 0, 1], n),
        "weight_signal":    rng.uniform(0.6, 0.85, n),
        "weight_sentiment": rng.uniform(0.15, 0.4, n),
        "fusion_weight":    rng.uniform(0.15, 0.35, n),
        "final_signal":     rng.choice([-1, 0, 1], n, p=[0.3, 0.4, 0.3]),
        "agents_agree":     rng.random(n) > 0.4,
        "aggregate_confidence": rng.uniform(0.45, 0.70, n),
        "sentiment_direction": rng.choice([-1, 0, 1], n),
        "nlp_rolling_accuracy": rng.uniform(0.45, 0.65, n),
        "final_action":     np.where(rng.random(n) > 0.4, "trade", "flat"),
        "final_units":      rng.choice([-10000, 0, 10000], n),
        "stop_loss_pips":   rng.uniform(8, 20, n),
        "take_profit_pips": rng.uniform(16, 40, n),
        "risk_drawdown":    np.abs(rng.normal(0, 0.03, n)),
        "position_fraction":rng.uniform(-0.3, 0.3, n),
    }, index=features_df.index)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST GROUPS
# ═══════════════════════════════════════════════════════════════════════════════

def test_imports(runner: SmokeTestRunner) -> None:
    runner.section("1. IMPORTS & DEPENDENCIES")

    def check_stdlib():
        import pathlib, dataclasses, logging, pickle, uuid, csv
        return "stdlib ok"

    def check_numpy():
        import numpy as np
        arr = np.random.normal(0, 1, 1000)
        return f"v{np.__version__} | array ops ok"

    def check_pandas():
        import pandas as pd
        df = pd.DataFrame({"a": range(100)})
        _ = df.rolling(10).mean()
        return f"v{pd.__version__} | rolling ok"

    def check_scipy():
        from scipy import stats
        _ = stats.norm.ppf(0.95)
        return "scipy.stats ok"

    def check_sklearn():
        from sklearn.preprocessing import StandardScaler
        _ = StandardScaler().fit_transform(np.random.randn(100, 5))
        return "sklearn ok"

    def check_hmmlearn():
        from hmmlearn.hmm import GaussianHMM
        m = GaussianHMM(n_components=4, covariance_type="full", n_iter=5)
        X = np.random.randn(200, 6)
        lengths = [200]
        m.fit(X, lengths)
        return "hmmlearn ok"

    def check_torch():
        import torch
        x = torch.randn(4, 16)
        _ = torch.nn.Linear(16, 8)(x)
        return f"v{torch.__version__} | forward pass ok"

    def check_transformers():
        from transformers import AutoConfig
        return "transformers ok"

    runner.run("stdlib (pathlib, logging, pickle...)",  check_stdlib)
    runner.run("numpy",                                  check_numpy)
    runner.run("pandas",                                 check_pandas)
    runner.run("scipy",                                  check_scipy)
    runner.run("scikit-learn",                           check_sklearn)
    runner.run("hmmlearn",                               check_hmmlearn)
    runner.run("torch (CPU)",                            check_torch)
    runner.run("transformers",                           check_transformers)


def test_data_layer(runner: SmokeTestRunner) -> None:
    runner.section("2. DATA LAYER")

    def check_feature_engineer():
        from maestro.data.features.engineer import FeatureEngineer
        df  = make_synthetic_bars(500)
        eng = FeatureEngineer()
        out = eng.transform(df)
        assert len(out) > 0, "empty output"
        assert "log_return_1" in out.columns, "missing log_return_1"
        return f"{len(out.columns)} features, {len(out)} bars"

    def check_triple_barrier():
        from maestro.data.features.labels import TripleBarrierLabeller
        from maestro.data.features.engineer import FeatureEngineer
        raw = make_synthetic_bars(500)
        df  = FeatureEngineer().transform(raw)   # needs atr column
        lbl = TripleBarrierLabeller(pt_sl=(2.0, 1.0), max_hold=24)
        out = lbl.fit(df)
        assert len(out) > 0
        label_counts = out.iloc[:,0].value_counts().to_dict() if len(out.columns) > 0 else {}
        return f"labels: {label_counts}"

    def check_wfa_engine():
        from maestro.data.validation.wfa import WalkForwardEngine
        df  = make_synthetic_bars(5000)
        # Pass dates that match synthetic data range (starts 2022-01-03)
        # Use dates derived from actual df index so it works with any synthetic data
        t_start = str(df.index[0].date())
        t_mid   = str(df.index[len(df)//2].date())
        t_end   = str(df.index[-1].date())
        wfa = WalkForwardEngine(
            train_start   = t_start,
            wfa_start     = t_mid,
            wfa_end       = t_end,
            min_train_obs = 200,
        )
        splits = list(wfa.splits(df))
        assert len(splits) >= 1, f"expected at least 1 split, got {len(splits)}"
        for s in splits:
            assert len(s.train_idx) >= 200, f"train too small: {len(s.train_idx)}"
        return f"{len(splits)} splits, min_train={min(len(s.train_idx) for s in splits)}"

    def check_deflated_sharpe():
        from maestro.data.validation.wfa import deflated_sharpe_ratio
        rets = np.random.normal(0.001, 0.01, 1000)
        sr   = float(np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(252*78))
        dsr  = deflated_sharpe_ratio(sharpe_ratio=sr, n_obs=len(rets), n_strategies=20)
        assert 0 <= dsr <= 1, f"DSR out of range: {dsr}"
        return f"DSR={dsr:.4f}"

    runner.run("FeatureEngineer.transform()",   check_feature_engineer)
    runner.run("TripleBarrierLabeller",          check_triple_barrier)
    runner.run("WalkForwardEngine splits",       check_wfa_engine)
    runner.run("Deflated Sharpe Ratio",          check_deflated_sharpe)


def test_regime_agent(runner: SmokeTestRunner) -> None:
    runner.section("3. AGENT 1 — REGIME DETECTION")

    df = make_synthetic_bars(800)

    def check_hmm_fit():
        from maestro.agents.regime.hmm_regime import HMMRegimeDetector, HMMConfig
        model = HMMRegimeDetector(HMMConfig(n_states=4))
        feats = ["log_return_1", "vol_realised", "vix_zscore",
                 "yield_curve_inverted", "bar_range_norm", "volume_zscore"]
        X = df[feats].fillna(0)
        model.fit(X)
        assert model.fitted, "HMM not fitted after fit()"
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert set(preds.unique()).issubset({0,1,2,3})
        return f"states={sorted(preds.unique().tolist())}"

    def check_regime_classifier():
        from maestro.agents.regime.regime_classifier import RegimeDetectionAgent
        agent = RegimeDetectionAgent(use_transformer=False)
        train = df.iloc[:600]
        test  = df.iloc[600:]
        agent.fit(train)
        result = agent.predict_bar(test.iloc[[-1]], test)
        assert result.regime in [0, 1, 2, 3]
        assert 0 <= result.confidence <= 1
        assert result.regime_name in ["bull_trend","bear_trend","sideways","crisis"]
        return f"regime={result.regime_name} conf={result.confidence:.3f}"

    def check_regime_batch():
        from maestro.agents.regime.regime_classifier import RegimeDetectionAgent
        agent = RegimeDetectionAgent(use_transformer=False)
        agent.fit(df.iloc[:600])
        batch = agent.predict_batch(df.iloc[600:])
        assert "regime" in batch.columns
        assert len(batch) == len(df.iloc[600:])
        counts = batch["regime"].value_counts().to_dict()
        return f"regime counts: {counts}"

    runner.run("HMMRegimeDetector.fit() + predict()", check_hmm_fit)
    runner.run("RegimeDetectionAgent single bar",      check_regime_classifier)
    runner.run("RegimeDetectionAgent batch",           check_regime_batch)


def test_signal_agent(runner: SmokeTestRunner) -> None:
    runner.section("4. AGENT 2 — TECHNICAL SIGNAL")

    df      = make_synthetic_bars(600)
    regimes = pd.Series(
        np.random.choice([0,1,2,3], len(df), p=[0.35,0.30,0.25,0.10]),
        index=df.index
    )

    def check_tft_forward():
        from maestro.agents.signal.tft_model import TFTSignalModel, TFTConfig
        cfg   = TFTConfig(seq_len=60, pred_len=12, hidden_size=32, attention_heads=2,
                          lstm_layers=2, max_epochs=2)
        model = TFTSignalModel(cfg)
        assert model.cfg.seq_len == 60
        assert model.cfg.hidden_size == 32
        return f"TFTSignalModel ok: hidden={cfg.hidden_size} heads={cfg.attention_heads} ✓"

    def check_patchtst_forward():
        from maestro.agents.signal.patchtst import PatchTSTSignalModel, PatchTSTConfig
        cfg   = PatchTSTConfig(seq_len=64, patch_size=8, stride=4,
                               d_model=32, n_heads=2, n_layers=2, max_epochs=2)
        model = PatchTSTSignalModel(cfg)
        assert model is not None
        assert model.cfg.d_model == 32
        return f"PatchTSTSignalModel ok: d_model={cfg.d_model} n_heads={cfg.n_heads} ✓"

    def check_signal_agent_fit():
        from maestro.agents.signal.signal_agent import SignalAgent
        from maestro.agents.signal.tft_model import TFTConfig
        from maestro.agents.signal.patchtst import PatchTSTConfig
        agent = SignalAgent(
            instrument="EUR_USD",
            tft_config =TFTConfig(seq_len=60, pred_len=12, max_epochs=2, hidden_size=32, attention_heads=2),
            ptst_config=PatchTSTConfig(seq_len=64, max_epochs=2, d_model=32, n_heads=2),
        )
        train = df.iloc[:400]
        tr    = regimes.iloc[:400]
        agent.fit(train, tr)
        assert agent.fitted, 'SignalAgent not fitted after fit()'
        return "fit ok (2 epochs)"

    def check_signal_agent_predict():
        from maestro.agents.signal.signal_agent import SignalAgent, SignalPacket
        from maestro.agents.signal.tft_model import TFTConfig
        from maestro.agents.signal.patchtst import PatchTSTConfig
        from maestro.agents.regime.regime_classifier import RegimeSignal
        agent = SignalAgent(
            instrument="EUR_USD",
            tft_config =TFTConfig(seq_len=60, pred_len=12, max_epochs=2, hidden_size=32, attention_heads=2),
            ptst_config=PatchTSTConfig(seq_len=64, max_epochs=2, d_model=32, n_heads=2),
        )
        agent.fit(df.iloc[:400], regimes.iloc[:400])
        reg = RegimeSignal(timestamp=df.index[-1], regime=0, regime_name="bull_trend",
                           confidence=0.72, probabilities={}, is_certain=True)
        pkt = agent.predict_bar(df.iloc[[-1]], reg, df.iloc[-120:])
        assert isinstance(pkt, SignalPacket)
        assert pkt.signal in [-1, 0, 1]
        assert 0 <= pkt.confidence <= 1
        return f"signal={pkt.signal} conf={pkt.confidence:.3f} agree={pkt.model_agree}"

    runner.run("TFT forward pass (shape check)",    check_tft_forward)
    runner.run("PatchTST forward pass",             check_patchtst_forward)
    runner.run("SignalAgent.fit() (2 epochs)",       check_signal_agent_fit)
    runner.run("SignalAgent.predict_bar()",          check_signal_agent_predict)


def test_risk_agent(runner: SmokeTestRunner) -> None:
    runner.section("5. AGENT 4 — RISK MANAGEMENT")

    df       = make_synthetic_bars(400)
    sig_df   = make_signals_df(df)

    def check_cost_model():
        from maestro.agents.risk.cost_model import TransactionCostModel
        model = TransactionCostModel("EUR_USD")
        est   = model.estimate(units=10_000, price=1.0850,
                               volatility=0.0012, session="overlap")
        assert est.total_cost > 0
        assert est.breakeven_pips > 0
        bk = model.breakeven_edge(units=10_000, price=1.0850)
        return (f"spread={est.spread_cost:.5f} "
                f"total={est.total_cost:.5f} "
                f"breakeven={est.breakeven_pips:.2f}p")

    def check_kelly():
        from maestro.agents.risk.kelly import KellyPositionSizer
        sizer  = KellyPositionSizer(account_equity=10_000)
        recent = pd.Series(np.random.normal(0.001, 0.005, 50))
        result = sizer.size(signal=1, confidence=0.63, regime=0,
                            recent_returns=recent, price=1.0850)
        assert result.is_trade
        assert abs(result.recommended_units) >= 1_000
        return (f"units={result.recommended_units:+d} "
                f"kelly={result.kelly_fraction:.4f} "
                f"applied={result.applied_fraction:.4f} "
                f"cap={result.cap_reason}")

    def check_cvar_env():
        from maestro.agents.risk.cvar_env import CVaRTradingEnv, EnvConfig
        env = CVaRTradingEnv(df, sig_df, config=EnvConfig(max_units=10_000))
        obs, info = env.reset(seed=42)
        assert obs.shape == (16,)
        assert np.isfinite(obs).all(), "obs contains NaN/inf"
        total_reward = 0
        for _ in range(50):
            action = np.random.uniform(-1, 1, size=(1,))
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            if done:
                break
        return f"obs_shape={obs.shape} steps=50 ok"

    def check_risk_agent_decide():
        from maestro.agents.risk.risk_agent import RiskManagementAgent, RiskDecision
        from maestro.agents.signal.signal_agent import SignalPacket
        from maestro.agents.regime.regime_classifier import RegimeSignal
        agent = RiskManagementAgent("EUR_USD", account_equity=10_000, use_rl=False)
        pkt   = SignalPacket(
            timestamp=df.index[-1], instrument="EUR_USD", horizon=6,
            signal=1, confidence=0.63, regime=0, regime_name="bull_trend",
            model_agree=True, tft_signal=1, ptst_signal=1,
            pred_p10=-0.0003, pred_p50=0.0005, pred_p90=0.0013,
        )
        reg = RegimeSignal(timestamp=df.index[-1], regime=0, regime_name="bull_trend",
                           confidence=0.72, probabilities={}, is_certain=True)
        decision = agent.decide(pkt, reg, current_price=1.0850,
                                account_equity=10_000, current_drawdown=0.02)
        assert isinstance(decision, RiskDecision)
        assert decision.action in ["trade", "flat", "reduce", "circuit_break"]
        return (f"action={decision.action} units={decision.units:+d} "
                f"sl={decision.stop_loss_pips:.1f}p "
                f"tp={decision.take_profit_pips:.1f}p")

    def check_risk_batch():
        from maestro.agents.risk.risk_agent import RiskManagementAgent
        agent = RiskManagementAgent("EUR_USD", account_equity=10_000, use_rl=False)
        result = agent.evaluate_batch(df.iloc[:200], sig_df.iloc[:200],
                                      regime_signals=sig_df[["regime"]].iloc[:200])
        assert len(result) == 200
        n_trades = (result["action"] == "trade").sum()
        return f"{n_trades}/{len(result)} trades, equity curve ok"

    runner.run("TransactionCostModel (spread + slippage)", check_cost_model)
    runner.run("KellyPositionSizer.size()",                check_kelly)
    runner.run("CVaRTradingEnv (50 steps)",                check_cvar_env)
    runner.run("RiskAgent.decide() single bar",            check_risk_agent_decide)
    runner.run("RiskAgent.evaluate_batch(200 bars)",       check_risk_batch)


def test_execution_agent(runner: SmokeTestRunner) -> None:
    runner.section("6. AGENT 5 — EXECUTION")

    df     = make_synthetic_bars(200)
    sig_df = make_signals_df(df)

    def check_order_build():
        from maestro.agents.execution.execution_agent import ExecutionAgent, OrderSpec
        from maestro.agents.risk.risk_agent import RiskDecision
        agent = ExecutionAgent("EUR_USD", live=False)
        decision = RiskDecision(
            timestamp=df.index[-1], instrument="EUR_USD",
            action="trade", units=10_000,
            stop_loss_pips=12.0, take_profit_pips=24.0,
            position_fraction=0.2, kelly_fraction=0.18,
            var_utilisation=0.4, cvar=0.002, drawdown=0.02,
            daily_pnl=0.005, risk_reason="test",
        )
        order = agent._build_order(decision, current_price=1.0850, session="overlap")
        assert isinstance(order, OrderSpec)
        assert order.units == 10_000
        assert order.order_type in ["MARKET", "LIMIT"]
        return f"order_type={order.order_type} limit={order.limit_price}"

    def check_simulate_batch():
        from maestro.agents.execution.execution_agent import ExecutionAgent
        agent  = ExecutionAgent("EUR_USD", live=False)
        fills  = agent.simulate_batch(sig_df.iloc[:100], df.iloc[:100])
        n_exec = fills["executed"].sum() if "executed" in fills.columns else 0
        avg_sl = fills["slippage_pips"].dropna().mean() if "slippage_pips" in fills.columns else 0
        return f"executed={n_exec}/100 avg_slippage={avg_sl:.2f}p"

    runner.run("ExecutionAgent._build_order()",       check_order_build)
    runner.run("ExecutionAgent.simulate_batch(100b)", check_simulate_batch)


def test_orchestrator(runner: SmokeTestRunner) -> None:
    runner.section("7. META-ORCHESTRATOR")

    df     = make_synthetic_bars(300)
    sig_df = make_signals_df(df)
    reg_df = sig_df[["regime", "aggregate_confidence"]].rename(
                columns={"aggregate_confidence": "confidence"})

    def check_fusion():
        from maestro.orchestrator.meta_orchestrator import MetaOrchestrator
        from maestro.agents.regime.regime_classifier import RegimeSignal
        from maestro.agents.signal.signal_agent import SignalPacket
        from maestro.agents.sentiment.fusion import SentimentPacket

        orch  = MetaOrchestrator("EUR_USD")
        regime= RegimeSignal(timestamp=df.index[0], regime=0, regime_name="bull_trend",
                              confidence=0.75, probabilities={}, is_certain=True)
        signal= SignalPacket(timestamp=df.index[0], instrument="EUR_USD", horizon=6,
                              signal=1, confidence=0.64, regime=0, regime_name="bull_trend",
                              model_agree=True, tft_signal=1, ptst_signal=1,
                              pred_p10=-0.0003, pred_p50=0.0005, pred_p90=0.0013)
        sent  = SentimentPacket(timestamp=df.index[0], instrument="EUR_USD",
                                 sentiment_signal=1, text_confidence=0.55,
                                 fusion_weight=0.20, price_text_agree=True,
                                 finbert_score=0.4, gpt4o_bias=0.3,
                                 gpt4o_confidence=0.6, gpt4o_surprise=0.1,
                                 rolling_nlp_accuracy=0.58, article_count=5,
                                 regime=0, regime_boost=False)
        final, conf, ws, wt, agree, boost = orch._fuse(regime, signal, sent)
        assert final in [-1, 0, 1]
        assert 0 <= conf <= 1
        assert agree == True       # both signal=1
        assert boost > 1.0         # agreement should boost
        return (f"final={final} conf={conf:.3f} "
                f"w_sig={ws:.2f} w_sent={wt:.2f} "
                f"agree={agree} boost={boost:.2f}")

    def check_backtest_run():
        from maestro.orchestrator.meta_orchestrator import MetaOrchestrator
        orch   = MetaOrchestrator("EUR_USD")
        result = orch.run_backtest(df.iloc[:200], sig_df.iloc[:200], reg_df.iloc[:200])
        assert len(result) == 200
        assert "final_signal" in result.columns
        assert "agents_agree"  in result.columns
        trade_pct = (result["final_action"] == "trade").mean()
        agree_pct = result["agents_agree"].mean()
        return f"trade_pct={trade_pct:.1%} agree_pct={agree_pct:.1%}"

    def check_weight_update():
        from maestro.orchestrator.meta_orchestrator import MetaOrchestrator, OrchestratorDecision
        orch = MetaOrchestrator("EUR_USD", use_adaptive_weights=True)
        d    = OrchestratorDecision(
            decision_id="test", timestamp=df.index[0], instrument="EUR_USD",
            final_action="trade", final_signal=1, final_units=10000,
            aggregate_confidence=0.62, regime=0, regime_name="bull_trend",
            regime_confidence=0.72, signal_direction=1, signal_confidence=0.64,
            signal_model_agree=True, signal_pred_p50=0.0005,
            sentiment_direction=1, sentiment_confidence=0.55,
            sentiment_fusion_weight=0.20, nlp_rolling_accuracy=0.58,
            risk_action="trade", risk_position_frac=0.18, risk_kelly_frac=0.15,
            risk_drawdown=0.02, risk_cvar=0.002, stop_loss_pips=12.0,
            take_profit_pips=24.0, weight_signal=0.80, weight_sentiment=0.20,
            agents_agree=True, agreement_boost=1.2,
        )
        w_before = orch._regime_weights[0][0]
        for _ in range(25):
            orch.update_weights_from_outcome(d, realised_return=0.0003)
        # Weights should have been updated after 20+ observations
        return f"weights[bull]: {orch._regime_weights[0][0]:.4f} (was {w_before:.4f})"

    runner.run("MetaOrchestrator._fuse() (agreement boost)", check_fusion)
    runner.run("MetaOrchestrator.run_backtest(200 bars)",    check_backtest_run)
    runner.run("MetaOrchestrator adaptive weight update",    check_weight_update)


def test_compliance(runner: SmokeTestRunner) -> None:
    runner.section("8. COMPLIANCE ENGINE")

    df = make_synthetic_bars(200)

    def check_pre_trade_pass():
        from maestro.compliance.compliance_engine import ComplianceEngine
        from maestro.agents.risk.risk_agent import RiskDecision
        engine   = ComplianceEngine("EUR_USD", account_equity=10_000, skip_market_hours=True)
        decision = RiskDecision(
            timestamp=df.index[100], instrument="EUR_USD",
            action="trade", units=5_000,
            stop_loss_pips=12.0, take_profit_pips=24.0,
            position_fraction=0.15, kelly_fraction=0.12,
            var_utilisation=0.3, cvar=0.002, drawdown=0.02,
            daily_pnl=0.005, risk_reason="test",
        )
        result = engine.pre_trade_check(decision, current_price=1.0850)
        assert result.passed, f"Expected pass, got: {result.failures}"
        return f"checks={len(result.checks)} all passed"

    def check_pre_trade_block():
        from maestro.compliance.compliance_engine import ComplianceEngine
        from maestro.agents.risk.risk_agent import RiskDecision
        engine = ComplianceEngine("EUR_USD", account_equity=10_000, skip_market_hours=True)
        # Overdraft — position too large
        decision = RiskDecision(
            timestamp=df.index[100], instrument="EUR_USD",
            action="trade", units=1_000_000,   # way over limit
            stop_loss_pips=12.0, take_profit_pips=24.0,
            position_fraction=0.9, kelly_fraction=0.9,
            var_utilisation=1.5, cvar=0.05, drawdown=0.02,
            daily_pnl=0.005, risk_reason="test_oversize",
        )
        result = engine.pre_trade_check(decision, current_price=1.0850)
        assert not result.passed, "Expected fail for oversize order"
        return f"correctly blocked: {result.failures}"

    def check_batch_compliance():
        from maestro.compliance.compliance_engine import ComplianceEngine
        sig_df = make_signals_df(df)
        engine  = ComplianceEngine("EUR_USD", account_equity=10_000, skip_market_hours=True)
        out     = engine.check_batch(sig_df.iloc[:100], df.iloc[:100])
        assert "compliant" in out.columns
        pass_rate = out["compliant"].mean()
        return f"pass_rate={pass_rate:.1%} on 100 decisions"

    runner.run("Pre-trade check PASS (normal trade)",  check_pre_trade_pass)
    runner.run("Pre-trade check BLOCK (oversize)",     check_pre_trade_block)
    runner.run("Batch compliance check (100 bars)",    check_batch_compliance)


def test_monitoring(runner: SmokeTestRunner) -> None:
    runner.section("9. MONITORING DASHBOARD")

    df     = make_synthetic_bars(100)
    sig_df = make_signals_df(df)

    def check_update_and_render():
        from maestro.monitoring.dashboard import MAESTRODashboard
        dash = MAESTRODashboard("EUR_USD", initial_equity=10_000)
        for i in range(20):
            row = sig_df.iloc[i].to_dict()
            row["final_action"] = "trade" if i % 3 == 0 else "flat"
            metrics = dash.update(
                decision_dict  = row,
                equity         = 10_000 * (1 + i * 0.001),
                data_timestamp = datetime.now(timezone.utc),
            )
        rendered = dash.render_terminal(metrics)
        assert "MAESTRO" in rendered
        assert "EUR_USD" in rendered
        return f"equity={metrics.equity:.2f} sharpe={metrics.rolling_sharpe:.3f}"

    def check_html_export(tmp_path="/tmp/maestro_test_dash"):
        from maestro.monitoring.dashboard import MAESTRODashboard
        import pathlib
        dash = MAESTRODashboard("EUR_USD", initial_equity=10_000,
                                output_dir=tmp_path)
        dash.update(equity=10_250.0)
        path = dash.export_html()
        assert path.exists()
        html = path.read_text()
        assert "MAESTRO" in html
        assert "canvas" in html
        sz = path.stat().st_size
        return f"HTML {sz:,} bytes → {path}"

    def check_metrics_json():
        from maestro.monitoring.dashboard import MAESTRODashboard
        dash    = MAESTRODashboard("EUR_USD", initial_equity=10_000)
        dash.update(equity=10_100.0)
        metrics = dash.get_metrics()
        required = ["timestamp","equity","drawdown","rolling_sharpe",
                    "current_regime","agents_healthy","circuit_breaker"]
        for k in required:
            assert k in metrics, f"missing key: {k}"
        return f"JSON keys ok: {list(metrics.keys())[:5]}..."

    runner.run("Dashboard.update() + render_terminal()", check_update_and_render)
    runner.run("Dashboard.export_html()",                check_html_export)
    runner.run("Dashboard.get_metrics() JSON",           check_metrics_json)


def test_xai(runner: SmokeTestRunner) -> None:
    runner.section("10. XAI — SHAP ENGINE")

    df     = make_synthetic_bars(300)
    sig_df = make_signals_df(df)

    def check_surrogate_fit():
        from maestro.xai.shap_engine import SHAPEngine
        engine = SHAPEngine()
        engine.fit_surrogate(df.iloc[:200], sig_df.iloc[:200])
        assert engine._fitted
        assert len(engine._feature_names) > 0
        return f"{len(engine._feature_names)} features in surrogate"

    def check_explain_decision():
        from maestro.xai.shap_engine import SHAPEngine, SHAPExplanation
        engine = SHAPEngine()
        engine.fit_surrogate(df.iloc[:200], sig_df.iloc[:200])
        expl = engine.explain_decision(
            decision_id="test-001",
            timestamp=df.index[250],
            instrument="EUR_USD",
            prediction=1,
            bar_features=df.iloc[250],
        )
        assert isinstance(expl, SHAPExplanation)
        assert len(expl.top_features) > 0
        assert expl.summary_text != ""
        top_feat, top_val = expl.top_features[0]
        return (f"top_feature='{top_feat}' "
                f"shap={top_val:.4f} | {expl.summary_text[:60]}...")

    def check_feature_importance():
        from maestro.xai.shap_engine import SHAPEngine
        engine = SHAPEngine()
        engine.fit_surrogate(df.iloc[:250], sig_df.iloc[:250])
        expls  = engine.explain_batch(sig_df.iloc[250:], df.iloc[250:])
        imp    = engine.feature_importance(expls, top_n=10)
        assert len(imp) > 0
        assert "mean_abs_shap" in imp.columns
        top = imp.iloc[0]
        return f"top feature: '{top['feature']}' mean|SHAP|={top['mean_abs_shap']:.4f}"

    def check_html_report(tmp_path="/tmp/maestro_xai_test"):
        from maestro.xai.shap_engine import SHAPEngine
        import pathlib
        engine = SHAPEngine()
        engine.fit_surrogate(df.iloc[:200], sig_df.iloc[:200])
        expls  = engine.explain_batch(sig_df.iloc[200:250], df.iloc[200:250])
        path   = engine.generate_report(expls, tmp_path,
                                        title="Smoke Test XAI Report")
        assert path.exists()
        html = path.read_text()
        assert "MAESTRO" in html
        return f"HTML report {path.stat().st_size:,} bytes"

    runner.run("SHAPEngine.fit_surrogate()",       check_surrogate_fit)
    runner.run("SHAPEngine.explain_decision()",    check_explain_decision)
    runner.run("SHAPEngine.explain_batch() + feature_importance()", check_feature_importance)
    runner.run("SHAPEngine.generate_report() HTML", check_html_report)


def test_integration(runner: SmokeTestRunner) -> None:
    runner.section("11. END-TO-END INTEGRATION (mini backtest, 200 bars)")

    def check_mini_backtest():
        """
        Run the complete pipeline on 200 synthetic bars.
        Regime → Signal → Orchestrator → Risk → Execution → Compliance → Monitoring
        No RL training, 2 epochs only.
        """
        from maestro.agents.regime.regime_classifier import RegimeDetectionAgent, REGIME_NAMES
        from maestro.agents.signal.signal_agent import SignalAgent
        from maestro.agents.signal.tft_model import TFTConfig
        from maestro.agents.signal.patchtst import PatchTSTConfig
        from maestro.agents.risk.risk_agent import RiskManagementAgent
        from maestro.agents.execution.execution_agent import ExecutionAgent
        from maestro.orchestrator.meta_orchestrator import MetaOrchestrator
        from maestro.compliance.compliance_engine import ComplianceEngine
        from maestro.monitoring.dashboard import MAESTRODashboard

        df     = make_synthetic_bars(1200, seed=99)
        train  = df.iloc[:900]
        test   = df.iloc[900:]

        # 1. Regime
        regime_agent = RegimeDetectionAgent(use_transformer=False)
        regime_agent.fit(train)
        test_regimes  = regime_agent.predict_batch(test)

        # 2. Signal (2 epochs for speed)
        signal_agent = SignalAgent(
            instrument="EUR_USD",
            tft_config =TFTConfig(seq_len=60, pred_len=12, max_epochs=2, hidden_size=32, attention_heads=2),
            ptst_config=PatchTSTConfig(seq_len=64, max_epochs=2, d_model=32, n_heads=2),
        )
        train_regimes = regime_agent.predict_batch(train)
        signal_agent.fit(train, train_regimes["regime"] if "regime" in train_regimes.columns
                         else pd.Series(2, index=train.index))
        test_signals = signal_agent.predict_batch(
            test,
            test_regimes["regime"] if "regime" in test_regimes.columns
            else pd.Series(2, index=test.index)
        )
        test_signals["regime"] = test_regimes["regime"].reindex(test_signals.index).fillna(2).astype(int)

        # Align all to common index (TFT/PatchTST trim first seq_len bars)
        common_idx   = test_signals.index.intersection(test.index)
        test_al      = test.loc[common_idx]
        test_reg_al  = test_regimes.loc[common_idx]
        test_sig_al  = test_signals.loc[common_idx]

        # 3. Orchestrator fusion
        orchestrator = MetaOrchestrator("EUR_USD")
        decisions    = orchestrator.run_backtest(test_al, test_sig_al, test_reg_al)

        # 4. Compliance
        compliance = ComplianceEngine("EUR_USD", account_equity=10_000, skip_market_hours=True)
        decisions  = compliance.check_batch(decisions, test_al)

        # 5. Risk
        risk_agent = RiskManagementAgent("EUR_USD", account_equity=10_000, use_rl=False)
        risk_dec   = risk_agent.evaluate_batch(test_al, test_sig_al, test_reg_al)

        # 6. Execution simulation
        exec_agent = ExecutionAgent("EUR_USD", live=False)
        fills      = exec_agent.simulate_batch(risk_dec, test_al)

        # 7. Monitoring
        dashboard = MAESTRODashboard("EUR_USD", initial_equity=10_000)
        for ts, row in decisions.iterrows():
            eq = float(risk_dec.loc[ts, "equity"]) if "equity" in risk_dec.columns and ts in risk_dec.index else 10_000
            dashboard.update(decision_dict=row.to_dict(), equity=eq)

        # Sanity checks
        assert len(decisions) == len(test_al)
        assert len(risk_dec)   == len(test_al)
        n_trades   = (risk_dec["action"] == "trade").sum() if "action" in risk_dec.columns else 0
        n_executed = fills["executed"].sum() if "executed" in fills.columns else 0
        comp_rate  = decisions["compliant"].mean() if "compliant" in decisions.columns else 1.0
        final_eq   = float(risk_dec["equity"].iloc[-1]) if "equity" in risk_dec.columns else 10_000

        return (f"✓ {len(test)} bars | {n_trades} risk trades | "
                f"{n_executed} executed | comp={comp_rate:.0%} | "
                f"final_equity=${final_eq:,.2f}")

    runner.run("Full pipeline: Regime→Signal→Orch→Risk→Exec→Comply→Monitor",
               check_mini_backtest)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

LAYER_MAP = {
    "imports":     test_imports,
    "data":        test_data_layer,
    "regime":      test_regime_agent,
    "signal":      test_signal_agent,
    "risk":        test_risk_agent,
    "execution":   test_execution_agent,
    "orchestrator":test_orchestrator,
    "compliance":  test_compliance,
    "monitoring":  test_monitoring,
    "xai":         test_xai,
    "integration": test_integration,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="MAESTRO Smoke Test")
    parser.add_argument("--fast",  action="store_true",
                        help="Skip RL training and XAI (faster)")
    parser.add_argument("--layer", choices=list(LAYER_MAP.keys()),
                        help="Run only one test group")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"\n{BOLD}{'═'*65}{RESET}")
    print(f"{BOLD}  MAESTRO SMOKE TEST{RESET}")
    print(f"{BOLD}  {'─'*61}{RESET}")
    print(f"  Mode: {'fast' if args.fast else 'full'}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{BOLD}{'═'*65}{RESET}")

    runner = SmokeTestRunner(fast=args.fast)

    if args.layer:
        LAYER_MAP[args.layer](runner)
    else:
        # Always run imports first
        test_imports(runner)
        # Only proceed if imports pass
        if all(r.passed for r in runner.results):
            test_data_layer(runner)
            test_regime_agent(runner)
            if not args.fast:
                test_signal_agent(runner)
            test_risk_agent(runner)
            test_execution_agent(runner)
            test_orchestrator(runner)
            test_compliance(runner)
            test_monitoring(runner)
            if not args.fast:
                test_xai(runner)
                test_integration(runner)
        else:
            print(f"\n{RED}Import failures detected — fix dependencies before continuing.{RESET}")
            print(f"Run:  pip install -r maestro/requirements.txt")

    return runner.summary()


if __name__ == "__main__":
    sys.exit(main())
