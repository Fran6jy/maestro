"""
Verify the confidence-calibration fix on the ALREADY-TRAINED split_000 models.
No retraining — loads saved weights, re-runs inference with fixed ensemble code,
and reports trade count AND hit ratio (the honest test: is there real edge?).
"""
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING)

from maestro.data.pipeline.ingestion import DataPipeline
from maestro.agents.regime.regime_classifier import RegimeDetectionAgent
from maestro.agents.signal.signal_agent import SignalAgent

MODEL_DIR = r"C:\tmp\maestro_models\EUR_USD\split_000"

# Split 000 test window (from the decision parquet we inspected)
TEST_START = "2023-01-02"
TEST_END   = "2023-02-01"

print("Loading data...")
df = DataPipeline().load("EUR_USD", "M5")
test_df = df[(df.index >= pd.Timestamp(TEST_START, tz="UTC")) &
             (df.index <= pd.Timestamp(TEST_END, tz="UTC"))].dropna()
print(f"Test window: {len(test_df)} bars ({test_df.index[0]} -> {test_df.index[-1]})")

print("Loading trained agents (split_000)...")
regime_agent = RegimeDetectionAgent.load(MODEL_DIR + r"\regime")
signal_agent = SignalAgent.load(MODEL_DIR + r"\signal")

print("Running inference with FIXED confidence code...")
test_regimes = regime_agent.predict_batch(test_df)
regime_series = test_regimes["regime"] if "regime" in test_regimes.columns else pd.Series(2, index=test_df.index)
signals = signal_agent.predict_batch(test_df, regime_series)

# ── Confidence distribution ───────────────────────────────────────────────────
conf = signals["confidence"]
print("\n=== CONFIDENCE DISTRIBUTION (fixed) ===")
print(f"  min={conf.min():.3f}  median={conf.median():.3f}  "
      f"mean={conf.mean():.3f}  max={conf.max():.3f}")
for thr in [0.30, 0.40, 0.52, 0.55, 0.62]:
    print(f"  bars with confidence > {thr:.2f}: {(conf > thr).sum():5d} ({100*(conf>thr).mean():.1f}%)")

# ── Actionable signals & hit ratio ────────────────────────────────────────────
# Use the per-regime threshold the system actually applies
from maestro.agents.signal.signal_agent import CONFIDENCE_THRESHOLDS
actionable = signals.apply(
    lambda r: r["signal"] != 0 and r["confidence"] >= CONFIDENCE_THRESHOLDS.get(int(r["regime"]), 0.55),
    axis=1
)
n_actionable = int(actionable.sum())
print(f"\n=== ACTIONABLE SIGNALS ===")
print(f"  Actionable (signal != 0 AND conf >= regime threshold): {n_actionable} / {len(signals)} ({100*n_actionable/len(signals):.1f}%)")
print(f"  Non-flat signals (any confidence): {(signals['signal'] != 0).sum()}")

# ── HONEST TEST: hit ratio of actionable signals vs actual forward returns ─────
fwd = test_df["log_return_1"].reindex(signals.index).fillna(0)
act_idx = signals.index[actionable]
if len(act_idx) > 0:
    pred_dir = signals.loc[act_idx, "signal"].values
    real_dir = np.sign(fwd.loc[act_idx].values)
    hits = (pred_dir == real_dir)
    hit_ratio = hits.mean()
    print(f"\n=== HONEST EDGE TEST (the number that matters) ===")
    print(f"  Hit ratio of actionable signals: {hit_ratio*100:.1f}%  ({hits.sum()}/{len(hits)})")
    print(f"  MSc baseline: 37.6%  |  coin flip: 50%  |  target: 55%")
    if hit_ratio > 0.53:
        print("  -> Signal shows edge above chance. Worth a full re-run.")
    elif hit_ratio < 0.47:
        print("  -> Signal is INVERSELY predictive (possible sign error worth investigating).")
    else:
        print("  -> Hit ratio ~50%: signals are essentially NOISE. Trading them is gambling.")
        print("     Do NOT proceed to live trading on this. The model needs work, not the gate.")
else:
    print("\n  Still no actionable signals even after fix — threshold may need review,")
    print("  or the model genuinely has near-zero directional conviction.")
