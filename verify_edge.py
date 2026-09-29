"""
The decisive test: do the model's directional calls have ANY edge?
Ignore the confidence gate entirely — just ask: when the ensemble says BUY or
SELL, how often is it right vs the actual next-bar return direction?
If ~50%, there is no edge and no calibration tweak will create one.
"""
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING)

from maestro.data.pipeline.ingestion import DataPipeline
from maestro.agents.regime.regime_classifier import RegimeDetectionAgent
from maestro.agents.signal.signal_agent import SignalAgent

MODEL_DIR = r"C:\tmp\maestro_models\EUR_USD\split_000"

df = DataPipeline().load("EUR_USD", "M5")
test_df = df[(df.index >= pd.Timestamp("2023-01-02", tz="UTC")) &
             (df.index <= pd.Timestamp("2023-02-01", tz="UTC"))].dropna()

regime_agent = RegimeDetectionAgent.load(MODEL_DIR + r"\regime")
signal_agent = SignalAgent.load(MODEL_DIR + r"\signal")

test_regimes = regime_agent.predict_batch(test_df)
regime_series = test_regimes["regime"]
signals = signal_agent.predict_batch(test_df, regime_series)

# Forward return at the PRIMARY horizon (6 bars), not just 1 bar — the signal
# is a 6-bar-ahead forecast, so we must evaluate it against the 6-bar return.
h = signal_agent.primary_horizon
close = test_df["close"].reindex(signals.index)
fwd_h = (close.shift(-h) / close - 1.0)          # h-bar forward return
fwd_1 = test_df["log_return_1"].reindex(signals.index).shift(-1)  # next-bar

for label, fwd in [(f"{h}-bar forward", fwd_h), ("1-bar forward", fwd_1)]:
    nonflat = signals["signal"] != 0
    idx = signals.index[nonflat]
    pred = signals.loc[idx, "signal"].values
    real = np.sign(fwd.loc[idx].fillna(0).values)
    valid = real != 0
    pred, real = pred[valid], real[valid]
    if len(pred) == 0:
        print(f"\n{label}: no valid pairs")
        continue
    hit = (pred == real).mean()
    print(f"\n=== EDGE TEST vs {label} return ===")
    print(f"  Directional calls evaluated: {len(pred)}")
    print(f"  Hit ratio: {hit*100:.1f}%   (50% = coin flip, MSc baseline 37.6%, target 55%)")
    # binomial significance: is it distinguishable from 50%?
    se = np.sqrt(0.25 / len(pred))
    z = (hit - 0.5) / se
    print(f"  z-score vs 50%: {z:+.2f}  (|z|>1.96 = statistically significant at 5%)")
    if abs(z) < 1.96:
        print("  -> NOT distinguishable from random. No tradeable edge in this window.")
    elif hit > 0.5:
        print("  -> Statistically significant POSITIVE edge.")
    else:
        print("  -> Significant INVERSE edge (sign convention worth checking).")

print("\nNote: single split, quick-mode model (20 epochs, ~9.8k train bars).")
print("A real verdict needs the full run across all 39 splits.")
