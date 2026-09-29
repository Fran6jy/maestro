"""
Aggregated edge test across ALL completed full-quality splits.
Pools directional calls from every saved split model evaluated on its OWN
test window (taken from that split's decision parquet index), then computes
one pooled hit rate + z-score. Far stronger than a single-split test.

Runs on CPU (CUDA hidden) to avoid contending with the live GPU backtest.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # force CPU; don't touch the busy GPU

import glob
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.ERROR)

from maestro.data.pipeline.ingestion import DataPipeline
from maestro.agents.regime.regime_classifier import RegimeDetectionAgent
from maestro.agents.signal.signal_agent import SignalAgent

MODELS = r"C:\tmp\maestro_models\EUR_USD"
OUTS   = r"C:\tmp\maestro_outputs\EUR_USD"

print("Loading feature data...")
feat = DataPipeline().load("EUR_USD", "M5")

split_dirs = sorted(glob.glob(MODELS + r"\split_*"))
print(f"Found {len(split_dirs)} saved split models\n")

pooled_pred_6, pooled_real_6 = [], []
per_split = []

for sd in split_dirs:
    sid = os.path.basename(sd).split("_")[-1]
    dec_path = os.path.join(OUTS, f"decisions_split_{sid}.parquet")
    if not (os.path.exists(os.path.join(sd, "signal")) and os.path.exists(dec_path)):
        continue
    try:
        idx = pd.read_parquet(dec_path).index
        test_df = feat.reindex(idx).dropna()
        if len(test_df) < 500:
            continue
        regime_agent = RegimeDetectionAgent.load(os.path.join(sd, "regime"))
        signal_agent = SignalAgent.load(os.path.join(sd, "signal"))
        regimes = regime_agent.predict_batch(test_df)["regime"]
        signals = signal_agent.predict_batch(test_df, regimes)

        h = signal_agent.primary_horizon
        close = test_df["close"].reindex(signals.index)
        fwd6 = (close.shift(-h) / close - 1.0)

        nonflat = signals["signal"] != 0
        sidx = signals.index[nonflat]
        pred = signals.loc[sidx, "signal"].values
        real = np.sign(fwd6.loc[sidx].fillna(0).values)
        valid = real != 0
        pred, real = pred[valid], real[valid]
        if len(pred) == 0:
            continue
        hit = (pred == real).mean()
        per_split.append((sid, len(pred), hit))
        pooled_pred_6.append(pred)
        pooled_real_6.append(real)
        print(f"  split {sid}: {len(pred):5d} calls, hit {hit*100:.1f}%")
    except Exception as e:
        print(f"  split {sid}: ERROR {e}")

print()
if pooled_pred_6:
    P = np.concatenate(pooled_pred_6)
    R = np.concatenate(pooled_real_6)
    hit = (P == R).mean()
    n = len(P)
    se = np.sqrt(0.25 / n)
    z = (hit - 0.5) / se
    print("=" * 58)
    print(f"POOLED across {len(per_split)} splits @ {signal_agent.primary_horizon}-bar horizon")
    print(f"  Total directional calls: {n}")
    print(f"  Pooled hit ratio: {hit*100:.2f}%")
    print(f"  z-score vs 50%:   {z:+.2f}   (|z|>1.96 = significant at 5%)")
    print(f"  95% CI on hit:    [{(hit-1.96*se)*100:.2f}%, {(hit+1.96*se)*100:.2f}%]")
    print("=" * 58)
    # naive cost check: edge in pips needed
    print(f"  Directional edge over 50%: {(hit-0.5)*100:+.2f} pp")
    print("  NOTE: hit rate alone ignores win/loss SIZE asymmetry and costs.")
    print("  A full PnL-net-of-cost read comes from the backtest equity curve.")
else:
    print("No pooled calls — check that splits trained and saved correctly.")
