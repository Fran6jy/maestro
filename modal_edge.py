"""
Modal harness — parallel cross-split edge test for MAESTRO.

Fans out the 39 walk-forward splits as independent GPU containers, trains the
full-quality regime + signal models on each split's training window, runs
inference on its test window, and returns directional hit-rate stats. The local
entrypoint pools all splits into one hit ratio + z-score.

Run:
    modal run maestro/modal_edge.py                  # all splits
    modal run maestro/modal_edge.py --max-splits 8   # first N splits only

Prereq (one-time data upload):
    modal volume create maestro-data
    modal volume put maestro-data "C:\\tmp\\maestro_data\\EUR_USD_features.parquet" /EUR_USD_features.parquet
"""
import os
import modal

app = modal.App("maestro-edge")

# ── Container image: CUDA torch + scientific stack + the maestro package ───────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "numpy", "pandas", "pyarrow", "scikit-learn", "scipy",
        "hmmlearn", "statsmodels", "pyyaml", "python-dateutil",
        "requests", "urllib3",
    )
    # copy only the package source; exclude the multi-GB local venv + junk
    .add_local_dir(
        "maestro",
        remote_path="/root/maestro",
        ignore=["venv", "venv/**", "**/__pycache__", "**/*.pyc", ".git", ".git/**",
                "**/*.pt", "**/*.pkl", "**/*.parquet"],
    )
)

data_vol = modal.Volume.from_name("maestro-data", create_if_missing=True)


# ── Per-split worker (one GPU container per split) ─────────────────────────────
@app.function(image=image, gpu="A10G", volumes={"/data": data_vol}, timeout=10800)
def run_split(split_id: int, instrument: str = "EUR_USD") -> dict:
    """Resilient wrapper: a failing split returns an error record rather than
    raising, so one bad split can never abort the whole fan-out."""
    import traceback
    try:
        return _run_split_impl(split_id, instrument)
    except Exception as exc:  # noqa: BLE001 — we want every failure recorded
        return {
            "split_id": split_id, "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc()[-1500:],
        }


def _run_split_impl(split_id: int, instrument: str = "EUR_USD") -> dict:
    import sys
    sys.path.insert(0, "/root")
    os.environ["MAESTRO_DATA_DIR"]   = "/data"
    os.environ["MAESTRO_MODEL_DIR"]  = "/tmp/models"
    os.environ["MAESTRO_OUTPUT_DIR"] = "/tmp/out"

    import numpy as np
    import pandas as pd
    from maestro.data.validation.wfa import WalkForwardEngine
    from maestro.agents.regime.regime_classifier import RegimeDetectionAgent
    from maestro.agents.signal.signal_agent import SignalAgent
    from maestro.agents.signal.tft_model import TFTConfig, HORIZONS
    from maestro.agents.signal.patchtst import PatchTSTConfig

    VAL_MONTHS, HORIZON, EPOCHS = 2, 6, 40

    # Read the feature parquet directly — avoids constructing the data
    # connectors (OANDA/FRED/News), which demand credentials we don't need
    # here. The volume holds {instrument}_features.parquet.
    df = pd.read_parquet(f"/data/{instrument}_features.parquet")
    df = df.sort_index()
    start, end = df.index[0], df.index[-1]
    wfa = WalkForwardEngine(
        train_start=str((start + pd.DateOffset(months=6)).date()),
        wfa_start=str((start + pd.DateOffset(months=12)).date()),
        wfa_end=str(end.date()),
    )
    splits = list(wfa.splits(df))
    if split_id >= len(splits):
        return {"split_id": split_id, "status": "out_of_range", "n_splits": len(splits)}

    split = splits[split_id]
    train_df = df.loc[split.train_idx]
    test_df  = df.loc[split.test_idx]
    if len(train_df) < 500:
        return {"split_id": split_id, "status": "skip_small_train", "n": len(train_df)}

    val_cut  = split.train_end - pd.DateOffset(months=VAL_MONTHS)
    val_mask = train_df.index >= val_cut
    val_df   = train_df[val_mask] if val_mask.sum() > 200 else None
    fit_df   = train_df[~val_mask] if val_df is not None else train_df
    fit_df   = fit_df.dropna()
    val_df   = val_df.dropna() if val_df is not None else None
    test_df  = test_df.dropna()
    if len(fit_df) < 100:
        return {"split_id": split_id, "status": "skip_nan", "n": len(fit_df)}

    # ── Train regime (HMM + Transformer) ──
    regime_agent = RegimeDetectionAgent(use_transformer=True)
    try:
        regime_agent.fit(fit_df, val_df=val_df)
    except Exception:
        regime_agent = RegimeDetectionAgent(use_transformer=False)
        regime_agent.fit(fit_df)
    train_regimes = regime_agent.predict_batch(fit_df)["regime"]
    val_regimes   = regime_agent.predict_batch(val_df)["regime"] if val_df is not None else None

    # ── Train signal (TFT + PatchTST) ──
    signal_agent = SignalAgent(
        instrument=instrument, primary_horizon=HORIZON,
        tft_config=TFTConfig(seq_len=120, pred_len=max(HORIZONS), max_epochs=EPOCHS, patience=6),
        ptst_config=PatchTSTConfig(seq_len=128, max_epochs=EPOCHS, patience=6),
    )
    signal_agent.fit(fit_df, train_regimes, val_df=val_df, val_regimes=val_regimes)

    # ── Inference + edge stats on test window ──
    test_regimes = regime_agent.predict_batch(test_df)["regime"]
    signals = signal_agent.predict_batch(test_df, test_regimes)

    close = test_df["close"].reindex(signals.index)
    fwd = (close.shift(-HORIZON) / close - 1.0)
    nonflat = signals["signal"] != 0
    sidx = signals.index[nonflat]
    pred = signals.loc[sidx, "signal"].values
    real = np.sign(fwd.loc[sidx].fillna(0).values)
    valid = real != 0
    pred, real = pred[valid], real[valid]
    n = int(len(pred))
    hits = int((pred == real).sum())

    return {
        "split_id": split_id, "status": "ok",
        "test_start": str(test_df.index[0]), "test_end": str(test_df.index[-1]),
        "n_calls": n, "n_hits": hits,
        "hit_rate": (hits / n) if n else None,
    }


# ── Fan-out entrypoint ─────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(max_splits: int = 39, instrument: str = "EUR_USD"):
    import numpy as np

    ids = list(range(max_splits))
    args = [(i, instrument) for i in ids]
    results = list(run_split.starmap(args, return_exceptions=True))
    # Modal-level failures (e.g. FunctionTimeoutError) arrive as exception objects
    results = [
        r if not isinstance(r, Exception)
        else {"split_id": i, "status": "error", "error": f"{type(r).__name__}: {r}"}
        for i, r in zip(ids, results)
    ]

    ok = [r for r in results if r.get("status") == "ok" and r.get("n_calls", 0) > 0]
    print("\n" + "=" * 64)
    print(f"PER-SPLIT (instrument={instrument})")
    for r in sorted(results, key=lambda x: x["split_id"]):
        if r.get("status") == "ok":
            print(f"  split {r['split_id']:2d}: {r['n_calls']:5d} calls  "
                  f"hit {(r['hit_rate'] or 0)*100:5.1f}%   {r['test_start'][:10]}→{r['test_end'][:10]}")
        elif r.get("status") == "error":
            print(f"  split {r['split_id']:2d}: ERROR — {r.get('error')}")
        else:
            print(f"  split {r['split_id']:2d}: {r['status']}")
    n_err = sum(1 for r in results if r.get("status") == "error")
    if n_err:
        print(f"\n  ({n_err} split(s) errored — excluded from pooled stats; see above)")

    total_calls = sum(r["n_calls"] for r in ok)
    total_hits  = sum(r["n_hits"] for r in ok)
    print("=" * 64)
    if total_calls:
        hit = total_hits / total_calls
        se = np.sqrt(0.25 / total_calls)
        z = (hit - 0.5) / se
        print(f"POOLED across {len(ok)} splits @ 6-bar horizon")
        print(f"  Total directional calls: {total_calls}")
        print(f"  Pooled hit ratio: {hit*100:.2f}%")
        print(f"  z-score vs 50%:   {z:+.2f}   (|z|>1.96 = significant at 5%)")
        print(f"  95% CI:           [{(hit-1.96*se)*100:.2f}%, {(hit+1.96*se)*100:.2f}%]")
        verdict = ("EDGE (significant)" if z > 1.96 else
                   "INVERSE edge" if z < -1.96 else
                   "NO detectable edge")
        print(f"  Verdict: {verdict}")
    else:
        print("No directional calls produced.")
    print("=" * 64)
