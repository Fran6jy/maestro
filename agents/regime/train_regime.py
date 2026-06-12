"""
maestro/agents/regime/train_regime.py
=======================================
Walk-forward training harness for Agent 1.

This is the script you run to train the Regime Detection Agent
across all WFA windows and produce a validated performance report.

What it does
------------
1. Loads the feature dataset from Parquet
2. Iterates through all WFA splits (expanding window)
3. On each split: fits RegimeDetectionAgent, predicts on test window
4. Aggregates: regime accuracy, transition stability, confidence calibration
5. Saves per-split models + a summary report

Usage
-----
    python -m maestro.agents.regime.train_regime --instrument EUR_USD
    python -m maestro.agents.regime.train_regime --instrument EUR_USD --granularity H1
    python -m maestro.agents.regime.train_regime --hmm-only   # skip Transformer (fast)
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from maestro.agents.regime.hmm_regime import HMMConfig
from maestro.agents.regime.regime_classifier import RegimeDetectionAgent, REGIME_NAMES
from maestro.agents.regime.transformer_regime import TransformerConfig
from maestro.data.pipeline.ingestion import DataPipeline
from maestro.data.validation.wfa import WalkForwardEngine, compute_metrics

logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.environ.get("MAESTRO_MODEL_DIR", "/tmp/maestro_models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_regime_agent(
    instrument:      str  = "EUR_USD",
    granularity:     str  = "M5",
    hmm_only:        bool = False,
    save_models:     bool = True,
    val_months:      int  = 2,
) -> pd.DataFrame:
    """
    Full walk-forward training + evaluation of Agent 1.

    Parameters
    ----------
    instrument   : e.g. "EUR_USD"
    granularity  : e.g. "M5", "H1", "D"
    hmm_only     : skip Transformer (faster, lower accuracy)
    save_models  : persist each split's model to disk
    val_months   : months held out from each training window for val/early-stopping

    Returns
    -------
    pd.DataFrame — per-split performance metrics
    """
    # ── Load data ──────────────────────────────────────────────────────────────
    logger.info("Loading %s %s features...", instrument, granularity)
    pipeline = DataPipeline()
    df       = pipeline.load(instrument, granularity)
    DataPipeline.quality_report(df, name=f"{instrument}_{granularity}")

    # ── WFA engine ────────────────────────────────────────────────────────────
    wfa     = WalkForwardEngine()
    n_splits= wfa.n_splits(df)
    logger.info("WFA: %d splits to process", n_splits)

    # ── Per-split results ──────────────────────────────────────────────────────
    split_results = []

    for split in wfa.splits(df):
        logger.info("─" * 55)
        logger.info("Split %d/%d: train=%s→%s | test=%s→%s",
                    split.split_id + 1, n_splits,
                    split.train_start.date(), split.train_end.date(),
                    split.test_start.date(), split.test_end.date())

        train_df = df.loc[split.train_idx]
        test_df  = df.loc[split.test_idx]

        # Hold out last val_months of train for validation
        val_cutoff = split.train_end - pd.DateOffset(months=val_months)
        val_mask   = train_df.index >= val_cutoff
        val_df     = train_df[val_mask] if val_mask.sum() > 50 else None
        fit_df     = train_df[~val_mask] if val_df is not None else train_df

        # ── Fit agent ──────────────────────────────────────────────────────────
        agent = RegimeDetectionAgent(
            hmm_config         = HMMConfig(n_states=4, n_iter=200),
            transformer_config = TransformerConfig(
                seq_len=60, d_model=64, n_heads=4,
                n_layers=4, max_epochs=30, patience=6,
            ) if not hmm_only else None,
            use_transformer    = not hmm_only,
        )

        try:
            agent.fit(fit_df, val_df=val_df)
        except Exception as exc:
            logger.error("Split %d fit failed: %s", split.split_id, exc)
            continue

        # ── Predict on test ────────────────────────────────────────────────────
        try:
            result_df = agent.predict_batch(test_df)
        except Exception as exc:
            logger.error("Split %d predict failed: %s", split.split_id, exc)
            continue

        # ── Evaluate ───────────────────────────────────────────────────────────
        metrics = _evaluate_split(result_df, test_df, split.split_id)
        split_results.append(metrics)

        logger.info(
            "Split %d results: confidence=%.3f | certain_pct=%.1f%% | "
            "regime_entropy=%.3f",
            split.split_id,
            metrics["mean_confidence"],
            metrics["pct_certain"] * 100,
            metrics["regime_entropy"],
        )

        # ── Save model ─────────────────────────────────────────────────────────
        if save_models:
            split_dir = MODEL_DIR / instrument / f"split_{split.split_id:03d}"
            agent.save(split_dir)

        # ── Save predictions ───────────────────────────────────────────────────
        pred_path = MODEL_DIR / instrument / f"predictions_split_{split.split_id:03d}.parquet"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_parquet(pred_path)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    if not split_results:
        logger.error("No splits completed successfully.")
        return pd.DataFrame()

    summary = pd.DataFrame(split_results)
    _log_summary(summary)

    # Save summary
    summary_path = MODEL_DIR / instrument / "regime_wfa_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Summary saved → %s", summary_path)

    return summary


def _evaluate_split(result_df: pd.DataFrame, test_df: pd.DataFrame, split_id: int) -> dict:
    """Compute per-split regime quality metrics."""
    regimes = result_df["regime"]
    confs   = result_df["confidence"]

    # Regime entropy (higher = less decisive = worse)
    regime_counts = regimes.value_counts(normalize=True)
    entropy = float(-(regime_counts * np.log2(regime_counts + 1e-10)).sum())

    # Regime stability (how often does regime stay the same bar-to-bar)
    stability = float((regimes == regimes.shift(1)).mean())

    # Average duration of each regime run (consecutive same-regime bars)
    runs        = (regimes != regimes.shift(1)).cumsum()
    run_lengths = regimes.groupby(runs).count()
    avg_duration= float(run_lengths.mean())

    # Transition matrix sparsity (are transitions realistic?)
    trans = result_df["regime_name"].value_counts().to_dict()

    return {
        "split_id":        split_id,
        "test_start":      result_df.index[0],
        "test_end":        result_df.index[-1],
        "n_bars":          len(result_df),
        "mean_confidence": float(confs.mean()),
        "pct_certain":     float((confs >= 0.55).mean()),
        "regime_entropy":  entropy,
        "regime_stability":stability,
        "avg_regime_dur":  avg_duration,
        **{f"pct_{REGIME_NAMES[i]}": float((regimes == i).mean()) for i in range(4)},
    }


def _log_summary(summary: pd.DataFrame) -> None:
    logger.info("=" * 55)
    logger.info("AGENT 1 — WFA Summary (%d splits)", len(summary))
    logger.info("  Mean confidence:  %.3f ± %.3f",
                summary["mean_confidence"].mean(), summary["mean_confidence"].std())
    logger.info("  Mean certainty %%: %.1f%%",
                summary["pct_certain"].mean() * 100)
    logger.info("  Mean stability:   %.3f", summary["regime_stability"].mean())
    logger.info("  Mean entropy:     %.3f", summary["regime_entropy"].mean())
    for i in range(4):
        col = f"pct_{REGIME_NAMES[i]}"
        if col in summary.columns:
            logger.info("  %s: %.1f%%", REGIME_NAMES[i], summary[col].mean() * 100)
    logger.info("=" * 55)


def main() -> None:
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Train MAESTRO Agent 1: Regime Detection")
    parser.add_argument("--instrument",  default="EUR_USD")
    parser.add_argument("--granularity", default="M5")
    parser.add_argument("--hmm-only",    action="store_true")
    parser.add_argument("--no-save",     action="store_true")
    args = parser.parse_args()

    train_regime_agent(
        instrument  = args.instrument,
        granularity = args.granularity,
        hmm_only    = args.hmm_only,
        save_models = not args.no_save,
    )


if __name__ == "__main__":
    main()
