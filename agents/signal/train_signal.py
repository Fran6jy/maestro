"""
maestro/agents/signal/train_signal.py
=======================================
Walk-forward training harness for Agent 2 — Technical Signal Agent.

Tracks hit ratio across all WFA splits — the primary metric we use
to prove we've beaten the MSc's 37% baseline.

Target: hit ratio > 55% on out-of-sample test windows.

Usage
-----
    python -m maestro.agents.signal.train_signal --instrument EUR_USD
    python -m maestro.agents.signal.train_signal --instrument GBP_USD --horizon 6
    python -m maestro.agents.signal.train_signal --tft-only   # skip PatchTST
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from maestro.agents.regime.regime_classifier import RegimeDetectionAgent
from maestro.agents.signal.patchtst import PatchTSTSignalModel, HORIZONS
from maestro.agents.signal.signal_agent import SignalAgent
from maestro.agents.signal.tft_model import TFTConfig
from maestro.agents.signal.patchtst import PatchTSTConfig
from maestro.data.pipeline.ingestion import DataPipeline
from maestro.data.validation.wfa import WalkForwardEngine, compute_metrics, deflated_sharpe_ratio

logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.environ.get("MAESTRO_MODEL_DIR", "/tmp/maestro_models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_signal_agent(
    instrument:      str  = "EUR_USD",
    granularity:     str  = "M5",
    primary_horizon: int  = 6,
    val_months:      int  = 2,
    save_models:     bool = True,
) -> pd.DataFrame:
    """
    Full walk-forward training + evaluation of Agent 2.

    Returns pd.DataFrame with per-split performance metrics including
    the critical hit ratio comparison against the 37% MSc baseline.
    """
    logger.info("=" * 60)
    logger.info("MAESTRO Agent 2: Technical Signal — WFA Training")
    logger.info("Instrument: %s | Granularity: %s | Horizon: %d bars",
                instrument, granularity, primary_horizon)
    logger.info("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────────
    pipeline = DataPipeline()
    df       = pipeline.load(instrument, granularity)
    logger.info("Loaded: %d bars × %d features", len(df), len(df.columns))

    # ── WFA engine ────────────────────────────────────────────────────────────
    wfa      = WalkForwardEngine()
    n_splits = wfa.n_splits(df)
    logger.info("WFA: %d splits", n_splits)

    split_results = []

    for split in wfa.splits(df):
        logger.info("─" * 60)
        logger.info("Split %d/%d | train=%s→%s | test=%s→%s",
                    split.split_id + 1, n_splits,
                    split.train_start.date(), split.train_end.date(),
                    split.test_start.date(), split.test_end.date())

        train_df = df.loc[split.train_idx]
        test_df  = df.loc[split.test_idx]

        # Validation slice (last val_months of training window)
        val_cut  = split.train_end - pd.DateOffset(months=val_months)
        val_mask = train_df.index >= val_cut
        val_df   = train_df[val_mask] if val_mask.sum() > 200 else None
        fit_df   = train_df[~val_mask] if val_df is not None else train_df

        # ── Step 1: Fit Regime Agent to get regime labels ─────────────────────
        logger.info("  Fitting Regime Agent for regime labels...")
        regime_agent = RegimeDetectionAgent(use_transformer=False)  # HMM-only for speed
        try:
            regime_agent.fit(fit_df)
            train_regimes = regime_agent.predict(fit_df)
            val_regimes   = regime_agent.predict(val_df) if val_df is not None else None
            test_regimes  = regime_agent.predict(test_df)
        except Exception as exc:
            logger.error("  Regime agent failed: %s — using flat regime", exc)
            train_regimes = pd.Series(2, index=fit_df.index)
            val_regimes   = pd.Series(2, index=val_df.index) if val_df is not None else None
            test_regimes  = pd.Series(2, index=test_df.index)

        # ── Step 2: Fit Signal Agent ──────────────────────────────────────────
        logger.info("  Fitting Signal Agent...")
        agent = SignalAgent(
            instrument      = instrument,
            primary_horizon = primary_horizon,
            tft_config      = TFTConfig(
                seq_len=120, pred_len=max(HORIZONS),
                max_epochs=40, patience=8, batch_size=64,
            ),
            ptst_config     = PatchTSTConfig(
                seq_len=128, patch_size=16, stride=8,
                max_epochs=40, patience=8,
            ),
        )
        try:
            agent.fit(
                fit_df, train_regimes,
                val_df=val_df, val_regimes=val_regimes
            )
        except Exception as exc:
            logger.error("  Signal agent fit failed: %s", exc)
            continue

        # ── Step 3: Predict + evaluate ────────────────────────────────────────
        try:
            result_df = agent.predict_batch(test_df, test_regimes)
        except Exception as exc:
            logger.error("  Signal agent predict failed: %s", exc)
            continue

        # Compute ground-truth hit ratios
        metrics = _evaluate_signals(result_df, test_df, primary_horizon, split.split_id)
        split_results.append(metrics)

        logger.info(
            "  HIT RATIO: %.1f%% (target >55%%) | "
            "actionable: %.1f%% | model_agree: %.1f%%",
            metrics["hit_ratio"] * 100,
            metrics["pct_actionable"] * 100,
            metrics["pct_model_agree"] * 100,
        )

        # ── Save ──────────────────────────────────────────────────────────────
        if save_models:
            split_dir = MODEL_DIR / instrument / f"agent2_split_{split.split_id:03d}"
            agent.save(split_dir)

        pred_path = MODEL_DIR / instrument / f"agent2_preds_split_{split.split_id:03d}.parquet"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_parquet(pred_path)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    if not split_results:
        logger.error("No splits completed.")
        return pd.DataFrame()

    summary = pd.DataFrame(split_results)
    _log_summary(summary, instrument, primary_horizon)

    summary_path = MODEL_DIR / instrument / "agent2_wfa_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Summary saved → %s", summary_path)
    return summary


def _evaluate_signals(
    result_df:  pd.DataFrame,
    features_df: pd.DataFrame,
    horizon:     int,
    split_id:   int,
) -> dict:
    """
    Compute hit ratio and related metrics for one WFA split.

    Hit ratio = fraction of non-flat, actionable signals that correctly
    predicted the future direction. This is the number we compare
    directly to the MSc's 37% baseline.
    """
    # Ground truth: actual future direction at horizon h
    close       = features_df["close"].reindex(result_df.index)
    future_ret  = np.log(
        features_df["close"].shift(-horizon).reindex(result_df.index) / close
    )
    threshold   = 0.0002
    true_dir    = np.where(future_ret >  threshold,  1,
                  np.where(future_ret < -threshold, -1, 0))

    signals     = result_df["signal"].values if "signal" in result_df.columns else np.zeros(len(result_df))
    actionable  = result_df["is_actionable"].values if "is_actionable" in result_df.columns else signals != 0

    # Hit ratio: only on actionable (non-flat) bars
    act_mask    = actionable & (signals != 0)
    n_active    = act_mask.sum()
    hit_ratio   = float((signals[act_mask] == true_dir[act_mask]).mean()) if n_active > 0 else 0.0

    # Regime-stratified hit ratios
    regime_hits = {}
    if "regime" in result_df.columns:
        from maestro.agents.regime.regime_classifier import REGIME_NAMES
        for r in range(4):
            r_mask = (result_df["regime"].values == r) & act_mask
            if r_mask.sum() >= 10:
                regime_hits[f"hit_ratio_{REGIME_NAMES[r]}"] = float(
                    (signals[r_mask] == true_dir[r_mask]).mean()
                )

    # Strategy returns (signal × next-bar return, no costs yet)
    strategy_ret = pd.Series(
        signals * future_ret.fillna(0).values,
        index = result_df.index
    )
    strategy_ret = strategy_ret[act_mask]

    # Performance metrics
    if len(strategy_ret) > 0 and strategy_ret.std() > 0:
        sharpe   = float(strategy_ret.mean() / strategy_ret.std() * np.sqrt(252 * 78))
        cum_ret  = (1 + strategy_ret).prod() - 1
        cum_prod = (1 + strategy_ret).cumprod()
        peak     = cum_prod.cummax()
        max_dd   = float(((cum_prod - peak) / peak).min())
    else:
        sharpe = cum_ret = max_dd = 0.0

    return {
        "split_id":          split_id,
        "test_start":        result_df.index[0],
        "test_end":          result_df.index[-1],
        "n_bars":            len(result_df),
        "n_active":          int(n_active),
        "hit_ratio":         hit_ratio,
        "pct_actionable":    float(act_mask.mean()),
        "pct_model_agree":   float(result_df["model_agree"].mean()) if "model_agree" in result_df.columns else 0.0,
        "sharpe_gross":      sharpe,
        "cum_return_gross":  cum_ret,
        "max_drawdown":      max_dd,
        "msc_baseline":      0.3756,    # MSc linear regression hit ratio
        "vs_baseline":       hit_ratio - 0.3756,
        **regime_hits,
    }


def _log_summary(summary: pd.DataFrame, instrument: str, horizon: int) -> None:
    hit_mean = summary["hit_ratio"].mean()
    hit_std  = summary["hit_ratio"].std()
    baseline = 0.3756

    logger.info("=" * 60)
    logger.info("AGENT 2 WFA SUMMARY — %s h=%d (%d splits)", instrument, horizon, len(summary))
    logger.info("")
    logger.info("  Hit Ratio:    %.1f%% ± %.1f%%", hit_mean * 100, hit_std * 100)
    logger.info("  MSc Baseline: %.1f%%", baseline * 100)
    logger.info("  Improvement:  %+.1f pp", (hit_mean - baseline) * 100)
    logger.info("  Target (>55%%): %s", "✓ ACHIEVED" if hit_mean > 0.55 else "✗ NOT YET")
    logger.info("")
    logger.info("  Sharpe (gross): %.3f ± %.3f",
                summary["sharpe_gross"].mean(), summary["sharpe_gross"].std())
    logger.info("  Max Drawdown:   %.1f%%", summary["max_drawdown"].mean() * 100)
    logger.info("  Actionable %%:   %.1f%%", summary["pct_actionable"].mean() * 100)
    logger.info("  Model Agreement: %.1f%%", summary["pct_model_agree"].mean() * 100)

    # Per-regime hit ratios
    for r_name in ["bull_trend", "bear_trend", "sideways", "crisis"]:
        col = f"hit_ratio_{r_name}"
        if col in summary.columns:
            logger.info("  %s hit ratio: %.1f%%", r_name, summary[col].mean() * 100)

    logger.info("=" * 60)


def main() -> None:
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Train MAESTRO Agent 2: Signal")
    parser.add_argument("--instrument",  default="EUR_USD")
    parser.add_argument("--granularity", default="M5")
    parser.add_argument("--horizon",     type=int, default=6)
    parser.add_argument("--no-save",     action="store_true")
    args = parser.parse_args()

    train_signal_agent(
        instrument      = args.instrument,
        granularity     = args.granularity,
        primary_horizon = args.horizon,
        save_models     = not args.no_save,
    )


if __name__ == "__main__":
    main()
