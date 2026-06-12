"""
maestro/agents/sentiment/train_sentiment.py
=============================================
Walk-forward evaluation harness for Agent 3 — LLM Sentiment Agent.

Unlike Agents 1 and 2, Agent 3 has no "training" in the traditional
sense — FinBERT and GPT-4o come pretrained. What we evaluate here is:

  1. NLP directional accuracy vs realised price moves (per WFA split)
  2. Fusion weight calibration — is the rolling accuracy signal useful?
  3. Incremental contribution — does Agent 3 improve overall hit ratio
     when combined with Agent 2's signals?
  4. Regime-conditioned accuracy — is NLP signal more valuable in
     crisis vs trending regimes? (PhD contribution hypothesis)

This is the evidence for the paper:
  "LLM-augmented macro signals: do they improve Forex prediction?"

Usage
-----
    python -m maestro.agents.sentiment.train_sentiment \
        --instrument EUR_USD \
        --news-path /tmp/maestro_data/news.parquet
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from maestro.agents.regime.regime_classifier import RegimeDetectionAgent, REGIME_NAMES
from maestro.agents.sentiment.finbert_agent import FinBERTScorer
from maestro.agents.sentiment.fusion import SentimentFusionAgent
from maestro.agents.signal.patchtst import PatchTSTSignalModel
from maestro.data.pipeline.ingestion import DataPipeline
from maestro.data.validation.wfa import WalkForwardEngine

logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.environ.get("MAESTRO_MODEL_DIR", "/tmp/maestro_models"))
DATA_DIR  = Path(os.environ.get("MAESTRO_DATA_DIR", "/tmp/maestro_data"))


def evaluate_sentiment_agent(
    instrument:  str  = "EUR_USD",
    granularity: str  = "M5",
    news_path:   str | None = None,
    finbert_device: str = "cpu",
) -> pd.DataFrame:
    """
    Walk-forward evaluation of Agent 3.

    Measures NLP directional accuracy per split and regime,
    and quantifies the incremental contribution to overall signal quality.

    Returns
    -------
    pd.DataFrame — per-split evaluation metrics
    """
    logger.info("=" * 60)
    logger.info("MAESTRO Agent 3: Sentiment — WFA Evaluation")
    logger.info("Instrument: %s | Granularity: %s", instrument, granularity)
    logger.info("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────────
    pipeline   = DataPipeline()
    features_df = pipeline.load(instrument, granularity)

    # Load pre-scored news (or use empty DataFrame if not available)
    news_df = _load_news(news_path)
    has_news = not news_df.empty
    if not has_news:
        logger.warning(
            "No news data found. Running in price-only mode.\n"
            "Run DataPipeline().run_full() to fetch and cache news."
        )

    # ── WFA engine ────────────────────────────────────────────────────────────
    wfa      = WalkForwardEngine()
    n_splits = wfa.n_splits(features_df)
    logger.info("WFA: %d splits", n_splits)

    split_results = []

    for split in wfa.splits(features_df):
        logger.info("─" * 60)
        logger.info("Split %d/%d | %s → %s",
                    split.split_id + 1, n_splits,
                    split.test_start.date(), split.test_end.date())

        test_df = features_df.loc[split.test_idx]

        # ── Regime labels ──────────────────────────────────────────────────────
        train_df    = features_df.loc[split.train_idx]
        regime_agent= RegimeDetectionAgent(use_transformer=False)
        try:
            regime_agent.fit(train_df)
            regime_signals = regime_agent.predict_batch(test_df)
        except Exception as exc:
            logger.warning("Regime agent failed: %s", exc)
            regime_signals = pd.DataFrame({"regime": 2}, index=test_df.index)

        # ── Filter news to test window ─────────────────────────────────────────
        test_news = pd.DataFrame()
        if has_news and "published_at" in news_df.columns:
            mask = (
                (news_df["published_at"] >= split.test_start - pd.Timedelta(hours=24)) &
                (news_df["published_at"] <= split.test_end)
            )
            test_news = news_df[mask].copy()

        # ── Initialise and run Agent 3 ─────────────────────────────────────────
        agent = SentimentFusionAgent(instrument=instrument, finbert_device=finbert_device)

        # Score news (if any)
        if not test_news.empty:
            try:
                agent.load_models()
                test_news = agent.score_news(test_news)
            except Exception as exc:
                logger.warning("FinBERT scoring failed: %s", exc)
                test_news = pd.DataFrame()

        # Generate signals
        try:
            sentiment_signals = agent.generate_signals(
                test_df, test_news, regime_signals
            )
        except Exception as exc:
            logger.error("Signal generation failed: %s", exc)
            continue

        # ── Evaluate ───────────────────────────────────────────────────────────
        metrics = _evaluate_split(
            sentiment_signals, test_df, regime_signals, split.split_id
        )
        split_results.append(metrics)
        logger.info(
            "  NLP accuracy: %.1f%% | crisis NLP acc: %.1f%% | "
            "articles/bar: %.2f | fusion_weight: %.3f",
            metrics["nlp_accuracy"] * 100,
            metrics.get("nlp_accuracy_crisis", 0.0) * 100,
            metrics["articles_per_bar"],
            metrics["avg_fusion_weight"],
        )

    # ── Aggregate ─────────────────────────────────────────────────────────────
    if not split_results:
        logger.error("No splits completed.")
        return pd.DataFrame()

    summary = pd.DataFrame(split_results)
    _log_summary(summary, instrument)

    out_path = MODEL_DIR / instrument / "agent3_wfa_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    logger.info("Summary saved → %s", out_path)
    return summary


def _evaluate_split(
    sentiment_df:  pd.DataFrame,
    features_df:   pd.DataFrame,
    regime_signals:pd.DataFrame,
    split_id:      int,
) -> dict:
    """Compute NLP accuracy metrics for one WFA split."""

    future_ret  = features_df["log_return_1"].shift(-1).reindex(sentiment_df.index)
    threshold   = 0.0002
    true_dir    = np.where(future_ret >  threshold,  1,
                  np.where(future_ret < -threshold, -1, 0))

    nlp_sigs = sentiment_df["sentiment_signal"].values if "sentiment_signal" in sentiment_df.columns else np.zeros(len(sentiment_df))
    active   = nlp_sigs != 0

    # Overall NLP accuracy
    nlp_accuracy = float((nlp_sigs[active] == true_dir[active]).mean()) if active.sum() > 0 else 0.5

    # Per-regime accuracy
    regime_acc = {}
    if "regime" in sentiment_df.columns:
        for r in range(4):
            r_mask = (sentiment_df["regime"].values == r) & active
            if r_mask.sum() >= 5:
                regime_acc[f"nlp_accuracy_{REGIME_NAMES[r]}"] = float(
                    (nlp_sigs[r_mask] == true_dir[r_mask]).mean()
                )

    # Incremental value: accuracy boost when text agrees with price
    price_agree_acc  = 0.0
    price_disagree_acc = 0.0
    if "price_text_agree" in sentiment_df.columns:
        agree_mask    = sentiment_df["price_text_agree"].values & active
        disagree_mask = (~sentiment_df["price_text_agree"].values) & active
        if agree_mask.sum() > 5:
            price_agree_acc = float((nlp_sigs[agree_mask] == true_dir[agree_mask]).mean())
        if disagree_mask.sum() > 5:
            price_disagree_acc = float((nlp_sigs[disagree_mask] == true_dir[disagree_mask]).mean())

    return {
        "split_id":             split_id,
        "test_start":           sentiment_df.index[0],
        "test_end":             sentiment_df.index[-1],
        "n_bars":               len(sentiment_df),
        "n_active_signals":     int(active.sum()),
        "pct_active":           float(active.mean()),
        "nlp_accuracy":         nlp_accuracy,
        "articles_per_bar":     float(sentiment_df["article_count"].mean()) if "article_count" in sentiment_df.columns else 0.0,
        "avg_fusion_weight":    float(sentiment_df["fusion_weight"].mean()) if "fusion_weight" in sentiment_df.columns else 0.0,
        "pct_price_text_agree": float(sentiment_df["price_text_agree"].mean()) if "price_text_agree" in sentiment_df.columns else 0.0,
        "price_agree_acc":      price_agree_acc,
        "price_disagree_acc":   price_disagree_acc,
        "agree_lift":           price_agree_acc - nlp_accuracy,
        **regime_acc,
    }


def _log_summary(summary: pd.DataFrame, instrument: str) -> None:
    logger.info("=" * 60)
    logger.info("AGENT 3 WFA SUMMARY — %s (%d splits)", instrument, len(summary))
    logger.info("")
    logger.info("  NLP Accuracy:    %.1f%% ± %.1f%%",
                summary["nlp_accuracy"].mean() * 100,
                summary["nlp_accuracy"].std() * 100)
    logger.info("  Avg Fusion Weight: %.3f", summary["avg_fusion_weight"].mean())
    logger.info("  Articles/bar:      %.2f", summary["articles_per_bar"].mean())
    logger.info("  Price+Text agree accuracy: %.1f%%",
                summary["price_agree_acc"].mean() * 100)
    logger.info("  Price+Text DISagree accuracy: %.1f%%",
                summary["price_disagree_acc"].mean() * 100)
    logger.info("  Agreement lift:    %+.1f pp",
                summary["agree_lift"].mean() * 100)

    # Crisis vs non-crisis
    crisis_col = "nlp_accuracy_crisis"
    trend_col  = "nlp_accuracy_bull_trend"
    if crisis_col in summary.columns:
        logger.info("  NLP accuracy in crisis:  %.1f%%",
                    summary[crisis_col].mean() * 100)
    if trend_col in summary.columns:
        logger.info("  NLP accuracy in bull:    %.1f%%",
                    summary[trend_col].mean() * 100)

    logger.info("")
    logger.info("  PhD Hypothesis: NLP signal is most valuable in crisis regime?")
    if crisis_col in summary.columns and trend_col in summary.columns:
        crisis_acc = summary[crisis_col].mean()
        trend_acc  = summary[trend_col].mean()
        supported  = crisis_acc > trend_acc
        logger.info("  → Crisis NLP acc (%.1f%%) > Trend NLP acc (%.1f%%): %s",
                    crisis_acc * 100, trend_acc * 100,
                    "SUPPORTED ✓" if supported else "NOT SUPPORTED ✗")
    logger.info("=" * 60)


def _load_news(news_path: str | None) -> pd.DataFrame:
    """Load news from Parquet if available, else return empty."""
    paths_to_try = [
        news_path,
        str(DATA_DIR / "news.parquet"),
        str(DATA_DIR / "news_features.parquet"),
    ]
    for p in paths_to_try:
        if p and Path(p).exists():
            df = pd.read_parquet(p)
            logger.info("Loaded news: %d articles from %s", len(df), p)
            return df
    return pd.DataFrame()


def main() -> None:
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Evaluate MAESTRO Agent 3: Sentiment")
    parser.add_argument("--instrument",   default="EUR_USD")
    parser.add_argument("--granularity",  default="M5")
    parser.add_argument("--news-path",    default=None)
    parser.add_argument("--finbert-device", default="cpu")
    args = parser.parse_args()

    evaluate_sentiment_agent(
        instrument     = args.instrument,
        granularity    = args.granularity,
        news_path      = args.news_path,
        finbert_device = args.finbert_device,
    )


if __name__ == "__main__":
    main()
