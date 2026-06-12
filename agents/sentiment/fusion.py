"""
maestro/agents/sentiment/fusion.py
=====================================
Adaptive Sentiment Fusion — the core of Agent 3.

What "adaptive" means here
---------------------------
The key innovation over the MSc's static NewsAPI approach is that
the fusion weight between text-based signals (FinBERT + GPT-4o) and
price-based signals (Agent 2) adapts in real time based on:

  1. Rolling NLP accuracy   — how well has sentiment predicted price
                              direction over the last N bars?
  2. Regime state           — LLM signal gets more weight in "crisis"
                              regime, where price patterns break down
  3. Sentiment intensity    — high-intensity articles (very positive or
                              very negative) get higher weight
  4. Article freshness      — recent articles weighted more heavily
  5. Model agreement        — when FinBERT + GPT-4o agree, boost weight

Fusion architecture
--------------------
  FinBERT bar signal    ─┐
  GPT-4o macro signal   ─┤→ sentiment_composite → adaptive_weight(t)
  Article count          ─┘                              ↓
                                                  SentimentPacket
  Regime (Agent 1)   ────────────────────────────────────↑
  Signal confidence (Agent 2) ────────────────────────────↑

The SentimentPacket tells downstream agents:
  - What the text says about direction
  - How confident to be in that text signal
  - Whether to override price signal (e.g., unexpected FOMC)

This directly addresses the MSc's failure mode: sentiment was
computed but never properly integrated into the trading decision.

Output SentimentPacket fields
------------------------------
  sentiment_signal:      {-1, 0, +1}
  text_confidence:       [0,1] — confidence in the text signal
  price_text_agree:      bool  — does text agree with technical signal?
  fusion_weight:         [0,1] — how much to weight text vs price
  finbert_score:         raw FinBERT composite
  gpt4o_eur_usd_bias:    raw GPT-4o EUR/USD call
  gpt4o_gbp_usd_bias:    raw GPT-4o GBP/USD call
  rolling_nlp_accuracy:  [0,1] — how well NLP has been predicting lately
  regime_boost:          bool  — fusion_weight boosted by crisis regime
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from maestro.agents.sentiment.finbert_agent import FinBERTScorer
from maestro.agents.sentiment.gpt4o_agent import GPT4oMacroAnalyst
from maestro.agents.regime.regime_classifier import RegimeSignal, REGIME_NAMES

logger = logging.getLogger(__name__)

# Regime-conditioned base fusion weights (text_weight)
# How much to weight NLP signal vs price signal per regime
REGIME_TEXT_WEIGHTS = {
    0: 0.20,   # bull_trend:  price patterns clear → low text weight
    1: 0.25,   # bear_trend:  slightly elevated text weight
    2: 0.20,   # sideways:    range-bound → text can clarify breakout direction
    3: 0.50,   # crisis:      price patterns break down → text crucial
}

# Rolling accuracy window for adaptive weight adjustment
ACCURACY_WINDOW = 20   # bars


@dataclass
class SentimentPacket:
    """
    Output of Agent 3. Consumed by Meta-Orchestrator and Risk Agent.

    Fields
    ------
    timestamp:            bar datetime (UTC)
    instrument:           e.g. "EUR_USD"
    sentiment_signal:     {-1=bearish, 0=neutral, +1=bullish}
    text_confidence:      calibrated confidence in NLP signal
    fusion_weight:        how much this NLP signal should weight vs price
    price_text_agree:     True if NLP agrees with Agent 2 signal direction
    finbert_score:        raw FinBERT composite score [-1,+1]
    gpt4o_bias:           GPT-4o instrument-specific bias [-1,+1]
    gpt4o_confidence:     GPT-4o self-assessed confidence
    gpt4o_surprise:       GPT-4o surprise factor [-1,+1]
    rolling_nlp_accuracy: [0,1] — recent NLP directional accuracy
    article_count:        number of articles in window
    regime:               regime ID at this bar
    regime_boost:         True if crisis regime boosted fusion weight
    """
    timestamp:            pd.Timestamp
    instrument:           str
    sentiment_signal:     int
    text_confidence:      float
    fusion_weight:        float
    price_text_agree:     bool
    finbert_score:        float
    gpt4o_bias:           float
    gpt4o_confidence:     float
    gpt4o_surprise:       float
    rolling_nlp_accuracy: float
    article_count:        int
    regime:               int
    regime_boost:         bool
    rationale:            str = ""

    def __repr__(self) -> str:
        direction = {1: "BULLISH", -1: "BEARISH", 0: "NEUTRAL"}.get(
            self.sentiment_signal, "?"
        )
        return (
            f"SentimentPacket({self.timestamp.strftime('%Y-%m-%d %H:%M')} | "
            f"{self.instrument} | {direction} | "
            f"fw={self.fusion_weight:.2f} | conf={self.text_confidence:.2f} | "
            f"nlp_acc={self.rolling_nlp_accuracy:.2f})"
        )

    def to_dict(self) -> dict:
        return {
            "timestamp":             self.timestamp,
            "instrument":            self.instrument,
            "sentiment_signal":      self.sentiment_signal,
            "text_confidence":       self.text_confidence,
            "fusion_weight":         self.fusion_weight,
            "price_text_agree":      self.price_text_agree,
            "finbert_score":         self.finbert_score,
            "gpt4o_bias":            self.gpt4o_bias,
            "gpt4o_confidence":      self.gpt4o_confidence,
            "gpt4o_surprise":        self.gpt4o_surprise,
            "rolling_nlp_accuracy":  self.rolling_nlp_accuracy,
            "article_count":         self.article_count,
            "regime":                self.regime,
            "regime_boost":          self.regime_boost,
            "rationale":             self.rationale,
        }


class SentimentFusionAgent:
    """
    Agent 3 — LLM Sentiment & Macro Agent.

    Combines FinBERT + GPT-4o signals with adaptive weighting
    that learns from rolling NLP accuracy vs realised returns.

    Usage
    -----
    >>> agent = SentimentFusionAgent(instrument="EUR_USD")
    >>> agent.load_models()
    >>> # One-time: score news articles
    >>> scored_news = agent.score_news(news_df)
    >>> # Per bar: generate SentimentPackets
    >>> packets = agent.generate_signals(
    ...     ohlcv_df, scored_news, regime_signals,
    ...     price_signals=signal_agent_output
    ... )
    """

    def __init__(
        self,
        instrument:      str  = "EUR_USD",
        finbert_device:  str  = "cpu",
        max_gpt4o_daily: int  = 50,
    ) -> None:
        self.instrument     = instrument
        self.finbert        = FinBERTScorer(device=finbert_device)
        self.gpt4o          = GPT4oMacroAnalyst(max_daily_calls=max_gpt4o_daily)
        self._models_loaded = False

        # Adaptive state — updated each bar
        self._nlp_history:   list[tuple[int, int]] = []  # (nlp_signal, true_direction)
        self._weight_history: list[float]           = []

    # ── Load models ───────────────────────────────────────────────────────────
    def load_models(self) -> "SentimentFusionAgent":
        """Load FinBERT from HuggingFace (downloads once, ~440MB)."""
        self.finbert.load()
        self._models_loaded = True
        logger.info("Agent 3: FinBERT loaded | GPT-4o ready (API key required)")
        return self

    # ── Score news articles ───────────────────────────────────────────────────
    def score_news(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run both FinBERT and GPT-4o over a news DataFrame.

        Parameters
        ----------
        news_df : output of NewsConnector.fetch_recent()

        Returns
        -------
        news_df with additional columns:
          FinBERT: sentiment, pos_prob, neg_prob, sentiment_score, intensity
          GPT-4o:  gpt4o_eur_usd_bias, gpt4o_gbp_usd_bias, gpt4o_confidence,
                   gpt4o_surprise_factor, gpt4o_rationale, gpt4o_horizon
        """
        if not self._models_loaded:
            self.load_models()

        logger.info("Agent 3: Scoring %d news articles...", len(news_df))

        # FinBERT scoring
        scored = self.finbert.score_dataframe(news_df)

        # GPT-4o macro analysis (high-impact events only)
        scored = self.gpt4o.analyse_batch(scored)

        return scored

    # ── Generate bar-level signals ────────────────────────────────────────────
    def generate_signals(
        self,
        ohlcv_df:       pd.DataFrame,
        scored_news:    pd.DataFrame,
        regime_signals: pd.DataFrame,
        price_signals:  pd.DataFrame | None = None,
        window_hours:   float = 4.0,
    ) -> pd.DataFrame:
        """
        Generate SentimentPackets aligned to OHLCV bars.

        Parameters
        ----------
        ohlcv_df        : OHLCV DataFrame with DatetimeIndex
        scored_news     : output of score_news()
        regime_signals  : output of RegimeDetectionAgent.predict_batch()
        price_signals   : output of SignalAgent.predict_batch() — used to
                          check text/price agreement and compute NLP accuracy
        window_hours    : lookback window for news aggregation

        Returns
        -------
        pd.DataFrame — one row per bar, all SentimentPacket fields as columns
        """
        bar_index = ohlcv_df.index

        # ── Step 1: Aggregate FinBERT to bars ─────────────────────────────────
        finbert_bars = self.finbert.aggregate_to_bars(
            scored_news, bar_index, window_hours=window_hours
        )

        # ── Step 2: Aggregate GPT-4o to bars ──────────────────────────────────
        gpt4o_bars = self.gpt4o.aggregate_to_bars(
            scored_news, bar_index,
            instrument=self.instrument, window_hours=window_hours
        )

        # ── Step 3: Fuse into composite sentiment per bar ─────────────────────
        packets = []
        true_returns = ohlcv_df["log_return_1"].shift(-1).fillna(0)   # for rolling accuracy

        for i, ts in enumerate(bar_index):
            # Regime at this bar
            regime = 2   # default sideways
            if ts in regime_signals.index and "regime" in regime_signals.columns:
                regime = int(regime_signals.loc[ts, "regime"])

            # FinBERT signal
            fb_score     = float(finbert_bars.loc[ts, "finbert_score"])      if ts in finbert_bars.index else 0.0
            fb_intensity = float(finbert_bars.loc[ts, "finbert_intensity"])  if ts in finbert_bars.index else 0.0
            fb_count     = int(finbert_bars.loc[ts, "finbert_article_count"])if ts in finbert_bars.index else 0
            fb_signal    = int(finbert_bars.loc[ts, "finbert_signal"])       if ts in finbert_bars.index else 0

            # GPT-4o signal
            bias_col   = "gpt4o_macro_signal"
            gpt4o_sig  = int(gpt4o_bars.loc[ts, bias_col])              if ts in gpt4o_bars.index else 0
            gpt4o_conf = float(gpt4o_bars.loc[ts, "gpt4o_confidence"])  if ts in gpt4o_bars.index else 0.0
            gpt4o_surp = float(gpt4o_bars.loc[ts, "gpt4o_surprise"])    if ts in gpt4o_bars.index else 0.0

            # Raw GPT-4o bias for the specific instrument
            instrument_bias_col = (
                "gpt4o_eur_usd_bias" if self.instrument == "EUR_USD" else "gpt4o_gbp_usd_bias"
            )
            gpt4o_bias = 0.0
            if ts in scored_news.index and instrument_bias_col in scored_news.columns:
                window_mask = (
                    (scored_news["published_at"] > ts - pd.Timedelta(hours=window_hours)) &
                    (scored_news["published_at"] <= ts)
                ) if "published_at" in scored_news.columns else pd.Series(False, index=scored_news.index)
                if window_mask.any():
                    gpt4o_bias = float(scored_news.loc[window_mask, instrument_bias_col].mean())

            # ── Composite sentiment signal ─────────────────────────────────────
            # Weighted combination: FinBERT (0.4) + GPT-4o (0.6)
            # GPT-4o gets higher weight as it has richer contextual reasoning
            finbert_weight = 0.4
            gpt4o_weight   = 0.6

            # Boost GPT-4o weight on high-surprise events
            if abs(gpt4o_surp) > 0.4 and gpt4o_conf > 0.5:
                gpt4o_weight   = 0.75
                finbert_weight = 0.25

            # When no GPT-4o signal, fall back fully to FinBERT
            if gpt4o_conf < 0.15:
                finbert_weight = 1.0
                gpt4o_weight   = 0.0

            raw_composite = finbert_weight * fb_score + gpt4o_weight * gpt4o_bias
            composite_sig = int(np.sign(raw_composite)) if abs(raw_composite) > 0.10 else 0

            # Text confidence: higher when both models agree + high intensity
            models_agree   = (fb_signal == gpt4o_sig) or (fb_count == 0)
            text_conf_base = max(fb_intensity, gpt4o_conf)
            text_conf      = text_conf_base * (1.15 if models_agree else 0.85)
            text_conf      = min(float(text_conf), 1.0)

            # ── Adaptive fusion weight ─────────────────────────────────────────
            base_weight   = REGIME_TEXT_WEIGHTS.get(regime, 0.20)
            regime_boost  = regime == 3   # crisis

            # Adjust by rolling NLP accuracy
            rolling_acc   = self._compute_rolling_accuracy()
            if rolling_acc > 0.60:
                # NLP has been accurate recently → increase its weight
                weight_adj = min(base_weight * 1.3, 0.6)
            elif rolling_acc < 0.45:
                # NLP has been unreliable → decrease its weight
                weight_adj = max(base_weight * 0.7, 0.05)
            else:
                weight_adj = base_weight

            # Cap fusion weight in non-crisis regimes
            fusion_weight = min(weight_adj, 0.6 if regime == 3 else 0.35)

            # ── Price/text agreement ───────────────────────────────────────────
            price_text_agree = False
            if price_signals is not None and ts in price_signals.index and "signal" in price_signals.columns:
                price_sig        = int(price_signals.loc[ts, "signal"])
                price_text_agree = (composite_sig == price_sig) and (composite_sig != 0)

            # ── Update NLP history (for next bar's rolling accuracy) ───────────
            if i > 0 and composite_sig != 0:
                actual_dir = int(np.sign(true_returns.iloc[i - 1])) if abs(true_returns.iloc[i-1]) > 1e-5 else 0
                if actual_dir != 0:
                    self._nlp_history.append((composite_sig, actual_dir))
                    if len(self._nlp_history) > ACCURACY_WINDOW * 3:
                        self._nlp_history = self._nlp_history[-ACCURACY_WINDOW * 3:]

            packets.append(SentimentPacket(
                timestamp            = ts,
                instrument           = self.instrument,
                sentiment_signal     = composite_sig,
                text_confidence      = text_conf,
                fusion_weight        = fusion_weight,
                price_text_agree     = price_text_agree,
                finbert_score        = fb_score,
                gpt4o_bias           = gpt4o_bias,
                gpt4o_confidence     = gpt4o_conf,
                gpt4o_surprise       = gpt4o_surp,
                rolling_nlp_accuracy = rolling_acc,
                article_count        = fb_count,
                regime               = regime,
                regime_boost         = regime_boost,
            ).to_dict())

        result = pd.DataFrame(packets).set_index("timestamp")
        self._log_summary(result)
        return result

    # ── Rolling NLP accuracy ──────────────────────────────────────────────────
    def _compute_rolling_accuracy(self) -> float:
        """
        Compute rolling directional accuracy of the NLP signal
        over the last ACCURACY_WINDOW active predictions.
        """
        if len(self._nlp_history) < 5:
            return 0.50   # insufficient data → assume neutral

        recent = self._nlp_history[-ACCURACY_WINDOW:]
        correct = sum(1 for nlp, true in recent if nlp == true)
        return correct / len(recent)

    # ── Diagnostics ───────────────────────────────────────────────────────────
    def _log_summary(self, result: pd.DataFrame) -> None:
        n       = len(result)
        bullish = (result["sentiment_signal"] ==  1).sum()
        bearish = (result["sentiment_signal"] == -1).sum()
        neutral = (result["sentiment_signal"] ==  0).sum()
        avg_fw  = result["fusion_weight"].mean()
        avg_acc = result["rolling_nlp_accuracy"].mean()
        agree   = result["price_text_agree"].mean() if "price_text_agree" in result.columns else 0.0

        logger.info(
            "Sentiment signals: BULL=%d (%.1f%%) | BEAR=%d (%.1f%%) | "
            "NEUTRAL=%d (%.1f%%) | avg_fusion_weight=%.3f | "
            "rolling_nlp_acc=%.3f | price_text_agree=%.1f%%",
            bullish, bullish/n*100,
            bearish, bearish/n*100,
            neutral, neutral/n*100,
            avg_fw, avg_acc, agree * 100,
        )

    # ── Convenience: live single-bar ──────────────────────────────────────────
    def update_bar(
        self,
        bar_ts:        pd.Timestamp,
        regime_signal: RegimeSignal,
        recent_news:   pd.DataFrame,
        price_signal:  int = 0,
    ) -> SentimentPacket:
        """
        Generate a SentimentPacket for the current live bar.
        Called once per bar in the live trading loop.

        Parameters
        ----------
        bar_ts        : current bar timestamp
        regime_signal : latest RegimeSignal from Agent 1
        recent_news   : scored news articles from the last few hours
        price_signal  : Agent 2's direction call for this bar

        Returns
        -------
        SentimentPacket ready for the Meta-Orchestrator
        """
        if not self._models_loaded:
            self.load_models()

        # Score any new articles
        if not recent_news.empty and "sentiment_score" not in recent_news.columns:
            recent_news = self.score_news(recent_news)

        # Aggregate to this bar
        mock_ohlcv = pd.DataFrame({"log_return_1": [0.0]}, index=[bar_ts])
        mock_ohlcv.index = pd.DatetimeIndex([bar_ts], tz="UTC")

        mock_regime = pd.DataFrame(
            {"regime": [regime_signal.regime]},
            index=pd.DatetimeIndex([bar_ts], tz="UTC")
        )
        mock_prices = pd.DataFrame(
            {"signal": [price_signal]},
            index=pd.DatetimeIndex([bar_ts], tz="UTC")
        )

        result = self.generate_signals(
            mock_ohlcv, recent_news, mock_regime,
            price_signals=mock_prices, window_hours=4.0
        )

        if bar_ts in result.index:
            row = result.loc[bar_ts]
            return SentimentPacket(**{
                k: row[k] for k in SentimentPacket.__dataclass_fields__
                if k != "timestamp"
            }, timestamp=bar_ts)

        return SentimentPacket(
            timestamp=bar_ts, instrument=self.instrument,
            sentiment_signal=0, text_confidence=0.0, fusion_weight=0.1,
            price_text_agree=False, finbert_score=0.0, gpt4o_bias=0.0,
            gpt4o_confidence=0.0, gpt4o_surprise=0.0,
            rolling_nlp_accuracy=0.5, article_count=0,
            regime=regime_signal.regime, regime_boost=False,
        )
