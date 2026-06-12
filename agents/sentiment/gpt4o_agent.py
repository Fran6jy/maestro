"""
maestro/agents/sentiment/gpt4o_agent.py
=========================================
GPT-4o macro analyst for high-impact Forex events.

Why GPT-4o on top of FinBERT?
-------------------------------
FinBERT is excellent at sentence-level sentiment but cannot reason
about complex macro causality. For example:

  Headline: "Fed holds rates steady, signals patience"

  FinBERT: probably "neutral" (factual statement)
  GPT-4o:  "Dovish — Fed patience implies rates will fall → USD bearish
             in medium term, particularly vs EUR where ECB is less dovish"

GPT-4o understands:
  - Central bank stance (hawkish / dovish / neutral)
  - Inter-market relationships (Fed rate → USD/EUR impact)
  - Event context (is this surprising vs consensus?)
  - Forward guidance language ("transitory", "data-dependent", "appropriate")
  - Geopolitical risk transmission to Forex

Rate limiting and cost management
----------------------------------
GPT-4o is expensive at scale. This module is rate-limited to:
  - Only called for HIGH-IMPACT events (FOMC, NFP, CPI, ECB, BoE releases)
  - Maximum 50 calls per day in production
  - Results cached indefinitely (same text = same analysis)
  - Falls back to FinBERT-only if API fails or budget exceeded

Structured output
------------------
GPT-4o is prompted to return JSON with:
  - eur_usd_bias:   float [-1,+1]  — impact on EUR/USD
  - gbp_usd_bias:   float [-1,+1]  — impact on GBP/USD
  - confidence:     float [0,1]    — model's self-assessed confidence
  - horizon:        str            — "immediate" | "hours" | "days" | "weeks"
  - rationale:      str            — chain-of-thought explanation (for XAI)
  - event_type:     str            — "central_bank" | "macro_data" | "geopolitical"
  - surprise_factor:float [-1,+1]  — vs market consensus
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# High-impact Forex keywords that trigger GPT-4o analysis
HIGH_IMPACT_KEYWORDS = {
    # Central banks
    "federal reserve", "fed", "fomc", "powell", "jerome powell",
    "ecb", "european central bank", "lagarde",
    "bank of england", "boe", "bailey", "andrew bailey",
    "interest rate", "rate decision", "rate hike", "rate cut",
    "quantitative easing", "qe", "tapering", "balance sheet",
    "forward guidance", "hawkish", "dovish", "pivot",
    # Key data releases
    "nonfarm payroll", "nfp", "unemployment rate", "jobs report",
    "cpi", "inflation", "pce", "core inflation",
    "gdp", "retail sales", "ism manufacturing", "pmi",
    # Market-moving events
    "recession", "banking crisis", "credit crunch",
    "geopolitical", "sanctions", "trade war", "tariff",
}

GPT4O_SYSTEM_PROMPT = """You are a professional Forex macro analyst specialising in EUR/USD and GBP/USD.

When given a financial news headline or article, analyse its impact on EUR/USD and GBP/USD exchange rates.

Always respond with valid JSON only — no preamble, no markdown, no explanation outside the JSON.

JSON schema (all fields required):
{
  "eur_usd_bias": <float -1 to +1, positive = EUR strengthens vs USD>,
  "gbp_usd_bias": <float -1 to +1, positive = GBP strengthens vs USD>,
  "confidence": <float 0 to 1, your confidence in this assessment>,
  "horizon": <"immediate" | "hours" | "days" | "weeks">,
  "event_type": <"central_bank" | "macro_data" | "geopolitical" | "risk_sentiment" | "other">,
  "surprise_factor": <float -1 to +1, +1 = very hawkish/positive surprise, -1 = very dovish/negative surprise>,
  "rationale": <string, max 150 words, chain-of-thought explanation>,
  "key_entities": <list of strings, central banks / currencies / instruments mentioned>
}

Guidelines:
- Positive EUR/USD bias = EUR is likely to strengthen relative to USD
- Negative EUR/USD bias = USD is likely to strengthen (EUR weakens)
- Surprise factor measures deviation from market consensus
- Be precise: "Fed holds rates" when a hike was expected = surprise_factor around -0.6
- If the news is irrelevant to Forex, set both biases to 0.0 and confidence to 0.1
"""


@dataclass
class MacroAnalysis:
    """Structured GPT-4o output for one news item."""
    text_hash:      str
    eur_usd_bias:   float         # [-1, +1]
    gbp_usd_bias:   float         # [-1, +1]
    confidence:     float         # [0, 1]
    horizon:        str           # immediate | hours | days | weeks
    event_type:     str
    surprise_factor:float         # [-1, +1]
    rationale:      str
    key_entities:   list[str]     = field(default_factory=list)
    timestamp:      datetime      = field(default_factory=lambda: datetime.now(timezone.utc))
    model:          str           = "gpt-4o"
    tokens_used:    int           = 0

    @classmethod
    def neutral(cls, text_hash: str = "") -> "MacroAnalysis":
        return cls(
            text_hash=text_hash, eur_usd_bias=0.0, gbp_usd_bias=0.0,
            confidence=0.1, horizon="immediate", event_type="other",
            surprise_factor=0.0, rationale="Not relevant to Forex."
        )

    def to_dict(self) -> dict:
        return {
            "eur_usd_bias":    self.eur_usd_bias,
            "gbp_usd_bias":    self.gbp_usd_bias,
            "confidence":      self.confidence,
            "horizon":         self.horizon,
            "event_type":      self.event_type,
            "surprise_factor": self.surprise_factor,
            "rationale":       self.rationale,
        }


class GPT4oMacroAnalyst:
    """
    GPT-4o-powered macro impact analyser for Forex.

    Usage
    -----
    >>> analyst = GPT4oMacroAnalyst()
    >>> result  = analyst.analyse("Fed raises rates by 25bps, signals one more hike")
    >>> result.eur_usd_bias
    -0.6

    >>> # Batch — automatically filters to high-impact only
    >>> results = analyst.analyse_batch(news_df)

    >>> # Aggregate to bar-level signal
    >>> bar_signals = analyst.aggregate_to_bars(results_df, ohlcv_index)
    """

    def __init__(
        self,
        model:        str = "gpt-4o",
        max_daily_calls: int = 50,
        cache_path:   str | None = None,
    ) -> None:
        self.model           = model
        self.max_daily_calls = max_daily_calls
        self.cache_path      = Path(cache_path or os.path.expanduser(
            "~/.cache/maestro/gpt4o_cache.pkl"
        ))
        self._cache: dict[str, MacroAnalysis] = {}
        self._daily_call_count = 0
        self._last_reset_day   = datetime.now(timezone.utc).date()
        self._load_cache()

    # ── Single article analysis ───────────────────────────────────────────────
    def analyse(
        self,
        text:         str,
        force_call:   bool = False,
    ) -> MacroAnalysis:
        """
        Analyse one news text for Forex macro impact.

        GPT-4o is only called if:
          1. Text contains high-impact keywords, OR force_call=True
          2. Daily call budget not exceeded
          3. Not already cached

        Falls back to neutral analysis otherwise.
        """
        h = _hash(text)

        # Return cached result
        if h in self._cache:
            logger.debug("GPT-4o cache hit for: %.60s...", text)
            return self._cache[h]

        # Check if high-impact
        if not force_call and not self._is_high_impact(text):
            return MacroAnalysis.neutral(h)

        # Check daily budget
        self._refresh_daily_counter()
        if self._daily_call_count >= self.max_daily_calls:
            logger.warning(
                "GPT-4o daily limit (%d) reached — returning neutral", self.max_daily_calls
            )
            return MacroAnalysis.neutral(h)

        # Call GPT-4o
        result = self._call_gpt4o(text, h)
        self._cache[h] = result
        self._daily_call_count += 1
        self._save_cache()
        return result

    # ── Batch analysis ────────────────────────────────────────────────────────
    def analyse_batch(
        self,
        df: pd.DataFrame,
        text_col: str = "full_text",
        timestamp_col: str = "published_at",
    ) -> pd.DataFrame:
        """
        Analyse all articles in a news DataFrame.

        Only high-impact articles are sent to GPT-4o (budget management).
        Others receive neutral scores or are skipped.

        Returns
        -------
        Input DataFrame with additional columns from MacroAnalysis.to_dict()
        """
        if df.empty:
            return df

        texts   = df[text_col].fillna("").tolist()
        results = []

        high_impact_count = 0
        for text in texts:
            result = self.analyse(text)
            results.append(result.to_dict())
            if result.confidence > 0.1:
                high_impact_count += 1

        result_df = pd.DataFrame(results, index=df.index)
        result_df.columns = [f"gpt4o_{c}" for c in result_df.columns]
        out = pd.concat([df, result_df], axis=1)

        logger.info(
            "GPT-4o batch complete: %d articles | %d high-impact | "
            "%d API calls today",
            len(df), high_impact_count, self._daily_call_count
        )
        return out

    # ── Aggregate to bar-level ────────────────────────────────────────────────
    def aggregate_to_bars(
        self,
        analysed_df:  pd.DataFrame,
        bar_index:    pd.DatetimeIndex,
        instrument:   str   = "EUR_USD",
        window_hours: float = 6.0,
    ) -> pd.DataFrame:
        """
        Aggregate GPT-4o macro signals to OHLCV bar timestamps.

        For each bar, looks back window_hours and selects the highest-
        confidence analysis, weighted by confidence and recency.

        Returns
        -------
        pd.DataFrame with columns:
          gpt4o_macro_signal, gpt4o_confidence, gpt4o_surprise,
          gpt4o_horizon_score, gpt4o_event_count
        Index matches bar_index
        """
        bias_col = "gpt4o_eur_usd_bias" if instrument == "EUR_USD" else "gpt4o_gbp_usd_bias"
        required  = [bias_col, "gpt4o_confidence", "gpt4o_surprise_factor"]
        if analysed_df.empty or not all(c in analysed_df.columns for c in required):
            return self._zero_macro(bar_index)

        ts_col = "published_at"
        if ts_col not in analysed_df.columns:
            return self._zero_macro(bar_index)

        analysed_df = analysed_df.copy()
        analysed_df[ts_col] = pd.to_datetime(analysed_df[ts_col], utc=True)
        window_td = pd.Timedelta(hours=window_hours)

        horizon_map = {"immediate": 1.0, "hours": 0.7, "days": 0.4, "weeks": 0.2}
        results = []

        for bar_ts in bar_index:
            start = bar_ts - window_td
            mask  = (
                (analysed_df[ts_col] > start) &
                (analysed_df[ts_col] <= bar_ts)
            )
            window = analysed_df[mask]

            if window.empty:
                results.append(self._zero_row())
                continue

            # Weight by confidence × recency
            confs     = window["gpt4o_confidence"].fillna(0).values
            hours_ago = (bar_ts - window[ts_col]).dt.total_seconds().values / 3600
            recency   = 0.8 ** hours_ago
            weights   = confs * recency
            w_sum     = weights.sum()

            if w_sum < 1e-8:
                results.append(self._zero_row())
                continue

            weights /= w_sum

            biases    = window[bias_col].fillna(0).values
            surprises = window["gpt4o_surprise_factor"].fillna(0).values
            horizons  = window["gpt4o_horizon"].fillna("hours").map(
                horizon_map
            ).fillna(0.5).values if "gpt4o_horizon" in window.columns else np.ones(len(window)) * 0.5

            agg_bias     = float(np.dot(weights, biases))
            agg_surprise = float(np.dot(weights, surprises))
            agg_horizon  = float(np.dot(weights, horizons))
            agg_conf     = float(confs.max())   # max confidence in window

            signal = int(round(agg_bias * 2)) if abs(agg_bias) > 0.1 else 0
            signal = max(-1, min(1, signal))

            results.append({
                "gpt4o_macro_signal":  signal,
                "gpt4o_confidence":    agg_conf,
                "gpt4o_surprise":      agg_surprise,
                "gpt4o_horizon_score": agg_horizon,
                "gpt4o_event_count":   len(window),
            })

        import numpy as np
        return pd.DataFrame(results, index=bar_index)

    # ── GPT-4o API call ───────────────────────────────────────────────────────
    def _call_gpt4o(self, text: str, text_hash: str) -> MacroAnalysis:
        """Make a single GPT-4o API call with retry."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai>=1.3.0")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set — returning neutral")
            return MacroAnalysis.neutral(text_hash)

        client = OpenAI(api_key=api_key)

        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model       = self.model,
                    messages    = [
                        {"role": "system", "content": GPT4O_SYSTEM_PROMPT},
                        {"role": "user",   "content": f"Analyse this news for Forex impact:\n\n{text[:2000]}"},
                    ],
                    max_tokens      = 400,
                    temperature     = 0.1,    # low temperature = consistent structured output
                    response_format = {"type": "json_object"},
                )
                raw     = response.choices[0].message.content
                tokens  = response.usage.total_tokens
                parsed  = json.loads(raw)

                return MacroAnalysis(
                    text_hash       = text_hash,
                    eur_usd_bias    = float(parsed.get("eur_usd_bias",    0.0)),
                    gbp_usd_bias    = float(parsed.get("gbp_usd_bias",    0.0)),
                    confidence      = float(parsed.get("confidence",      0.5)),
                    horizon         = str(parsed.get("horizon",           "hours")),
                    event_type      = str(parsed.get("event_type",        "other")),
                    surprise_factor = float(parsed.get("surprise_factor", 0.0)),
                    rationale       = str(parsed.get("rationale",         "")),
                    key_entities    = list(parsed.get("key_entities",     [])),
                    tokens_used     = tokens,
                )

            except json.JSONDecodeError as exc:
                logger.warning("GPT-4o JSON parse error (attempt %d): %s", attempt + 1, exc)
                if attempt == 2:
                    return MacroAnalysis.neutral(text_hash)

            except Exception as exc:
                wait = 2 ** attempt
                logger.warning("GPT-4o API error (attempt %d): %s — retrying in %ds",
                               attempt + 1, exc, wait)
                time.sleep(wait)
                if attempt == 2:
                    return MacroAnalysis.neutral(text_hash)

        return MacroAnalysis.neutral(text_hash)

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _is_high_impact(text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in HIGH_IMPACT_KEYWORDS)

    def _refresh_daily_counter(self) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self._last_reset_day:
            self._daily_call_count = 0
            self._last_reset_day   = today

    @staticmethod
    def _zero_row() -> dict:
        return {
            "gpt4o_macro_signal":  0,
            "gpt4o_confidence":    0.0,
            "gpt4o_surprise":      0.0,
            "gpt4o_horizon_score": 0.5,
            "gpt4o_event_count":   0,
        }

    @staticmethod
    def _zero_macro(index: pd.DatetimeIndex) -> pd.DataFrame:
        import numpy as np
        return pd.DataFrame({
            "gpt4o_macro_signal":  0,
            "gpt4o_confidence":    0.0,
            "gpt4o_surprise":      0.0,
            "gpt4o_horizon_score": 0.5,
            "gpt4o_event_count":   0,
        }, index=index)

    def _load_cache(self) -> None:
        if self.cache_path.exists():
            with open(self.cache_path, "rb") as f:
                self._cache = pickle.load(f)
            logger.info("GPT-4o cache loaded: %d entries", len(self._cache))

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump(self._cache, f)


def _hash(text: str) -> str:
    return hashlib.md5(text.lower().strip().encode()).hexdigest()
