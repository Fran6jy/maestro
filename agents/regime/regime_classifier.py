"""
maestro/agents/regime/regime_classifier.py
============================================
Regime Detection Ensemble — the complete Agent 1.

Combines HMM (probabilistic structure) + Transformer (discriminative
power) into a single, calibrated regime signal via soft voting with
confidence-weighted fusion.

Fusion strategy
---------------
1. HMM outputs  P_HMM(regime | X)   — 4-class probability vector
2. Transformer outputs P_TRF(regime | X) — 4-class probability vector
3. Ensemble:  P_final = α × P_HMM + (1-α) × P_TRF
   where α is tuned on the WFA validation window (default 0.4)
4. Final regime  = argmax P_final
5. Confidence    = max(P_final) — only act on high-confidence regimes

Regime broadcast protocol
-------------------------
The classifier outputs a RegimeSignal dataclass that is passed to
every other agent on each bar. Agents use .regime and .confidence
to gate their behaviour — low-confidence bars → agents reduce position size.

Regime IDs (consistent across all agents)
-----------------------------------------
0 = bull_trend   → Signal agents: momentum strategies preferred
1 = bear_trend   → Signal agents: mean-reversion + short bias
2 = sideways     → Signal agents: range / BB strategies preferred
3 = crisis       → Risk agent: reduce all positions; LLM agent: elevated weight
"""
from __future__ import annotations

import json

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from maestro.agents.regime.hmm_regime import (
    HMMConfig,
    HMMRegimeDetector,
    REGIME_NAMES,
)
from maestro.agents.regime.transformer_regime import (
    TransformerConfig,
    TransformerRegimeClassifier,
)

logger = logging.getLogger(__name__)

N_REGIMES = 4
CONFIDENCE_THRESHOLD = 0.55    # below this → "uncertain" regime


# ── Output signal ─────────────────────────────────────────────────────────────
@dataclass
class RegimeSignal:
    """
    Broadcasted output of Agent 1. Consumed by all downstream agents.

    Fields
    ------
    timestamp    : bar datetime (UTC)
    regime       : int {0,1,2,3} — most probable regime
    regime_name  : str e.g. "bull_trend"
    confidence   : float [0,1] — max probability across regimes
    probabilities: dict[str, float] — full probability distribution
    is_certain   : bool — True if confidence > CONFIDENCE_THRESHOLD
    source       : str — "hmm" | "transformer" | "ensemble"
    """
    timestamp:     pd.Timestamp
    regime:        int
    regime_name:   str
    confidence:    float
    probabilities: dict[str, float]
    is_certain:    bool
    source:        str = "ensemble"

    def __repr__(self) -> str:
        return (
            f"RegimeSignal({self.timestamp.date()} | "
            f"{self.regime_name} | conf={self.confidence:.3f} | "
            f"certain={self.is_certain})"
        )

    def to_dict(self) -> dict:
        return {
            "timestamp":   self.timestamp,
            "regime":      self.regime,
            "regime_name": self.regime_name,
            "confidence":  self.confidence,
            "is_certain":  self.is_certain,
            **{f"prob_{k}": v for k, v in self.probabilities.items()},
        }


# ── Main Agent ────────────────────────────────────────────────────────────────
class RegimeDetectionAgent:
    """
    Agent 1 — Market Regime Detection.

    The single authoritative source of regime information for all
    other MAESTRO agents. Combines HMM + Transformer via soft voting.

    Usage
    -----
    >>> agent = RegimeDetectionAgent()
    >>> agent.fit(train_df)                        # fit both models
    >>> signals = agent.predict_batch(test_df)     # pd.DataFrame of RegimeSignals
    >>> signal  = agent.predict_bar(bar_features)  # single RegimeSignal (live)

    Walk-Forward Integration
    ------------------------
    >>> for split in wfa_engine.splits(features_df):
    ...     agent = RegimeDetectionAgent()
    ...     agent.fit(features_df.loc[split.train_idx])
    ...     results = agent.predict_batch(features_df.loc[split.test_idx])
    """

    def __init__(
        self,
        hmm_config:         HMMConfig | None         = None,
        transformer_config: TransformerConfig | None = None,
        hmm_weight:         float                    = 0.4,
        use_transformer:    bool                     = True,
    ) -> None:
        self.hmm         = HMMRegimeDetector(hmm_config)
        self.transformer = TransformerRegimeClassifier(transformer_config) if use_transformer else None
        self.hmm_weight  = hmm_weight          # α in ensemble formula
        self.use_transformer = use_transformer
        self.fitted      = False

    # ── Fit ───────────────────────────────────────────────────────────────────
    def fit(
        self,
        train_df: pd.DataFrame,
        val_df:   pd.DataFrame | None = None,
    ) -> "RegimeDetectionAgent":
        """
        Fit both HMM and Transformer on the training window.

        The Transformer uses HMM predictions as pseudo-labels
        (self-supervised bootstrapping), then refines on validation data.

        Parameters
        ----------
        train_df : feature DataFrame (full training window)
        val_df   : optional held-out window for Transformer early stopping
        """
        logger.info("=" * 55)
        logger.info("Agent 1: Regime Detection — Training")
        logger.info("Train bars: %d | Features: %d", len(train_df), len(train_df.columns))

        # ── Step 1: Fit HMM ───────────────────────────────────────────────────
        logger.info("Step 1/3: Fitting HMM...")
        self.hmm.fit(train_df)

        # ── Step 2: Generate HMM pseudo-labels for Transformer ────────────────
        if self.use_transformer:
            logger.info("Step 2/3: Generating HMM pseudo-labels...")
            hmm_labels = self.hmm.predict(train_df, causal=False)   # labels for past data: hindsight is fine

            val_labels = None
            if val_df is not None:
                val_labels = self.hmm.predict(val_df, causal=False)

            # ── Step 3: Fit Transformer ───────────────────────────────────────
            logger.info("Step 3/3: Training Transformer classifier...")
            self.transformer.fit(
                train_df, hmm_labels,
                val_df=val_df, val_labels=val_labels
            )
        else:
            logger.info("Step 2–3: Transformer disabled — HMM only.")

        # ── Optional: tune ensemble weight on val set ──────────────────────────
        if val_df is not None and self.use_transformer:
            self.hmm_weight = self._tune_weight(val_df)
            logger.info("Ensemble weight tuned: α(HMM)=%.2f | α(TRF)=%.2f",
                        self.hmm_weight, 1 - self.hmm_weight)

        self.fitted = True
        logger.info("Agent 1 training complete.")
        logger.info("=" * 55)
        return self

    # ── Batch prediction ──────────────────────────────────────────────────────
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regime signals for all bars in df.

        Returns
        -------
        pd.DataFrame with columns:
          regime, regime_name, confidence, is_certain,
          prob_bull_trend, prob_bear_trend, prob_sideways, prob_crisis,
          hmm_regime, transformer_regime (if available)
        """
        self._check_fitted()
        logger.info("Agent 1: Predicting regimes for %d bars...", len(df))

        # HMM probabilities
        hmm_proba = self.hmm.predict_proba(df)     # (n, 4)
        hmm_preds = self.hmm.predict(df)

        # Transformer probabilities
        if self.use_transformer and self.transformer.fitted:
            trf_proba = self.transformer.predict_proba(df)   # (n, 4) with NaN for first seq_len-1
            trf_preds = self.transformer.predict(df)
        else:
            trf_proba = pd.DataFrame(
                np.full((len(df), N_REGIMES), 1/N_REGIMES),
                index   = df.index,
                columns = [f"transformer_prob_{REGIME_NAMES[i]}" for i in range(N_REGIMES)]
            )
            trf_preds = pd.Series(np.zeros(len(df)), index=df.index, dtype=int)

        # ── Ensemble fusion ────────────────────────────────────────────────────
        hmm_arr = hmm_proba.values     # (n, 4)
        trf_arr = trf_proba.values     # (n, 4)

        # Handle NaN in transformer (early bars without full window)
        trf_nan  = np.isnan(trf_arr).any(axis=1)
        trf_arr  = trf_arr.copy()   # ensure writable (fix for numpy read-only views)
        trf_arr[trf_nan] = 1.0 / N_REGIMES    # uniform prior for uncertain bars

        ensemble = self.hmm_weight * hmm_arr + (1 - self.hmm_weight) * trf_arr
        regimes  = ensemble.argmax(axis=1)
        confs    = ensemble.max(axis=1)

        # Build output DataFrame
        prob_cols = [f"prob_{REGIME_NAMES[i]}" for i in range(N_REGIMES)]
        result = pd.DataFrame(ensemble, index=df.index, columns=prob_cols)
        result["regime"]      = regimes
        result["regime_name"] = [REGIME_NAMES[r] for r in regimes]
        result["confidence"]  = confs
        result["is_certain"]  = confs >= CONFIDENCE_THRESHOLD
        result["hmm_regime"]  = hmm_preds.values
        result["trf_regime"]  = trf_preds.values
        result["trf_uncertain"] = trf_nan

        self._log_regime_distribution(result)
        return result

    # ── Single-bar live prediction ─────────────────────────────────────────────
    def predict_bar(
        self,
        bar: pd.Series | pd.DataFrame,
        history_df: pd.DataFrame | None = None,
    ) -> RegimeSignal:
        """
        Predict regime for a single incoming bar (live trading).

        Parameters
        ----------
        bar        : a single-row DataFrame or Series of features
        history_df : optional — last seq_len bars needed for Transformer.
                     If None, HMM-only prediction is returned.

        Returns
        -------
        RegimeSignal — the authoritative regime broadcast for this bar.
        """
        self._check_fitted()

        # Normalise input
        if isinstance(bar, pd.Series):
            bar_df = bar.to_frame().T
        else:
            bar_df = bar

        timestamp = bar_df.index[0]

        # HMM prediction (doesn't need history window)
        hmm_proba = self.hmm.predict_proba(bar_df).values[0]   # (4,)

        # Transformer prediction (needs seq_len bars)
        if self.use_transformer and self.transformer.fitted and history_df is not None:
            ctx = pd.concat([history_df.tail(self.transformer.cfg.seq_len - 1), bar_df])
            trf_proba = self.transformer.predict_proba(ctx)
            if not trf_proba.empty and not np.isnan(trf_proba.values[-1]).any():
                trf_arr = trf_proba.values[-1]
            else:
                trf_arr = np.ones(N_REGIMES) / N_REGIMES
        else:
            trf_arr = np.ones(N_REGIMES) / N_REGIMES

        ensemble   = self.hmm_weight * hmm_proba + (1 - self.hmm_weight) * trf_arr
        regime     = int(ensemble.argmax())
        confidence = float(ensemble.max())
        source     = "ensemble" if self.use_transformer else "hmm"

        return RegimeSignal(
            timestamp    = timestamp,
            regime       = regime,
            regime_name  = REGIME_NAMES[regime],
            confidence   = confidence,
            probabilities= {REGIME_NAMES[i]: float(ensemble[i]) for i in range(N_REGIMES)},
            is_certain   = confidence >= CONFIDENCE_THRESHOLD,
            source       = source,
        )

    # ── Weight tuning ──────────────────────────────────────────────────────────
    def _tune_weight(self, val_df: pd.DataFrame) -> float:
        """
        Grid-search HMM weight α on validation set to maximise accuracy.
        Tests α ∈ {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}.
        """
        hmm_proba = self.hmm.predict_proba(val_df).values
        trf_proba = self.transformer.predict_proba(val_df).values
        hmm_preds = self.hmm.predict(val_df).values

        best_alpha  = self.hmm_weight
        best_agree  = -1.0

        for alpha in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            ens      = alpha * hmm_proba + (1 - alpha) * trf_proba
            ens_pred = ens.argmax(axis=1)
            # Agreement with HMM as proxy for stability
            agree = (ens_pred == hmm_preds).mean()
            if agree > best_agree:
                best_agree = agree
                best_alpha = alpha

        return best_alpha

    # ── Diagnostics ───────────────────────────────────────────────────────────
    def _log_regime_distribution(self, result: pd.DataFrame) -> None:
        counts = result["regime_name"].value_counts()
        total  = len(result)
        dist   = " | ".join(f"{k}: {v/total:.1%}" for k, v in counts.items())
        certain_pct = result["is_certain"].mean()
        logger.info("Regime distribution: %s | certain: %.1f%%", dist, certain_pct * 100)

    def regime_transition_analysis(self, result: pd.DataFrame) -> pd.DataFrame:
        """
        Analyse regime transitions — how often does the market switch states?
        Returns a transition count matrix.
        """
        regimes = result["regime_name"]
        trans   = pd.crosstab(
            regimes.shift(1).rename("from"),
            regimes.rename("to"),
        )
        return trans

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, model_dir: str | Path) -> None:
        """Save both sub-models to a directory."""
        self._check_fitted()
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        self.hmm.save(model_dir / "hmm_regime.pkl")
        if self.use_transformer and self.transformer.fitted:
            self.transformer.save(model_dir / "transformer_regime.pt")
        # The HMM's share of the vote is tuned on validation data during fit(), so it has to
        # travel with the model; otherwise a reloaded model votes differently.
        (model_dir / "ensemble.json").write_text(json.dumps({"hmm_weight": self.hmm_weight}))
        logger.info("Agent 1 saved → %s", model_dir)

    @classmethod
    def load(cls, model_dir: str | Path) -> "RegimeDetectionAgent":
        """Load a saved RegimeDetectionAgent."""
        model_dir = Path(model_dir)
        agent = cls.__new__(cls)
        agent.hmm         = HMMRegimeDetector.load(model_dir / "hmm_regime.pkl")
        agent.use_transformer = (model_dir / "transformer_regime.pt").exists()
        if agent.use_transformer:
            agent.transformer = TransformerRegimeClassifier.load(
                model_dir / "transformer_regime.pt"
            )
        else:
            agent.transformer = None
        ensemble = model_dir / "ensemble.json"
        agent.hmm_weight  = json.loads(ensemble.read_text())["hmm_weight"] if ensemble.exists() else 0.4
        agent.fitted      = True
        logger.info("Agent 1 loaded from %s", model_dir)
        return agent

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("Agent not fitted. Call .fit(train_df) first.")
