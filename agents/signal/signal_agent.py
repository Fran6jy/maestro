"""
maestro/agents/signal/signal_agent.py
=======================================
Agent 2 — Technical Signal Agent.

Combines TFT + PatchTST into a single directional signal via
regime-conditioned ensemble weighting. This is what feeds the
Risk Agent and ultimately the Execution Agent.

Regime-conditioned weighting rationale
----------------------------------------
Different models excel in different regimes:

  bull_trend  → TFT dominates: captures momentum continuation
  bear_trend  → TFT dominates: captures momentum continuation (short)
  sideways    → PatchTST dominates: patch-level mean-reversion patterns
  crisis      → Neither trusted fully; position size reduced by Risk Agent
                Both weighted equally; regime uncertainty dominates

The weights are tuned per-split on the validation window using
a grid search over the [0,1] weight space.

Signal output
-------------
The agent outputs a SignalPacket per bar containing:
  - signal:      final directional call {-1, 0, +1}
  - confidence:  calibrated confidence score [0,1]
  - horizon:     which horizon is being traded (configurable)
  - regime:      regime label at time of signal
  - model_agree: whether TFT and PatchTST agree (higher = more reliable)
  - metadata:    per-model breakdown for XAI
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from maestro.agents.signal.tft_model import TFTSignalModel, TFTConfig, HORIZONS
from maestro.agents.signal.patchtst import PatchTSTSignalModel, PatchTSTConfig
from maestro.agents.regime.regime_classifier import RegimeSignal, REGIME_NAMES

logger = logging.getLogger(__name__)

# Regime-conditioned default weights [TFT_weight, PatchTST_weight]
# Tuned empirically — overridden by val-set optimization per WFA split
DEFAULT_REGIME_WEIGHTS = {
    0: (0.65, 0.35),   # bull_trend:  TFT stronger
    1: (0.65, 0.35),   # bear_trend:  TFT stronger
    2: (0.35, 0.65),   # sideways:    PatchTST stronger
    3: (0.50, 0.50),   # crisis:      equal weight; Risk Agent handles uncertainty
}

# Confidence thresholds per regime (crisis demands higher bar)
CONFIDENCE_THRESHOLDS = {
    0: 0.52,   # bull_trend
    1: 0.52,   # bear_trend
    2: 0.55,   # sideways (mean-reversion harder to call)
    3: 0.62,   # crisis (must be very confident to trade)
}


@dataclass
class SignalPacket:
    """
    Output of Agent 2. Consumed by Agent 4 (Risk) and Agent 5 (Execution).

    Fields
    ------
    timestamp   : bar datetime (UTC)
    instrument  : e.g. "EUR_USD"
    horizon     : prediction horizon in bars
    signal      : {-1=sell, 0=flat, +1=buy}
    confidence  : ensemble confidence score [0,1]
    regime      : regime ID from Agent 1
    regime_name : human-readable regime name
    model_agree : True if TFT and PatchTST agree on direction
    tft_signal  : raw TFT directional call
    ptst_signal : raw PatchTST directional call
    pred_p10    : TFT 10th percentile return (downside)
    pred_p50    : TFT median return prediction
    pred_p90    : TFT 90th percentile return (upside)
    """
    timestamp:   pd.Timestamp
    instrument:  str
    horizon:     int
    signal:      int
    confidence:  float
    regime:      int
    regime_name: str
    model_agree: bool
    tft_signal:  int
    ptst_signal: int
    pred_p10:    float
    pred_p50:    float
    pred_p90:    float
    metadata:    dict = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        """True when signal is non-flat AND above regime confidence threshold."""
        return (
            self.signal != 0
            and self.confidence >= CONFIDENCE_THRESHOLDS.get(self.regime, 0.55)
        )

    def __repr__(self) -> str:
        direction = {1: "BUY", -1: "SELL", 0: "FLAT"}.get(self.signal, "?")
        return (
            f"SignalPacket({self.timestamp.strftime('%Y-%m-%d %H:%M')} | "
            f"{self.instrument} | {direction} h={self.horizon} | "
            f"conf={self.confidence:.3f} | {self.regime_name} | "
            f"agree={self.model_agree})"
        )

    def to_dict(self) -> dict:
        return {
            "timestamp":   self.timestamp,
            "instrument":  self.instrument,
            "horizon":     self.horizon,
            "signal":      self.signal,
            "confidence":  self.confidence,
            "regime":      self.regime,
            "regime_name": self.regime_name,
            "model_agree": self.model_agree,
            "tft_signal":  self.tft_signal,
            "ptst_signal": self.ptst_signal,
            "pred_p10":    self.pred_p10,
            "pred_p50":    self.pred_p50,
            "pred_p90":    self.pred_p90,
            "is_actionable": self.is_actionable,
        }


class SignalAgent:
    """
    Agent 2 — Technical Signal Agent.

    Trains TFT + PatchTST, fuses their outputs with regime-conditioned
    weights, and emits SignalPackets consumed by downstream agents.

    Usage
    -----
    >>> agent = SignalAgent(instrument="EUR_USD")
    >>> agent.fit(train_df, train_regimes)
    >>> packets = agent.predict_batch(test_df, test_regimes)
    >>> packet  = agent.predict_bar(bar_df, regime_signal)   # live

    Walk-Forward Integration
    ------------------------
    >>> for split in wfa.splits(features_df):
    ...     agent = SignalAgent(instrument="EUR_USD")
    ...     agent.fit(features_df.loc[split.train_idx], regimes.loc[split.train_idx])
    ...     packets = agent.predict_batch(features_df.loc[split.test_idx],
    ...                                   regimes.loc[split.test_idx])
    """

    def __init__(
        self,
        instrument:    str  = "EUR_USD",
        primary_horizon: int = 6,      # default: 30-min horizon (6 × 5min bars)
        tft_config:    TFTConfig | None      = None,
        ptst_config:   PatchTSTConfig | None = None,
    ) -> None:
        self.instrument      = instrument
        self.primary_horizon = primary_horizon
        self.instrument_id   = 0 if instrument == "EUR_USD" else 1

        self.tft  = TFTSignalModel(tft_config)
        self.ptst = PatchTSTSignalModel(ptst_config)

        # Per-regime ensemble weights (tuned on val set)
        self.regime_weights: dict[int, tuple[float, float]] = dict(DEFAULT_REGIME_WEIGHTS)
        self.fitted = False

    # ── Fit ───────────────────────────────────────────────────────────────────
    def fit(
        self,
        train_df:      pd.DataFrame,
        train_regimes: pd.Series,
        val_df:        pd.DataFrame | None = None,
        val_regimes:   pd.Series | None    = None,
    ) -> "SignalAgent":
        """
        Train both TFT and PatchTST on the training window.

        Parameters
        ----------
        train_df      : full feature DataFrame
        train_regimes : pd.Series[int] from Agent 1, same index as train_df
        val_df        : optional held-out window for early stopping + weight tuning
        val_regimes   : regime labels for val_df
        """
        logger.info("=" * 55)
        logger.info("Agent 2: Technical Signal — Training")
        logger.info("Instrument: %s | Bars: %d", self.instrument, len(train_df))

        # ── Build horizon labels for PatchTST ─────────────────────────────────
        logger.info("Step 1/3: Building horizon labels...")
        train_labels = PatchTSTSignalModel.build_horizon_labels(train_df)
        val_labels   = PatchTSTSignalModel.build_horizon_labels(val_df) if val_df is not None else None

        # ── Fit TFT ───────────────────────────────────────────────────────────
        logger.info("Step 2/3: Fitting TFT...")
        self.tft.fit(
            train_df, train_regimes, self.instrument_id,
            val_df=val_df, val_regimes=val_regimes
        )

        # ── Fit PatchTST ──────────────────────────────────────────────────────
        logger.info("Step 3/3: Fitting PatchTST...")
        self.ptst.fit(
            train_df, train_labels,
            val_df=val_df, val_labels=val_labels
        )

        # ── Tune ensemble weights on validation set ────────────────────────────
        if val_df is not None and val_regimes is not None:
            logger.info("Tuning regime-conditioned ensemble weights...")
            self.regime_weights = self._tune_weights(val_df, val_regimes, val_labels)
            for regime_id, (w_tft, w_ptst) in self.regime_weights.items():
                logger.info("  %s: TFT=%.2f PatchTST=%.2f",
                            REGIME_NAMES[regime_id], w_tft, w_ptst)

        self.fitted = True
        logger.info("Agent 2 training complete.")
        logger.info("=" * 55)
        return self

    # ── Batch predict ─────────────────────────────────────────────────────────
    def predict_batch(
        self,
        df:      pd.DataFrame,
        regimes: pd.Series,
    ) -> pd.DataFrame:
        """
        Generate SignalPackets for all bars in df.

        Returns
        -------
        pd.DataFrame — one row per bar, all SignalPacket fields as columns.
        """
        self._check_fitted()
        logger.info("Agent 2: Predicting signals for %d bars...", len(df))

        # Get raw model outputs
        tft_out  = self.tft.predict(df, regimes, self.instrument_id)
        ptst_out = self.ptst.predict(df)

        # Align indices (PatchTST and TFT may trim different numbers of rows)
        common_idx = tft_out.index.intersection(ptst_out.index)
        tft_out    = tft_out.loc[common_idx]
        ptst_out   = ptst_out.loc[common_idx]
        reg_aligned= regimes.reindex(common_idx).fillna(2).astype(int)

        packets = []
        h       = self.primary_horizon
        h_col   = h if h in HORIZONS else HORIZONS[0]

        for idx in common_idx:
            regime     = int(reg_aligned.loc[idx])
            w_tft, w_ptst = self.regime_weights.get(regime, (0.5, 0.5))

            # TFT outputs
            tft_sig  = int(tft_out.loc[idx, f"signal_{h_col}"])  if f"signal_{h_col}" in tft_out.columns else 0
            tft_conf = float(tft_out.loc[idx, f"confidence_{h_col}"]) if f"confidence_{h_col}" in tft_out.columns else 0.0
            p10      = float(tft_out.loc[idx, f"pred_p10_{h_col}"]) if f"pred_p10_{h_col}" in tft_out.columns else 0.0
            p50      = float(tft_out.loc[idx, f"pred_p50_{h_col}"]) if f"pred_p50_{h_col}" in tft_out.columns else 0.0
            p90      = float(tft_out.loc[idx, f"pred_p90_{h_col}"]) if f"pred_p90_{h_col}" in tft_out.columns else 0.0

            # PatchTST outputs
            ptst_sig  = int(ptst_out.loc[idx, f"patchtst_signal_{h_col}"]) if f"patchtst_signal_{h_col}" in ptst_out.columns else 0
            ptst_conf = float(ptst_out.loc[idx, f"patchtst_conf_{h_col}"]) if f"patchtst_conf_{h_col}" in ptst_out.columns else 0.0

            # Ensemble — weighted average of each model's *signed confidence*.
            # tft_conf and ptst_conf are both calibrated directional confidences
            # in [0,1]; the signed vote is sign × confidence ∈ [-1,1]. Because the
            # regime weights sum to 1, the combination is a true weighted average
            # that stays in [-1,1] — so |combined| is a genuine directional
            # confidence comparable to the per-regime threshold (≈0.52).
            tft_vote  = tft_sig  * tft_conf
            ptst_vote = ptst_sig * ptst_conf
            combined  = w_tft * tft_vote + w_ptst * ptst_vote   # in [-1, 1]

            final_signal = int(np.sign(combined)) if abs(combined) > 0.05 else 0
            raw_conf     = min(abs(combined), 1.0)
            # Boost confidence when both models agree on a non-flat direction
            model_agree  = (tft_sig == ptst_sig) and (tft_sig != 0)
            if model_agree:
                raw_conf = min(raw_conf * 1.15, 1.0)

            packets.append(SignalPacket(
                timestamp   = idx,
                instrument  = self.instrument,
                horizon     = h_col,
                signal      = final_signal,
                confidence  = raw_conf,
                regime      = regime,
                regime_name = REGIME_NAMES.get(regime, "unknown"),
                model_agree = model_agree,
                tft_signal  = tft_sig,
                ptst_signal = ptst_sig,
                pred_p10    = p10,
                pred_p50    = p50,
                pred_p90    = p90,
            ).to_dict())

        result = pd.DataFrame(packets).set_index("timestamp")
        self._log_signal_summary(result)
        return result

    # ── Live single-bar predict ───────────────────────────────────────────────
    def predict_bar(
        self,
        bar_df:        pd.DataFrame,
        regime_signal: RegimeSignal,
        history_df:    pd.DataFrame | None = None,
    ) -> SignalPacket:
        """
        Generate a single SignalPacket for one incoming bar (live trading).

        Parameters
        ----------
        bar_df        : single-row feature DataFrame for current bar
        regime_signal : RegimeSignal from Agent 1
        history_df    : recent history needed for windowed models

        Returns
        -------
        SignalPacket — ready to pass to Risk Agent
        """
        self._check_fitted()

        regime = regime_signal.regime
        h      = self.primary_horizon

        # Need full context window for inference
        if history_df is not None:
            ctx_df = pd.concat([history_df, bar_df]).tail(
                max(self.tft.cfg.seq_len, self.ptst.cfg.seq_len) + max(HORIZONS) + 10
            )
        else:
            ctx_df = bar_df

        ctx_regimes = pd.Series(regime, index=ctx_df.index)

        # TFT signal
        try:
            tft_out = self.tft.predict(ctx_df, ctx_regimes, self.instrument_id)
            tft_sig  = int(tft_out[f"signal_{h}"].iloc[-1]) if f"signal_{h}" in tft_out.columns else 0
            tft_conf = float(tft_out[f"confidence_{h}"].iloc[-1]) if f"confidence_{h}" in tft_out.columns else 0.0
            p50      = float(tft_out[f"pred_p50_{h}"].iloc[-1]) if f"pred_p50_{h}" in tft_out.columns else 0.0
            p10      = float(tft_out[f"pred_p10_{h}"].iloc[-1]) if f"pred_p10_{h}" in tft_out.columns else 0.0
            p90      = float(tft_out[f"pred_p90_{h}"].iloc[-1]) if f"pred_p90_{h}" in tft_out.columns else 0.0
        except Exception as e:
            logger.warning("TFT live predict failed: %s", e)
            tft_sig, tft_conf, p50, p10, p90 = 0, 0.0, 0.0, 0.0, 0.0

        # PatchTST signal
        try:
            ptst_out  = self.ptst.predict(ctx_df)
            ptst_sig  = int(ptst_out[f"patchtst_signal_{h}"].iloc[-1]) if f"patchtst_signal_{h}" in ptst_out.columns else 0
            ptst_conf = float(ptst_out[f"patchtst_conf_{h}"].iloc[-1]) if f"patchtst_conf_{h}" in ptst_out.columns else 0.0
        except Exception as e:
            logger.warning("PatchTST live predict failed: %s", e)
            ptst_sig, ptst_conf = 0, 0.0

        # Ensemble — weighted average of signed confidences (see predict_batch)
        w_tft, w_ptst    = self.regime_weights.get(regime, (0.5, 0.5))
        combined         = w_tft * (tft_sig * tft_conf) + w_ptst * (ptst_sig * ptst_conf)
        final_signal     = int(np.sign(combined)) if abs(combined) > 0.05 else 0
        raw_conf         = min(abs(combined), 1.0)
        model_agree      = (tft_sig == ptst_sig) and (tft_sig != 0)
        if model_agree:
            raw_conf = min(raw_conf * 1.15, 1.0)

        return SignalPacket(
            timestamp   = bar_df.index[-1],
            instrument  = self.instrument,
            horizon     = h,
            signal      = final_signal,
            confidence  = raw_conf,
            regime      = regime,
            regime_name = REGIME_NAMES.get(regime, "unknown"),
            model_agree = model_agree,
            tft_signal  = tft_sig,
            ptst_signal = ptst_sig,
            pred_p10    = p10,
            pred_p50    = p50,
            pred_p90    = p90,
        )

    # ── Weight tuning ─────────────────────────────────────────────────────────
    def _tune_weights(
        self,
        val_df:      pd.DataFrame,
        val_regimes: pd.Series,
        val_labels:  pd.DataFrame | None,
    ) -> dict[int, tuple[float, float]]:
        """
        Per-regime grid search over TFT weight α ∈ {0.2, 0.35, 0.5, 0.65, 0.8}.
        Selects α that maximises hit ratio on validation set per regime.
        """
        if val_labels is None:
            return dict(DEFAULT_REGIME_WEIGHTS)

        h = self.primary_horizon
        label_col = f"label_{h}"
        if label_col not in val_labels.columns:
            return dict(DEFAULT_REGIME_WEIGHTS)

        try:
            tft_preds  = self.tft.predict(val_df, val_regimes, self.instrument_id)
            ptst_preds = self.ptst.predict(val_df)
        except Exception as e:
            logger.warning("Weight tuning failed: %s — using defaults", e)
            return dict(DEFAULT_REGIME_WEIGHTS)

        common = tft_preds.index.intersection(ptst_preds.index).intersection(val_labels.index)
        if len(common) < 50:
            return dict(DEFAULT_REGIME_WEIGHTS)

        true_labels = val_labels.loc[common, label_col]
        tft_sigs    = tft_preds.loc[common, f"signal_{h}"] if f"signal_{h}" in tft_preds.columns else pd.Series(0, index=common)
        ptst_sigs   = ptst_preds.loc[common, f"patchtst_signal_{h}"] if f"patchtst_signal_{h}" in ptst_preds.columns else pd.Series(0, index=common)
        reg_vals    = val_regimes.reindex(common).fillna(2).astype(int)

        tuned = {}
        alphas = [0.2, 0.35, 0.5, 0.65, 0.8]

        for regime_id in range(4):
            mask = reg_vals == regime_id
            if mask.sum() < 20:
                tuned[regime_id] = DEFAULT_REGIME_WEIGHTS[regime_id]
                continue

            t_sig   = tft_sigs[mask].values
            p_sig   = ptst_sigs[mask].values
            truth   = true_labels[mask].values

            best_alpha   = 0.5
            best_hit     = -1.0

            for alpha in alphas:
                score   = alpha * t_sig + (1 - alpha) * p_sig
                pred    = np.where(score > 0.1, 1, np.where(score < -0.1, -1, 0))
                active  = pred != 0
                if active.sum() < 5:
                    continue
                hit = (pred[active] == truth[active]).mean()
                if hit > best_hit:
                    best_hit   = hit
                    best_alpha = alpha

            tuned[regime_id] = (best_alpha, 1 - best_alpha)

        return tuned

    # ── Diagnostics ───────────────────────────────────────────────────────────
    def _log_signal_summary(self, result: pd.DataFrame) -> None:
        n      = len(result)
        buys   = (result["signal"] ==  1).sum()
        sells  = (result["signal"] == -1).sum()
        flats  = (result["signal"] ==  0).sum()
        agree  = result["model_agree"].mean()
        act    = result["is_actionable"].mean() if "is_actionable" in result.columns else 0.0
        logger.info(
            "Signal summary: BUY=%d (%.1f%%) | SELL=%d (%.1f%%) | FLAT=%d (%.1f%%) "
            "| agree=%.1f%% | actionable=%.1f%%",
            buys, buys/n*100, sells, sells/n*100, flats, flats/n*100,
            agree*100, act*100
        )

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, model_dir: str | Path) -> None:
        import pickle
        self._check_fitted()
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        self.tft.save(model_dir / "tft_signal.pt")
        self.ptst.save(model_dir / "patchtst_signal.pt")
        with open(model_dir / "signal_agent_meta.pkl", "wb") as f:
            pickle.dump({
                "instrument":      self.instrument,
                "primary_horizon": self.primary_horizon,
                "instrument_id":   self.instrument_id,
                "regime_weights":  self.regime_weights,
            }, f)
        logger.info("Signal Agent saved → %s", model_dir)

    @classmethod
    def load(cls, model_dir: str | Path) -> "SignalAgent":
        import pickle
        model_dir = Path(model_dir)
        with open(model_dir / "signal_agent_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        agent = cls(
            instrument      = meta["instrument"],
            primary_horizon = meta["primary_horizon"],
        )
        agent.instrument_id   = meta["instrument_id"]
        agent.regime_weights  = meta["regime_weights"]
        agent.tft  = TFTSignalModel.load(model_dir / "tft_signal.pt")
        agent.ptst = PatchTSTSignalModel.load(model_dir / "patchtst_signal.pt")
        agent.fitted = True
        logger.info("Signal Agent loaded from %s", model_dir)
        return agent

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("SignalAgent not fitted. Call .fit() first.")
