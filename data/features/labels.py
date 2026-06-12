"""
maestro/data/features/labels.py
=================================
Label generation using the Triple-Barrier Method (Lopez de Prado, 2018).

Why Triple-Barrier instead of simple next-bar return sign?
-----------------------------------------------------------
Simple labels (next bar > 0 → buy) are noisy and don't encode the
actual trade outcome (hit TP, hit SL, or expire). The triple-barrier
method creates labels that map directly to real trading outcomes.

The three barriers:
  - Upper barrier (take profit):  price rises by `pt_sl[0] × ATR`  → label +1
  - Lower barrier (stop loss):    price falls by `pt_sl[1] × ATR`  → label -1
  - Vertical barrier (time stop): max_hold bars elapse              → label 0

This gives a richer, more realistic target that trains models
to predict actual profitable events rather than noise.

References
----------
Lopez de Prado, M. (2018). Advances in Financial Machine Learning.
  Chapter 3: Labels. Wiley.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Triple Barrier Labeller ───────────────────────────────────────────────────
class TripleBarrierLabeller:
    """
    Generates trade labels via the triple-barrier method.

    Parameters
    ----------
    pt_sl       : (profit_target_multiplier, stop_loss_multiplier) as ATR multiples
    max_hold    : maximum number of bars to hold (vertical barrier)
    min_ret     : minimum absolute return to generate a non-zero label (filters noise)
    volatility_col : column name of ATR or vol estimate in feature DataFrame

    Usage
    -----
    >>> labeller = TripleBarrierLabeller(pt_sl=(2.0, 1.0), max_hold=10)
    >>> labels = labeller.fit(features_df)
    >>> labels["label"].value_counts()
    """

    def __init__(
        self,
        pt_sl: tuple[float, float] = (2.0, 1.0),
        max_hold: int = 10,
        min_ret: float = 0.0001,
        volatility_col: str = "atr",
    ) -> None:
        self.pt_sl          = pt_sl
        self.max_hold       = max_hold
        self.min_ret        = min_ret
        self.volatility_col = volatility_col

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply triple-barrier labelling to a feature DataFrame.

        Requires columns: close, {volatility_col} (e.g. atr)

        Returns
        -------
        pd.DataFrame with additional columns:
          - label:        {-1, 0, +1}
          - ret:          actual return over holding period
          - holding_bars: number of bars held
          - barrier_hit:  'tp' | 'sl' | 'vertical'
        """
        if self.volatility_col not in df.columns:
            raise ValueError(f"Column '{self.volatility_col}' not found. "
                             f"Run FeatureEngineer.transform() first.")

        close  = df["close"].values
        vol    = df[self.volatility_col].values
        n      = len(df)
        labels = np.zeros(n, dtype=int)
        rets   = np.zeros(n)
        holds  = np.zeros(n, dtype=int)
        hits   = np.empty(n, dtype=object)
        hits[:] = "vertical"

        pt_mult, sl_mult = self.pt_sl

        for i in range(n - 1):
            if np.isnan(vol[i]) or vol[i] == 0:
                continue

            entry = close[i]
            pt    = entry + pt_mult * vol[i]    # upper barrier
            sl    = entry - sl_mult * vol[i]    # lower barrier
            end   = min(i + self.max_hold, n - 1)

            for j in range(i + 1, end + 1):
                p = close[j]
                if p >= pt:
                    labels[i] = 1
                    rets[i]   = (p - entry) / entry
                    holds[i]  = j - i
                    hits[i]   = "tp"
                    break
                elif p <= sl:
                    labels[i] = -1
                    rets[i]   = (p - entry) / entry
                    holds[i]  = j - i
                    hits[i]   = "sl"
                    break
            else:
                # Vertical barrier hit
                p         = close[end]
                ret       = (p - entry) / entry
                rets[i]   = ret
                holds[i]  = end - i
                # Label 0 for flat; assign direction if large enough
                if abs(ret) >= self.min_ret:
                    labels[i] = int(np.sign(ret))

        result = df.copy()
        result["label"]        = labels
        result["ret"]          = rets
        result["holding_bars"] = holds
        result["barrier_hit"]  = hits

        # Drop last max_hold rows (labels unreliable near end of data)
        result = result.iloc[:-self.max_hold]

        label_counts = pd.Series(labels[:-self.max_hold]).value_counts().to_dict()
        logger.info(
            "Triple-barrier labels: %d total | +1: %d | 0: %d | -1: %d",
            len(result),
            label_counts.get(1, 0),
            label_counts.get(0, 0),
            label_counts.get(-1, 0),
        )
        return result


# ── Meta-label (for model confidence filtering) ───────────────────────────────
def add_meta_labels(
    df: pd.DataFrame,
    primary_signal_col: str,
    label_col: str = "label",
) -> pd.DataFrame:
    """
    Meta-labelling (Lopez de Prado, 2018, Chapter 4).

    The meta-label is 1 when the primary signal agrees with the true label,
    and 0 when it does not. A secondary model trained on meta-labels learns
    *when* to trust the primary signal — improving precision at the cost of recall.

    Parameters
    ----------
    primary_signal_col : column containing primary model predictions {-1, 0, 1}
    label_col          : column containing true triple-barrier labels

    Returns
    -------
    DataFrame with added column 'meta_label' ∈ {0, 1}
    """
    signal = df[primary_signal_col]
    label  = df[label_col]

    # Meta-label = 1 when signal is non-zero AND agrees with true label
    meta = ((signal != 0) & (signal == label)).astype(int)
    df["meta_label"] = meta

    precision = meta[signal != 0].mean() if (signal != 0).any() else 0
    logger.info(
        "Meta-labels added: %d positive (precision=%.3f)",
        meta.sum(), precision
    )
    return df


# ── Sample weights (return-based) ─────────────────────────────────────────────
def compute_sample_weights(
    df: pd.DataFrame,
    ret_col: str = "ret",
    method: str = "returns",
) -> pd.Series:
    """
    Compute sample weights to reduce the contribution of low-information events.

    Methods
    -------
    'returns'   : weight by absolute return (large moves more informative)
    'uniform'   : equal weights (baseline)
    'time_decay': recent observations get higher weight (exp decay)

    Returns
    -------
    pd.Series of weights, same index as df, normalised to sum to len(df)
    """
    n = len(df)

    if method == "returns":
        abs_rets = df[ret_col].abs().fillna(0)
        weights  = abs_rets / abs_rets.sum() * n

    elif method == "time_decay":
        # Exponential decay: most recent bar has weight 1.0
        decay  = np.exp(np.linspace(-1, 0, n))
        weights = pd.Series(decay / decay.sum() * n, index=df.index)

    else:  # uniform
        weights = pd.Series(np.ones(n), index=df.index)

    weights.name = "sample_weight"
    logger.info("Sample weights computed (method=%s): mean=%.4f std=%.4f",
                method, weights.mean(), weights.std())
    return weights
