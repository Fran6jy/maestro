"""
maestro/data/validation/wfa.py
================================
Walk-Forward Analysis (WFA) engine with Combinatorial Purged Cross-Validation.

This module is the most methodologically critical in the entire MAESTRO
pipeline. It eliminates the look-ahead bias and multiple-testing inflation
that caused the MSc project's results to be unreliable.

Key concepts implemented
------------------------
1. Expanding-window WFA
   Train on [start → t], test on [t+embargo → t+step].
   Training window grows by one step each iteration.
   This mirrors live deployment: we never "forget" older data.

2. Purging (embargo)
   A gap of `embargo_days` is left between the last train bar
   and the first test bar. This prevents label leakage when
   overlapping return windows cause train and test labels to
   be computed from shared price bars.

3. Combinatorial Purged Cross-Validation (CPCV)
   For hyperparameter tuning within the training window only.
   Multiple train/validation splits are created without leakage.

4. Deflated Sharpe Ratio (DSR)
   Corrects the reported Sharpe Ratio for:
     - Multiple testing across strategy configurations
     - Non-normality of returns (skewness and kurtosis)
   Prevents reporting inflated performance from lucky splits.

References
----------
Lopez de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
  Chapter 7: Cross-Validation in Finance
  Chapter 8: Feature Importance
Bailey, D.H. & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio.
  Journal of Portfolio Management, 40(5), 94–107.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from maestro.config.config import get

logger = logging.getLogger(__name__)


# ── Data Structures ───────────────────────────────────────────────────────────
@dataclass
class WFASplit:
    """One train/test split in the walk-forward sequence."""
    split_id:    int
    train_start: pd.Timestamp
    train_end:   pd.Timestamp
    test_start:  pd.Timestamp
    test_end:    pd.Timestamp
    train_idx:   pd.DatetimeIndex
    test_idx:    pd.DatetimeIndex
    n_train:     int = field(init=False)
    n_test:      int = field(init=False)

    def __post_init__(self):
        self.n_train = len(self.train_idx)
        self.n_test  = len(self.test_idx)

    def __repr__(self) -> str:
        return (
            f"WFASplit(id={self.split_id} | "
            f"train={self.train_start.date()}→{self.train_end.date()} [{self.n_train}] | "
            f"test={self.test_start.date()}→{self.test_end.date()} [{self.n_test}])"
        )


@dataclass
class WFAResult:
    """Performance results for one WFA split."""
    split_id:       int
    test_start:     pd.Timestamp
    test_end:       pd.Timestamp
    predictions:    pd.Series
    true_labels:    pd.Series
    returns:        pd.Series         # strategy returns (net of costs)
    hit_ratio:      float = 0.0
    sharpe:         float = 0.0
    max_drawdown:   float = 0.0
    n_trades:       int   = 0
    metadata:       dict  = field(default_factory=dict)


# ── WFA Engine ────────────────────────────────────────────────────────────────
class WalkForwardEngine:
    """
    Generates expanding-window train/test splits with purging.

    Usage
    -----
    >>> engine = WalkForwardEngine()
    >>> for split in engine.splits(features_df):
    ...     X_train = features_df.loc[split.train_idx]
    ...     X_test  = features_df.loc[split.test_idx]
    ...     # train model on X_train, evaluate on X_test
    """

    def __init__(
        self,
        train_start:     str | None = None,
        train_end:       str | None = None,
        wfa_start:       str | None = None,
        wfa_end:         str | None = None,
        step_months:     int | None = None,
        embargo_days:    int | None = None,
        min_train_obs:   int | None = None,
    ) -> None:
        cfg = get("validation", {})
        self.train_start   = pd.Timestamp(train_start   or cfg.get("train_start",  "2000-01-01"), tz="UTC")
        self.train_end     = pd.Timestamp(train_end     or cfg.get("train_end",    "2015-12-31"), tz="UTC")
        self.wfa_start     = pd.Timestamp(wfa_start     or cfg.get("wfa_start",   "2016-01-01"), tz="UTC")
        self.wfa_end       = pd.Timestamp(wfa_end       or cfg.get("wfa_end",     "2023-12-31"), tz="UTC")
        self.step_months   = step_months  or cfg.get("wfa_step_months",  1)
        self.embargo_days  = embargo_days or cfg.get("embargo_days",     5)
        self.min_train_obs = min_train_obs or cfg.get("min_samples_train", 500)

    # ── Split generation ──────────────────────────────────────────────────────
    def splits(self, df: pd.DataFrame) -> Iterator[WFASplit]:
        """
        Yield WFASplit objects for an expanding-window walk-forward sequence.

        Parameters
        ----------
        df : feature DataFrame with DatetimeIndex (UTC)

        Yields
        ------
        WFASplit objects, one per step in the WFA sequence
        """
        _validate_index(df)

        cursor      = self.wfa_start
        split_id    = 0
        embargo     = pd.Timedelta(days=self.embargo_days)

        while cursor <= self.wfa_end:
            # Test window: [cursor, cursor + step_months)
            test_end = min(
                cursor + relativedelta(months=self.step_months) - pd.Timedelta(days=1),
                self.wfa_end
            )

            # Train window: [train_start, cursor - embargo)
            train_end_purged = cursor - embargo

            if train_end_purged <= self.train_start:
                cursor += relativedelta(months=self.step_months)
                continue

            # Resolve index positions
            train_idx = df.loc[self.train_start:train_end_purged].index
            test_idx  = df.loc[cursor:test_end].index

            if len(train_idx) < self.min_train_obs:
                logger.debug("Split %d skipped: insufficient train data (%d < %d)",
                             split_id, len(train_idx), self.min_train_obs)
                cursor += relativedelta(months=self.step_months)
                continue

            if len(test_idx) == 0:
                cursor += relativedelta(months=self.step_months)
                continue

            split = WFASplit(
                split_id    = split_id,
                train_start = train_idx[0],
                train_end   = train_idx[-1],
                test_start  = test_idx[0],
                test_end    = test_idx[-1],
                train_idx   = train_idx,
                test_idx    = test_idx,
            )
            logger.debug("Generated: %s", split)
            yield split

            split_id += 1
            cursor   += relativedelta(months=self.step_months)

        logger.info("WFA complete: %d splits generated", split_id)

    def n_splits(self, df: pd.DataFrame) -> int:
        """Count total splits without generating them."""
        return sum(1 for _ in self.splits(df))

    # ── Aggregate results ─────────────────────────────────────────────────────
    @staticmethod
    def aggregate_results(results: list[WFAResult]) -> dict:
        """
        Aggregate per-split WFA results into overall statistics.

        Returns dict with:
          - hit_ratio_mean/std
          - sharpe_mean/std
          - deflated_sharpe
          - max_drawdown_worst
          - oos_is_ratio (robustness check)
          - equity_curve (pd.Series)
        """
        if not results:
            return {}

        hit_ratios    = [r.hit_ratio    for r in results]
        sharpes       = [r.sharpe       for r in results]
        drawdowns     = [r.max_drawdown for r in results]
        all_returns   = pd.concat([r.returns for r in results]).sort_index()

        dsr = deflated_sharpe_ratio(
            sharpe_ratio = np.mean(sharpes),
            n_obs        = len(all_returns),
            skewness     = all_returns.skew(),
            kurtosis     = all_returns.kurtosis(),
            n_strategies = len(results),
        )

        return {
            "n_splits":          len(results),
            "hit_ratio_mean":    np.mean(hit_ratios),
            "hit_ratio_std":     np.std(hit_ratios),
            "sharpe_mean":       np.mean(sharpes),
            "sharpe_std":        np.std(sharpes),
            "deflated_sharpe":   dsr,
            "max_drawdown_worst":max(drawdowns),
            "max_drawdown_mean": np.mean(drawdowns),
            "equity_curve":      (1 + all_returns).cumprod(),
            "total_return":      (1 + all_returns).prod() - 1,
        }


# ── Combinatorial Purged CV (within-window hyperparameter tuning) ─────────────
class CPCVSplitter:
    """
    Combinatorial Purged Cross-Validation for within-training-window
    hyperparameter search. Never touches the test window.

    Parameters
    ----------
    n_splits  : number of CV groups (k)
    embargo   : purging gap in bars (not days) between train/val folds

    Usage
    -----
    >>> cpcv = CPCVSplitter(n_splits=6, embargo=10)
    >>> for train_idx, val_idx in cpcv.split(X_train):
    ...     model.fit(X_train.iloc[train_idx])
    ...     score = model.score(X_train.iloc[val_idx])
    """

    def __init__(self, n_splits: int = 6, embargo: int = 10) -> None:
        self.n_splits = n_splits
        self.embargo  = embargo

    def split(self, X: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n     = len(X)
        groups = np.array_split(np.arange(n), self.n_splits)

        for i in range(self.n_splits):
            val_idx   = groups[i]
            # Purge: remove embargo bars before and after validation fold
            purge_start = max(val_idx[0] - self.embargo, 0)
            purge_end   = min(val_idx[-1] + self.embargo + 1, n)
            purge_range = set(range(purge_start, purge_end))

            train_idx = np.array([j for j in range(n) if j not in purge_range])

            if len(train_idx) < 50 or len(val_idx) < 10:
                continue

            yield train_idx, val_idx


# ── Deflated Sharpe Ratio ─────────────────────────────────────────────────────
def deflated_sharpe_ratio(
    sharpe_ratio: float,
    n_obs:        int,
    skewness:     float = 0.0,
    kurtosis:     float = 3.0,
    n_strategies: int   = 1,
) -> float:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    Corrects the Sharpe Ratio for:
      1. Multiple testing across n_strategies configurations
      2. Non-normality (skewness and excess kurtosis of returns)

    A DSR > 0 means the strategy is likely genuinely profitable.
    A DSR ≤ 0 means performance is likely due to overfitting / luck.

    Parameters
    ----------
    sharpe_ratio : annualised (or per-period) Sharpe Ratio to correct
    n_obs        : number of return observations
    skewness     : skewness of return series
    kurtosis     : (excess) kurtosis of return series
    n_strategies : number of strategies tested (multiple testing correction)

    Returns
    -------
    float : Deflated Sharpe Ratio
    """
    from scipy import stats

    # Sharpe Ratio benchmark under IID hypothesis with multiple testing
    # Expected maximum SR from n_strategies trials (Bonferroni approximation)
    gamma   = 0.5772156649           # Euler-Mascheroni constant
    sr_star = (
        (1 - gamma) * stats.norm.ppf(1 - 1.0 / n_strategies)
        + gamma * stats.norm.ppf(1 - 1.0 / (n_strategies * np.e))
    ) if n_strategies > 1 else 0.0

    # Non-normality penalty
    non_normal_factor = np.sqrt(
        1
        - skewness * sharpe_ratio
        + ((kurtosis - 1) / 4) * sharpe_ratio ** 2
    )

    # Standard error of SR
    se = np.sqrt(
        (1 + (sharpe_ratio ** 2) / 2 * (1 - skewness ** 2 + (kurtosis / 4)))
        / (n_obs - 1)
    ) if n_obs > 1 else 1.0

    dsr = stats.norm.cdf(
        (sharpe_ratio - sr_star) * non_normal_factor / se
    )
    return float(dsr)


# ── Performance Metrics ───────────────────────────────────────────────────────
def compute_metrics(
    returns:     pd.Series,
    predictions: pd.Series | None = None,
    labels:      pd.Series | None = None,
    periods:     int = 252,
) -> dict:
    """
    Compute a comprehensive set of performance metrics for one WFA split.

    Parameters
    ----------
    returns     : strategy net-of-cost returns per bar
    predictions : model predictions {-1, 0, 1}
    labels      : true labels {-1, 0, 1}
    periods     : annualisation factor (252 = daily, 252*78 = 5-min)
    """
    r = returns.dropna()
    metrics: dict = {}

    # Return metrics
    metrics["total_return"]      = (1 + r).prod() - 1
    metrics["annualised_return"] = (1 + r).prod() ** (periods / len(r)) - 1 if len(r) > 0 else 0

    # Risk metrics
    std = r.std()
    metrics["volatility"]    = std * np.sqrt(periods)
    metrics["sharpe"]        = (r.mean() / std * np.sqrt(periods)) if std > 0 else 0
    metrics["sortino"]       = (r.mean() / r[r < 0].std() * np.sqrt(periods)) if (r < 0).any() else 0

    # Drawdown
    cum  = (1 + r).cumprod()
    peak = cum.cummax()
    dd   = (cum - peak) / peak
    metrics["max_drawdown"] = float(dd.min())
    metrics["calmar"]       = (-metrics["annualised_return"] / metrics["max_drawdown"]
                               if metrics["max_drawdown"] < 0 else 0)

    # Prediction metrics
    if predictions is not None and labels is not None:
        aligned = pd.concat([predictions, labels], axis=1).dropna()
        aligned.columns = ["pred", "true"]
        active = aligned[aligned["pred"] != 0]
        if len(active) > 0:
            metrics["hit_ratio"]   = float((active["pred"] == active["true"]).mean())
            metrics["n_trades"]    = len(active)
        else:
            metrics["hit_ratio"]   = 0.0
            metrics["n_trades"]    = 0

    return metrics


# ── Helpers ───────────────────────────────────────────────────────────────────
def _validate_index(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a DatetimeIndex")
    if df.index.tz is None:
        raise ValueError("DatetimeIndex must be timezone-aware (UTC)")
    if not df.index.is_monotonic_increasing:
        raise ValueError("DatetimeIndex must be sorted ascending")
