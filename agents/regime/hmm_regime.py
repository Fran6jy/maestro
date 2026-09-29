"""
maestro/agents/regime/hmm_regime.py
=====================================
Hidden Markov Model (HMM) for market regime detection.

Why HMM?
--------
Financial markets switch between latent states (regimes) that are
not directly observable — only their effects on price/vol/returns are.
HMMs model exactly this: a sequence of hidden states that emit
observable features, with transitions governed by a stochastic matrix.

The 4 regimes we learn
-----------------------
0 — Low-Vol Trend (Bull)    : rising prices, low volatility, trending
1 — High-Vol Trend (Bear)   : falling prices, elevated vol, trending
2 — Sideways / Mean-Revert  : low directional movement, range-bound
3 — Crisis / Shock          : extreme volatility, fat-tailed moves

These labels are soft-assigned post-hoc by inspecting each state's
mean return and volatility — HMM learns the states unsupervised.

Architecture
------------
- GaussianHMM with full covariance matrices
- Input features: log-return, realised vol, VIX z-score, yield curve,
                  bar range, volume z-score (6 dimensions)
- Trained on 2000–2015 data; evaluated on 2016–2023 WFA windows
- Outputs: state sequence + state probabilities (soft assignment)

Dependencies: hmmlearn, numpy, pandas, scikit-learn
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Regime label mapping (assigned post-hoc from state statistics)
REGIME_NAMES = {
    0: "bull_trend",
    1: "bear_trend",
    2: "sideways",
    3: "crisis",
}

# Features used by the HMM (must exist in feature DataFrame)
HMM_FEATURES = [
    "log_return_1",      # direction signal
    "vol_realised",      # volatility level
    "vix_zscore",        # macro vol signal (from FRED)
    "yield_curve_inverted",  # recession indicator
    "bar_range_norm",    # intrabar volatility
    "volume_zscore",     # activity level
]


@dataclass
class HMMConfig:
    n_states:        int   = 4          # number of hidden states
    n_iter:          int   = 200        # EM training iterations
    tol:             float = 1e-4       # convergence tolerance
    covariance_type: str   = "full"     # full | diag | tied | spherical
    random_state:    int   = 42
    min_train_bars:  int   = 500        # minimum bars to fit


class HMMRegimeDetector:
    """
    Gaussian HMM-based market regime detector.

    Usage
    -----
    >>> detector = HMMRegimeDetector()
    >>> detector.fit(train_features_df)
    >>> regimes = detector.predict(test_features_df)
    >>> probs   = detector.predict_proba(test_features_df)

    Output regimes
    --------------
    pd.Series of int {0,1,2,3} — mapped to REGIME_NAMES after label assignment.
    """

    def __init__(self, config: HMMConfig | None = None) -> None:
        self.cfg     = config or HMMConfig()
        self.model   = None
        self.scaler  = StandardScaler()
        self.fitted  = False
        self._state_to_regime: dict[int, int] = {}   # HMM state → canonical regime id
        self._regime_stats: dict[int, dict]   = {}   # per-state statistics

    # ── Fit ───────────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> "HMMRegimeDetector":
        """
        Fit the HMM on historical feature data.

        Parameters
        ----------
        df : feature DataFrame containing HMM_FEATURES columns.
             Missing macro columns (vix_zscore, yield_curve_inverted)
             are zero-filled with a warning — model still runs.

        Returns self for chaining.
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            raise ImportError("Install hmmlearn: pip install hmmlearn")

        X, lengths = self._prepare(df)

        if X.shape[0] < self.cfg.min_train_bars:
            raise ValueError(
                f"Need at least {self.cfg.min_train_bars} bars to fit HMM, "
                f"got {X.shape[0]}"
            )

        logger.info(
            "Fitting GaussianHMM: %d states | %d bars | %d features",
            self.cfg.n_states, X.shape[0], X.shape[1]
        )

        # Try full covariance first; fall back to diagonal if the matrix
        # becomes numerically singular (common with correlated financial features).
        for cov_type in [self.cfg.covariance_type, "diag"]:
            try:
                self.model = GaussianHMM(
                    n_components     = self.cfg.n_states,
                    covariance_type  = cov_type,
                    n_iter           = self.cfg.n_iter,
                    tol              = self.cfg.tol,
                    random_state     = self.cfg.random_state,
                    min_covar        = 1e-3,
                    verbose          = False,
                )
                self.model.fit(X, lengths)
                if cov_type != self.cfg.covariance_type:
                    logger.warning("HMM fell back to covariance_type='%s'", cov_type)
                break
            except (ValueError, np.linalg.LinAlgError) as exc:
                if cov_type == "diag":
                    raise
                logger.warning("HMM covariance_type='%s' failed (%s), retrying with 'diag'", cov_type, exc)
        self.fitted = True

        # Assign canonical regime labels based on state statistics
        self._assign_regime_labels(df, X)

        logger.info(
            "HMM fitted. Log-likelihood: %.2f | State mapping: %s",
            self.model.score(X), self._state_to_regime
        )
        return self

    # ── Predict ───────────────────────────────────────────────────────────────
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict regime for each bar.

        Returns
        -------
        pd.Series[int] with values in {0,1,2,3} mapped to REGIME_NAMES.
        Index matches df.index.
        """
        self._check_fitted()
        X, _ = self._prepare(df)
        raw_states = self.model.predict(X)
        regimes    = np.array([self._state_to_regime.get(s, s) for s in raw_states])
        return pd.Series(regimes, index=df.index, name="hmm_regime", dtype=int)

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict posterior state probabilities for each bar.

        Returns
        -------
        pd.DataFrame[float] shape (n_bars, n_states)
        Columns: regime_{name} (e.g. 'regime_bull_trend')
        """
        self._check_fitted()
        X, _ = self._prepare(df)
        _, posteriors = self.model.score_samples(X)

        # Reorder columns to canonical regime order
        cols = {
            self._state_to_regime.get(s, s): f"hmm_prob_{REGIME_NAMES[self._state_to_regime.get(s,s)]}"
            for s in range(self.cfg.n_states)
        }
        proba_df = pd.DataFrame(
            posteriors,
            index   = df.index,
            columns = [cols[s] for s in sorted(cols.keys())]
        )
        return proba_df

    def predict_with_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: returns regimes + probabilities in one DataFrame."""
        regimes = self.predict(df).rename("hmm_regime")
        probas  = self.predict_proba(df)
        return pd.concat([regimes, probas], axis=1)

    # ── Label assignment ──────────────────────────────────────────────────────
    def _assign_regime_labels(self, df: pd.DataFrame, X: np.ndarray) -> None:
        """
        Map HMM states → canonical regime IDs using state statistics.

        Logic
        -----
        State with highest mean return   → bull_trend  (0)
        State with lowest mean return    → bear_trend  (1)
        State with highest vol           → crisis      (3)
        Remaining state                  → sideways    (2)
        """
        raw_states = self.model.predict(X)
        ret_col    = HMM_FEATURES.index("log_return_1")
        vol_col    = HMM_FEATURES.index("vol_realised")

        state_means = {}
        state_vols  = {}
        for s in range(self.cfg.n_states):
            mask = raw_states == s
            if mask.sum() == 0:
                # Degenerate state with no assigned bars (happens with the
                # 'diag' covariance fallback). Record zeroed stats so every
                # state is present in _regime_stats — the label-assignment
                # log loop and regime_summary() index all states.
                state_means[s] = 0.0
                state_vols[s]  = 0.0
                self._regime_stats[s] = {
                    "mean_return": 0.0, "mean_vol": 0.0,
                    "n_bars": 0, "pct_bars": 0.0,
                }
            else:
                state_means[s] = float(X[mask, ret_col].mean())
                state_vols[s]  = float(X[mask, vol_col].mean())
                self._regime_stats[s] = {
                    "mean_return": state_means[s],
                    "mean_vol":    state_vols[s],
                    "n_bars":      int(mask.sum()),
                    "pct_bars":    float(mask.sum() / len(raw_states)),
                }

        # Sort states by mean return (desc) and volatility (desc)
        by_return = sorted(state_means, key=state_means.get, reverse=True)
        by_vol    = sorted(state_vols,  key=state_vols.get,  reverse=True)

        crisis_state = by_vol[0]                                      # highest vol → crisis
        bull_state   = by_return[0] if by_return[0] != crisis_state else by_return[1]
        bear_state   = by_return[-1] if by_return[-1] != crisis_state else by_return[-2]
        sideways_state = next(
            s for s in range(self.cfg.n_states)
            if s not in {crisis_state, bull_state, bear_state}
        )

        self._state_to_regime = {
            bull_state:     0,   # bull_trend
            bear_state:     1,   # bear_trend
            sideways_state: 2,   # sideways
            crisis_state:   3,   # crisis
        }

        for state, regime in self._state_to_regime.items():
            logger.info(
                "  HMM state %d → %s | ret=%.5f | vol=%.4f | pct=%.1f%%",
                state, REGIME_NAMES[regime],
                state_means[state], state_vols[state],
                self._regime_stats[state]["pct_bars"] * 100,
            )

    # ── Feature preparation ───────────────────────────────────────────────────
    def _prepare(self, df: pd.DataFrame) -> tuple[np.ndarray, list[int]]:
        """Extract, fill, scale HMM features. Returns (X, lengths)."""
        available = [f for f in HMM_FEATURES if f in df.columns]
        missing   = [f for f in HMM_FEATURES if f not in df.columns]

        if missing:
            logger.warning(
                "HMM: missing features filled with 0: %s. "
                "Ensure macro data (FRED) is joined before fitting.", missing
            )

        X_df = df[available].copy()
        for col in missing:
            X_df[col] = 0.0
        X_df = X_df[HMM_FEATURES]    # enforce column order

        # Fill NaN with column medians (don't drop rows — preserves sequence)
        X_df = X_df.fillna(X_df.median())
        X    = X_df.values.astype(np.float64)

        # Fit scaler only during training (fit=True handled externally via fitted flag)
        if not self.fitted:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        return X, [len(X)]

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        """Pickle the fitted model + scaler + label mapping."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model":            self.model,
                "scaler":           self.scaler,
                "state_to_regime":  self._state_to_regime,
                "regime_stats":     self._regime_stats,
                "config":           self.cfg,
            }, f)
        logger.info("HMM saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "HMMRegimeDetector":
        """Load a saved HMM detector."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(config=data["config"])
        obj.model              = data["model"]
        obj.scaler             = data["scaler"]
        obj._state_to_regime   = data["state_to_regime"]
        obj._regime_stats      = data["regime_stats"]
        obj.fitted             = True
        logger.info("HMM loaded from %s", path)
        return obj

    # ── Utils ─────────────────────────────────────────────────────────────────
    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("HMM not fitted. Call .fit(df) first.")

    def regime_summary(self) -> pd.DataFrame:
        """Return a DataFrame summarising each regime's statistics."""
        rows = []
        for state, stats in self._regime_stats.items():
            regime_id = self._state_to_regime.get(state, state)
            rows.append({
                "regime_id":   regime_id,
                "regime_name": REGIME_NAMES.get(regime_id, "unknown"),
                "hmm_state":   state,
                **stats,
            })
        return pd.DataFrame(rows).sort_values("regime_id").reset_index(drop=True)

    def transition_matrix(self) -> pd.DataFrame:
        """Return the HMM transition probability matrix with regime labels."""
        self._check_fitted()
        names = [REGIME_NAMES[self._state_to_regime[s]]
                 for s in range(self.cfg.n_states)]
        return pd.DataFrame(
            self.model.transmat_,
            index   = names,
            columns = names,
        ).round(4)
