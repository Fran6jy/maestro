"""
maestro/agents/risk/kelly.py
==============================
Fractional Kelly position sizing with regime-conditioned caps.

Why Kelly?
----------
Kelly Criterion (Kelly, 1956) gives the mathematically optimal
fraction of capital to bet to maximise long-run geometric growth.
For a binary bet with win probability p and win/loss ratio b:

    f* = (bp - q) / b      where q = 1 - p

For continuous returns (our case), the Kelly fraction is:

    f* = μ / σ²             where μ = mean return, σ² = variance

Problems with full Kelly in practice
--------------------------------------
1. Estimates of μ and σ are noisy → full Kelly often overbets
2. Full Kelly has very high short-term volatility
3. Drawdowns under full Kelly can be severe even when strategy is correct

MAESTRO uses fractional Kelly (25% of full Kelly) which:
  - Reduces bet size by 75% → dramatically smoother equity curve
  - Achieves ~75% of the long-run growth rate
  - Significantly reduces max drawdown

Regime conditioning
--------------------
The Kelly cap is further reduced based on:
  - Current regime (crisis → maximum 10% of capital)
  - Recent hit ratio (below breakeven → reduce)
  - Current drawdown (approaching limit → de-risk)
  - Signal confidence (low confidence → half-Kelly)

This implements the "proportional Kelly" concept where position size
is continuously adapted to current information quality.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Regime-conditioned Kelly caps (fraction of full Kelly allowed)
REGIME_KELLY_CAPS = {
    0: 0.25,   # bull_trend:  standard 25% fractional Kelly
    1: 0.25,   # bear_trend:  standard 25% fractional Kelly
    2: 0.20,   # sideways:    reduced (lower edge in range markets)
    3: 0.10,   # crisis:      minimal (high uncertainty, wide spreads)
}

# Maximum position as fraction of account equity per regime
REGIME_MAX_POSITION = {
    0: 0.20,   # bull:    up to 20% of equity
    1: 0.20,   # bear:    up to 20% of equity
    2: 0.15,   # sideways:up to 15% of equity
    3: 0.05,   # crisis:  maximum 5% of equity
}


@dataclass
class KellyResult:
    """Output of Kelly position sizing calculation."""
    recommended_units:  int
    kelly_fraction:     float   # f* before caps
    applied_fraction:   float   # f* after all caps and adjustments
    cap_reason:         str     # why the cap was applied
    max_units:          int     # account-based maximum
    edge_estimate:      float   # estimated edge (μ / σ²)
    is_trade:           bool    # True if edge is positive after costs


class KellyPositionSizer:
    """
    Fractional Kelly position sizer with regime-adaptive caps.

    Usage
    -----
    >>> sizer = KellyPositionSizer(account_equity=10_000)
    >>> result = sizer.size(
    ...     signal=1,
    ...     confidence=0.62,
    ...     regime=0,
    ...     recent_returns=recent_return_series,
    ...     current_drawdown=0.03,
    ...     price=1.0850,
    ... )
    >>> result.recommended_units
    12500
    """

    def __init__(
        self,
        account_equity:  float = 10_000.0,
        leverage:        float = 30.0,        # ESMA max retail leverage EUR/USD
        min_units:       int   = 1_000,
        max_units:       int   = 100_000,
        base_kelly_frac: float = 0.25,        # 25% fractional Kelly
        pip_size:        float = 0.0001,
    ) -> None:
        self.equity          = account_equity
        self.leverage        = leverage
        self.min_units       = min_units
        self.max_units       = min(max_units, int(account_equity * leverage))
        self.base_kelly_frac = base_kelly_frac
        self.pip_size        = pip_size

    # ── Main sizing function ──────────────────────────────────────────────────
    def size(
        self,
        signal:           int,           # {-1, 0, +1}
        confidence:       float,         # [0, 1] from SignalAgent
        regime:           int,           # {0,1,2,3} from RegimeAgent
        recent_returns:   pd.Series,     # last N strategy returns
        current_drawdown: float = 0.0,
        price:            float = 1.0,
        transaction_cost: float = 0.00008,
    ) -> KellyResult:
        """
        Compute recommended position size.

        Parameters
        ----------
        signal           : directional signal {-1=short, 0=flat, +1=long}
        confidence       : model confidence [0,1]
        regime           : current market regime
        recent_returns   : recent strategy returns (for μ/σ estimation)
        current_drawdown : current drawdown fraction [0,1]
        price            : current instrument price
        transaction_cost : one-way cost in log-return units

        Returns
        -------
        KellyResult with recommended_units and full breakdown
        """
        # Flat signal → zero position
        if signal == 0:
            return KellyResult(
                recommended_units=0, kelly_fraction=0.0, applied_fraction=0.0,
                cap_reason="flat_signal", max_units=self.max_units,
                edge_estimate=0.0, is_trade=False,
            )

        # ── Step 1: Estimate edge from recent returns ──────────────────────────
        ret = recent_returns.dropna()
        if len(ret) < 5:
            # Insufficient history → use minimum size
            edge    = confidence - 0.5   # crude edge estimate from confidence
            mu      = edge * 0.001
            sigma2  = 0.001 ** 2
        else:
            mu      = float(ret.mean())
            sigma2  = float(ret.var()) + 1e-10

        # Net edge after transaction costs
        net_mu   = mu - transaction_cost
        kelly_f  = net_mu / sigma2

        # Edge must be positive to trade
        if net_mu <= 0:
            return KellyResult(
                recommended_units=0, kelly_fraction=kelly_f, applied_fraction=0.0,
                cap_reason="negative_edge_after_costs", max_units=self.max_units,
                edge_estimate=kelly_f, is_trade=False,
            )

        # ── Step 2: Apply regime-conditioned Kelly cap ─────────────────────────
        regime_cap = REGIME_KELLY_CAPS.get(regime, self.base_kelly_frac)
        applied_f  = min(kelly_f, regime_cap)
        cap_reason = "regime_cap"

        if applied_f == kelly_f:
            cap_reason = "within_kelly"

        # ── Step 3: Confidence scaling ─────────────────────────────────────────
        # Below 0.55 confidence → scale down linearly to 50%
        if confidence < 0.55:
            conf_scale  = 0.5 + confidence   # maps [0, 0.55] → [0.5, 1.05]
            applied_f  *= min(conf_scale, 1.0)
            cap_reason   = "low_confidence"

        # ── Step 4: Drawdown-based de-risking ──────────────────────────────────
        # Above 8% drawdown → scale down position proportionally
        dd_threshold = 0.08
        if current_drawdown > dd_threshold:
            dd_scale    = 1.0 - (current_drawdown - dd_threshold) / (0.15 - dd_threshold)
            applied_f  *= max(0.1, dd_scale)
            cap_reason   = "drawdown_derisking"

        # ── Step 5: Account-based maximum position ────────────────────────────
        max_pos_frac = REGIME_MAX_POSITION.get(regime, 0.15)
        max_notional = self.equity * self.leverage * max_pos_frac
        max_units    = int(max_notional / price)
        max_units    = min(max_units, self.max_units)

        # ── Step 6: Convert fraction → units ──────────────────────────────────
        # Position size = Kelly fraction × account equity × leverage / price
        notional        = self.equity * self.leverage * applied_f
        raw_units       = int(notional / price)
        recommended     = max(self.min_units, min(raw_units, max_units))
        recommended    *= signal   # apply direction

        logger.debug(
            "Kelly sizing: signal=%+d | conf=%.3f | regime=%d | "
            "kelly_f=%.4f → applied_f=%.4f | units=%d | cap=%s",
            signal, confidence, regime, kelly_f, applied_f, recommended, cap_reason
        )

        return KellyResult(
            recommended_units = recommended,
            kelly_fraction    = kelly_f,
            applied_fraction  = applied_f,
            cap_reason        = cap_reason,
            max_units         = max_units,
            edge_estimate     = net_mu / np.sqrt(sigma2),   # Sharpe-like
            is_trade          = True,
        )

    # ── Update equity ─────────────────────────────────────────────────────────
    def update_equity(self, new_equity: float) -> None:
        """Update account equity after trades. Called after each trade fills."""
        self.equity    = new_equity
        self.max_units = int(new_equity * self.leverage)

    # ── Portfolio-level sizing ────────────────────────────────────────────────
    def size_portfolio(
        self,
        signals:    dict[str, int],       # {instrument: signal}
        confs:      dict[str, float],
        regime:     int,
        recent_rets:dict[str, pd.Series],
        prices:     dict[str, float],
        current_drawdown: float = 0.0,
    ) -> dict[str, KellyResult]:
        """
        Size all instruments simultaneously with correlation adjustment.

        For EUR/USD + GBP/USD, apply a correlation haircut since both
        are USD pairs and tend to move together (correlation ~0.7-0.85).
        Running both at full Kelly would overdiversify risk.
        """
        results   = {}
        n_active  = sum(1 for s in signals.values() if s != 0)

        # Correlation haircut: reduce each position by √(1/n_active) factor
        # This is a simplified version of the Markowitz optimal sizing
        corr_factor = 1.0 / np.sqrt(max(n_active, 1))

        for instr, sig in signals.items():
            ret = recent_rets.get(instr, pd.Series(dtype=float))
            p   = prices.get(instr, 1.0)
            c   = confs.get(instr, 0.5)

            result = self.size(sig, c, regime, ret, current_drawdown, p)

            # Apply correlation haircut to correlated pairs
            if n_active > 1 and result.is_trade:
                adjusted_units = int(result.recommended_units * corr_factor)
                result = KellyResult(
                    recommended_units = adjusted_units,
                    kelly_fraction    = result.kelly_fraction,
                    applied_fraction  = result.applied_fraction * corr_factor,
                    cap_reason        = f"{result.cap_reason}+correlation",
                    max_units         = result.max_units,
                    edge_estimate     = result.edge_estimate,
                    is_trade          = abs(adjusted_units) >= self.min_units,
                )

            results[instr] = result

        return results
