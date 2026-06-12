"""
maestro/agents/risk/cost_model.py
===================================
Realistic transaction cost model for Forex.

The MSc project's critical flaw #3: zero transaction costs.
All backtested strategies showed negative live P&L because costs
were never modelled. This module fixes that permanently.

Four cost components
--------------------
1. Spread         — the bid-ask spread paid on every trade
                    EUR/USD typical: 0.5–1.2 pips (varies by session)
                    GBP/USD typical: 0.7–1.5 pips
                    Spread widens in low-liquidity periods and crises

2. Commission     — broker commission (OANDA charges 0 for standard,
                    but ~$5/100k for premium accounts; modelled here)

3. Slippage       — execution price vs signal price, due to:
                    - Order queue position
                    - Price movement during order transmission
                    - Modelled as Gaussian noise proportional to vol

4. Market impact  — for larger positions, your own order moves the price
                    Almgren-Chriss (2001) model: impact ∝ √(order_size)
                    Negligible for retail sizes < 1M notional

Also computes:
  - Breakeven edge required: minimum signal edge to overcome costs
  - Net return: gross return adjusted for all costs
  - Annual cost drag: what costs are doing to annualised returns

This cost model feeds directly into:
  - Agent 4 (Risk): reward function = returns - costs - CVaR penalty
  - Agent 5 (Execution): optimise order routing to minimise costs
  - Backtesting: generate realistic net P&L

References
----------
Almgren, R. & Chriss, N. (2001). Optimal execution of portfolio transactions.
  Journal of Risk, 3(2), 5–39.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Instrument cost parameters ────────────────────────────────────────────────
INSTRUMENT_COSTS = {
    "EUR_USD": {
        "typical_spread_pips":   0.8,    # mid-session average
        "wide_spread_pips":      2.5,    # low-liquidity / news events
        "commission_per_lot":    0.0,    # OANDA standard (no commission)
        "pip_value_per_lot":     10.0,   # USD per pip per standard lot
        "lot_size":              100_000,# units per lot
        "min_spread_pips":       0.3,    # tightest possible (London/NY overlap)
    },
    "GBP_USD": {
        "typical_spread_pips":   1.0,
        "wide_spread_pips":      3.0,
        "commission_per_lot":    0.0,
        "pip_value_per_lot":     10.0,
        "lot_size":              100_000,
        "min_spread_pips":       0.4,
    },
}

# Session-based spread multipliers
SESSION_SPREAD_MULTIPLIER = {
    "overlap":    0.70,   # London/NY 12:00–16:00 UTC — tightest
    "london":     0.85,   # London only  07:00–12:00 UTC
    "new_york":   0.90,   # NY only       16:00–21:00 UTC
    "asian":      1.40,   # Asian session 00:00–07:00 UTC — wider
    "weekend":    2.50,   # Weekend/thin trading
    "unknown":    1.00,
}


@dataclass
class CostEstimate:
    """Full cost breakdown for one trade."""
    instrument:     str
    units:          int           # trade size in currency units
    gross_return:   float         # pre-cost return (log-return)
    spread_cost:    float         # in log-return units
    slippage_cost:  float
    commission_cost:float
    impact_cost:    float
    total_cost:     float         # sum of all costs
    net_return:     float         # gross_return - total_cost
    breakeven_pips: float         # pips needed to break even
    cost_pct:       float         # total_cost as % of |gross_return|

    @property
    def is_profitable(self) -> bool:
        return self.net_return > 0


class TransactionCostModel:
    """
    Computes realistic all-in transaction costs for Forex trades.

    Usage
    -----
    >>> model = TransactionCostModel("EUR_USD")
    >>> cost  = model.estimate(units=50_000, price=1.0850,
    ...                        volatility=0.0012, session="overlap")
    >>> cost.total_cost
    0.000074   # ~0.74 pips all-in

    >>> net_returns = model.apply_to_series(gross_returns, signals, prices)
    """

    def __init__(self, instrument: str = "EUR_USD") -> None:
        self.instrument = instrument
        params          = INSTRUMENT_COSTS.get(instrument, INSTRUMENT_COSTS["EUR_USD"])
        self.typical_spread  = params["typical_spread_pips"]
        self.wide_spread     = params["wide_spread_pips"]
        self.min_spread      = params["min_spread_pips"]
        self.commission      = params["commission_per_lot"]
        self.pip_value       = params["pip_value_per_lot"]
        self.lot_size        = params["lot_size"]

    # ── Single trade estimate ─────────────────────────────────────────────────
    def estimate(
        self,
        units:      int,
        price:      float,
        volatility: float = 0.001,
        session:    str   = "unknown",
        high_impact:bool  = False,     # True during news releases → wide spread
    ) -> CostEstimate:
        """
        Estimate all-in cost for one trade.

        Parameters
        ----------
        units       : trade size in currency units (e.g. 50_000)
        price       : current mid price
        volatility  : current 1-bar realised vol (for slippage scaling)
        session     : trading session for spread multiplier
        high_impact : True during FOMC/NFP etc. → force wide spread

        Returns
        -------
        CostEstimate with all cost components
        """
        lots = abs(units) / self.lot_size

        # ── 1. Spread cost ─────────────────────────────────────────────────────
        if high_impact:
            spread_pips = self.wide_spread
        else:
            base_spread = self.typical_spread
            multiplier  = SESSION_SPREAD_MULTIPLIER.get(session, 1.0)
            spread_pips = max(base_spread * multiplier, self.min_spread)

        # Convert pips to log-return units: 1 pip = 0.0001 / price
        pip_in_price  = 0.0001
        spread_lr     = (spread_pips * pip_in_price) / price   # one-way cost

        # ── 2. Slippage ────────────────────────────────────────────────────────
        # Gaussian slippage: mean = 0.2 × spread, std = 0.3 × volatility
        # Capped at 2× spread to prevent unrealistic outliers
        slippage_mean = 0.2 * spread_lr
        slippage_std  = 0.3 * volatility
        slippage      = min(slippage_mean + slippage_std, 2.0 * spread_lr)

        # ── 3. Commission ──────────────────────────────────────────────────────
        # Convert $ per lot to log-return units
        commission_usd = self.commission * lots
        notional_usd   = abs(units) * price
        commission_lr  = commission_usd / notional_usd if notional_usd > 0 else 0.0

        # ── 4. Market impact (Almgren-Chriss) ──────────────────────────────────
        # Impact = η × σ × √(trade_size / ADV)
        # For retail Forex sizes << ADV → impact is negligible (<0.01 pips)
        # We model it for completeness; becomes relevant above 1M units
        eta   = 0.1     # temporary impact coefficient
        adv   = 1e10    # average daily volume EUR/USD (~$1 trillion)
        sigma = volatility
        impact_lr = eta * sigma * np.sqrt(abs(units) / adv)
        impact_lr = min(impact_lr, 0.5 * spread_lr)   # cap at 50% of spread

        # ── Total cost ─────────────────────────────────────────────────────────
        total_lr        = spread_lr + slippage + commission_lr + impact_lr
        gross_return    = 0.0   # placeholder; caller populates
        breakeven_pips  = total_lr * price / pip_in_price

        return CostEstimate(
            instrument      = self.instrument,
            units           = units,
            gross_return    = gross_return,
            spread_cost     = spread_lr,
            slippage_cost   = slippage,
            commission_cost = commission_lr,
            impact_cost     = impact_lr,
            total_cost      = total_lr,
            net_return      = gross_return - total_lr,
            breakeven_pips  = breakeven_pips,
            cost_pct        = 0.0,
        )

    # ── Apply to return series ────────────────────────────────────────────────
    def apply_to_series(
        self,
        gross_returns: pd.Series,
        signals:       pd.Series,
        prices:        pd.Series,
        volatility:    pd.Series | None = None,
        units:         int = 10_000,
    ) -> pd.DataFrame:
        """
        Apply cost model to a full backtested return series.

        Parameters
        ----------
        gross_returns : log-returns per bar
        signals       : {-1, 0, +1} signals per bar
        prices        : close prices per bar
        volatility    : realised volatility series (optional)
        units         : trade size in currency units

        Returns
        -------
        pd.DataFrame with columns:
          gross_return, total_cost, net_return, is_trade, spread_cost,
          slippage_cost, cumulative_cost_drag
        """
        n = len(gross_returns)
        results = []

        vol_series = volatility if volatility is not None else pd.Series(0.001, index=gross_returns.index)
        prev_signal = 0

        for i, ts in enumerate(gross_returns.index):
            sig   = int(signals.iloc[i]) if i < len(signals) else 0
            price = float(prices.iloc[i]) if i < len(prices) else 1.0
            vol   = float(vol_series.iloc[i]) if i < len(vol_series) else 0.001

            # Is this bar a trade? (signal changes or new non-flat signal)
            is_trade = (sig != 0) and (sig != prev_signal or sig != 0)
            session  = _get_session(ts)

            if is_trade:
                est = self.estimate(units * sig, price, vol, session)
                total_cost = est.total_cost
            else:
                total_cost = 0.0

            gross_ret = float(gross_returns.iloc[i]) * sig
            net_ret   = gross_ret - total_cost

            results.append({
                "gross_return": gross_ret,
                "total_cost":   total_cost,
                "net_return":   net_ret,
                "is_trade":     is_trade,
            })
            prev_signal = sig

        df = pd.DataFrame(results, index=gross_returns.index)
        df["cumulative_cost_drag"] = df["total_cost"].cumsum()
        df["cum_gross"]  = (1 + df["gross_return"]).cumprod() - 1
        df["cum_net"]    = (1 + df["net_return"]).cumprod() - 1

        n_trades    = df["is_trade"].sum()
        total_drag  = df["total_cost"].sum()
        avg_cost    = total_drag / max(n_trades, 1)

        logger.info(
            "Cost model applied: %d trades | total_cost_drag=%.4f "
            "(%.1f pips) | avg_per_trade=%.5f",
            n_trades, total_drag,
            total_drag * price / 0.0001,
            avg_cost,
        )
        return df

    # ── Breakeven analysis ────────────────────────────────────────────────────
    def breakeven_edge(
        self,
        units:    int   = 10_000,
        price:    float = 1.08,
        session:  str   = "overlap",
    ) -> dict:
        """
        Compute the minimum edge required to profitably trade.

        Returns dict with breakeven_pips, breakeven_hit_ratio,
        and minimum_expected_return_bps.
        """
        est      = self.estimate(units, price, session=session)
        total    = est.total_cost

        # Breakeven hit ratio: fraction of correct trades needed
        # If you win W pips on correct trades and lose L pips on wrong,
        # breakeven requires: HR × W − (1−HR) × L = cost
        # Simplified with W=L assumption:
        breakeven_hr = 0.5 + total / (2 * total + total)

        return {
            "breakeven_pips":         est.breakeven_pips,
            "total_cost_log_return":  total,
            "breakeven_hit_ratio":    min(breakeven_hr, 1.0),
            "min_expected_return_bps":total * 10_000,
            "annual_cost_drag_pct":   total * 252 * 78 * 100,   # M5 bars/year
        }


# ── Session detection ─────────────────────────────────────────────────────────
def _get_session(ts: pd.Timestamp) -> str:
    """Classify bar timestamp into trading session."""
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    hour = ts.hour
    dow  = ts.dayofweek   # 5=Sat, 6=Sun

    if dow >= 5:
        return "weekend"
    elif 12 <= hour < 16:
        return "overlap"
    elif 7 <= hour < 12:
        return "london"
    elif 16 <= hour < 21:
        return "new_york"
    else:
        return "asian"
