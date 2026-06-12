"""
maestro/agents/risk/cvar_env.py
=================================
CVaR-aware Reinforcement Learning trading environment.

This is a custom Gymnasium environment that trains a PPO agent to
manage position sizing with explicit tail-risk constraints.

Why RL for position sizing?
----------------------------
Traditional approaches (fixed fractional, Kelly) are static — they
don't adapt to changing market conditions, regime transitions, or
sequences of losses. RL learns a dynamic policy that:
  - Increases size after wins in high-confidence regimes
  - Cuts size after drawdown sequences (loss aversion)
  - Goes flat in crisis/uncertain conditions
  - Explicitly penalises tail events (CVaR penalty in reward)

The CVaR reward shaping
------------------------
Standard RL reward = net return per step.
MAESTRO reward = net_return - λ_CVaR × CVaR₉₅ - λ_DD × drawdown_excess

Where:
  CVaR₉₅  = Expected loss in the worst 5% of outcomes (tail risk)
  λ_CVaR  = 0.5 (penalises tail events)
  λ_DD    = 1.0 (penalises drawdown beyond 10% threshold)

This trains the agent to actively avoid the fat-tailed loss
distributions typical of unmanaged Forex strategies.

State space (what the RL agent observes)
-----------------------------------------
  [0]    current portfolio return (normalised)
  [1]    current drawdown
  [2]    realised volatility (20-bar)
  [3]    VaR₉₅ estimate (rolling)
  [4]    current regime (0-3, normalised)
  [5]    regime confidence
  [6]    signal confidence from Agent 2
  [7]    sentiment fusion weight from Agent 3
  [8]    recent hit ratio (last 20 trades)
  [9]    current position fraction (-1 to +1)
  [10]   bars since last trade
  [11]   current CVaR (rolling 20-bar)
  [12]   spread cost (current session)
  [13-15] last 3 bar returns (momentum)

Action space
------------
  Continuous: [-1, +1] → target position fraction
    -1 = full short (max_units short)
     0 = flat
    +1 = full long (max_units long)

  Final units = action × max_units × kelly_fraction

References
----------
Tamar, A. et al. (2015). Policy gradient for coherent risk measures.
  NeurIPS 2015.
Chow, Y. et al. (2017). Risk-Constrained Reinforcement Learning with
  Percentile Risk Criteria. JMLR 18(167).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OBS_DIM     = 16     # state space dimension
MAX_UNITS   = 100_000
KELLY_CAP   = 0.25   # maximum Kelly fraction allowed


@dataclass
class EnvConfig:
    max_units:         int   = MAX_UNITS
    kelly_cap:         float = KELLY_CAP
    lambda_cvar:       float = 0.5        # CVaR penalty weight
    lambda_drawdown:   float = 1.0        # drawdown penalty weight
    drawdown_threshold:float = 0.10       # drawdown beyond this → penalty
    cvar_window:       int   = 20         # rolling CVaR window
    cvar_alpha:        float = 0.95       # CVaR confidence level
    circuit_breaker_dd:float = 0.15       # force flat if DD > 15%
    max_hold_bars:     int   = 24         # max bars to hold a position
    transaction_cost:  float = 0.00008   # ~0.8 pip per trade (log-return)
    reward_scale:      float = 100.0      # scale rewards for RL stability


class CVaRTradingEnv:
    """
    CVaR-aware Forex trading environment for PPO/SAC training.

    Implements the Gymnasium interface (step/reset/observation_space/action_space)
    without inheriting from gymnasium.Env to avoid the hard dependency.
    Import gymnasium only when actually training.

    Usage
    -----
    >>> env = CVaRTradingEnv(features_df, signals_df)
    >>> obs, info = env.reset()
    >>> for _ in range(1000):
    ...     action = policy(obs)
    ...     obs, reward, done, truncated, info = env.step(action)
    """

    def __init__(
        self,
        features_df:  pd.DataFrame,
        signals_df:   pd.DataFrame,
        config:       EnvConfig | None = None,
    ) -> None:
        """
        Parameters
        ----------
        features_df : feature DataFrame with OHLCV + technical features
        signals_df  : output of SignalAgent.predict_batch() — provides
                      signal, confidence, regime, regime_confidence cols
        config      : EnvConfig with RL hyperparameters
        """
        self.cfg         = config or EnvConfig()
        self.features    = features_df
        self.signals     = signals_df.reindex(features_df.index)

        # Align returns
        self.returns     = features_df["log_return_1"].fillna(0).values
        self.prices      = features_df["close"].values
        self.vol         = features_df.get("vol_realised",
                           pd.Series(0.001, index=features_df.index)).fillna(0.001).values
        self.vix         = features_df.get("vix_zscore",
                           pd.Series(0.0, index=features_df.index)).fillna(0).values

        self.n           = len(features_df)
        self._t          = 0
        self._reset_state()

        # Gymnasium-compatible spaces (defined as dicts for portability)
        self.observation_space_shape = (OBS_DIM,)
        self.action_space_low        = np.array([-1.0], dtype=np.float32)
        self.action_space_high       = np.array([ 1.0], dtype=np.float32)

    # ── Gymnasium interface ───────────────────────────────────────────────────
    def reset(
        self,
        seed: int | None = None,
        start_idx: int | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset environment to start of episode."""
        if seed is not None:
            np.random.seed(seed)
        self._t = start_idx if start_idx is not None else 0
        self._reset_state()
        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step.

        Parameters
        ----------
        action : np.ndarray shape (1,) in [-1, +1]

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        if self._t >= self.n - 1:
            return self._get_obs(), 0.0, True, False, {}

        position_fraction = float(np.clip(action[0], -1.0, 1.0))

        # Apply circuit breaker
        if self._max_drawdown > self.cfg.circuit_breaker_dd:
            position_fraction = 0.0
            logger.debug("Circuit breaker active at t=%d (DD=%.1f%%)",
                         self._t, self._max_drawdown * 100)

        # Units = fraction × max_units × Kelly cap
        target_units  = position_fraction * self.cfg.max_units

        # Transaction cost if position changed
        position_changed = abs(target_units - self._current_units) > self.cfg.max_units * 0.05
        cost = self.cfg.transaction_cost if position_changed else 0.0

        # Execute: realised return at t+1
        self._t          += 1
        bar_return        = self.returns[self._t]
        strategy_return   = position_fraction * bar_return - cost

        # Update portfolio state
        self._portfolio_value  *= (1 + strategy_return)
        self._peak_value        = max(self._peak_value, self._portfolio_value)
        self._current_drawdown  = (self._peak_value - self._portfolio_value) / self._peak_value
        self._max_drawdown      = max(self._max_drawdown, self._current_drawdown)
        self._current_units     = target_units
        self._bars_in_trade     = self._bars_in_trade + 1 if target_units != 0 else 0

        # Track return history for CVaR
        self._return_history.append(strategy_return)
        if len(self._return_history) > self.cfg.cvar_window * 2:
            self._return_history.pop(0)

        # Update trade tracking
        if position_changed and target_units != 0:
            self._trade_returns.append(strategy_return)
            if len(self._trade_returns) > 20:
                self._trade_returns.pop(0)

        # ── Reward shaping ─────────────────────────────────────────────────────
        reward = self._compute_reward(strategy_return, position_fraction)

        # Episode termination
        terminated = (
            self._t >= self.n - 1 or
            self._portfolio_value <= 0.5 or   # 50% loss → terminate
            self._max_drawdown >= 0.40         # 40% catastrophic DD
        )

        info = {
            "t":               self._t,
            "portfolio_value": self._portfolio_value,
            "drawdown":        self._current_drawdown,
            "max_drawdown":    self._max_drawdown,
            "strategy_return": strategy_return,
            "position":        position_fraction,
            "cvar":            self._rolling_cvar(),
        }
        return self._get_obs(), reward, terminated, False, info

    # ── Reward function ───────────────────────────────────────────────────────
    def _compute_reward(
        self, strategy_return: float, position_fraction: float
    ) -> float:
        """
        CVaR-penalised reward function.

        R(t) = net_return(t)
               - λ_CVaR × max(0, CVaR₉₅ - CVaR_threshold)
               - λ_DD   × max(0, drawdown - DD_threshold)²
               + concentration_bonus   (reward for decisive positions)
        """
        # Base return reward
        r_base = strategy_return * self.cfg.reward_scale

        # CVaR penalty (tail risk)
        cvar = self._rolling_cvar()
        cvar_threshold = 0.005   # ~0.5% CVaR is acceptable
        cvar_penalty   = self.cfg.lambda_cvar * max(0, cvar - cvar_threshold)

        # Drawdown penalty (quadratic — punishes large drawdowns much more)
        dd_excess    = max(0, self._current_drawdown - self.cfg.drawdown_threshold)
        dd_penalty   = self.cfg.lambda_drawdown * (dd_excess ** 2) * 10.0

        # Regime-aware concentration bonus:
        # Reward decisive non-flat positions in high-confidence regimes
        regime       = self._current_regime()
        sig_conf     = self._current_signal_conf()
        concentration_bonus = 0.0
        if abs(position_fraction) > 0.3 and sig_conf > 0.55 and regime != 3:
            concentration_bonus = 0.001 * abs(position_fraction) * self.cfg.reward_scale

        # MiFID II: penalise excessive trading (>5 position changes per session)
        overtrading_penalty = 0.0
        if self._bars_in_trade == 1 and self._trade_count_session > 5:
            overtrading_penalty = 0.002 * self.cfg.reward_scale

        reward = r_base - cvar_penalty - dd_penalty + concentration_bonus - overtrading_penalty
        return float(reward)

    # ── Observation ───────────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        """Build observation vector for the RL policy."""
        t = min(self._t, self.n - 1)

        # Rolling volatility (20-bar)
        vol_window = max(0, t - 20)
        ret_window = self.returns[vol_window:t + 1]
        rolling_vol = float(np.std(ret_window)) if len(ret_window) > 2 else 0.001

        # Rolling VaR₉₅
        var_95 = float(np.percentile(ret_window, 5)) if len(ret_window) > 5 else -0.002

        # Regime and signal from aligned signals DataFrame
        regime         = self._current_regime()
        sig_conf       = self._current_signal_conf()
        fusion_weight  = self._current_fusion_weight()
        regime_conf    = self._current_regime_conf()

        # Recent hit ratio
        if len(self._trade_returns) >= 5:
            hit_ratio = float(np.mean([r > 0 for r in self._trade_returns]))
        else:
            hit_ratio = 0.5

        cvar = self._rolling_cvar()

        # Last 3 bar returns (momentum context)
        last3 = [float(self.returns[max(0, t - i)]) for i in range(3)]

        obs = np.array([
            np.clip((self._portfolio_value - 1.0) / 0.1, -5, 5),  # normalised PnL
            np.clip(self._current_drawdown / 0.15, 0, 3),          # normalised DD
            np.clip(rolling_vol / 0.005, 0, 5),                    # normalised vol
            np.clip(-var_95 / 0.005, 0, 5),                        # normalised VaR (pos)
            regime / 3.0,                                           # regime [0,1]
            regime_conf,                                            # regime confidence
            sig_conf,                                               # signal confidence
            fusion_weight,                                          # NLP fusion weight
            hit_ratio,                                              # recent hit ratio
            np.clip(self._current_units / self.cfg.max_units, -1, 1), # current position
            min(self._bars_in_trade / self.cfg.max_hold_bars, 1),  # hold duration
            np.clip(cvar / 0.005, 0, 5),                           # CVaR
            self._get_spread_cost(),                                # current spread
            last3[0] * 100, last3[1] * 100, last3[2] * 100,        # recent returns
        ], dtype=np.float32)

        return obs

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _rolling_cvar(self) -> float:
        """CVaR₉₅: Expected Shortfall in worst 5% of recent returns."""
        hist = self._return_history
        if len(hist) < 10:
            return 0.002
        arr    = np.array(hist)
        cutoff = np.percentile(arr, (1 - self.cfg.cvar_alpha) * 100)
        tail   = arr[arr <= cutoff]
        return float(-tail.mean()) if len(tail) > 0 else 0.002

    def _reset_state(self) -> None:
        self._portfolio_value  = 1.0
        self._peak_value       = 1.0
        self._current_drawdown = 0.0
        self._max_drawdown     = 0.0
        self._current_units    = 0.0
        self._bars_in_trade    = 0
        self._trade_count_session = 0
        self._return_history: list[float] = []
        self._trade_returns:  list[float] = []

    def _current_regime(self) -> int:
        t = min(self._t, self.n - 1)
        idx = self.features.index[t]
        if idx in self.signals.index and "regime" in self.signals.columns:
            return int(self.signals.loc[idx, "regime"])
        return 2

    def _current_signal_conf(self) -> float:
        t = min(self._t, self.n - 1)
        idx = self.features.index[t]
        if idx in self.signals.index and "confidence" in self.signals.columns:
            return float(self.signals.loc[idx, "confidence"])
        return 0.5

    def _current_fusion_weight(self) -> float:
        t = min(self._t, self.n - 1)
        idx = self.features.index[t]
        if idx in self.signals.index and "fusion_weight" in self.signals.columns:
            return float(self.signals.loc[idx, "fusion_weight"])
        return 0.2

    def _current_regime_conf(self) -> float:
        t = min(self._t, self.n - 1)
        idx = self.features.index[t]
        if idx in self.signals.index and "confidence" in self.signals.columns:
            return float(self.signals.loc[idx, "confidence"])
        return 0.5

    def _get_spread_cost(self) -> float:
        t   = min(self._t, self.n - 1)
        ts  = self.features.index[t]
        from maestro.agents.risk.cost_model import _get_session, TransactionCostModel
        session = _get_session(ts)
        from maestro.agents.risk.cost_model import SESSION_SPREAD_MULTIPLIER, INSTRUMENT_COSTS
        instr   = INSTRUMENT_COSTS.get("EUR_USD", {})
        spread  = instr.get("typical_spread_pips", 0.8) * SESSION_SPREAD_MULTIPLIER.get(session, 1.0)
        return spread / 10.0   # normalise to ~[0,1] range

    # ── Gymnasium wrapper factory ─────────────────────────────────────────────
    def as_gymnasium_env(self):
        """
        Wrap this environment as a proper gymnasium.Env subclass.
        Only call this when you have gymnasium installed and are about
        to start training — avoids the import for backtesting.
        """
        import gymnasium as gym

        env_instance = self

        class WrappedEnv(gym.Env):
            metadata = {"render_modes": []}

            def __init__(self):
                super().__init__()
                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(OBS_DIM,), dtype=np.float32
                )
                self.action_space = gym.spaces.Box(
                    low=-1.0, high=1.0, shape=(1,), dtype=np.float32
                )

            def reset(self, seed=None, options=None):
                return env_instance.reset(seed=seed)

            def step(self, action):
                return env_instance.step(action)

            def render(self):
                pass

        return WrappedEnv()
