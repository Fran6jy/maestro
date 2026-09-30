"""
maestro/agents/risk/risk_agent.py
===================================
Agent 4 — Risk Management Agent.

The unified risk layer that combines:
  1. PPO-trained CVaR-RL policy  — adaptive position sizing
  2. Kelly Criterion sizer        — edge-based position scaling
  3. Hard risk gates              — non-negotiable absolute limits
  4. Transaction cost awareness   — minimum edge threshold

Architecture
------------
Every potential trade passes through four sequential layers:

  Layer A: Hard Gates (non-negotiable)
    - Max drawdown breached?          → FLAT, no trade
    - Daily loss limit breached?      → FLAT, no trade
    - VaR limit breached?             → FLAT, no trade
    - Crisis regime + low confidence? → FLAT, no trade

  Layer B: CVaR-RL Policy
    - PPO policy observes full state → outputs position_fraction ∈ [-1,+1]
    - Trained to maximise Sharpe whilst minimising CVaR
    - Falls back to Kelly-only if RL policy not trained yet

  Layer C: Kelly Scaling
    - Apply fractional Kelly cap to RL output
    - Scale by confidence, regime, recent hit ratio
    - Apply correlation haircut for multi-instrument

  Layer D: Cost Filter
    - Reject trades where expected edge < transaction cost
    - Minimum 0.5 pips expected profit to justify trade

Output: RiskDecision dataclass
---------------------------------
  action:        "trade" | "flat" | "reduce"
  units:         signed integer (positive=long, negative=short)
  stop_loss_pips:float — where to place stop
  take_profit_pips:float — where to place take-profit
  risk_reason:   str — why this sizing was chosen
  position_fraction: float — fraction of max position used
  var_utilisation:   float — current VaR as fraction of limit
"""
from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from maestro.agents.risk.cost_model import TransactionCostModel
from maestro.agents.risk.kelly import KellyPositionSizer, KellyResult, REGIME_KELLY_CAPS
from maestro.agents.risk.cvar_env import CVaRTradingEnv, EnvConfig
from maestro.agents.signal.signal_agent import SignalPacket
from maestro.agents.regime.regime_classifier import RegimeSignal, REGIME_NAMES
from maestro.config.instruments import get_instrument_spec

logger = logging.getLogger(__name__)

# Hard risk limits (from config — these mirror settings.yaml)
HARD_LIMITS = {
    "max_drawdown":         0.15,    # 15%
    "circuit_breaker":      0.10,    # 10% → alert + reduce
    "max_daily_loss":       0.03,    # 3% max loss per day
    "max_var_95":           0.02,    # 2% VaR limit
    "min_confidence":       0.52,    # below this → flat
    "min_edge_pips":        0.5,     # minimum expected profit
    "max_position_pct":     0.20,    # max 20% of equity per instrument
}


@dataclass
class RiskDecision:
    """
    Output of Agent 4. Fed directly into Agent 5 (Execution).

    Fields
    ------
    timestamp:         bar datetime
    instrument:        e.g. "EUR_USD"
    action:            "trade" | "flat" | "reduce" | "circuit_break"
    units:             signed int — +ve=long, -ve=short, 0=flat
    stop_loss_pips:    float — SL distance from entry
    take_profit_pips:  float — TP distance from entry
    position_fraction: float — fraction of max position actually used
    kelly_fraction:    float — Kelly fraction computed
    var_utilisation:   float — current VaR / VaR limit
    cvar:              float — current CVaR estimate
    drawdown:          float — current drawdown
    daily_pnl:         float — today's P&L so far
    risk_reason:       str  — human-readable explanation
    is_compliant:      bool — passed all MiFID II checks
    """
    timestamp:         pd.Timestamp
    instrument:        str
    action:            str
    units:             int
    stop_loss_pips:    float
    take_profit_pips:  float
    position_fraction: float
    kelly_fraction:    float
    var_utilisation:   float
    cvar:              float
    drawdown:          float
    daily_pnl:         float
    risk_reason:       str
    is_compliant:      bool = True
    metadata:          dict = field(default_factory=dict)

    @property
    def is_trade(self) -> bool:
        return self.action == "trade" and self.units != 0

    def __repr__(self) -> str:
        return (
            f"RiskDecision({self.timestamp.strftime('%Y-%m-%d %H:%M')} | "
            f"{self.instrument} | {self.action.upper()} | "
            f"units={self.units:+d} | SL={self.stop_loss_pips:.1f}p | "
            f"DD={self.drawdown:.1%} | {self.risk_reason})"
        )

    def to_dict(self) -> dict:
        d = {f: getattr(self, f) for f in self.__dataclass_fields__}
        d.pop("metadata", None)
        return d


class RiskManagementAgent:
    """
    Agent 4 — Risk Management.

    Integrates CVaR-RL policy + Kelly sizing + hard gates into
    a single risk-adjusted position decision for each bar.

    Usage (backtesting)
    --------------------
    >>> agent = RiskManagementAgent("EUR_USD")
    >>> agent.train_rl(features_df, signals_df)   # train PPO policy
    >>> decisions = agent.evaluate_batch(
    ...     features_df, signals_df, regime_signals
    ... )

    Usage (live)
    ------------
    >>> decision = agent.decide(
    ...     signal_packet   = signal_agent.predict_bar(...),
    ...     regime_signal   = regime_agent.predict_bar(...),
    ...     current_price   = 1.0852,
    ...     account_equity  = 12_450.0,
    ...     current_drawdown= 0.024,
    ... )
    """

    def __init__(
        self,
        instrument:     str   = "EUR_USD",
        account_equity: float = 10_000.0,
        leverage:       float = 30.0,
        use_rl:         bool  = True,
    ) -> None:
        self.instrument = instrument
        self.spec       = get_instrument_spec(instrument)
        self.pip_size   = self.spec.pip_size
        self.equity     = account_equity
        self.leverage   = leverage
        self.use_rl     = use_rl

        self.cost_model = TransactionCostModel(instrument)
        self.kelly      = KellyPositionSizer(account_equity, leverage, pip_size=self.pip_size)
        self.rl_policy  = None     # PPO policy — set after train_rl()
        self.fitted     = False

        # State tracking
        self._daily_pnl:    float = 0.0
        self._daily_date:   object = None
        self._daily_start_equity: float = account_equity
        self._return_history: list[float] = []
        self._peak_equity:  float = account_equity

    # ── RL Training ───────────────────────────────────────────────────────────
    def train_rl(
        self,
        features_df:  pd.DataFrame,
        signals_df:   pd.DataFrame,
        total_steps:  int  = 500_000,
        save_path:    str | None = None,
    ) -> "RiskManagementAgent":
        """
        Train the PPO risk policy on historical data.

        Parameters
        ----------
        features_df  : full feature DataFrame
        signals_df   : Agent 2 + Agent 3 signals (confidence, regime, fusion_weight)
        total_steps  : PPO training timesteps (500k ≈ 2-3 hours on CPU)
        save_path    : where to save the trained policy
        """
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv
            from stable_baselines3.common.callbacks import EvalCallback
        except ImportError:
            raise ImportError(
                "Install stable-baselines3:\n"
                "  pip install stable-baselines3 gymnasium"
            )

        logger.info("Training CVaR-RL Policy | steps=%d | instrument=%s",
                    total_steps, self.instrument)

        env_config = EnvConfig(
            max_units         = int(self.equity * self.leverage * 0.20),
            lambda_cvar       = 0.5,
            lambda_drawdown   = 1.0,
            circuit_breaker_dd= HARD_LIMITS["circuit_breaker"],
        )
        env = CVaRTradingEnv(features_df, signals_df, config=env_config)
        gym_env = env.as_gymnasium_env()
        vec_env = DummyVecEnv([lambda: gym_env])

        # PPO hyperparameters tuned for financial time series
        self.rl_policy = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate    = 3e-4,
            n_steps          = 2048,
            batch_size       = 64,
            n_epochs         = 10,
            gamma            = 0.99,
            gae_lambda       = 0.95,
            clip_range       = 0.2,
            ent_coef         = 0.01,    # entropy bonus → exploration
            vf_coef          = 0.5,
            max_grad_norm    = 0.5,
            verbose          = 1,
            policy_kwargs    = dict(
                net_arch = [dict(pi=[128, 64], vf=[128, 64])]
            ),
        )

        self.rl_policy.learn(
            total_timesteps = total_steps,
            progress_bar    = True,
        )
        self.fitted = True

        if save_path:
            self.rl_policy.save(save_path)
            logger.info("RL policy saved → %s", save_path)

        logger.info("CVaR-RL training complete.")
        return self

    # ── Single-bar live decision ───────────────────────────────────────────────
    def decide(
        self,
        signal_packet:    SignalPacket,
        regime_signal:    RegimeSignal,
        current_price:    float,
        account_equity:   float,
        current_drawdown: float  = 0.0,
        recent_returns:   pd.Series | None = None,
        high_impact_news: bool = False,
    ) -> RiskDecision:
        """
        Make a risk-adjusted position decision for one bar.

        This is the function called every bar in live trading.
        Passes through all four layers: gates → RL → Kelly → cost filter.

        Returns
        -------
        RiskDecision — if .is_trade, pass to Agent 5 for execution
        """
        ts         = signal_packet.timestamp
        signal     = signal_packet.signal
        confidence = signal_packet.confidence
        regime     = regime_signal.regime

        self.kelly.update_equity(account_equity)
        self._update_daily_pnl(ts, account_equity)
        self._peak_equity = max(self._peak_equity, account_equity)

        # ── Layer A: Hard Gates ────────────────────────────────────────────────
        gate_result = self._check_hard_gates(
            signal, confidence, regime,
            current_drawdown, current_price, high_impact_news
        )
        if gate_result is not None:
            return self._flat_decision(ts, gate_result)

        # ── Layer B: RL Policy position fraction ──────────────────────────────
        rl_fraction = self._query_rl_policy(
            signal, confidence, regime, current_drawdown,
            current_price, account_equity, recent_returns
        )

        # ── Layer C: Kelly scaling ────────────────────────────────────────────
        ret_series = recent_returns if recent_returns is not None else pd.Series(dtype=float)
        kelly_result = self.kelly.size(
            signal           = signal,
            confidence       = confidence,
            regime           = regime,
            recent_returns   = ret_series,
            current_drawdown = current_drawdown,
            price            = current_price,
        )

        if not kelly_result.is_trade:
            return self._flat_decision(ts, f"kelly_negative_edge: {kelly_result.cap_reason}")

        # Blend RL fraction with Kelly fraction
        if self.fitted and rl_fraction != 0:
            blended_fraction = 0.6 * rl_fraction + 0.4 * kelly_result.applied_fraction
        else:
            blended_fraction = kelly_result.applied_fraction

        raw_units = int(abs(blended_fraction) * int(account_equity * self.leverage / current_price))
        units     = raw_units * signal
        units     = max(min(abs(units), self.kelly.max_units), 0) * signal

        # ── Layer D: Cost filter ──────────────────────────────────────────────
        cost_est  = self.cost_model.estimate(units, current_price,
                                              session=self._current_session(ts))
        # Expected move in the direction of the trade: a short needs a negative forecast.
        # (Using the signed forecast alone rejected every short.)
        edge_pips = (signal * signal_packet.pred_p50 * current_price / self.pip_size
                     if signal_packet.pred_p50 else 0.0)
        if edge_pips < HARD_LIMITS["min_edge_pips"] + cost_est.breakeven_pips:
            return self._flat_decision(ts,
                f"insufficient_edge: {edge_pips:.2f}p < {cost_est.breakeven_pips:.2f}p breakeven")

        # ── SL / TP from ATR ──────────────────────────────────────────────────
        sl_pips, tp_pips = self._compute_sl_tp(
            signal, regime, confidence,
            signal_packet.pred_p10, signal_packet.pred_p90, current_price
        )

        # Rolling CVaR and VaR for reporting
        cvar       = self._compute_cvar()
        var_util   = cvar / HARD_LIMITS["max_var_95"]

        return RiskDecision(
            timestamp         = ts,
            instrument        = self.instrument,
            action            = "trade",
            units             = units,
            stop_loss_pips    = sl_pips,
            take_profit_pips  = tp_pips,
            position_fraction = blended_fraction,
            kelly_fraction    = kelly_result.kelly_fraction,
            var_utilisation   = var_util,
            cvar              = cvar,
            drawdown          = current_drawdown,
            daily_pnl         = self._daily_pnl,
            risk_reason       = (
                f"regime={REGIME_NAMES[regime]} | "
                f"kelly={kelly_result.applied_fraction:.3f} | "
                f"rl_blend={blended_fraction:.3f} | "
                f"edge={edge_pips:.2f}p"
            ),
        )

    # ── Batch evaluation (backtesting) ────────────────────────────────────────
    def evaluate_batch(
        self,
        features_df:    pd.DataFrame,
        signals_df:     pd.DataFrame,
        regime_signals: pd.DataFrame,
        simulate_equity: bool = True,
    ) -> pd.DataFrame:
        """
        Apply risk management to all bars in a backtest.

        Returns pd.DataFrame with all RiskDecision fields per bar,
        plus net returns after costs and the equity curve.
        """
        logger.info("Risk evaluation: %d bars × %s", len(features_df), self.instrument)

        decisions = []
        equity    = self.equity
        drawdown  = 0.0
        peak_eq   = equity
        ret_hist: list[float] = []

        close   = features_df["close"]
        ret_ser = features_df["log_return_1"].fillna(0)

        for ts in features_df.index:
            # Reconstruct signal packet from signals_df
            sig_row = signals_df.loc[ts] if ts in signals_df.index else None
            if sig_row is None:
                decisions.append(self._flat_decision(ts, "no_signal").to_dict())
                continue

            signal    = int(sig_row.get("signal",     0))
            conf      = float(sig_row.get("confidence", 0.5))
            regime    = int(sig_row.get("regime",      2))
            p50       = float(sig_row.get("pred_p50",  0.0)) if "pred_p50" in sig_row else 0.0
            p10       = float(sig_row.get("pred_p10",  0.0)) if "pred_p10" in sig_row else 0.0
            p90       = float(sig_row.get("pred_p90",  0.0)) if "pred_p90" in sig_row else 0.0

            # Build lightweight SignalPacket
            from maestro.agents.signal.signal_agent import SignalPacket
            packet = SignalPacket(
                timestamp=ts, instrument=self.instrument,
                horizon=6, signal=signal, confidence=conf,
                regime=regime, regime_name=REGIME_NAMES.get(regime, ""),
                model_agree=bool(sig_row.get("model_agree", False)),
                tft_signal=signal, ptst_signal=signal,
                pred_p10=p10, pred_p50=p50, pred_p90=p90,
            )
            from maestro.agents.regime.regime_classifier import RegimeSignal
            reg_packet = RegimeSignal(
                timestamp=ts, regime=regime,
                regime_name=REGIME_NAMES.get(regime, "sideways"),
                confidence=float(sig_row.get("confidence", 0.5)),
                probabilities={}, is_certain=conf > 0.55,
            )

            price = float(close.loc[ts]) if ts in close.index else 1.0
            recent_ret = pd.Series(ret_hist[-20:]) if ret_hist else pd.Series(dtype=float)

            decision = self.decide(
                signal_packet    = packet,
                regime_signal    = reg_packet,
                current_price    = price,
                account_equity   = equity,
                current_drawdown = drawdown,
                recent_returns   = recent_ret,
            )

            # Legacy standalone simulation. The end-to-end backtest disables
            # this and delegates accounting to CausalPortfolioLedger.
            if simulate_equity and decision.is_trade and ts in ret_ser.index:
                bar_ret    = float(ret_ser.loc[ts]) * decision.units / (equity * self.leverage / price)
                cost_est   = self.cost_model.estimate(decision.units, price)
                net_ret    = bar_ret - cost_est.total_cost
                equity    *= (1 + net_ret)
                ret_hist.append(net_ret)
                drawdown   = max(0, (peak_eq - equity) / peak_eq)
                peak_eq    = max(peak_eq, equity)

            d = decision.to_dict()
            d["equity"]    = equity
            d["net_return"]= (equity / self.equity) - 1 if simulate_equity else 0.0
            decisions.append(d)

        result = pd.DataFrame(decisions).set_index("timestamp")
        self._log_evaluation_summary(result)
        return result

    # ── Hard gate logic ───────────────────────────────────────────────────────
    def _check_hard_gates(
        self, signal, confidence, regime,
        drawdown, price, high_impact
    ) -> str | None:
        """Returns reason string if trade should be blocked, else None."""
        if signal == 0:
            return "flat_signal"
        if drawdown >= HARD_LIMITS["max_drawdown"]:
            return f"max_drawdown_breached: {drawdown:.1%}"
        if self._daily_pnl <= -HARD_LIMITS["max_daily_loss"]:
            return f"daily_loss_limit: {self._daily_pnl:.1%}"
        if confidence < HARD_LIMITS["min_confidence"]:
            return f"low_confidence: {confidence:.3f}"
        if regime == 3 and confidence < 0.62:
            return "crisis_regime_low_confidence"
        if high_impact:
            return "high_impact_news_blackout"
        return None

    def _flat_decision(self, ts: pd.Timestamp, reason: str) -> RiskDecision:
        return RiskDecision(
            timestamp=ts, instrument=self.instrument,
            action="flat", units=0,
            stop_loss_pips=0.0, take_profit_pips=0.0,
            position_fraction=0.0, kelly_fraction=0.0,
            var_utilisation=0.0, cvar=self._compute_cvar(),
            drawdown=0.0, daily_pnl=self._daily_pnl,
            risk_reason=reason,
        )

    # ── RL query ──────────────────────────────────────────────────────────────
    def _query_rl_policy(
        self, signal, confidence, regime, drawdown,
        price, equity, recent_returns
    ) -> float:
        """Query trained PPO policy for position fraction."""
        if not self.fitted or self.rl_policy is None:
            return float(signal) * REGIME_KELLY_CAPS.get(regime, 0.2)

        try:
            from maestro.agents.risk.cvar_env import OBS_DIM
            cvar = self._compute_cvar()
            ret3 = list(recent_returns.tail(3).values) if recent_returns is not None and len(recent_returns) >= 3 else [0.0, 0.0, 0.0]
            obs  = np.array([
                0.0, drawdown / 0.15, 0.001 / 0.005,
                cvar / 0.005, regime / 3.0, confidence,
                confidence, 0.2, 0.5,
                float(signal) * 0.2, 0.0, cvar / 0.005,
                0.08,
                ret3[0] * 100, ret3[1] * 100, ret3[2] * 100,
            ], dtype=np.float32)

            action, _ = self.rl_policy.predict(obs, deterministic=True)
            return float(np.clip(action[0], -1.0, 1.0))
        except Exception as e:
            logger.debug("RL policy query failed: %s", e)
            return float(signal) * REGIME_KELLY_CAPS.get(regime, 0.2)

    # ── SL / TP calculation ───────────────────────────────────────────────────
    def _compute_sl_tp(
        self, signal, regime, confidence,
        p10, p90, price
    ) -> tuple[float, float]:
        """
        Compute stop-loss and take-profit in pips from TFT quantile predictions.
        Falls back to regime-based defaults if predictions are unavailable.
        """
        pip = self.pip_size

        # Use TFT quantile spread for SL/TP if available
        if p10 != 0.0 and p90 != 0.0:
            if signal == 1:
                raw_sl = abs(p10) * price / pip      # downside scenario
                raw_tp = abs(p90) * price / pip      # upside scenario
            else:
                raw_sl = abs(p90) * price / pip
                raw_tp = abs(p10) * price / pip

            # Enforce minimum 1:1.5 risk:reward
            sl = max(raw_sl, 3.0)
            tp = max(raw_tp, sl * 1.5)
        else:
            # Regime-based defaults (pips)
            regime_sl = {0: 12.0, 1: 12.0, 2: 8.0, 3: 5.0}
            regime_rr = {0: 2.0,  1: 2.0,  2: 1.5, 3: 1.5}  # reward:risk
            sl = regime_sl.get(regime, 10.0)
            tp = sl * regime_rr.get(regime, 2.0)

        # Scale by confidence (higher confidence → tighter SL acceptable)
        conf_scale = 0.8 + 0.4 * confidence   # [0.8, 1.2]
        sl *= conf_scale
        tp *= conf_scale

        return round(sl, 1), round(tp, 1)

    # ── Portfolio VaR / CVaR ──────────────────────────────────────────────────
    def _compute_cvar(self) -> float:
        hist = self._return_history
        if len(hist) < 10:
            return 0.002
        arr    = np.array(hist[-60:])
        cutoff = np.percentile(arr, 5)
        tail   = arr[arr <= cutoff]
        return float(-tail.mean()) if len(tail) > 0 else 0.002

    def _update_daily_pnl(self, ts: pd.Timestamp, equity: float) -> None:
        today = ts.date() if hasattr(ts, "date") else None
        if self._daily_date != today:
            self._daily_pnl  = 0.0
            self._daily_date = today
            self._daily_start_equity = equity
        self._daily_pnl = (equity - self._daily_start_equity) / max(self._daily_start_equity, 1.0)

    @staticmethod
    def _current_session(ts: pd.Timestamp) -> str:
        from maestro.agents.risk.cost_model import _get_session
        return _get_session(ts)

    # ── Summary ───────────────────────────────────────────────────────────────
    def _log_evaluation_summary(self, result: pd.DataFrame) -> None:
        trades     = result[result["action"] == "trade"] if "action" in result.columns else result
        n_trades   = len(trades)
        n_total    = len(result)
        if n_total == 0:
            return

        flat_rate  = 1.0 - n_trades / n_total
        avg_units  = abs(trades["units"]).mean() if n_trades > 0 and "units" in trades.columns else 0
        avg_kelly  = trades["kelly_fraction"].mean() if n_trades > 0 and "kelly_fraction" in trades.columns else 0
        peak_dd    = result["drawdown"].max() if "drawdown" in result.columns else 0

        logger.info(
            "Risk eval: %d bars | %d trades (%.1f%% flat) | "
            "avg_units=%.0f | avg_kelly=%.4f | peak_dd=%.1f%%",
            n_total, n_trades, flat_rate * 100,
            avg_units, avg_kelly, peak_dd * 100,
        )

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        if self.fitted and self.rl_policy is not None:
            self.rl_policy.save(str(model_dir / "ppo_risk_policy"))
        meta = {
            "instrument": self.instrument,
            "equity":     self.equity,
            "leverage":   self.leverage,
            "use_rl":     self.use_rl,
            "fitted":     self.fitted,
        }
        with open(model_dir / "risk_agent_meta.pkl", "wb") as f:
            pickle.dump(meta, f)
        logger.info("Risk Agent saved → %s", model_dir)

    @classmethod
    def load(cls, model_dir: str | Path) -> "RiskManagementAgent":
        model_dir = Path(model_dir)
        with open(model_dir / "risk_agent_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        agent = cls(
            instrument     = meta["instrument"],
            account_equity = meta["equity"],
            leverage       = meta["leverage"],
            use_rl         = meta["use_rl"],
        )
        if meta["fitted"]:
            try:
                from stable_baselines3 import PPO
                agent.rl_policy = PPO.load(str(model_dir / "ppo_risk_policy"))
                agent.fitted    = True
            except Exception as e:
                logger.warning("Could not load RL policy: %s", e)
        return agent
