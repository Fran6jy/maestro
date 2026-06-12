"""
maestro/orchestrator/meta_orchestrator.py
===========================================
MAESTRO Meta-Orchestrator — the top-level coordination layer.

What the orchestrator does
---------------------------
The orchestrator is the single decision-maker that aggregates signals
from all five agents and produces the final trading action:

  Agent 1 (Regime)    → regime_id, regime_confidence
  Agent 2 (Signal)    → technical_signal, signal_confidence
  Agent 3 (Sentiment) → sentiment_signal, fusion_weight, nlp_accuracy
  Agent 4 (Risk)      → position_size, SL/TP, CVaR, drawdown state
  Agent 5 (Execution) → receives final decision, handles order routing

The orchestrator's role is NOT to override agents but to:
  1. Dynamically re-weight each agent's contribution per regime
  2. Detect and handle agent disagreements
  3. Trigger circuit breakers when the system is in an unusual state
  4. Log the full decision provenance for explainability (XAI)
  5. Coordinate the live trading loop timing

Dynamic weighting (the key innovation)
----------------------------------------
In the PhD proposal, this is the "meta-policy" — a PPO agent that
learns optimal agent weights as a function of market state. In the
first implementation, we use a rule-based weighting scheme that
can be replaced by a trained PPO policy in a later phase.

Regime-conditioned agent weights:
                      Agent2  Agent3  (Agent1 always informs context)
  bull_trend:         0.80    0.20    (technicals dominate)
  bear_trend:         0.75    0.25    (technicals, slightly more text)
  sideways:           0.70    0.30    (range + some macro context)
  crisis:             0.40    0.60    (LLM/macro context dominates)

Agreement amplification
------------------------
When Agent 2 and Agent 3 agree on direction:
  Final signal strength is boosted by ×1.2 (max cap: 1.0)
When they disagree:
  Signal is attenuated by ×0.6 (high uncertainty)
When one agent is not confident:
  The other agent's weight increases to compensate

Decision provenance
--------------------
Every decision is logged as an OrchestratorDecision with the full
agent contribution breakdown — this is what feeds the XAI layer
to explain any trade to regulators / thesis examiners.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from maestro.agents.regime.regime_classifier import RegimeSignal, REGIME_NAMES
from maestro.agents.signal.signal_agent import SignalPacket
from maestro.agents.sentiment.fusion import SentimentPacket
from maestro.agents.risk.risk_agent import RiskDecision

logger = logging.getLogger(__name__)

# Regime-conditioned base weights [signal_weight, sentiment_weight]
REGIME_AGENT_WEIGHTS = {
    0: (0.80, 0.20),   # bull_trend
    1: (0.75, 0.25),   # bear_trend
    2: (0.70, 0.30),   # sideways
    3: (0.40, 0.60),   # crisis
}

# Minimum aggregate confidence to pass to execution
MIN_AGGREGATE_CONFIDENCE = 0.50


@dataclass
class OrchestratorDecision:
    """
    The complete, provenance-tracked decision of the MAESTRO system.

    Every field is logged for XAI / regulatory explainability.
    """
    decision_id:          str
    timestamp:            pd.Timestamp
    instrument:           str

    # Final decision
    final_action:         str           # "trade" | "flat" | "circuit_break"
    final_signal:         int           # {-1, 0, +1}
    final_units:          int
    aggregate_confidence: float

    # Agent inputs
    regime:               int
    regime_name:          str
    regime_confidence:    float

    signal_direction:     int
    signal_confidence:    float
    signal_model_agree:   bool
    signal_pred_p50:      float

    sentiment_direction:  int
    sentiment_confidence: float
    sentiment_fusion_weight: float
    nlp_rolling_accuracy: float

    risk_action:          str
    risk_position_frac:   float
    risk_kelly_frac:      float
    risk_drawdown:        float
    risk_cvar:            float
    stop_loss_pips:       float
    take_profit_pips:     float

    # Orchestrator weights used
    weight_signal:        float
    weight_sentiment:     float
    agents_agree:         bool
    agreement_boost:      float

    # Diagnostics
    flat_reason:          str = ""
    provenance:           dict = field(default_factory=dict)

    def __repr__(self) -> str:
        direction = {1: "BUY", -1: "SELL", 0: "FLAT"}.get(self.final_signal, "?")
        return (
            f"OrchestratorDecision({self.timestamp.strftime('%Y-%m-%d %H:%M')} | "
            f"{self.instrument} | {direction} {abs(self.final_units)} | "
            f"conf={self.aggregate_confidence:.3f} | {self.regime_name} | "
            f"agree={self.agents_agree})"
        )

    def to_dict(self) -> dict:
        d = {f: getattr(self, f) for f in self.__dataclass_fields__}
        d.pop("provenance", None)
        return d


class MetaOrchestrator:
    """
    MAESTRO Meta-Orchestrator.

    Coordinates all five agents and produces final trading decisions
    with full provenance tracking.

    Usage (live trading loop)
    --------------------------
    >>> orch = MetaOrchestrator(instrument="EUR_USD")
    >>> orch.load_agents(
    ...     regime_agent=regime_agent,
    ...     signal_agent=signal_agent,
    ...     sentiment_agent=sentiment_agent,
    ...     risk_agent=risk_agent,
    ...     execution_agent=execution_agent,
    ... )
    >>> for bar in live_bars:
    ...     decision = orch.process_bar(bar)
    ...     if decision.final_action == "trade":
    ...         execution_agent.execute(decision, bar)

    Usage (backtesting — batch mode)
    ----------------------------------
    >>> decisions_df = orch.run_backtest(features_df, news_df)
    """

    def __init__(
        self,
        instrument:        str  = "EUR_USD",
        use_adaptive_weights: bool = True,
    ) -> None:
        self.instrument           = instrument
        self.use_adaptive_weights = use_adaptive_weights

        # Agent handles (set via load_agents())
        self.regime_agent    = None
        self.signal_agent    = None
        self.sentiment_agent = None
        self.risk_agent      = None
        self.execution_agent = None

        # Decision history (for rolling stats + XAI)
        self._decision_history: list[OrchestratorDecision] = []
        self._regime_weights: dict[int, tuple[float, float]] = dict(REGIME_AGENT_WEIGHTS)

        # Performance tracking for adaptive weight adjustment
        self._regime_signal_accuracy:    dict[int, list[float]] = {r: [] for r in range(4)}
        self._regime_sentiment_accuracy: dict[int, list[float]] = {r: [] for r in range(4)}

    def load_agents(
        self,
        regime_agent    = None,
        signal_agent    = None,
        sentiment_agent = None,
        risk_agent      = None,
        execution_agent = None,
    ) -> "MetaOrchestrator":
        """Attach fitted agent instances."""
        self.regime_agent    = regime_agent
        self.signal_agent    = signal_agent
        self.sentiment_agent = sentiment_agent
        self.risk_agent      = risk_agent
        self.execution_agent = execution_agent
        logger.info(
            "Orchestrator loaded: regime=%s signal=%s sentiment=%s risk=%s exec=%s",
            "✓" if regime_agent    else "✗",
            "✓" if signal_agent    else "✗",
            "✓" if sentiment_agent else "✗",
            "✓" if risk_agent      else "✗",
            "✓" if execution_agent else "✗",
        )
        return self

    # ── Single-bar live processing ────────────────────────────────────────────
    def process_bar(
        self,
        bar_df:       pd.DataFrame,
        history_df:   pd.DataFrame,
        recent_news:  pd.DataFrame | None = None,
    ) -> OrchestratorDecision:
        """
        Process one incoming bar through the full agent pipeline.

        Parameters
        ----------
        bar_df      : single-row DataFrame for the current bar
        history_df  : recent history (for windowed models)
        recent_news : scored news articles from Agent 3

        Returns
        -------
        OrchestratorDecision — pass to ExecutionAgent if is_trade
        """
        ts = bar_df.index[-1]

        # ── Agent 1: Regime ────────────────────────────────────────────────────
        regime_signal = self._get_regime(bar_df, history_df)

        # ── Agent 2: Technical Signal ─────────────────────────────────────────
        signal_packet = self._get_signal(bar_df, history_df, regime_signal)

        # ── Agent 3: Sentiment ────────────────────────────────────────────────
        sentiment_packet = self._get_sentiment(bar_df, regime_signal, signal_packet, recent_news)

        # ── Orchestrator fusion ───────────────────────────────────────────────
        final_signal, agg_conf, w_sig, w_sent, agree, boost = self._fuse(
            regime_signal, signal_packet, sentiment_packet
        )

        # ── Agent 4: Risk ─────────────────────────────────────────────────────
        risk_decision = self._get_risk(
            signal_packet, regime_signal,
            bar_df, history_df, final_signal, agg_conf
        )

        # ── Build orchestrator decision ───────────────────────────────────────
        decision = OrchestratorDecision(
            decision_id           = uuid.uuid4().hex[:12],
            timestamp             = ts,
            instrument            = self.instrument,
            final_action          = risk_decision.action,
            final_signal          = final_signal,
            final_units           = risk_decision.units,
            aggregate_confidence  = agg_conf,
            regime                = regime_signal.regime,
            regime_name           = regime_signal.regime_name,
            regime_confidence     = regime_signal.confidence,
            signal_direction      = signal_packet.signal,
            signal_confidence     = signal_packet.confidence,
            signal_model_agree    = signal_packet.model_agree,
            signal_pred_p50       = signal_packet.pred_p50,
            sentiment_direction   = sentiment_packet.sentiment_signal,
            sentiment_confidence  = sentiment_packet.text_confidence,
            sentiment_fusion_weight = sentiment_packet.fusion_weight,
            nlp_rolling_accuracy  = sentiment_packet.rolling_nlp_accuracy,
            risk_action           = risk_decision.action,
            risk_position_frac    = risk_decision.position_fraction,
            risk_kelly_frac       = risk_decision.kelly_fraction,
            risk_drawdown         = risk_decision.drawdown,
            risk_cvar             = risk_decision.cvar,
            stop_loss_pips        = risk_decision.stop_loss_pips,
            take_profit_pips      = risk_decision.take_profit_pips,
            weight_signal         = w_sig,
            weight_sentiment      = w_sent,
            agents_agree          = agree,
            agreement_boost       = boost,
            flat_reason           = risk_decision.risk_reason if risk_decision.action != "trade" else "",
        )

        self._decision_history.append(decision)
        self._log_decision(decision)
        return decision

    # ── Batch backtest ────────────────────────────────────────────────────────
    def run_backtest(
        self,
        features_df:    pd.DataFrame,
        signal_df:      pd.DataFrame,       # pre-computed Agent 2 output
        regime_df:      pd.DataFrame,       # pre-computed Agent 1 output
        sentiment_df:   pd.DataFrame | None = None,  # pre-computed Agent 3 output
    ) -> pd.DataFrame:
        """
        Run the full orchestration pipeline on pre-computed agent outputs.

        This is the fast backtest path — agents are run once externally
        and their outputs are passed in, avoiding re-training per bar.

        Returns
        -------
        pd.DataFrame — OrchestratorDecision per bar
        """
        logger.info("Orchestrator backtest: %d bars × %s", len(features_df), self.instrument)
        decisions = []

        for ts in features_df.index:
            # Read pre-computed agent outputs for this bar
            regime       = int(regime_df.loc[ts, "regime"])          if ts in regime_df.index else 2
            regime_conf  = float(regime_df.loc[ts, "confidence"])     if ts in regime_df.index and "confidence" in regime_df.columns else 0.5
            sig          = int(signal_df.loc[ts, "signal"])           if ts in signal_df.index and "signal" in signal_df.columns else 0
            sig_conf     = float(signal_df.loc[ts, "confidence"])     if ts in signal_df.index and "confidence" in signal_df.columns else 0.5
            model_agree  = bool(signal_df.loc[ts, "model_agree"])     if ts in signal_df.index and "model_agree" in signal_df.columns else False
            p50          = float(signal_df.loc[ts, "pred_p50"])       if ts in signal_df.index and "pred_p50" in signal_df.columns else 0.0
            p10          = float(signal_df.loc[ts, "pred_p10"])       if ts in signal_df.index and "pred_p10" in signal_df.columns else 0.0
            p90          = float(signal_df.loc[ts, "pred_p90"])       if ts in signal_df.index and "pred_p90" in signal_df.columns else 0.0

            sent_sig     = int(sentiment_df.loc[ts, "sentiment_signal"]) if sentiment_df is not None and ts in sentiment_df.index else 0
            sent_conf    = float(sentiment_df.loc[ts, "text_confidence"]) if sentiment_df is not None and ts in sentiment_df.index and "text_confidence" in sentiment_df.columns else 0.3
            fusion_w     = float(sentiment_df.loc[ts, "fusion_weight"])   if sentiment_df is not None and ts in sentiment_df.index and "fusion_weight" in sentiment_df.columns else 0.2
            nlp_acc      = float(sentiment_df.loc[ts, "rolling_nlp_accuracy"]) if sentiment_df is not None and ts in sentiment_df.index and "rolling_nlp_accuracy" in sentiment_df.columns else 0.5

            # Build lightweight packets
            from maestro.agents.signal.signal_agent import SignalPacket
            from maestro.agents.regime.regime_classifier import RegimeSignal
            from maestro.agents.sentiment.fusion import SentimentPacket

            regime_pkt = RegimeSignal(
                timestamp=ts, regime=regime, regime_name=REGIME_NAMES.get(regime, "sideways"),
                confidence=regime_conf, probabilities={}, is_certain=regime_conf > 0.55
            )
            signal_pkt = SignalPacket(
                timestamp=ts, instrument=self.instrument, horizon=6,
                signal=sig, confidence=sig_conf, regime=regime,
                regime_name=REGIME_NAMES.get(regime, "sideways"),
                model_agree=model_agree, tft_signal=sig, ptst_signal=sig,
                pred_p10=p10, pred_p50=p50, pred_p90=p90,
            )
            sent_pkt = SentimentPacket(
                timestamp=ts, instrument=self.instrument,
                sentiment_signal=sent_sig, text_confidence=sent_conf,
                fusion_weight=fusion_w, price_text_agree=(sig == sent_sig),
                finbert_score=0.0, gpt4o_bias=0.0, gpt4o_confidence=0.0,
                gpt4o_surprise=0.0, rolling_nlp_accuracy=nlp_acc,
                article_count=0, regime=regime, regime_boost=(regime == 3),
            )

            # Fuse
            final_sig, agg_conf, w_sig, w_sent, agree, boost = self._fuse(
                regime_pkt, signal_pkt, sent_pkt
            )

            # Risk decision from pre-computed signal df
            risk_action    = "trade" if sig != 0 and sig_conf >= 0.52 else "flat"
            units          = int(signal_df.loc[ts, "signal"]) * 10_000 if ts in signal_df.index else 0
            sl_pips        = float(signal_df.loc[ts, "stop_loss_pips"])    if ts in signal_df.index and "stop_loss_pips" in signal_df.columns else 12.0
            tp_pips        = float(signal_df.loc[ts, "take_profit_pips"])  if ts in signal_df.index and "take_profit_pips" in signal_df.columns else 24.0

            decisions.append({
                "timestamp":            ts,
                "instrument":           self.instrument,
                "final_action":         risk_action,
                "final_signal":         final_sig,
                "final_units":          units if risk_action == "trade" else 0,
                "aggregate_confidence": agg_conf,
                "regime":               regime,
                "regime_name":          REGIME_NAMES.get(regime, "sideways"),
                "regime_confidence":    regime_conf,
                "signal_direction":     sig,
                "signal_confidence":    sig_conf,
                "sentiment_direction":  sent_sig,
                "sentiment_confidence": sent_conf,
                "fusion_weight":        fusion_w,
                "agents_agree":         agree,
                "agreement_boost":      boost,
                "weight_signal":        w_sig,
                "weight_sentiment":     w_sent,
                "stop_loss_pips":       sl_pips,
                "take_profit_pips":     tp_pips,
            })

        result = pd.DataFrame(decisions).set_index("timestamp")
        self._log_backtest_summary(result)
        return result

    # ── Fusion logic ──────────────────────────────────────────────────────────
    def _fuse(
        self,
        regime:    RegimeSignal,
        signal:    SignalPacket,
        sentiment: SentimentPacket,
    ) -> tuple[int, float, float, float, bool, float]:
        """
        Fuse Agent 2 + Agent 3 signals under Agent 1 regime context.

        Returns
        -------
        (final_signal, aggregate_confidence,
         weight_signal, weight_sentiment, agents_agree, agreement_boost)
        """
        # Base weights from regime
        w_sig, w_sent = self._regime_weights.get(regime.regime, (0.75, 0.25))

        # Override with adaptive fusion weight from Agent 3 if available
        if sentiment.fusion_weight > 0:
            w_sent = sentiment.fusion_weight
            w_sig  = 1.0 - w_sent

        # Compute weighted signal score
        sig_score  = signal.signal    * signal.confidence    * w_sig
        sent_score = sentiment.sentiment_signal * sentiment.text_confidence * w_sent
        raw_score  = sig_score + sent_score

        # Agreement detection
        agents_agree = (
            signal.signal != 0 and
            sentiment.sentiment_signal != 0 and
            signal.signal == sentiment.sentiment_signal
        )
        agreement_boost = 1.0
        if agents_agree:
            agreement_boost = 1.20
            raw_score      *= agreement_boost
        elif signal.signal != 0 and sentiment.sentiment_signal != 0 and signal.signal != sentiment.sentiment_signal:
            # Active disagreement → attenuate
            raw_score      *= 0.60

        # Final signal
        threshold    = 0.08
        final_signal = int(np.sign(raw_score)) if abs(raw_score) > threshold else 0

        # Aggregate confidence
        agg_conf = min(abs(raw_score), 1.0)
        if not regime.is_certain:
            agg_conf *= 0.85    # regime uncertainty penalty

        return final_signal, agg_conf, w_sig, w_sent, agents_agree, agreement_boost

    # ── Agent stub calls (for live mode) ──────────────────────────────────────
    def _get_regime(self, bar_df, history_df) -> RegimeSignal:
        if self.regime_agent:
            return self.regime_agent.predict_bar(bar_df, history_df)
        from maestro.agents.regime.regime_classifier import RegimeSignal
        return RegimeSignal(
            timestamp=bar_df.index[-1], regime=2, regime_name="sideways",
            confidence=0.5, probabilities={}, is_certain=False
        )

    def _get_signal(self, bar_df, history_df, regime_signal) -> SignalPacket:
        if self.signal_agent:
            return self.signal_agent.predict_bar(bar_df, regime_signal, history_df)
        from maestro.agents.signal.signal_agent import SignalPacket
        return SignalPacket(
            timestamp=bar_df.index[-1], instrument=self.instrument,
            horizon=6, signal=0, confidence=0.5, regime=regime_signal.regime,
            regime_name=regime_signal.regime_name, model_agree=False,
            tft_signal=0, ptst_signal=0, pred_p10=0.0, pred_p50=0.0, pred_p90=0.0
        )

    def _get_sentiment(self, bar_df, regime_signal, signal_packet, recent_news) -> SentimentPacket:
        if self.sentiment_agent:
            return self.sentiment_agent.update_bar(
                bar_df.index[-1], regime_signal, recent_news or pd.DataFrame(),
                price_signal=signal_packet.signal
            )
        from maestro.agents.sentiment.fusion import SentimentPacket
        return SentimentPacket(
            timestamp=bar_df.index[-1], instrument=self.instrument,
            sentiment_signal=0, text_confidence=0.3, fusion_weight=0.2,
            price_text_agree=False, finbert_score=0.0, gpt4o_bias=0.0,
            gpt4o_confidence=0.0, gpt4o_surprise=0.0, rolling_nlp_accuracy=0.5,
            article_count=0, regime=regime_signal.regime, regime_boost=False
        )

    def _get_risk(self, signal_packet, regime_signal, bar_df, history_df,
                  final_signal, agg_conf) -> RiskDecision:
        if self.risk_agent:
            price  = float(bar_df["close"].iloc[-1]) if "close" in bar_df.columns else 1.0
            equity = self.risk_agent.equity
            recent = pd.Series([d.final_signal * 0.001 for d in self._decision_history[-20:]])
            # Override signal packet confidence with aggregate
            import copy
            pkt = copy.copy(signal_packet)
            pkt.signal     = final_signal
            pkt.confidence = agg_conf
            return self.risk_agent.decide(pkt, regime_signal, price, equity,
                                          recent_returns=recent)
        from maestro.agents.risk.risk_agent import RiskDecision
        return RiskDecision(
            timestamp=signal_packet.timestamp, instrument=self.instrument,
            action="flat" if final_signal == 0 else "trade",
            units=final_signal * 10_000,
            stop_loss_pips=12.0, take_profit_pips=24.0,
            position_fraction=abs(agg_conf) * 0.2, kelly_fraction=0.1,
            var_utilisation=0.0, cvar=0.002, drawdown=0.0, daily_pnl=0.0,
            risk_reason="orchestrator_default",
        )

    # ── Adaptive weight update ────────────────────────────────────────────────
    def update_weights_from_outcome(
        self,
        decision:      OrchestratorDecision,
        realised_return: float,
    ) -> None:
        """
        Update regime-conditioned weights based on observed outcome.
        Called after each bar's return is known (next bar open).

        This is the online learning component of the meta-orchestrator.
        """
        if not self.use_adaptive_weights:
            return

        regime = decision.regime
        correct = int(np.sign(realised_return)) == decision.final_signal if decision.final_signal != 0 else None
        if correct is None:
            return

        # Track per-agent accuracy
        sig_correct  = int(np.sign(realised_return)) == decision.signal_direction    if decision.signal_direction != 0 else None
        sent_correct = int(np.sign(realised_return)) == decision.sentiment_direction if decision.sentiment_direction != 0 else None

        if sig_correct is not None:
            self._regime_signal_accuracy[regime].append(float(sig_correct))
            if len(self._regime_signal_accuracy[regime]) > 50:
                self._regime_signal_accuracy[regime].pop(0)
        if sent_correct is not None:
            self._regime_sentiment_accuracy[regime].append(float(sent_correct))
            if len(self._regime_sentiment_accuracy[regime]) > 50:
                self._regime_sentiment_accuracy[regime].pop(0)

        # Adjust weights based on rolling accuracy (min 20 observations)
        sig_hist  = self._regime_signal_accuracy[regime]
        sent_hist = self._regime_sentiment_accuracy[regime]

        if len(sig_hist) >= 20 and len(sent_hist) >= 20:
            sig_acc  = np.mean(sig_hist[-20:])
            sent_acc = np.mean(sent_hist[-20:])
            total    = sig_acc + sent_acc + 1e-8
            new_w_sig  = sig_acc  / total
            new_w_sent = sent_acc / total
            # Smooth update (EMA with 0.95 memory)
            old_w_sig, old_w_sent = self._regime_weights.get(regime, (0.75, 0.25))
            self._regime_weights[regime] = (
                0.95 * old_w_sig  + 0.05 * new_w_sig,
                0.95 * old_w_sent + 0.05 * new_w_sent,
            )

    # ── Diagnostics ───────────────────────────────────────────────────────────
    def _log_decision(self, d: OrchestratorDecision) -> None:
        direction = {1: "BUY", -1: "SELL", 0: "FLAT"}.get(d.final_signal, "?")
        logger.info(
            "%s | %s %s | conf=%.3f | %s | sig=%+d(%.2f) sent=%+d(%.2f) "
            "agree=%s boost=%.2f",
            d.timestamp.strftime("%Y-%m-%d %H:%M"),
            self.instrument, direction,
            d.aggregate_confidence, d.regime_name,
            d.signal_direction, d.signal_confidence,
            d.sentiment_direction, d.sentiment_confidence,
            "✓" if d.agents_agree else "✗", d.agreement_boost,
        )

    def _log_backtest_summary(self, result: pd.DataFrame) -> None:
        n      = len(result)
        trades = (result["final_action"] == "trade").sum() if "final_action" in result.columns else 0
        agree  = result["agents_agree"].mean() if "agents_agree" in result.columns else 0.0
        agg_c  = result["aggregate_confidence"].mean() if "aggregate_confidence" in result.columns else 0.0
        logger.info(
            "Orchestrator backtest: %d bars | %d trades (%.1f%%) | "
            "agent_agree=%.1f%% | avg_conf=%.3f",
            n, trades, trades/n*100, agree*100, agg_c
        )

    # ── Decision log export ───────────────────────────────────────────────────
    def export_decisions(self) -> pd.DataFrame:
        """Export full decision history as DataFrame for XAI analysis."""
        if not self._decision_history:
            return pd.DataFrame()
        return pd.DataFrame([d.to_dict() for d in self._decision_history]).set_index("timestamp")

    def current_weights(self) -> dict:
        """Return current regime-conditioned weights (for monitoring)."""
        return {
            REGIME_NAMES[r]: {"signal": ws, "sentiment": wt}
            for r, (ws, wt) in self._regime_weights.items()
        }
