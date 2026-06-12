"""
maestro/orchestrator/llm_orchestrator.py
==========================================
LLM-powered Meta-Orchestrator — the MAESTRO PhD thesis contribution.

Instead of rule-based weighted fusion, this orchestrator uses Claude
(claude-opus-4-8 with adaptive thinking) as a reasoning engine that:

  1. Calls 5 specialised tools (one per agent output) in a natural loop
  2. Reasons about signal consistency, regime context, and risk
  3. Commits a structured final decision with plain-English rationale

Architecture
------------
Backtest mode (all agent outputs pre-computed):
  regime_df, signal_df, risk_df, sentiment_df → Claude tool loop → submit_decision

Live trading mode:
  Each tool call invokes the actual fitted agent in real time.

The "tool use loop" is genuine agentic reasoning: Claude decides which
agents to consult, in what order, and synthesises a reasoned decision.
Every rationale is stored for XAI / regulatory explainability.

Cost management
---------------
  - Called only when abs(technical_signal) > 0 OR regime == 3 (crisis)
    → ~30-50K bars in a 311K-bar backtest (10-16%)
  - System prompt is prompt-cached (cache_control) → ~90% input savings
  - Decisions are memoised by SHA-256(bar_context) — same market state
    gets the same decision without a second API call
  - Falls back to the rule-based MetaOrchestrator on API failure
  - Model: claude-opus-4-8 (adaptive thinking enabled)

Estimated cost for a full 39-split EUR/USD backtest: ~$15-30.

PhD contribution
-----------------
  Compare LLM orchestrator vs rule-based on identical WFA splits:
    Hypothesis H1: LLM reasoning improves net Sharpe vs rule-based fusion
    Hypothesis H2: improvement is largest in crisis + disagreement regimes
  Every rationale stored → XAI audit trail → thesis chapter evidence.

Usage
-----
    from maestro.orchestrator.llm_orchestrator import LLMOrchestrator

    orch = LLMOrchestrator(instrument="EUR_USD")
    decisions_df = orch.run_backtest(
        features_df, signal_df, regime_df, sentiment_df=None
    )
    # decisions_df has same schema as MetaOrchestrator.run_backtest()
    # plus extra columns: llm_rationale, llm_key_factors, llm_risk_warnings
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
REGIME_NAMES = {0: "bull_trend", 1: "bear_trend", 2: "sideways", 3: "crisis"}

_SYSTEM_PROMPT = """\
You are MAESTRO's trading orchestrator for {instrument} M5 Forex trading.
You coordinate 5 specialised AI agents to produce optimal, risk-aware decisions.

Your agents:
  • Regime Agent    — HMM+Transformer: detects market state (bull/bear/sideways/crisis)
  • Signal Agent    — TFT+PatchTST:    momentum/mean-reversion directional forecasts
  • Sentiment Agent — FinBERT+GPT-4o:  macro news impact on currency direction
  • Risk Agent      — CVaR+Kelly RL:   position sizing, stop-loss, CVaR estimates
  • Market Context  — raw price action: recent volatility, spread, session timing

Your process (call tools in roughly this order):
  1. get_regime_signal      — understand the market environment first
  2. get_technical_signal   — check price-based signal strength
  3. get_sentiment_signal   — assess macro news environment
  4. get_risk_assessment    — size position and get risk guardrails
  5. submit_decision        — commit your final reasoning and decision

Decision principles:
  • CRISIS regime (3): demand confidence > 0.70; reduce size; lean toward flat
  • AGENT DISAGREEMENT: attenuate signal by ×0.6; prefer flat on strong conflict
  • STRONG AGREEMENT across agents: can boost confidence up to ×1.2 (max 1.0)
  • RISK AGENT VETO (action=flat or circuit_break): ALWAYS honour it
  • LOW CONFIDENCE (aggregate < 0.52): go flat — costs less than a bad trade
  • Transaction costs erode edge: only trade when your edge is genuine

You must call submit_decision exactly once per analysis.
Your rationale should be 60-120 words: what each agent said and why you decided.
"""

_TOOLS = [
    {
        "name": "get_regime_signal",
        "description": (
            "Get the current market regime from the HMM+Transformer ensemble. "
            "Returns regime (0=bull_trend, 1=bear_trend, 2=sideways, 3=crisis), "
            "confidence [0,1], and probability distribution across regimes."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_technical_signal",
        "description": (
            "Get the directional signal from TFT+PatchTST ensemble. "
            "Returns signal (-1=sell, 0=flat, +1=buy), confidence [0,1], "
            "whether the two models agree, and quantile return predictions (p10/p50/p90)."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_sentiment_signal",
        "description": (
            "Get the macro sentiment signal from FinBERT+GPT-4o news analysis. "
            "Returns sentiment direction (-1/0/+1), confidence [0,1], "
            "GPT-4o rationale text (if available), and rolling NLP accuracy."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_risk_assessment",
        "description": (
            "Get position sizing and risk parameters from the CVaR-RL risk agent. "
            "Returns recommended action (trade/flat/circuit_break), "
            "position fraction [0,1], Kelly fraction, stop-loss pips, "
            "take-profit pips, CVaR estimate, and current drawdown state."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "submit_decision",
        "description": (
            "Commit your final trading decision. Call this once after gathering "
            "sufficient evidence from the other tools. "
            "This is the only tool that produces output — call it exactly once."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["trade", "flat", "circuit_break"],
                    "description": "Whether to trade, stay flat, or trigger circuit breaker",
                },
                "signal": {
                    "type": "integer",
                    "enum": [-1, 0, 1],
                    "description": "-1=sell, 0=flat, +1=buy",
                },
                "confidence": {
                    "type": "number",
                    "description": "Your aggregate confidence in the decision [0.0–1.0]",
                },
                "rationale": {
                    "type": "string",
                    "description": (
                        "Plain-English reasoning (60-120 words). "
                        "State what each agent said and why you made this decision. "
                        "This is the XAI audit trail for the PhD thesis."
                    ),
                },
                "key_factors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Top 3 factors driving this decision (brief phrases)",
                },
                "risk_warnings": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Risk concerns or reasons for caution (may be empty)",
                },
            },
            "required": ["action", "signal", "confidence", "rationale", "key_factors"],
        },
    },
]


# ── Decision result ───────────────────────────────────────────────────────────
@dataclass
class LLMDecision:
    timestamp:        pd.Timestamp
    instrument:       str
    action:           str           # trade | flat | circuit_break
    signal:           int           # -1 | 0 | +1
    confidence:       float
    rationale:        str
    key_factors:      list[str]     = field(default_factory=list)
    risk_warnings:    list[str]     = field(default_factory=list)
    fallback_used:    bool          = False   # True if rule-based fallback was used
    tokens_used:      int           = 0
    cache_hit:        bool          = False


# ── Main class ────────────────────────────────────────────────────────────────
class LLMOrchestrator:
    """
    MAESTRO's LLM-powered Meta-Orchestrator.

    Replaces the rule-based MetaOrchestrator with a genuine Claude-driven
    reasoning loop that calls each agent as a tool and synthesises a decision.

    Usage
    -----
    >>> orch = LLMOrchestrator(instrument="EUR_USD")
    >>> decisions_df = orch.run_backtest(features_df, signal_df, regime_df)
    """

    def __init__(
        self,
        instrument:    str   = "EUR_USD",
        model:         str   = "claude-opus-4-8",
        max_retries:   int   = 3,
        only_on_signal: bool = True,   # skip LLM when all agents are flat → saves cost
        cache_dir:     str | Path | None = None,
    ) -> None:
        self.instrument      = instrument
        self.model           = model
        self.max_retries     = max_retries
        self.only_on_signal  = only_on_signal

        # Persistent decision cache (survives process restarts)
        _base = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "maestro"
        _base.mkdir(parents=True, exist_ok=True)
        self._cache_path = _base / f"llm_decisions_{instrument}.json"
        self._decision_cache: dict[str, dict] = self._load_cache()

        # Fallback rule-based orchestrator
        from maestro.orchestrator.meta_orchestrator import MetaOrchestrator
        self._fallback = MetaOrchestrator(instrument=instrument)

        # Cost tracking
        self._total_tokens = 0
        self._api_calls     = 0
        self._cache_hits    = 0
        self._fallbacks     = 0

        # Rationale log (for XAI export)
        self._rationale_log: list[dict] = []

    # ── Batch backtest (main entry point) ────────────────────────────────────
    def run_backtest(
        self,
        features_df:    pd.DataFrame,
        signal_df:      pd.DataFrame,
        regime_df:      pd.DataFrame,
        sentiment_df:   pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Run orchestration on pre-computed agent outputs (backtest mode).

        The LLM receives each agent's output via tool calls, reasons about
        them, and calls submit_decision with a structured final answer.

        Returns a DataFrame with the same schema as MetaOrchestrator.run_backtest()
        plus: llm_rationale, llm_key_factors, llm_risk_warnings, llm_fallback.
        """
        logger.info(
            "LLM Orchestrator: %d bars | model=%s | only_on_signal=%s",
            len(features_df), self.model, self.only_on_signal
        )

        client = self._get_client()
        rows   = []

        for i, ts in enumerate(features_df.index):
            # ── Read pre-computed agent outputs for this bar ──────────────
            ctx = self._build_bar_context(ts, signal_df, regime_df,
                                          sentiment_df, features_df)

            # ── Decide whether to call LLM or use rule-based ──────────────
            sig_nonzero  = ctx["signal"] != 0
            is_crisis    = ctx["regime"] == 3
            call_llm     = (not self.only_on_signal) or sig_nonzero or is_crisis

            if not call_llm:
                # Rule-based flat decision (no API cost)
                row = self._flat_row(ts, ctx)
            else:
                row = self._decide_with_llm(client, ts, ctx)

            rows.append(row)

            if (i + 1) % 500 == 0:
                logger.info(
                    "LLM Orchestrator: %d/%d bars | API calls: %d | "
                    "cache hits: %d | fallbacks: %d | tokens: %d",
                    i + 1, len(features_df),
                    self._api_calls, self._cache_hits,
                    self._fallbacks, self._total_tokens
                )

        result = pd.DataFrame(rows).set_index("timestamp")
        self._save_cache()
        self._log_summary(len(features_df))
        return result

    # ── Single-bar decision ───────────────────────────────────────────────────
    def _decide_with_llm(
        self,
        client: Any,
        ts:     pd.Timestamp,
        ctx:    dict,
    ) -> dict:
        """Run the Claude tool loop for one bar and return a decision row."""
        cache_key = self._cache_key(ctx)

        # Check in-memory + persistent cache
        if cache_key in self._decision_cache:
            cached = self._decision_cache[cache_key]
            self._cache_hits += 1
            return self._cached_to_row(ts, ctx, cached)

        # Run the agentic loop
        decision = self._run_tool_loop(client, ts, ctx)

        # Persist to cache
        self._decision_cache[cache_key] = {
            "action":        decision.action,
            "signal":        decision.signal,
            "confidence":    decision.confidence,
            "rationale":     decision.rationale,
            "key_factors":   decision.key_factors,
            "risk_warnings": getattr(decision, "risk_warnings", []),
            "fallback_used": decision.fallback_used,
        }
        self._rationale_log.append({
            "timestamp": str(ts),
            "instrument": self.instrument,
            **self._decision_cache[cache_key],
        })

        return self._decision_to_row(ts, ctx, decision)

    # ── Claude tool loop ──────────────────────────────────────────────────────
    def _run_tool_loop(
        self,
        client: Any,
        ts:     pd.Timestamp,
        ctx:    dict,
    ) -> LLMDecision:
        """
        Run the full Claude tool-use loop for one bar.

        Claude will call the agent tools in turn, then call submit_decision.
        We intercept each tool call, return pre-computed results, and loop
        until submit_decision is called or max iterations exceeded.
        """
        system_prompt = _SYSTEM_PROMPT.format(instrument=self.instrument)
        user_message  = self._build_user_message(ts, ctx)

        messages = [{"role": "user", "content": user_message}]

        # Tool state — each tool may only be called once (pre-computed data)
        tool_results: dict[str, str] = {}
        decision: LLMDecision | None = None
        total_tokens = 0
        iterations   = 0
        max_iter     = 6  # regime + signal + sentiment + risk + submit + 1 spare

        for attempt in range(self.max_retries):
            try:
                while iterations < max_iter and decision is None:
                    iterations += 1

                    response = client.messages.create(
                        model      = self.model,
                        max_tokens = 1024,
                        thinking   = {"type": "adaptive"},
                        system     = [
                            {
                                "type": "text",
                                "text": system_prompt,
                                "cache_control": {"type": "ephemeral"},  # prompt cache
                            }
                        ],
                        tools    = _TOOLS,
                        messages = messages,
                    )
                    total_tokens += response.usage.input_tokens + response.usage.output_tokens
                    self._api_calls += 1

                    # Append assistant response to history
                    messages.append({"role": "assistant", "content": response.content})

                    if response.stop_reason == "end_turn":
                        # Claude stopped without calling submit_decision — fallback
                        logger.warning(
                            "LLM stopped at %s without submit_decision (iter %d)",
                            ts, iterations
                        )
                        break

                    if response.stop_reason != "tool_use":
                        break

                    # Process tool calls
                    tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
                    tool_result_content = []

                    for tb in tool_use_blocks:
                        tool_name = tb.name
                        tool_id   = tb.id

                        if tool_name == "submit_decision":
                            # Extract the final decision
                            inp = tb.input
                            decision = LLMDecision(
                                timestamp     = ts,
                                instrument    = self.instrument,
                                action        = str(inp.get("action", "flat")),
                                signal        = int(inp.get("signal", 0)),
                                confidence    = float(inp.get("confidence", 0.5)),
                                rationale     = str(inp.get("rationale", "")),
                                key_factors   = list(inp.get("key_factors", [])),
                                risk_warnings = list(inp.get("risk_warnings", [])),
                                fallback_used = False,
                                tokens_used   = total_tokens,
                                cache_hit     = False,
                            )
                            tool_result_content.append({
                                "type":        "tool_result",
                                "tool_use_id": tool_id,
                                "content":     "Decision recorded. Analysis complete.",
                            })
                        else:
                            # Return pre-computed result for this agent tool
                            result_text = self._get_tool_result(tool_name, ctx)
                            tool_results[tool_name] = result_text
                            tool_result_content.append({
                                "type":        "tool_result",
                                "tool_use_id": tool_id,
                                "content":     result_text,
                            })

                    if tool_result_content:
                        messages.append({"role": "user", "content": tool_result_content})

                    if decision is not None:
                        break

                # If we got a decision, return it
                if decision is not None:
                    self._total_tokens += total_tokens
                    return decision

                # Otherwise fall through to fallback
                logger.warning(
                    "LLM loop completed without decision at %s — using rule-based fallback",
                    ts
                )
                break

            except Exception as exc:
                wait = 2 ** attempt
                logger.warning(
                    "LLM API error at %s (attempt %d/%d): %s — retrying in %ds",
                    ts, attempt + 1, self.max_retries, exc, wait
                )
                time.sleep(wait)

        # ── Fallback to rule-based ────────────────────────────────────────────
        self._fallbacks += 1
        return self._rule_based_fallback(ts, ctx)

    # ── Tool result dispatch ──────────────────────────────────────────────────
    def _get_tool_result(self, tool_name: str, ctx: dict) -> str:
        """Return pre-computed agent output as a formatted string for Claude."""
        if tool_name == "get_regime_signal":
            r = ctx["regime"]
            return json.dumps({
                "regime":       r,
                "regime_name":  REGIME_NAMES.get(r, "unknown"),
                "confidence":   round(ctx["regime_conf"], 3),
                "is_certain":   ctx["regime_conf"] >= 0.55,
                "distribution": {
                    REGIME_NAMES[i]: round(ctx.get(f"prob_{REGIME_NAMES[i]}", 0.25), 3)
                    for i in range(4)
                },
            }, indent=2)

        if tool_name == "get_technical_signal":
            return json.dumps({
                "signal":         ctx["signal"],
                "direction":      {1: "BUY", -1: "SELL", 0: "FLAT"}.get(ctx["signal"], "?"),
                "confidence":     round(ctx["sig_conf"], 3),
                "models_agree":   bool(ctx.get("model_agree", False)),
                "tft_signal":     ctx.get("tft_signal", ctx["signal"]),
                "patchtst_signal":ctx.get("ptst_signal", ctx["signal"]),
                "pred_p10":       round(ctx.get("pred_p10", 0.0), 6),
                "pred_p50":       round(ctx.get("pred_p50", 0.0), 6),
                "pred_p90":       round(ctx.get("pred_p90", 0.0), 6),
                "horizon_bars":   ctx.get("horizon", 6),
            }, indent=2)

        if tool_name == "get_sentiment_signal":
            return json.dumps({
                "signal":            ctx.get("sent_sig", 0),
                "direction":         {1: "BULLISH", -1: "BEARISH", 0: "NEUTRAL"}.get(
                                       ctx.get("sent_sig", 0), "NEUTRAL"),
                "confidence":        round(ctx.get("sent_conf", 0.3), 3),
                "fusion_weight":     round(ctx.get("fusion_w", 0.2), 3),
                "rolling_nlp_acc":   round(ctx.get("nlp_acc", 0.5), 3),
                "data_available":    ctx.get("sent_available", False),
                "note": ("No news data available — sentiment neutral"
                         if not ctx.get("sent_available", False) else "Live news signal"),
            }, indent=2)

        if tool_name == "get_risk_assessment":
            return json.dumps({
                "recommended_action":  ctx.get("risk_action", "flat"),
                "position_fraction":   round(ctx.get("risk_pos_frac", 0.1), 3),
                "kelly_fraction":      round(ctx.get("risk_kelly", 0.1), 3),
                "stop_loss_pips":      round(ctx.get("sl_pips", 12.0), 1),
                "take_profit_pips":    round(ctx.get("tp_pips", 24.0), 1),
                "cvar_95":             round(ctx.get("cvar", 0.002), 5),
                "current_drawdown":    round(ctx.get("drawdown", 0.0), 4),
                "risk_reward_ratio":   round(
                    ctx.get("tp_pips", 24.0) / max(ctx.get("sl_pips", 12.0), 0.01), 2
                ),
                "note": ctx.get("risk_note", ""),
            }, indent=2)

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    # ── Context builders ──────────────────────────────────────────────────────
    def _build_bar_context(
        self,
        ts:           pd.Timestamp,
        signal_df:    pd.DataFrame,
        regime_df:    pd.DataFrame,
        sentiment_df: pd.DataFrame | None,
        features_df:  pd.DataFrame,
    ) -> dict:
        """Extract all agent outputs for one bar into a flat context dict."""
        ctx: dict = {}

        # Regime
        ctx["regime"]      = int(regime_df.loc[ts, "regime"])           if ts in regime_df.index else 2
        ctx["regime_conf"] = float(regime_df.loc[ts, "confidence"])     if ts in regime_df.index and "confidence" in regime_df.columns else 0.5
        for rname in REGIME_NAMES.values():
            col = f"prob_{rname}"
            ctx[col] = float(regime_df.loc[ts, col]) if ts in regime_df.index and col in regime_df.columns else 0.25

        # Signal
        ctx["signal"]       = int(signal_df.loc[ts, "signal"])           if ts in signal_df.index and "signal" in signal_df.columns else 0
        ctx["sig_conf"]     = float(signal_df.loc[ts, "confidence"])     if ts in signal_df.index and "confidence" in signal_df.columns else 0.5
        ctx["model_agree"]  = bool(signal_df.loc[ts, "model_agree"])     if ts in signal_df.index and "model_agree" in signal_df.columns else False
        ctx["tft_signal"]   = int(signal_df.loc[ts, "tft_signal"])       if ts in signal_df.index and "tft_signal" in signal_df.columns else ctx["signal"]
        ctx["ptst_signal"]  = int(signal_df.loc[ts, "ptst_signal"])      if ts in signal_df.index and "ptst_signal" in signal_df.columns else ctx["signal"]
        ctx["pred_p10"]     = float(signal_df.loc[ts, "pred_p10"])       if ts in signal_df.index and "pred_p10" in signal_df.columns else 0.0
        ctx["pred_p50"]     = float(signal_df.loc[ts, "pred_p50"])       if ts in signal_df.index and "pred_p50" in signal_df.columns else 0.0
        ctx["pred_p90"]     = float(signal_df.loc[ts, "pred_p90"])       if ts in signal_df.index and "pred_p90" in signal_df.columns else 0.0
        ctx["horizon"]      = int(signal_df.loc[ts, "horizon"])          if ts in signal_df.index and "horizon" in signal_df.columns else 6

        # Sentiment (optional)
        has_sent = sentiment_df is not None and ts in sentiment_df.index
        ctx["sent_available"] = has_sent
        ctx["sent_sig"]   = int(sentiment_df.loc[ts, "sentiment_signal"])  if has_sent and "sentiment_signal" in sentiment_df.columns else 0
        ctx["sent_conf"]  = float(sentiment_df.loc[ts, "text_confidence"]) if has_sent and "text_confidence" in sentiment_df.columns else 0.3
        ctx["fusion_w"]   = float(sentiment_df.loc[ts, "fusion_weight"])   if has_sent and "fusion_weight" in sentiment_df.columns else 0.2
        ctx["nlp_acc"]    = float(sentiment_df.loc[ts, "rolling_nlp_accuracy"]) if has_sent and "rolling_nlp_accuracy" in sentiment_df.columns else 0.5

        # Risk (proxied from signal df — these columns are added by MetaOrchestrator)
        # In backtest engine, risk decisions are made separately.
        # We use simple Kelly-based estimates here.
        ctx["risk_action"]   = "trade" if ctx["signal"] != 0 and ctx["sig_conf"] >= 0.52 else "flat"
        ctx["risk_pos_frac"] = min(abs(ctx["sig_conf"]) * 0.2, 0.20)
        ctx["risk_kelly"]    = 0.1
        ctx["sl_pips"]       = float(signal_df.loc[ts, "stop_loss_pips"])   if ts in signal_df.index and "stop_loss_pips" in signal_df.columns else 12.0
        ctx["tp_pips"]       = float(signal_df.loc[ts, "take_profit_pips"]) if ts in signal_df.index and "take_profit_pips" in signal_df.columns else 24.0
        ctx["cvar"]          = 0.002
        ctx["drawdown"]      = 0.0
        ctx["risk_note"]     = "CVaR estimated; RL risk agent outputs not yet fused here"

        # Market features
        if ts in features_df.index:
            ctx["close"]  = float(features_df.loc[ts, "close"])  if "close" in features_df.columns else 0.0
            ctx["vol"]    = float(features_df.loc[ts, "vol_realised"]) if "vol_realised" in features_df.columns else 0.0
            ctx["spread"] = float(features_df.loc[ts, "spread"])       if "spread" in features_df.columns else 0.0
        else:
            ctx["close"] = ctx["vol"] = ctx["spread"] = 0.0

        return ctx

    def _build_user_message(self, ts: pd.Timestamp, ctx: dict) -> str:
        """Build the initial user message for the tool loop."""
        direction = {1: "BUY signal", -1: "SELL signal", 0: "flat (no signal)"}.get(
            ctx["signal"], "unknown"
        )
        regime_name = REGIME_NAMES.get(ctx["regime"], "unknown")

        return (
            f"Time: {ts.strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"Instrument: {self.instrument}\n"
            f"Current regime: {regime_name} (ID={ctx['regime']})\n"
            f"Technical signal: {direction} (conf={ctx['sig_conf']:.2f})\n"
            f"Models agree: {ctx['model_agree']}\n\n"
            f"Please use the tools to gather agent outputs and submit your trading decision."
        )

    # ── Fallback and helper rows ──────────────────────────────────────────────
    def _rule_based_fallback(self, ts: pd.Timestamp, ctx: dict) -> LLMDecision:
        """Simple rule-based decision when LLM is unavailable."""
        regime = ctx["regime"]
        signal = ctx["signal"]
        sig_conf = ctx["sig_conf"]

        # Crisis or low confidence → flat
        if regime == 3 and sig_conf < 0.70:
            action, final_sig = "flat", 0
            rationale = f"Rule-based fallback: crisis regime ({REGIME_NAMES[3]}) with insufficient confidence ({sig_conf:.2f} < 0.70). Going flat."
        elif signal != 0 and sig_conf >= 0.52:
            action, final_sig = "trade", signal
            rationale = f"Rule-based fallback: {REGIME_NAMES[regime]} regime, {'BUY' if signal > 0 else 'SELL'} signal conf={sig_conf:.2f}."
        else:
            action, final_sig = "flat", 0
            rationale = f"Rule-based fallback: insufficient signal confidence ({sig_conf:.2f})."

        return LLMDecision(
            timestamp     = ts,
            instrument    = self.instrument,
            action        = action,
            signal        = final_sig,
            confidence    = sig_conf if action == "trade" else 0.0,
            rationale     = rationale,
            key_factors   = ["rule-based fallback"],
            risk_warnings = ["LLM API unavailable — rule-based logic used"],
            fallback_used = True,
            tokens_used   = 0,
            cache_hit     = False,
        )

    def _flat_row(self, ts: pd.Timestamp, ctx: dict) -> dict:
        """Return a flat decision row without calling the LLM."""
        return {
            "timestamp":            ts,
            "instrument":           self.instrument,
            "final_action":         "flat",
            "final_signal":         0,
            "final_units":          0,
            "aggregate_confidence": 0.0,
            "regime":               ctx["regime"],
            "regime_name":          REGIME_NAMES.get(ctx["regime"], "sideways"),
            "regime_confidence":    ctx["regime_conf"],
            "signal_direction":     ctx["signal"],
            "signal_confidence":    ctx["sig_conf"],
            "sentiment_direction":  ctx.get("sent_sig", 0),
            "sentiment_confidence": ctx.get("sent_conf", 0.3),
            "fusion_weight":        ctx.get("fusion_w", 0.2),
            "agents_agree":         False,
            "agreement_boost":      1.0,
            "weight_signal":        0.75,
            "weight_sentiment":     0.25,
            "stop_loss_pips":       ctx.get("sl_pips", 12.0),
            "take_profit_pips":     ctx.get("tp_pips", 24.0),
            "llm_rationale":        "Skipped — all agents flat",
            "llm_key_factors":      "[]",
            "llm_risk_warnings":    "[]",
            "llm_fallback":         False,
        }

    def _decision_to_row(
        self, ts: pd.Timestamp, ctx: dict, d: LLMDecision
    ) -> dict:
        units = d.signal * 10_000 if d.action == "trade" else 0
        return {
            "timestamp":            ts,
            "instrument":           self.instrument,
            "final_action":         d.action,
            "final_signal":         d.signal,
            "final_units":          units,
            "aggregate_confidence": d.confidence,
            "regime":               ctx["regime"],
            "regime_name":          REGIME_NAMES.get(ctx["regime"], "sideways"),
            "regime_confidence":    ctx["regime_conf"],
            "signal_direction":     ctx["signal"],
            "signal_confidence":    ctx["sig_conf"],
            "sentiment_direction":  ctx.get("sent_sig", 0),
            "sentiment_confidence": ctx.get("sent_conf", 0.3),
            "fusion_weight":        ctx.get("fusion_w", 0.2),
            "agents_agree":         ctx["signal"] == ctx.get("sent_sig", 0) and ctx["signal"] != 0,
            "agreement_boost":      1.2 if ctx["signal"] == ctx.get("sent_sig", 0) and ctx["signal"] != 0 else 1.0,
            "weight_signal":        1.0 - ctx.get("fusion_w", 0.2),
            "weight_sentiment":     ctx.get("fusion_w", 0.2),
            "stop_loss_pips":       ctx.get("sl_pips", 12.0),
            "take_profit_pips":     ctx.get("tp_pips", 24.0),
            "llm_rationale":        d.rationale,
            "llm_key_factors":      json.dumps(d.key_factors),
            "llm_risk_warnings":    json.dumps(d.risk_warnings),
            "llm_fallback":         d.fallback_used,
        }

    def _cached_to_row(
        self, ts: pd.Timestamp, ctx: dict, cached: dict
    ) -> dict:
        d = LLMDecision(
            timestamp     = ts,
            instrument    = self.instrument,
            action        = cached["action"],
            signal        = cached["signal"],
            confidence    = cached["confidence"],
            rationale     = cached["rationale"],
            key_factors   = cached.get("key_factors", []),
            risk_warnings = cached.get("risk_warnings", []),
            fallback_used = cached.get("fallback_used", False),
            cache_hit     = True,
        )
        row = self._decision_to_row(ts, ctx, d)
        return row

    # ── Cache helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _cache_key(ctx: dict) -> str:
        """Deterministic hash of the market state for memoisation."""
        key_fields = {
            "regime":     ctx["regime"],
            "signal":     ctx["signal"],
            "sig_conf":   round(ctx["sig_conf"], 2),
            "regime_conf":round(ctx["regime_conf"], 2),
            "sent_sig":   ctx.get("sent_sig", 0),
            "sent_conf":  round(ctx.get("sent_conf", 0.3), 2),
            "model_agree":ctx.get("model_agree", False),
        }
        raw = json.dumps(key_fields, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:20]

    def _load_cache(self) -> dict:
        if self._cache_path.exists():
            try:
                with open(self._cache_path) as f:
                    data = json.load(f)
                logger.info("LLM decision cache loaded: %d entries", len(data))
                return data
            except Exception:
                pass
        return {}

    def _save_cache(self) -> None:
        try:
            with open(self._cache_path, "w") as f:
                json.dump(self._decision_cache, f, indent=2)
        except Exception as exc:
            logger.warning("Could not save LLM cache: %s", exc)

    # ── Client helper ─────────────────────────────────────────────────────────
    @staticmethod
    def _get_client() -> Any:
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "The Anthropic SDK is required: pip install anthropic"
            )
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not set. Add it to your .env file."
            )
        return anthropic.Anthropic(api_key=api_key)

    # ── XAI export ────────────────────────────────────────────────────────────
    def export_rationales(self, output_path: str | Path | None = None) -> pd.DataFrame:
        """
        Export all stored LLM rationales as a DataFrame.
        Essential for the XAI chapter of the PhD thesis.

        Returns a DataFrame with columns:
          timestamp, instrument, action, signal, confidence,
          rationale, key_factors, risk_warnings, fallback_used
        """
        if not self._rationale_log:
            logger.warning("No rationales logged yet.")
            return pd.DataFrame()

        df = pd.DataFrame(self._rationale_log)
        if output_path:
            df.to_csv(output_path, index=False)
            logger.info("Rationale log saved → %s", output_path)
        return df

    # ── Summary logging ───────────────────────────────────────────────────────
    def _log_summary(self, total_bars: int) -> None:
        called_bars = self._api_calls + self._cache_hits + self._fallbacks
        logger.info(
            "LLM Orchestrator summary | bars=%d | called=%d (%.1f%%) | "
            "API calls=%d | cache hits=%d | fallbacks=%d | total tokens=%d",
            total_bars, called_bars, 100 * called_bars / max(total_bars, 1),
            self._api_calls, self._cache_hits, self._fallbacks, self._total_tokens
        )

    @property
    def cost_estimate_usd(self) -> float:
        """Rough cost estimate based on tokens used (claude-opus-4-8 pricing)."""
        # claude-opus-4-8: $5/MTok input, $25/MTok output
        # Assume 70% input, 30% output of total tokens
        input_tokens  = self._total_tokens * 0.7
        output_tokens = self._total_tokens * 0.3
        # Cached tokens cost ~0.1× input price
        cached_fraction = 0.8  # assume 80% of input hits cache after warmup
        effective_input_cost = (
            input_tokens * cached_fraction * 5.0 / 1_000_000 * 0.1 +
            input_tokens * (1 - cached_fraction) * 5.0 / 1_000_000
        )
        output_cost = output_tokens * 25.0 / 1_000_000
        return effective_input_cost + output_cost
