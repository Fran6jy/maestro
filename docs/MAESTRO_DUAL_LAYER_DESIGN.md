# MAESTRO — Dual-Layer Design: Scientific Core + Explainable Demo

**Purpose:** Define two strictly separated layers over the existing MAESTRO system:
- **Layer 1 — Scientific core**: the trading logic and its thesis-grade evaluation.
- **Layer 2 — Experience layer**: a read-only interpretation/visualisation of the
  core's outputs, for viva, video, and portfolio.

**Non-negotiable invariant:** Layer 2 *interprets* Layer 1 outputs. It has no write
path into trading logic. This is enforced structurally (see §3, §5), not by
convention — the demo cannot change a single trade.

**Honest framing:** Empirical testing of the current signal gives ~51.7% directional
hit rate (not significant vs 50%). The hypothesis (§2.1) and metrics (§2.4) are
therefore built around **robustness and risk-adjusted behaviour**, where the
architecture can demonstrably act, rather than raw-return outperformance, which it
cannot presuppose. A rigorous negative result is an acceptable, publishable outcome.

---

## 1. System overview (unified architecture)

### 1.1 The two layers at a glance

```
  ┌──────────────────────── LAYER 1: SCIENTIFIC CORE ─────────────────────────┐
  │                                                                            │
  │   DATA  →  REGIME  →  SIGNAL  →  RISK  →  ORCHESTRATOR  →  DECISION[]       │
  │  (causal)  (Agent1)  (Agent2)  (Agent3)   (fusion+guards)  (immutable)     │
  │                          │                                      │          │
  │                      SENTIMENT (Agent 5, optional)              │          │
  │                                                                 │          │
  │                                          EXECUTION / BACKTEST ◄──┤          │
  │                                          + EVALUATION FRAMEWORK ◄┤          │
  └─────────────────────────────────────────────────────────────────┼─────────┘
                                                                      │ (read-only)
  ┌──────────────────────── LAYER 2: EXPERIENCE ─────────────────────▼─────────┐
  │   SHADOW EXPLAINER  →  {State, Decision, Reason}                            │
  │        │                                                                   │
  │        ├─► Market Replay   ├─► Control Room   ├─► Decision Timeline         │
  │        └─► Emotion Layer (calm / uncertain / dangerous / trending)         │
  └────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 How the agents interact (Layer 1)

1. **Regime Agent (Agent 1, HMM + Transformer)** sets the market "weather" — a
   regime label + confidence. It runs first; every other agent receives it.
2. **Signal Agent (Agent 2, TFT + PatchTST)** produces a directional view,
   *conditioned* on the regime (regime is an input feature, not a gate).
3. **Sentiment Agent (Agent 5, FinBERT + GPT-4o)**, when news is available, adds a
   macro-sentiment view.
4. **Risk Agent (Agent 3, CVaR-RL + Kelly)** sizes the position under risk
   constraints, scaled down in adverse regimes.
5. **Meta-Orchestrator** fuses the views, applies live-readiness guards, and emits
   the immutable `Decision`. (Two variants exist: a deterministic rule-based fusion,
   and an optional LLM reasoning orchestrator using claude-opus-4-8 with tool use.)
6. **Execution Agent (Agent 4)** applies the cost model and simulates/places fills.

### 1.3 How data flows (causal, leakage-controlled)

```
For each bar t (and for each WFA split during training):
   features(≤t)  ─► Agent1.predict(t)  ─► regime_state
   features(≤t), regime_state ─► Agent2.signal(t) ─► directional view + confidence
   news(≤t), regime_state     ─► Agent5.score(t)  ─► sentiment view  (optional)
   views, regime_state, equity ─► Agent3.size(t)   ─► position + risk params
   all of the above            ─► Orchestrator.fuse + guard ─► Decision (frozen)
   Decision ─► Execution (costs, fills) ─► equity update
   Decision ─► Evaluation collector (metrics; no feedback)
   Decision ─► Layer 2 shadow explainer (text; no feedback)
```

Everything an agent sees at `t` is available at `t`. Fitting happens on the training
window only; the test window is never used to fit a parameter or scaler.

### 1.4 Where the interpretation layer attaches

At exactly one point: the **immutable `Decision` object** emitted by the
orchestrator. Layer 2 subscribes to the Decision stream (live) or reads the persisted
Decision log (backtest replay). It never touches agents, features, or models. Because
`Decision` is frozen and the explainer is a pure function, the attachment is provably
side-effect-free.

---

## 2. Thesis-grade scientific framework (Layer 1)

### 2.1 Research hypothesis

**Primary hypothesis (H1).**
> Multi-agent decomposition with shared regime conditioning improves the
> **robustness and risk-adjusted** behaviour of the system in non-stationary
> markets, relative to (a) ablated variants and (b) naive baselines — measured by
> Sortino, maximum drawdown, and CVaR, net of costs.

**Secondary hypotheses.**
- **H2 (regime value):** removing regime conditioning degrades risk-adjusted metrics
  more in volatile/crisis regimes than in calm regimes.
- **H3 (interpretability fidelity):** the shadow-explanation layer reproduces the
  causal drivers of each decision without altering any decision (faithfulness +
  zero-effect verification).

**Falsifiability.** If ablations show no statistically significant difference in
risk-adjusted metrics across splits, H1 is rejected and the thesis reports a negative
result — that multi-agent decomposition did not improve robustness on this data at
this frequency. This is a valid scientific outcome and is explicitly anticipated.

**What MAESTRO does *not* claim.** It does not claim to predict price direction
profitably (current evidence: ~51.7% hit rate). The contribution is methodological:
the decomposition, the regime conditioning, the evaluation rigour, and the faithful
interpretability layer.

### 2.2 Methodology

**Walk-forward validation (expanding window).** Train on `[T0, T1]`, test on
`[T1+embargo, T2]`, then advance. The training window grows each step, mirroring live
deployment where older data is retained. Implemented in `data/validation/wfa.py`.

**Training/testing separation.** Per split: all fitting (regime models, feature
scalers, signal models, risk parameters) occurs strictly within `[T0, T1]`. Fitted
objects are frozen before any test-window inference. No global normalisation across
the full dataset.

**Embargo / purging.** A gap (`embargo_days`) separates train and test so that
overlapping label horizons cannot leak future information into training labels
(Lopez de Prado, *Advances in Financial Machine Learning*, ch. 7).

**Regime conditioning logic.** The regime label is injected as a static covariate to
the signal models (TFT static input, PatchTST feature). Models thus learn
regime-specific temporal patterns. Crucially, regime is used to *condition* the
signal, and to *scale risk* later — it is not a hard pre-filter on signals, which
keeps the ablation clean (you can remove regime conditioning without removing the
signal model).

### 2.3 Experimental design

**Baselines.**
- **Buy-and-hold** the instrument (passive benchmark).
- **Naive always-on** signal with equal sizing, no regime, no guards (floor).
- **Rebuilt MSc strategies** (SMA, contrarian, Bollinger, linear and logistic
  regression) re-run in this harness — see `backtesting/baselines.py` and
  `docs/THESIS_NARRATIVE.md`. The MSc report's own figures (Sharpe 0.599, hit
  rate 37.56%) come from two unrelated experiments and do not reproduce, so they
  are cited only as reported history, never as a comparable benchmark.

**Ablation studies** (identical WFA, one component removed at a time; paired
across-split comparison via Wilcoxon signed-rank):

| Variant | Removed | Tests |
|---|---|---|
| Full | — | baseline system |
| −Regime | regime fixed to neutral | H2: does regime conditioning help? |
| −Risk | Kelly/CVaR → fixed unit sizing | does risk sizing cut drawdown/CVaR? |
| −Orchestrator | naive signal pass-through | does fusion + guards add value? |
| −Sentiment | sentiment disabled | does news add anything (data-dependent)? |
| Naive | all of the above | floor benchmark |

Hyperparameters are frozen across variants so differences are attributable to the
removed component, not to retuning.

### 2.4 Evaluation metrics (and why each matters)

| Metric | Definition | Why it matters here |
|---|---|---|
| **Sharpe** | mean / std of returns, annualised | standard risk-adjusted return; reported as **Deflated Sharpe** to correct multiple testing |
| **Sortino** | mean / downside-deviation | penalises only harmful volatility; **primary DV** because the architecture targets downside control |
| **Max drawdown** | worst peak-to-trough equity loss | the metric a regime-gated allocator should most improve; ties to the 15% risk limit |
| **Turnover** | volume traded / equity | proxy for cost exposure and overfitting (a strategy that churns is suspect) |
| **Hit rate** | fraction of correct directional calls | direct test of signal skill (currently ~51.7% — the honest headline) |
| **CVaR (95%)** | expected loss in the worst 5% | tail-risk control; portfolio-level constraint in v2 |

Reported per split (distribution, not just mean), net of spread + slippage. The
**Deflated Sharpe Ratio** (Bailey & Lopez de Prado, 2014; in `wfa.py`) corrects a
raw Sharpe for non-normality and for the number of configurations tested.

### 2.5 Statistical validity

**Overfitting risks & defences.**
- Multiple testing across configs → Deflated Sharpe + frozen hyperparameters across
  ablations.
- Expanding-window WFA gives many out-of-sample splits → report the *distribution*
  of metrics, and use paired non-parametric tests across splits, not a single number.
- Turnover monitored as an overfitting smell test.

**Leakage prevention.**
- Per-split fitting only; scalers/models frozen before test inference.
- Embargo gap between train and test.
- Causal feature audit: a shift-and-recompute test asserts no feature at `t` uses
  data from `> t`.
- (v2) synchronized multi-asset splits + no interpolation across market gaps.

**Robustness checks.**
- Cost-sensitivity: re-run with doubled spread/slippage; a strategy that only works
  at zero cost is rejected.
- Regime-stratified performance: confirm results are not driven by one regime.
- Sub-period stability: compare early vs late splits for decay.
- Significance: hit rate tested against 50% with a binomial z-test.

---

## 3. Explainable trading layer design

### 3.1 The Decision object (the single interface)

```python
@dataclass(frozen=True)
class Decision:
    timestamp:        pd.Timestamp
    asset:            str
    regime_state:     str          # market state (regime label)
    signal_strength:  float         # calibrated [0,1] directional confidence
    risk_decision:    float         # position fraction / target weight
    execution_action: str           # open_long | open_short | hold | close | none
    final_action:     str           # BUY | SELL | NO_TRADE
    confidence:       float         # aggregate [0,1]
    reason_codes:     list          # machine-readable drivers
```

`frozen=True` ⇒ immutable. Every downstream consumer (execution, evaluation,
explanation) can only **read** it.

### 3.2 From Decision to human-readable shadow explanation

```python
def explain(decision: Decision) -> ShadowNarrative: ...   # pure function

@dataclass(frozen=True)
class ShadowNarrative:
    state:   str   # what is happening in the market
    action:  str   # what MAESTRO is doing
    reason:  str   # why, in plain language
```

The explainer maps `regime_state` + `reason_codes` + `final_action` to three
sentences via deterministic templates (auditable), optionally rephrased by an LLM for
fluency **without** access to raw model internals and **without** the ability to
change `final_action`.

**Worked example.**

| Field | Value |
|---|---|
| regime_state | `crisis` (vol_high, risk_off), confidence 0.88 |
| signal_strength | 0.41 (below crisis threshold 0.62) |
| risk_decision | 0.0 |
| final_action | `NO_TRADE` |
| reason_codes | `["crisis_regime", "confidence_below_threshold", "vol_suppression"]` |

→ ShadowNarrative:
- **State:** "The market is unstable — high volatility and a flight to safety."
- **Action:** "MAESTRO is staying out of the market (no trade)."
- **Reason:** "Conditions are dangerous and the signal isn't confident enough to
  justify the risk, so it waits."

The narrative restates the *actual* drivers (`reason_codes`) — it does not invent
rationale. This faithfulness is testable (§5).

---

## 4. Demo / user experience design (Layer 2)

All components read the persisted Decision log + equity curve from a completed
backtest. No model code is imported. Implemented as an offline dashboard
(HTML/JS over exported JSON) or a lightweight app — either way, **read-only**.

### 4.1 Market Replay Mode
- Animated price line per asset with a time scrubber over the backtest window.
- Decision markers overlaid at their timestamps: ▲ BUY, ▼ SELL, ⏸ NO_TRADE.
- Equity curve grows beneath the price as the replay advances.
- Play/pause/step; speed control.

### 4.2 Agent "Control Room"
Six role panels, each showing that agent's output at the replay cursor:

| Agent | Role persona | Shows |
|---|---|---|
| Regime | **Weather Reporter** | current regime + confidence ("Stormy, 88%") |
| Signal | **Analyst** | direction + strength ("Mild buy, 0.41") |
| Risk | **Safety Officer** | position size + limits ("Size 0 — too risky") |
| Execution | **Trader** | the order action ("Standing down") |
| Sentiment | **News Interpreter** | macro tone ("Headlines: risk-off") |
| Orchestrator | **Head Decision Maker** | the final call + one-line reason |

### 4.3 Decision Timeline View
A scrollable, timestamped list. For each event:
- market **state** (regime),
- each agent's **opinion** (compact),
- the **final decision**,
- the **outcome** (realised PnL once the horizon resolves).

### 4.4 Emotion Layer (simplified states)
A presentation-only relabelling of regime state into a mood, for intuition:

| Regime signal | Mood | Visual cue |
|---|---|---|
| low vol, trending | **calm** | steady blue |
| low confidence / mixed | **uncertain** | amber |
| high vol / liquidity stress | **dangerous** | red, pulsing |
| strong directional + trend | **trending** | green arrow |

This mood has **zero** effect on logic — it is a colour/label over the existing
`regime_state` field, nothing more.

---

## 5. Bridging mechanism (raw ML → decisions → explanations → UI)

This is the critical seam. Three deterministic stages, each a pure transformation.

### 5.1 Raw ML outputs → structured Decision object
- **Calibration:** model confidences are mapped to [0,1] (TFT: directional
  probability from quantile spread; PatchTST: softmax margin) — *done in core, see
  `signal_agent.predict_batch`*.
- **Confidence thresholds (per regime):** a non-flat signal becomes a candidate trade
  only if `confidence ≥ threshold(regime)` (e.g. 0.52 trend, 0.62 crisis). Below
  threshold → `final_action = NO_TRADE`, `reason_codes += "confidence_below_threshold"`.
- **Disagreement handling:** if signal and sentiment oppose each other, aggregate
  confidence is attenuated (×0.6) and a `reason_codes += "agent_disagreement"` is
  recorded; strong conflict → NO_TRADE.
- **Risk override:** if the Risk Agent returns `action=flat`/`circuit_break`, it
  **wins unconditionally** — `final_action = NO_TRADE`, `reason_codes += "risk_override"`.
  This guarantees risk control is never overruled by a bullish signal.
- The orchestrator writes all drivers into `reason_codes` so the explanation is
  grounded in the real cause, not reconstructed.

### 5.2 Decision object → human explanation
- Template lookup keyed on `(regime_state, final_action, dominant reason_code)`.
- Each `reason_code` has a plain-language fragment; the dominant one (by a fixed
  priority order: `risk_override > crisis_regime > confidence_below_threshold >
  agent_disagreement > normal`) determines the sentence.
- Optional LLM rephrase for fluency, constrained to preserve meaning and forbidden
  from changing the action.

### 5.3 Explanation → visual UI components
- `state` → Control Room "Weather Reporter" + Emotion Layer colour.
- `action` → Replay marker + "Trader" panel.
- `reason` → Decision Timeline row + tooltip.
- `confidence` → marker opacity / gauge fill.

### 5.4 Faithfulness & zero-effect guarantees (testable)
- **Zero-effect test:** run the backtest twice — once with Layer 2 attached, once
  without — and assert the Decision logs are byte-identical. Any difference fails CI.
- **Faithfulness test:** assert every sentence fragment in a ShadowNarrative maps to a
  `reason_code` actually present in that Decision (no invented reasons).

---

## 6. Demo narrative script (for presentation)

*A 90-second walkthrough for a non-expert audience. Plain language; the colours and
panels referenced are Layer 2 visuals.*

> **[Calm market]**
> "We're watching EUR/USD. The screen is blue and steady — MAESTRO's Weather Reporter
> says the market is *calm*. The Analyst sees a mild upward drift but isn't very
> confident. The Safety Officer is relaxed. Because nothing is compelling, the Head
> Decision Maker holds back: small, uncertain signals aren't worth the trading cost.
> **No trade.**"
>
> **[Volatility enters]**
> "Now the chart starts to jump. The screen shifts to *amber* — the Weather Reporter
> flags rising volatility and growing *uncertainty*. The Analyst's view flickers
> between up and down. Notice MAESTRO does **not** rush in. The Safety Officer has
> already started shrinking how much it would ever risk here."
>
> **[Stress / danger]**
> "Volatility spikes and the screen turns *red* — a *dangerous*, risk-off state.
> Headlines turn negative; the News Interpreter agrees. Even though the Analyst now
> shouts 'sell!', the Safety Officer overrides everything: in conditions this unstable,
> the signal simply isn't confident enough to justify the danger. The Head Decision
> Maker announces: **No trade — stand down.** The timeline records the reason in plain
> English: *'Markets are unstable and the signal isn't strong enough — waiting is
> safer.'*"
>
> **[Resolution]**
> "Minutes later the storm passes. The screen fades back toward amber, then blue. Had
> MAESTRO traded into the chaos, it would have been whipsawed. By *not* acting, it
> protected the capital. That restraint — knowing when **not** to trade — is the
> behaviour we measure."

*Key teaching point for the audience: the interesting decision was a NO-TRADE, and the
system explained itself in one sentence anyone can follow.*

---

## 7. Key innovation summary

**What is novel in MAESTRO.**
- A **faithful, structurally-isolated interpretability layer**: every decision carries
  a grounded plain-language explanation, and the explanation layer is provably unable
  to change any trade (frozen Decision object + pure explainer + zero-effect CI test).
- A **multi-agent decomposition** where each function (regime, signal, risk,
  execution, sentiment, orchestration) is a separately testable, separately ablatable
  module — enabling clean attribution of where value (or none) comes from.
- An **honest evaluation framework** that treats a negative result as a valid outcome
  and is built to detect it (Deflated Sharpe, paired across-split tests, cost
  sensitivity, leakage audit).

**Why multi-agent decomposition matters.**
A monolithic model is a black box you can only accept or reject. Decomposition lets
you ask *which part works*: ablate the regime layer and measure the change; ablate
risk sizing and watch drawdown move. It turns "is the system good?" into a set of
falsifiable sub-questions — the essence of a defensible thesis.

**Why regime-conditioning improves robustness.**
Markets are non-stationary; a model tuned on calm data fails in crises. Conditioning
the signal on regime, and scaling risk down in adverse regimes, lets the same system
behave differently as the world changes — the mechanism by which it can improve
*downside* metrics (Sortino, drawdown, CVaR) even when raw directional skill is weak.

**Why the interpretability layer is a research contribution.**
Most "explainable trading" post-rationalises with feature attributions that may not
reflect the true decision path. MAESTRO's explanation is generated from the actual
`reason_codes` the orchestrator used, and a CI test guarantees the explanation never
alters the decision. That combination — *grounded* and *provably inert* — is the
contribution: interpretability that is both faithful and safe.

---

## Appendix — layer separation guarantees (summary)

| Concern | Guarantee | Enforced by |
|---|---|---|
| Demo can't change trades | Decision logs identical with/without Layer 2 | frozen `Decision` + zero-effect CI test |
| Explanation can't invent reasons | every fragment maps to a real `reason_code` | faithfulness test |
| Evaluation can't feed back | metrics computed from logs only | one-way data flow |
| No leakage | per-split fitting, embargo, causal audit | WFA engine + unit tests |
| Risk never overruled | `risk_override` wins unconditionally | orchestrator precedence rule |
