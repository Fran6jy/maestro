# MAESTRO v2 — Cross-Asset Regime-Driven Capital Allocation Engine

**Status:** Design specification (not yet implemented)
**Scope:** Extends single-asset MAESTRO (EUR/USD, GBP/USD) to a 4-asset portfolio
allocator (EUR/USD, GBP/USD, XAU/USD, SPY) with a shared global regime layer,
portfolio-level CVaR risk, an explainability layer, and a thesis-grade evaluation
framework.

---

## 0. Research framing (read first)

### 0.1 The honest starting point

Empirical testing of the existing single-asset signal (split_000, EUR/USD, M5)
gives a directional hit rate of **51.7% (z = +0.84 vs 50%)** — not statistically
distinguishable from random. The confidence-calibration pipeline has been repaired
(see `tft_model.predict` and `signal_agent.predict_batch`), so this is an honest
measurement of model skill, not an artefact.

**Implication for v2:** more assets and more architecture do not manufacture
directional edge. The design below is therefore optimised to test a claim that
remains valid under weak signals.

### 0.2 The research claim (restated to be testable under weak signals)

> *Multi-agent decomposition + shared regime conditioning improves the
> **robustness and risk-adjusted** behaviour of a cross-asset allocator in
> non-stationary markets, relative to (a) a naive equal-weight portfolio and
> (b) ablated variants of the same system.*

This is falsifiable through ablation (§7) and does **not** require beating a
buy-and-hold benchmark on raw return. The measurable dependent variables are
Sortino, max drawdown, portfolio CVaR, turnover, and regime-conditioned stability
— all of which a regime-gated risk allocator can move even when per-asset
directional hit rate is ≈ 50%.

### 0.3 Three explicit layers (kept strictly separate throughout)

| Layer | Definition | May influence trades? |
|---|---|---|
| **Model logic** | Anything that produces or sizes a position | Yes — this *is* the strategy |
| **Explanation layer** | Human-readable narration of a decision already made | **No** — read-only, post-hoc |
| **Evaluation framework** | Metrics, ablations, significance tests | No — observes, never feeds back |

The explanation layer (§5) consumes the immutable `Decision` object and emits text.
It has no write path into model logic. This separation is enforced by making the
explainer a pure function `explain(decision, context) -> str`.

---

## 1. Full architecture design

### 1.1 Layer stack (top = global, bottom = per-asset)

```
                    ┌───────────────────────────────────────────┐
   GLOBAL           │  GLOBAL REGIME LAYER  (shared "weather")   │
   (asset-agnostic) │  risk_on/off · vol_hi/lo · trend/MR ·      │
                    │  liquidity_stress                          │
                    └───────────────────┬───────────────────────┘
                                        │ GlobalRegimeState (1 per timestamp)
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                ▼               ▼                ▼               
  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐          
  │ FX MODULE│    │ FX MODULE│    │GOLD MODULE│   │EQTY MODULE│   PER-ASSET
  │ EUR/USD  │    │ GBP/USD  │    │ XAU/USD  │    │   SPY    │          
  │ feat+sig │    │ feat+sig │    │ feat+sig │    │ feat+sig │          
  └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘          
       │ AssetSignal   │               │               │                
       └───────────────┴───────┬───────┴───────────────┘                
                               ▼                                        
                  ┌─────────────────────────┐                          
   PORTFOLIO      │  PORTFOLIO RISK ENGINE   │  cross-asset CVaR,       
                  │  (cross-asset allocator) │  exposure caps,          
                  └────────────┬────────────┘  global de-risk          
                               ▼                                        
                  ┌─────────────────────────┐                          
   DECISION       │   META-ORCHESTRATOR      │  conflict resolution,    
                  │   (portfolio manager)    │  regime-weighted fusion  
                  └────────────┬────────────┘                          
                               ▼                                        
                  ┌─────────────────────────┐                          
                  │  Decision[] (per asset)  │  ← THE ONLY interface    
                  └──────┬───────────┬───────┘    to execution/eval     
                         ▼           ▼                                  
              ┌────────────────┐  ┌────────────────────┐               
              │ EXECUTION /    │  │ EXPLANATION LAYER  │  (read-only)   
              │ BACKTEST SINK  │  │ (human shadow)     │               
              └────────────────┘  └────────────────────┘               
```

### 1.2 Reuse vs new build (mapped to existing repo)

| Component | Status | Path |
|---|---|---|
| Regime detector (HMM/Transformer) | **Refactor** → global, asset-agnostic | `agents/regime/` |
| TFT / PatchTST signal models | **Reuse**, instantiate per asset | `agents/signal/` |
| Risk agent (CVaR/Kelly) | **Major upgrade** → portfolio allocator | `agents/risk/` |
| Execution agent | Reuse, add per-asset cost models | `agents/execution/` |
| Meta-orchestrator | Upgrade → conflict resolution + allocation | `orchestrator/` |
| WFA engine | Upgrade → synchronized multi-asset splits | `data/validation/wfa.py` |
| Explanation layer | **New** | `xai/shadow_model.py` (new) |
| Demo layer | **New, isolated** | `demo/` (new package) |
| `Decision` object | **New, central** | `orchestrator/decision.py` (new) |

---

## 2. Data flow (text description)

### 2.1 Per-timestamp flow (inference / backtest, one synchronized bar)

```
1.  ALIGNED BAR t  (see §2.3 calendar alignment)
       │  contains: per-asset OHLCV+features, shared macro (FRED)
       ▼
2.  GLOBAL REGIME LAYER.predict(t)
       │  input : cross-asset return panel, VIX, yield curve, breadth,
       │          realised-vol panel, FX-funding proxy  (ALL ≤ t)
       │  output: GlobalRegimeState{risk, vol, style, liquidity, conf}
       ▼
3.  FOR each asset a in {EURUSD, GBPUSD, XAUUSD, SPY}:
       AssetModule[a].signal(t, GlobalRegimeState)
         │  input : asset-local features (≤ t) + regime as conditioning
         │  output: AssetSignal{direction, strength, horizon, model_conf}
       ▼
4.  PORTFOLIO RISK ENGINE.allocate({AssetSignal}, GlobalRegimeState, equity_t)
         │  output: target weights w_a, total gross exposure, per-asset caps,
         │          portfolio CVaR estimate, de-risk flag
       ▼
5.  META-ORCHESTRATOR.resolve({AssetSignal}, weights, GlobalRegimeState)
         │  resolves direction conflicts, applies regime confidence scaling,
         │  applies LIVE-READINESS GUARDS (§6) → may force global NO_TRADE
         │  output: Decision[a] for each asset (immutable)
       ▼
6a. EXECUTION/BACKTEST SINK  ← Decision[]   (applies cost model, updates equity)
6b. EXPLANATION LAYER        ← Decision[]   (read-only narration)
6c. EVALUATION COLLECTOR     ← Decision[] + fills (metrics, no feedback)
```

Steps 6a/6b/6c consume the **same immutable `Decision[]`**. 6b and 6c never write
back. This is the structural guarantee that interpretability and evaluation cannot
alter trading logic.

### 2.2 Training / fitting flow (per WFA split, leakage-controlled)

```
For split s with train window [T0, T1] and test window [T1+embargo, T2]:
  1. Fit GLOBAL REGIME LAYER on cross-asset panel restricted to [T0,T1] only.
  2. For each asset: fit feature scalers + signal model on [T0,T1] only,
     using regime labels produced by the split-s regime model (not future).
  3. Fit / calibrate PORTFOLIO RISK ENGINE on [T0,T1] (covariance, CVaR params).
  4. Freeze ALL fitted objects. Run inference on [T1+embargo, T2].
  5. Persist per-split artefacts + Decision log for evaluation.
```

Nothing fitted on test data; embargo gap prevents label bleed from overlapping
return horizons (Lopez de Prado purging, already in `wfa.py`).

### 2.3 Calendar alignment (critical, often-missed rigor point)

The four assets do **not** share a trading calendar:

| Asset | Hours (UTC) | Gaps |
|---|---|---|
| EUR/USD, GBP/USD | ~Sun 22:00 → Fri 22:00, continuous | weekend |
| XAU/USD | ~Sun 23:00 → Fri 22:00, daily ~1h break | weekend |
| SPY | 14:30 → 21:00, Mon–Fri only | overnight + weekend |

**Rule:** a synchronized timestamp exists only where **all required inputs exist
causally**. Design decisions:

- The **master clock is the union of FX M5 timestamps** (most continuous).
- For an asset that is **closed** at time `t` (e.g. SPY overnight), its
  `AssetSignal` is set to `direction=0, tradeable=False`. The allocator may still
  *hold* an existing SPY position but cannot open/close until the asset reopens.
- SPY features that require its own bars (momentum, gaps) are **forward-filled
  from the last close** and explicitly flagged `stale=True`; the signal model
  receives the staleness flag as a feature so it can learn to discount stale state.
- **No interpolation across a gap** (would inject look-ahead). Gaps are represented,
  not filled with synthetic prices.
- The global regime layer only consumes inputs available at `t`; when SPY is closed
  its contribution to breadth/risk-appetite features uses the last *causal* value
  with a staleness flag, never a future open.

This alignment policy is itself a thesis methods-section contribution.

---

## 3. Module breakdown with interfaces

All interfaces are typed dataclasses. Signatures below are the contract; bodies
reuse existing model code where noted.

### 3.1 Global Regime Layer

```python
@dataclass(frozen=True)
class GlobalRegimeState:
    timestamp:        pd.Timestamp
    risk_state:       str      # "risk_on" | "risk_off" | "neutral"
    vol_state:        str      # "vol_high" | "vol_low"
    style_state:      str      # "trend" | "mean_reversion"
    liquidity_state:  str      # "normal" | "stress"
    confidence:       float    # [0,1] max class prob of the regime ensemble
    probabilities:    dict     # full class distribution (for XAI + audit)
    is_certain:       bool     # confidence >= threshold

class GlobalRegimeLayer:
    def fit(self, panel: pd.DataFrame) -> "GlobalRegimeLayer": ...
    def predict_batch(self, panel: pd.DataFrame) -> pd.DataFrame: ...   # one row/ts
    def predict_bar(self, row: pd.Series) -> GlobalRegimeState: ...
```

**Inputs (`panel`)** — strictly cross-asset, asset-agnostic, all causal:
cross-asset realised-vol average, return dispersion/breadth, VIX & VIX term proxy,
2s10s yield curve, DXY/USD-strength proxy, FX-funding (cross-currency basis proxy),
gold/SPY ratio (risk appetite). **Deliberately excludes** any single asset's raw
price so the layer cannot become an EUR/USD detector in disguise.

**Implementation:** reuse the HMM+Transformer ensemble from `agents/regime/`, but
trained on the cross-asset panel and emitting the 4 orthogonal state axes (each a
small classifier head) rather than one 4-class label. Output is **identical for all
assets at a given timestamp** — enforced by construction (single call per `t`).

### 3.2 Asset Module (one instance per asset)

```python
@dataclass(frozen=True)
class AssetSignal:
    timestamp:   pd.Timestamp
    asset:       str
    direction:   int       # -1 | 0 | +1
    strength:    float      # [0,1] calibrated directional confidence
    horizon:     int        # bars
    model_conf:  float      # model self-confidence (for XAI)
    tradeable:   bool       # False when market closed / stale
    metadata:    dict       # per-model breakdown (XAI)

class AssetModule:
    asset_class: str        # "fx" | "gold" | "equity"
    def fit(self, df, regime_states, val_df=None) -> "AssetModule": ...
    def signal_batch(self, df, regime_states) -> pd.DataFrame: ...   # AssetSignal rows
```

Three concrete subclasses share the interface, differ in feature pipeline:

- **FXModule** (EUR/USD, GBP/USD): macro-sensitive features, rate differential,
  USD-strength bias, cross-pair correlation. Signal models = existing TFT+PatchTST.
- **GoldModule** (XAU/USD): real-yield / inflation-surprise features, inverse-USD
  handling, risk-off beta, heavier volatility-regime conditioning.
- **EquityModule** (SPY): momentum/trend-persistence features, overnight-gap
  features, risk-appetite beta, staleness flags for closed sessions.

Each has an **independent feature pipeline and scaler** (fit per split, per asset —
no shared fitting across assets → no cross-asset leakage). The **only shared input**
is the `GlobalRegimeState`, passed as conditioning.

### 3.3 Portfolio Risk Engine

```python
@dataclass(frozen=True)
class PortfolioAllocation:
    timestamp:        pd.Timestamp
    weights:          dict     # asset -> target weight (signed, sums |w| ≤ gross_cap)
    gross_exposure:   float    # total |w|
    per_asset_cap:    dict     # asset -> max |w|
    portfolio_cvar:   float    # estimated 95% CVaR of the target portfolio
    derisk_flag:      bool     # True → scale everything toward cash
    reason_codes:     list     # machine reasons (feeds XAI)

class PortfolioRiskEngine:
    def fit(self, train_panel, regime_states) -> "PortfolioRiskEngine": ...
    def allocate(self, signals: dict[str, AssetSignal],
                 regime: GlobalRegimeState, equity: float) -> PortfolioAllocation: ...
```

Responsibilities:
- **Total exposure**: scaled down in `vol_high` / `stress` regimes.
- **Per-asset weights**: proportional to `direction * strength`, then risk-parity
  adjusted by the **rolling cross-asset covariance** (fit on train window only).
- **Correlation-aware scaling**: two highly correlated longs (e.g. gold + risk-off
  FX) are jointly capped so portfolio CVaR, not per-trade risk, is the binding
  constraint.
- **Portfolio CVaR constraint**: solve for weights s.t. estimated 95% CVaR ≤ budget;
  if infeasible, shrink gross exposure.
- **Global de-risk**: `liquidity_state == "stress"` forces `derisk_flag=True` →
  weights scaled toward zero regardless of signals.

Upgrade path from existing `agents/risk/`: keep Kelly/CVaR primitives, wrap them in
a portfolio optimiser (covariance + CVaR constraint) instead of per-trade sizing.

### 3.4 Meta-Orchestrator (enhanced)

```python
class MetaOrchestrator:
    def resolve(self, signals: dict[str, AssetSignal],
                allocation: PortfolioAllocation,
                regime: GlobalRegimeState,
                guards: GuardState) -> list[Decision]: ...
```

- **Conflict resolution**: when two correlated assets give opposing directions, the
  one with lower calibrated `strength` is attenuated or zeroed (configurable rule,
  logged for XAI).
- **Regime confidence scaling**: final `confidence` multiplied by
  `regime.confidence`; uncertain regimes shrink all convictions.
- **Guard application**: if any live-readiness guard (§6) is tripped, emit
  `final_action = NO_TRADE` for **all** assets ("no-trade global state").
- **Output**: one `Decision` per asset (or a single global NO_TRADE set).

### 3.5 Unified Decision object (the only ML→execution interface)

```python
@dataclass(frozen=True)
class Decision:
    timestamp:        pd.Timestamp
    asset:            str
    regime_state:     GlobalRegimeState
    signal_strength:  float          # calibrated [0,1]
    risk_allocation:  float          # target weight from PortfolioAllocation
    execution_action: str            # "open_long" | "open_short" | "hold" | "close" | "none"
    final_action:     str            # "BUY" | "SELL" | "NO_TRADE"
    confidence:       float          # aggregate [0,1]
    reason_codes:     list           # machine-readable drivers (XAI input)
```

`frozen=True` makes it immutable: downstream layers cannot mutate a decision, only
read it. Execution, explanation, and evaluation all bind to this single type.

---

## 4. Decision pipeline explanation

1. **Regime first.** The global layer sets the "weather". Every downstream module
   receives the same `GlobalRegimeState` — no asset can see a different regime.
2. **Signals are conditioned, not gated, by regime.** Asset modules receive regime
   as an input feature; they still emit a raw directional view. (Gating happens
   later, at allocation — keeps the signal model honest and the ablation clean.)
3. **Risk allocates at the portfolio level**, not per trade. This is where regime
   actually reduces exposure and where correlation is handled.
4. **Orchestrator resolves and guards.** Conflicts resolved, guards applied,
   confidence scaled. Produces immutable `Decision[]`.
5. **Sinks consume.** Execution applies costs; explanation narrates; evaluation
   measures. None of the three can change the decision.

Determinism: given the same fitted artefacts and the same input panel up to `t`,
the pipeline is deterministic (no RNG in the inference path), which is required for
reproducible backtests and for the ablation studies to be comparable.

---

## 5. Explanation / "human shadow" layer design

A **pure, post-hoc** narrator. Signature:

```python
def explain(decision: Decision) -> ShadowNarrative: ...

@dataclass(frozen=True)
class ShadowNarrative:
    state:   str   # what is happening in the market
    action:  str   # what MAESTRO is doing
    reason:  str   # why, in plain language
```

- **Template-based first** (deterministic, auditable): maps `regime_state` +
  `reason_codes` to sentence templates. Example:
  - State: "High volatility, risk-off regime"
  - Action: "Reduce equity exposure, hold gold"
  - Reason: "Markets are unstable; capital moved toward safer assets"
- **Optional LLM rephrasing** (claude-opus-4-8) for fluency — but only rephrases the
  template output; it never sees raw signals and cannot change `final_action`. This
  keeps the narrative faithful and prevents the LLM from inventing rationale.
- Generated for **every timestamp in backtest mode** and stored alongside the
  Decision log → becomes the XAI appendix of the thesis.

**Guarantee:** `explain` takes a `Decision` and returns text. It has no reference to
any model object and no return path into the pipeline. Interpretability cannot alter
trading logic — enforced by the type system, not by convention.

---

## 6. Risk controls specification (live-readiness guards)

Evaluated every bar by a `GuardEngine` *before* `Decision[]` is emitted. Any trip →
global `NO_TRADE`.

| Guard | Trigger | Action |
|---|---|---|
| Max leverage | gross exposure > `max_gross` (e.g. 3×) | scale weights down to cap |
| Max daily loss | day PnL < −`daily_loss_pct` (e.g. 3%) | NO_TRADE rest of day, flatten |
| Volatility suppression | realised vol > `vol_cap` percentile | suppress new entries |
| Liquidity stress | regime `liquidity_state == stress` | global de-risk → NO_TRADE new |
| Trade frequency cap | trades/asset/day > `N` | block further entries that asset |
| Execution cost gate | expected edge < spread+slippage | NO_TRADE that asset |
| Max portfolio drawdown | equity drawdown > `max_dd` (e.g. 15%) | flatten all, halt new |

```python
@dataclass(frozen=True)
class GuardState:
    tripped:    bool
    global_no_trade: bool
    reasons:    list           # which guards fired (XAI + audit)
```

Cost model: per-asset spread + slippage (FX tight, gold wider, SPY commission+spread).
Already partially present in `agents/risk/cost_model.py`; extend per asset class.

These guards are **model logic** (they change trades), distinct from the explanation
layer. They are evaluated and logged so the evaluation framework can report how often
each fired and with what PnL consequence.

---

## 7. Experimental validation plan (thesis-grade)

### 7.1 Metrics (per asset, and portfolio-level)

Sharpe, Sortino, max drawdown, hit rate, turnover, 95% CVaR, Calmar, profit factor.
All reported **net of costs**. Sharpe additionally reported as **Deflated Sharpe
Ratio** (already in `wfa.py`) to correct for multiple testing across configs.

### 7.2 Ablation studies (the core experiment — tests the research claim)

Run the identical synchronized WFA on each variant; compare distributions across
splits with a paired test (Wilcoxon signed-rank across splits):

| Variant | What is removed | Hypothesis it tests |
|---|---|---|
| **Full** | nothing | baseline system |
| **−Regime** | global regime layer fixed to "neutral" | does regime conditioning help robustness? |
| **−Orchestrator** | naive sum of signals, no conflict resolution | does fusion add value? |
| **−Risk** | equal-weight allocation, no CVaR/correlation | does portfolio risk reduce drawdown/CVaR? |
| **−All (naive)** | equal-weight, always-on, no guards | floor benchmark |
| **Buy&Hold** | passive 1/N portfolio | market benchmark |

Primary dependent variables: **Sortino, max drawdown, portfolio CVaR** (the metrics
the architecture can move under weak signals). Raw return is reported but is *not*
the success criterion.

### 7.3 Cross-asset evaluation

- Per-asset-class performance table.
- **Return correlation matrix** across the four assets (realised), to show the
  allocator is diversifying rather than concentrating.
- Regime-conditioned breakdown: performance within each `risk/vol/style/liquidity`
  state — the central evidence that regime conditioning changes behaviour.

### 7.4 Leakage & integrity audit (must pass before any result is reported)

- Synchronized splits: every asset uses the **same** train/test boundaries + embargo.
- Regime layer fit on train only; verified by a unit test that fails if any
  test-window timestamp influences a fitted parameter.
- Feature causality test: shift-and-recompute check that no feature at `t` uses data
  from `> t`.
- Calendar test: assert no synthetic prices were interpolated across market gaps.

### 7.5 Honest reporting stance

If ablations show the full system improves Sortino / drawdown / CVaR **but not raw
return**, that is the result — and it supports the restated claim (§0.2). If even
risk-adjusted metrics are indistinguishable from the naive benchmark across splits,
the thesis reports a **negative result**: multi-agent decomposition did not improve
robustness on this asset set at this frequency. Both outcomes are publishable; the
design does not presuppose success.

---

## 8. Demo / presentation layer design (isolated, no trading-logic changes)

Lives in a separate `demo/` package. Consumes the persisted Decision log + equity
curve produced by a completed backtest. **Read-only.** No model code imported into
trading path.

- **9.1 Market Replay**: animated per-asset price with Decision markers (BUY/SELL/
  NO_TRADE) overlaid at their timestamps; scrubber over the backtest window.
- **9.2 Control Room**: five agent panels (Regime = weather, Signal = analyst,
  Risk = safety officer, Execution = trader, Orchestrator = portfolio manager), each
  showing that agent's current output for the replay cursor.
- **9.3 Decision Timeline**: scrollable list of `{state, decision, reason, outcome}`
  rows from the ShadowNarrative + realised PnL.
- **9.4 Emotional Regime Mapping**: maps `GlobalRegimeState` to a mood
  (calm / uncertain / dangerous / trending) for intuitive display only — a
  presentation relabelling of regime state, with **zero** effect on logic.

Implementation: offline (HTML/JS reading exported JSON), or a small dashboard.
Because it reads only the immutable Decision log, it provably cannot alter results.

---

## 9. Build order (suggested, incremental + testable)

1. `Decision` + `GlobalRegimeState` + `AssetSignal` dataclasses (`orchestrator/decision.py`).
2. Calendar-alignment module + tests (the hardest correctness piece — do it first).
3. Global regime layer refactor + leakage unit test.
4. Gold + Equity modules (FX already exists); per-asset feature pipelines.
5. Portfolio risk engine (covariance + CVaR optimiser).
6. Orchestrator upgrade + guard engine.
7. Multi-asset synchronized WFA.
8. Explanation layer (template first, LLM rephrase optional).
9. Evaluation framework + ablation harness.
10. Demo layer (last, purely on exported logs).

Each step ships with a unit test; nothing proceeds to evaluation until the leakage
audit (§7.4) passes.

---

## 10. Open risks & honest caveats

- **Signal edge is unproven.** Single-asset hit rate ≈ 51.7%. v2 may improve
  risk-adjusted metrics via allocation/regime, but will not create directional edge
  that the models do not have.
- **SPY/FX frequency mismatch.** M5 is natural for FX, awkward for SPY (6.5h/day).
  Consider evaluating SPY at a coarser bar, or restrict SPY trading to RTH only.
- **Data acquisition.** XAU/USD and SPY M5 history must be sourced and quality-checked
  (gaps, splits/dividends for SPY) before any of this runs.
- **Overfitting surface grows** with 4 assets × regime conditioning. The ablation +
  deflated-Sharpe + paired-test design is the defence; keep hyperparameters fixed
  across ablation variants.
```
