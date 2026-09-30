# MAESTRO — Thesis Narrative

> **My MSc asked the right questions but couldn't answer them rigorously. The PhD
> rebuilds the same strategies under a leakage-free, cost-aware walk-forward
> evaluation, shows what they actually achieve, then tests whether a multi-agent
> system does better.**

Every chapter, experiment and piece of code should serve this sentence. If a task
doesn't, it is future work.

## Research question

Does a multi-agent, regime-aware trading system beat simple machine-learning
baselines on EUR/USD — after transaction costs, under a leakage-free walk-forward
evaluation, both in backtest and in live forward testing?

Success is not defined as "MAESTRO is profitable". It is defined as answering the
question with evidence an examiner cannot pick apart. A clear "no" is a valid result.

## Chapter plan and evidence status

| # | Chapter | Evidence | Status |
|---|---|---|---|
| 1 | Introduction | Research question above | To write |
| 2 | Literature review | ML in trading; backtest overfitting (López de Prado; Bailey & López de Prado); multi-agent systems; regime detection; explainable AI | To write |
| 3 | **Revisiting the MSc** | What the MSc claimed; why its figures don't reproduce (regression re-run on the same window gives 46.6% vs the reported 37.56%; the metric counts flat bars as misses; Sharpe 0.599 came from a separate daily SMA test; scored in-sample; no costs; live results were a handful of trades) | Done — see `memory/msc_baseline.md`, replication scripts |
| 4 | **Evaluation framework** | 243 walk-forward test months (2006–2026), retrained quarterly on the latest 12 months, 5-day embargo; one shared evaluator (`backtesting/baselines.py`); explicit timing convention; look-ahead tests (`tests/test_baselines.py`); macro inputs stamped at publication time, with tests (`data/features/macro.py`, `tests/test_data_timing.py`); holdout sealed from 7 Mar 2026 (`data/holdout.py`); power test (`backtesting/power_test.py`); cost model; metric definitions | Done |
| 5 | **Baselines under the framework** | 20 years: learning baselines hit 51.9–52.6% and earn +0.12–0.70 pips/trade before costs (Bollinger best, +0.70); all lose after 0.8 pips; buy-and-hold ~0 over 20 years (its 2023–26 lead was that period); real OANDA spread median 0.9–1.6 pips by year | Done — `OUTPUT_DIR/maestro/refit3_roll12_fast/scores` |
| 6 | **MAESTRO** | Architecture (regime, signal, risk, execution, sentiment, orchestrator); confidence-calibration fix; TFT fixes (target scaling, horizon indexing, one-bar lag); explainability layer | Built |
| 7 | **MAESTRO vs baselines** | Being re-run: the regime detector's HMM smoothed over whole test blocks (a look-ahead leak, fixed in 5a9f47b), so every earlier MAESTRO number (20-year result, power test, risk layer) is invalid. Findings that don't depend on MAESTRO's numbers stand: its designed confidence gates and Kelly sizing assume confidence it never reached; the risk agent's cost filter rejected every short (fixed). Then: monthly vs quarterly retraining, horizon sweep, confidence buckets | Re-running |
| 8 | **Live forward test** | MSc strategies and MAESTRO side by side on the OANDA practice account for 4–8 weeks: every strategy paper-traded at live quotes, one MAESTRO variant also sending practice orders; live decisions tested equal to the backtest's; daily reconciliation | Paper shakedown running |
| 9 | Discussion and conclusion | Statistical vs economic significance; what the extra complexity bought (or didn't); limitations; future work | To write |

## Parked as future work

These are built or designed, but they don't serve the core sentence yet:

- Cross-asset v2 (gold, SPY) — `docs/MAESTRO_V2_ARCHITECTURE.md`
- LLM orchestrator — possible extra ablation in chapter 7, not a dependency
- Demo / presentation layer — `docs/MAESTRO_DUAL_LAYER_DESIGN.md`, for the viva, not the evidence.
  Stage 1 exists: the plain-language findings page built from real results by
  `demo/findings/build_page.py`. It grows as chapters 7 and 8 produce results.
- Live trading with real money — out of scope
