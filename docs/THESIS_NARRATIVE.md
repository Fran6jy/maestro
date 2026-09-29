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
| 4 | **Evaluation framework** | 39 walk-forward test months, retrained quarterly on the latest 12 months, with embargo; one shared evaluator (`backtesting/baselines.py`); explicit timing convention; tests proving no look-ahead (`tests/test_baselines.py`); cost model; metric definitions | Done |
| 5 | **Baselines under the framework** | MSc strategies rebuilt and scored: all 48–51% hit rate; ML baselines have positive gross Sharpe but lose after costs; only buy-and-hold is positive | Done — `C:\tmp\maestro_outputs\baselines\` |
| 6 | **MAESTRO** | Architecture (regime, signal, risk, execution, sentiment, orchestrator); confidence-calibration fix; TFT fixes (target scaling, horizon indexing, one-bar lag); explainability layer | Built |
| 7 | **MAESTRO vs baselines** | Same evaluator, three versions fixed before results: as designed never trades (confidence never clears its thresholds); most confident 10% hits 50.0%, Sharpe −0.46 gross / −3.13 net; every signal 49.7%, −0.82 / −3.67. No edge even before costs, unlike the simple models; loses less than logistic regression only by trading less. Still to add: confidence-bucket analysis, ablations | Core result done |
| 8 | **Live forward test** | MSc strategies and MAESTRO side by side on the OANDA practice account for 4–8 weeks; daily reconciliation of live decisions and fills against the backtest | **Next** |
| 9 | Discussion and conclusion | Statistical vs economic significance; what the extra complexity bought (or didn't); limitations; future work | To write |

## Parked as future work

These are built or designed, but they don't serve the core sentence yet:

- Cross-asset v2 (gold, SPY) — `docs/MAESTRO_V2_ARCHITECTURE.md`
- LLM orchestrator — possible extra ablation in chapter 7, not a dependency
- Demo / presentation layer — `docs/MAESTRO_DUAL_LAYER_DESIGN.md`, for the viva, not the evidence.
  Stage 1 exists: the plain-language findings page built from real results by
  `demo/findings/build_page.py`. It grows as chapters 7 and 8 produce results.
- Live trading with real money — out of scope
