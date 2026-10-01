# MAESTRO

**PhD research: can a team of AI agents trade EUR/USD profitably once every trade pays its costs?**

Live site: **[maestro-research.vercel.app](https://maestro-research.vercel.app)** — an interactive
walkthrough of the results, including a Strategy Lab where you can re-price every strategy at any
trading cost.

> My MSc asked the right questions but couldn't answer them rigorously. This PhD rebuilds the same
> strategies under a leakage-free, cost-aware walk-forward evaluation, shows what they actually
> achieve, then tests whether a multi-agent system does better.

---

## Findings so far

**The MSc figures don't reproduce.** The MSc reported a 37.56% hit ratio and a Sharpe ratio of 0.599.
They came from two unrelated experiments: a linear regression scored on the same month it was trained
on, and a separate daily moving-average strategy. Re-running the regression on the same month gives
46.6%; the original figure mostly reflects how flat price bars were scored.

**Rebuilt under one evaluator over 20 years, no baseline survives trading costs.**
EUR/USD 5-minute bars, 243 walk-forward test months (Jan 2006 to Feb 2026), every model retrained
each quarter on the latest 12 months, 0.8 pips per round trip, £10,000 start:

| Strategy | Right about next move | Trades | Pips per trade before costs | Sharpe before costs | Sharpe after costs | £10,000 becomes |
|---|---:|---:|---:|---:|---:|---:|
| Buy and hold (benchmark) | 50.2% | 243 | −0.40 | −0.01 | −0.02 | £9,637 |
| Bollinger bands 20 / 2σ | 52.6% | 51,966 | +0.70 | 2.07 | −0.30 | £6,474 |
| Moving-average crossover 20/200 | 49.4% | 13,302 | +0.07 | 0.01 | −0.43 | £4,254 |
| Contrarian, 3 bars | 52.0% | 415,067 | +0.20 | 3.49 | −10.40 | < £0.01 |
| Logistic regression, 5 moves | 52.3% | 450,504 | +0.21 | 3.84 | −10.96 | < £0.01 |
| Linear regression, 5 moves | 51.9% | 662,098 | +0.14 | 3.69 | −15.48 | < £0.01 |
| Linear regression, 1 move | 51.9% | 757,250 | +0.12 | 3.51 | −18.48 | < £0.01 |
| Coin flip (sanity check) | 50.0% | 735,364 | +0.00 | 0.11 | −19.25 | < £0.01 |

Short-horizon EUR/USD has a real, small edge (52–53% of next moves called right, a positive Sharpe
before costs), but it is worth only 0.1–0.7 pips per trade. OANDA's real EUR/USD spread, measured on
every bar, has a median of 0.9–1.6 pips depending on the year (1.5–1.6 since 2022). Buy and hold earns
nothing over the full 20 years; its good showing in 2023–2026 was that period.

**MAESTRO is the most accurate forecaster tested, and it still doesn't beat costs.** Same months,
same costs, same scoring code:

| MAESTRO version | Right about next move | Trades | Pips per trade before costs | Sharpe after costs | £10,000 becomes |
|---|---:|---:|---:|---:|---:|
| Every signal | 52.7% | 63,749 | +0.28 | −2.51 | £661 |
| Most confident 10% | 53.1% | 39,112 | +0.23 | −2.50 | £1,569 |
| With a cost check (risk agent) | 53.9% | 12,784 | +0.30 | −0.89 | £6,061 |
| As designed (confidence thresholds) | 1 trade in 20 years | | | | £10,004 |

Its best version calls 54% of moves right, more than any baseline, but earns about a third of a
trade's cost and ends a little behind Bollinger bands (£6,474). A power test planted a known edge
in real prices: at a 55% planted edge MAESTRO found it (58% of its most confident calls right) and
made money after costs; at 52% it found nothing.

**Corrected after two look-ahead bugs.** Found and fixed on 30 September 2026: macro data reached
the models before it was published, and MAESTRO's regime detector labelled each bar using the whole
three-month test block it sat in (the HMM's Viterbi path and forward-backward smoothing), so a bar's
regime depended on later prices. Everything above was re-run on the fixed code; the fixes barely
changed MAESTRO's results, and the baselines use prices only. The cost-check version was first tried
after results from before the fix had been seen, so the sealed holdout is its real test.

**How the evaluation is checked.**
- Macro inputs (VIX, yields, rates, CPI) reach a bar only once they were public.
- Regimes are forward-filtered: a bar's regime uses only bars up to it. Tests prove that rewriting
  later prices cannot change an earlier regime, and that the old method could.
- Data from 7 March 2026 is sealed and has influenced no decision. It is opened once, for a final
  confirmation run of a frozen design.
- A power test plants a known edge in real prices to check the pipeline can find one.
- Every retraining block's dates were checked across the machines that trained it, and the ways
  of turning MAESTRO's forecasts into trades are fixed before any result is seen.

## Live trial

A 4–8 week forward test on an OANDA **practice** account (demo money only) runs MAESTRO and the
baselines side by side on every 5-minute bar, to measure how far live trading departs from the
backtest. It runs in one Docker container on a small always-on server (`deploy/live/`).

- Every strategy is paper-traded at OANDA's live bid/ask; one MAESTRO variant also sends real orders
  to the practice account, so real spreads, fills and delays are measured too.
- Live decisions are the backtest's decisions: a model trained, saved and reloaded, fed only data up
  to each bar, gives the backtest's forecast on 40 of 40 test bars, and every strategy's live rule
  is tested bar for bar against the backtest's.
- `live/oanda.py` can only reach OANDA's practice server and never resends an order.
- Before the trial starts, the 20-year backtest's expected range for every strategy over 1–65 days
  is committed (`web/src/data/live_expectations.json`); the live results are judged against it.
- Every night a snapshot (returns, pips, fills, health; no prices) is published to a public
  record, checked automatically, and shown on the site's live page.

Status: paper-only shakedown since 30 September 2026. The trial proper starts once the corrected
MAESTRO results fix which variant trades and the holdout confirmation has run.

## Status

| Step | Status |
|---|---|
| Re-examine the MSc figures | Done |
| Leakage-free, cost-aware evaluator with look-ahead tests | Done |
| 20 years of data, daily ingestion, macro timing fixed, holdout sealed | Done |
| MSc strategies scored over 20 years | Done: small real edge, nothing beats costs |
| MAESTRO over 20 years, power test, risk layer | Done (re-run after the regime fix): best hit rate, still loses after costs |
| Horizon sweep (1-hour, 4-hour, daily bars) | Done: slower bars don't help; no result distinguishable from luck |
| Monthly vs quarterly retraining | Next |
| Live trial on an OANDA **practice** account | Paper shakedown running |
| Cross-asset extension (gold, S&P 500) | Later |

## How MAESTRO works

Six specialist agents, organised like a trading desk so each part can be tested and switched off on
its own:

| Agent | Role | Under the hood |
|---|---|---|
| Regime | Reads what kind of market this is | Hidden Markov model + Transformer |
| Signal | Forecasts the next 30 minutes | Temporal Fusion Transformer + PatchTST |
| Sentiment | Scores macro and central-bank news | FinBERT + GPT-4o |
| Orchestrator | Weighs the views and makes the call | Regime-weighted fusion, optional LLM reasoning |
| Risk | Sizes or vetoes every trade | CVaR limits, Kelly sizing, reinforcement learning |
| Execution | Places orders and records real costs | Cost-aware order placement |

## Repository layout

The evaluated pipeline:

```
backtesting/     baselines.py (the shared evaluator), maestro_runner.py, power_test.py, risk_layer.py
agents/          regime and signal agents (trained by maestro_runner), risk agent (tested by risk_layer)
data/            raw store and daily ingestion, feature engineering, macro timing, walk-forward splits
live/            the live trial: forecasts, decisions, paper ledger, practice orders
cloud/           runs retraining blocks on Kaggle's free GPUs
deploy/live/     Docker setup for the live trial
demo/            data export for the website
web/             the public website (Next.js, deployed on Vercel)
tests/           look-ahead, data timing, causal regimes, store, runner and live-parity tests
docs/            thesis narrative and design documents
```

Parked: part of the MAESTRO design, not yet in the evaluated pipeline:

```
agents/          sentiment (FinBERT + GPT-4o) and execution agents, per-agent training scripts
orchestrator/    rule-based and LLM orchestrators
compliance/, monitoring/, xai/   limits, dashboards and SHAP explanations for the original live loop
backtesting/     backtest_engine.py and portfolio_ledger.py (the original single-run backtest)
trader/, shared/ multi-asset paper-trading groundwork for the cross-asset extension
tests/system_smoke.py   end-to-end smoke run of the original system (python -m maestro.tests.system_smoke)
```

## Reproducing the results

Python 3.11. The package imports as `maestro`, so run commands from the folder **above** this one.

```bash
python -m venv maestro/venv
maestro/venv/Scripts/pip install -r maestro/requirements.txt
```

Training MAESTRO needs CUDA PyTorch (`pip install torch==2.10.0 --index-url
https://download.pytorch.org/whl/cu128`, plus `hmmlearn statsmodels`).

Price data is not in the repository: OANDA's prices may not be redistributed. Build your own copy
with an OANDA account (a free practice account works) and a free FRED API key, set as
`OANDA_API_KEY` and `FRED_API_KEY` (a `.env` file works):

```bash
python -m maestro.data.pipeline.store fetch --start 2005-01-01   # candles + FRED into $MAESTRO_DATA_DIR/raw
python -m maestro.data.pipeline.store sync                       # rebuild the feature table
python -m maestro.backtesting.maestro_runner --refit-months 3 --train-months 12 --fast
python -m pytest maestro/tests                                   # look-ahead, timing, store, runner, live
python -m maestro.demo.export_web_data                           # rebuild the website's data
```

The 20-year run is 81 retraining blocks of 10–15 minutes each on a laptop RTX 3070 Ti. Every block
is saved as it finishes, so a run can stop and resume, and blocks can be split across machines
(`--shard`, `--reverse`). `cloud/kaggle/` runs the same blocks on Kaggle's free GPUs.

The methodology is documented in full on the site's
[method page](https://maestro-research.vercel.app/method) and in
[`docs/THESIS_NARRATIVE.md`](docs/THESIS_NARRATIVE.md).

## The website

```bash
cd web
npm install
npm run dev        # http://localhost:3000
vercel deploy --prod
```

The site never computes results itself. It reads `web/src/data/*.json`, produced by
`demo/export_web_data.py` from the tested pipeline.

## Disclaimer

Research, not investment advice. All results come from historical simulation; no real money has
been traded. Any live testing uses an OANDA practice account only.
