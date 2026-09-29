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

**Rebuilt under one evaluator, nothing beats buy and hold after costs, MAESTRO included.**
EUR/USD 5-minute bars, 39 walk-forward test months (Jan 2023 to Mar 2026), every model retrained
each quarter on the latest 12 months, 0.8 pips per round trip, £10,000 start:

| Strategy | Right about next move | Trades | Sharpe before costs | Sharpe after costs | £10,000 becomes |
|---|---:|---:|---:|---:|---:|
| Buy and hold (benchmark) | 50.4% | 39 | 0.27 | **0.26** | **£10,667** |
| MAESTRO as designed | n/a | 0 | 0.00 | 0.00 | £10,000 |
| Moving-average crossover 20/200 | 49.7% | 2,089 | −0.04 | −0.67 | £8,517 |
| Bollinger bands 20 / 2σ | 51.2% | 7,990 | 1.95 | −1.28 | £7,963 |
| MAESTRO, most confident 10% | 50.0% | 3,539 | −0.46 | −3.13 | £7,405 |
| MAESTRO, every signal | 49.7% | 4,949 | −0.82 | −3.67 | £6,298 |
| Logistic regression, 5 moves | 50.9% | 50,915 | 1.92 | −11.94 | £399 |
| Contrarian, 3 bars | 50.9% | 63,239 | 2.36 | −15.35 | £182 |
| Linear regression, 5 moves | 50.8% | 106,835 | 2.12 | −21.71 | £6.86 |
| Linear regression, 1 move | 50.7% | 116,552 | 0.64 | −24.09 | £2.49 |
| Coin flip (sanity check) | 49.9% | 115,013 | −0.01 | −24.33 | £2.33 |

Short-horizon EUR/USD has a real, small tendency to reverse: the simple models show a positive
Sharpe before costs. That edge per trade is far smaller than the spread, so only buy and hold ends
positive.

**MAESTRO has no edge even before costs.** Its forecasts are right 49.7–50.0% of the time and its
Sharpe before costs is negative. It loses less than the busy MSc models only because it trades far
less. As designed, its confidence never cleared the thresholds it was built with, so it never traded.
The three ways of turning its forecasts into trades were fixed before any result was seen.

## Status

| Step | Status |
|---|---|
| Re-examine the MSc figures | Done |
| Leakage-free, cost-aware evaluator with look-ahead tests | Done |
| MSc strategies rebuilt and scored | Done |
| MAESTRO scored on the same evaluator | Done: no edge before or after costs |
| Side-by-side live trial on an OANDA **practice** account | Next |
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

```
agents/          the six agents (regime, signal, sentiment, risk, execution)
orchestrator/    rule-based and LLM orchestrators
backtesting/     walk-forward engine, causal portfolio ledger, baselines.py (the shared evaluator)
data/            connectors, feature engineering, labelling, walk-forward splits
trader/          paper-trading engine, portfolio allocator, read-only OANDA client
shared/          contracts shared between research and trading code
tests/           unit tests, including no-look-ahead checks for the evaluator
demo/            data export for the website
web/             the public website (Next.js, deployed on Vercel)
docs/            thesis narrative and design documents
```

## Reproducing the results

Python 3.11. The package imports as `maestro`, so run commands from the folder **above** this one.

```bash
python -m venv maestro/venv
maestro/venv/Scripts/pip install -r maestro/requirements.txt
```

Training MAESTRO needs CUDA PyTorch (`pip install torch==2.10.0 --index-url
https://download.pytorch.org/whl/cu128`, plus `hmmlearn statsmodels`). The full run takes about four
hours on a laptop RTX 3070 Ti and saves each quarterly block as it finishes, so it can resume.

Price data (OANDA EUR/USD 5-minute features) is not in the repository. Point
`MAESTRO_DATA_DIR` at a folder containing `EUR_USD_features.parquet` (a `.env` file works).

```bash
python -m maestro.backtesting.maestro_runner --refit-months 3 --train-months 12   # train MAESTRO, score everything
python -m pytest maestro/tests/test_baselines.py   # the no-look-ahead tests
python -m maestro.demo.export_web_data             # rebuild the website's data
```

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
