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

**Rebuilt under one evaluator, no MSc strategy survives trading costs.** EUR/USD 5-minute bars,
39 walk-forward months (Jan 2023 to Mar 2026), 0.8 pips per round trip, £10,000 start:

| Strategy | Right about next move | Trades | Sharpe before costs | Sharpe after costs | £10,000 becomes |
|---|---:|---:|---:|---:|---:|
| Buy and hold (benchmark) | 50.4% | 39 | 0.27 | **0.26** | **£10,667** |
| Moving-average crossover 20/200 | 49.7% | 2,089 | −0.04 | −0.67 | £8,517 |
| Bollinger bands 20 / 2σ | 51.2% | 7,990 | 1.95 | −1.28 | £7,963 |
| Logistic regression, 5 moves | 51.1% | 63,738 | 2.62 | −14.82 | £188 |
| Contrarian, 3 bars | 50.9% | 63,239 | 2.36 | −15.35 | £182 |
| Linear regression, 5 moves | 50.8% | 114,572 | 1.91 | −23.25 | £3.86 |
| Linear regression, 1 move | 50.8% | 116,552 | 1.36 | −23.98 | £3.03 |
| Coin flip (sanity check) | 50.1% | 115,108 | 0.31 | −24.66 | £2.51 |

Short-horizon EUR/USD has a real, small tendency to reverse (positive Sharpe before costs), but the
edge per trade is far smaller than the spread. Only buy and hold ends positive.

**Next:** MAESTRO's six-agent system goes through exactly the same evaluator. To count as progress it
has to beat the logistic regression and buy and hold after costs. A clear "no" is a valid result.

## Status

| Step | Status |
|---|---|
| Re-examine the MSc figures | Done |
| Leakage-free, cost-aware evaluator with look-ahead tests | Done |
| MSc strategies rebuilt and scored | Done |
| MAESTRO scored on the same evaluator | Next |
| Side-by-side live trial on an OANDA **practice** account | Planned |
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

Price data (OANDA EUR/USD 5-minute features) is not in the repository. Point
`MAESTRO_DATA_DIR` at a folder containing `EUR_USD_features.parquet` (a `.env` file works).

```bash
python -m maestro.backtesting.baselines            # score every MSc strategy
python -m pytest maestro/tests                     # includes the no-look-ahead tests
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
