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

**Rebuilt under one evaluator over 20 years, no strategy survives trading costs, MAESTRO included.**
EUR/USD 5-minute bars, 243 walk-forward test months (Jan 2006 to Feb 2026), every model retrained
each quarter on the latest 12 months, 0.8 pips per round trip, £10,000 start:

| Strategy | Right about next move | Trades | Pips per trade before costs | Sharpe before costs | Sharpe after costs | £10,000 becomes |
|---|---:|---:|---:|---:|---:|---:|
| MAESTRO as designed | n/a | 0 | n/a | 0.00 | 0.00 | £10,000 |
| Buy and hold (benchmark) | 50.2% | 243 | −0.40 | −0.01 | −0.02 | £9,637 |
| Bollinger bands 20 / 2σ | 52.6% | 51,966 | +0.70 | 2.07 | −0.30 | £6,474 |
| Moving-average crossover 20/200 | 49.4% | 13,302 | +0.07 | 0.01 | −0.43 | £4,254 |
| MAESTRO, most confident 10% | **53.0%** | 40,563 | +0.20 | 0.84 | −2.60 | £1,351 |
| MAESTRO, every signal | 52.7% | 66,933 | +0.27 | 1.27 | −2.66 | £530 |
| Contrarian, 3 bars | 52.0% | 415,067 | +0.20 | 3.49 | −10.40 | < £0.01 |
| Logistic regression, 5 moves | 52.3% | 450,504 | +0.21 | 3.84 | −10.96 | < £0.01 |
| Linear regression, 5 moves | 51.9% | 662,098 | +0.14 | 3.69 | −15.48 | < £0.01 |
| Linear regression, 1 move | 51.9% | 757,250 | +0.12 | 3.51 | −18.48 | < £0.01 |
| Coin flip (sanity check) | 50.0% | 735,364 | +0.00 | 0.11 | −19.25 | < £0.01 |

**MAESTRO learned a real edge, and it faded.** Over 20 years it calls the next move right more often
than any other strategy, and it earns 0.2–0.3 pips per trade before costs, on a par with the simple
models. By era, the edge before costs (most confident 10% / every signal) was:

| | 2006–10 | 2011–15 | 2016–20 | 2021–26 |
|---|---:|---:|---:|---:|
| MAESTRO, pips per trade before costs | +0.19 / +0.26 | +0.36 / +0.46 | +0.19 / +0.23 | about 0 |
| Logistic regression | +0.16 | +0.29 | +0.23 | +0.14 |
| Bollinger bands | +0.76 | +0.77 | +0.74 | +0.54 |

The simple models kept part of their edge after 2020; MAESTRO lost all of it. An earlier test over
2023–2026 alone found MAESTRO with no edge at all, which this explains.

**No edge is large enough to pay for trading.** The best, Bollinger bands, earns 0.70 pips per trade
before costs against the 0.8 pips assumed here. OANDA's real EUR/USD spread, measured on every bar,
has a median of 0.9–1.6 pips depending on the year (1.5–1.6 since 2022), so real costs are higher
still. Buy and hold earns nothing over the full 20 years; its good showing in 2023–2026 was that period.

**The test could have found an edge.** In a power test a known momentum edge was planted in real prices
and every feature rebuilt. At a 55% edge MAESTRO found it (57.7% hit, Sharpe 6.0 after costs); at a
52% edge logistic regression found it but MAESTRO largely did not. MAESTRO is therefore less sensitive
than simple models to small edges, the kind a real market might hold.

**How the evaluation was checked.**
- Macro inputs (VIX, yields, rates, CPI) reach a bar only once they were public. An earlier version
  stamped them at midnight on the day they describe, and CPI six weeks before release.
- Data from 7 March 2026 is sealed and has influenced no decision. It is opened once, for a final
  confirmation run of a frozen design.
- Every MAESTRO block's dates were checked across the two machines that trained them.
- The three ways of turning MAESTRO's forecasts into trades were fixed before any result was seen.

## Status

| Step | Status |
|---|---|
| Re-examine the MSc figures | Done |
| Leakage-free, cost-aware evaluator with look-ahead tests | Done |
| 20 years of data, macro timing fixed, holdout sealed | Done |
| MSc strategies and MAESTRO scored over 20 years | Done: real but fading edge, nothing beats costs |
| Power test: can the pipeline find a planted edge? | Done |
| Monthly vs quarterly retraining | Running |
| Horizon sweep (1-hour, 4-hour, daily bars) and the risk agent | Next |
| Side-by-side live trial on an OANDA **practice** account | After that |
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
cloud/           runs retraining blocks on Kaggle's free GPUs
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
https://download.pytorch.org/whl/cu128`, plus `hmmlearn statsmodels`).

Price data is not in the repository: OANDA's prices may not be redistributed. Build your own copy
with an OANDA account (a free practice account works) and a free FRED API key, set as
`OANDA_API_KEY` and `FRED_API_KEY` (a `.env` file works):

```bash
python -m maestro.data.pipeline.store fetch --start 2005-01-01   # candles + FRED into $MAESTRO_DATA_DIR/raw
python -m maestro.data.pipeline.store sync                       # rebuild the feature table
python -m maestro.backtesting.maestro_runner --refit-months 3 --train-months 12 --fast
python -m pytest maestro/tests --ignore=maestro/tests/smoke_test.py   # look-ahead, timing, store, runner
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
