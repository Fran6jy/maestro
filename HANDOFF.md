# MAESTRO — Handoff

Where the project stands and how to pick it up. Last updated 30 September 2026.

## The one-line direction

Everything serves this sentence (see [`docs/THESIS_NARRATIVE.md`](docs/THESIS_NARRATIVE.md)):

> My MSc asked the right questions but couldn't answer them rigorously. The PhD rebuilds the same
> strategies under a leakage-free, cost-aware walk-forward evaluation, shows what they actually
> achieve, then tests whether a multi-agent system does better.

If a task doesn't serve it, it is future work.

## What is done

| Area | Result | Where |
|---|---|---|
| MSc re-examined | 37.56% / 0.599 come from two unrelated tests; the regression re-runs at 46.6% | README, site `/method` |
| Shared evaluator | One scorer for every strategy, 243 walk-forward months (2006–2026), retraining on the latest 12 months, 5-day embargo, 0.8 pip round trip | `backtesting/baselines.py` |
| Data | OANDA EUR/USD + GBP/USD 5-minute candles with bid/ask from 2005, FRED macro stamped at publication time, ingested daily by GitHub Actions into a private repo | `data/pipeline/store.py`, `data/features/macro.py` |
| Holdout | Everything from 7 Mar 2026 is sealed; `--holdout` opens it once, for the frozen design | `data/holdout.py` |
| Tests | 59 passing: look-ahead, data timing, store layout, power-test maths, runner scheduling, trader | `tests/` |
| Power test | Finds a planted 55% edge; mostly misses a 52% edge that logistic regression finds | `backtesting/power_test.py` |
| MAESTRO vs baselines, 20 years | Real but fading edge before costs; nothing beats costs; see below | `backtesting/maestro_runner.py` |
| Paper-trading groundwork | Instrument specs, causal portfolio ledger, paper broker/engine, allocator, read-only OANDA client | `trader/`, `backtesting/portfolio_ledger.py` |
| Public website | Live, but still shows the 2023–2026 results; update after the monthly run | [maestro-research.vercel.app](https://maestro-research.vercel.app), `web/` |

## Chapter 7 result: MAESTRO vs the baselines, 20 years

Quarterly retraining on the latest 12 months, 81 blocks, 243 test months traded blind (Jan 2006 to
Feb 2026), 0.8 pips per round trip. Full table in the README.

| Version (fixed before results) | Hit | Pips/trade before costs | Sharpe gross | Sharpe net |
|---|---:|---:|---:|---:|
| As designed (per-regime confidence thresholds) | n/a | n/a (never trades) | 0.00 | 0.00 |
| Most confident 10% (cut-off from previous 5 days) | 53.0% | +0.20 | 0.84 | −2.60 |
| Every signal | 52.7% | +0.27 | 1.27 | −2.66 |
| Logistic regression (reference) | 52.3% | +0.21 | 3.84 | −10.96 |
| Bollinger bands (best baseline) | 52.6% | +0.70 | 2.07 | −0.30 |
| Buy and hold (reference) | 50.2% | n/a | −0.01 | −0.02 |

MAESTRO's edge before costs peaked in 2011–15 (+0.36–0.46 pips/trade) and was about zero by
2021–26, while logistic regression and Bollinger kept part of theirs. Real OANDA spreads (median
0.9–1.6 pips by year) are wider than the 0.8 pips assumed, so real results would be worse.

Bugs found and fixed on the way (they belong in the thesis methods chapter):

- **Macro look-ahead**: FRED values were stamped at midnight on the date they describe (each bar saw
  that day's closing VIX; CPI arrived six weeks before release). Now stamped at publication time.
- **TFT collapse**: trained on raw 5-minute returns (~1e-4), it output one constant for every bar,
  so MAESTRO said BUY on 100% of bars. Targets are now scaled per horizon (`target_scale`).
- **TFT horizons**: "horizon 6" read the single-bar forecast at step 3. Targets are now cumulative
  and horizon h reads step h−1.
- **One-bar lag**: TFT and PatchTST forecasts were stamped a bar late (conservative, not a leak).
- **Data gaps**: ~0.3% of bars have a NaN candle feature; positions carry through them (≤ 1 hour)
  instead of paying fake round trips.

The earlier Modal "edge" figure (50.82%) and the 2023–26-only results came from code before these
fixes. Cite the 20-year run.

Outputs: `C:\tmp\maestro_outputs\maestro\refit3_roll12_fast\` (per-block signals, `scores\`).

## What is next

1. **Monthly vs quarterly retraining** — running on Kaggle (`refit1_roll12_fast`, 242 blocks).
2. **Horizon sweep** — the same comparison on 1-hour, 4-hour and daily bars.
3. **Risk agent** in the backtest.
4. Update the website and README once 1–3 are in; then the holdout confirmation run of the frozen
   design; then Chapter 8, the side-by-side OANDA **practice** trial.

## Environment

- **Run from the folder above the repo** (`C:\Users\fran6\Downloads`); the package imports as `maestro`.
- **venv**: `maestro/venv` has the core packages plus CUDA PyTorch, hmmlearn, statsmodels and the
  Kaggle CLI. To rebuild the extras:
  ```bash
  maestro/venv/Scripts/pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
  maestro/venv/Scripts/pip install hmmlearn statsmodels kaggle
  ```
- **Raw data**: `C:\tmp\maestro_data\raw\` is a clone of the private repo `Fran6jy/maestro-data`,
  updated daily at 22:30 UTC by its GitHub Actions workflow (secrets `OANDA_API_KEY`,
  `FRED_API_KEY`). Never commit data to this public repo.
- **Features**: `C:\tmp\maestro_data\EUR_USD_features.parquet`, rebuilt by `store sync`.
  `EUR_USD_features_legacy_2022_2026.parquet` is the file the old site numbers came from.
- **Outputs**: `C:\tmp\maestro_outputs\maestro\<design>\` (design tags such as `refit3_roll12_fast`).
- **Compute**: laptop RTX 3070 Ti (about 12 minutes per block), or Kaggle's free T4 ×2 (about 30
  GPU-hours a week) through `cloud/kaggle/`. Kaggle token: `~/.kaggle/access_token`. Private dataset
  `fran6jy/maestro-raw` holds the raw store for Kaggle; re-upload it yourself if it needs refreshing.
  **Do not use Modal** (no budget; September went over the free credit).

## Common commands

```bash
python -m maestro.data.pipeline.store sync                                   # pull raw data, rebuild features
python -m maestro.backtesting.maestro_runner --refit-months 3 --train-months 12 --fast   # resumable
python -m maestro.backtesting.maestro_runner --refit-months 3 --train-months 12 --fast --score-only
python -m maestro.cloud.kaggle.launch --refit-months 1 --train-months 12 --reverse       # 6 h Kaggle session
python -m maestro.cloud.kaggle.collect                                       # bring Kaggle's blocks back
python -m maestro.backtesting.power_test                                     # planted-edge check
python -m pytest maestro/tests --ignore=maestro/tests/smoke_test.py
python -m maestro.demo.export_web_data                                       # refresh web/src/data/*.json
cd maestro/web && npm run build && vercel deploy --prod --yes                # redeploy the site
```

## Website

- Vercel project `maestro-research` (scope `fran6jy-7215s-projects`), public at
  https://maestro-research.vercel.app. Hobby plan, no cost.
- Next.js 16 in `web/`; read `web/AGENTS.md` before changing it (Next 16 has breaking changes).
- Sections: ridgeline hero from real EUR/USD days, Strategy Lab with exact cost re-pricing, MSc replay,
  agents control room, roadmap, `/method`. It reads only `web/src/data/*.json`.
- Your name is deliberately not on the site; add it in `web/src/components/Footer.tsx` if wanted.

## Known issues

- `backtesting/backtest_engine.py` annualises Sharpe with `sqrt(252 * 78)` (stock-market hours).
  FX trades around the clock, so its Sharpe figures are understated by about 1.9×. The shared
  evaluator avoids this by using daily returns.
- The rest of the pipeline has not been re-tested on pandas 3.
- `config/settings.yaml` now lists eight instruments, so `DataPipeline.run_full()` would fetch all of them.
- The LLM orchestrator, cross-asset v2 design and demo layer are parked as future work.

## Rules for this repository

- You (Fran6jy) are the sole author and contributor: no co-author trailers on commits or PR footers.
- The repository is public: never commit `.env`, API keys, account IDs or raw price data.
- Live testing uses an OANDA **practice** account only.
