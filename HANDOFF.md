# MAESTRO — Handoff

Where the project stands and how to pick it up. Last updated 30 September 2026 (evening).

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
| Shared evaluator | One scorer for every strategy, 243 walk-forward months (2006–2026), retraining on the latest 12 months, 5-day embargo, 0.8 pip round trip, costs charged on turnover | `backtesting/baselines.py` |
| Data | OANDA EUR/USD + GBP/USD 5-minute candles with bid/ask from 2005, FRED macro stamped at publication time, ingested daily by GitHub Actions into a private repo; 1-hour, 4-hour and daily bars built from them | `data/pipeline/store.py`, `data/features/macro.py` |
| Holdout | Everything from 7 Mar 2026 is sealed; `--holdout` opens it once, for the frozen design | `data/holdout.py` |
| Baselines, 20 years | Small real edge (52–53% hit, +0.1–0.7 pips/trade before costs); none beats costs | README table |
| Regime look-ahead fixed | HMM regimes now forward-filtered; all MAESTRO results before the fix are invalid and being re-run | `agents/regime/hmm_regime.py` |
| Live trial | Built and in paper-only shakedown on a private always-on server | `live/`, `deploy/live/` |
| Tests | 82 passing: look-ahead, data timing, causal regimes, store, runner, live parity, practice-only client | `tests/` |
| Public website | Live; MAESTRO section flagged as being re-run | [maestro-research.vercel.app](https://maestro-research.vercel.app), `web/` |

## MAESTRO: being re-run

Every MAESTRO number produced before commit `5a9f47b` used regimes that saw later bars (the HMM's
Viterbi path and forward-backward smoothing over whole three-month test blocks). That includes the
20-year result (53% hit, an edge that faded after 2020), the power test and the risk-layer tests.
Those outputs are kept for the record in `C:\tmp\maestro_outputs\_leaky_regime\` and must not be
cited. The re-run of the 20-year quarterly design (`refit3_roll12_fast`) is split between the laptop
(forward) and Kaggle (backward).

Bugs found and fixed so far (they belong in the thesis methods chapter):

- **Macro look-ahead**: FRED values were stamped at midnight on the date they describe (each bar saw
  that day's closing VIX; CPI arrived six weeks before release). Now stamped at publication time.
- **Regime look-ahead**: see above. Predictions forward-filter; training labels may use hindsight.
- **TFT collapse**: trained on raw 5-minute returns (~1e-4), it output one constant for every bar.
  Targets are now scaled per horizon (`target_scale`).
- **TFT horizons**: "horizon 6" read the single-bar forecast at step 3. Targets are now cumulative.
- **One-bar lag**: TFT and PatchTST forecasts were stamped a bar late (conservative, not a leak).
- **Risk agent cost filter**: used the signed forecast, so it rejected every short.
- **Regime model persistence**: a reloaded model forgot its tuned HMM vote weight (live only).
- **`SignalAgent.predict_bar`**: its TFT forecast is an hour stale; the live trial doesn't use it.
- **Data gaps**: ~0.3% of bars have a NaN candle feature; positions carry through them (≤ 1 hour).

## What is next

1. Finish the causal 20-year re-run; collect Kaggle's blocks (`cloud.kaggle.collect`), score it.
2. Re-run the power test and the risk layer on the fixed code; then the monthly-retraining
   comparison and the horizon sweep (`--granularity H1|H4|D`).
3. Freeze the design; run the sealed holdout once (`--holdout`).
4. Choose the MAESTRO variant that trades on the practice account; train the live model
   (`live.deploy`), copy it to the server, restart with `--orders <variant>`. The 4–8 week trial
   starts then. Build the daily live-vs-backtest reconciliation during the shakedown.
5. Update the website and README with the corrected results.

## Live trial

- Runs in Docker on a private always-on ARM server (details kept outside this public repo). Its
  folder `~/maestro-live` holds `Dockerfile`, `docker-compose.yml`, a `.env` with only the four
  trial settings, and `state/` (models, `journal.db`, `status.json`, `live.log`).
- Paper-only shakedown since 30 Sep 2026 with the model `state/models/shakedown-2026-09-30`.
  No practice orders are sent until a variant is chosen and confirmed.
- Update the model: `python -m maestro.live.deploy --out <dir>` on the laptop, copy the folder to
  `state/models/`, point `state/models/current` at it, then `docker compose restart`.
- Watch: `cat state/status.json` (last bar, targets, any error) or `docker compose logs -f`.
- Each cycle re-filters regimes from the model's start date; if cycles slow down as that history
  grows, make the filter incremental.

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
- **Features**: `C:\tmp\maestro_data\EUR_USD[_H1|_H4|_D]_features.parquet`, rebuilt by `store sync`.
- **Outputs**: `C:\tmp\maestro_outputs\maestro\<design>\` (tags such as `refit3_roll12_fast`,
  `H1_refit3_roll12_fast`).
- **Compute**: laptop RTX 3070 Ti (about 12 minutes per block), or Kaggle's free T4 ×2 (about 30
  GPU-hours a week) through `cloud/kaggle/` (token in `~/.kaggle/access_token`; private dataset
  `fran6jy/maestro-raw`). **Do not use Modal** (no budget; September went over the free credit).

## Common commands

```bash
python -m maestro.data.pipeline.store sync --granularity M5 H1 H4 D          # pull raw data, rebuild features
python -m maestro.backtesting.maestro_runner --refit-months 3 --train-months 12 --fast   # resumable
python -m maestro.backtesting.maestro_runner --refit-months 3 --train-months 12 --fast --score-only
python -m maestro.cloud.kaggle.launch --refit-months 3 --train-months 12 --reverse --commit <pushed>
python -m maestro.cloud.kaggle.collect                                       # bring Kaggle's blocks back
python -m maestro.backtesting.power_test                                     # planted-edge check
python -m maestro.backtesting.risk_layer --refit-months 3 --train-months 12 --fast
python -m maestro.live.deploy --out C:\tmp\maestro_live\models\<date>       # live model
python -m pytest maestro/tests --ignore=maestro/tests/smoke_test.py
python -m maestro.demo.export_web_data                                       # refresh web/src/data/*.json
cd maestro/web && npm run build && vercel deploy --prod --yes                # redeploy the site
```

## Website

- Vercel project `maestro-research` (scope `fran6jy-7215s-projects`), public at
  https://maestro-research.vercel.app. Hobby plan, no cost.
- Next.js 16 in `web/`; read `web/AGENTS.md` before changing it (Next 16 has breaking changes).
- It reads only `web/src/data/*.json`. The Strategy Lab currently shows the 2023–2026 baselines and
  a notice that MAESTRO's results are being re-run; replace both with the corrected 20-year results.
- Your name is deliberately not on the site; add it in `web/src/components/Footer.tsx` if wanted.

## Known issues

- `backtesting/backtest_engine.py` annualises Sharpe with `sqrt(252 * 78)` (stock-market hours).
  FX trades around the clock, so its Sharpe figures are understated by about 1.9×. The shared
  evaluator avoids this by using daily returns.
- MAESTRO's regime labels depend on where the HMM filter starts (it is slow to forget); the live
  trial pins the start exactly as a backtest block does.
- Kaggle's output download drops large files over SSL; `collect` retries, and stragglers can be
  fetched one at a time with `kaggle kernels output ... --file-pattern`.
- `config/settings.yaml` lists eight instruments, so `DataPipeline.run_full()` would fetch all of them.
- The LLM orchestrator, cross-asset v2 design and demo layer are parked as future work.

## Rules for this repository

- You (Fran6jy) are the sole author and contributor: no co-author trailers on commits or PR footers.
- The repository is public: never commit `.env`, API keys, account IDs or raw price data.
- Live testing uses an OANDA **practice** account only.
