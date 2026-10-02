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
| Live trial | Built and in paper-only shakedown on a private always-on server; nightly public snapshot, health check and a hidden live page built | `live/`, `deploy/live/`, `web/src/app/live/` |
| Tests | 86 passing: look-ahead, data timing, causal regimes, store, runner, live parity, live snapshot, practice-only client | `tests/` |
| Public website | Live; MAESTRO's corrected 20-year results in the Strategy Lab (cost-check version included) | [maestro-research.vercel.app](https://maestro-research.vercel.app), `web/` |

## MAESTRO: re-run after the regime fix

Every MAESTRO number produced before commit `5a9f47b` used regimes that saw later bars (the HMM's
Viterbi path and forward-backward smoothing over whole three-month test blocks). Those outputs are
kept for the record in `C:\tmp\maestro_outputs\_leaky_regime\` and must not be cited.

The 20-year quarterly design (`refit3_roll12_fast`, 81 blocks, laptop forward + Kaggle backward) was
re-run on 1 Oct 2026. The leak made almost no difference. At 0.8 pips:

| Variant | Hit | Pips/trade before costs | Sharpe after costs | Months profitable |
|---|---:|---:|---:|---:|
| MAESTRO, every signal | 52.7% | 0.28 | −2.51 | 51/243 |
| MAESTRO, top 10% confidence | 53.1% | 0.23 | −2.50 | 54/243 |
| MAESTRO + cost filter (`risk_cost_filter`) | 53.9% | 0.30 | −0.89 | 74/243 |
| MAESTRO as designed (`maestro_gated`) | 1 trade in 20 years | | | |
| Bollinger bands (best baseline) | 52.6% | 0.70 | −0.30 | 101/243 |

MAESTRO is the most accurate forecaster tested but earns a third of the cost per trade; no variant
beats Bollinger after costs. The leak had flattered the cost filter (Sharpe −0.54 → −0.89). The
power test (`power_test --fast`, 4 blocks of 2024-25 per level) passes on the fixed code: with no
planted edge MAESTRO's top 10% hit 53.1%; at a planted 52% it found nothing (50.1%); at a planted 55%
it found it (top 10% 58.4%, every signal 56.1%) and made money after costs (Sharpe 7.0 and 7.8).
So a moderate edge would be found; MAESTRO's real-data losses are not a pipeline failure.

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

1. ~~Horizon sweep~~ done 1 Oct 2026 (`H1_refit3_roll12_fast`, `H4_refit3_roll12_fast`,
   `D_refit3_expanding_fast`; risk layer on H1 too). Slower bars don't rescue MAESTRO. The one
   positive MAESTRO result anywhere, hourly top 10% (+0.36 pips/trade after 0.8 pips, Sharpe +0.14,
   121/243 months), is noise: t = 0.69 on daily returns, 12 of 21 years positive, 2009 alone is more
   than all of its profit, negative at 1.5 pips, and it is the best of many variants tried. On 4-hour
   and daily bars MAESTRO is below 50% hit and loses; only the moving-average crossover is faintly
   positive (Sharpe 0.25 and 0.22, also not significant). The hourly cost filter loses (−0.11).
2. The monthly-retraining comparison (`refit1_roll12_fast`, 243 blocks), started 1 Oct 2026:
   Kaggle backwards (`launch --refit-months 1 --train-months 12 --reverse --budget-hours 11`,
   relaunch each session until done; `collect` after each) and the laptop forwards (log
   `C:\tmp\maestro_outputs\maestro\monthly_laptop.log`). Score with `--score-only`, then run
   `risk_layer --refit-months 1 --train-months 12 --fast`.
3. Freeze the design; run the sealed holdout once (`--holdout`).
4. Regenerate the live expectations from the frozen design and commit them *before* the trial
   (`python -m maestro.live.expectations`): they are the trial's published prediction.
5. The practice-order variant is decided (2 Oct 2026): **`maestro_top10`**, about 6 trades a day.
   The cost check (`risk_cost_filter`) was the first choice but trades in bursts: whole quarters
   with no trades (2021 Q1, 2025 Q1), 28 qualifying bars in 2026 Q1, and none in the shakedown's
   first 1.5 days (largest live forecast 0.67 pips against its 1.3-pip bar). Practice orders exist
   to measure real fills, spreads and delays, which needs orders, so they go to `maestro_top10`;
   the cost check is still paper-traded and featured beside it on the live page. Train the live
   model (`live.deploy`), copy it to the server, restart with `--orders maestro_top10` and
   `MAESTRO_LIVE_PHASE=trial`. The 4–8 week trial starts then; go public as in `deploy/live/README.md`.

## Live trial

- Runs in Docker on a private always-on ARM server (details kept outside this public repo). Its
  folder `~/maestro-live` holds `Dockerfile`, `docker-compose.yml`, a `.env` with only the four
  trial settings, and `state/` (models, `journal.db`, `status.json`, `live.log`).
- Paper-only shakedown since 30 Sep 2026 with the model `state/models/shakedown-2026-09-30`.
  No practice orders are sent until a variant is chosen and confirmed.
- Update the model: `python -m maestro.live.deploy --out <dir>` on the laptop, copy the folder to
  `state/models/`, point `state/models/current` at it, then `docker compose restart`.
- Watch: `docker exec maestro-live python -m maestro.live.report` (health, every strategy in pips,
  practice fills), `cat state/status.json` or `docker compose logs -f`.
- Nightly snapshot: `maestro-publish` pushes `live/snapshot.py`'s output to the `maestro-live`
  repository at 00:20 UTC; that repository's Action runs `live/check.py` and emails on failure.
  The live page (`web/src/app/live/`) reads it and stays a 404 until `LIVE_PUBLIC=1` on Vercel.
  Preview it locally with `LIVE_SNAPSHOT_FILE=<snapshot.json>` in `web/.env.local` and `npm run dev`.
- Live quoted spreads in the shakedown average about 0.8 pips, well under the 1.5-pip median of the
  candles' bid/ask since 2022; worth explaining in the thesis (quote source and timing differ).
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
  `H1_refit3_roll12_fast`). Research runs keep each block's forecasts (`block_NN.parquet`) and its
  settings (`block_NN.json`), not the trained models: a model is trained, forecasts its test
  months and is discarded, and any block can be retrained from its settings.
- **Live models**: the only saved models. `live.deploy` writes one to
  `C:\tmp\maestro_live\models\<name>\` (regime, signal, `meta.json`, seed forecasts); a copy goes to
  the server's `state/models/<name>/`, and the `current` link there picks the one in use.
- **Backups**: the research outputs and live models exist only on the laptop, so they are copied to
  the server as `~/maestro-backup/outputs-<date>.tar.gz` after each big run:
  ```bash
  cd C:/tmp && tar czf - maestro_outputs maestro_live/models | ssh <server> "cat > ~/maestro-backup/outputs-$(date +%F).tar.gz"
  ```
  First backup 2 Oct 2026 (129 MB, 1,781 files). Code is on GitHub, data in `maestro-data`, the live
  trial on the server and in `maestro-live`, so nothing else is laptop-only except three secrets:
  the server's SSH key (keep a copy in a password manager; without it, recover through Oracle's
  console), `.env` (reissue from OANDA and FRED) and the Kaggle token (reissue on Kaggle).
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
python -m pytest maestro/tests
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
