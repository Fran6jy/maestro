# MAESTRO — Handoff

Where the project stands and how to pick it up. Last updated 29 September 2026.

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
| Shared evaluator | One scorer for every strategy, 39 walk-forward months, quarterly retraining on the latest 12 months, 5-day embargo, 0.8 pip round trip | `backtesting/baselines.py` |
| No-look-ahead tests | 13 tests, all passing (scorer, retraining plan, MAESTRO positions) | `tests/test_baselines.py` |
| MSc strategies rebuilt | All lose after costs; only buy and hold is positive | README table |
| MAESTRO vs baselines | No edge before or after costs; see below | `backtesting/maestro_runner.py` |
| Paper-trading groundwork | Instrument specs, causal portfolio ledger, paper broker/engine, allocator, read-only OANDA client — 35 tests pass | `trader/`, `backtesting/portfolio_ledger.py` |
| Public website | Live, interactive, built only from exported results | [maestro-research.vercel.app](https://maestro-research.vercel.app), `web/` |

## Chapter 7 result: MAESTRO vs the baselines (done)

Every model is retrained each quarter on the latest 12 months (13 retrainings, 39 test months
traded blind), scored by the same code at 0.8 pips per round trip.

| Version (fixed before results) | Hit | Trades | Sharpe gross | Sharpe net | £10k becomes |
|---|---:|---:|---:|---:|---:|
| As designed (per-regime confidence thresholds) | n/a | 0 | 0.00 | 0.00 | £10,000 |
| Most confident 10% (cut-off from previous 5 days) | 50.0% | 3,539 | −0.46 | −3.13 | £7,405 |
| Every signal | 49.7% | 4,949 | −0.82 | −3.67 | £6,298 |
| Logistic regression (reference) | 50.9% | 50,915 | 1.92 | −11.94 | £399 |
| Buy and hold (reference) | 50.4% | 39 | 0.27 | 0.26 | £10,667 |

MAESTRO has no edge even before costs. It loses less than the busy MSc models only because it trades
less; moving-average and Bollinger still beat it. As designed, it never cleared its confidence thresholds.

Bugs found and fixed on the way (they belong in the thesis methods chapter):

- **TFT collapse**: trained on raw 5-minute returns (~1e-4), it output one constant for every bar,
  so MAESTRO said BUY on 100% of bars. Targets are now scaled per horizon (`target_scale`).
- **TFT horizons**: "horizon 6" read the single-bar forecast at step 3. Targets are now cumulative
  and horizon h reads step h−1.
- **One-bar lag**: TFT and PatchTST forecasts were stamped a bar late (conservative, not a leak).
- **PatchTST speed**: per-channel Python loop batched; identical output, 11× faster.
- **Data gaps**: ~0.3% of bars have a NaN candle feature; positions carry through them (≤ 1 hour)
  instead of paying fake round trips.

The earlier Modal "edge" figure (50.82% over 25,198 calls) came from the buggy TFT. Don't cite it.

Outputs: `C:\tmp\maestro_outputs\maestro\refit3_roll12\` (per-block signals, `scores/`, run log).
efit3_roll12\` (per-block signals, `scores/`, run log).

## What is next: Chapter 8, live practice trial

MSc strategies and MAESTRO side by side on the OANDA **practice** account for 4–8 weeks, with daily
reconciliation of live decisions and fills against the backtest. Open questions to settle first:
which MAESTRO version to run live (most confident 10% is the natural choice), and whether to keep
the confidence-bucket and ablation analyses in Chapter 7.

## Environment

- **Run from the folder above the repo** (`C:\Users\fran6\Downloads`); the package imports as `maestro`.
- **venv**: `maestro/venv` has the core packages (numpy, pandas 3, scikit-learn, scipy, pyarrow,
  pyyaml, python-dotenv, pytest) plus CUDA PyTorch, hmmlearn and statsmodels. To rebuild the extras:
  ```bash
  maestro/venv/Scripts/pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
  maestro/venv/Scripts/pip install hmmlearn statsmodels
  ```
- **Data**: `C:\tmp\maestro_data\EUR_USD_features.parquet` (restored from the Modal volume
  `maestro-data`). Earlier saved models and outputs under `C:\tmp` were lost and are regenerable.
- **Evaluator outputs**: `C:\tmp\maestro_outputs\baselines\<design>\` and `C:\tmp\maestro_outputs\maestro\<design>\`
  (design tags such as `refit3_roll12`; the site reads the MAESTRO run's `scores\` folder).
- **Device**: model configs auto-select CUDA when available and reload saved models onto the current device.

## Common commands

```bash
python -m maestro.backtesting.maestro_runner --refit-months 3 --train-months 12   # ~4 h on the GPU, resumable
python -m maestro.backtesting.maestro_runner --score-only                         # re-score saved blocks
python -m maestro.backtesting.baselines --refit-months 3 --train-months 12        # baselines only
python -m pytest maestro/tests/test_baselines.py    # no-look-ahead tests (tests/smoke_test.py is a script, not pytest)
python -m maestro.demo.export_web_data              # refresh web/src/data/*.json
cd maestro/web && npm run build && vercel deploy --prod --yes   # redeploy the site
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
