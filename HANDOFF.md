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
| Shared evaluator | One scorer for every strategy, 39 expanding walk-forward months, 5-day embargo, 0.8 pip round trip | `backtesting/baselines.py` |
| No-look-ahead tests | 8 tests, all passing | `tests/test_baselines.py` |
| MSc strategies rebuilt | All lose after costs; only buy and hold is positive | README table |
| MAESTRO edge check | Pooled directional hit 50.82% over 25,198 calls (z = +2.61): statistically detectable, economically too small (breakeven needs ~58–68% at 0.8–1.8 pips) | `modal_edge.py` |
| Paper-trading groundwork | Instrument specs, causal portfolio ledger, paper broker/engine, allocator, read-only OANDA client — 35 tests pass | `trader/`, `backtesting/portfolio_ledger.py` |
| Public website | Live, interactive, built only from exported results | [maestro-research.vercel.app](https://maestro-research.vercel.app), `web/` |

## What is next: Chapter 7, MAESTRO vs the baselines

Score MAESTRO with the **same** evaluator as the MSc strategies and add it to the table and the site.

1. Save MAESTRO's per-bar signals for every walk-forward split (the Modal run only returned hit rates).
2. Score two versions: **as designed** (trades only above its confidence threshold) and **always in the
   market** (the like-for-like comparison with the MSc models).
3. Compare against logistic regression and buy and hold after costs.
4. Re-export data and redeploy the website.

### Decision still open: how to run it

There is **no budget for Modal**, so the run goes on the laptop GPU (RTX 3070 Ti, 8 GB, ~7× faster
than CPU once CUDA PyTorch is installed).

| Option | Laptop time | Consequence |
|---|---|---|
| Retrain every quarter on the latest 12 months | ~3 hours | Re-score the MSc strategies the same way (seconds) and update site copy |
| Keep monthly retrains on all history | ~30–40 hours | No design change; must be resumable across several nights |

Whichever is chosen, the runner should save each split as it finishes so a sleep or crash does not
lose work (the first expanding-window laptop run was killed mid-way).

## Environment

- **Run from the folder above the repo** (`C:\Users\fran6\Downloads`); the package imports as `maestro`.
- **venv**: `maestro/venv` was recreated with core packages only (numpy, pandas 3, scikit-learn,
  scipy, pyarrow, pyyaml, python-dotenv, pytest). Model training also needs:
  ```bash
  maestro/venv/Scripts/pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
  maestro/venv/Scripts/pip install hmmlearn statsmodels
  ```
- **Data**: `C:\tmp\maestro_data\EUR_USD_features.parquet` (restored from the Modal volume
  `maestro-data`). Earlier saved models and outputs under `C:\tmp` were lost and are regenerable.
- **Evaluator outputs**: `C:\tmp\maestro_outputs\baselines\`.
- **Device**: model configs auto-select CUDA when available and reload saved models onto the current device.

## Common commands

```bash
python -m maestro.backtesting.baselines             # re-score the MSc strategies
python -m pytest maestro/tests                      # full test suite
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
