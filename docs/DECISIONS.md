# Decision log

Every decision that shaped the results, in date order, with the evidence it rested on and the
commit that recorded it. This is the record to defend the work from: what was decided, when,
on what numbers, and what was rejected. Numbers are at the 0.8-pip reference cost unless
stated. `HANDOFF.md` has the operating detail; `README.md` the public summary.

Conventions that hold throughout: every strategy is scored by one shared evaluator
(`backtesting/baselines.py`); a position decided at a bar's close earns the next bar; costs are
charged on turnover; test months are never seen in training; data from 7 March 2026 is sealed
until the one confirmation run below. Commits are by the author alone.

## 2026-09-29

- **Rebuild the MSc strategies under one evaluator** (`78e7cd9`). The MSc's 37.56% hit rate and
  0.599 Sharpe came from two unrelated tests and do not reproduce; the regression re-runs at
  46.6% on its own month. Decision: every result on the site and in the thesis comes from the
  shared evaluator, never from the original scripts.
- **MAESTRO scored with quarterly retraining on the latest 12 months** (`a36d09f`). Three ways
  of turning forecasts into trades were fixed before any result was seen: as designed
  (confidence thresholds), most confident 10% (cut-off from the previous five trading days),
  every signal.
- **Macro look-ahead found and fixed; holdout sealed** (`6e51c38`). FRED values had been stamped
  at midnight on the day they describe (each bar saw that day's closing VIX; CPI arrived weeks
  before release). Now stamped at publication time. Everything from 7 March 2026 sealed.
- **Power test added** (`7d503e9`): a known edge planted in real prices, to show the pipeline
  can find one. Rejected alternative: trusting a negative result without a positive control.
- **Faster TFT settings adopted** (`7771e4c`, `487a2e6`) after an ablation on the same blocks:
  batch 256, patience 10. bf16 rejected (worse); a higher learning rate rejected. Every
  20-year result uses these settings (`_fast` tag); the ablation showed no loss of accuracy.

## 2026-09-30

- **Regime look-ahead found; every MAESTRO result withdrawn** (`5a9f47b`, `35c52dd`). The HMM
  labelled each bar from the whole three-month test block around it (Viterbi path, forward-
  backward smoothing), so a label could depend on later prices. Fixed to forward filtering.
  Decision: all MAESTRO numbers produced before the fix are invalid, kept only for the record
  in `_leaky_regime/`, and the site says so. The baselines use prices only and were unaffected.
- **Costs charged on turnover** (`dcd552f`), so a risk layer that scales positions pays for
  what it trades. **Risk agent's cost filter fixed for shorts and its tests pre-registered**
  (`752713c`): the filter had used the signed forecast and rejected every short.
- **Horizon sweep designed** (`483a031`): the same designs on 1-hour, 4-hour and daily bars,
  to test whether slower bars let the edge outgrow the costs.
- **Live trial built on an OANDA practice account only** (`3af771f`, `4a0ad50`, `cf5c6dd`). The
  client refuses any server but OANDA's practice API. Paper-only shakedown started 22:20 UTC
  with a model trained after the regime fix.

## 2026-10-01

- **Corrected 20-year MAESTRO results** (81 blocks, laptop and Kaggle; block dates checked
  identical across machines) (`d566f1b`). Every signal: 52.7% hit, +0.28 pips/trade before
  costs, Sharpe −2.51. Top 10%: 53.1%, +0.23, −2.50. Cost check: 53.9%, +0.30, −0.89,
  74/243 months profitable. As designed: 1 trade in 20 years. Bollinger bands: 52.6%, +0.70,
  −0.30, 101/243. The leak had barely mattered (cost check −0.54 → −0.89; others within 0.2).
  Decision: MAESTRO is the most accurate forecaster tested and loses after costs; results
  restored to the site with a dated correction note.
- **Power test passed on the fixed code** (`d566f1b`): no planted edge, top 10% 53.1%; planted
  52%, 50.1% (nothing found); planted 55%, 58.4% and profitable after costs (Sharpe 7.0). So a
  moderate edge would be found; the real-data loss is not a pipeline failure.
- **Horizon sweep: slower bars do not help** (`df4e8a1`). The one positive MAESTRO result
  anywhere, hourly top 10% (Sharpe +0.14, +0.36 pips/trade), is judged noise: t = 0.69 on daily
  returns, 12 of 21 years positive, 2009 alone exceeds its whole profit, negative at 1.5 pips,
  and it is the best of more than fifteen variants tried. 4-hour and daily MAESTRO are below
  50% hit. Rejected: reporting the hourly result as a finding.
- **Caveat recorded on the cost check** (`d566f1b`, method page): it was first tried after
  results from before the bug fixes had been seen, so it is less blind than the other three
  variants. The sealed holdout and the live trial are its real test.
- **Live trial published nightly, judged against pre-registered ranges** (`bbd6e93`). For
  every strategy and trial length 1–65 days, the backtest's 5/25/50/75/95th percentiles are
  written to `web/src/data/live_expectations.json` and committed before the trial; the live
  page draws them. Snapshots go to the public `maestro-live` repository, never rewritten.
- **Practice-order variant chosen: the cost check** (`69fd867`), as MAESTRO's closest to
  break-even. Superseded the next day (below).

## 2026-10-02

- **Practice orders moved to the top-10% variant** (`14312b4`). Evidence: the cost check trades
  in bursts (no trades in 2021 Q1 or 2025 Q1; 28 qualifying bars in 2026 Q1; none in the first
  1.5 days of the shakedown, largest live forecast 0.67 pips against its 1.3-pip bar). Practice
  orders exist to measure real fills, spreads and delays, which needs orders. The cost check
  stays paper-traded and featured beside it. Rejected: keeping it on orders and accepting a
  trial with possibly no fills.
- **Monthly retraining compared** (243 blocks) (`9611ed3`). Monthly vs quarterly: top 10%
  Sharpe −2.22 vs −2.50; every signal −2.23 vs −2.51; cost check −0.70 vs −0.89 (80 vs 74
  months profitable). Paired daily returns, monthly minus quarterly: t = +0.70, +1.91, +0.45.
  Nothing becomes profitable; Bollinger still beats every MAESTRO version.
- **Design frozen** (`16a107e`): quarterly retraining on the latest 12 months, 5-minute bars,
  fast TFT settings, the four variants as they stand. Reason: monthly is not significantly
  better and is three times costlier to run live. Nothing in MAESTRO is tuned or changed after
  this commit.
- **Predictions stamped from the frozen, sealed backtest** (`e8b067d`), before the holdout was
  opened or the trial started. `created_at` 2026-10-02T19:45:13Z, `code_commit` 16a107e,
  backtest 2006-01-02 to 2026-03-06.
- **Holdout opened once** (`08ad895` added the folder options). Command:
  `maestro_runner --refit-months 3 --train-months 12 --fast --holdout --out <separate folder>`,
  seeded with the frozen blocks 0–79; blocks covering January 2026 onwards retrained with
  March whole. The frozen results folder is untouched. An earlier launch that would have
  written into the frozen folder was stopped before its first block finished (verified: 81
  blocks, scores unchanged). Results are reported whatever they say; see the entry below.
- **Holdout result: the finding holds.** 7 March to 2 October 2026, 180 trading days, four
  retraining blocks. Top 10%: 1,042 trades, 51.6% hit, +0.15 pips/trade before costs, Sharpe
  −4.38, £10,000 → £9,440. Every signal: 51.0%, Sharpe −5.14, £9,301. Cost check: 50 trades,
  +£44 (too few trades to read). As designed: no trades. Bollinger bands: Sharpe −1.69, £9,574.
  Moving-average crossover: +0.10, £10,031. Logistic regression: £4,588. After 65 days every
  featured strategy sat inside its pre-registered 5–95% range: top 10% −£113 against
  [−449, +35]; cost check +£32 against [−277, +65]; Bollinger −£83 against [−448, +449];
  logistic regression −£1,839 against [−2,999, −1,152]. Conclusion confirmed on data no
  decision touched: MAESTRO's accuracy is real and too small for the costs. Decision: none
  (none allowed); the trial proceeds with the top-10% variant on practice orders.

## Standing rules

- A "no" is a valid result and is reported as plainly as a "yes".
- Any number shown publicly traces to a CSV written by the shared evaluator.
- The holdout is opened once. The live trial is the only test after it.
- Practice orders, the trial phase and public listing switch on only on the author's say-so.
