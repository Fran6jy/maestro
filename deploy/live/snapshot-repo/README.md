# MAESTRO live trial: nightly snapshots

The public record of the MAESTRO live trial: every strategy trading EUR/USD on every
5-minute bar with pretend money at live prices, and one MAESTRO version also placing
orders on an OANDA practice account. No real money is involved.

- `snapshot.json` is the latest snapshot; the website's live page reads it.
- `history/<date>.json` is each night's snapshot, written once and never changed.

A snapshot holds returns, pips, counts and differences, never OANDA's prices. It is
built by `live/snapshot.py` in [Fran6jy/maestro](https://github.com/Fran6jy/maestro),
which documents every field, and pushed from the trial's server just after midnight UTC.
What the backtest expected before the trial started is in that repository's
`web/src/data/live_expectations.json`.
