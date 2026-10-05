# MAESTRO live trial: Foundation snapshots

The public record of the MAESTRO live trial: every strategy trading EUR/USD on every
5-minute bar with pretend money at live prices, and one MAESTRO version also placing
orders on an OANDA practice account. No real money is involved.

- `snapshot.json` is the latest Dawn, Day or Dusk observation; the website reads it.
- `official.json` is the latest official Dawn close.
- `history/<date>.json` is each official daily close, written once and never changed.
- `history/foundation/<date>-<edition>.json` preserves every immutable observation.

A snapshot holds returns, pips, counts and differences, never OANDA's prices. It is
built by `live/snapshot.py` in [Fran6jy/maestro](https://github.com/Fran6jy/maestro),
which documents every field. Dawn is published at 00:20 UTC and is the only official
daily result. Day at 12:20 and Dusk at 20:20 are clearly provisional.
What the backtest expected before the trial started is in that repository's
`web/src/data/live_expectations.json`.
