"""
maestro/live/loop.py
====================
The live trial's 5-minute loop: OANDA practice account only.

After each 5-minute bar closes:
  1. append new candles to the store in MAESTRO_RAW_DIR (FRED once a day)
  2. skip the bar if the journal already has it, so a restart never repeats a bar
  3. build features from the last 30 days and forecast the latest bar (live/predictor)
  4. decide every strategy with the backtest's own rules (live/strategies)
  5. paper-fill every strategy at OANDA's live quote (live/ledger)
  6. for ORDER_STRATEGY only, move the practice account to its target with a
     market order tagged with the bar time, so the same bar is never ordered twice
  7. write status.json (a heartbeat anyone can check)

    python -m maestro.live.loop --models /state/models/current --orders maestro_top10
    python -m maestro.live.loop --models ... --once          # one cycle, then exit
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from maestro.data.pipeline import store
from maestro.live.ledger import Journal
from maestro.live.oanda import PracticeClient
from maestro.live.predictor import LiveModel
from maestro.live.strategies import BASELINES, MAESTRO_VARIANTS, baseline_targets, maestro_targets

logger = logging.getLogger(__name__)

INSTRUMENT, OTHER = "EUR_USD", "GBP_USD"
FEATURE_DAYS = 30        # live features from a 30-day window match full history (see store.features_from)
HISTORY_MONTHS = 13      # candles kept for the baselines' 12-month training window
SETTLE_SECONDS = 10      # wait after a bar closes so OANDA marks it complete


def next_bar_close(now: datetime) -> datetime:
    t = pd.Timestamp(now).floor("5min") + pd.Timedelta(minutes=5)
    return (t + pd.Timedelta(seconds=SETTLE_SECONDS)).to_pydatetime()


class LiveTrial:
    def __init__(self, models: Path, state: Path, order_strategy: str | None, order_units: int) -> None:
        self.client = PracticeClient()
        self.state = Path(state)
        self.journal = Journal(self.state / "journal.db")
        self.model = LiveModel.load(models)
        self.train_window = (pd.Timestamp(self.model.meta["train_start"]), pd.Timestamp(self.model.meta["train_end"]))
        if order_strategy and order_strategy not in MAESTRO_VARIANTS + BASELINES:
            raise ValueError(f"unknown order strategy {order_strategy}")
        self.order_strategy, self.order_units = order_strategy, order_units
        self.fred_day = None
        self.carry: dict[str, tuple[float, int]] = {}
        seed = Path(models) / "seed_forecasts.parquet"
        if seed.exists() and self.journal.confidence_history(1).empty:
            # Start the top-10% cut-off from the history a backtest would give it.
            for bar, row in pd.read_parquet(seed).iterrows():
                self.journal.record_forecast(bar, row.to_dict())
            self.journal.commit()

    # ── One cycle ────────────────────────────────────────────────────────────
    def refresh_data(self) -> None:
        start = (pd.Timestamp.now(tz="UTC") - pd.DateOffset(months=HISTORY_MONTHS)).strftime("%Y-%m-%d")
        for inst in (INSTRUMENT, OTHER):
            store.update_candles(inst, start)
        today = pd.Timestamp.now(tz="UTC").date()
        if self.fred_day != today or not (store.RAW / "fred").exists():
            store.update_fred()
            self.fred_day = today

    def cycle(self) -> dict | None:
        self.refresh_data()
        eur, gbp = store.load_candles(INSTRUMENT), store.load_candles(OTHER)
        bar = eur.index[-1]
        if self.journal.seen(bar):
            return None
        since = bar - pd.Timedelta(days=FEATURE_DAYS)
        feats = store.features_from(eur.loc[since:], gbp.loc[since:], store.load_fred())

        targets: dict[str, float] = {}
        forecast = None
        if feats.loc[bar].notna().all():
            forecast = self.model.forecast_latest(feats)
            history = self.journal.confidence_history(1440)
            targets.update(maestro_targets(forecast, history, float(eur["close"].iloc[-1]), INSTRUMENT))
            self.journal.record_forecast(bar, forecast.to_dict())
            self.carry = {k: (v, 0) for k, v in targets.items()}
        else:
            # The backtest holds the last decision through at most an hour of unscorable bars.
            self.journal.record_forecast(bar, {"unscorable": True})
            for name in MAESTRO_VARIANTS:
                value, age = self.carry.get(name, (0.0, 0))
                targets[name] = value if age < 12 else 0.0
                self.carry[name] = (value, age + 1)
        close = eur["close"]
        train_idx = close.index[(close.index >= self.train_window[0]) & (close.index <= self.train_window[1])]
        targets.update(baseline_targets(close, train_idx))

        _, bid, ask = self.client.quote(INSTRUMENT)
        mid = float(eur["close"].iloc[-1])
        for name, target in targets.items():
            self.journal.step(bar, name, target, mid, bid, ask)
        if self.order_strategy:
            self.place_order(bar, targets[self.order_strategy])
        self.journal.commit()
        return {"bar": str(bar), "forecast": None if forecast is None else forecast.to_dict(), "targets": targets}

    def place_order(self, bar: pd.Timestamp, target: float) -> None:
        tag = f"{bar:%Y%m%d%H%M}-{self.order_strategy}"
        if self.journal.db.execute("SELECT 1 FROM orders WHERE tag = ?", (tag,)).fetchone():
            return                                            # this bar was already ordered
        units = int(round(target * self.order_units)) - self.client.net_units(INSTRUMENT)
        if units == 0:
            return
        try:
            fill = self.client.market_order(INSTRUMENT, units, tag)
            self.journal.record_order(tag, bar, self.order_strategy, units, "filled", fill)
        except Exception as exc:  # noqa: BLE001 — recorded; the next bar re-aims at the target
            logger.error("order %s failed: %s", tag, exc)
            self.journal.record_order(tag, bar, self.order_strategy, units, "failed", error=str(exc)[:300])

    def write_status(self, result: dict | None, error: str | None = None) -> None:
        status = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "error": error,
                  "last_cycle": result, "model": self.model.meta.get("deployed_at"),
                  "order_strategy": self.order_strategy, "order_units": self.order_units}
        (self.state / "status.json").write_text(json.dumps(status, indent=2, default=str))

    def run(self, once: bool = False) -> None:
        while True:
            if not once:
                wait = (next_bar_close(datetime.now(timezone.utc)) - datetime.now(timezone.utc)).total_seconds()
                time.sleep(max(0.0, wait))
            if store.market_closed(pd.Timestamp.now(tz="UTC")) and not once:
                self.write_status(None, "market closed")
                continue
            try:
                result = self.cycle()
                if result:
                    logger.info("bar %s | %s", result["bar"],
                                " ".join(f"{k}={v:+.0f}" for k, v in result["targets"].items()))
                self.write_status(result)
            except Exception as exc:  # noqa: BLE001 — keep the loop alive; the error is in status.json
                logger.exception("cycle failed")
                self.write_status(None, f"{type(exc).__name__}: {exc}"[:500])
            if once:
                return


def main() -> None:
    p = argparse.ArgumentParser(description="MAESTRO live trial (OANDA practice only)")
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--state", type=Path, default=Path("/state"))
    p.add_argument("--orders", default=None, help="strategy whose target is sent to the practice account")
    p.add_argument("--units", type=int, default=10_000, help="practice units per unit of position")
    p.add_argument("--once", action="store_true")
    args = p.parse_args()
    args.state.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(args.state / "live.log", encoding="utf-8")])
    LiveTrial(args.models, args.state, args.orders, args.units).run(once=args.once)


if __name__ == "__main__":
    main()
