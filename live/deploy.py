"""
maestro/live/deploy.py
======================
Train and package MAESTRO for the live trial (run where there is a GPU).

Trains on the latest 12 months, ending EMBARGO_DAYS before the newest bar exactly
as a backtest block's training ends before its first test month, with the frozen
design's settings. It pins the regime filter's start (see LiveModel.start_filter)
and saves forecasts for the last 1,440 bars, so the top-10% cut-off starts from
the history a backtest would give it. The live model must know the most recent
market, so this unlocks the sealed holdout: run it only after the holdout
confirmation of the frozen design.

    python -m maestro.live.deploy --out C:\\tmp\\maestro_live\\models\\2026-10-05
"""
from __future__ import annotations

import argparse
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from maestro.backtesting.maestro_runner import EPOCHS, load_features, predict_bars, train_agents
from maestro.data.holdout import unlock
from maestro.live.predictor import LiveModel

logger = logging.getLogger(__name__)

EMBARGO_DAYS = 5          # as data/validation/wfa.py leaves between training and testing
SEED_BARS = 1440          # the top-10% cut-off's lookback


def deploy(out: Path, train_months: int = 12, epochs: int = EPOCHS, fast: bool = True) -> LiveModel:
    unlock()
    df = load_features("EUR_USD")
    newest = df.index[-1]
    train_end = newest - pd.Timedelta(days=EMBARGO_DAYS)
    train_idx = df.index[(df.index > train_end - pd.DateOffset(months=train_months)) & (df.index <= train_end)]
    logger.info("training on %s -> %s (%d bars)", train_idx[0], train_idx[-1], len(train_idx))
    regime, signal, meta = train_agents(df, train_idx, "EUR_USD", epochs, fast)

    commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    meta.update({"train_start": str(train_idx[0]), "train_end": str(train_idx[-1]),
                 "data_end": str(newest), "commit": commit, "fast": fast, "epochs": epochs,
                 "deployed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    model = LiveModel(regime, signal, meta)
    model.start_filter(newest, df)
    out = Path(out)
    model.save(out)
    seed = predict_bars(regime, signal, df, df.index[-SEED_BARS:])
    seed.to_parquet(out / "seed_forecasts.parquet")
    logger.info("saved %s (seeded %d forecasts)", out, len(seed))
    return model


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description="Train and package MAESTRO for the live trial")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--train-months", type=int, default=12)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    args = p.parse_args()
    deploy(args.out, args.train_months, args.epochs)


if __name__ == "__main__":
    main()
