"""
maestro/backtesting/maestro_runner.py
=====================================
Chapter 7: MAESTRO on the shared evaluator.

Trains MAESTRO's regime agent (HMM + Transformer) and signal agent (TFT +
PatchTST) on exactly the same retraining plan as the MSc baselines
(`baselines.refit_plan`), on the local GPU, then scores the resulting positions
with exactly the same `simulate` / `summarise` code as every other strategy.

Three versions are scored, because how MAESTRO turns forecasts into positions
is a genuine design choice. All three were fixed before any full-run result
was seen:
  maestro_gated    trades only when its confidence clears the per-regime
                   threshold in signal_agent.CONFIDENCE_THRESHOLDS (as designed)
  maestro_top10    trades only its 10% most confident calls, the cut-off taken
                   from the previous five trading days (causal, adapts to each
                   retrained model's own confidence scale)
  maestro_ungated  trades every non-flat signal, ignoring confidence
                   (the like-for-like comparison with the MSc models)

Resumable: each retraining block's signals are written to disk the moment the
block finishes, and completed blocks are skipped on the next run. A sleep,
crash or Ctrl+C loses at most the block in progress.

    python -m maestro.backtesting.maestro_runner --smoke                          # 1 block, 1 epoch
    python -m maestro.backtesting.maestro_runner --refit-months 3 --train-months 12
    python -m maestro.backtesting.maestro_runner --refit-months 3 --train-months 12 --score-only
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from maestro.backtesting.baselines import (
    OUTPUT_DIR, design_tag, feature_file, load_close, refit_plan, run as score_strategies,
)

logger = logging.getLogger(__name__)

VAL_MONTHS = 2          # last part of each training window used for early stopping
HORIZON = 6             # primary forecast horizon in bars (30 minutes)
EPOCHS = 40             # same settings as the Modal edge run
WARMUP_BARS = 400       # history fed before each test block so sequence models have context
# --fast: the TFT is the bottleneck (LSTMs over 120-bar windows). A 4x batch cuts its
# time per epoch ~3.7x; with the learning rate unchanged its validation loss stays within
# ~0.6% of the standard settings (a 2e-3 rate, or bf16, made it clearly worse, and bf16
# was also slower on this GPU). Epochs are cheap, so early stopping gets more patience.
FAST_TFT = {"batch_size": 256, "patience": 10}


def load_features(instrument: str, granularity: str = "M5") -> pd.DataFrame:
    """Full feature parquet, with the sealed holdout dropped unless unlocked."""
    from maestro.data.holdout import seal
    return seal(pd.read_parquet(feature_file(instrument, granularity)).sort_index())


def train_and_predict(df: pd.DataFrame, train_idx: pd.Index, test_idx: pd.Index,
                      instrument: str, epochs: int, fast: bool = False,
                      granularity: str = "M5") -> tuple[pd.DataFrame, dict]:
    """Fit regime + signal agents on train_idx, forecast every bar of test_idx."""
    from maestro.agents.regime.regime_classifier import RegimeDetectionAgent
    from maestro.agents.regime.transformer_regime import TransformerConfig
    from maestro.agents.signal.patchtst import PatchTSTConfig
    from maestro.agents.signal.signal_agent import SignalAgent
    from maestro.agents.signal.tft_model import HORIZONS, TFTConfig

    train_df = df.loc[train_idx]
    val_cut = train_df.index[-1] - pd.DateOffset(months=VAL_MONTHS)
    val_mask = train_df.index >= val_cut
    val_df = train_df[val_mask].dropna() if val_mask.sum() > 200 else None
    fit_df = (train_df[~val_mask] if val_df is not None else train_df).dropna()
    if len(fit_df) < MIN_FIT_BARS.get(granularity, 1000):
        raise ValueError(f"only {len(fit_df)} usable training bars after dropping missing features")

    meta = {"fit_bars": len(fit_df), "val_bars": 0 if val_df is None else len(val_df)}

    regime = RegimeDetectionAgent(transformer_config=TransformerConfig(max_epochs=min(50, max(1, epochs))))
    try:
        regime.fit(fit_df, val_df=val_df)
        meta["regime_model"] = "hmm+transformer"
    except Exception as exc:  # noqa: BLE001 — record and fall back, as the backtest engine does
        logger.warning("Regime transformer failed (%s); falling back to HMM only", exc)
        regime = RegimeDetectionAgent(use_transformer=False)
        regime.fit(fit_df)
        meta["regime_model"] = "hmm"

    signal = SignalAgent(
        instrument=instrument, primary_horizon=HORIZON,
        tft_config=TFTConfig(seq_len=120, pred_len=max(HORIZONS), max_epochs=epochs,
                             **{"patience": 6, **(FAST_TFT if fast else {})}),
        ptst_config=PatchTSTConfig(seq_len=128, max_epochs=epochs, patience=6),
    )
    t0 = time.time()
    signal.fit(fit_df, regime.predict_batch(fit_df)["regime"], val_df=val_df,
               val_regimes=regime.predict_batch(val_df)["regime"] if val_df is not None else None)
    meta.update({"fast": fast, "signal_fit_minutes": round((time.time() - t0) / 60, 2),
                 "tft_best_val_loss": getattr(signal.tft, "best_loss", None),
                 "tft_epochs": getattr(signal.tft, "epochs_run", None)})

    # Forecast the test block with a run-up of earlier (already known) bars for context.
    start = df.index.get_indexer([test_idx[0]])[0]
    infer = df.iloc[max(0, start - WARMUP_BARS): df.index.get_indexer([test_idx[-1]])[0] + 1].dropna()
    regimes = regime.predict_batch(infer)["regime"]
    out = signal.predict_batch(infer, regimes)
    cols = ["signal", "confidence", "regime", "model_agree", "tft_signal", "ptst_signal", "pred_p50"]
    out = out[[c for c in cols if c in out.columns]]
    out = out[out.index.isin(test_idx)]
    meta["signal_bars"] = len(out)
    return out, meta


TOP_SHARE = 0.10        # maestro_top10: share of calls confident enough to trade
BARS_PER_DAY = {"M5": 288, "H1": 24, "H4": 6, "D": 1}
MIN_FIT_BARS = {"M5": 1000, "H1": 1000, "H4": 600, "D": 300}   # below this a block is not trained
# Long bars start the design once a training window holds enough bars (the regime model's
# HMM alone needs 500); shorter bars start at the first walk-forward month, as before.
MIN_TRAIN_BARS = {"M5": 0, "H1": 0, "H4": 1200, "D": 650}


def position_windows(granularity: str = "M5") -> tuple[int, int, int]:
    """(cut-off lookback, its minimum history, max bars to carry a decision) for a bar length.
    On 5-minute bars: five trading days, one day and one hour, as fixed for Chapter 7.
    Longer bars keep the same spans in time, with a floor so a cut-off never rests on a
    handful of values."""
    per_day = BARS_PER_DAY[granularity]
    return max(5 * per_day, 60), max(per_day, 20), max(1, per_day // 24)


def positions(signals: pd.DataFrame, bars: pd.Index, granularity: str = "M5") -> dict[str, pd.Series]:
    """Turn saved forecasts into the two position series that get scored.

    About 0.3% of bars have a missing feature (candle shape is 0/0 when
    high == low), so the models cannot score them. Going flat on those bars
    would charge MAESTRO an extra round trip for a data artefact, so the
    previous decision is carried forward instead, which stays causal.
    """
    from maestro.agents.signal.signal_agent import CONFIDENCE_THRESHOLDS
    sig = signals["signal"].astype(float)
    threshold = signals["regime"].astype(int).map(CONFIDENCE_THRESHOLDS).fillna(0.55)
    gated = sig.where(signals["confidence"] >= threshold, 0.0)
    # Cut-off from strictly earlier bars only (shift(1)), so no bar sees its own confidence rank.
    lookback, min_history, max_carry = position_windows(granularity)
    cutoff = (signals["confidence"].rolling(lookback, min_periods=min_history)
              .quantile(1 - TOP_SHARE).shift(1))
    top = sig.where(signals["confidence"] >= cutoff, 0.0)
    bars = bars[(bars >= signals.index[0]) & (bars <= signals.index[-1])]
    carry = lambda s: s.reindex(bars).ffill(limit=max_carry).fillna(0.0)
    return {"maestro_gated": carry(gated), "maestro_top10": carry(top), "maestro_ungated": carry(sig)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description="Chapter 7: MAESTRO on the shared evaluator")
    p.add_argument("--instrument", default="EUR_USD")
    p.add_argument("--granularity", default="M5", choices=list(BARS_PER_DAY),
                   help="bar length (features from: store sync --granularity ...)")
    p.add_argument("--refit-months", type=int, default=3)
    p.add_argument("--train-months", type=int, default=12)
    p.add_argument("--expanding", action="store_true", help="train on all history instead of a rolling window")
    p.add_argument("--epochs", type=int, default=None, help=f"max training epochs (default {EPOCHS}, smoke 1)")
    p.add_argument("--smoke", action="store_true", help="first block only, 1 epoch unless --epochs is given")
    p.add_argument("--score-only", action="store_true", help="skip training; score the blocks already saved")
    p.add_argument("--holdout", action="store_true",
                   help="unlock the sealed holdout (maestro.data.holdout): final confirmation run only")
    p.add_argument("--fast", action="store_true", help=f"faster TFT training: {FAST_TFT}")
    p.add_argument("--shard", default=None, metavar="I/N",
                   help="train only blocks with block_id %% N == I (split work across GPUs or machines)")
    p.add_argument("--reverse", action="store_true", help="train the latest blocks first")
    p.add_argument("--skip", default="", metavar="IDS",
                   help="block ids already trained elsewhere, e.g. 0-40,52 (a cloud session resuming)")
    p.add_argument("--limit", type=int, default=None,
                   help="train at most this many new blocks, then exit (one process per block)")
    p.add_argument("--max-hours", type=float, default=None,
                   help="stop starting new blocks when a block might not finish in this many hours")
    args = p.parse_args()
    if args.holdout:
        from maestro.data.holdout import unlock
        unlock()

    train_months = None if args.expanding else args.train_months
    epochs = args.epochs or (1 if args.smoke else EPOCHS)
    prefix = "" if args.granularity == "M5" else f"{args.granularity}_"
    tag = prefix + design_tag(args.refit_months, train_months) + ("_fast" if args.fast else "")
    out_dir = OUTPUT_DIR / "maestro" / (tag + ("_smoke" if args.smoke else ""))

    pooled = run_design(load_close(args.instrument, args.granularity), out_dir, args.instrument,
                        lambda: load_features(args.instrument, args.granularity),
                        args.refit_months, train_months,
                        epochs, max_blocks=1 if args.smoke else None, train=not args.score_only,
                        fast=args.fast, shard=tuple(map(int, args.shard.split("/"))) if args.shard else None,
                        reverse=args.reverse, skip=parse_ids(args.skip), limit=args.limit,
                        deadline=time.time() + args.max_hours * 3600 if args.max_hours else None,
                        granularity=args.granularity)
    if pooled is not None:
        print_summary(pooled)


def parse_ids(text: str) -> set[int]:
    """'0-3,7' -> {0, 1, 2, 3, 7}"""
    ids: set[int] = set()
    for part in filter(None, text.split(",")):
        lo, _, hi = part.partition("-")
        ids.update(range(int(lo), int(hi or lo) + 1))
    return ids


def format_ids(ids) -> str:
    """{0, 1, 2, 3, 7} -> '0-3,7' (the inverse of parse_ids)"""
    out, run = [], []
    for i in sorted(ids):
        if run and i != run[-1] + 1:
            out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
            run = []
        run.append(i)
    if run:
        out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
    return ",".join(out)


def run_design(close: pd.Series, out_dir: Path, instrument: str, features, refit_months: int = 3,
               train_months: int | None = 12, epochs: int = EPOCHS, max_blocks: int | None = None,
               train: bool = True, fast: bool = False, shard: tuple[int, int] | None = None,
               reverse: bool = False, deadline: float | None = None,
               skip: set[int] | None = None, limit: int | None = None,
               granularity: str = "M5") -> pd.DataFrame | None:
    """Train MAESTRO block by block (resumable), then score it with every baseline.

    features : callable returning the feature table, only called if a block needs training.
    shard    : (i, n) trains only blocks with block_id % n == i, so several GPUs or machines
               can share one design; blocks are independent, so the results are identical.
    skip     : block ids trained elsewhere; not trained here (their files are merged later)
    limit    : train at most this many new blocks in this call
    deadline : epoch seconds after which no new block is started if it might not finish
               (a cloud session's time limit); finished blocks are kept either way.
    Returns the pooled scores, or None if some blocks are still missing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    blocks = refit_plan(close, refit_months, train_months, min_train_bars=MIN_TRAIN_BARS[granularity])
    full = len(blocks)
    if max_blocks:
        blocks = blocks[:max_blocks]
    logger.info("Design %s: %d retraining blocks, %d test months, output %s",
                design_tag(refit_months, train_months), len(blocks),
                sum(len(b.splits) for b in blocks), out_dir)

    todo = [b for b in blocks if not (out_dir / f"block_{b.block_id:02d}.parquet").exists()]
    if train and todo:
        import torch
        logger.info("Device: %s", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
        df = features()
        mine = [b for b in blocks if (shard is None or b.block_id % shard[1] == shard[0])
                and b.block_id not in (skip or set())]
        longest = 20 * 60.0                                  # assume 20 min until one has been timed
        trained = 0
        for b in (reversed(mine) if reverse else mine):
            path = out_dir / f"block_{b.block_id:02d}.parquet"
            if path.exists():
                continue
            if limit is not None and trained >= limit:
                break
            if deadline and time.time() + 1.3 * longest > deadline:
                logger.info("Stopping before block %d: it might not finish before the deadline", b.block_id)
                break
            t0 = time.time()
            logger.info("block %d/%d: training on %d bars (%s -> %s), testing %s -> %s",
                        b.block_id + 1, len(blocks), len(b.train_idx), b.train_idx[0].date(),
                        b.train_idx[-1].date(), b.splits[0].test_start.date(), b.splits[-1].test_end.date())
            sig, meta = train_and_predict(df, b.train_idx, b.test_idx, instrument, epochs, fast, granularity)
            meta.update({"block_id": b.block_id, "minutes": round((time.time() - t0) / 60, 1)})
            tmp = path.with_suffix(".tmp")
            sig.to_parquet(tmp)
            tmp.replace(path)                                  # atomic: never a half-written block
            (out_dir / f"block_{b.block_id:02d}.json").write_text(json.dumps(meta, indent=2))
            longest = max(longest, time.time() - t0)
            trained += 1
            logger.info("block %d done in %.1f min (%d signals)", b.block_id, meta["minutes"], len(sig))

    files = [out_dir / f"block_{b.block_id:02d}.parquet" for b in blocks]
    missing = [f for f in files if not f.exists()]
    if missing:
        logger.warning("Only %d of %d blocks saved; scoring requires all. Re-run to resume.",
                       len(files) - len(missing), len(files))
        return None
    signals = pd.concat(pd.read_parquet(f) for f in files).sort_index()
    pos = positions(signals, close.index, granularity)
    pooled, _ = score_strategies(instrument, strategies=None,
                                 max_splits=blocks[-1].splits[-1].split_id + 1 if len(blocks) < full else None,
                                 refit_months=refit_months, train_months=train_months,
                                 external=pos, out_dir=out_dir / "scores", close=close,
                                 min_train_bars=MIN_TRAIN_BARS[granularity])
    return pooled


def print_summary(pooled: pd.DataFrame) -> None:
    view = pooled[pooled["cost"] == "spread"].set_index("strategy")
    gross = pooled[pooled["cost"] == "gross"].set_index("strategy")
    print(f"\n{'strategy':<17}{'trades':>9}{'hit':>8}{'gross Sh':>10}{'net Sh':>9}{'net pips':>11}")
    for name in ["maestro_gated", "maestro_top10", "maestro_ungated", "logreg_lag5", "buy_hold"]:
        r = view.loc[name]
        print(f"{name:<17}{r['n_trades']:>9.0f}{r['hit_directional'] * 100:>7.1f}%"
              f"{gross.loc[name, 'sharpe']:>10.2f}{r['sharpe']:>9.2f}{r['net_pips_total']:>11.0f}")


if __name__ == "__main__":
    main()
