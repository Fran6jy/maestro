"""
maestro/data/pipeline/store.py
==============================
The raw data store, its daily ingestion, and the feature rebuild.

Raw data lives in its own private git repository (maestro-data), because OANDA
prices must not be redistributed through the public code repo. A scheduled
GitHub Actions workflow there runs `fetch` every day after the New York close,
so ingestion never depends on this laptop. Layout of the raw store:

    candles/EUR_USD/2026/2026-08.parquet      one file per pair per finished month, and one
    candles/EUR_USD/2026/2026-09-29.parquet   per day of the month in progress: mid OHLC, bid/ask
    candles/GBP_USD/...                       close, tick volume (complete candles only)
    fred/2026.parquet                         FRED observations by year, each with the time
                                              it became public (maestro.data.features.macro)
    status.json                               what the last fetch did and whether its checks passed

Past months never change; when a month ends its day files are folded into one
month file, so the repository grows by about 15 MB a year. Features are never stored
there: `sync` pulls the raw store and rebuilds EUR_USD_features.parquet in full,
so rolling indicators never have a seam where old and new data meet.

    python -m maestro.data.pipeline.store fetch               # CI: append new candles + FRED, check
    python -m maestro.data.pipeline.store fetch --start 2005-01-01   # an empty store
    python -m maestro.data.pipeline.store sync                # laptop: git pull, rebuild features
    python -m maestro.data.pipeline.store status

The sealed holdout (maestro.data.holdout) is ingested like everything else; the
research loaders are what keep it out of sight. Triple-barrier labels are no
longer stored: no model or evaluator reads them.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from maestro.backtesting.baselines import DATA_DIR

logger = logging.getLogger(__name__)

RAW = Path(os.environ.get("MAESTRO_RAW_DIR", str(DATA_DIR / "raw")))
INSTRUMENTS = ["EUR_USD", "GBP_USD"]
TARGET = "EUR_USD"
GRANULARITY = "M5"
BATCH = 5000                 # OANDA's per-request candle limit
PIP = 1e-4
OANDA_URL = os.environ.get("OANDA_BASE_URL", "https://api-fxpractice.oanda.com")


def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, compression="snappy")
    tmp.replace(path)


# ── Candles ──────────────────────────────────────────────────────────────────
def candle_dir(instrument: str) -> Path:
    return RAW / "candles" / instrument


def _files(instrument: str) -> list[Path]:
    return sorted(candle_dir(instrument).glob("*/*.parquet"))


def _merge(parts: list[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(parts)
    return df[~df.index.duplicated(keep="last")].sort_index()


def load_candles(instrument: str) -> pd.DataFrame:
    files = _files(instrument)
    return _merge([pd.read_parquet(f) for f in files]) if files else pd.DataFrame()


def last_candle(instrument: str) -> pd.Timestamp | None:
    # A month file sorts after its own day files, so look at the last two.
    files = _files(instrument)[-2:]
    return max(pd.read_parquet(f).index.max() for f in files) if files else None


def _open_month(now: pd.Timestamp) -> pd.Timestamp:
    return now.tz_convert("UTC").normalize().replace(day=1)


def write_candles(instrument: str, df: pd.DataFrame, now: pd.Timestamp | None = None) -> int:
    """Merge candles into their files: one per finished month, one per day of the
    month still in progress. Returns the number of files written."""
    open_month = _open_month(now or pd.Timestamp.now(tz="UTC"))
    in_open = df.index >= open_month
    written = 0
    closed, current = df[~in_open], df[in_open]
    groups = list(closed.groupby(closed.index.strftime("%Y-%m")))
    groups += list(current.groupby(current.index.strftime("%Y-%m-%d")))
    for name, part in groups:
        path = candle_dir(instrument) / name[:4] / f"{name}.parquet"
        if path.exists():
            part = _merge([pd.read_parquet(path), part])
        _atomic_parquet(part, path)
        written += 1
    return written


def compact(instrument: str, now: pd.Timestamp | None = None) -> int:
    """Fold the day files of every finished month into that month's file."""
    open_month = _open_month(now or pd.Timestamp.now(tz="UTC"))
    days = [f for f in _files(instrument) if len(f.stem) == 10 and pd.Timestamp(f.stem, tz="UTC") < open_month]
    for month in sorted({f.stem[:7] for f in days}):
        members = [f for f in days if f.stem.startswith(month)]
        path = candle_dir(instrument) / month[:4] / f"{month}.parquet"
        parts = ([pd.read_parquet(path)] if path.exists() else []) + [pd.read_parquet(f) for f in members]
        _atomic_parquet(_merge(parts), path)
        for f in members:
            f.unlink()
        logger.info("%s: folded %d day files into %s", instrument, len(members), path.name)
    return len(days)


def _parse(candles: list[dict]) -> pd.DataFrame:
    idx = pd.to_datetime([c["time"] for c in candles], utc=True)
    f = lambda side, k: [float(c[side][k]) for c in candles]
    return pd.DataFrame({
        "open": f("mid", "o"), "high": f("mid", "h"), "low": f("mid", "l"), "close": f("mid", "c"),
        "bid_close": f("bid", "c"), "ask_close": f("ask", "c"),
        "volume": [int(c["volume"]) for c in candles],
    }, index=idx).rename_axis("time")


def _oanda(session: requests.Session, instrument: str, params: dict) -> dict:
    url = f"{OANDA_URL}/v3/instruments/{instrument}/candles"
    for attempt in range(5):
        try:
            resp = session.get(url, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt == 4:
                raise RuntimeError(f"OANDA {instrument}: {exc}") from exc
            time.sleep(2 ** attempt)


def update_candles(instrument: str, start: str = "2005-01-01") -> dict:
    """Append every complete candle after the last one stored (or after `start`)."""
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {os.environ['OANDA_API_KEY']}"
    last = last_candle(instrument)
    cursor = last if last is not None else pd.Timestamp(start, tz="UTC") - pd.Timedelta(seconds=1)
    added, days = 0, 0
    while True:
        data = _oanda(session, instrument, {
            "granularity": GRANULARITY, "price": "MBA", "count": BATCH, "includeFirst": "false",
            "from": cursor.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        })
        got = data.get("candles", [])
        complete = [c for c in got if c.get("complete")]
        if complete:
            batch = _parse(complete)
            days += write_candles(instrument, batch)
            added += len(batch)
            cursor = batch.index[-1]
            if added % (BATCH * 40) < len(batch):
                logger.info("  %s: +%s candles, up to %s", instrument, f"{added:,}", cursor)
        if not complete or len(complete) < len(got):         # nothing newer, or reached the forming candle
            break
        time.sleep(0.05)
    folded = compact(instrument)
    logger.info("%s: +%s candles in %d files, last %s", instrument, f"{added:,}", days, cursor)
    return {"added": added, "last": str(cursor), "day_files_folded": folded}


# ── FRED ─────────────────────────────────────────────────────────────────────
def load_fred() -> pd.DataFrame:
    return pd.concat(pd.read_parquet(f) for f in sorted((RAW / "fred").glob("*.parquet")))


def update_fred() -> dict:
    """Refetch every series (small); rewrite a year's file only if it changed."""
    from maestro.data.features.macro import fetch_fred
    raw = fetch_fred()
    for year, part in raw.groupby(raw["date"].dt.year):
        part = part.sort_values(["series", "date"]).reset_index(drop=True)
        path = RAW / "fred" / f"{year}.parquet"
        if path.exists() and pd.read_parquet(path).equals(part):
            continue
        _atomic_parquet(part, path)
    return {s: str(g["date"].max().date()) for s, g in raw.groupby("series")}


# ── Checks ───────────────────────────────────────────────────────────────────
def market_closed(now: pd.Timestamp) -> bool:
    """FX closes Friday ~21:00 UTC to Sunday ~21:00 UTC, and on 25 December and 1 January."""
    wd, h = now.weekday(), now.hour
    return (wd == 5 or (wd == 6 and h < 22) or (wd == 4 and h >= 21)
            or (now.month, now.day) in {(12, 25), (1, 1)})


def raw_checks(instrument: str, now: pd.Timestamp) -> dict:
    """Checks on the last ~two months of candles (the part a fetch can have changed)."""
    files = [f for f in _files(instrument) if f.stem[:7] >= f"{now - pd.DateOffset(months=1):%Y-%m}"]
    raw = pd.concat(pd.read_parquet(f) for f in files).sort_index()
    ix = raw.index
    return {
        "monotonic_unique": bool(ix.is_monotonic_increasing and not ix.duplicated().any()),
        "ohlc_consistent": bool(((raw["high"] >= raw[["open", "close"]].max(axis=1) - 1e-9)
                                 & (raw["low"] <= raw[["open", "close"]].min(axis=1) + 1e-9)).all()),
        "spread_positive": bool((raw["ask_close"] >= raw["bid_close"]).all()),
        "fresh": bool(market_closed(now) or (now - ix[-1]) < pd.Timedelta(hours=3)),
    }


# ── Features ─────────────────────────────────────────────────────────────────
def build_features(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rebuild the full feature table for TARGET from the raw store.

    raw : optional replacement candles for TARGET (the power test passes prices
          with a planted edge); the other pair and FRED always come from the store.
    """
    from maestro.data.features.engineer import FeatureEngineer, add_cross_pair_features
    from maestro.data.features.macro import macro_frame

    fe = FeatureEngineer()
    ohlcv = ["open", "high", "low", "close", "volume"]
    raw = load_candles(TARGET) if raw is None else raw
    feats = fe.transform(raw[ohlcv], drop_nan=False)
    feats["bid_close"], feats["ask_close"] = raw["bid_close"], raw["ask_close"]
    feats["spread_pips"] = (raw["ask_close"] - raw["bid_close"]) / PIP

    feats = pd.concat([feats, macro_frame(load_fred(), feats.index)], axis=1)

    other = next(i for i in INSTRUMENTS if i != TARGET)
    other_feats = fe.transform(load_candles(other)[ohlcv], drop_nan=False)
    feats, _ = add_cross_pair_features(feats, other_feats)
    return feats


# ── Commands ─────────────────────────────────────────────────────────────────
def fetch(start: str) -> dict:
    now = pd.Timestamp.now(tz="UTC")
    status = {"fetched_at": now.isoformat(timespec="seconds")}
    status["candles"] = {i: update_candles(i, start) for i in INSTRUMENTS}
    status["fred"] = update_fred()
    status["checks"] = {i: raw_checks(i, now) for i in INSTRUMENTS}
    status["ok"] = all(all(c.values()) for c in status["checks"].values())
    (RAW / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    return status


def sync() -> dict:
    """Pull the raw store (if it is a git clone) and rebuild the feature table."""
    if (RAW / ".git").exists():
        subprocess.run(["git", "-C", str(RAW), "pull", "--ff-only", "--quiet"], check=True)
    feats = build_features()
    path = DATA_DIR / f"{TARGET}_features.parquet"
    legacy = DATA_DIR / f"{TARGET}_features_legacy_2022_2026.parquet"
    if path.exists() and not legacy.exists():
        path.replace(legacy)        # keep the file the published Chapter 5-7 numbers came from
    _atomic_parquet(feats, path)
    status = {"rows": len(feats), "cols": feats.shape[1],
              "first": str(feats.index[0]), "last": str(feats.index[-1]),
              "ok": bool(feats["VIXCLS"].iloc[-2000:].notna().any())}
    logger.info("Features rebuilt: %s rows x %d cols, %s -> %s", f"{status['rows']:,}", status["cols"],
                status["first"], status["last"])
    return status


def main() -> None:
    log_dir = DATA_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_dir / f"ingest_{datetime.now(timezone.utc):%Y%m}.log", encoding="utf-8")],
    )
    p = argparse.ArgumentParser(description="MAESTRO raw store, ingestion and feature rebuild")
    p.add_argument("mode", choices=["fetch", "sync", "status"])
    p.add_argument("--start", default="2005-01-01", help="first date for an empty store")
    args = p.parse_args()
    if args.mode == "status":
        print((RAW / "status.json").read_text())
        return
    status = fetch(args.start) if args.mode == "fetch" else sync()
    logger.log(logging.INFO if status["ok"] else logging.ERROR, "%s %s", args.mode,
               "OK" if status["ok"] else f"FAILED CHECKS: {status.get('checks')}")
    raise SystemExit(0 if status["ok"] else 1)


if __name__ == "__main__":
    main()
