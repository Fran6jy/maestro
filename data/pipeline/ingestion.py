"""
maestro/data/pipeline/ingestion.py
====================================
Master data pipeline orchestrator.

Ties together: OANDA → features → labels → WFA splits → storage.

Running this once builds the complete historical dataset
needed to train all 5 MAESTRO agents.

Usage (CLI)
-----------
    python -m maestro.data.pipeline.ingestion --mode full
    python -m maestro.data.pipeline.ingestion --mode incremental
    python -m maestro.data.pipeline.ingestion --instrument EUR_USD --granularity M5

Usage (Python)
--------------
    >>> from maestro.data.pipeline.ingestion import DataPipeline
    >>> pipeline = DataPipeline()
    >>> pipeline.run_full()
"""
from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd

from maestro.config.config import get, instrument_ids
from maestro.data.connectors.fred import FREDConnector
from maestro.data.connectors.newsapi import NewsConnector
from maestro.data.connectors.oanda import OANDAConnector
from maestro.data.features.engineer import FeatureEngineer, add_cross_pair_features
from maestro.data.features.labels import TripleBarrierLabeller, compute_sample_weights

logger = logging.getLogger(__name__)

# Load .env before resolving paths (critical on Windows).
try:
    from dotenv import load_dotenv as _load_dotenv
    _here = Path(__file__).resolve()
    for _p in [_here.parent, _here.parent.parent, _here.parent.parent.parent, Path.cwd()]:
        if (_p / ".env").exists():
            _load_dotenv(_p / ".env", override=False)
            break
except ImportError:
    pass

_default_data = str(Path.home() / "maestro_data")
DATA_DIR = Path(os.environ.get("MAESTRO_DATA_DIR", _default_data))
DATA_DIR.mkdir(parents=True, exist_ok=True)


class DataPipeline:
    """
    Orchestrates the full data build:
      1. Fetch OHLCV from OANDA (historical + incremental)
      2. Fetch macro data from FRED
      3. Fetch news from NewsAPI
      4. Feature engineering
      5. Triple-barrier labelling
      6. Sample weight computation
      7. Persist to Parquet (local) + optionally S3

    All DataFrames are saved as Parquet files with snappy compression
    for fast loading during model training.
    """

    def __init__(self) -> None:
        self.oanda   = OANDAConnector()
        self.fred    = FREDConnector()
        self.news    = NewsConnector()
        self.fe      = FeatureEngineer()
        self.labeller = TripleBarrierLabeller(
            pt_sl    = (2.0, 1.0),
            max_hold = 10,
            min_ret  = 0.0001,
        )

    # ── Full historical build ─────────────────────────────────────────────────
    def run_full(
        self,
        start:       str  = "2020-01-01",
        end:         str | None = None,
        granularity: str  = "M5",
        save:        bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Full pipeline run: fetch all instruments + macro + news,
        engineer features, generate labels, save outputs.

        Returns
        -------
        dict mapping dataset name → DataFrame:
          {
            "EUR_USD_features": ...,
            "GBP_USD_features": ...,
            "macro": ...,
            "news":  ...,
          }
        """
        logger.info("=" * 60)
        logger.info("MAESTRO Data Pipeline — Full Build")
        logger.info("Instruments: %s | Granularity: %s | Start: %s",
                    instrument_ids(), granularity, start)
        logger.info("=" * 60)

        results: dict[str, pd.DataFrame] = {}

        # ── Step 1: OHLCV ─────────────────────────────────────────────────────
        logger.info("STEP 1: Fetching OHLCV data...")
        ohlcv_data: dict[str, pd.DataFrame] = {}
        for instr in instrument_ids():
            logger.info("  Fetching %s %s...", instr, granularity)
            try:
                df = self.oanda.fetch_historical(instr, granularity, start=start, end=end)
                ohlcv_data[instr] = df
                logger.info("  ✓ %s: %d bars", instr, len(df))
            except Exception as exc:
                logger.error("  ✗ %s failed: %s", instr, exc)

        # ── Step 2: Macro data ────────────────────────────────────────────────
        logger.info("STEP 2: Fetching macro data (FRED)...")
        macro_df = pd.DataFrame()
        try:
            macro_df = self.fred.fetch_all_series(start=start, end=end)
            logger.info("  ✓ Macro: %d rows × %d columns", len(macro_df), len(macro_df.columns))
        except Exception as exc:
            logger.error("  ✗ FRED fetch failed: %s", exc)

        # ── Step 3: News (recent only) ────────────────────────────────────────
        logger.info("STEP 3: Fetching recent news...")
        news_df = pd.DataFrame()
        try:
            news_df = self.news.fetch_recent(lookback_days=30)
            logger.info("  ✓ News: %d articles", len(news_df))
        except Exception as exc:
            logger.error("  ✗ News fetch failed: %s", exc)

        # ── Step 4: Feature engineering ───────────────────────────────────────
        logger.info("STEP 4: Engineering features...")
        feature_dfs: dict[str, pd.DataFrame] = {}
        for instr, ohlcv in ohlcv_data.items():
            if ohlcv.empty:
                continue
            logger.info("  Engineering features for %s...", instr)
            features = self.fe.transform(ohlcv, drop_nan=False)

            # Attach macro features if available
            if not macro_df.empty:
                macro_aligned = FREDConnector.align_to_ohlcv(macro_df, features.index)
                features = pd.concat([features, macro_aligned], axis=1)

            feature_dfs[instr] = features
            logger.info("  ✓ %s: %d bars × %d features", instr, len(features), len(features.columns))

        # Cross-pair features (requires both EUR/USD and GBP/USD)
        if "EUR_USD" in feature_dfs and "GBP_USD" in feature_dfs:
            logger.info("  Adding cross-pair features...")
            feature_dfs["EUR_USD"], feature_dfs["GBP_USD"] = add_cross_pair_features(
                feature_dfs["EUR_USD"], feature_dfs["GBP_USD"]
            )

        # ── Step 5: Triple-barrier labels ──────────────────────────────────────
        logger.info("STEP 5: Generating triple-barrier labels...")
        labelled_dfs: dict[str, pd.DataFrame] = {}
        for instr, features in feature_dfs.items():
            if "atr" not in features.columns:
                logger.warning("  ATR column missing for %s — skipping labelling", instr)
                labelled_dfs[instr] = features
                continue
            labelled = self.labeller.fit(features)
            labelled["sample_weight"] = compute_sample_weights(labelled, ret_col="ret")
            labelled_dfs[instr] = labelled
            logger.info("  ✓ %s: %d labelled bars", instr, len(labelled))

        results.update({f"{k}_features": v for k, v in labelled_dfs.items()})
        results["macro"] = macro_df
        results["news"]  = news_df

        # ── Step 6: Save ──────────────────────────────────────────────────────
        if save:
            logger.info("STEP 6: Saving to Parquet...")
            self._save_all(results)

        logger.info("=" * 60)
        logger.info("Pipeline complete. Datasets: %s", list(results.keys()))
        logger.info("=" * 60)
        return results

    # ── Incremental update ────────────────────────────────────────────────────
    def run_incremental(
        self,
        granularity: str = "M5",
        lookback_bars: int = 500,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch only the most recent N bars for all instruments.
        Used for daily/intraday updates in live trading mode.
        """
        logger.info("Running incremental data update (last %d bars)...", lookback_bars)
        results: dict[str, pd.DataFrame] = {}

        for instr in instrument_ids():
            try:
                # Load existing dataset
                path = DATA_DIR / f"{instr}_{granularity}_features.parquet"
                if path.exists():
                    existing = pd.read_parquet(path)
                    last_date = existing.index[-1].strftime("%Y-%m-%d")
                else:
                    existing = pd.DataFrame()
                    last_date = "2024-01-01"

                # Fetch new bars
                new_ohlcv = self.oanda.fetch_historical(
                    instr, granularity, start=last_date
                )
                if new_ohlcv.empty:
                    continue

                # Engineer features
                new_features = self.fe.transform(new_ohlcv, drop_nan=False)

                # Append to existing (deduplicate by index)
                if not existing.empty:
                    # Only take columns present in both
                    common_cols = existing.columns.intersection(new_features.columns)
                    combined = pd.concat([
                        existing[common_cols],
                        new_features[common_cols]
                    ])
                    combined = combined[~combined.index.duplicated(keep="last")]
                    combined = combined.sort_index()
                else:
                    combined = new_features

                results[instr] = combined
                logger.info("  ✓ %s: +%d new bars (total: %d)", instr, len(new_ohlcv), len(combined))

                # Save updated dataset
                self._save_df(combined, f"{instr}_{granularity}_features")

            except Exception as exc:
                logger.error("  ✗ Incremental update failed for %s: %s", instr, exc)

        return results

    # ── Loaders ───────────────────────────────────────────────────────────────
    def load(
        self,
        instrument: str,
        granularity: str = "M5",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """
        Load a pre-built feature dataset from Parquet.

        Parameters
        ----------
        instrument  : e.g. "EUR_USD"
        granularity : e.g. "M5"
        start, end  : optional date filters

        Returns
        -------
        pd.DataFrame with all features and labels
        """
        # Support both naming conventions (with/without granularity)
        path = DATA_DIR / f"{instrument}_{granularity}_features.parquet"
        if not path.exists():
            path = DATA_DIR / f"{instrument}_features.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}\n"
                f"Run DataPipeline().run_full() to build it first."
            )
        from maestro.data.holdout import seal
        df = seal(pd.read_parquet(path).sort_index())
        if start:
            df = df[df.index >= pd.Timestamp(start, tz="UTC")]
        if end:
            df = df[df.index <= pd.Timestamp(end, tz="UTC")]
        logger.info("Loaded %s %s: %d rows × %d cols", instrument, granularity, len(df), len(df.columns))
        return df

    # ── Persistence ───────────────────────────────────────────────────────────
    def _save_all(self, results: dict[str, pd.DataFrame]) -> None:
        for name, df in results.items():
            if df is not None and not df.empty:
                self._save_df(df, name)

    def _save_df(self, df: pd.DataFrame, name: str) -> None:
        path = DATA_DIR / f"{name}.parquet"
        df.to_parquet(path, compression="snappy", engine="pyarrow")
        size_mb = path.stat().st_size / 1_048_576
        logger.info("  Saved %s → %s (%.1f MB)", name, path, size_mb)

    # ── Data quality report ───────────────────────────────────────────────────
    @staticmethod
    def quality_report(df: pd.DataFrame, name: str = "dataset") -> None:
        """Print a data quality summary to the log."""
        null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        high_null = null_pct[null_pct > 5]

        logger.info("─" * 50)
        logger.info("Data Quality Report: %s", name)
        logger.info("  Rows: %d | Columns: %d", len(df), len(df.columns))
        logger.info("  Date range: %s → %s", df.index[0].date(), df.index[-1].date())
        logger.info("  Memory: %.1f MB", df.memory_usage(deep=True).sum() / 1_048_576)
        if not high_null.empty:
            logger.warning("  Columns with >5%% nulls: %s", high_null.to_dict())
        if "label" in df.columns:
            vc = df["label"].value_counts(normalize=True)
            logger.info("  Label distribution: %s", vc.round(3).to_dict())
        logger.info("─" * 50)


# ── CLI entry point ───────────────────────────────────────────────────────────
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="MAESTRO Data Pipeline")
    parser.add_argument("--mode",        choices=["full", "incremental"], default="full")
    parser.add_argument("--instrument",  default=None)
    parser.add_argument("--granularity", default="M5")
    parser.add_argument("--start",       default="2020-01-01")
    parser.add_argument("--end",         default=None)
    args = parser.parse_args()

    pipeline = DataPipeline()

    if args.mode == "full":
        pipeline.run_full(
            start=args.start,
            end=args.end,
            granularity=args.granularity,
        )
    else:
        pipeline.run_incremental(granularity=args.granularity)


if __name__ == "__main__":
    main()
