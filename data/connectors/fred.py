"""
maestro/data/connectors/fred.py
================================
FRED (Federal Reserve Economic Data) connector for macroeconomic series:
  - Fed Funds Rate, Yield Curve, VIX, CPI, Unemployment
  - Daily/monthly data aligned to OHLCV index via forward-fill
  - All series are regime-relevant signals for Agent 1

Dependencies: requests, pandas, fredapi (optional)
"""
from __future__ import annotations

import logging
import time
from datetime import datetime

import pandas as pd
import requests

from maestro.config.config import get, get_secret

logger = logging.getLogger(__name__)


class FREDConnector:
    """
    Fetches macroeconomic data from the FRED API and
    returns clean, forward-filled DataFrames ready for
    feature engineering.

    Usage
    -----
    >>> fred = FREDConnector()
    >>> df = fred.fetch_all_series(start="2000-01-01")
    >>> df.head()
    """

    def __init__(self) -> None:
        self.base_url = get("data_sources.fred.base_url")
        self.api_key  = get_secret(get("data_sources.fred.token_env_var", "FRED_API_KEY"))
        self.series   = get("data_sources.fred.series", [])

    # ── Single series ─────────────────────────────────────────────────────────
    def fetch_series(
        self,
        series_id: str,
        start: str = "2000-01-01",
        end: str | None = None,
    ) -> pd.Series:
        """
        Fetch a single FRED series as a daily pd.Series (forward-filled).

        Parameters
        ----------
        series_id : FRED series identifier, e.g. "DFF", "T10Y2Y", "VIXCLS"
        start     : start date (ISO string)
        end       : end date (ISO string). Defaults to today.

        Returns
        -------
        pd.Series with DatetimeIndex (UTC, daily frequency)
        """
        url = f"{self.base_url}/series/observations"
        params = {
            "series_id":             series_id,
            "api_key":               self.api_key,
            "file_type":             "json",
            "observation_start":     start,
            "sort_order":            "asc",
        }
        if end:
            params["observation_end"] = end

        logger.info("Fetching FRED series: %s from %s", series_id, start)

        for attempt in range(3):
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                observations = resp.json().get("observations", [])
                break
            except requests.RequestException as exc:
                if attempt == 2:
                    raise FREDError(f"Failed to fetch {series_id}: {exc}") from exc
                time.sleep(2 ** attempt)

        if not observations:
            logger.warning("No data returned for series %s", series_id)
            return pd.Series(name=series_id, dtype=float)

        # Build series — FRED uses "." for missing values
        records = {}
        for obs in observations:
            val = obs.get("value", ".")
            if val != ".":
                try:
                    records[obs["date"]] = float(val)
                except ValueError:
                    pass

        s = pd.Series(records, name=series_id)
        s.index = pd.DatetimeIndex(s.index, tz="UTC")
        s = s.sort_index()

        logger.info("  → %d observations for %s (%s → %s)",
                    len(s), series_id, s.index[0].date(), s.index[-1].date())
        return s

    # ── All configured series ─────────────────────────────────────────────────
    def fetch_all_series(
        self,
        start: str = "2000-01-01",
        end: str | None = None,
        resample_freq: str = "D",
    ) -> pd.DataFrame:
        """
        Fetch all series defined in config, align to a common daily index,
        and forward-fill gaps (weekends, holidays).

        Returns
        -------
        pd.DataFrame with one column per series, daily DatetimeIndex (UTC)
        Columns: DFF, T10Y2Y, VIXCLS, CPIAUCSL, UNRATE, DEXUSEU, DEXUSUK
        """
        frames: dict[str, pd.Series] = {}

        for series_cfg in self.series:
            sid = series_cfg["id"]
            try:
                frames[sid] = self.fetch_series(sid, start=start, end=end)
                time.sleep(0.5)   # respect FRED rate limit (120 req/min)
            except FREDError as exc:
                logger.error("Skipping %s: %s", sid, exc)

        if not frames:
            return pd.DataFrame()

        # Align all series to a common date range (union)
        df = pd.DataFrame(frames)
        df = df.resample(resample_freq).last()          # daily cadence
        df = df.ffill()                                  # forward-fill missing
        df = df.dropna(how="all")

        # ── Derived macro features ─────────────────────────────────────────────
        if "T10Y2Y" in df.columns:
            # Yield curve inversion flag (strong recession predictor)
            df["yield_curve_inverted"] = (df["T10Y2Y"] < 0).astype(int)

        if "DFF" in df.columns:
            # Rate hike/cut momentum (1-month change)
            df["rate_change_1m"] = df["DFF"].diff(21)

        if "VIXCLS" in df.columns:
            # High-vol regime flag (VIX > 25 = elevated risk)
            df["vix_high"] = (df["VIXCLS"] > 25).astype(int)
            # VIX z-score (rolling 252-day)
            vix_mean = df["VIXCLS"].rolling(252).mean()
            vix_std  = df["VIXCLS"].rolling(252).std()
            df["vix_zscore"] = (df["VIXCLS"] - vix_mean) / vix_std

        if "CPIAUCSL" in df.columns:
            # YoY inflation rate
            df["inflation_yoy"] = df["CPIAUCSL"].pct_change(252) * 100

        logger.info(
            "Macro dataset built: %d rows × %d columns (%s → %s)",
            len(df), len(df.columns),
            df.index[0].date(), df.index[-1].date()
        )
        return df

    # ── Align macro data to OHLCV index ───────────────────────────────────────
    @staticmethod
    def align_to_ohlcv(
        macro_df: pd.DataFrame,
        ohlcv_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Reindex macro data to match the OHLCV bar index via forward-fill.
        This ensures every OHLCV bar has an associated macro signal.

        Parameters
        ----------
        macro_df    : daily macro DataFrame (from fetch_all_series)
        ohlcv_index : DatetimeIndex of the OHLCV data (any granularity)

        Returns
        -------
        pd.DataFrame with same index as ohlcv_index, all macro columns present
        """
        # Reindex to OHLCV frequency, then forward-fill macro signals
        combined_index = macro_df.index.union(ohlcv_index)
        aligned = macro_df.reindex(combined_index).ffill()
        return aligned.reindex(ohlcv_index)


# ── Custom Exceptions ─────────────────────────────────────────────────────────
class FREDError(Exception):
    """Raised when FRED API calls fail."""
    pass
