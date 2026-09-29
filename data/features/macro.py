"""
maestro/data/features/macro.py
==============================
Macro features from FRED, each attached to a bar only once it was public.

The old pipeline stamped every FRED value at midnight on the date it describes
and forward-filled it through that day, so each 5-minute bar already saw that
day's closing VIX, and monthly CPI six weeks before it was published. Here every
observation carries the time it became available, and a bar only sees values
that were available by then:

  daily market series   VIXCLS, T10Y2Y, DFF: FRED posts day D's value the next
                        US morning, so it is used from D + 1 day, 14:00 UTC.
  weekly H.10 release   DEXUSEU, DEXUSUK: the Fed publishes a week's daily rates
                        the following Monday, so they are used from that Monday
                        at 21:00 UTC.
  monthly releases      CPIAUCSL, UNRATE: ALFRED gives the first-published value
                        and its release date; it is used from 14:00 UTC that day
                        (after the 08:30 ET release). Later revisions are never seen.

Derived features keep their old names so the models need no change, but they
are computed on each series' own observations (the old versions used calendar
days, so "252 days" meant eight months, not a year).
"""
from __future__ import annotations

import logging
import os
import time

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"

DAILY_SERIES   = ["VIXCLS", "T10Y2Y", "DFF"]
WEEKLY_H10     = ["DEXUSEU", "DEXUSUK"]
MONTHLY_SERIES = ["CPIAUCSL", "UNRATE"]
ALL_SERIES     = DAILY_SERIES + WEEKLY_H10 + MONTHLY_SERIES

DAILY_LAG       = pd.Timedelta(days=1, hours=14)
H10_TIME        = pd.Timedelta(hours=21)
RELEASE_TIME    = pd.Timedelta(hours=14)
MONTHLY_FALLBACK = pd.DateOffset(months=1, days=19)   # if ALFRED has no release date


def available_at(series: str, date: pd.Series, release: pd.Series | None = None) -> pd.Series:
    """UTC time at which the observation for `date` became public."""
    date = pd.to_datetime(date, utc=True)
    if series in DAILY_SERIES:
        return date + DAILY_LAG
    if series in WEEKLY_H10:
        return date + pd.to_timedelta(7 - date.dt.weekday, unit="D") + H10_TIME
    if series in MONTHLY_SERIES:
        rel = pd.to_datetime(release, utc=True) if release is not None else pd.Series(pd.NaT, index=date.index)
        return rel.fillna(date + MONTHLY_FALLBACK) + RELEASE_TIME
    raise ValueError(f"no availability rule for {series}")


# ── Fetching ─────────────────────────────────────────────────────────────────
def _observations(series: str, api_key: str, start: str, **extra) -> pd.DataFrame:
    params = {"series_id": series, "api_key": api_key, "file_type": "json",
              "observation_start": start, **extra}
    for attempt in range(4):
        try:
            resp = requests.get(FRED_URL, params=params, timeout=60)
            resp.raise_for_status()
            return pd.DataFrame(resp.json()["observations"])
        except (requests.RequestException, KeyError) as exc:
            if attempt == 3:
                raise RuntimeError(f"FRED {series}: {exc}") from exc
            time.sleep(2 ** attempt)


def fetch_fred(start: str = "2003-01-01", api_key: str | None = None) -> pd.DataFrame:
    """Every configured series in long form: series, date, value, available_at.

    Monthly series come from ALFRED as first releases, so revisions made after
    the fact never reach the features.
    """
    api_key = api_key or os.environ["FRED_API_KEY"]
    frames = []
    for sid in ALL_SERIES:
        if sid in MONTHLY_SERIES:
            raw = _observations(sid, api_key, start, realtime_start="1776-07-04",
                                realtime_end="9999-12-31", output_type=4)
            release = raw["realtime_start"]
        else:
            raw = _observations(sid, api_key, start)
            release = None
        df = pd.DataFrame({
            "series": sid,
            "date": pd.to_datetime(raw["date"], utc=True),
            "value": pd.to_numeric(raw["value"], errors="coerce"),   # FRED uses "." for missing
        })
        df["available_at"] = available_at(sid, df["date"], release)
        frames.append(df.dropna(subset=["value"]))
        time.sleep(0.3)
    out = pd.concat(frames, ignore_index=True)
    logger.info("FRED: %d observations across %d series", len(out), out["series"].nunique())
    return out


# ── Features ─────────────────────────────────────────────────────────────────
def _derived(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Each output column as (available_at, value), computed on native observations."""
    def obs(sid: str) -> pd.DataFrame:
        return raw[raw["series"] == sid].sort_values("date").set_index("date")

    cols: dict[str, pd.DataFrame] = {}
    for sid in ALL_SERIES:
        o = obs(sid)
        cols[sid] = pd.DataFrame({"t": o["available_at"], "v": o["value"]})

    vix = obs("VIXCLS")
    z = (vix["value"] - vix["value"].rolling(252).mean()) / vix["value"].rolling(252).std()
    cols["vix_zscore"] = pd.DataFrame({"t": vix["available_at"], "v": z})
    cols["vix_high"] = pd.DataFrame({"t": vix["available_at"], "v": (vix["value"] > 25).astype(float)})

    curve = obs("T10Y2Y")
    cols["yield_curve_inverted"] = pd.DataFrame({"t": curve["available_at"], "v": (curve["value"] < 0).astype(float)})

    dff = obs("DFF")["value"]
    month_ago = dff.reindex(dff.index - pd.Timedelta(days=30), method="ffill").to_numpy()
    cols["rate_change_1m"] = pd.DataFrame({"t": obs("DFF")["available_at"], "v": dff.to_numpy() - month_ago})

    cpi = obs("CPIAUCSL")
    cols["inflation_yoy"] = pd.DataFrame({"t": cpi["available_at"], "v": cpi["value"].pct_change(12) * 100})
    return cols


def macro_frame(raw: pd.DataFrame, bars: pd.DatetimeIndex) -> pd.DataFrame:
    """Macro columns for each bar, using only values public at that bar's time."""
    left = pd.DataFrame({"t": bars})
    out = pd.DataFrame(index=bars)
    for name, col in _derived(raw).items():
        col = col.dropna().sort_values("t").drop_duplicates("t", keep="last")
        col["t"] = col["t"].astype(left["t"].dtype)
        out[name] = pd.merge_asof(left, col, on="t", direction="backward")["v"].to_numpy()
    return out
