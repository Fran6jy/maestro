"""
maestro/data/features/engineer.py
===================================
Technical feature engineering for OHLCV data.

All features are designed to be:
  - Leak-free: only use information available at bar close time
  - Stationary: returns/z-scores preferred over raw prices
  - Interpretable: every feature maps to a known trading concept
  - Consistent: same pipeline used for training AND live inference

Feature Groups
--------------
1. Returns          — raw, log, lagged
2. Trend            — SMA, EMA, MACD, ADX
3. Momentum         — RSI, Stochastic, ROC, Williams %R
4. Volatility       — ATR, Bollinger Bands, realised vol, Garman-Klass
5. Volume           — OBV, volume z-score, VWAP proxy
6. Microstructure   — spread proxy, bar range, body-to-range ratio
7. Calendar         — hour-of-day, day-of-week, day-of-month (cyclical encode)
8. Cross-pair       — EUR/USD vs GBP/USD correlation, spread
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from maestro.config.config import get

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Transforms raw OHLCV DataFrame into a rich feature matrix.

    Usage
    -----
    >>> fe = FeatureEngineer()
    >>> features = fe.transform(ohlcv_df)
    >>> features.shape
    (48523, 87)

    Notes
    -----
    - All feature computations are vectorised (no Python loops).
    - NaN rows introduced by rolling windows are kept (caller decides
      whether to drop them — WFA engine handles this appropriately).
    - No look-ahead: all windows are strictly backward-looking.
    """

    def __init__(self) -> None:
        cfg = get("features", {})
        self.sma_windows    = cfg.get("sma_windows",    [10, 20, 50, 100, 200])
        self.ema_windows    = cfg.get("ema_windows",    [12, 26, 50])
        self.rsi_period     = cfg.get("rsi_period",     14)
        self.atr_period     = cfg.get("atr_period",     14)
        self.bb_period      = cfg.get("bb_period",      20)
        self.bb_std         = cfg.get("bb_std",         2.0)
        self.macd_fast      = cfg.get("macd_fast",      12)
        self.macd_slow      = cfg.get("macd_slow",      26)
        self.macd_signal    = cfg.get("macd_signal",    9)
        self.vol_lookback   = cfg.get("vol_lookback",   20)
        self.return_lags    = cfg.get("return_lags",    [1, 2, 3, 5, 10])

    # ── Main entry point ──────────────────────────────────────────────────────
    def transform(
        self,
        df: pd.DataFrame,
        drop_nan: bool = False,
    ) -> pd.DataFrame:
        """
        Full feature engineering pipeline.

        Parameters
        ----------
        df        : OHLCV DataFrame with columns [open, high, low, close, volume]
                    and DatetimeIndex (UTC)
        drop_nan  : if True, drop rows where any feature is NaN

        Returns
        -------
        pd.DataFrame with all original + engineered columns
        """
        _validate_ohlcv(df)
        out = df.copy()

        out = self._add_returns(out)
        out = self._add_trend(out)
        out = self._add_momentum(out)
        out = self._add_volatility(out)
        out = self._add_volume(out)
        out = self._add_microstructure(out)
        out = self._add_calendar(out)

        n_features = len(out.columns) - len(df.columns)
        logger.info("Feature engineering complete: +%d features → %d total columns",
                    n_features, len(out.columns))

        if drop_nan:
            before = len(out)
            out = out.dropna()
            logger.info("Dropped %d NaN rows (%d → %d)", before - len(out), before, len(out))

        return out

    # ── 1. Returns ────────────────────────────────────────────────────────────
    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df["close"]
        df["return_1"]     = c.pct_change(1)
        df["log_return_1"] = np.log(c / c.shift(1))

        for lag in self.return_lags:
            df[f"return_{lag}"]     = c.pct_change(lag)
            df[f"log_return_{lag}"] = np.log(c / c.shift(lag))

        # Return z-score (rolling 252 bars)
        r = df["return_1"]
        df["return_zscore_252"] = (r - r.rolling(252).mean()) / r.rolling(252).std()

        return df

    # ── 2. Trend ──────────────────────────────────────────────────────────────
    def _add_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        c = df["close"]

        # SMA
        for w in self.sma_windows:
            df[f"sma_{w}"] = c.rolling(w).mean()
            df[f"sma_{w}_dist"] = (c - df[f"sma_{w}"]) / df[f"sma_{w}"]  # % distance

        # SMA crossover signal (short/long)
        if 20 in self.sma_windows and 200 in self.sma_windows:
            df["sma_cross_signal"] = np.where(
                df["sma_20"] > df["sma_200"], 1,
                np.where(df["sma_20"] < df["sma_200"], -1, 0)
            )

        # EMA
        for w in self.ema_windows:
            df[f"ema_{w}"] = c.ewm(span=w, adjust=False).mean()

        # MACD
        ema_fast   = c.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow   = c.ewm(span=self.macd_slow, adjust=False).mean()
        macd_line  = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        df["macd"]          = macd_line
        df["macd_signal"]   = signal_line
        df["macd_histogram"]= macd_line - signal_line
        df["macd_cross"]    = np.where(macd_line > signal_line, 1, -1)

        # ADX (Average Directional Index) — trend strength
        df = self._add_adx(df, period=14)

        return df

    def _add_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()

        up_move   = h - h.shift(1)
        down_move = l.shift(1) - l
        plus_dm   = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        plus_di  = 100 * pd.Series(plus_dm,  index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr

        dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()

        df["adx"]       = adx
        df["adx_plus"]  = plus_di
        df["adx_minus"] = minus_di
        df["trend_strong"] = (adx > 25).astype(int)
        return df

    # ── 3. Momentum ───────────────────────────────────────────────────────────
    def _add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        c, h, l = df["close"], df["high"], df["low"]

        # RSI
        delta = c.diff()
        gain  = delta.clip(lower=0).ewm(com=self.rsi_period - 1, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(com=self.rsi_period - 1, adjust=False).mean()
        rs    = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi_overbought"]  = (df["rsi"] > 70).astype(int)
        df["rsi_oversold"]    = (df["rsi"] < 30).astype(int)

        # Stochastic %K and %D
        low_14  = l.rolling(14).min()
        high_14 = h.rolling(14).max()
        stoch_k = 100 * (c - low_14) / (high_14 - low_14).replace(0, np.nan)
        df["stoch_k"] = stoch_k
        df["stoch_d"] = stoch_k.rolling(3).mean()

        # Rate of Change
        for p in [5, 10, 20]:
            df[f"roc_{p}"] = c.pct_change(p) * 100

        # Williams %R
        df["williams_r"] = -100 * (high_14 - c) / (high_14 - low_14).replace(0, np.nan)

        # CCI (Commodity Channel Index)
        typical = (h + l + c) / 3
        mean_dev = typical.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        df["cci"] = (typical - typical.rolling(20).mean()) / (0.015 * mean_dev)

        return df

    # ── 4. Volatility ─────────────────────────────────────────────────────────
    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        c, h, l, o = df["close"], df["high"], df["low"], df["open"]
        log_r = np.log(c / c.shift(1))

        # Realized volatility (annualised)
        df["vol_realised"] = log_r.rolling(self.vol_lookback).std() * np.sqrt(252)

        # Garman-Klass volatility (uses O/H/L/C — more efficient than close-to-close)
        gk = (0.5 * (np.log(h / l) ** 2)
              - (2 * np.log(2) - 1) * (np.log(c / o) ** 2))
        df["vol_gk"] = np.sqrt(gk.rolling(self.vol_lookback).mean() * 252)

        # Bollinger Bands
        sma = c.rolling(self.bb_period).mean()
        std = c.rolling(self.bb_period).std()
        upper = sma + self.bb_std * std
        lower = sma - self.bb_std * std
        df["bb_upper"]    = upper
        df["bb_lower"]    = lower
        df["bb_middle"]   = sma
        df["bb_width"]    = (upper - lower) / sma     # bandwidth (normalised)
        df["bb_pct"]      = (c - lower) / (upper - lower).replace(0, np.nan)  # %B
        df["bb_squeeze"]  = (df["bb_width"] < df["bb_width"].rolling(126).quantile(0.2)).astype(int)

        # ATR (Average True Range)
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs()
        ], axis=1).max(axis=1)
        df["atr"]           = tr.ewm(com=self.atr_period - 1, adjust=False).mean()
        df["atr_normalised"]= df["atr"] / c         # ATR as % of price

        # High-vol regime flag
        df["vol_regime_high"] = (df["vol_realised"] > df["vol_realised"].rolling(252).quantile(0.75)).astype(int)

        return df

    # ── 5. Volume ─────────────────────────────────────────────────────────────
    def _add_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        v, c = df["volume"], df["close"]

        # Volume z-score (rolling 20 bars)
        v_mean = v.rolling(20).mean()
        v_std  = v.rolling(20).std()
        df["volume_zscore"] = (v - v_mean) / v_std.replace(0, np.nan)
        df["volume_high"]   = (df["volume_zscore"] > 2).astype(int)

        # On-Balance Volume (OBV)
        direction = np.sign(c.diff())
        df["obv"] = (v * direction).cumsum()
        df["obv_sma20"] = df["obv"].rolling(20).mean()
        df["obv_trend"] = np.where(df["obv"] > df["obv_sma20"], 1, -1)

        # Volume-weighted price proxy (no true VWAP without tick data)
        typical = (df["high"] + df["low"] + c) / 3
        df["vwap_proxy"] = (typical * v).rolling(20).sum() / v.rolling(20).sum()

        return df

    # ── 6. Microstructure ────────────────────────────────────────────────────
    def _add_microstructure(self, df: pd.DataFrame) -> pd.DataFrame:
        h, l, o, c = df["high"], df["low"], df["open"], df["close"]
        bar_range = h - l

        # Candle body relative to range
        df["body_ratio"]   = (c - o).abs() / bar_range.replace(0, np.nan)
        df["upper_shadow"]  = (h - pd.concat([o, c], axis=1).max(axis=1)) / bar_range.replace(0, np.nan)
        df["lower_shadow"]  = (pd.concat([o, c], axis=1).min(axis=1) - l) / bar_range.replace(0, np.nan)
        df["bar_direction"] = np.where(c >= o, 1, -1)

        # Bar range normalised (volatility proxy)
        df["bar_range_norm"] = bar_range / c

        # Close position within bar (0 = at low, 1 = at high)
        df["close_position"] = (c - l) / bar_range.replace(0, np.nan)

        return df

    # ── 7. Calendar ───────────────────────────────────────────────────────────
    def _add_calendar(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cyclical encoding of time features using sin/cos transforms.
        This preserves the circular nature (hour 23 is close to hour 0).
        """
        idx = df.index

        # Hour of day [0–23] — relevant for intraday data
        hour = idx.hour
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

        # Day of week [0–4] — Mon=0, Fri=4
        dow = idx.dayofweek
        df["dow_sin"] = np.sin(2 * np.pi * dow / 5)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 5)

        # Day of month [1–31]
        dom = idx.day
        df["dom_sin"] = np.sin(2 * np.pi * dom / 31)
        df["dom_cos"] = np.cos(2 * np.pi * dom / 31)

        # London/NY overlap session flag (12:00–16:00 UTC) — highest liquidity
        df["session_overlap"] = ((hour >= 12) & (hour < 16)).astype(int)
        # London only (07:00–12:00 UTC)
        df["session_london"]  = ((hour >= 7) & (hour < 12)).astype(int)

        return df


# ── Cross-pair features ───────────────────────────────────────────────────────
def add_cross_pair_features(
    eur_usd: pd.DataFrame,
    gbp_usd: pd.DataFrame,
    window: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add cross-pair correlation and spread features.
    Requires both EUR/USD and GBP/USD feature DataFrames.

    Returns updated (eur_usd, gbp_usd) DataFrames with additional columns:
      - cross_correlation_{window}: rolling Pearson correlation of log-returns
      - cross_spread: GBP/USD close - EUR/USD close (proxy for cable/euro spread)
      - cross_spread_zscore: z-score of above
    """
    # Align indices
    common_idx = eur_usd.index.intersection(gbp_usd.index)
    e = eur_usd.loc[common_idx, "log_return_1"]
    g = gbp_usd.loc[common_idx, "log_return_1"]

    corr = e.rolling(window).corr(g)
    spread = gbp_usd.loc[common_idx, "close"] - eur_usd.loc[common_idx, "close"]
    spread_z = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()

    for df, name in [(eur_usd, "EUR_USD"), (gbp_usd, "GBP_USD")]:
        df.loc[common_idx, f"cross_corr_{window}"]  = corr.values
        df.loc[common_idx, "cross_spread"]           = spread.values
        df.loc[common_idx, "cross_spread_zscore"]    = spread_z.values

    return eur_usd, gbp_usd


# ── Validation helper ─────────────────────────────────────────────────────────
def _validate_ohlcv(df: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a DatetimeIndex")
    if df.index.tz is None:
        raise ValueError("DatetimeIndex must be timezone-aware (UTC)")
