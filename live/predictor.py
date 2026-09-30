"""
maestro/live/predictor.py
=========================
MAESTRO's forecast for the latest complete bar, computed the way the backtest does.

The TFT reads the next 12 bars' calendar features (hour, weekday, session flags)
when forecasting a bar, so a frame that ends at the latest bar gives it no
forecast there. SignalAgent.predict_bar works around this by taking the TFT's
last row, which is 12 bars (an hour) stale. Here the frame is extended with the
next 12 bars' calendar features instead, which are known in advance, so the
latest bar gets the same forecast the backtest would give it. Regimes are
filtered causally, as in the backtest.

    model = LiveModel.load(models_dir)
    row = model.forecast_latest(features)     # signal, confidence, regime, pred_p50, ...
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from maestro.backtesting.maestro_runner import SIGNAL_COLUMNS
from maestro.data.pipeline.store import MINUTES

CONTEXT_BARS = 2000       # a week of 5-minute bars: model windows plus time for the regime filter to settle


def future_calendar(last: pd.Timestamp, n: int, granularity: str = "M5") -> pd.DataFrame:
    """Calendar features for the n bars after `last` (they depend only on the timestamp)."""
    from maestro.data.features.engineer import FeatureEngineer
    step = pd.Timedelta(minutes=MINUTES[granularity])
    idx = pd.DatetimeIndex([last + step * (k + 1) for k in range(n)])
    return FeatureEngineer()._add_calendar(pd.DataFrame(index=idx))


class LiveModel:
    def __init__(self, regime, signal, meta: dict | None = None, granularity: str = "M5") -> None:
        self.regime, self.signal, self.meta, self.granularity = regime, signal, meta or {}, granularity

    # ── Persistence ──────────────────────────────────────────────────────────
    def save(self, directory: Path) -> None:
        directory = Path(directory)
        self.regime.save(directory / "regime")
        self.signal.save(directory / "signal")
        (directory / "meta.json").write_text(json.dumps({**self.meta, "granularity": self.granularity},
                                                        indent=2, default=str))

    @classmethod
    def load(cls, directory: Path) -> "LiveModel":
        from maestro.agents.regime.regime_classifier import RegimeDetectionAgent
        from maestro.agents.signal.signal_agent import SignalAgent
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text())
        return cls(RegimeDetectionAgent.load(directory / "regime"), SignalAgent.load(directory / "signal"),
                   meta, meta.get("granularity", "M5"))

    # ── Forecast ─────────────────────────────────────────────────────────────
    def start_filter(self, first_live_bar: pd.Timestamp, features: pd.DataFrame) -> None:
        """Pin where the regime filter starts: WARMUP_BARS before the first live bar,
        exactly as a backtest block starts its filter before its first test bar.
        The HMM's regime filter depends on where it starts, so this makes live
        regimes equal the backtest's for the same deployment."""
        from maestro.backtesting.maestro_runner import WARMUP_BARS
        # Count back over all rows, then drop incomplete ones, exactly as predict_bars does:
        # the HMM is slow to forget where it started, so the same start gives the same regimes.
        first = features.index.get_indexer([first_live_bar])[0]
        self.meta["filter_start"] = str(features.index[max(0, first - WARMUP_BARS)])

    def forecast_latest(self, features: pd.DataFrame) -> pd.Series:
        """The forecast for the last complete bar in `features` (complete rows only).

        Regimes are filtered from meta["filter_start"] (set by start_filter at
        deployment); the signal models see the latest CONTEXT_BARS bars.
        """
        from maestro.agents.signal.tft_model import HORIZONS
        if "filter_start" not in self.meta:
            raise RuntimeError("call start_filter() when the model is deployed")
        complete = features.dropna()
        history = complete.loc[pd.Timestamp(self.meta["filter_start"]):]
        regimes = self.regime.predict_batch(history)["regime"]
        ctx = complete.tail(CONTEXT_BARS)
        ahead = future_calendar(ctx.index[-1], max(HORIZONS), self.granularity)
        frame = pd.concat([ctx, ahead])
        out = self.signal.predict_batch(frame, regimes.reindex(frame.index).ffill())
        row = out.loc[ctx.index[-1], [c for c in SIGNAL_COLUMNS if c in out.columns]]
        return row.rename(ctx.index[-1])
