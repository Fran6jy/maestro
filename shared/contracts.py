"""Stable boundary between MAESTRO's prediction and trading systems."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from maestro.config.instruments import get_instrument_spec


@dataclass(frozen=True)
class PredictionSignal:
    """A forecast only; deliberately contains no order or position fields."""

    instrument: str
    timestamp: datetime
    data_as_of: datetime
    expires_at: datetime
    horizon_bars: int
    direction: int
    expected_return: float
    confidence: float
    uncertainty: float
    model_version: str

    def __post_init__(self) -> None:
        get_instrument_spec(self.instrument)
        for name in ("timestamp", "data_as_of", "expires_at"):
            value = getattr(self, name)
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{name} must be timezone-aware")
        if self.direction not in {-1, 0, 1}:
            raise ValueError("direction must be -1, 0, or 1")
        if self.horizon_bars < 1:
            raise ValueError("horizon_bars must be positive")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError("uncertainty must be in [0, 1]")
        if self.data_as_of > self.timestamp:
            raise ValueError("data_as_of cannot be later than the prediction timestamp")
        if self.expires_at <= self.timestamp:
            raise ValueError("expires_at must be later than the prediction timestamp")
        if not self.model_version.strip():
            raise ValueError("model_version is required")

    def is_fresh(self, now: datetime | None = None) -> bool:
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None or current.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        return self.timestamp <= current < self.expires_at

    def to_dict(self) -> dict:
        payload = asdict(self)
        for name in ("timestamp", "data_as_of", "expires_at"):
            payload[name] = payload[name].isoformat()
        return payload

