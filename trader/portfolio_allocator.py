"""Cost-aware portfolio allocator for FX majors and XAU/USD.

The allocator is intentionally broker-agnostic. It converts immutable
``PredictionSignal`` objects plus current quotes into target positions. It does
not place orders; execution and broker reconciliation remain separate concerns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Iterable, Mapping

from maestro.config.instruments import InstrumentSpec, get_instrument_spec
from maestro.shared.contracts import PredictionSignal


@dataclass(frozen=True)
class MarketQuote:
    instrument: str
    timestamp: datetime
    bid: float
    ask: float
    stop_distance_pct: float
    tradeable: bool = True
    financing_return: float = 0.0

    def __post_init__(self) -> None:
        get_instrument_spec(self.instrument)
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("quote timestamp must be timezone-aware")
        if self.bid <= 0 or self.ask <= 0 or self.ask < self.bid:
            raise ValueError("quote must satisfy 0 < bid <= ask")
        if self.stop_distance_pct <= 0:
            raise ValueError("stop_distance_pct must be positive")
        if self.financing_return < 0:
            raise ValueError("financing_return must be a non-negative cost")

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread_return(self) -> float:
        return (self.ask - self.bid) / self.mid


@dataclass(frozen=True)
class PortfolioLimits:
    max_open_positions: int = 3
    max_total_risk_pct: float = 0.0100
    max_fx_risk_pct: float = 0.0075
    max_metal_risk_pct: float = 0.0025
    max_risk_per_position_pct: float = 0.0025
    max_position_notional_pct: float = 0.50
    max_net_currency_exposure_pct: float = 0.50
    max_gross_currency_exposure_pct: float = 1.00
    min_confidence: float = 0.55
    max_uncertainty: float = 0.45
    min_net_edge_bps: float = 1.0
    estimated_slippage_bps_per_side: float = 0.25
    max_quote_age_seconds: float = 15.0

    def __post_init__(self) -> None:
        positive = (
            self.max_open_positions,
            self.max_total_risk_pct,
            self.max_fx_risk_pct,
            self.max_metal_risk_pct,
            self.max_risk_per_position_pct,
            self.max_position_notional_pct,
            self.max_net_currency_exposure_pct,
            self.max_gross_currency_exposure_pct,
            self.max_quote_age_seconds,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("portfolio limits must be positive")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be in [0, 1]")
        if not 0 <= self.max_uncertainty <= 1:
            raise ValueError("max_uncertainty must be in [0, 1]")


@dataclass(frozen=True)
class TargetPosition:
    instrument: str
    units: int
    target_notional_usd: float
    risk_usd: float
    expected_net_return: float
    score: float
    stop_distance_pct: float
    model_version: str
    prediction_timestamp: datetime


@dataclass(frozen=True)
class RejectedCandidate:
    instrument: str
    reason: str
    detail: str = ""


@dataclass(frozen=True)
class AllocationResult:
    as_of: datetime
    equity: float
    targets: tuple[TargetPosition, ...]
    rejected: tuple[RejectedCandidate, ...]
    net_exposure_usd: Mapping[str, float] = field(default_factory=dict)
    gross_exposure_usd: Mapping[str, float] = field(default_factory=dict)
    total_risk_usd: float = 0.0
    fx_risk_usd: float = 0.0
    metal_risk_usd: float = 0.0


@dataclass(frozen=True)
class _Candidate:
    signal: PredictionSignal
    quote: MarketQuote
    spec: InstrumentSpec
    expected_net_return: float
    score: float


class PortfolioAllocator:
    """Rank predictions and construct a constrained target portfolio."""

    def __init__(self, limits: PortfolioLimits | None = None) -> None:
        self.limits = limits or PortfolioLimits()

    def allocate(
        self,
        predictions: Iterable[PredictionSignal],
        quotes: Mapping[str, MarketQuote],
        equity: float,
        as_of: datetime | None = None,
    ) -> AllocationResult:
        now = as_of or datetime.now(timezone.utc)
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware")
        if equity <= 0:
            raise ValueError("equity must be positive")

        latest = self._latest_predictions(predictions)
        candidates: list[_Candidate] = []
        rejected: list[RejectedCandidate] = []
        for instrument, signal in latest.items():
            candidate, rejection = self._screen(signal, quotes.get(instrument), now)
            if candidate is not None:
                candidates.append(candidate)
            elif rejection is not None:
                rejected.append(rejection)
        candidates.sort(key=lambda item: item.score, reverse=True)

        targets: list[TargetPosition] = []
        net_exposure: dict[str, float] = {}
        gross_exposure: dict[str, float] = {}
        total_risk = fx_risk = metal_risk = 0.0

        for candidate in candidates:
            if len(targets) >= self.limits.max_open_positions:
                rejected.append(RejectedCandidate(candidate.spec.instrument, "position_limit"))
                continue

            class_used = metal_risk if candidate.spec.asset_class == "metal" else fx_risk
            class_cap = equity * (
                self.limits.max_metal_risk_pct
                if candidate.spec.asset_class == "metal"
                else self.limits.max_fx_risk_pct
            )
            risk_budget = min(
                equity * self.limits.max_risk_per_position_pct,
                equity * self.limits.max_total_risk_pct - total_risk,
                class_cap - class_used,
            )
            if risk_budget <= 0:
                rejected.append(RejectedCandidate(candidate.spec.instrument, "risk_budget_exhausted"))
                continue

            desired_notional = min(
                risk_budget / candidate.quote.stop_distance_pct,
                equity * self.limits.max_position_notional_pct,
            )
            exposure_cap = self._available_exposure_notional(
                candidate.spec,
                candidate.signal.direction,
                equity,
                net_exposure,
                gross_exposure,
            )
            allowed_notional = min(desired_notional, exposure_cap)
            signed_units = self._units_for_notional(
                candidate.spec,
                allowed_notional,
                candidate.quote.mid,
                candidate.signal.direction,
            )
            if abs(signed_units) < candidate.spec.min_units:
                rejected.append(RejectedCandidate(candidate.spec.instrument, "currency_exposure_limit"))
                continue

            actual_notional = self._notional_usd(candidate.spec, signed_units, candidate.quote.mid)
            actual_risk = actual_notional * candidate.quote.stop_distance_pct
            target = TargetPosition(
                instrument=candidate.spec.instrument,
                units=signed_units,
                target_notional_usd=actual_notional,
                risk_usd=actual_risk,
                expected_net_return=candidate.expected_net_return,
                score=candidate.score,
                stop_distance_pct=candidate.quote.stop_distance_pct,
                model_version=candidate.signal.model_version,
                prediction_timestamp=candidate.signal.timestamp,
            )
            targets.append(target)
            total_risk += actual_risk
            if candidate.spec.asset_class == "metal":
                metal_risk += actual_risk
            else:
                fx_risk += actual_risk
            self._add_exposure(candidate.spec, signed_units, candidate.quote.mid, net_exposure, gross_exposure)

        return AllocationResult(
            as_of=now,
            equity=equity,
            targets=tuple(targets),
            rejected=tuple(rejected),
            net_exposure_usd=dict(net_exposure),
            gross_exposure_usd=dict(gross_exposure),
            total_risk_usd=total_risk,
            fx_risk_usd=fx_risk,
            metal_risk_usd=metal_risk,
        )

    @staticmethod
    def _latest_predictions(predictions: Iterable[PredictionSignal]) -> dict[str, PredictionSignal]:
        latest: dict[str, PredictionSignal] = {}
        for signal in predictions:
            previous = latest.get(signal.instrument)
            if previous is None or signal.timestamp > previous.timestamp:
                latest[signal.instrument] = signal
        return latest

    def _screen(
        self,
        signal: PredictionSignal,
        quote: MarketQuote | None,
        now: datetime,
    ) -> tuple[_Candidate | None, RejectedCandidate | None]:
        instrument = signal.instrument
        if not signal.is_fresh(now):
            return None, RejectedCandidate(instrument, "stale_prediction")
        if quote is None:
            return None, RejectedCandidate(instrument, "missing_quote")
        if quote.instrument != instrument:
            return None, RejectedCandidate(instrument, "quote_instrument_mismatch")
        if not quote.tradeable:
            return None, RejectedCandidate(instrument, "instrument_not_tradeable")
        quote_age = (now - quote.timestamp).total_seconds()
        if quote_age < 0 or quote_age > self.limits.max_quote_age_seconds:
            return None, RejectedCandidate(instrument, "stale_quote", f"age={quote_age:.1f}s")
        if signal.direction == 0:
            return None, RejectedCandidate(instrument, "flat_prediction")
        if signal.confidence < self.limits.min_confidence:
            return None, RejectedCandidate(instrument, "low_confidence")
        if signal.uncertainty > self.limits.max_uncertainty:
            return None, RejectedCandidate(instrument, "high_uncertainty")

        directional_edge = signal.direction * signal.expected_return
        if directional_edge <= 0:
            return None, RejectedCandidate(instrument, "inconsistent_expected_return")
        round_trip_cost = (
            quote.spread_return
            + 2.0 * self.limits.estimated_slippage_bps_per_side / 10_000.0
            + quote.financing_return
        )
        net_edge = directional_edge - round_trip_cost
        if net_edge * 10_000.0 < self.limits.min_net_edge_bps:
            return None, RejectedCandidate(
                instrument,
                "insufficient_net_edge",
                f"net_edge_bps={net_edge * 10_000.0:.2f}",
            )
        score = net_edge * signal.confidence * (1.0 - signal.uncertainty) / quote.stop_distance_pct
        return _Candidate(signal, quote, get_instrument_spec(instrument), net_edge, score), None

    def _available_exposure_notional(
        self,
        spec: InstrumentSpec,
        direction: int,
        equity: float,
        net: Mapping[str, float],
        gross: Mapping[str, float],
    ) -> float:
        net_limit = equity * self.limits.max_net_currency_exposure_pct
        gross_limit = equity * self.limits.max_gross_currency_exposure_pct
        available = float("inf")
        for currency, sign in ((spec.base_currency, direction), (spec.quote_currency, -direction)):
            current_net = net.get(currency, 0.0)
            current_gross = gross.get(currency, 0.0)
            gross_room = gross_limit - current_gross
            net_room = net_limit - current_net if sign > 0 else net_limit + current_net
            available = min(available, max(0.0, gross_room), max(0.0, net_room))
        return available

    @staticmethod
    def _units_for_notional(
        spec: InstrumentSpec,
        notional_usd: float,
        price: float,
        direction: int,
    ) -> int:
        if spec.quote_currency == "USD":
            raw_units = notional_usd / price
        elif spec.base_currency == "USD":
            raw_units = notional_usd
        else:
            return 0
        steps = math.floor(raw_units / spec.min_units)
        units = min(steps * spec.min_units, spec.max_units)
        return int(direction * units)

    @staticmethod
    def _notional_usd(spec: InstrumentSpec, signed_units: int, price: float) -> float:
        if spec.quote_currency == "USD":
            return abs(signed_units) * price
        if spec.base_currency == "USD":
            return abs(signed_units)
        raise ValueError(f"{spec.instrument} requires a home-currency conversion")

    def _add_exposure(
        self,
        spec: InstrumentSpec,
        signed_units: int,
        price: float,
        net: dict[str, float],
        gross: dict[str, float],
    ) -> None:
        notional = self._notional_usd(spec, signed_units, price)
        direction = 1 if signed_units > 0 else -1
        for currency, signed_notional in (
            (spec.base_currency, direction * notional),
            (spec.quote_currency, -direction * notional),
        ):
            net[currency] = net.get(currency, 0.0) + signed_notional
            gross[currency] = gross.get(currency, 0.0) + notional
