"""Read-only translation boundary from OANDA payloads to trader contracts.

This module intentionally defines no order-creation or position-close method.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Mapping, Protocol

from maestro.config.instruments import get_instrument_spec
from maestro.trader.order_planner import BrokerPosition, BrokerSnapshot
from maestro.trader.portfolio_allocator import MarketQuote


class OANDAReadClient(Protocol):
    def get_open_positions(self) -> list[dict]: ...
    def get_account_summary(self) -> dict: ...


class HedgedPositionError(ValueError):
    """Raised because a single net target cannot safely represent both sides."""


class OANDAReadOnlyAdapter:
    def __init__(self, client: OANDAReadClient) -> None:
        self._client = client

    def snapshot(
        self,
        as_of: datetime | None = None,
        processed_client_ids: Iterable[str] = (),
    ) -> BrokerSnapshot:
        timestamp = as_of or datetime.now(timezone.utc)
        positions = self.parse_positions(self._client.get_open_positions())
        return BrokerSnapshot(timestamp, positions, frozenset(processed_client_ids))

    def account_equity(self) -> float:
        summary = self._client.get_account_summary()
        nav = float(summary.get("nav", 0.0))
        if nav <= 0:
            raise ValueError("OANDA account NAV must be positive")
        return nav

    @staticmethod
    def parse_positions(raw_positions: Iterable[Mapping]) -> dict[str, BrokerPosition]:
        positions: dict[str, BrokerPosition] = {}
        for raw in raw_positions:
            instrument = str(raw.get("instrument", ""))
            get_instrument_spec(instrument)
            long_side = raw.get("long") or {}
            short_side = raw.get("short") or {}
            long_units = int(float(long_side.get("units", 0)))
            short_units = int(float(short_side.get("units", 0)))
            if long_units and short_units:
                raise HedgedPositionError(
                    f"{instrument} has simultaneous long and short positions; manual resolution required"
                )
            units = long_units + short_units
            if units == 0:
                continue
            side = long_side if units > 0 else short_side
            average_price = float(side.get("averagePrice", 0.0))
            positions[instrument] = BrokerPosition(instrument, units, average_price)
        return positions

    @staticmethod
    def quote_from_tick(
        tick: Mapping,
        stop_distance_pct: float,
        financing_return: float = 0.0,
    ) -> MarketQuote:
        instrument = str(tick.get("instrument", ""))
        get_instrument_spec(instrument)
        raw_time = tick.get("time")
        if isinstance(raw_time, datetime):
            timestamp = raw_time
        elif isinstance(raw_time, str):
            timestamp = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
        else:
            raise ValueError("OANDA tick is missing a timestamp")
        return MarketQuote(
            instrument=instrument,
            timestamp=timestamp,
            bid=float(tick["bid"]),
            ask=float(tick["ask"]),
            stop_distance_pct=stop_distance_pct,
            tradeable=bool(tick.get("tradeable", True)),
            financing_return=financing_return,
        )

