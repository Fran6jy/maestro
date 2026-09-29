"""Canonical instrument metadata shared by prediction, risk, and execution."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InstrumentSpec:
    instrument: str
    display: str
    asset_class: str
    base_currency: str
    quote_currency: str
    pip_size: float
    price_precision: int
    min_units: int
    max_units: int


INSTRUMENTS: dict[str, InstrumentSpec] = {
    "EUR_USD": InstrumentSpec("EUR_USD", "EUR/USD", "fx", "EUR", "USD", 0.0001, 5, 1, 10_000_000),
    "GBP_USD": InstrumentSpec("GBP_USD", "GBP/USD", "fx", "GBP", "USD", 0.0001, 5, 1, 10_000_000),
    "USD_JPY": InstrumentSpec("USD_JPY", "USD/JPY", "fx", "USD", "JPY", 0.01, 3, 1, 10_000_000),
    "USD_CHF": InstrumentSpec("USD_CHF", "USD/CHF", "fx", "USD", "CHF", 0.0001, 5, 1, 10_000_000),
    "USD_CAD": InstrumentSpec("USD_CAD", "USD/CAD", "fx", "USD", "CAD", 0.0001, 5, 1, 10_000_000),
    "AUD_USD": InstrumentSpec("AUD_USD", "AUD/USD", "fx", "AUD", "USD", 0.0001, 5, 1, 10_000_000),
    "NZD_USD": InstrumentSpec("NZD_USD", "NZD/USD", "fx", "NZD", "USD", 0.0001, 5, 1, 10_000_000),
    # OANDA represents one XAU_USD unit as one ounce. Broker/account limits are
    # still fetched live; these values are conservative local validation caps.
    "XAU_USD": InstrumentSpec("XAU_USD", "Gold / USD", "metal", "XAU", "USD", 0.01, 3, 1, 10_000),
}


def get_instrument_spec(instrument: str) -> InstrumentSpec:
    """Return metadata or fail closed for an unsupported instrument."""
    try:
        return INSTRUMENTS[instrument]
    except KeyError as exc:
        supported = ", ".join(sorted(INSTRUMENTS))
        raise ValueError(f"Unsupported instrument {instrument!r}; supported: {supported}") from exc

