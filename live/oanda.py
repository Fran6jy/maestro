"""
maestro/live/oanda.py
=====================
OANDA client for the live trial: practice accounts only.

The only server address in this module is OANDA's practice API. The client
refuses to start if configured with any other address, or if
OANDA_ENVIRONMENT is anything but "practice", so no setting can point it at a
real-money account. Orders are never retried automatically, because a retry
after a timeout can fill twice; each order carries a unique client tag so the
loop can check whether it went through.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import pandas as pd
import requests

PRACTICE_URL = "https://api-fxpractice.oanda.com"


class PracticeOnlyError(RuntimeError):
    """Raised on any configuration that could reach a real-money account."""


@dataclass(frozen=True)
class Fill:
    order_tag: str
    instrument: str
    units: int
    price: float
    time: pd.Timestamp
    bid: float | None
    ask: float | None


def _secret(name: str) -> str:
    return os.environ[name].strip().strip("\"'﻿").strip()


class PracticeClient:
    def __init__(self, api_key: str | None = None, account_id: str | None = None,
                 base_url: str | None = None, session: requests.Session | None = None) -> None:
        base = (base_url or os.environ.get("OANDA_BASE_URL") or PRACTICE_URL).rstrip("/")
        if base != PRACTICE_URL:
            raise PracticeOnlyError(f"refusing to use {base}: the live trial runs on OANDA practice only")
        if os.environ.get("OANDA_ENVIRONMENT", "practice").strip().lower() != "practice":
            raise PracticeOnlyError("OANDA_ENVIRONMENT must be 'practice' for the live trial")
        self.base = base
        self.account = account_id or _secret("OANDA_ACCOUNT_ID")
        self.session = session or requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key or _secret('OANDA_API_KEY')}",
                                     "Content-Type": "application/json"})

    # ── Reading (retried) ────────────────────────────────────────────────────
    def _get(self, path: str, params: dict | None = None) -> dict:
        for attempt in range(4):
            try:
                resp = self.session.get(f"{self.base}{path}", params=params, timeout=30)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

    def candles(self, instrument: str, count: int = 500, granularity: str = "M5") -> pd.DataFrame:
        """The latest complete candles: mid OHLC, bid/ask close, tick volume."""
        from maestro.data.pipeline.store import _parse
        data = self._get(f"/v3/instruments/{instrument}/candles",
                         {"granularity": granularity, "price": "MBA", "count": count})
        return _parse([c for c in data.get("candles", []) if c.get("complete")])

    def quote(self, instrument: str) -> tuple[pd.Timestamp, float, float]:
        data = self._get(f"/v3/accounts/{self.account}/pricing", {"instruments": instrument})
        p = data["prices"][0]
        return pd.Timestamp(p["time"]), float(p["bids"][0]["price"]), float(p["asks"][0]["price"])

    def net_units(self, instrument: str) -> int:
        data = self._get(f"/v3/accounts/{self.account}/openPositions")
        for pos in data.get("positions", []):
            if pos["instrument"] == instrument:
                return int(pos["long"]["units"]) + int(pos["short"]["units"])
        return 0

    def summary(self) -> dict:
        return self._get(f"/v3/accounts/{self.account}/summary")["account"]

    # ── Orders (never retried) ───────────────────────────────────────────────
    def market_order(self, instrument: str, units: int, tag: str) -> Fill:
        """Fill-or-kill market order for `units` (positive buys, negative sells)."""
        body = {"order": {"type": "MARKET", "instrument": instrument, "units": str(int(units)),
                          "timeInForce": "FOK", "positionFill": "DEFAULT",
                          "clientExtensions": {"id": tag, "tag": "maestro-live"}}}
        resp = self.session.post(f"{self.base}/v3/accounts/{self.account}/orders", json=body, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        fill = data.get("orderFillTransaction")
        if fill is None:
            reason = data.get("orderCancelTransaction", {}).get("reason", "unknown")
            raise RuntimeError(f"order {tag} not filled: {reason}")
        price = fill.get("fullPrice") or {}
        bid = float(price["bids"][0]["price"]) if price.get("bids") else None
        ask = float(price["asks"][0]["price"]) if price.get("asks") else None
        return Fill(tag, instrument, int(float(fill["units"])), float(fill["price"]),
                    pd.Timestamp(fill["time"]), bid, ask)
