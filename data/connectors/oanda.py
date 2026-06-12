"""
maestro/data/connectors/oanda.py
=================================
Production OANDA connector for:
  - Historical OHLCV data (full range, auto-paginated)
  - Real-time tick streaming (async generator)
  - Order management (used later by Execution Agent)

Design principles
-----------------
- Stateless: each method is independently callable
- Resilient: exponential back-off on 429/5xx errors
- Typed: every public method is fully type-annotated
- Observable: structured logging on every request

Dependencies: requests, pandas, python-dateutil, tqdm
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Generator, Iterator

import pandas as pd
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from dateutil.relativedelta import relativedelta

from maestro.config.config import get, get_secret

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
GRANULARITY_SECONDS: dict[str, int] = {
    "S5": 5, "S10": 10, "S15": 15, "S30": 30,
    "M1": 60, "M2": 120, "M4": 240, "M5": 300,
    "M10": 600, "M15": 900, "M30": 1800,
    "H1": 3600, "H2": 7200, "H3": 10800, "H4": 14400,
    "H6": 21600, "H8": 28800, "H12": 43200,
    "D": 86400, "W": 604800, "M": 2592000,
}


# ── Data Models ──────────────────────────────────────────────────────────────
@dataclass
class OANDACandle:
    """Single OHLCV candle from OANDA API."""
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    complete: bool


@dataclass
class OANDAConfig:
    """Runtime config resolved from settings.yaml + env vars."""
    base_url: str = field(default_factory=lambda: get("data_sources.oanda.base_url"))
    account_id: str = field(default_factory=lambda: get_secret(
        get("data_sources.oanda.account_env_var", "OANDA_ACCOUNT_ID")))
    token: str = field(default_factory=lambda: get_secret(
        get("data_sources.oanda.token_env_var", "OANDA_API_TOKEN")))
    max_candles: int = field(default_factory=lambda: get(
        "data_sources.oanda.max_candles_per_request", 5000))
    retry_attempts: int = field(default_factory=lambda: get(
        "data_sources.oanda.retry_attempts", 3))
    retry_backoff: float = field(default_factory=lambda: get(
        "data_sources.oanda.retry_backoff_seconds", 2.0))


# ── Main Connector ────────────────────────────────────────────────────────────
class OANDAConnector:
    """
    Thin, resilient wrapper around the OANDA v20 REST API.

    Usage
    -----
    >>> conn = OANDAConnector()
    >>> df = conn.fetch_historical("EUR_USD", "M5",
    ...         start="2023-01-01", end="2023-12-31")
    >>> for tick in conn.stream_prices(["EUR_USD", "GBP_USD"]):
    ...     print(tick)
    """

    def __init__(self, config: OANDAConfig | None = None) -> None:
        self.cfg = config or OANDAConfig()
        self._session = self._build_session()

    # ── HTTP layer ────────────────────────────────────────────────────────────
    def _build_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update({
            "Authorization": f"Bearer {self.cfg.token}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "RFC3339",
        })
        # Disable proxies and trust_env to prevent Windows proxy/PAC hangs
        session.trust_env = False
        session.proxies = {"http": None, "https": None}
        return session

    def _get(self, endpoint: str, params: dict | None = None) -> dict:
        """
        GET with exponential back-off retry on 429 / 5xx.
        Raises OANDAError on persistent failure.
        """
        url = f"{self.cfg.base_url}/{endpoint}"
        for attempt in range(1, self.cfg.retry_attempts + 1):
            try:
                resp = self._session.get(url, params=params, timeout=(5, 30), verify=False)

                if resp.status_code == 200:
                    return resp.json()

                if resp.status_code == 429:
                    wait = self.cfg.retry_backoff * (2 ** (attempt - 1))
                    logger.warning("Rate limited — waiting %.1fs (attempt %d/%d)",
                                   wait, attempt, self.cfg.retry_attempts)
                    time.sleep(wait)
                    continue

                if resp.status_code >= 500:
                    wait = self.cfg.retry_backoff * attempt
                    logger.warning("Server error %d — retrying in %.1fs",
                                   resp.status_code, wait)
                    time.sleep(wait)
                    continue

                # 4xx client errors — don't retry
                logger.error("OANDA API error %d: %s", resp.status_code, resp.text)
                resp.raise_for_status()

            except requests.RequestException as exc:
                if attempt == self.cfg.retry_attempts:
                    raise OANDAError(f"Request failed after {attempt} attempts: {exc}") from exc
                time.sleep(self.cfg.retry_backoff * attempt)

        raise OANDAError(f"Exhausted {self.cfg.retry_attempts} retry attempts for {url}")

    # ── Historical data ───────────────────────────────────────────────────────
    def fetch_historical(
        self,
        instrument: str,
        granularity: str,
        start: str | datetime,
        end: str | datetime | None = None,
        price: str = "M",          # M=mid, B=bid, A=ask
    ) -> pd.DataFrame:
        """
        Fetch complete OHLCV history for an instrument, auto-paginating
        through OANDA's 5000-candle limit.

        Parameters
        ----------
        instrument  : OANDA instrument code, e.g. "EUR_USD"
        granularity : e.g. "M5", "H1", "D"
        start       : ISO date string or datetime (UTC)
        end         : ISO date string or datetime (UTC). Defaults to now.
        price       : "M" (mid), "B" (bid), "A" (ask)

        Returns
        -------
        pd.DataFrame with columns: [open, high, low, close, volume]
            Index: pd.DatetimeIndex (UTC)
        """
        start_dt = _parse_dt(start)
        end_dt = _parse_dt(end) if end else datetime.now(timezone.utc)

        logger.info(
            "Fetching %s %s from %s to %s",
            instrument, granularity,
            start_dt.date(), end_dt.date()
        )

        # Cap end_dt to last weekday to avoid infinite loop on weekends/holidays
        from datetime import timezone as _tz
        _now = datetime.now(timezone.utc)
        _days_back = (_now.weekday() - 4) % 7  # days since last Friday
        _last_friday = _now - pd.Timedelta(days=_days_back)
        _last_friday = _last_friday.replace(hour=21, minute=55, second=0, microsecond=0)
        if end_dt > _last_friday:
            end_dt = _last_friday

        all_candles: list[OANDACandle] = []
        cursor = start_dt
        prev_cursor = None

        while cursor < end_dt:
            # Safety: break if cursor stopped advancing (e.g. market closed)
            if prev_cursor is not None and cursor <= prev_cursor:
                break
            prev_cursor = cursor

            batch = self._fetch_candle_batch(
                instrument, granularity, cursor, end_dt, price
            )
            if not batch:
                break

            all_candles.extend(batch)
            new_cursor = batch[-1].time + pd.Timedelta(seconds=1)

            # Only advance if we actually moved forward
            if new_cursor <= cursor:
                break
            cursor = new_cursor

            print(f"  {instrument} {granularity}: {len(all_candles):,} candles up to {batch[-1].time.date()}", flush=True)

            # Brief pause to respect rate limits
            time.sleep(0.05)

        if not all_candles:
            logger.warning("No candles returned for %s %s", instrument, granularity)
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = _candles_to_dataframe(all_candles)
        logger.info(
            "Completed: %d candles for %s %s (%s → %s)",
            len(df), instrument, granularity,
            df.index[0].date(), df.index[-1].date()
        )
        return df

    # Granularity → minutes per bar (used to cap chunk size)
    _GRAN_MINUTES: dict[str, int] = {
        "M1": 1, "M2": 2, "M4": 4, "M5": 5, "M10": 10, "M15": 15,
        "M30": 30, "H1": 60, "H2": 120, "H3": 180, "H4": 240,
        "H6": 360, "H8": 480, "H12": 720, "D": 1440, "W": 10080,
    }
    _MAX_CANDLES_PER_REQUEST = 4500   # OANDA hard cap is 5000; use 4500 for safety

    def _fetch_candle_batch(
        self,
        instrument: str,
        granularity: str,
        from_dt: datetime,
        to_dt: datetime,
        price: str,
    ) -> list[OANDACandle]:
        """Fetch a single batch from OANDA, chunking by time to stay under 5000-candle cap."""
        # Work out how many minutes each candle covers, then cap each HTTP request
        # to at most _MAX_CANDLES_PER_REQUEST candles worth of time.
        mins_per_bar  = self._GRAN_MINUTES.get(granularity, 5)
        chunk_minutes = mins_per_bar * self._MAX_CANDLES_PER_REQUEST
        chunk_delta   = pd.Timedelta(minutes=chunk_minutes)

        # Cap the end of this chunk so we never ask for more than the limit
        chunk_end = min(from_dt + chunk_delta, to_dt)

        # OANDA v20: use 'from' + 'count' (not from+to) to avoid the
        # "count exceeded" error that occurs with large date ranges.
        actual_count = max(1, int((chunk_end - from_dt).total_seconds() / 60 / mins_per_bar))
        actual_count = min(actual_count, self._MAX_CANDLES_PER_REQUEST)

        params = {
            "granularity": granularity,
            "from":  from_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "count": actual_count,
            "price": price,
        }
        endpoint = f"v3/instruments/{instrument}/candles"
        data = self._get(endpoint, params)
        raw_candles = data.get("candles", [])
        return [_parse_candle(c, price) for c in raw_candles if c.get("complete", True)]

    # ── Fetch multiple instruments ─────────────────────────────────────────────
    def fetch_all_instruments(
        self,
        granularity: str,
        start: str,
        end: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch historical data for all configured instruments.

        Returns
        -------
        dict mapping instrument_id → pd.DataFrame
        """
        from maestro.config.config import instrument_ids
        results = {}
        for instr in instrument_ids():
            try:
                results[instr] = self.fetch_historical(instr, granularity, start, end)
            except OANDAError as exc:
                logger.error("Failed to fetch %s: %s", instr, exc)
        return results

    # ── Account info ──────────────────────────────────────────────────────────
    def get_account_summary(self) -> dict:
        """Return account NAV, balance, open positions, margin used."""
        data = self._get(f"v3/accounts/{self.cfg.account_id}/summary")
        acc = data.get("account", {})
        return {
            "balance":          float(acc.get("balance", 0)),
            "nav":              float(acc.get("NAV", 0)),
            "unrealized_pl":    float(acc.get("unrealizedPL", 0)),
            "realized_pl":      float(acc.get("pl", 0)),
            "margin_used":      float(acc.get("marginUsed", 0)),
            "margin_available": float(acc.get("marginAvailable", 0)),
            "open_trade_count": int(acc.get("openTradeCount", 0)),
        }

    def get_open_positions(self) -> list[dict]:
        """Return all currently open positions."""
        data = self._get(f"v3/accounts/{self.cfg.account_id}/openPositions")
        return data.get("positions", [])

    def place_market_order(
        self,
        instrument: str,
        units: int,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        client_id: str | None = None,
    ) -> dict:
        """Submit a market order to OANDA."""
        order_body: dict = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(units),
                "timeInForce": "FOK",
                "positionFill": "DEFAULT",
            }
        }
        if client_id:
            order_body["order"]["clientExtensions"] = {"id": client_id}
        if stop_loss:
            order_body["order"]["stopLossOnFill"] = {
                "price": f"{stop_loss:.5f}",
                "timeInForce": "GTC",
            }
        if take_profit:
            order_body["order"]["takeProfitOnFill"] = {
                "price": f"{take_profit:.5f}",
                "timeInForce": "GTC",
            }
        
        endpoint = f"v3/accounts/{self.cfg.account_id}/orders"
        url = f"{self.cfg.base_url}/{endpoint}"
        logger.info("Placing LIVE MARKET order on OANDA: %+d %s", units, instrument)
        resp = self._session.post(url, json=order_body, timeout=10)
        resp.raise_for_status()
        res = resp.json()
        
        fill_tx = res.get("orderFillTransaction", {})
        if fill_tx:
            res["price"] = fill_tx.get("price")
            res["id"] = fill_tx.get("id")
        else:
            res["price"] = res.get("orderCreateTransaction", {}).get("price")
            res["id"] = res.get("orderCreateTransaction", {}).get("id")
            
        return res

    def place_limit_order(
        self,
        instrument: str,
        units: int,
        price: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        time_in_force: str = "GFD",
        client_id: str | None = None,
    ) -> dict:
        """Submit a limit order to OANDA."""
        order_body: dict = {
            "order": {
                "type": "LIMIT",
                "instrument": instrument,
                "units": str(units),
                "price": f"{price:.5f}",
                "timeInForce": time_in_force,
                "positionFill": "DEFAULT",
            }
        }
        if client_id:
            order_body["order"]["clientExtensions"] = {"id": client_id}
        if stop_loss:
            order_body["order"]["stopLossOnFill"] = {
                "price": f"{stop_loss:.5f}",
                "timeInForce": "GTC",
            }
        if take_profit:
            order_body["order"]["takeProfitOnFill"] = {
                "price": f"{take_profit:.5f}",
                "timeInForce": "GTC",
            }
        
        endpoint = f"v3/accounts/{self.cfg.account_id}/orders"
        url = f"{self.cfg.base_url}/{endpoint}"
        logger.info("Placing LIVE LIMIT order on OANDA: %+d %s @ %.5f", units, instrument, price)
        resp = self._session.post(url, json=order_body, timeout=10)
        resp.raise_for_status()
        res = resp.json()
        
        create_tx = res.get("orderCreateTransaction", {})
        if create_tx:
            res["price"] = create_tx.get("price")
            res["id"] = create_tx.get("id")
            
        return res


    # ── Live price streaming ───────────────────────────────────────────────────
    def stream_prices(
        self,
        instruments: list[str] | None = None,
        max_ticks: int | None = None,
    ) -> Generator[dict, None, None]:
        """
        Async-compatible generator yielding real-time price ticks.

        Parameters
        ----------
        instruments : list of OANDA instrument codes. Defaults to configured instruments.
        max_ticks   : stop after this many ticks (useful for testing). None = infinite.

        Yields
        ------
        dict with keys: instrument, time, bid, ask, mid, tradeable

        Example
        -------
        >>> for tick in conn.stream_prices(["EUR_USD"], max_ticks=100):
        ...     print(f"{tick['instrument']}: {tick['mid']:.5f}")
        """
        if instruments is None:
            from maestro.config.config import instrument_ids
            instruments = instrument_ids()

        instr_str = ",".join(instruments)
        url = (
            f"{self.cfg.base_url}/v3/accounts/{self.cfg.account_id}"
            f"/pricing/stream?instruments={instr_str}"
        )
        headers = {
            "Authorization": f"Bearer {self.cfg.token}",
            "Accept-Datetime-Format": "RFC3339",
        }

        tick_count = 0
        logger.info("Starting price stream for: %s", instr_str)

        while True:
            try:
                with requests.get(url, headers=headers, stream=True, timeout=30) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        import json
                        msg = json.loads(line)
                        if msg.get("type") == "PRICE":
                            tick = _parse_tick(msg)
                            yield tick
                            tick_count += 1
                            if max_ticks and tick_count >= max_ticks:
                                logger.info("Reached max_ticks=%d — stopping stream", max_ticks)
                                return

            except (requests.RequestException, ConnectionError) as exc:
                logger.warning("Stream interrupted: %s — reconnecting in 5s", exc)
                time.sleep(5)


# ── Order Management (used by Execution Agent) ────────────────────────────────
class OANDAOrderManager:
    """
    Thin wrapper for order creation and management.
    Called exclusively by the Execution Agent — not directly by other agents.
    """

    def __init__(self, connector: OANDAConnector | None = None) -> None:
        self._conn = connector or OANDAConnector()

    def create_market_order(
        self,
        instrument: str,
        units: int,          # positive = long, negative = short
        stop_loss_pips: float | None = None,
        take_profit_pips: float | None = None,
        client_order_id: str | None = None,
    ) -> dict:
        """
        Submit a market order with optional SL/TP.

        Returns the full OANDA order response dict including fill price.
        """
        order_body: dict = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(units),
                "timeInForce": "FOK",    # Fill-or-Kill
                "positionFill": "DEFAULT",
            }
        }

        if client_order_id:
            order_body["order"]["clientExtensions"] = {"id": client_order_id}

        # SL / TP — resolved after fill price known (simplified here)
        # In production: use trailingStopLossOnFill / takeProfitOnFill
        if stop_loss_pips:
            order_body["order"]["stopLossOnFill"] = {
                "distance": str(round(stop_loss_pips * 0.0001, 5))
            }
        if take_profit_pips:
            order_body["order"]["takeProfitOnFill"] = {
                "distance": str(round(take_profit_pips * 0.0001, 5))
            }

        endpoint = f"v3/accounts/{self._conn.cfg.account_id}/orders"
        url = f"{self._conn.cfg.base_url}/{endpoint}"

        resp = self._conn._session.post(url, json=order_body, timeout=10)
        resp.raise_for_status()
        result = resp.json()

        logger.info(
            "Order submitted: %s %+d units | Fill: %s",
            instrument, units,
            result.get("orderFillTransaction", {}).get("price", "pending")
        )
        return result

    def close_position(self, instrument: str, long_units: str = "ALL", short_units: str = "NONE") -> dict:
        """Close all or part of an open position."""
        url = (
            f"{self._conn.cfg.base_url}/v3/accounts"
            f"/{self._conn.cfg.account_id}/positions/{instrument}/close"
        )
        body = {"longUnits": long_units, "shortUnits": short_units}
        resp = self._conn._session.put(url, json=body, timeout=10)
        resp.raise_for_status()
        logger.info("Closed position: %s (long=%s, short=%s)", instrument, long_units, short_units)
        return resp.json()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _parse_dt(dt: str | datetime) -> datetime:
    """Parse string or datetime → UTC-aware datetime."""
    if isinstance(dt, datetime):
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return pd.Timestamp(dt, tz="UTC").to_pydatetime()


def _parse_candle(raw: dict, price: str = "M") -> OANDACandle:
    """Convert raw OANDA candle dict to typed OANDACandle."""
    price_key = {"M": "mid", "B": "bid", "A": "ask"}.get(price, "mid")
    ohlc = raw.get(price_key, {})
    return OANDACandle(
        time=pd.Timestamp(raw["time"], tz="UTC").to_pydatetime(),
        open=float(ohlc.get("o", 0)),
        high=float(ohlc.get("h", 0)),
        low=float(ohlc.get("l", 0)),
        close=float(ohlc.get("c", 0)),
        volume=int(raw.get("volume", 0)),
        complete=raw.get("complete", True),
    )


def _parse_tick(raw: dict) -> dict:
    """Parse a streaming PRICE message into a clean tick dict."""
    bids = raw.get("bids", [{}])
    asks = raw.get("asks", [{}])
    bid = float(bids[0].get("price", 0)) if bids else 0.0
    ask = float(asks[0].get("price", 0)) if asks else 0.0
    return {
        "instrument": raw.get("instrument"),
        "time":       pd.Timestamp(raw.get("time"), tz="UTC"),
        "bid":        bid,
        "ask":        ask,
        "mid":        round((bid + ask) / 2, 5),
        "spread":     round(ask - bid, 5),
        "tradeable":  raw.get("tradeable", True),
    }


def _candles_to_dataframe(candles: list[OANDACandle]) -> pd.DataFrame:
    """Convert list of OANDACandle → clean pd.DataFrame."""
    records = [
        {
            "open":   c.open,
            "high":   c.high,
            "low":    c.low,
            "close":  c.close,
            "volume": c.volume,
        }
        for c in candles
    ]
    idx = pd.DatetimeIndex([c.time for c in candles], name="datetime")
    df = pd.DataFrame(records, index=idx)
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    return df


# ── Custom Exceptions ─────────────────────────────────────────────────────────
class OANDAError(Exception):
    """Raised when OANDA API calls fail after all retries."""
    pass
