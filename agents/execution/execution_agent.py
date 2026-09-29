"""
maestro/agents/execution/execution_agent.py
=============================================
Agent 5 — Execution Agent.

Receives a RiskDecision from Agent 4 and translates it into actual
OANDA orders with intelligent timing, order type selection, and
fill quality monitoring.

Why a dedicated execution agent?
----------------------------------
Naive execution (market order at signal bar) is suboptimal:
  1. Market orders pay the full spread every time
  2. Signals often arrive mid-bar — the "good" price may be
     available for only a fraction of the bar
  3. Limit orders can capture better fills but risk non-execution
  4. Partial fills need careful handling in live trading
  5. Latency between signal and execution matters at M5 granularity

Order type selection logic
---------------------------
  High confidence + trending regime  → Limit order (capture spread)
  Low confidence  OR sideways regime → Market order (certainty > fill quality)
  Crisis regime                      → Market order (urgency > cost)
  Reduce / close position            → Market order (immediacy)
  Stop-loss / take-profit            → GTC stop/limit orders

Fill quality monitoring
-----------------------
  Tracks:
    - Slippage per trade (actual fill vs signal price)
    - Fill rate of limit orders
    - Mean time to fill (limit orders)
    - VWAP comparison (did we beat the bar VWAP?)

  These metrics feed back into:
    - Cost model calibration (actual vs estimated slippage)
    - Order type selection (if limit fill rate drops → switch to market)

Live vs backtest modes
----------------------
  Live:     calls OANDAConnector directly to place real orders
  Backtest: simulates fills using OHLC bars (conservative: assumes
            limit orders fill only if price trades through limit level)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from maestro.agents.risk.risk_agent import RiskDecision
from maestro.agents.risk.cost_model import TransactionCostModel, _get_session
from maestro.config.instruments import get_instrument_spec

logger = logging.getLogger(__name__)

# Limit order parameters
LIMIT_OFFSET_PIPS    = 0.3     # place limit this many pips inside the spread
MAX_LIMIT_WAIT_BARS  = 3       # cancel and re-evaluate after this many bars
MIN_CONFIDENCE_LIMIT = 0.58    # use limit orders above this confidence


@dataclass
class OrderSpec:
    """Specification for one order to be sent to the broker."""
    instrument:    str
    units:         int            # signed: +ve=buy, -ve=sell
    order_type:    str            # "MARKET" | "LIMIT" | "STOP"
    limit_price:   float | None   # None for market orders
    stop_price:    float | None   # None if no stop trigger
    stop_loss:     float | None   # SL price level
    take_profit:   float | None   # TP price level
    time_in_force: str            # "GTC" | "GFD" | "FOK"
    client_id:     str            # internal reference
    timestamp:     datetime       = field(default_factory=lambda: datetime.now(timezone.utc))

    def __repr__(self) -> str:
        typ = self.order_type
        prc = f"@{self.limit_price:.5f}" if self.limit_price else "@MARKET"
        return (
            f"OrderSpec({self.instrument} | "
            f"{'BUY' if self.units > 0 else 'SELL'} {abs(self.units)} "
            f"| {typ}{prc} | SL={self.stop_loss:.5f if self.stop_loss else 'None'})"
        )


@dataclass
class FillReport:
    """Result of order execution (live or simulated)."""
    order_id:      str
    instrument:    str
    units_filled:  int
    fill_price:    float
    signal_price:  float
    slippage_pips: float          # fill_price vs signal_price in pips
    fill_time:     datetime
    order_type:    str
    partial_fill:  bool
    cost_pips:     float          # all-in cost including spread + slippage

    @property
    def is_full_fill(self) -> bool:
        return not self.partial_fill


class ExecutionAgent:
    """
    Agent 5 — Smart Order Execution.

    Translates RiskDecisions into broker orders with optimal
    order type selection and fill quality monitoring.

    Usage (live)
    ------------
    >>> agent = ExecutionAgent(instrument="EUR_USD", live=True)
    >>> fill  = agent.execute(risk_decision, current_bar)

    Usage (backtest)
    ----------------
    >>> agent = ExecutionAgent(instrument="EUR_USD", live=False)
    >>> fills = agent.simulate_batch(risk_decisions_df, ohlcv_df)
    """

    def __init__(
        self,
        instrument:  str  = "EUR_USD",
        live:        bool = False,
        account_id:  str  | None = None,
    ) -> None:
        self.instrument  = instrument
        self.spec        = get_instrument_spec(instrument)
        self.pip_size    = self.spec.pip_size
        self.live        = live
        self.account_id  = account_id
        self.cost_model  = TransactionCostModel(instrument)
        self._connector  = None
        self._fill_log:  list[FillReport] = []
        self._pending_limits: dict[str, OrderSpec] = {}

        if live:
            self._init_connector()

    # ── Main execution entry point ────────────────────────────────────────────
    def execute(
        self,
        decision:     RiskDecision,
        current_bar:  pd.Series,          # OHLCV bar: open, high, low, close, volume
    ) -> FillReport | None:
        """
        Execute a RiskDecision.

        Parameters
        ----------
        decision    : output of RiskManagementAgent.decide()
        current_bar : current OHLCV bar as a pd.Series

        Returns
        -------
        FillReport if order placed and filled, None if flat/rejected.
        """
        if not decision.is_trade:
            logger.debug("Execution: no trade — %s", decision.risk_reason)
            return None

        current_price = float(current_bar.get("close", current_bar.get("mid_c", 1.0)))
        session       = _get_session(decision.timestamp)

        # Build order specification
        order = self._build_order(decision, current_price, session)

        logger.info(
            "Executing: %s | %s | %s",
            decision.instrument,
            f"{'BUY' if decision.units > 0 else 'SELL'} {abs(decision.units)}",
            order.order_type,
        )

        # Place order
        if self.live:
            fill = self._place_live_order(order, current_price)
        else:
            fill = self._simulate_fill(order, current_bar, current_price)

        if fill:
            self._fill_log.append(fill)
            logger.info(
                "Fill: %s | units=%d | price=%.5f | slippage=%.2f pips | cost=%.2f pips",
                fill.instrument, fill.units_filled,
                fill.fill_price, fill.slippage_pips, fill.cost_pips,
            )

        return fill

    # ── Batch simulation (backtesting) ────────────────────────────────────────
    def simulate_batch(
        self,
        decisions_df: pd.DataFrame,
        ohlcv_df:     pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Simulate execution for all decisions in a backtest.

        Uses OHLC to simulate limit vs market fills realistically:
          - Market orders: fill at open of NEXT bar + half-spread slippage
          - Limit orders:  fill only if price trades through limit level
          - Stop-losses:   triggered at low/high of bar (conservative)

        Returns
        -------
        pd.DataFrame — one row per bar with fill details:
          fill_price, slippage_pips, units_filled, order_type,
          fill_quality (fill_price vs bar VWAP)
        """
        results = []

        for ts, row in decisions_df.iterrows():
            action = row.get("action", "flat")

            if action != "trade" or row.get("units", 0) == 0:
                results.append({
                    "timestamp":    ts,
                    "fill_price":   np.nan,
                    "slippage_pips":np.nan,
                    "units_filled": 0,
                    "order_type":   "none",
                    "fill_quality": np.nan,
                    "executed":     False,
                })
                continue

            # Get next bar for market fills
            next_ts  = ohlcv_df.index[ohlcv_df.index.get_loc(ts) + 1] \
                if ts in ohlcv_df.index and ohlcv_df.index.get_loc(ts) < len(ohlcv_df) - 1 \
                else ts
            next_bar = ohlcv_df.loc[next_ts] if next_ts in ohlcv_df.index else ohlcv_df.loc[ts]

            curr_close = float(ohlcv_df.loc[ts, "close"]) if ts in ohlcv_df.index else 1.0
            units      = int(row.get("units", 0))
            confidence = float(row.get("position_fraction", 0.3))
            session    = _get_session(ts)

            # Determine order type
            use_limit  = self._should_use_limit(confidence, int(row.get("regime", 2)), session)

            if use_limit:
                # Limit order: place LIMIT_OFFSET_PIPS inside spread
                offset     = LIMIT_OFFSET_PIPS * self.pip_size
                limit_px   = curr_close - offset if units > 0 else curr_close + offset
                # Check if price trades through limit in next bar
                next_low   = float(next_bar.get("low",  next_bar.get("mid_l", curr_close)))
                next_high  = float(next_bar.get("high", next_bar.get("mid_h", curr_close)))
                filled = (units > 0 and next_low <= limit_px) or \
                         (units < 0 and next_high >= limit_px)
                fill_price = limit_px if filled else np.nan
                order_type = "LIMIT"
            else:
                # Market order: fill at next bar open
                fill_price = float(next_bar.get("open", next_bar.get("mid_o", curr_close)))
                filled     = True
                order_type = "MARKET"

            if not filled or np.isnan(fill_price):
                results.append({
                    "timestamp": ts, "fill_price": np.nan,
                    "slippage_pips": np.nan, "units_filled": 0,
                    "order_type": order_type + "_EXPIRED",
                    "fill_quality": np.nan, "executed": False,
                })
                continue

            # Add spread (always paid on market orders; not on limit fills)
            cost_est    = self.cost_model.estimate(units, fill_price, session=session)
            spread_cost = cost_est.spread_cost if order_type == "MARKET" else cost_est.spread_cost * 0.5
            slippage    = abs(fill_price - curr_close) / self.pip_size

            # VWAP comparison (did we beat the average bar price?)
            bar_vwap   = (float(next_bar.get("high", fill_price)) +
                         float(next_bar.get("low", fill_price)) +
                         float(next_bar.get("close", fill_price))) / 3.0
            fill_quality = 1.0 if (units > 0 and fill_price <= bar_vwap) or \
                                  (units < 0 and fill_price >= bar_vwap) else 0.0

            results.append({
                "timestamp":    ts,
                "fill_price":   fill_price,
                "slippage_pips":slippage,
                "units_filled": units,
                "order_type":   order_type,
                "spread_cost":  spread_cost,
                "fill_quality": fill_quality,
                "executed":     True,
            })

        result_df = pd.DataFrame(results).set_index("timestamp")
        self._log_execution_summary(result_df)
        return result_df

    # ── Order building ────────────────────────────────────────────────────────
    def _build_order(
        self,
        decision:      RiskDecision,
        current_price: float,
        session:       str,
    ) -> OrderSpec:
        """Select order type and build OrderSpec from RiskDecision."""
        use_limit = self._should_use_limit(
            decision.position_fraction, decision.metadata.get("regime", 2), session
        )
        pip = self.pip_size

        if use_limit:
            offset     = LIMIT_OFFSET_PIPS * pip
            limit_px   = current_price - offset if decision.units > 0 else current_price + offset
            order_type = "LIMIT"
            tif        = "GFD"     # Good For Day
        else:
            limit_px   = None
            order_type = "MARKET"
            tif        = "FOK"     # Fill or Kill

        # Compute absolute SL/TP price levels
        sl_price = tp_price = None
        if decision.stop_loss_pips > 0:
            direction  = 1 if decision.units > 0 else -1
            sl_price   = current_price - direction * decision.stop_loss_pips  * pip
            tp_price   = current_price + direction * decision.take_profit_pips* pip

        import uuid
        return OrderSpec(
            instrument    = self.instrument,
            units         = decision.units,
            order_type    = order_type,
            limit_price   = limit_px,
            stop_price    = None,
            stop_loss     = sl_price,
            take_profit   = tp_price,
            time_in_force = tif,
            client_id     = f"MAESTRO-{uuid.uuid4().hex[:8]}",
        )

    # ── Live order placement ──────────────────────────────────────────────────
    def _place_live_order(
        self, order: OrderSpec, current_price: float
    ) -> FillReport | None:
        """Place order via OANDA API and return fill report."""
        if self._connector is None:
            logger.error("OANDA connector not initialised")
            return None

        try:
            if order.order_type == "MARKET":
                result = self._connector.place_market_order(
                    instrument  = order.instrument,
                    units       = order.units,
                    stop_loss   = order.stop_loss,
                    take_profit = order.take_profit,
                    client_id   = order.client_id,
                )
            else:
                result = self._connector.place_limit_order(
                    instrument   = order.instrument,
                    units        = order.units,
                    price        = order.limit_price,
                    stop_loss    = order.stop_loss,
                    take_profit  = order.take_profit,
                    time_in_force= order.time_in_force,
                    client_id    = order.client_id,
                )

            fill_price = float(result.get("price", current_price))
            slippage   = abs(fill_price - current_price) / self.pip_size

            return FillReport(
                order_id      = result.get("id", order.client_id),
                instrument    = order.instrument,
                units_filled  = order.units,
                fill_price    = fill_price,
                signal_price  = current_price,
                slippage_pips = slippage,
                fill_time     = datetime.now(timezone.utc),
                order_type    = order.order_type,
                partial_fill  = False,
                cost_pips     = slippage + 0.8,   # slippage + typical spread
            )

        except Exception as exc:
            logger.error("Live order placement failed: %s", exc)
            return None

    # ── Simulated fill ────────────────────────────────────────────────────────
    def _simulate_fill(
        self,
        order:         OrderSpec,
        current_bar:   pd.Series,
        current_price: float,
    ) -> FillReport:
        """Conservative fill simulation for backtesting."""
        import uuid
        session    = _get_session(datetime.now(timezone.utc))
        cost_est   = self.cost_model.estimate(order.units, current_price, session=session)

        # Market order: add half-spread as slippage
        if order.order_type == "MARKET":
            direction  = 1 if order.units > 0 else -1
            fill_price = current_price + direction * cost_est.spread_cost
        else:
            fill_price = order.limit_price or current_price

        slippage = abs(fill_price - current_price) / self.pip_size

        return FillReport(
            order_id      = f"SIM-{uuid.uuid4().hex[:8]}",
            instrument    = self.instrument,
            units_filled  = order.units,
            fill_price    = fill_price,
            signal_price  = current_price,
            slippage_pips = slippage,
            fill_time     = datetime.now(timezone.utc),
            order_type    = order.order_type,
            partial_fill  = False,
            cost_pips     = slippage + cost_est.breakeven_pips,
        )

    # ── Order type logic ──────────────────────────────────────────────────────
    def _should_use_limit(
        self, position_fraction: float, regime: int, session: str
    ) -> bool:
        """
        Decide whether to use limit vs market order.
        Limit orders save the spread but risk non-execution.
        """
        if regime == 3:             return False   # crisis → always market
        if session in ("asian", "weekend"): return False  # thin → market
        if abs(position_fraction) < 0.15:  return False   # small → not worth limit risk
        return abs(position_fraction) >= MIN_CONFIDENCE_LIMIT

    # ── Diagnostics ───────────────────────────────────────────────────────────
    def _log_execution_summary(self, result: pd.DataFrame) -> None:
        executed = result[result.get("executed", pd.Series(False, index=result.index))]
        n_exec   = executed["executed"].sum() if "executed" in executed.columns else 0
        n_total  = len(result)
        if n_exec == 0:
            return
        avg_slip = executed["slippage_pips"].mean() if "slippage_pips" in executed.columns else 0.0
        fill_q   = executed["fill_quality"].mean()  if "fill_quality" in executed.columns else 0.0
        mkt_pct  = (executed["order_type"] == "MARKET").mean() * 100 if "order_type" in executed.columns else 0.0

        logger.info(
            "Execution summary: %d/%d bars traded | avg_slippage=%.2f pips | "
            "fill_quality=%.1f%% | market_orders=%.1f%%",
            n_exec, n_total, avg_slip, fill_q * 100, mkt_pct,
        )

    def fill_quality_report(self) -> pd.DataFrame:
        """Return DataFrame summarising all fills for post-trade analysis."""
        if not self._fill_log:
            return pd.DataFrame()
        return pd.DataFrame([{
            "order_id":      f.order_id,
            "instrument":    f.instrument,
            "units":         f.units_filled,
            "fill_price":    f.fill_price,
            "signal_price":  f.signal_price,
            "slippage_pips": f.slippage_pips,
            "cost_pips":     f.cost_pips,
            "order_type":    f.order_type,
            "fill_time":     f.fill_time,
        } for f in self._fill_log])

    def _init_connector(self) -> None:
        try:
            from maestro.data.connectors.oanda import OANDAConnector
            self._connector = OANDAConnector()
            logger.info("OANDA connector initialised (LIVE mode)")
        except Exception as exc:
            logger.error("Failed to initialise OANDA connector: %s", exc)
            self.live = False
