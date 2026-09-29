"""Causal, fill-driven accounting for a single instrument backtest.

A decision stamped at bar ``t`` is not allowed to change the portfolio until
the open of bar ``t+1``.  This prevents the strategy from earning the return
that was already used to form its decision.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from maestro.agents.risk.cost_model import TransactionCostModel, _get_session
from maestro.config.instruments import InstrumentSpec, get_instrument_spec


@dataclass(frozen=True)
class LedgerConfig:
    initial_equity: float = 10_000.0
    max_drawdown: float = 0.15
    force_close: bool = True


class CausalPortfolioLedger:
    """Turn target-position decisions into next-bar fills and portfolio P&L."""

    def __init__(self, instrument: str, config: LedgerConfig | None = None) -> None:
        self.instrument = instrument
        self.spec = get_instrument_spec(instrument)
        self.cfg = config or LedgerConfig()
        self.cost_model = TransactionCostModel(instrument)

    def run(self, decisions: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
        if not {"open", "close"}.issubset(bars.columns):
            raise ValueError("bars must contain open and close columns")
        if not bars.index.is_monotonic_increasing or not bars.index.is_unique:
            raise ValueError("bars index must be unique and increasing")

        aligned = decisions.reindex(bars.index)
        equity = float(self.cfg.initial_equity)
        peak_equity = equity
        position = 0
        previous_close: float | None = None
        rows: list[dict] = []

        for i, (ts, bar) in enumerate(bars.iterrows()):
            open_price = float(bar["open"])
            close_price = float(bar["close"])
            if not np.isfinite(open_price) or not np.isfinite(close_price) or open_price <= 0:
                raise ValueError(f"invalid price at {ts}")

            # Position held overnight earns the close-to-open gap before today's
            # rebalance. The new target then earns only today's open-to-close move.
            start_equity = equity
            gap_pnl = 0.0 if previous_close is None else self._pnl(position, open_price - previous_close, open_price)
            equity_at_open = equity + gap_pnl
            peak_equity = max(peak_equity, equity_at_open)

            decision_ts = bars.index[i - 1] if i > 0 else pd.NaT
            decision = aligned.iloc[i - 1] if i > 0 else None
            desired = self._target_units(decision) if decision is not None else 0

            drawdown_before = max(0.0, (peak_equity - equity_at_open) / peak_equity)
            if drawdown_before >= self.cfg.max_drawdown:
                desired = 0

            delta = desired - position
            cost = self._transaction_cost(delta, open_price, ts)
            position = desired
            intrabar_pnl = self._pnl(position, close_price - open_price, close_price)
            gross_pnl = gap_pnl + intrabar_pnl
            net_pnl = gross_pnl - cost
            equity = equity_at_open + intrabar_pnl - cost
            peak_equity = max(peak_equity, equity)
            drawdown = max(0.0, (peak_equity - equity) / peak_equity)

            rows.append({
                "timestamp": ts,
                "decision_timestamp": decision_ts,
                "instrument": self.instrument,
                "target_units": desired,
                "units": position,
                "delta_units": delta,
                "fill_price": open_price if delta else np.nan,
                "executed": bool(delta),
                "gross_pnl": gross_pnl,
                "transaction_cost": cost,
                "net_pnl": net_pnl,
                "start_equity": start_equity,
                "gross_return": gross_pnl / start_equity if start_equity else 0.0,
                "net_return": net_pnl / start_equity if start_equity else 0.0,
                "equity": equity,
                "drawdown": drawdown,
                "regime": self._value(decision, "regime", np.nan),
                "signal": self._value(decision, "final_signal", self._value(decision, "signal", 0)),
            })
            previous_close = close_price

        if rows and self.cfg.force_close and position:
            final = rows[-1]
            close_cost = self._transaction_cost(-position, float(bars["close"].iloc[-1]), bars.index[-1])
            final["delta_units"] -= position
            final["transaction_cost"] += close_cost
            final["net_pnl"] -= close_cost
            final["net_return"] = final["net_pnl"] / final["start_equity"] if final["start_equity"] else 0.0
            final["equity"] -= close_cost
            final["drawdown"] = max(0.0, (peak_equity - final["equity"]) / peak_equity)
            final["executed"] = True
            final["units"] = 0

        return pd.DataFrame(rows).set_index("timestamp")

    @staticmethod
    def _value(row: pd.Series | None, key: str, default):
        if row is None or key not in row or pd.isna(row[key]):
            return default
        return row[key]

    def _target_units(self, row: pd.Series | None) -> int:
        if row is None:
            return 0
        compliant = bool(self._value(row, "compliant", True))
        action = str(self._value(row, "final_action", self._value(row, "action", "flat")))
        units = int(self._value(row, "final_units", self._value(row, "units", 0)))
        if not compliant or action not in {"trade", "reduce"}:
            return 0
        return int(np.clip(units, -self.spec.max_units, self.spec.max_units))

    def _pnl(self, units: int, price_change: float, current_price: float) -> float:
        quote_pnl = units * price_change
        if self.spec.quote_currency == "USD":
            return quote_pnl
        if self.spec.base_currency == "USD":
            return quote_pnl / current_price
        raise ValueError(
            f"{self.instrument} needs a point-in-time {self.spec.quote_currency}/USD conversion rate"
        )

    def _transaction_cost(self, delta_units: int, price: float, ts: pd.Timestamp) -> float:
        if delta_units == 0:
            return 0.0
        estimate = self.cost_model.estimate(delta_units, price, session=_get_session(ts))
        home_notional = abs(delta_units) * price if self.spec.quote_currency == "USD" else abs(delta_units)
        return estimate.total_cost * home_notional
