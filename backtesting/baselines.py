"""
maestro/backtesting/baselines.py
================================
Re-implementation of the MSc (2022/23) strategies inside the MAESTRO
evaluation harness, so they can be compared with MAESTRO on identical terms:
same data, same walk-forward splits, same cost assumptions, same metric code.

The original MSc notebook is lost. Strategies are rebuilt from the report
("A Comparative Study of Machine Learning Approaches for Predictive Modeling
in Algorithmic Trading", Coventry University, 2022/23). Where the report does
not state a parameter, a documented default is used — see STRATEGY_NOTES.

Timing convention (no look-ahead)
---------------------------------
position[t] is decided with information up to and including bar t's close,
and is held during bar t+1 (it earns the return from close[t] to close[t+1]).
Every window starts flat and is forced flat on its last bar, so every trade
pays exactly one round-trip cost.

Metric definitions (shared by every strategy, including MAESTRO later)
---------------------------------------------------------------------
hit_directional : share of bars, while in a position and the price moved,
                  where the position's sign matched the move.
hit_msc_style   : the MSc report's definition — winning bars / all bars in a
                  position (bars where the price did not move count as misses).
win_rate        : share of trades whose PnL after costs is positive.
sharpe          : daily-aggregated net returns, annualised with sqrt(252).

Usage
-----
    python -m maestro.backtesting.baselines                  # EUR_USD, all splits
    python -m maestro.backtesting.baselines --max-splits 3
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv as _load_dotenv
    _here = Path(__file__).resolve()
    for _d in [_here.parent.parent, _here.parent.parent.parent, Path.cwd()]:
        if (_d / ".env").exists():
            _load_dotenv(str(_d / ".env"), override=True)
            break
except ImportError:
    pass

DATA_DIR   = Path(os.environ.get("MAESTRO_DATA_DIR",   str(Path.home() / "maestro_data")))
OUTPUT_DIR = Path(os.environ.get("MAESTRO_OUTPUT_DIR", str(Path.home() / "maestro_outputs")))

PIP_SIZE = {"EUR_USD": 1e-4, "GBP_USD": 1e-4}


# ── Strategies ────────────────────────────────────────────────────────────────
# Signature: (close, returns, train_idx, test_idx, split_id) -> positions on test_idx
StrategyFn = Callable[[pd.Series, pd.Series, pd.Index, pd.Index, int], pd.Series]

STRATEGY_NOTES = {
    "buy_hold":       "Benchmark: always long.",
    "random":         "Sanity check: seeded random +/-1 each bar. Should land near 50% "
                      "directional hit and lose roughly the full cost of every trade.",
    "sma_20_200":     "MSc 4.1.1: SMA crossover (20, 200). The MSc ran it on DAILY bars; "
                      "here it runs on M5 bars inside the same harness as MAESTRO.",
    "contrarian_3":   "MSc 4.3.2: short when the rolling mean of returns is positive, long "
                      "when negative. Window not stated in the report; 3 used "
                      "(tpqoa / Hilpisch default).",
    "bollinger_20_2": "MSc 4.3.4: mean reversion. Short above the upper band, long below the "
                      "lower band, exit when price crosses the SMA. Window/width not stated "
                      "in the report; 20 / 2 std used.",
    "linreg_lag1":    "MSc 4.2.1: OLS (no intercept, as in the MSc) of the next return on the "
                      "latest return; trade the sign of the prediction. Fit on train only.",
    "linreg_lag5":    "MSc 4.2.1: as linreg_lag1 with the 5 latest returns.",
    "logreg_lag5":    "MSc 4.3.5: logistic regression on the 5 latest returns predicting "
                      "up/down; features standardised on train. Fit on train only.",
}


def _lags(returns: pd.Series, n: int) -> pd.DataFrame:
    """Features at bar t: r_t, r_{t-1}, ..., r_{t-n+1} — all known at t's close."""
    return pd.concat({f"lag_{k}": returns.shift(k) for k in range(n)}, axis=1)


def _train_xy(returns: pd.Series, n_lags: int, train_idx: pd.Index) -> pd.DataFrame:
    """Training rows: lag features at t, target r_{t+1}, strictly inside train_idx."""
    X = _lags(returns, n_lags).loc[train_idx]
    y = returns.shift(-1).loc[train_idx].rename("y")
    # The last train row's target is the first bar after train_end (in the embargo
    # gap) — drop it so no data outside the train window is used.
    return X.join(y).iloc[:-1].dropna()


def buy_hold(close, returns, train_idx, test_idx, split_id):
    return pd.Series(1.0, index=test_idx)


def random_signal(close, returns, train_idx, test_idx, split_id):
    rng = np.random.default_rng(1000 + split_id)
    return pd.Series(rng.choice([-1.0, 1.0], size=len(test_idx)), index=test_idx)


def sma_crossover(short: int = 20, long: int = 200) -> StrategyFn:
    def strat(close, returns, train_idx, test_idx, split_id):
        s, l = close.rolling(short).mean(), close.rolling(long).mean()
        pos = pd.Series(np.where(s > l, 1.0, -1.0), index=close.index)
        return pos.where(l.notna(), 0.0).loc[test_idx]
    return strat


def contrarian(window: int = 3) -> StrategyFn:
    def strat(close, returns, train_idx, test_idx, split_id):
        return (-np.sign(returns.rolling(window).mean())).fillna(0.0).loc[test_idx]
    return strat


def bollinger_reversion(window: int = 20, n_std: float = 2.0) -> StrategyFn:
    def strat(close, returns, train_idx, test_idx, split_id):
        sma = close.rolling(window).mean()
        sd = close.rolling(window).std()
        dist = close - sma
        pos = pd.Series(np.nan, index=close.index)
        pos[close > sma + n_std * sd] = -1.0
        pos[close < sma - n_std * sd] = 1.0
        pos[(dist * dist.shift(1)) < 0] = 0.0          # crossed the SMA -> exit
        return pos.ffill().fillna(0.0).loc[test_idx]
    return strat


def linreg(n_lags: int) -> StrategyFn:
    def strat(close, returns, train_idx, test_idx, split_id):
        tr = _train_xy(returns, n_lags, train_idx)
        feats = [c for c in tr.columns if c != "y"]
        coef = np.linalg.lstsq(tr[feats].values, tr["y"].values, rcond=None)[0]
        X_te = _lags(returns, n_lags).loc[test_idx].fillna(0.0)
        return pd.Series(np.sign(X_te.values @ coef), index=test_idx)
    return strat


def logreg(n_lags: int) -> StrategyFn:
    def strat(close, returns, train_idx, test_idx, split_id):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        tr = _train_xy(returns, n_lags, train_idx)
        tr = tr[tr["y"] != 0]
        feats = [c for c in tr.columns if c != "y"]
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(tr[feats], (tr["y"] > 0).astype(int))
        X_te = _lags(returns, n_lags).loc[test_idx].fillna(0.0)
        p_up = model.predict_proba(X_te[feats])[:, 1]
        return pd.Series(np.where(p_up > 0.5, 1.0, -1.0), index=test_idx)
    return strat


STRATEGIES: dict[str, StrategyFn] = {
    "buy_hold":       buy_hold,
    "random":         random_signal,
    "sma_20_200":     sma_crossover(20, 200),
    "contrarian_3":   contrarian(3),
    "bollinger_20_2": bollinger_reversion(20, 2.0),
    "linreg_lag1":    linreg(1),
    "linreg_lag5":    linreg(5),
    "logreg_lag5":    logreg(5),
}


# ── Evaluation (shared by every system) ───────────────────────────────────────
@dataclass
class WindowResult:
    bars:   pd.DataFrame   # per bar: held, r_log, pnl_pips, turnover, close
    trades: pd.DataFrame   # per trade: direction, gross_pips, n_bars


def simulate(positions: pd.Series, close: pd.Series, pip: float) -> WindowResult:
    """Run a position series over one test window. Pure bookkeeping, no costs yet."""
    close = close.loc[positions.index]
    pos = positions.astype(float).fillna(0.0).copy()
    pos.iloc[-1] = 0.0                                   # flat at window end
    held = pos.shift(1).fillna(0.0)                      # exposure during bar t
    bars = pd.DataFrame({
        "held":     held,
        "r_log":    np.log(close / close.shift(1)).fillna(0.0),
        "pnl_pips": held * close.diff().fillna(0.0) / pip,
        "turnover": (pos - pos.shift(1).fillna(0.0)).abs(),
        "close":    close,
    })
    # A trade is a run in one direction; its size may change along the way (a risk layer
    # scaling a position), which moves turnover and therefore cost, not the trade count.
    direction = np.sign(held)
    run_id = (direction != direction.shift(1)).cumsum()
    in_mkt = held != 0
    trades = (bars[in_mkt]
              .assign(size=held.abs())
              .groupby(run_id[in_mkt])
              .agg(direction=("held", lambda h: float(np.sign(h.iloc[0]))),
                   gross_pips=("pnl_pips", "sum"), n_bars=("held", "size"), size=("size", "mean"))
              .reset_index(drop=True))
    return WindowResult(bars, trades)


def _net_log(bars: pd.DataFrame, cost_pips: float, pip: float) -> pd.Series:
    """Per-bar net log return: position return minus cost booked when the position changes."""
    cost_log = bars["turnover"] * (cost_pips / 2.0) * pip / bars["close"]
    return bars["held"] * bars["r_log"] - cost_log


def daily_net_returns(results: list[WindowResult], cost_pips: float, pip: float) -> pd.Series:
    """Daily net log returns across one or more windows (for equity curves)."""
    net_log = _net_log(pd.concat([r.bars for r in results]), cost_pips, pip)
    return net_log.groupby(net_log.index.normalize()).sum()


def summarise(results: list[WindowResult], cost_pips: float, pip: float) -> dict:
    """Pool one or more windows and compute every metric at a given round-trip cost."""
    bars = pd.concat([r.bars for r in results])
    trades = pd.concat([r.trades for r in results], ignore_index=True)
    n_trades = len(trades)
    in_mkt = bars["held"] != 0
    moved = bars["r_log"] != 0

    both = in_mkt & moved
    hit_dir = float((np.sign(bars["held"]) == np.sign(bars["r_log"]))[both].mean()) if both.any() else np.nan
    hit_msc = float(((bars["held"] * bars["r_log"]) > 0)[in_mkt].mean()) if in_mkt.any() else np.nan

    # Costs follow turnover: half a round trip per unit of position change. For unit positions
    # this is exactly one round trip per trade; for scaled positions it charges what was traded.
    net_pips_total = float(trades["gross_pips"].sum() - bars["turnover"].sum() * cost_pips / 2.0)
    net_trade_pips = trades["gross_pips"] - cost_pips * trades["size"]

    net_log = _net_log(bars, cost_pips, pip)
    daily = net_log.groupby(net_log.index.normalize()).sum()
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    equity = np.exp(net_log.cumsum())
    max_dd = float((1.0 - equity / equity.cummax()).max())

    return {
        "n_trades":           n_trades,
        "exposure":           float(in_mkt.mean()),
        "trades_per_day":     n_trades / max(len(daily), 1),
        "hit_directional":    hit_dir,
        "hit_msc_style":      hit_msc,
        "win_rate":           float((net_trade_pips > 0).mean()) if n_trades else np.nan,
        "net_pips_total":     net_pips_total,
        "net_pips_per_trade": net_pips_total / n_trades if n_trades else np.nan,
        "sharpe":             sharpe,
        "max_drawdown":       max_dd,
    }


# ── Walk-forward runner ───────────────────────────────────────────────────────
def load_close(instrument: str, granularity: str = "M5") -> pd.Series:
    """Read the feature parquet directly (avoids building credentialed connectors).

    The sealed holdout (maestro.data.holdout) is dropped unless unlocked.
    """
    from maestro.data.holdout import seal
    return seal(pd.read_parquet(feature_file(instrument, granularity), columns=["close"])["close"].sort_index())


def feature_file(instrument: str, granularity: str = "M5") -> Path:
    """The feature table for one bar length. Longer bars never fall back to the
    5-minute file: silently scoring the wrong bars would be worse than failing."""
    names = [f"{instrument}_{granularity}_features.parquet"]
    if granularity == "M5":
        names.append(f"{instrument}_features.parquet")
    for name in names:
        if (DATA_DIR / name).exists():
            return DATA_DIR / name
    raise FileNotFoundError(f"No {granularity} feature parquet for {instrument} in {DATA_DIR}; "
                            f"run: python -m maestro.data.pipeline.store sync --granularity {granularity}")


def make_wfa(close: pd.Series):
    """Identical split configuration to backtest_engine and modal_edge."""
    from maestro.data.validation.wfa import WalkForwardEngine
    start, end = close.index[0], close.index[-1]
    return WalkForwardEngine(
        train_start=str((start + pd.DateOffset(months=6)).date()),
        wfa_start=str((start + pd.DateOffset(months=12)).date()),
        wfa_end=str(end.date()),
    )


@dataclass
class RefitBlock:
    """A group of consecutive monthly test windows that share one trained model."""
    block_id: int
    train_idx: pd.Index
    splits: list          # WFASplit objects, one per monthly test window

    @property
    def test_idx(self) -> pd.Index:
        return self.splits[0].test_idx.append([s.test_idx for s in self.splits[1:]])


def design_tag(refit_months: int, train_months: int | None) -> str:
    """Folder-safe name of a retraining design, e.g. 'refit1_expanding' or 'refit3_roll12'."""
    return f"refit{refit_months}_" + ("expanding" if train_months is None else f"roll{train_months}")


def refit_plan(close: pd.Series, refit_months: int = 1, train_months: int | None = None,
               max_splits: int | None = None, min_train_bars: int = 0) -> list[RefitBlock]:
    """
    Group the monthly walk-forward test windows into retraining blocks.

    refit_months=1, train_months=None  -> the original design: retrain before every
                                          month on all data so far (expanding window).
    refit_months=3, train_months=12    -> retrain every quarter on the latest 12 months.
                                          Every month is still traded blind.

    The training window always ends `embargo_days` before the block's first test month,
    so months two and three of a block are even further out of sample.

    min_train_bars drops leading blocks whose training window is shorter, so on long bars
    (4-hour, daily) the design starts once there is enough history. Every strategy is
    scored on the same remaining months.
    """
    splits = [s for s in make_wfa(close).splits(close.to_frame()) if len(s.train_idx) >= 500]
    if max_splits is not None:
        splits = [s for s in splits if s.split_id < max_splits]
    blocks = []
    for b, start in enumerate(range(0, len(splits), refit_months)):
        group = splits[start:start + refit_months]
        train_idx = group[0].train_idx
        if train_months is not None:
            cutoff = group[0].test_start - pd.DateOffset(months=train_months)
            train_idx = train_idx[train_idx >= cutoff]
        blocks.append(RefitBlock(b, train_idx, group))
    while blocks and len(blocks[0].train_idx) < min_train_bars:
        blocks.pop(0)
    return [RefitBlock(i, b.train_idx, b.splits) for i, b in enumerate(blocks)]


def default_cost_scenarios(instrument: str) -> dict[str, float]:
    """Round-trip costs in pips. 'spread' reuses the shared cost model's typical spread."""
    from maestro.agents.risk.cost_model import INSTRUMENT_COSTS
    spread = INSTRUMENT_COSTS[instrument]["typical_spread_pips"]
    return {"gross": 0.0, "spread": spread, "spread+slippage": spread + 0.7}


def run(instrument: str = "EUR_USD", max_splits: int | None = None,
        strategies: list[str] | None = None, out_dir: Path | None = None,
        refit_months: int = 1, train_months: int | None = None,
        external: dict[str, pd.Series] | None = None,
        close: pd.Series | None = None, min_train_bars: int = 0,
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Score strategies over the walk-forward test months.

    external : optional {name: position series} produced elsewhere (e.g. MAESTRO),
               covering the test bars. They are scored with exactly the same
               simulate/summarise code as the built-in strategies.
    close    : optional price series to score on instead of the stored data
               (used by the power test, which plants a known edge in prices).
    """
    pip = PIP_SIZE[instrument]
    close = load_close(instrument) if close is None else close
    returns = np.log(close / close.shift(1))
    names = list(strategies or STRATEGIES)
    external = external or {}
    all_names = names + list(external)
    costs = default_cost_scenarios(instrument)
    blocks = refit_plan(close, refit_months, train_months, max_splits, min_train_bars)

    windows: dict[str, list[WindowResult]] = {n: [] for n in all_names}
    split_rows = []
    for block in blocks:
        # Each built-in strategy is fitted once per block, then traded month by month.
        block_pos = {n: STRATEGIES[n](close, returns, block.train_idx, block.test_idx, block.block_id)
                     for n in names}
        for split in block.splits:
            for name in all_names:
                src = block_pos[name] if name in block_pos else external[name]
                pos = src.reindex(split.test_idx).fillna(0.0)
                res = simulate(pos, close, pip)
                windows[name].append(res)
                s = summarise([res], cost_pips=costs["spread"], pip=pip)
                split_rows.append({"split_id": split.split_id, "block_id": block.block_id,
                                   "strategy": name, "test_start": split.test_start,
                                   "test_end": split.test_end, **s})
        logger.info("block %d done: %d month(s) from %s, trained on %d bars",
                    block.block_id, len(block.splits), block.splits[0].test_start.date(),
                    len(block.train_idx))

    pooled_rows = []
    per_split = pd.DataFrame(split_rows)
    names = all_names
    for name in names:
        for label, c in costs.items():
            s = summarise(windows[name], cost_pips=c, pip=pip)
            pooled_rows.append({"strategy": name, "cost": label, "cost_pips": c, **s})
        net = per_split.loc[per_split["strategy"] == name, "net_pips_total"]
        for row in pooled_rows[-len(costs):]:
            row["splits_profitable_at_spread"] = f"{int((net > 0).sum())}/{len(net)}"
    pooled = pd.DataFrame(pooled_rows)

    out = out_dir or OUTPUT_DIR / "baselines" / design_tag(refit_months, train_months)
    out.mkdir(parents=True, exist_ok=True)
    (out / "design.json").write_text(json.dumps({
        "instrument": instrument, "refit_months": refit_months, "train_months": train_months,
        "blocks": len(blocks), "months": int(per_split["split_id"].nunique()),
    }, indent=2), encoding="utf-8")
    pooled.to_csv(out / f"{instrument}_baselines_pooled.csv", index=False)
    per_split.to_csv(out / f"{instrument}_baselines_per_split.csv", index=False)
    daily = pd.DataFrame({
        f"{name}|{label}": daily_net_returns(windows[name], costs[label], pip)
        for name in names for label in ("gross", "spread")
    })
    daily.to_csv(out / f"{instrument}_baselines_daily.csv")
    logger.info("Saved results -> %s", out)
    return pooled, per_split


def _print_report(pooled: pd.DataFrame, n_splits: int) -> None:
    view = pooled[pooled["cost"] == "spread"].set_index("strategy")
    gross = pooled[pooled["cost"] == "gross"].set_index("strategy")["sharpe"]
    print(f"\nMSc strategies rebuilt in the MAESTRO harness — pooled over {n_splits} splits")
    print(f"(net figures at the 'spread' cost of {view['cost_pips'].iloc[0]:.1f} pips round trip)\n")
    header = (f"{'strategy':<16}{'trades':>8}{'hit dir':>9}{'hit MSc':>9}{'win':>7}"
              f"{'pips/trade':>11}{'net pips':>11}{'Sharpe':>8}{'gross Sh':>9}{'maxDD':>8}{'splits+':>9}")
    print(header)
    print("-" * len(header))
    for name, r in view.iterrows():
        print(f"{name:<16}{r['n_trades']:>8.0f}{r['hit_directional']*100:>8.1f}%{r['hit_msc_style']*100:>8.1f}%"
              f"{r['win_rate']*100:>6.1f}%{r['net_pips_per_trade']:>11.2f}{r['net_pips_total']:>11.0f}"
              f"{r['sharpe']:>8.2f}{gross[name]:>9.2f}{r['max_drawdown']*100:>7.1f}%"
              f"{r['splits_profitable_at_spread']:>9}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description="MSc baselines in the MAESTRO harness")
    p.add_argument("--instrument", default="EUR_USD")
    p.add_argument("--max-splits", type=int, default=None)
    p.add_argument("--strategies", nargs="+", default=None, choices=list(STRATEGIES))
    p.add_argument("--refit-months", type=int, default=1,
                   help="retrain every N test months (default 1)")
    p.add_argument("--train-months", type=int, default=None,
                   help="rolling training window in months (default: all history)")
    p.add_argument("--holdout", action="store_true",
                   help="unlock the sealed holdout (maestro.data.holdout): final confirmation run only")
    args = p.parse_args()
    if args.holdout:
        from maestro.data.holdout import unlock
        unlock()
    pooled, per_split = run(args.instrument, args.max_splits, args.strategies,
                            refit_months=args.refit_months, train_months=args.train_months)
    _print_report(pooled, per_split["split_id"].nunique())


if __name__ == "__main__":
    main()
