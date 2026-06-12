"""
maestro/agents/risk/train_risk.py
===================================
Walk-forward training and evaluation harness for Agent 4.

Unlike Agents 1-3, Agent 4 has two training modes:

  Mode A — RL Training (use_rl=True):
    Trains a PPO policy on each WFA training window,
    evaluates on the test window. Expensive but adaptive.

  Mode B — Kelly-only (use_rl=False):
    Skips RL training; uses fractional Kelly + hard gates only.
    Faster, interpretable, strong baseline.

The key metrics evaluated here that the MSc project missed entirely:
  - Net Sharpe ratio (after all transaction costs)
  - Maximum drawdown
  - Calmar ratio (annualised return / max drawdown)
  - Hit ratio (directional accuracy of actual trades placed)
  - Cost drag (how much transaction costs eroded returns)
  - CVaR₉₅ (average loss in worst 5% of days)
  - Profit factor (gross wins / gross losses)

Target performance (from PhD proposal):
  Sharpe > 1.5 | Sortino > 2.0 | Max DD < 15% | Hit ratio > 55%

Usage
-----
    python -m maestro.agents.risk.train_risk --instrument EUR_USD
    python -m maestro.agents.risk.train_risk --instrument EUR_USD --kelly-only
    python -m maestro.agents.risk.train_risk --rl-steps 200000
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from maestro.agents.regime.regime_classifier import RegimeDetectionAgent, REGIME_NAMES
from maestro.agents.risk.risk_agent import RiskManagementAgent, HARD_LIMITS
from maestro.agents.signal.signal_agent import SignalAgent
from maestro.agents.signal.patchtst import PatchTSTSignalModel, HORIZONS
from maestro.agents.signal.tft_model import TFTConfig
from maestro.agents.signal.patchtst import PatchTSTConfig
from maestro.data.pipeline.ingestion import DataPipeline
from maestro.data.validation.wfa import WalkForwardEngine

logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.environ.get("MAESTRO_MODEL_DIR", "/tmp/maestro_models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Annualisation factor for M5 data: 252 trading days × 78 bars/day
ANNUAL_FACTOR = np.sqrt(252 * 78)


def train_risk_agent(
    instrument:  str   = "EUR_USD",
    granularity: str   = "M5",
    use_rl:      bool  = True,
    rl_steps:    int   = 300_000,
    val_months:  int   = 2,
    account_equity: float = 10_000.0,
    save_models: bool  = True,
) -> pd.DataFrame:
    """
    Full walk-forward training + evaluation of Agent 4.

    Trains the complete stack (Regime → Signal → Risk) per WFA window
    and reports net performance metrics for the PhD thesis.

    Returns
    -------
    pd.DataFrame — per-split performance summary
    """
    logger.info("=" * 65)
    logger.info("MAESTRO Agent 4: Risk Management — WFA Training")
    logger.info("Instrument: %s | RL: %s | RL steps: %d | Equity: $%.0f",
                instrument, use_rl, rl_steps if use_rl else 0, account_equity)
    logger.info("=" * 65)

    # ── Load features ─────────────────────────────────────────────────────────
    pipeline    = DataPipeline()
    features_df = pipeline.load(instrument, granularity)
    logger.info("Loaded %d bars × %d features", len(features_df), len(features_df.columns))

    wfa      = WalkForwardEngine()
    n_splits = wfa.n_splits(features_df)
    logger.info("WFA: %d splits", n_splits)

    split_results = []

    for split in wfa.splits(features_df):
        logger.info("─" * 65)
        logger.info("Split %d/%d | train=%s→%s | test=%s→%s",
                    split.split_id + 1, n_splits,
                    split.train_start.date(), split.train_end.date(),
                    split.test_start.date(), split.test_end.date())

        train_df = features_df.loc[split.train_idx]
        test_df  = features_df.loc[split.test_idx]

        val_cut = split.train_end - pd.DateOffset(months=val_months)
        val_mask= train_df.index >= val_cut
        val_df  = train_df[val_mask]  if val_mask.sum() > 200  else None
        fit_df  = train_df[~val_mask] if val_df is not None    else train_df

        # ── Step 1: Regime Agent ──────────────────────────────────────────────
        logger.info("  [1/3] Fitting Regime Agent...")
        regime_agent = RegimeDetectionAgent(use_transformer=False)
        try:
            regime_agent.fit(fit_df)
            train_regimes = regime_agent.predict(fit_df)
            test_regimes  = regime_agent.predict(test_df)
        except Exception as exc:
            logger.warning("  Regime agent failed: %s — using sideways default", exc)
            train_regimes = pd.Series(2, index=fit_df.index)
            test_regimes  = pd.Series(2, index=test_df.index)

        # ── Step 2: Signal Agent ──────────────────────────────────────────────
        logger.info("  [2/3] Fitting Signal Agent...")
        signal_agent = SignalAgent(
            instrument      = instrument,
            primary_horizon = 6,
            tft_config      = TFTConfig(seq_len=120, pred_len=max(HORIZONS),
                                        max_epochs=30, patience=6),
            ptst_config     = PatchTSTConfig(seq_len=128, max_epochs=30, patience=6),
        )
        try:
            signal_agent.fit(fit_df, train_regimes,
                             val_df=val_df,
                             val_regimes=regime_agent.predict(val_df) if val_df is not None else None)
            test_signals = signal_agent.predict_batch(test_df, test_regimes)
        except Exception as exc:
            logger.error("  Signal agent failed: %s", exc)
            continue

        # Merge regime info into signals df for Risk Agent
        test_signals["regime"]     = test_regimes.reindex(test_signals.index).fillna(2).astype(int)
        test_signals["confidence"] = test_signals.get("confidence", pd.Series(0.5, index=test_signals.index))

        # ── Step 3: Risk Agent ────────────────────────────────────────────────
        logger.info("  [3/3] %s Risk Agent...",
                    "Training RL" if use_rl else "Configuring Kelly-only")

        risk_agent = RiskManagementAgent(
            instrument     = instrument,
            account_equity = account_equity,
            use_rl         = use_rl,
        )

        if use_rl:
            try:
                rl_save = str(MODEL_DIR / instrument / f"ppo_split_{split.split_id:03d}") if save_models else None
                risk_agent.train_rl(fit_df, test_signals, total_steps=rl_steps,
                                    save_path=rl_save)
            except Exception as exc:
                logger.warning("  RL training failed: %s — falling back to Kelly", exc)
                risk_agent.use_rl = False
                risk_agent.fitted = False

        # ── Evaluate on test window ───────────────────────────────────────────
        try:
            risk_decisions = risk_agent.evaluate_batch(
                test_df, test_signals,
                regime_signals=pd.DataFrame({"regime": test_regimes}, index=test_df.index)
            )
        except Exception as exc:
            logger.error("  Risk evaluation failed: %s", exc)
            continue

        # ── Compute metrics ───────────────────────────────────────────────────
        metrics = _compute_metrics(risk_decisions, test_df, split.split_id)
        split_results.append(metrics)

        logger.info(
            "  Results: Sharpe=%.3f | MaxDD=%.1f%% | HitRatio=%.1f%% | "
            "Trades=%d | CostDrag=%.2f%%",
            metrics["sharpe_net"],
            metrics["max_drawdown"] * 100,
            metrics["hit_ratio"] * 100,
            metrics["n_trades"],
            metrics["cost_drag"] * 100,
        )

        # ── Save ──────────────────────────────────────────────────────────────
        if save_models:
            risk_agent.save(MODEL_DIR / instrument / f"agent4_split_{split.split_id:03d}")
            risk_decisions.to_parquet(
                MODEL_DIR / instrument / f"agent4_decisions_split_{split.split_id:03d}.parquet"
            )

    # ── Aggregate ─────────────────────────────────────────────────────────────
    if not split_results:
        logger.error("No splits completed.")
        return pd.DataFrame()

    summary = pd.DataFrame(split_results)
    _log_final_summary(summary, instrument)

    out_path = MODEL_DIR / instrument / "agent4_wfa_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    logger.info("Summary saved → %s", out_path)
    return summary


def _compute_metrics(decisions: pd.DataFrame, features: pd.DataFrame, split_id: int) -> dict:
    """Compute full performance metrics from RiskDecision batch output."""

    trades   = decisions[decisions["action"] == "trade"] if "action" in decisions.columns else decisions
    n_total  = len(decisions)
    n_trades = len(trades)

    if n_trades == 0:
        return {
            "split_id": split_id,
            "n_bars": n_total, "n_trades": 0,
            "sharpe_net": 0.0, "sharpe_gross": 0.0,
            "sortino_net": 0.0, "calmar": 0.0,
            "max_drawdown": 0.0, "hit_ratio": 0.0,
            "profit_factor": 0.0, "cost_drag": 0.0,
            "cvar_95": 0.0, "trade_frequency": 0.0,
        }

    # Net returns (from equity curve)
    if "equity" in decisions.columns:
        eq        = decisions["equity"]
        net_rets  = eq.pct_change().fillna(0)
    else:
        net_rets  = pd.Series(0.0, index=decisions.index)

    # Gross returns (signal × bar return, no costs)
    close    = features["close"].reindex(decisions.index).fillna(method="ffill")
    bar_rets = features["log_return_1"].reindex(decisions.index).fillna(0)
    signals  = decisions.get("units", pd.Series(0, index=decisions.index))
    direction= np.sign(signals.values)
    gross_rets= pd.Series(direction * bar_rets.values, index=decisions.index)

    # ── Sharpe ────────────────────────────────────────────────────────────────
    net_mean  = net_rets.mean()
    net_std   = net_rets.std() + 1e-10
    sharpe_net= float(net_mean / net_std * ANNUAL_FACTOR)

    gross_mean  = gross_rets[gross_rets != 0].mean() if (gross_rets != 0).any() else 0
    gross_std   = gross_rets[gross_rets != 0].std()  + 1e-10 if (gross_rets != 0).any() else 1
    sharpe_gross= float(gross_mean / gross_std * ANNUAL_FACTOR)

    # ── Sortino (downside deviation only) ─────────────────────────────────────
    downside  = net_rets[net_rets < 0]
    sortino   = float(net_mean / (downside.std() + 1e-10) * ANNUAL_FACTOR) if len(downside) > 5 else 0.0

    # ── Max drawdown ──────────────────────────────────────────────────────────
    max_dd    = float(decisions["max_drawdown"].max()) if "max_drawdown" in decisions.columns else 0.0

    # ── Calmar ────────────────────────────────────────────────────────────────
    cum_net   = (1 + net_rets).prod() - 1
    ann_net   = (1 + cum_net) ** (252 * 78 / max(n_total, 1)) - 1
    calmar    = float(ann_net / max(max_dd, 0.001))

    # ── Hit ratio (directional accuracy on actual trades) ─────────────────────
    trade_signals = decisions[decisions.get("action", pd.Series("flat", index=decisions.index)) == "trade"]
    if len(trade_signals) > 0 and "units" in trade_signals.columns:
        trade_dir  = np.sign(trade_signals["units"].values)
        trade_rets = bar_rets.reindex(trade_signals.index).fillna(0).values
        hit_ratio  = float((np.sign(trade_rets) == trade_dir).mean())
    else:
        hit_ratio  = 0.5

    # ── Profit factor ─────────────────────────────────────────────────────────
    active_rets = net_rets[net_rets != 0]
    wins  = active_rets[active_rets > 0].sum()
    losses= abs(active_rets[active_rets < 0].sum()) + 1e-10
    profit_factor = float(wins / losses)

    # ── Cost drag ─────────────────────────────────────────────────────────────
    cost_drag = float(sharpe_gross - sharpe_net) / ANNUAL_FACTOR if sharpe_gross != 0 else 0.0

    # ── CVaR₉₅ ────────────────────────────────────────────────────────────────
    cutoff   = np.percentile(net_rets.values, 5)
    tail     = net_rets.values[net_rets.values <= cutoff]
    cvar_95  = float(-tail.mean()) if len(tail) > 0 else 0.0

    # ── Regime-stratified Sharpe ──────────────────────────────────────────────
    regime_sharpes = {}
    if "regime" in decisions.columns:
        for r in range(4):
            mask = decisions["regime"] == r
            if mask.sum() > 50:
                r_rets = net_rets[mask]
                r_sh   = float(r_rets.mean() / (r_rets.std() + 1e-10) * ANNUAL_FACTOR)
                regime_sharpes[f"sharpe_{REGIME_NAMES[r]}"] = r_sh

    return {
        "split_id":       split_id,
        "test_start":     decisions.index[0],
        "test_end":       decisions.index[-1],
        "n_bars":         n_total,
        "n_trades":       n_trades,
        "trade_frequency":float(n_trades / n_total),
        "sharpe_net":     sharpe_net,
        "sharpe_gross":   sharpe_gross,
        "sortino_net":    sortino,
        "calmar":         calmar,
        "max_drawdown":   max_dd,
        "hit_ratio":      hit_ratio,
        "profit_factor":  profit_factor,
        "cost_drag":      cost_drag,
        "cvar_95":        cvar_95,
        "cum_net_return": float(cum_net),
        "msc_sharpe_baseline": 0.599,    # MSc project Sharpe for direct comparison
        **regime_sharpes,
    }


def _log_final_summary(summary: pd.DataFrame, instrument: str) -> None:
    logger.info("=" * 65)
    logger.info("AGENT 4 WFA FINAL SUMMARY — %s (%d splits)", instrument, len(summary))
    logger.info("")
    logger.info("  Net Sharpe:    %.3f ± %.3f  (target >1.5, MSc baseline: 0.599)",
                summary["sharpe_net"].mean(), summary["sharpe_net"].std())
    logger.info("  Sortino:       %.3f ± %.3f  (target >2.0)",
                summary["sortino_net"].mean(), summary["sortino_net"].std())
    logger.info("  Max Drawdown:  %.1f%%        (target <15%%)",
                summary["max_drawdown"].mean() * 100)
    logger.info("  Calmar:        %.3f",
                summary["calmar"].mean())
    logger.info("  Hit Ratio:     %.1f%%        (target >55%%, MSc: 37.56%%)",
                summary["hit_ratio"].mean() * 100)
    logger.info("  Profit Factor: %.3f",
                summary["profit_factor"].mean())
    logger.info("  Cost Drag:     %.4f",
                summary["cost_drag"].mean())
    logger.info("  CVaR₉₅:        %.4f",
                summary["cvar_95"].mean())
    logger.info("  Trade Frequency:%.1f%%",
                summary["trade_frequency"].mean() * 100)
    logger.info("")

    # vs MSc baseline
    sharpe_improve = summary["sharpe_net"].mean() - 0.599
    hit_improve    = summary["hit_ratio"].mean() - 0.3756
    logger.info("  vs MSc Sharpe:  %+.3f", sharpe_improve)
    logger.info("  vs MSc Hit:     %+.1f pp", hit_improve * 100)
    logger.info("")
    logger.info("  Regime-stratified Sharpe:")
    for r_name in ["bull_trend", "bear_trend", "sideways", "crisis"]:
        col = f"sharpe_{r_name}"
        if col in summary.columns:
            logger.info("    %-15s  %.3f", r_name, summary[col].mean())
    logger.info("=" * 65)


def main() -> None:
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Train MAESTRO Agent 4: Risk")
    parser.add_argument("--instrument",    default="EUR_USD")
    parser.add_argument("--granularity",   default="M5")
    parser.add_argument("--kelly-only",    action="store_true")
    parser.add_argument("--rl-steps",      type=int, default=300_000)
    parser.add_argument("--equity",        type=float, default=10_000.0)
    parser.add_argument("--no-save",       action="store_true")
    args = parser.parse_args()

    train_risk_agent(
        instrument     = args.instrument,
        granularity    = args.granularity,
        use_rl         = not args.kelly_only,
        rl_steps       = args.rl_steps,
        account_equity = args.equity,
        save_models    = not args.no_save,
    )


if __name__ == "__main__":
    main()
