"""
maestro/backtesting/backtest_engine.py
=========================================
MAESTRO End-to-End Backtest Engine.

This is the top-level integration module that orchestrates all
components of the MAESTRO system through a full walk-forward
backtest. It is the single script you run to generate the
complete performance evidence for the PhD thesis.

Architecture
------------
The backtest runs the complete pipeline per WFA split:

  ┌─────────────────────────────────────────────────────────────┐
  │  DATA PIPELINE                                              │
  │  FeatureEngineer → TripleBarrierLabeller → WalkForwardSplits│
  └────────────────────────────┬────────────────────────────────┘
                               │ train/val/test splits
  ┌────────────────────────────▼────────────────────────────────┐
  │  TRAINING (per split)                                       │
  │  Agent1.fit() → Agent2.fit() → Agent4.train_rl()           │
  └────────────────────────────┬────────────────────────────────┘
                               │ fitted agents
  ┌────────────────────────────▼────────────────────────────────┐
  │  INFERENCE (per test bar)                                   │
  │  Agent1 → Agent2 → Agent3 → Orchestrator.fuse()            │
  │       → Compliance.check() → Agent4.decide()               │
  │       → Agent5.execute() → Dashboard.update()              │
  └────────────────────────────┬────────────────────────────────┘
                               │ decisions + fills
  ┌────────────────────────────▼────────────────────────────────┐
  │  ANALYSIS                                                   │
  │  XAI.explain_batch() → PerformanceReport → HTML export     │
  └─────────────────────────────────────────────────────────────┘

PhD thesis outputs generated
------------------------------
  1. WFA performance summary CSV — Sharpe, Sortino, MaxDD, hit ratio
     per split and instrument, benchmarked against the MSc baseline.

  2. XAI report HTML — SHAP feature importance, agent contributions,
     sample trade explanations. Answers the "black box" criticism.

  3. Regime-stratified performance — proves the regime conditioning
     hypothesis: MAESTRO performs differently per regime.

  4. Cost attribution — shows exactly how much transaction costs
     eroded gross returns (addresses MSc's missing cost model).

  5. Equity curves — per-split and combined equity curve plots.

  6. Compliance log — proves MiFID II pre-trade checks pass rate.

Usage
-----
  Quick test (Kelly only, no RL, fast):
    python -m maestro.backtesting.backtest_engine --quick

  Full backtest (with RL, all instruments):
    python -m maestro.backtesting.backtest_engine \
        --instruments EUR_USD GBP_USD \
        --granularity M5 \
        --rl-steps 300000

  Reload existing models (no re-training):
    python -m maestro.backtesting.backtest_engine \
        --load-models /tmp/maestro_models/EUR_USD
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Load .env before resolving paths (critical on Windows)
try:
    from dotenv import load_dotenv as _load_dotenv
    _here = Path(__file__).resolve()
    for _d in [_here.parent, _here.parent.parent, _here.parent.parent.parent, Path.cwd(), Path.cwd().parent]:
        if (_d / ".env").exists():
            _load_dotenv(str(_d / ".env"), override=True)
            break
except Exception:
    pass

MODEL_DIR  = Path(os.environ.get("MAESTRO_MODEL_DIR",  str(Path.home() / "maestro_models")))
OUTPUT_DIR = Path(os.environ.get("MAESTRO_OUTPUT_DIR", str(Path.home() / "maestro_outputs")))
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Annualisation factor: 252 days × 78 M5 bars per day
ANNUAL_FACTOR = np.sqrt(252 * 78)
REGIME_LABELS = {0: "bull_trend", 1: "bear_trend", 2: "sideways", 3: "crisis"}


@dataclass
class BacktestConfig:
    instruments:    list[str]  = field(default_factory=lambda: ["EUR_USD"])
    granularity:    str        = "M5"
    account_equity: float      = 10_000.0
    leverage:       float      = 30.0
    use_rl:         bool       = True
    rl_steps:       int        = 300_000
    use_sentiment:  bool       = True      # requires FinBERT + news data
    val_months:     int        = 2
    horizon:        int        = 6         # primary trading horizon (bars)
    save_models:    bool       = True
    generate_xai:   bool       = True
    generate_html:  bool       = True
    quick_mode:     bool       = False     # skip RL, fewer epochs
    use_llm_orchestrator: bool = False    # PhD contribution: Claude reasoning layer


@dataclass
class SplitResult:
    """Performance results for one WFA split + instrument."""
    split_id:         int
    instrument:       str
    test_start:       pd.Timestamp
    test_end:         pd.Timestamp
    n_bars:           int
    n_trades:         int

    # Net performance (after costs)
    sharpe_net:       float
    sortino_net:      float
    calmar:           float
    max_drawdown:     float
    cum_return_net:   float
    hit_ratio:        float
    profit_factor:    float

    # Gross performance (before costs)
    sharpe_gross:     float
    cum_return_gross: float

    # Cost attribution
    total_cost_drag:  float
    avg_cost_per_trade: float

    # Risk metrics
    cvar_95:          float
    var_95:           float

    # Regime breakdown
    regime_sharpes:   dict = field(default_factory=dict)

    # vs MSc baselines
    msc_sharpe:       float = 0.599
    msc_hit_ratio:    float = 0.3756

    @property
    def sharpe_improvement(self) -> float:
        return self.sharpe_net - self.msc_sharpe

    @property
    def hit_ratio_improvement(self) -> float:
        return self.hit_ratio - self.msc_hit_ratio


class BacktestEngine:
    """
    End-to-end MAESTRO backtest engine.

    Trains and evaluates the complete MAESTRO system across all
    WFA windows, generating the full PhD thesis evidence package.
    """

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.cfg = config or BacktestConfig()
        self._split_results: list[SplitResult] = []

    # ── Main entry point ──────────────────────────────────────────────────────
    def run(self) -> pd.DataFrame:
        """
        Run the full backtest for all instruments.

        Returns
        -------
        pd.DataFrame — per-split performance metrics, all instruments
        """
        logger.info("=" * 70)
        logger.info("MAESTRO END-TO-END BACKTEST")
        logger.info("Instruments: %s | Granularity: %s | RL: %s | Quick: %s",
                    self.cfg.instruments, self.cfg.granularity,
                    self.cfg.use_rl, self.cfg.quick_mode)
        logger.info("=" * 70)

        start_time = time.time()

        all_results = []
        for instrument in self.cfg.instruments:
            logger.info("\n%s\nRunning %s...\n%s",
                        "═" * 70, instrument, "═" * 70)
            results = self._run_instrument(instrument)
            all_results.extend(results)

        summary = pd.DataFrame([r.__dict__ for r in all_results])
        if "regime_sharpes" in summary.columns:
            summary = summary.drop(columns=["regime_sharpes"])

        if summary.empty:
            logger.error("No results produced — check data path and errors above.")
            return summary
        self._log_final_summary(summary)
        self._save_outputs(summary, all_results)

        elapsed = time.time() - start_time
        logger.info("Backtest complete in %.1f minutes.", elapsed / 60)
        return summary

    # ── Per-instrument backtest ───────────────────────────────────────────────
    def _run_instrument(self, instrument: str) -> list[SplitResult]:
        from maestro.data.pipeline.ingestion import DataPipeline
        from maestro.data.validation.wfa import WalkForwardEngine
        from maestro.agents.regime.regime_classifier import RegimeDetectionAgent
        from maestro.agents.signal.signal_agent import SignalAgent
        from maestro.agents.signal.tft_model import TFTConfig
        from maestro.agents.signal.patchtst import PatchTSTConfig, HORIZONS
        from maestro.agents.risk.risk_agent import RiskManagementAgent
        from maestro.agents.execution.execution_agent import ExecutionAgent
        from maestro.orchestrator.meta_orchestrator import MetaOrchestrator
        from maestro.compliance.compliance_engine import ComplianceEngine
        from maestro.monitoring.dashboard import MAESTRODashboard

        # Load features
        pipeline = DataPipeline()
        try:
            df = pipeline.load(instrument, self.cfg.granularity)
        except Exception as exc:
            logger.error("Could not load %s data: %s", instrument, exc)
            return []

        logger.info("Loaded %s: %d bars × %d features", instrument, len(df), len(df.columns))

        # Derive WFA window from the actual data so the engine always adapts
        # to whatever date range is present — regardless of settings.yaml defaults.
        #   train_start : data_start + 6 months   (feature-engineering warmup)
        #   wfa_start   : data_start + 12 months  (enough post-warmup fit_df after
        #                 val_months=2 is carved out; ~10 clean months remain)
        #   wfa_end     : last bar in the data
        _df_start        = df.index[0]
        _df_end          = df.index[-1]
        _wfa_train_start = str((_df_start + pd.DateOffset(months=6)).date())
        _wfa_start       = str((_df_start + pd.DateOffset(months=12)).date())
        _wfa_end         = str(_df_end.date())
        wfa = WalkForwardEngine(
            train_start = _wfa_train_start,
            wfa_start   = _wfa_start,
            wfa_end     = _wfa_end,
        )
        n_splits = wfa.n_splits(df)

        instrument_results = []
        inst_out = OUTPUT_DIR / instrument
        inst_out.mkdir(parents=True, exist_ok=True)

        # News data (optional)
        news_df = self._load_news(instrument)

        dashboard   = MAESTRODashboard(instrument, self.cfg.account_equity)
        compliance  = ComplianceEngine(instrument, self.cfg.account_equity)

        for split in wfa.splits(df):
            logger.info("─" * 70)
            logger.info("[%s] Split %d/%d | test=%s→%s",
                        instrument, split.split_id + 1, n_splits,
                        split.test_start.date(), split.test_end.date())

            t0 = time.time()

            train_df = df.loc[split.train_idx]
            test_df  = df.loc[split.test_idx]

            # Skip splits with insufficient training data
            if len(train_df) < 500:
                logger.warning("Split %d skipped — only %d training bars",
                               split.split_id + 1, len(train_df))
                continue

            val_cut  = split.train_end - pd.DateOffset(months=self.cfg.val_months)
            val_mask = train_df.index >= val_cut
            val_df   = train_df[val_mask]  if val_mask.sum() > 200 else None
            fit_df   = train_df[~val_mask] if val_df is not None   else train_df

            # Final safety check — fit_df must have enough rows for scaler
            if len(fit_df) < 100:
                logger.warning("Split %d fit_df too small (%d rows) — skipping",
                               split.split_id + 1, len(fit_df))
                continue

            # Drop NaN rows — HMM and scalers cannot handle them
            fit_df  = fit_df.dropna()
            val_df  = val_df.dropna()  if val_df  is not None else None
            test_df = test_df.dropna()
            if len(fit_df) < 100:
                logger.warning("Split %d too many NaNs, only %d rows after dropna — skipping",
                               split.split_id + 1, len(fit_df))
                continue

            # ── TRAINING ──────────────────────────────────────────────────────

            # Agent 1: Regime
            regime_agent = RegimeDetectionAgent(use_transformer=not self.cfg.quick_mode)
            try:
                regime_agent.fit(fit_df, val_df=val_df)
            except Exception as exc:
                logger.warning("Regime agent fit failed: %s — using HMM-only", exc)
                try:
                    regime_agent = RegimeDetectionAgent(use_transformer=False)
                    regime_agent.fit(fit_df)
                except Exception as exc2:
                    logger.error("HMM-only fallback also failed: %s — skipping split", exc2)
                    continue

            # Agent 2: Signal
            tft_epochs  = 20 if self.cfg.quick_mode else 40
            ptst_epochs = 20 if self.cfg.quick_mode else 40
            signal_agent = SignalAgent(
                instrument      = instrument,
                primary_horizon = self.cfg.horizon,
                tft_config      = TFTConfig(seq_len=120, pred_len=max(HORIZONS),
                                            max_epochs=tft_epochs, patience=6),
                ptst_config     = PatchTSTConfig(seq_len=128, max_epochs=ptst_epochs, patience=6),
            )
            train_regimes = regime_agent.predict_batch(fit_df)["regime"]
            val_regimes   = regime_agent.predict_batch(val_df)["regime"] if val_df is not None else None
            try:
                signal_agent.fit(fit_df, train_regimes, val_df=val_df, val_regimes=val_regimes)
            except Exception as exc:
                logger.error("Signal agent fit failed: %s — skipping split", exc)
                continue

            # Agent 4: Risk
            risk_agent = RiskManagementAgent(
                instrument     = instrument,
                account_equity = self.cfg.account_equity,
                leverage       = self.cfg.leverage,
                use_rl         = self.cfg.use_rl and not self.cfg.quick_mode,
            )
            if risk_agent.use_rl:
                try:
                    risk_agent.train_rl(fit_df, pd.DataFrame(), total_steps=self.cfg.rl_steps)
                except Exception as exc:
                    logger.warning("RL training failed: %s — using Kelly-only", exc)
                    risk_agent.use_rl = False

            # ── INFERENCE ─────────────────────────────────────────────────────
            test_regimes  = regime_agent.predict_batch(test_df)
            test_signals  = signal_agent.predict_batch(test_df, test_regimes["regime"] if "regime" in test_regimes.columns else pd.Series(2, index=test_df.index))

            # Merge regime + signal info
            combined = test_signals.copy()
            if "regime" in test_regimes.columns:
                combined["regime"]          = test_regimes["regime"]
            if "confidence" in test_regimes.columns:
                combined["regime_confidence"]= test_regimes["confidence"]

            # Agent 3: Sentiment (if news available)
            sentiment_df = None
            if self.cfg.use_sentiment and not news_df.empty:
                try:
                    from maestro.agents.sentiment.fusion import SentimentFusionAgent
                    sentiment_agent = SentimentFusionAgent(instrument=instrument)
                    sentiment_agent.load_models()
                    scored_news  = sentiment_agent.score_news(news_df)
                    sentiment_df = sentiment_agent.generate_signals(
                        test_df, scored_news, test_regimes, combined
                    )
                except Exception as exc:
                    logger.warning("Sentiment agent failed: %s", exc)

            # Orchestrator fusion — rule-based or LLM-powered
            if self.cfg.use_llm_orchestrator:
                from maestro.orchestrator.llm_orchestrator import LLMOrchestrator
                orchestrator   = LLMOrchestrator(instrument=instrument)
                decisions_df   = orchestrator.run_backtest(
                    test_df, combined, test_regimes, sentiment_df
                )
                # Export rationales for XAI / thesis chapter
                rat_path = inst_out / f"llm_rationales_split_{split.split_id:03d}.csv"
                orchestrator.export_rationales(rat_path)
                logger.info(
                    "LLM Orchestrator: cost_estimate=$%.2f | rationales→%s",
                    orchestrator.cost_estimate_usd, rat_path
                )
            else:
                orchestrator = MetaOrchestrator(instrument=instrument)
                decisions_df = orchestrator.run_backtest(
                    test_df, combined, test_regimes, sentiment_df
                )

            # Risk sizes the orchestrator's fused signal. It must not bypass
            # fusion by consuming the raw technical signal directly.
            risk_inputs = combined.copy()
            risk_inputs["signal"] = decisions_df["final_signal"]
            risk_inputs["confidence"] = decisions_df["aggregate_confidence"]
            risk_inputs["regime"] = decisions_df["regime"]
            risk_decisions = risk_agent.evaluate_batch(
                test_df, risk_inputs,
                regime_signals=test_regimes,
                simulate_equity=False,
            )
            risk_decisions["regime"] = decisions_df["regime"]
            risk_decisions["final_signal"] = decisions_df["final_signal"]
            risk_decisions = compliance.check_batch(risk_decisions, test_df)

            # Authoritative accounting path: decision at t -> target fill at
            # next bar open -> costs and P&L in one causal ledger.
            from maestro.backtesting.portfolio_ledger import CausalPortfolioLedger, LedgerConfig
            ledger = CausalPortfolioLedger(
                instrument,
                LedgerConfig(initial_equity=self.cfg.account_equity),
            )
            ledger_df = ledger.run(risk_decisions, test_df)

            # ── METRICS ───────────────────────────────────────────────────────
            result = self._compute_split_metrics(
                ledger_df, ledger_df, test_df,
                test_regimes, split.split_id, instrument
            )
            instrument_results.append(result)

            # Dashboard update
            if "equity" in ledger_df.columns:
                final_equity = float(ledger_df["equity"].iloc[-1])
                dashboard.update(equity=final_equity)

            t_elapsed = time.time() - t0
            logger.info(
                "[%s] Split %d done in %.1fs: Sharpe=%.3f MaxDD=%.1f%% Hit=%.1f%%",
                instrument, split.split_id + 1, t_elapsed,
                result.sharpe_net, result.max_drawdown * 100, result.hit_ratio * 100
            )

            # Save per-split artefacts
            if self.cfg.save_models:
                sp = MODEL_DIR / instrument / f"split_{split.split_id:03d}"
                signal_agent.save(sp / "signal")
                risk_agent.save(sp / "risk")
                regime_agent.save(sp / "regime")

            ledger_df.to_parquet(inst_out / f"decisions_split_{split.split_id:03d}.parquet")

            # XAI (on a sample)
            if self.cfg.generate_xai and len(decisions_df) > 50:
                self._run_xai(decisions_df.head(200), test_df, instrument, split.split_id, inst_out)

        # Export monitoring dashboard HTML
        if self.cfg.generate_html:
            dashboard.export_html(inst_out)

        return instrument_results

    # ── Metric computation ────────────────────────────────────────────────────
    def _compute_split_metrics(
        self,
        decisions:   pd.DataFrame,
        fills:       pd.DataFrame,
        features:    pd.DataFrame,
        regimes:     pd.DataFrame,
        split_id:    int,
        instrument:  str,
    ) -> SplitResult:
        from maestro.agents.risk.cost_model import TransactionCostModel

        close    = features["close"].reindex(decisions.index).ffill()
        bar_rets = features["log_return_1"].reindex(decisions.index).fillna(0)
        units    = decisions.get("units", pd.Series(0, index=decisions.index)).fillna(0)
        direction= np.sign(units.values)

        # Prefer returns produced by the causal fill ledger.
        if "gross_return" in decisions.columns:
            gross_rets = decisions["gross_return"].fillna(0.0)
        else:
            gross_rets = pd.Series(direction * bar_rets.values, index=decisions.index)

        if "net_return" in decisions.columns:
            net_rets = decisions["net_return"].fillna(0.0)
        elif "equity" in decisions.columns:
            eq      = decisions["equity"].ffill()
            net_rets= eq.pct_change().fillna(0)
        else:
            cost_model = TransactionCostModel(instrument)
            cost_est   = cost_model.apply_to_series(bar_rets, pd.Series(direction, index=decisions.index), close)
            net_rets   = cost_est["net_return"]

        # Sharpe / Sortino
        def sharpe(s):
            m, sd = s.mean(), s.std() + 1e-10
            return float(m / sd * ANNUAL_FACTOR)

        def sortino(s):
            m  = s.mean()
            ds = s[s < 0].std() + 1e-10
            return float(m / ds * ANNUAL_FACTOR)

        # Flat bars remain in the series. Removing them and then annualising as
        # continuous M5 exposure materially inflates Sharpe and Sortino.
        sh_net   = sharpe(net_rets)   if len(net_rets)   > 10 else 0.0
        sh_gross = sharpe(gross_rets) if len(gross_rets) > 10 else 0.0
        so_net   = sortino(net_rets)  if len(net_rets)   > 10 else 0.0

        # Drawdown
        cum      = (1 + net_rets).cumprod()
        peak     = cum.cummax()
        max_dd   = float(((cum - peak) / peak).min()) * -1

        # Calmar
        cum_ret_net = float(cum.iloc[-1] - 1)
        ann_ret     = (1 + cum_ret_net) ** (252 * 78 / max(len(decisions), 1)) - 1
        calmar      = float(ann_ret / max(max_dd, 0.001))

        # Hit ratio
        active_mask  = direction != 0
        n_active     = active_mask.sum()
        hit_ratio    = float((gross_rets.values[active_mask] > 0).mean()) if n_active > 0 else 0.5

        # Profit factor
        wins  = net_rets[net_rets > 0].sum()
        losses= abs(net_rets[net_rets < 0].sum()) + 1e-10
        pf    = float(wins / losses)

        # Cost drag
        if "transaction_cost" in decisions.columns:
            cost_drag = float(decisions["transaction_cost"].sum() / self.cfg.account_equity)
        else:
            cost_drag = float(sh_gross - sh_net) / ANNUAL_FACTOR if sh_gross != 0 else 0.0

        # CVaR
        arr    = net_rets.values
        cut    = np.percentile(arr, 5)
        cvar   = float(-arr[arr <= cut].mean()) if (arr <= cut).any() else 0.0
        var_95 = float(-cut)

        # Regime-stratified Sharpe
        regime_sharpes = {}
        reg_col = "regime" if "regime" in decisions.columns else None
        if reg_col and reg_col in decisions.columns:
            for r in range(4):
                mask = decisions[reg_col].values == r
                if mask.sum() > 50:
                    r_rets = net_rets[mask]
                    regime_sharpes[REGIME_LABELS[r]] = float(
                        r_rets.mean() / (r_rets.std() + 1e-10) * ANNUAL_FACTOR
                    )

        n_trades = int(fills["executed"].fillna(False).sum()) if "executed" in fills.columns else int(active_mask.sum())
        avg_cost = float(cost_drag / max(n_trades, 1))

        return SplitResult(
            split_id          = split_id,
            instrument        = instrument,
            test_start        = decisions.index[0],
            test_end          = decisions.index[-1],
            n_bars            = len(decisions),
            n_trades          = n_trades,
            sharpe_net        = sh_net,
            sortino_net       = so_net,
            calmar            = calmar,
            max_drawdown      = max_dd,
            cum_return_net    = cum_ret_net,
            hit_ratio         = hit_ratio,
            profit_factor     = pf,
            sharpe_gross      = sh_gross,
            cum_return_gross  = float((1 + gross_rets).prod() - 1),
            total_cost_drag   = cost_drag,
            avg_cost_per_trade= avg_cost,
            cvar_95           = cvar,
            var_95            = var_95,
            regime_sharpes    = regime_sharpes,
        )

    # ── XAI ──────────────────────────────────────────────────────────────────
    def _run_xai(
        self,
        decisions_df: pd.DataFrame,
        features_df:  pd.DataFrame,
        instrument:   str,
        split_id:     int,
        output_dir:   Path,
    ) -> None:
        try:
            from maestro.xai.shap_engine import SHAPEngine
            engine = SHAPEngine()
            engine.fit_surrogate(features_df, decisions_df)
            explanations = engine.explain_batch(decisions_df, features_df)
            imp = engine.feature_importance(explanations)
            imp.to_csv(output_dir / f"feature_importance_split_{split_id:03d}.csv", index=False)
            if self.cfg.generate_html:
                engine.generate_report(explanations, output_dir,
                                       title=f"MAESTRO XAI — {instrument} Split {split_id}")
            engine.save(output_dir / f"shap_engine_split_{split_id:03d}.pkl")
        except Exception as exc:
            logger.warning("XAI failed (non-critical): %s", exc)

    # ── Save outputs ──────────────────────────────────────────────────────────
    def _save_outputs(self, summary: pd.DataFrame, results: list[SplitResult]) -> None:
        # Main summary CSV
        out_path = OUTPUT_DIR / "maestro_wfa_summary.csv"
        summary.to_csv(out_path, index=False)
        logger.info("Summary CSV → %s", out_path)

        # Per-instrument summaries
        for inst in self.cfg.instruments:
            inst_df = summary[summary["instrument"] == inst] if "instrument" in summary.columns else summary
            inst_df.to_csv(OUTPUT_DIR / inst / "wfa_summary.csv", index=False)

        # Final HTML report
        if self.cfg.generate_html:
            self._generate_html_report(summary)

    # ── Summary logging ───────────────────────────────────────────────────────
    def _log_final_summary(self, summary: pd.DataFrame) -> None:
        logger.info("\n" + "=" * 70)
        logger.info("MAESTRO FINAL BACKTEST RESULTS")
        logger.info("=" * 70)
        logger.info("")

        for inst in self.cfg.instruments:
            s = summary[summary["instrument"] == inst] if "instrument" in summary.columns else summary
            if s.empty:
                continue
            logger.info("  ── %s (%d splits) ──", inst, len(s))
            logger.info("  Net Sharpe:    %.3f ± %.3f  (target >1.5, MSc: 0.599)",
                        s["sharpe_net"].mean(), s["sharpe_net"].std())
            logger.info("  Sortino:       %.3f ± %.3f  (target >2.0)",
                        s["sortino_net"].mean(), s["sortino_net"].std())
            logger.info("  Max Drawdown:  %.1f%%        (target <15%%)",
                        s["max_drawdown"].mean() * 100)
            logger.info("  Hit Ratio:     %.1f%%        (target >55%%, MSc: 37.6%%)",
                        s["hit_ratio"].mean() * 100)
            logger.info("  Calmar:        %.3f",
                        s["calmar"].mean())
            logger.info("  Profit Factor: %.3f",
                        s["profit_factor"].mean())
            logger.info("  CVaR₉₅:        %.4f",
                        s["cvar_95"].mean())
            logger.info("  Cost Drag:     %.4f",
                        s["total_cost_drag"].mean())
            logger.info("")
            logger.info("  vs MSc Sharpe:    %+.3f", s["sharpe_net"].mean() - 0.599)
            logger.info("  vs MSc Hit Ratio: %+.1f pp",
                        (s["hit_ratio"].mean() - 0.3756) * 100)
            logger.info("")

        logger.info("  Targets achieved:")
        if not summary.empty:
            for metric, target, direction, label in [
                ("sharpe_net",    1.5,  1, "Sharpe >1.5"),
                ("sortino_net",   2.0,  1, "Sortino >2.0"),
                ("max_drawdown",  0.15,-1, "MaxDD <15%"),
                ("hit_ratio",     0.55, 1, "Hit Ratio >55%"),
            ]:
                if metric in summary.columns:
                    achieved = (
                        (summary[metric].mean() > target) if direction == 1
                        else (summary[metric].mean() < target)
                    )
                    logger.info("    %s: %s", label, "✓ ACHIEVED" if achieved else "✗ NOT YET")

        logger.info("=" * 70)

    # ── HTML report ───────────────────────────────────────────────────────────
    def _generate_html_report(self, summary: pd.DataFrame) -> Path:
        path = OUTPUT_DIR / "maestro_backtest_report.html"
        rows = ""
        for _, r in summary.iterrows():
            sh_cl = "pos" if r.get("sharpe_net", 0) > 1.5 else "neg" if r.get("sharpe_net", 0) < 1.0 else "neu"
            hr_cl = "pos" if r.get("hit_ratio", 0) > 0.55 else "neg"
            dd_cl = "neg" if r.get("max_drawdown", 0) > 0.12 else "pos"
            rows += (
                f"<tr>"
                f"<td>{r.get('instrument','')}</td>"
                f"<td>{r.get('split_id','')}</td>"
                f"<td>{pd.Timestamp(r['test_start']).strftime('%Y-%m') if 'test_start' in r.index else ''}</td>"
                f"<td class='{sh_cl}'>{r.get('sharpe_net',0):.3f}</td>"
                f"<td>{r.get('sortino_net',0):.3f}</td>"
                f"<td class='{dd_cl}'>{r.get('max_drawdown',0):.1%}</td>"
                f"<td class='{hr_cl}'>{r.get('hit_ratio',0):.1%}</td>"
                f"<td>{r.get('profit_factor',0):.2f}</td>"
                f"<td>{r.get('cum_return_net',0):.2%}</td>"
                f"<td>{r.get('cvar_95',0):.3%}</td>"
                f"</tr>\n"
            )

        html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>MAESTRO Backtest Report</title>
<style>
body{{font-family:'Segoe UI',sans-serif;background:#0d1117;color:#c9d1d9;padding:24px}}
h1{{color:#58a6ff;border-bottom:1px solid #30363d;padding-bottom:12px}}
h2{{color:#79c0ff;margin-top:32px}}
table{{border-collapse:collapse;width:100%;margin-top:12px;font-size:13px}}
th{{background:#161b22;color:#8b949e;text-align:left;padding:8px 12px;font-size:11px;text-transform:uppercase}}
td{{padding:7px 12px;border-bottom:1px solid #21262d}}
tr:hover td{{background:#161b22}}
.pos{{color:#3fb950;font-weight:600}}
.neg{{color:#f85149;font-weight:600}}
.neu{{color:#d29922;font-weight:600}}
.target-box{{display:inline-block;background:#161b22;border:1px solid #30363d;
              border-radius:8px;padding:12px 20px;margin:8px;min-width:160px}}
.target-box .val{{font-size:24px;font-weight:700;margin:4px 0}}
.target-box .lbl{{font-size:11px;color:#8b949e;text-transform:uppercase}}
</style></head><body>
<h1>🤖 MAESTRO Walk-Forward Backtest Report</h1>
<p style="color:#8b949e">PhD Research — Coventry University | {len(summary)} WFA splits | Generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>

<h2>📊 Performance vs MSc Baseline</h2>
<div>
  <div class="target-box">
    <div class="lbl">Net Sharpe (target >1.5)</div>
    <div class="val {'pos' if summary['sharpe_net'].mean()>1.5 else 'neg'}">{summary['sharpe_net'].mean():.3f}</div>
    <small style="color:#8b949e">MSc baseline: 0.599</small>
  </div>
  <div class="target-box">
    <div class="lbl">Hit Ratio (target >55%)</div>
    <div class="val {'pos' if summary['hit_ratio'].mean()>0.55 else 'neg'}">{summary['hit_ratio'].mean():.1%}</div>
    <small style="color:#8b949e">MSc baseline: 37.6%</small>
  </div>
  <div class="target-box">
    <div class="lbl">Max Drawdown (target <15%)</div>
    <div class="val {'pos' if summary['max_drawdown'].mean()<0.15 else 'neg'}">{summary['max_drawdown'].mean():.1%}</div>
  </div>
  <div class="target-box">
    <div class="lbl">Sortino (target >2.0)</div>
    <div class="val {'pos' if summary['sortino_net'].mean()>2.0 else 'neg'}">{summary['sortino_net'].mean():.3f}</div>
  </div>
</div>

<h2>📋 Per-Split Results</h2>
<table>
  <tr>
    <th>Instrument</th><th>Split</th><th>Period</th>
    <th>Sharpe</th><th>Sortino</th><th>Max DD</th>
    <th>Hit Ratio</th><th>Profit Factor</th><th>Cum Return</th><th>CVaR₉₅</th>
  </tr>
  {rows}
</table>

<p style="color:#484f58;font-size:11px;margin-top:40px">
  MAESTRO — Multi-Agent Ensemble System for Trading with Reinforcement Optimisation<br>
  Coventry University PhD Research | All returns net of transaction costs (spread + slippage)
</p>
</body></html>"""
        path.write_text(html, encoding="utf-8")
        logger.info("HTML report → %s", path)
        return path

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _load_news(self, instrument: str) -> pd.DataFrame:
        _data_dir = Path(os.environ.get("MAESTRO_DATA_DIR", str(Path.home() / "maestro_data")))
        paths = [
            _data_dir / "news.parquet",
            _data_dir / f"news_{instrument}.parquet",
        ]
        for p in paths:
            if p.exists():
                df = pd.read_parquet(p)
                logger.info("Loaded news: %d articles", len(df))
                return df
        return pd.DataFrame()


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="MAESTRO End-to-End Backtest")
    parser.add_argument("--instruments", nargs="+", default=["EUR_USD"])
    parser.add_argument("--granularity", default="M5")
    parser.add_argument("--equity",      type=float, default=10_000.0)
    parser.add_argument("--rl-steps",    type=int,   default=300_000)
    parser.add_argument("--quick",       action="store_true",
                        help="Quick mode: Kelly-only, fewer epochs, 1 instrument")
    parser.add_argument("--no-rl",            action="store_true")
    parser.add_argument("--no-sentiment",     action="store_true")
    parser.add_argument("--no-html",          action="store_true")
    parser.add_argument("--no-xai",           action="store_true")
    parser.add_argument("--llm-orchestrator", action="store_true",
                        help="PhD mode: replace rule-based fusion with Claude reasoning layer "
                             "(requires ANTHROPIC_API_KEY in .env)")
    args = parser.parse_args()

    cfg = BacktestConfig(
        instruments          = args.instruments,
        granularity          = args.granularity,
        account_equity       = args.equity,
        rl_steps      = args.rl_steps,
        use_rl        = not args.no_rl,
        use_sentiment        = not args.no_sentiment,
        generate_xai         = not args.no_xai,
        generate_html        = not args.no_html,
        quick_mode           = args.quick,
        use_llm_orchestrator = args.llm_orchestrator,
    )

    engine  = BacktestEngine(cfg)
    summary = engine.run()

    print(f"\nResults saved to: {OUTPUT_DIR}")
    if not summary.empty:
        print(f"\nMean Net Sharpe:  {summary['sharpe_net'].mean():.3f}")
        print(f"Mean Hit Ratio:   {summary['hit_ratio'].mean():.1%}")
        print(f"Mean Max DD:      {summary['max_drawdown'].mean():.1%}")


if __name__ == "__main__":
    main()
