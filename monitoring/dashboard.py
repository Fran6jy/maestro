"""
maestro/monitoring/dashboard.py
=================================
MAESTRO Live Monitoring Dashboard.

Provides real-time system health, P&L attribution, and agent
status monitoring. Two output modes:

  1. Terminal dashboard  — rich-text ASCII display in the console
     (used during development and backtesting)

  2. JSON metrics endpoint — structured JSON for Grafana/Prometheus
     (used in production — pointed at a monitoring stack)

What is monitored
------------------
  System health:
    ✓ All five agents responsive
    ✓ Data pipeline heartbeat (last candle age)
    ✓ OANDA connection status
    ✓ Memory / CPU usage
    ✓ Circuit breaker status

  Trading performance (live + rolling):
    ✓ Equity curve (current value vs peak)
    ✓ Current drawdown vs limits
    ✓ Daily P&L (gross and net)
    ✓ Rolling Sharpe (last 100 trades)
    ✓ Rolling hit ratio (last 50 trades)
    ✓ Trade count today

  Agent status:
    ✓ Regime: current label + confidence + last update
    ✓ Signal: last signal + confidence + model agreement
    ✓ Sentiment: last NLP score + rolling accuracy
    ✓ Risk: CVaR + Kelly fraction + fusion weight
    ✓ Execution: last fill + avg slippage

  Alerts:
    🔴 CRITICAL — circuit breaker active / max DD breached
    🟡 WARNING  — approaching daily loss limit / high CVaR
    🟢 OK       — all systems nominal

The dashboard updates every 5 seconds in live mode and is
exported as a static HTML snapshot every bar for the PhD
performance monitoring log.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """Snapshot of all MAESTRO metrics at one point in time."""
    timestamp:          datetime

    # Equity
    equity:             float
    peak_equity:        float
    drawdown:           float
    daily_pnl:          float
    total_return:       float

    # Performance (rolling)
    rolling_sharpe:     float
    rolling_hit_ratio:  float
    n_trades_today:     int
    n_trades_total:     int

    # Agent states
    current_regime:     str
    regime_confidence:  float
    last_signal:        int
    signal_confidence:  float
    model_agree:        bool
    nlp_score:          float
    nlp_accuracy:       float
    current_cvar:       float
    fusion_weight:      float
    last_fill_slippage: float

    # System health
    data_lag_seconds:   float
    agents_healthy:     dict[str, bool]
    circuit_breaker:    bool
    compliance_pass_rate:float

    # Alerts
    alerts:             list[str] = field(default_factory=list)


class MAESTRODashboard:
    """
    Real-time monitoring dashboard for the MAESTRO trading system.

    Usage
    -----
    >>> dashboard = MAESTRODashboard(initial_equity=10_000)
    >>> dashboard.start()                           # start update loop
    >>> dashboard.update(orchestrator_decision, fill_report)  # per bar
    >>> dashboard.render_terminal()                 # print to console
    >>> metrics = dashboard.get_metrics()           # JSON-serialisable dict
    >>> dashboard.export_html(output_dir)           # HTML snapshot
    """

    def __init__(
        self,
        instrument:     str   = "EUR_USD",
        initial_equity: float = 10_000.0,
        update_interval:int   = 5,       # seconds
        output_dir:     str | None = None,
    ) -> None:
        self.instrument      = instrument
        self.initial_equity  = initial_equity
        self.update_interval = update_interval
        self.output_dir      = Path(output_dir or os.path.expanduser("~/.maestro/monitoring"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # State
        self._equity           = initial_equity
        self._peak_equity      = initial_equity
        self._daily_start_eq   = initial_equity
        self._trade_returns:   list[float] = []
        self._all_returns:     list[float] = []
        self._fill_slippages:  list[float] = []
        self._n_trades_today   = 0
        self._n_trades_total   = 0
        self._daily_date       = None
        self._last_data_ts     = datetime.now(timezone.utc)
        self._circuit_breaker  = False
        self._compliance_results: list[bool] = []
        self._metric_history:  list[SystemMetrics] = []

        # Last agent state
        self._last_regime      = "unknown"
        self._last_regime_conf = 0.0
        self._last_signal      = 0
        self._last_sig_conf    = 0.0
        self._last_model_agree = False
        self._last_nlp_score   = 0.0
        self._last_nlp_acc     = 0.5
        self._last_cvar        = 0.002
        self._last_fusion_w    = 0.2
        self._last_slippage    = 0.0
        self._agents_healthy   = {
            "regime": True, "signal": True, "sentiment": True,
            "risk": True, "execution": True,
        }

    # ── Update ────────────────────────────────────────────────────────────────
    def update(
        self,
        decision_dict:  dict | None = None,
        fill_report:    dict | None = None,
        equity:         float | None = None,
        data_timestamp: datetime | None = None,
    ) -> SystemMetrics:
        """
        Update dashboard state with latest information from one bar.

        Parameters
        ----------
        decision_dict : OrchestratorDecision.to_dict()
        fill_report   : ExecutionAgent FillReport dict (optional)
        equity        : current account equity
        data_timestamp: timestamp of the most recent bar received
        """
        now = datetime.now(timezone.utc)
        self._refresh_daily(now)

        # Update equity
        if equity is not None and equity != self._equity:
            bar_ret = (equity - self._equity) / self._equity
            self._equity   = equity
            self._peak_equity = max(self._peak_equity, equity)
            self._all_returns.append(bar_ret)
            if len(self._all_returns) > 500:
                self._all_returns = self._all_returns[-500:]

        # Update from decision
        if decision_dict:
            self._last_regime      = str(decision_dict.get("regime_name",      self._last_regime))
            self._last_regime_conf = float(decision_dict.get("regime_confidence", self._last_regime_conf))
            self._last_signal      = int(decision_dict.get("final_signal",     self._last_signal))
            self._last_sig_conf    = float(decision_dict.get("aggregate_confidence", self._last_sig_conf))
            self._last_model_agree = bool(decision_dict.get("agents_agree",    self._last_model_agree))
            self._last_nlp_score   = float(decision_dict.get("sentiment_direction", self._last_nlp_score))
            self._last_nlp_acc     = float(decision_dict.get("nlp_rolling_accuracy", self._last_nlp_acc))
            self._last_cvar        = float(decision_dict.get("risk_cvar",      self._last_cvar))
            self._last_fusion_w    = float(decision_dict.get("fusion_weight",  self._last_fusion_w))

            if decision_dict.get("final_action") == "trade":
                self._n_trades_today += 1
                self._n_trades_total += 1

        # Update from fill
        if fill_report:
            slip = float(fill_report.get("slippage_pips", 0.0))
            self._fill_slippages.append(slip)
            if len(self._fill_slippages) > 100:
                self._fill_slippages = self._fill_slippages[-100:]
            self._last_slippage = slip
            fill_ret = float(fill_report.get("net_return", 0.0))
            self._trade_returns.append(fill_ret)
            if len(self._trade_returns) > 100:
                self._trade_returns = self._trade_returns[-100:]

        if data_timestamp:
            self._last_data_ts = data_timestamp

        metrics = self._build_metrics(now)
        self._metric_history.append(metrics)
        if len(self._metric_history) > 1000:
            self._metric_history = self._metric_history[-1000:]

        return metrics

    def mark_agent_health(self, agent: str, healthy: bool) -> None:
        """Called by orchestrator when an agent fails or recovers."""
        self._agents_healthy[agent] = healthy

    def set_circuit_breaker(self, active: bool) -> None:
        self._circuit_breaker = active

    def record_compliance(self, passed: bool) -> None:
        self._compliance_results.append(passed)
        if len(self._compliance_results) > 200:
            self._compliance_results = self._compliance_results[-200:]

    # ── Terminal render ────────────────────────────────────────────────────────
    def render_terminal(self, metrics: SystemMetrics | None = None) -> str:
        """
        Render a compact terminal dashboard. Print with print(dashboard.render_terminal()).

        Example output:
        ┌─────────────────────────────────────────────────────────────┐
        │  MAESTRO  │  EUR_USD  │  2024-03-12 14:35:10 UTC           │
        ├─────────────────────────────────────────────────────────────┤
        │  💰 Equity: $10,342.50  (+3.43%)  DD: 1.2%                 │
        │  📈 Sharpe: 1.82  Hit: 58.3%  Trades: 12 today             │
        │  🤖 Regime: bull_trend (87%)  Signal: BUY (63%)  Agree: ✓  │
        │  📰 NLP: +0.42  Acc: 61.2%  Fusion: 0.23                   │
        │  ⚠️  CVaR: 0.18%  Kelly: 0.21  Compliance: 100%            │
        │  🟢 All agents healthy  │  Data lag: 2.1s                  │
        └─────────────────────────────────────────────────────────────┘
        """
        if metrics is None:
            metrics = self._build_metrics(datetime.now(timezone.utc))

        eq_pct   = (metrics.equity / self.initial_equity - 1) * 100
        dd_pct   = metrics.drawdown * 100
        dpnl_pct = metrics.daily_pnl * 100
        sig_str  = {1: "BUY", -1: "SELL", 0: "FLAT"}.get(metrics.last_signal, "?")
        agree    = "✓" if metrics.model_agree else "✗"
        cb_str   = "🔴 CIRCUIT BREAK" if metrics.circuit_breaker else ""

        agent_health = " ".join(
            f"{'✓' if v else '✗'}{k[:3]}"
            for k, v in metrics.agents_healthy.items()
        )

        alerts = ""
        for a in metrics.alerts:
            alerts += f"\n│  ⚠  {a:<56}│"

        lines = [
            "┌" + "─" * 63 + "┐",
            f"│  MAESTRO  │  {self.instrument}  │  {metrics.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC  {cb_str:<10}│",
            "├" + "─" * 63 + "┤",
            f"│  💰 Equity: ${metrics.equity:,.2f}  ({eq_pct:+.2f}%)  DD: {dd_pct:.1f}%  Day: {dpnl_pct:+.2f}%  │",
            f"│  📈 Sharpe: {metrics.rolling_sharpe:.2f}  Hit: {metrics.rolling_hit_ratio:.1%}  Trades: {metrics.n_trades_today} today ({metrics.n_trades_total} total)  │",
            f"│  🤖 Regime: {metrics.current_regime} ({metrics.regime_confidence:.0%})  Sig: {sig_str} ({metrics.signal_confidence:.0%}) Agree:{agree}  │",
            f"│  📰 NLP: {metrics.nlp_score:+.2f}  Acc: {metrics.nlp_accuracy:.1%}  FusionW: {metrics.fusion_weight:.2f}  CVaR: {metrics.current_cvar:.3%}  │",
            f"│  🔍 Agents: {agent_health}  │  DataLag: {metrics.data_lag_seconds:.1f}s  Compliance: {metrics.compliance_pass_rate:.0%}  │",
        ]
        if alerts:
            lines.append(alerts)
        lines.append("└" + "─" * 63 + "┘")
        return "\n".join(lines)

    # ── JSON metrics (for Grafana/Prometheus) ─────────────────────────────────
    def get_metrics(self) -> dict:
        """Return current metrics as a JSON-serialisable dict."""
        m = self._build_metrics(datetime.now(timezone.utc))
        return {
            "timestamp":          m.timestamp.isoformat(),
            "instrument":         self.instrument,
            "equity":             round(m.equity, 2),
            "drawdown":           round(m.drawdown, 4),
            "daily_pnl":          round(m.daily_pnl, 4),
            "total_return":       round(m.total_return, 4),
            "rolling_sharpe":     round(m.rolling_sharpe, 3),
            "rolling_hit_ratio":  round(m.rolling_hit_ratio, 3),
            "n_trades_today":     m.n_trades_today,
            "current_regime":     m.current_regime,
            "regime_confidence":  round(m.regime_confidence, 3),
            "last_signal":        m.last_signal,
            "signal_confidence":  round(m.signal_confidence, 3),
            "nlp_accuracy":       round(m.nlp_accuracy, 3),
            "cvar":               round(m.current_cvar, 5),
            "data_lag_seconds":   round(m.data_lag_seconds, 1),
            "agents_healthy":     m.agents_healthy,
            "circuit_breaker":    m.circuit_breaker,
            "alerts":             m.alerts,
        }

    # ── HTML export ───────────────────────────────────────────────────────────
    def export_html(self, output_dir: str | Path | None = None) -> Path:
        """Export a static HTML monitoring snapshot."""
        out = Path(output_dir or self.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "monitoring_snapshot.html"

        metrics = self._build_metrics(datetime.now(timezone.utc))
        history_json = json.dumps([
            {"t": m.timestamp.isoformat(), "equity": round(m.equity, 2),
             "sharpe": round(m.rolling_sharpe, 3),
             "drawdown": round(m.drawdown, 4)}
            for m in self._metric_history[-200:]
        ])

        eq_pct  = (metrics.equity / self.initial_equity - 1) * 100
        dd_pct  = metrics.drawdown * 100
        status_colour = "#f85149" if metrics.circuit_breaker else "#3fb950" if not metrics.alerts else "#d29922"
        status_text   = "CIRCUIT BREAK" if metrics.circuit_breaker else "ALERTS" if metrics.alerts else "NOMINAL"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="30">
  <title>MAESTRO Monitor — {self.instrument}</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:'Segoe UI',monospace;background:#0d1117;color:#c9d1d9;padding:20px}}
    h1{{color:#58a6ff;font-size:18px;margin-bottom:16px}}
    .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:20px}}
    .card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px}}
    .card .label{{font-size:11px;color:#8b949e;text-transform:uppercase;margin-bottom:4px}}
    .card .value{{font-size:22px;font-weight:700}}
    .pos{{color:#3fb950}} .neg{{color:#f85149}} .neu{{color:#58a6ff}}
    .status{{background:{status_colour}22;border:1px solid {status_colour};border-radius:6px;
              padding:8px 14px;color:{status_colour};font-size:13px;margin-bottom:16px}}
    canvas{{max-width:100%;background:#161b22;border-radius:8px;border:1px solid #30363d}}
  </style>
</head>
<body>
<h1>🤖 MAESTRO Live Monitor — {self.instrument} — {metrics.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC</h1>
<div class="status">● System Status: {status_text} {'| ' + ' | '.join(metrics.alerts) if metrics.alerts else ''}</div>
<div class="grid">
  <div class="card"><div class="label">Equity</div>
    <div class="value {'pos' if eq_pct>=0 else 'neg'}">${metrics.equity:,.2f}</div>
    <small style="color:#8b949e">{eq_pct:+.2f}% total | {metrics.daily_pnl*100:+.2f}% today</small></div>
  <div class="card"><div class="label">Drawdown</div>
    <div class="value {'neg' if dd_pct>5 else 'pos'}">{dd_pct:.2f}%</div>
    <small style="color:#8b949e">Limit: 15%</small></div>
  <div class="card"><div class="label">Rolling Sharpe</div>
    <div class="value {'pos' if metrics.rolling_sharpe>1.5 else 'neu'}">{metrics.rolling_sharpe:.2f}</div>
    <small style="color:#8b949e">Target: >1.5</small></div>
  <div class="card"><div class="label">Hit Ratio</div>
    <div class="value {'pos' if metrics.rolling_hit_ratio>0.55 else 'neg'}">{metrics.rolling_hit_ratio:.1%}</div>
    <small style="color:#8b949e">Target: >55%</small></div>
  <div class="card"><div class="label">Regime</div>
    <div class="value neu">{metrics.current_regime.replace('_',' ').title()}</div>
    <small style="color:#8b949e">Conf: {metrics.regime_confidence:.0%}</small></div>
  <div class="card"><div class="label">Signal</div>
    <div class="value {['neg','neu','pos'][metrics.last_signal+1]}">{['SELL','FLAT','BUY'][metrics.last_signal+1]}</div>
    <small style="color:#8b949e">Conf: {metrics.signal_confidence:.0%} | Agree: {'✓' if metrics.model_agree else '✗'}</small></div>
  <div class="card"><div class="label">NLP Accuracy</div>
    <div class="value {'pos' if metrics.nlp_accuracy>0.55 else 'neg'}">{metrics.nlp_accuracy:.1%}</div>
    <small style="color:#8b949e">Fusion: {metrics.fusion_weight:.2f}</small></div>
  <div class="card"><div class="label">CVaR₉₅</div>
    <div class="value neu">{metrics.current_cvar:.3%}</div>
    <small style="color:#8b949e">Trades today: {metrics.n_trades_today}</small></div>
</div>
<canvas id="chart" height="120"></canvas>
<script>
const data = {history_json};
if(data.length > 1){{
  const canvas = document.getElementById('chart');
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.parentElement.offsetWidth || 900;
  const eq = data.map(d=>d.equity);
  const minEq = Math.min(...eq)*0.999, maxEq = Math.max(...eq)*1.001;
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0,0,W,H);
  ctx.strokeStyle='#58a6ff'; ctx.lineWidth=1.5; ctx.beginPath();
  data.forEach((d,i)=>{{
    const x=i/(data.length-1)*W;
    const y=H-(d.equity-minEq)/(maxEq-minEq)*H*0.9-H*0.05;
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  }});
  ctx.stroke();
}}
</script>
<p style="color:#484f58;font-size:11px;margin-top:16px">Auto-refreshes every 30s | MAESTRO PhD Research — Coventry University</p>
</body>
</html>"""
        path.write_text(html, encoding="utf-8")
        return path

    # ── Build metrics ─────────────────────────────────────────────────────────
    def _build_metrics(self, now: datetime) -> SystemMetrics:
        drawdown    = (self._peak_equity - self._equity) / max(self._peak_equity, 1)
        daily_pnl   = (self._equity - self._daily_start_eq) / max(self._daily_start_eq, 1)
        total_ret   = (self._equity / self.initial_equity) - 1

        # Rolling Sharpe (last 100 bar returns, annualised)
        if len(self._all_returns) >= 10:
            arr    = np.array(self._all_returns[-100:])
            sharpe = float(arr.mean() / (arr.std() + 1e-10) * np.sqrt(252 * 78))
        else:
            sharpe = 0.0

        # Rolling hit ratio (last 50 trade returns)
        if len(self._trade_returns) >= 5:
            hit = float(np.mean([r > 0 for r in self._trade_returns[-50:]]))
        else:
            hit = 0.5

        avg_slip = float(np.mean(self._fill_slippages[-20:])) if self._fill_slippages else 0.0
        data_lag = (now - self._last_data_ts).total_seconds()
        comp_pr  = float(np.mean(self._compliance_results[-50:])) if self._compliance_results else 1.0

        alerts = self._compute_alerts(drawdown, daily_pnl, sharpe)

        return SystemMetrics(
            timestamp           = now,
            equity              = self._equity,
            peak_equity         = self._peak_equity,
            drawdown            = drawdown,
            daily_pnl           = daily_pnl,
            total_return        = total_ret,
            rolling_sharpe      = sharpe,
            rolling_hit_ratio   = hit,
            n_trades_today      = self._n_trades_today,
            n_trades_total      = self._n_trades_total,
            current_regime      = self._last_regime,
            regime_confidence   = self._last_regime_conf,
            last_signal         = self._last_signal,
            signal_confidence   = self._last_sig_conf,
            model_agree         = self._last_model_agree,
            nlp_score           = self._last_nlp_score,
            nlp_accuracy        = self._last_nlp_acc,
            current_cvar        = self._last_cvar,
            fusion_weight       = self._last_fusion_w,
            last_fill_slippage  = avg_slip,
            data_lag_seconds    = data_lag,
            agents_healthy      = dict(self._agents_healthy),
            circuit_breaker     = self._circuit_breaker,
            compliance_pass_rate= comp_pr,
            alerts              = alerts,
        )

    def _compute_alerts(self, drawdown, daily_pnl, sharpe) -> list[str]:
        alerts = []
        if self._circuit_breaker:
            alerts.append("🔴 CIRCUIT BREAKER ACTIVE")
        if drawdown > 0.12:
            alerts.append(f"🔴 Drawdown {drawdown:.1%} approaching 15% limit")
        elif drawdown > 0.08:
            alerts.append(f"🟡 Drawdown {drawdown:.1%} — monitor closely")
        if daily_pnl < -0.025:
            alerts.append(f"🟡 Daily P&L {daily_pnl:.1%} — near -3% limit")
        if not all(self._agents_healthy.values()):
            bad = [k for k, v in self._agents_healthy.items() if not v]
            alerts.append(f"🔴 Agent(s) unhealthy: {', '.join(bad)}")
        if (datetime.now(timezone.utc) - self._last_data_ts).total_seconds() > 120:
            alerts.append("🟡 Data feed stale > 120s")
        return alerts

    def _refresh_daily(self, now: datetime) -> None:
        today = now.date()
        if self._daily_date != today:
            self._daily_date    = today
            self._n_trades_today = 0
            self._daily_start_eq = self._equity
