#!/usr/bin/env python
"""
run_live_trading.py
===================
MAESTRO Live/Demo Trading Loop.

Orchestrates the real-time execution of the MAESTRO agents:
1. Regime Agent (Hmm + Transformer)
2. Technical Signal Agent (TFT + PatchTST)
3. Risk Agent (Gates + Kelly + RL)
4. Compliance Engine (MiFID II limits)
5. Execution Agent (Smart order routing)
6. Monitoring Dashboard (Terminal render & HTML export)

Usage
-----
    venv\\Scripts\\python run_live_trading.py --dry-run
    venv\\Scripts\\python run_live_trading.py --one-bar --dry-run
"""
import sys
import os
import argparse
import time
import logging
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

# Adjust Python path to resolve 'maestro' package
THIS_FILE = os.path.abspath(__file__)
MAESTRO_DIR = os.path.dirname(THIS_FILE)       # Downloads\maestro
PARENT_DIR  = os.path.dirname(MAESTRO_DIR)     # Downloads
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from maestro.config.config import get, get_secret
from maestro.data.connectors.oanda import OANDAConnector
from maestro.data.features.engineer import FeatureEngineer
from maestro.agents.regime.regime_classifier import RegimeDetectionAgent
from maestro.agents.signal.signal_agent import SignalAgent
from maestro.agents.risk.risk_agent import RiskManagementAgent
from maestro.agents.execution.execution_agent import ExecutionAgent
from maestro.orchestrator.meta_orchestrator import MetaOrchestrator
from maestro.compliance.compliance_engine import ComplianceEngine
from maestro.monitoring.dashboard import MAESTRODashboard

# Configure Loguru-style console logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("live_runner")

# Colours for a premium CLI experience
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def print_banner(dry_run: bool):
    mode_str = f"{YELLOW}DRY-RUN (SIMULATING ORDERS){RESET}" if dry_run else f"{RED}{BOLD}LIVE EXECUTION (ACTIVE TRADING){RESET}"
    print(f"\n{BOLD}{BLUE}================================================================={RESET}")
    print(f"{BOLD}{BLUE}  MAESTRO LIVE EXECUTION RUNNER{RESET}")
    print(f"{BOLD}{BLUE}  ---------------------------------------------------------------{RESET}")
    print(f"  Mode:       {mode_str}")
    print(f"  Start Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"{BOLD}{BLUE}================================================================={RESET}\n")


def get_live_features(conn: OANDAConnector, fe: FeatureEngineer, instrument: str, start_time: datetime, last_macro_values: dict = None) -> pd.DataFrame:
    """Fetch EUR_USD and GBP_USD and compute live features including cross-pair correlation."""
    try:
        # Fetch lookback candles for both instruments to compute technical and cross-pair features
        df_eur = conn.fetch_historical("EUR_USD", "M5", start=start_time)
        df_gbp = conn.fetch_historical("GBP_USD", "M5", start=start_time)
        
        if df_eur.empty or df_gbp.empty:
            logger.error("OANDA returned empty historical data for EUR_USD or GBP_USD")
            return pd.DataFrame()
            
        feat_eur = fe.transform(df_eur, drop_nan=False)
        feat_gbp = fe.transform(df_gbp, drop_nan=False)
        
        # Add cross-pair features
        from maestro.data.features.engineer import add_cross_pair_features
        feat_eur, feat_gbp = add_cross_pair_features(feat_eur, feat_gbp)
        
        df_features = feat_eur if instrument == "EUR_USD" else feat_gbp
        
        # Fill in slowly-changing macro features from the historical dataset
        if last_macro_values:
            for col, val in last_macro_values.items():
                if col not in df_features.columns or df_features[col].isna().all():
                    df_features[col] = val
                    
        return df_features
    except Exception as exc:
        logger.error(f"Error computing live features: {exc}")
        return pd.DataFrame()


def sleep_until_next_bar(interval_mins: int = 5, buffer_secs: int = 5):
    """Calculates time remaining until the next bar close and sleeps."""
    now = datetime.now(timezone.utc)
    mins = now.minute
    next_mins = ((mins // interval_mins) + 1) * interval_mins
    
    # Rollover hour if next_mins is 60
    if next_mins >= 60:
        next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        next_time = now.replace(minute=next_mins, second=0, microsecond=0)
        
    sleep_secs = (next_time - now).total_seconds() + buffer_secs
    logger.info(f"Waiting {sleep_secs:.1f}s until next M5 bar close at {next_time + timedelta(seconds=buffer_secs)} UTC...")
    time.sleep(sleep_secs)


def main():
    parser = argparse.ArgumentParser(description="MAESTRO Live Execution Runner")
    parser.add_argument("--instrument", default="EUR_USD", choices=["EUR_USD", "GBP_USD"])
    parser.add_argument("--model-dir", default=r"C:\tmp\maestro_models\EUR_USD\split_002")
    parser.add_argument("--dry-run", action="store_true", help="Simulate orders instead of submitting to OANDA")
    parser.add_argument("--one-bar", action="store_true", help="Run once for the current bar and exit")
    parser.add_argument("--interval", type=int, default=5, help="Bar interval in minutes")
    args = parser.parse_args()

    print_banner(args.dry_run)

    # 1. Initialize OANDA connection & verify credentials
    logger.info("Initializing OANDA connector...")
    try:
        conn = OANDAConnector()
        acct_summary = conn.get_account_summary()
        logger.info(f"{GREEN}Connected to OANDA!{RESET} Account ID: {conn.cfg.account_id} | Balance: ${acct_summary['balance']:,.2f} | NAV: ${acct_summary['nav']:,.2f}")
    except Exception as exc:
        logger.error(f"{RED}Failed to connect to OANDA. Verify your credentials in .env file.{RESET}")
        logger.error(f"Error details: {exc}")
        sys.exit(1)

    # 2. Load trained agents
    logger.info(f"Loading agents from {args.model_dir}...")
    try:
        regime_agent = RegimeDetectionAgent.load(os.path.join(args.model_dir, "regime"))
        signal_agent = SignalAgent.load(os.path.join(args.model_dir, "signal"))
        risk_agent = RiskManagementAgent.load(os.path.join(args.model_dir, "risk"))
        logger.info(f"{GREEN}All agents loaded successfully!{RESET}")
    except Exception as exc:
        logger.error(f"{RED}Failed to load agents: {exc}{RESET}")
        sys.exit(1)

    # 3. Load macro baseline values from C:\tmp\maestro_data
    last_macro_values = {}
    macro_cols = ['DFF', 'T10Y2Y', 'VIXCLS', 'CPIAUCSL', 'UNRATE', 'DEXUSEU', 'DEXUSUK', 'yield_curve_inverted', 'rate_change_1m', 'vix_high', 'vix_zscore', 'inflation_yoy']
    hist_path = r"C:\tmp\maestro_data\EUR_USD_features.parquet"
    if os.path.exists(hist_path):
        try:
            logger.info("Loading macro data baselines from historical Parquet...")
            hist_feat = pd.read_parquet(hist_path)
            last_row = hist_feat[hist_feat.index <= pd.Timestamp.now(tz="UTC")].iloc[-1]
            last_macro_values = {c: last_row[c] for c in macro_cols if c in last_row}
            logger.info(f"Loaded {len(last_macro_values)} macro features (DFF={last_macro_values.get('DFF')}, VIXCLS={last_macro_values.get('VIXCLS')})")
        except Exception as exc:
            logger.warning(f"Could not load macro baselines: {exc}. Using zero-fill defaults.")
    else:
        logger.warning(f"Historical features file not found at {hist_path}. Using zero-fill defaults.")

    # 4. Initialize orchestrator, compliance engine, execution agent & dashboard
    orchestrator = MetaOrchestrator(args.instrument)
    orchestrator.load_agents(
        regime_agent=regime_agent,
        signal_agent=signal_agent,
        risk_agent=risk_agent,
    )
    
    compliance = ComplianceEngine(args.instrument, account_equity=acct_summary["nav"])
    
    # If dry-run, we simulate orders. If not, execution agent is set to live.
    execution = ExecutionAgent(args.instrument, live=not args.dry_run, account_id=conn.cfg.account_id)
    
    dashboard = MAESTRODashboard(args.instrument, initial_equity=acct_summary["nav"], output_dir=r"C:\tmp\maestro_outputs")

    fe = FeatureEngineer()

    # Main Execution Loop
    logger.info("Entering trading loop...")
    while True:
        try:
            if not args.one_bar:
                sleep_until_next_bar(args.interval)
            
            logger.info("Retrieving live market state...")
            # Fetch last 4 days of history to cover all indicator windows (SMAs up to 200 bars)
            lookback_start = datetime.now(timezone.utc) - timedelta(days=4)
            df_features = get_live_features(conn, fe, args.instrument, lookback_start, last_macro_values)
            
            if df_features.empty:
                logger.error("Could not construct features for the current bar. Retrying on next cycle.")
                if args.one_bar:
                    break
                continue
                
            # Process the latest bar
            latest_bar = df_features.iloc[-1:]
            history_df = df_features.iloc[:-1]
            latest_ts = latest_bar.index[-1]
            current_price = float(latest_bar["close"].iloc[-1])
            
            logger.info(f"Processing bar at {latest_ts} | Close: {current_price:.5f}")
            
            # 1. Pipeline Execution
            decision = orchestrator.process_bar(latest_bar, history_df)
            
            # 2. Compliance check
            dec_df = pd.DataFrame([decision.to_dict()]).set_index("timestamp")
            comp_df = compliance.check_batch(dec_df, latest_bar)
            is_compliant = bool(comp_df["compliant"].iloc[-1])
            
            # Update decision compliance status
            decision.is_compliant = is_compliant
            if not is_compliant:
                decision.final_action = "flat"
                decision.flat_reason = "compliance_block"
                logger.warning(f"{RED}Compliance Block triggered: {comp_df['compliance_reasons'].iloc[-1]}{RESET}")

            # 3. Execution
            fill = None
            if decision.final_action == "trade" and decision.final_units != 0:
                # Convert OrchestratorDecision to RiskDecision structure expected by ExecutionAgent
                from maestro.agents.risk.risk_agent import RiskDecision
                risk_dec = RiskDecision(
                    timestamp         = decision.timestamp,
                    instrument        = decision.instrument,
                    action            = decision.final_action,
                    units             = decision.final_units,
                    stop_loss_pips    = decision.stop_loss_pips,
                    take_profit_pips  = decision.take_profit_pips,
                    position_fraction = decision.risk_position_frac,
                    kelly_fraction    = decision.risk_kelly_frac,
                    var_utilisation   = decision.var_utilisation if hasattr(decision, 'var_utilisation') else 0.0,
                    cvar              = decision.risk_cvar,
                    drawdown          = decision.risk_drawdown,
                    daily_pnl         = 0.0,
                    risk_reason       = decision.flat_reason if decision.final_action != 'trade' else 'live_edge',
                    is_compliant      = decision.is_compliant,
                    metadata          = {"regime": decision.regime}
                )
                
                logger.info(f"{BOLD}{GREEN}SIGNAL TRIGGERED: {decision.final_action.upper()} {decision.final_units} units on {args.instrument}{RESET}")
                
                if args.dry_run:
                    logger.info(f"{YELLOW}[DRY-RUN] Simulating execution...{RESET}")
                    # Simulate fill using the latest bar
                    fill = execution._simulate_fill(
                        order = execution._build_order(risk_dec, current_price, "london_ny_overlap"),
                        current_bar = latest_bar.iloc[-1],
                        current_price = current_price
                    )
                else:
                    logger.info(f"{RED}[LIVE] Executing OANDA trade...{RESET}")
                    fill = execution.execute(risk_dec, latest_bar.iloc[-1])
            else:
                logger.info(f"No trade signal triggered. System state is FLAT. Reason: {decision.flat_reason or 'No edge'}")

            # 4. Update Monitoring
            acct_summary = conn.get_account_summary()
            metrics = dashboard.update(
                decision_dict  = decision.to_dict(),
                equity         = acct_summary["nav"],
                data_timestamp = latest_ts,
            )
            
            # Print dashboard output
            print("\n" + dashboard.render_terminal(metrics) + "\n")
            
            # Save HTML dashboard
            html_path = dashboard.export_html()
            logger.info(f"Dashboard HTML snapshot updated → {html_path}")

        except Exception as exc:
            logger.error(f"Exception occurred in live loop: {exc}")
            import traceback
            logger.error(traceback.format_exc())
            
        if args.one_bar:
            logger.info("Exit requested due to --one-bar flag.")
            break
            
        # Settle sleep to prevent tight looping on error
        time.sleep(10)


if __name__ == "__main__":
    main()
