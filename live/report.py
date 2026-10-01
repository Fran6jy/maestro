"""
maestro/live/report.py
======================
A short plain-text health and results report for whoever runs the live trial.

    docker exec maestro-live python -m maestro.live.report        # on the VM
    python -m maestro.live.report --state <copied state dir>       # anywhere
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from maestro.live.snapshot import build


def _age(at: str | None, now: pd.Timestamp) -> str:
    if not at:
        return "never"
    minutes = (now - pd.Timestamp(at)).total_seconds() / 60
    return f"{minutes:.0f} min ago" if minutes < 120 else f"{minutes / 60:.1f} h ago"


def render(snap: dict) -> str:
    now = pd.Timestamp(snap["generated_at"])
    hb, h = snap["heartbeat"], snap.get("health") or {}
    lines = [
        f"MAESTRO live trial ({snap['phase']})  {snap['generated_at']}",
        f"  heartbeat   {_age(hb.get('at'), now)}" + (f"  [{hb['error']}]" if hb.get("error") else ""),
        f"  model       deployed {snap['model'].get('deployed_at')}  trained to {snap['model'].get('train_end')}",
        f"  orders      {snap.get('order_strategy') or 'paper only'}",
    ]
    if not snap["strategies"]:
        return "\n".join(lines + ["  no bars recorded yet"])
    lines += [
        f"  since       {snap['start']}  ({len(snap['days'])} days, last bar {snap['last_bar']})",
        f"  bars        {h.get('bars_recorded')}/{h.get('bars_expected')} recorded, "
        f"{h.get('bars_missed')} missed, {h.get('bars_unscorable')} unscorable",
        "",
        f"  {'strategy':18s} {'trades':>6s} {'hit':>6s} {'gross':>8s} {'cost@ref':>8s} {'net@ref':>8s} {'net@quoted':>10s} {'pos':>4s}",
    ]
    for name, s in sorted(snap["strategies"].items()):
        hit = f"{s['hit']:.1%}" if s["hit"] is not None else "-"
        lines.append(f"  {name:18s} {s['trades']:6d} {hit:>6s} {s['gross_pips']:8.1f} {s['cost_ref_pips']:8.1f} "
                     f"{s['gross_pips'] - s['cost_ref_pips']:8.1f} {s['gross_pips'] - s['cost_quoted_pips']:10.1f} "
                     f"{s['position']:+4.0f}")
    o = snap["orders"]
    if o.get("filled") or o.get("failed"):
        lines += ["", f"  practice orders: {o['filled']} filled, {o['failed']} failed; vs paper "
                      f"{o['mean_extra_pips']:+.2f} pips/order ({o['total_extra_pips']:+.1f} total), "
                      f"spread {o['mean_spread_pips']:.2f}, median delay {o['median_delay_s']}s"]
    lines.append("\n  pips; net@ref charges the backtests' 0.8-pip cost, net@quoted the spread OANDA quoted")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Live trial report")
    p.add_argument("--state", type=Path, default=Path("/state"))
    p.add_argument("--phase", default="shakedown", choices=["shakedown", "trial"])
    args = p.parse_args()
    print(render(build(args.state, args.phase)))


if __name__ == "__main__":
    main()
