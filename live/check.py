"""
maestro/live/check.py
=====================
Fails (exit code 1) if the published snapshot says the live trial is unwell, so a
scheduled GitHub Action that runs it emails the owner.

Unwell means: a scheduled snapshot is overdue, the
loop's heartbeat was stale when the snapshot was taken (the loop stopped), the loop
reported an error other than the market being closed, or it missed bars.

    python -m maestro.live.check snapshot.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from maestro.data.pipeline.store import market_closed
from maestro.live.schedule import PUBLISH_SLOTS

PUBLISH_GRACE = pd.Timedelta(hours=1)
MAX_HEARTBEAT_AGE = pd.Timedelta(minutes=30)
MAX_MISSED_SHARE = 0.02


def problems(snap: dict, now: pd.Timestamp) -> list[str]:
    out = []
    at = pd.Timestamp(snap["generated_at"])
    due_by = now - PUBLISH_GRACE
    midnight = due_by.normalize()
    latest_slot = max(
        (midnight + offset for offset in PUBLISH_SLOTS.values() if offset <= due_by - midnight),
        default=midnight - pd.Timedelta(days=1) + PUBLISH_SLOTS["dusk"],
    )
    if at < latest_slot:
        out.append(f"snapshot is stale: expected a publication at or after {latest_slot.isoformat()}, "
                   f"latest snapshot was generated at {at.isoformat()}")
    hb = snap.get("heartbeat") or {}
    if not hb.get("at"):
        out.append("no heartbeat: the loop has never completed a cycle")
    elif at - pd.Timestamp(hb["at"]) > MAX_HEARTBEAT_AGE:
        out.append(f"heartbeat was {(at - pd.Timestamp(hb['at'])) / pd.Timedelta(minutes=1):.0f} min old "
                   f"when the snapshot was taken: the loop has stopped")
    if hb.get("error") and not (hb["error"] == "market closed" and market_closed(at)):
        out.append(f"loop error: {hb['error']}")
    h = snap.get("health") or {}
    if h.get("bars_expected"):
        share = h["bars_missed"] / h["bars_expected"]
        if share > MAX_MISSED_SHARE:
            out.append(f"{h['bars_missed']} of {h['bars_expected']} bars missed ({share:.1%}); "
                       f"latest: {', '.join(h.get('last_missed', [])[-3:])}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Check the live trial's published snapshot")
    p.add_argument("snapshot", type=Path)
    args = p.parse_args()
    snap = json.loads(args.snapshot.read_text())
    found = problems(snap, pd.Timestamp.now(tz="UTC"))
    for line in found:
        print(f"::error::{line}")
    if found:
        sys.exit(1)
    print(f"ok: snapshot {snap['generated_at']}, heartbeat {snap['heartbeat']['at']}")


if __name__ == "__main__":
    main()
