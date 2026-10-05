"""
maestro/live/publish.py
=======================
Publishes the live trial's Foundation snapshots by pushing them to a git repository.

Three immutable observations are written each UTC day: Dawn at 00:20 (the official
close for the previous whole UTC day), Day at 12:20 and Dusk at 20:20 (provisional
intraday views). snapshot.json always holds the latest observation; official.json
always holds the latest Dawn close. The legacy history/<date>.json path remains the
official daily record, while history/foundation/<date>-<edition>.json holds all three.

    python -m maestro.live.publish --repo git@github.com:Fran6jy/maestro-live.git \\
        --key /keys/maestro_live --work /publish            # loop, three times a day
    python -m maestro.live.publish ... --once                # publish now and exit
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
from pathlib import Path

import pandas as pd

from maestro.live.snapshot import build

logger = logging.getLogger(__name__)

PUBLISH_SLOTS = {
    "dawn": pd.Timedelta(minutes=20),
    "day": pd.Timedelta(hours=12, minutes=20),
    "dusk": pd.Timedelta(hours=20, minutes=20),
}
AUTHOR = ("Fran6jy", "30803931+Fran6jy@users.noreply.github.com")


def git(work: Path, *args: str, key: Path) -> str:
    env = {**os.environ,
           "GIT_SSH_COMMAND": f"ssh -i {key} -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new",
           "GIT_AUTHOR_NAME": AUTHOR[0], "GIT_AUTHOR_EMAIL": AUTHOR[1],
           "GIT_COMMITTER_NAME": AUTHOR[0], "GIT_COMMITTER_EMAIL": AUTHOR[1]}
    # The working copy is a host folder owned by the VM's user; the container runs as root.
    out = subprocess.run(["git", "-c", f"safe.directory={work}", "-C", str(work), *args],
                         env=env, capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {out.stderr.strip()[:500]}")
    return out.stdout


def checkout(repo: str, work: Path, key: Path) -> None:
    if (work / ".git").exists():
        git(work, "pull", "-q", "--ff-only", key=key)
        return
    work.mkdir(parents=True, exist_ok=True)
    git(work, "clone", "-q", repo, ".", key=key)


def publish(state: Path, phase: str, repo: str, work: Path, key: Path,
            edition: str | None = None, now: pd.Timestamp | None = None) -> str:
    checkout(repo, work, key)
    now = now or pd.Timestamp.now(tz="UTC")
    edition = edition or current_edition(now)
    official = edition == "dawn"
    cutoff = now.normalize() if official else None
    snap = build(state, phase, now, cutoff=cutoff, edition=edition, official=official)
    text = json.dumps(snap, separators=(",", ":"))
    elapsed = now - now.normalize()
    previous_day = official or (edition == "dusk" and elapsed < PUBLISH_SLOTS["dawn"])
    record_day = (now.normalize() - pd.Timedelta(days=int(previous_day))).strftime("%Y-%m-%d")
    (work / "history").mkdir(exist_ok=True)
    foundation = work / "history" / "foundation"
    foundation.mkdir(exist_ok=True)
    edition_path = foundation / f"{record_day}-{edition}.json"
    if edition_path.exists():
        return f"{record_day} {edition} already published"
    latest_path = work / "snapshot.json"
    seeded_official = False
    # On the first Foundation deployment, preserve the existing nightly whole-day
    # snapshot as official.json before a provisional edition replaces snapshot.json.
    if not official and not (work / "official.json").exists() and latest_path.exists():
        previous = json.loads(latest_path.read_text())
        if previous.get("cutoff"):
            (work / "official.json").write_text(json.dumps(previous, separators=(",", ":")))
            seeded_official = True
    latest_path.write_text(text)
    edition_path.write_text(text)
    files = ["snapshot.json", f"history/foundation/{edition_path.name}"]
    if seeded_official:
        files.append("official.json")
    if official:
        (work / "official.json").write_text(text)
        (work / "history" / f"{record_day}.json").write_text(text)
        files.extend(["official.json", f"history/{record_day}.json"])
    git(work, "add", *files, key=key)
    if not git(work, "status", "--porcelain", key=key).strip():
        return "nothing new"
    git(work, "commit", "-q", "-m", f"Snapshot {snap['generated_at']}", key=key)
    git(work, "push", "-q", key=key)
    kind = "official" if official else "provisional"
    return f"pushed {record_day} {edition} ({kind}, {len(text) / 1024:.0f} KB)"


def current_edition(now: pd.Timestamp) -> str:
    elapsed = now - now.normalize()
    passed = [name for name, offset in PUBLISH_SLOTS.items() if offset <= elapsed]
    return passed[-1] if passed else "dusk"


def next_run(now: pd.Timestamp) -> tuple[pd.Timestamp, str]:
    for edition, offset in PUBLISH_SLOTS.items():
        t = now.normalize() + offset
        if t > now:
            return t, edition
    return now.normalize() + pd.Timedelta(days=1) + PUBLISH_SLOTS["dawn"], "dawn"


def main() -> None:
    p = argparse.ArgumentParser(description="Publish Dawn, Day and Dusk live-trial snapshots")
    p.add_argument("--state", type=Path, default=Path("/state"))
    p.add_argument("--phase", default=os.environ.get("MAESTRO_LIVE_PHASE", "shakedown"),
                   choices=["shakedown", "trial"])
    p.add_argument("--repo", required=True)
    p.add_argument("--key", type=Path, required=True)
    p.add_argument("--work", type=Path, default=Path("/publish"))
    p.add_argument("--once", action="store_true")
    p.add_argument("--edition", choices=list(PUBLISH_SLOTS), default=None,
                   help="edition for --once; default: most recent scheduled edition")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    while True:
        edition = args.edition
        if not args.once:
            run_at, edition = next_run(pd.Timestamp.now(tz="UTC"))
            wait = (run_at - pd.Timestamp.now(tz="UTC")).total_seconds()
            logger.info("next %s publish in %.1f h", edition, wait / 3600)
            time.sleep(max(0.0, wait))
        try:
            logger.info(publish(args.state, args.phase, args.repo, args.work, args.key, edition=edition))
        except Exception:  # noqa: BLE001 — a failed edition becomes stale, which the check catches
            logger.exception("publish failed")
            if args.once:
                raise
        if args.once:
            return


if __name__ == "__main__":
    main()
