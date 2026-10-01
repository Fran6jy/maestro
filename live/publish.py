"""
maestro/live/publish.py
=======================
Publishes the live trial's snapshot once a night by pushing it to a git repository.

Each run writes snapshot.json (the latest) and history/<date>.json (that night's copy,
never rewritten), commits as the repository owner and pushes over SSH with a deploy
key that can write to that one repository only. The website reads snapshot.json, and
the history is the trial's public record.

It runs just after midnight UTC, once the UTC day the snapshot reports is complete.

    python -m maestro.live.publish --repo git@github.com:Fran6jy/maestro-live.git \\
        --key /keys/maestro_live --work /publish            # loop, once a night
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

PUBLISH_AT = pd.Timedelta(minutes=20)        # 00:20 UTC
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


def publish(state: Path, phase: str, repo: str, work: Path, key: Path) -> str:
    checkout(repo, work, key)
    now = pd.Timestamp.now(tz="UTC")
    snap = build(state, phase, now, cutoff=now.normalize())       # whole UTC days only
    text = json.dumps(snap, separators=(",", ":"))
    day = (now.normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")   # the last whole day it covers
    (work / "history").mkdir(exist_ok=True)
    (work / "snapshot.json").write_text(text)
    (work / "history" / f"{day}.json").write_text(text)
    git(work, "add", "snapshot.json", f"history/{day}.json", key=key)
    if not git(work, "status", "--porcelain", key=key).strip():
        return "nothing new"
    git(work, "commit", "-q", "-m", f"Snapshot {snap['generated_at']}", key=key)
    git(work, "push", "-q", key=key)
    return f"pushed {day} ({len(text) / 1024:.0f} KB)"


def next_run(now: pd.Timestamp) -> pd.Timestamp:
    t = now.normalize() + PUBLISH_AT
    return t if t > now else t + pd.Timedelta(days=1)


def main() -> None:
    p = argparse.ArgumentParser(description="Publish the live trial snapshot nightly")
    p.add_argument("--state", type=Path, default=Path("/state"))
    p.add_argument("--phase", default=os.environ.get("MAESTRO_LIVE_PHASE", "shakedown"),
                   choices=["shakedown", "trial"])
    p.add_argument("--repo", required=True)
    p.add_argument("--key", type=Path, required=True)
    p.add_argument("--work", type=Path, default=Path("/publish"))
    p.add_argument("--once", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    while True:
        if not args.once:
            wait = (next_run(pd.Timestamp.now(tz="UTC")) - pd.Timestamp.now(tz="UTC")).total_seconds()
            logger.info("next publish in %.1f h", wait / 3600)
            time.sleep(max(0.0, wait))
        try:
            logger.info(publish(args.state, args.phase, args.repo, args.work, args.key))
        except Exception:  # noqa: BLE001 — a failed night shows up as a stale snapshot, which the check catches
            logger.exception("publish failed")
            if args.once:
                raise
        if args.once:
            return


if __name__ == "__main__":
    main()
