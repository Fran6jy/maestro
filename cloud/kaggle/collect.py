"""
Bring a finished Kaggle session's blocks back to this machine.

Downloads the output of <user>/maestro-runner and copies every block it trained
into OUTPUT_DIR/maestro/<design>/ next to the laptop's own blocks. Existing
blocks are never overwritten, and a block is only taken if its parquet file
opens cleanly, so an interrupted download can't bring back a truncated block.
Kaggle's file server sometimes times out on single files; the download is
retried, and whatever complete blocks arrived are merged regardless.

    python -m maestro.cloud.kaggle.collect
    python -m maestro.cloud.kaggle.collect --from C:\\tmp\\maestro_outputs\\kaggle_output\\<stamp>
    python -m maestro.backtesting.maestro_runner --refit-months 1 --train-months 12 --fast --score-only
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

from maestro.backtesting.baselines import OUTPUT_DIR
from maestro.cloud.kaggle.launch import KERNEL, kaggle

TRIES = 3


def download(dest: Path) -> None:
    for attempt in range(1, TRIES + 1):
        try:
            kaggle("kernels", "output", KERNEL, "-p", str(dest))
            return
        except subprocess.CalledProcessError as exc:
            last = (exc.stdout or "").strip().splitlines()[-1:] or ["?"]
            print(f"download attempt {attempt}/{TRIES} incomplete: {last[0][:120]}")
    print("continuing with the files that did arrive")


def complete(f: Path) -> bool:
    try:
        return len(pd.read_parquet(f)) > 0
    except Exception:
        return False


def merge(src: Path) -> None:
    for design in sorted((src / "out" / "maestro").glob("*")):
        target = OUTPUT_DIR / "maestro" / design.name
        target.mkdir(parents=True, exist_ok=True)
        new = bad = 0
        for f in sorted(design.glob("block_*.parquet")):
            if (target / f.name).exists():
                continue
            if not complete(f):
                bad += 1
                continue
            shutil.copy2(f, target / f.name)
            meta = f.with_suffix(".json")
            if meta.exists():
                shutil.copy2(meta, target / meta.name)
            new += 1
        total = len(list(target.glob("block_*.parquet")))
        print(f"{design.name}: +{new} blocks from Kaggle ({bad} unreadable, skipped), {total} here in total")


def main() -> None:
    p = argparse.ArgumentParser(description="Collect a Kaggle session's blocks")
    p.add_argument("--from", dest="src", type=Path, default=None,
                   help="merge an output folder already downloaded instead of downloading")
    args = p.parse_args()
    if args.src is None:
        print(kaggle("kernels", "status", KERNEL).strip())
        args.src = OUTPUT_DIR / "kaggle_output" / datetime.now().strftime("%Y%m%d_%H%M%S")
        args.src.mkdir(parents=True, exist_ok=True)
        download(args.src)
    merge(args.src)
    print(f"logs and raw output: {args.src}")


if __name__ == "__main__":
    main()
