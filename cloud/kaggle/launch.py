"""
Start a MAESTRO training session on Kaggle's free GPUs.

Fills in cloud/kaggle/run.py (code commit, design, blocks already done) and
pushes it as the private script <user>/maestro-runner, attached to the private
dataset <user>/maestro-raw. Blocks already on this machine are skipped, so each
session continues where the last one (or the laptop) stopped.

    python -m maestro.cloud.kaggle.launch --refit-months 1 --train-months 12
    python -m maestro.cloud.kaggle.launch --refit-months 1 --train-months 12 --smoke   # 1 block, 1 epoch
    python -m maestro.cloud.kaggle.collect                                             # when it has finished

Needs the Kaggle CLI (pip install kaggle) with a token in ~/.kaggle/access_token.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from maestro.backtesting.baselines import OUTPUT_DIR, design_tag
from maestro.backtesting.maestro_runner import format_ids

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
USER = os.environ.get("KAGGLE_USERNAME", "fran6jy")
KERNEL = f"{USER}/maestro-runner"
DATASET = f"{USER}/maestro-raw"
MACHINE = "NvidiaTeslaT4"          # Kaggle's "GPU T4 x2"


def kaggle(*args: str) -> str:
    exe = Path(sys.executable).with_name("kaggle.exe")
    cmd = [str(exe) if exe.exists() else "kaggle", *args]
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def pushed_commit(ref: str = "HEAD") -> str:
    """The commit `ref` names, provided GitHub already has it (the Kaggle job clones the public repo)."""
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", ref],
                          check=True, capture_output=True, text=True).stdout.strip()
    remote = subprocess.run(["git", "-C", str(REPO), "branch", "-r", "--contains", head],
                            capture_output=True, text=True).stdout
    if "origin/" not in remote:
        raise SystemExit(f"Commit {head[:7]} is not on GitHub yet; push it first.")
    return head


def tag_for(refit_months: int, train_months: int | None, smoke: bool) -> str:
    return design_tag(refit_months, train_months) + "_fast" + ("_smoke" if smoke else "")


def main() -> None:
    p = argparse.ArgumentParser(description="Start a MAESTRO training session on Kaggle")
    p.add_argument("--refit-months", type=int, default=1)
    p.add_argument("--train-months", type=int, default=12)
    p.add_argument("--budget-hours", type=float, default=6.0,
                   help="Kaggle stops sessions at 12 h; shorter sessions lose less if one is killed")
    p.add_argument("--smoke", action="store_true", help="first block only, one epoch")
    p.add_argument("--commit", default="HEAD", help="code version to run (must be on GitHub)")
    p.add_argument("--reverse", action="store_true",
                   help="train the latest blocks first (while the laptop works forward on the same design)")
    args = p.parse_args()

    tag = tag_for(args.refit_months, args.train_months, args.smoke)
    done = {int(f.stem.split("_")[1]) for f in (OUTPUT_DIR / "maestro" / tag).glob("block_*.parquet")}
    runner_args = f"--refit-months {args.refit_months} --train-months {args.train_months} --fast"
    if args.smoke:
        runner_args += " --smoke --epochs 1"
    if args.reverse:
        runner_args += " --reverse"
    if done:
        runner_args += f" --skip {format_ids(done)}"

    stage = OUTPUT_DIR / "kaggle_push"
    stage.mkdir(parents=True, exist_ok=True)
    code = (HERE / "run.py").read_text(encoding="utf-8")
    code = (code.replace("__COMMIT__", pushed_commit(args.commit))
                .replace("__RUNNER_ARGS__", runner_args)
                .replace("__BUDGET_HOURS__", str(args.budget_hours)))
    (stage / "run.py").write_text(code, encoding="utf-8")
    (stage / "kernel-metadata.json").write_text(json.dumps({
        "id": KERNEL, "title": "maestro-runner", "code_file": "run.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": True, "machine_shape": MACHINE,
        "dataset_sources": [DATASET], "kernel_sources": [], "competition_sources": [],
    }, indent=2))

    print(f"design {tag}: {len(done)} blocks already here; runner args: {runner_args}")
    print(kaggle("kernels", "push", "-p", str(stage), "--accelerator", MACHINE))


if __name__ == "__main__":
    main()
