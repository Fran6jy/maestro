"""
MAESTRO retraining blocks on Kaggle's free GPUs (T4 x2).

Runs as a private Kaggle script. It clones the public code at a pinned commit,
rebuilds the features from the private dataset fran6jy/maestro-raw, then runs
one maestro_runner worker per GPU (--shard i/n) until the session budget is
nearly used. Finished blocks land in /kaggle/working/out and become the run's
output; collect them with cloud/kaggle/collect.py and the laptop scores the
full design with the same evaluator as every other run.

Blocks already trained (in earlier sessions or on the laptop) are passed in
as --skip by launch.py, so each session continues where the last one stopped.

Settings live in the constants below; cloud/kaggle/launch.py fills them in.
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

COMMIT = "__COMMIT__"                 # code version, pinned for reproducibility
RUNNER_ARGS = "__RUNNER_ARGS__"       # e.g. "--refit-months 1 --train-months 12 --fast"
BUDGET_HOURS = float("__BUDGET_HOURS__")  # Kaggle stops a session at 12 hours

t0 = time.time()
WORK = Path("/kaggle/working")
SRC = WORK / "src"
LOGS = WORK / "logs"
LOGS.mkdir(parents=True, exist_ok=True)


def sh(*cmd, **kw):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def find_input(marker: str) -> Path | None:
    """Where Kaggle mounted a dataset: the folder that contains `marker`
    (the mount path has changed between Kaggle versions, so search for it)."""
    return next((p.parent for p in sorted(Path("/kaggle/input").rglob(marker)) if p.is_dir()), None)


def cleanup() -> None:
    """Keep the output small: only blocks and logs are needed back on the laptop."""
    shutil.rmtree(SRC, ignore_errors=True)
    shutil.rmtree(WORK / "data", ignore_errors=True)


raw = find_input("candles")
print("raw data store:", raw, flush=True)
if raw is None:
    sh("find", "/kaggle/input", "-maxdepth", "4")
    raise SystemExit("maestro-raw is not attached to this script")

try:
    sh("git", "clone", "-q", "https://github.com/Fran6jy/maestro.git", str(SRC / "maestro"))
    sh("git", "-C", str(SRC / "maestro"), "checkout", "-q", COMMIT)
    sh(sys.executable, "-m", "pip", "install", "-q", "hmmlearn")

    env = {**os.environ,
           "PYTHONPATH": str(SRC),
           "MAESTRO_RAW_DIR": str(raw),
           "MAESTRO_DATA_DIR": str(WORK / "data"),
           "MAESTRO_OUTPUT_DIR": str(WORK / "out")}
    sh(sys.executable, "-m", "maestro.data.pipeline.store", "sync", env=env)
except Exception:
    cleanup()
    raise

import torch  # noqa: E402

n = max(1, torch.cuda.device_count())
hours_left = BUDGET_HOURS - (time.time() - t0) / 3600
print(f"{n} GPU(s): {[torch.cuda.get_device_name(i) for i in range(n)]}; {hours_left:.1f} h for training",
      flush=True)
workers = []
for i in range(n):
    log = open(LOGS / f"worker_{i}.log", "w")
    cmd = [sys.executable, "-u", "-m", "maestro.backtesting.maestro_runner", *RUNNER_ARGS.split(),
           "--shard", f"{i}/{n}", "--max-hours", f"{hours_left:.2f}"]
    workers.append(subprocess.Popen(cmd, env={**env, "CUDA_VISIBLE_DEVICES": str(i)},
                                    stdout=log, stderr=subprocess.STDOUT))
codes = [w.wait() for w in workers]
print("worker exit codes:", codes, flush=True)

cleanup()
for d in (WORK / "out" / "maestro").glob("*"):
    print(d.name, len(list(d.glob("block_*.parquet"))), "blocks", flush=True)
if any(codes):
    sys.exit(1)
