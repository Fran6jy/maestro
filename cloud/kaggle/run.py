"""
MAESTRO retraining blocks on Kaggle's free GPUs (T4 x2).

Runs as a private Kaggle script. It clones the public code at a pinned commit,
rebuilds the features from the private dataset fran6jy/maestro-raw, then runs
as many maestro_runner workers as fit in memory (at most one per GPU, --shard
i/n). Each worker trains one block per fresh process, so memory is returned
after every block, until the session budget is nearly used. A heartbeat with
free RAM and progress goes to the main log, which Kaggle keeps even if it kills
the session. Finished blocks land in /kaggle/working/out and become the run's
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
import threading
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

GB_PER_WORKER = 6.5     # peak RAM of one worker on a full block (measured 5.7 GB) plus margin
RETRIES = 2             # a worker survives this many failed blocks (e.g. a transient CUDA error)
OUT = WORK / "out" / "maestro"


def meminfo_gb(key: str) -> float:
    for line in open("/proc/meminfo"):
        if line.startswith(key + ":"):
            return int(line.split()[1]) / 1024 ** 2
    return float("nan")


def done_by(i: int, n: int) -> int:
    """Blocks of worker i's shard already saved in this session."""
    return sum(int(f.stem.split("_")[1]) % n == i for f in OUT.glob("*/block_*.parquet"))


gpus = torch.cuda.device_count()
total, free = meminfo_gb("MemTotal"), meminfo_gb("MemAvailable")
# One worker per GPU, but only as many as fit in memory at the same time.
n = max(1, min(max(gpus, 1), int((free - 2) // GB_PER_WORKER)))
deadline = t0 + BUDGET_HOURS * 3600
print(f"RAM {total:.1f} GB total, {free:.1f} GB free | GPUs: "
      f"{[torch.cuda.get_device_name(i) for i in range(gpus)]} | {n} worker(s), "
      f"{(deadline - time.time()) / 3600:.1f} h left", flush=True)

codes: dict[int, int] = {}


def worker(i: int) -> None:
    """Train one block per fresh process, so memory is returned after every block."""
    failures = 0
    with open(LOGS / f"worker_{i}.log", "a") as log:
        while True:
            before = done_by(i, n)
            cmd = [sys.executable, "-u", "-m", "maestro.backtesting.maestro_runner", *RUNNER_ARGS.split(),
                   "--shard", f"{i}/{n}", "--limit", "1",
                   "--max-hours", f"{(deadline - time.time()) / 3600:.2f}"]
            rc = subprocess.run(cmd, env={**env, "CUDA_VISIBLE_DEVICES": str(i)},
                                stdout=log, stderr=subprocess.STDOUT).returncode
            if rc != 0:
                failures += 1
                print(f"worker {i}: block process exited {rc} ({failures}/{RETRIES + 1})", flush=True)
                if failures > RETRIES:
                    codes[i] = rc
                    return
                continue
            if done_by(i, n) == before:      # nothing left for this worker, or no time for another block
                codes[i] = 0
                return


threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(n)]
for th in threads:
    th.start()
while any(th.is_alive() for th in threads):
    # Heartbeat in the main log, which Kaggle keeps even if the session is killed.
    for th in threads:
        th.join(timeout=300 / len(threads))
    print(f"[{(time.time() - t0) / 3600:.1f} h] RAM free {meminfo_gb('MemAvailable'):.1f} GB | "
          f"blocks done: {[done_by(i, n) for i in range(n)]}", flush=True)
print("worker exit codes:", codes, flush=True)

cleanup()
for d in OUT.glob("*"):
    print(d.name, len(list(d.glob("block_*.parquet"))), "blocks", flush=True)
if any(codes.values()):
    sys.exit(1)
