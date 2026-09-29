"""
Bring a finished Kaggle session's blocks back to this machine.

Downloads the output of <user>/maestro-runner and copies every block it trained
into OUTPUT_DIR/maestro/<design>/ next to the laptop's own blocks. Existing
blocks are never overwritten. Once every block of a design is present, score it:

    python -m maestro.cloud.kaggle.collect
    python -m maestro.backtesting.maestro_runner --refit-months 1 --train-months 12 --fast --score-only
"""
from __future__ import annotations

import shutil
from datetime import datetime

from maestro.backtesting.baselines import OUTPUT_DIR
from maestro.cloud.kaggle.launch import KERNEL, kaggle


def main() -> None:
    print(kaggle("kernels", "status", KERNEL).strip())
    dest = OUTPUT_DIR / "kaggle_output" / datetime.now().strftime("%Y%m%d_%H%M%S")
    dest.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", KERNEL, "-p", str(dest))
    for design in sorted((dest / "out" / "maestro").glob("*")):
        target = OUTPUT_DIR / "maestro" / design.name
        target.mkdir(parents=True, exist_ok=True)
        new = 0
        for f in sorted(design.glob("block_*")):
            if not (target / f.name).exists():
                shutil.copy2(f, target / f.name)
                new += f.suffix == ".parquet"
        total = len(list(target.glob("block_*.parquet")))
        print(f"{design.name}: +{new} blocks from Kaggle, {total} here in total")
    print(f"logs and raw output: {dest}")


if __name__ == "__main__":
    main()
