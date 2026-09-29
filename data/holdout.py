"""
maestro/data/holdout.py
=======================
The sealed holdout period.

Every design decision so far (bug fixes, MAESTRO's three position rules, the
quarterly retraining schedule) was made while looking at results up to
6 March 2026. Data from HOLDOUT_START onwards has never influenced anything,
so it is kept sealed: every loader drops it unless the holdout is explicitly
unlocked. It is opened once, for the final confirmation run of a frozen design.

Unlock with the environment variable MAESTRO_HOLDOUT=unlock (the runners'
--holdout flag sets it). Ingestion is not affected: new data keeps arriving
daily, it just stays invisible to research code until then.
"""
from __future__ import annotations

import os

import pandas as pd

HOLDOUT_START = pd.Timestamp("2026-03-07", tz="UTC")
_ENV = "MAESTRO_HOLDOUT"


def unlocked() -> bool:
    return os.environ.get(_ENV, "").lower() == "unlock"


def unlock() -> None:
    """Open the holdout for this process (used by --holdout flags)."""
    os.environ[_ENV] = "unlock"


def seal(obj):
    """Drop every row at or after HOLDOUT_START unless the holdout is unlocked.

    Works on anything with a DatetimeIndex (Series or DataFrame).
    """
    if unlocked():
        return obj
    return obj[obj.index < HOLDOUT_START]
