"""
maestro/config/config.py
========================
Centralised configuration loader. Reads settings.yaml and merges
with environment variables. All modules import from here.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Load .env as early as possible — must happen before any get_secret() calls.
try:
    from dotenv import load_dotenv as _load_dotenv
    _cfg_dir  = Path(__file__).resolve().parent   # maestro/config/
    _pkg_dir  = _cfg_dir.parent                   # maestro/
    _root_dir = _pkg_dir.parent                   # Downloads/
    for _d in [_pkg_dir, _root_dir, _cfg_dir, Path.cwd(), Path.cwd().parent]:
        _env_file = _d / ".env"
        if _env_file.exists():
            _load_dotenv(str(_env_file), override=True)
            break
except ImportError:
    pass

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "settings.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Load and cache the master config. Called once per process."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    logger.info("Config loaded from %s", CONFIG_PATH)
    return cfg


def get(key_path: str, default: Any = None) -> Any:
    """
    Dot-notation config access.

    Examples
    --------
    >>> get("markets.instruments")
    >>> get("risk.max_drawdown_pct")
    >>> get("data_sources.oanda.max_candles_per_request")
    """
    cfg = load_config()
    keys = key_path.split(".")
    val = cfg
    for k in keys:
        if not isinstance(val, dict) or k not in val:
            return default
        val = val[k]
    return val


def get_secret(env_var: str) -> str:
    """
    Retrieve a secret from the environment.
    Raises clearly if not set — never silently returns empty strings.
    """
    val = os.environ.get(env_var)
    if not val:
        raise EnvironmentError(
            f"Required environment variable '{env_var}' is not set. "
            f"Add it to your .env file or AWS Secrets Manager."
        )
    return val


def instruments() -> list[dict]:
    return get("markets.instruments", [])


def instrument_ids() -> list[str]:
    return [i["id"] for i in instruments()]
