"""Configuration loading and path helpers."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def p(key: str) -> Path:
    """Resolve a path from the ``paths`` section relative to the repo root."""
    cfg = load_config()
    path = ROOT / cfg["paths"][key]
    if path.suffix == "":
        path.mkdir(parents=True, exist_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(name: str = "loto", level: int = logging.INFO) -> logging.Logger:
    logs = p("logs_dir")
    logger = logging.getLogger("loto")
    if not logger.handlers:
        logger.setLevel(level)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        fh = logging.FileHandler(logs / "loto.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(sh)
        logger.addHandler(fh)
    return logging.getLogger(name if name.startswith("loto") else f"loto.{name}")


@dataclass(frozen=True)
class Game:
    pool: int = 42
    pick: int = 6

    @property
    def n_combinations(self) -> int:
        from math import comb

        return comb(self.pool, self.pick)


GAME = Game(pool=load_config()["game"]["pool_size"], pick=load_config()["game"]["pick"])

N_JOBS = max(1, (os.cpu_count() or 1))
