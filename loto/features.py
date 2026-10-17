"""Vectorised structural features of six-number combinations, plus exact enumeration.

All functions take an (n, 6) integer array of *sorted* combinations.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations

import numpy as np
import pandas as pd

from .config import ROOT, p, setup_logging

log = setup_logging("loto.features")

POOL = 42
PRIMES = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41])
IS_PRIME = np.zeros(POOL + 1, dtype=bool)
IS_PRIME[PRIMES] = True
DECADE_EDGES = [(1, 9), (10, 19), (20, 29), (30, 39), (40, 42)]
DECADE_LABELS = ["1-9", "10-19", "20-29", "30-39", "40-42"]
LOW_MAX = 21  # 1..21 low, 22..42 high


def decade_of(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 9, 0, np.minimum(x // 10, 4))


def structural_features(c: np.ndarray) -> pd.DataFrame:
    c = np.sort(np.asarray(c, dtype=np.int16), axis=1)
    diffs = np.diff(c, axis=1)
    dec = decade_of(c)
    dec_counts = np.stack([(dec == k).sum(1) for k in range(5)], axis=1)
    endings = c % 10
    end_counts = np.stack([(endings == k).sum(1) for k in range(10)], axis=1)
    consec = diffs == 1
    # longest run of consecutive numbers (length in numbers)
    run = np.ones(len(c), dtype=np.int8)
    cur = np.ones(len(c), dtype=np.int8)
    for j in range(5):
        cur = np.where(consec[:, j], cur + 1, 1)
        run = np.maximum(run, cur)
    pd_dec = dec_counts / 6.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(pd_dec > 0, pd_dec * np.log2(pd_dec), 0.0), axis=1)
    out = pd.DataFrame({
        "sum": c.sum(1).astype(np.int32),
        "mean": c.mean(1),
        "median": (c[:, 2] + c[:, 3]) / 2.0,
        "range": (c[:, 5] - c[:, 0]).astype(np.int16),
        "variance": c.var(1),
        "odd": (c % 2 == 1).sum(1).astype(np.int8),
        "low": (c <= LOW_MAX).sum(1).astype(np.int8),
        "consecutive_pairs": consec.sum(1).astype(np.int8),
        "longest_run": run,
        "min_gap": diffs.min(1).astype(np.int8),
        "max_gap": diffs.max(1).astype(np.int8),
        "mean_gap": diffs.mean(1),
        "primes": IS_PRIME[c].sum(1).astype(np.int8),
        "distinct_endings": (end_counts > 0).sum(1).astype(np.int8),
        "max_same_ending": end_counts.max(1).astype(np.int8),
        "mult3": (c % 3 == 0).sum(1).astype(np.int8),
        "mult5": (c % 5 == 0).sum(1).astype(np.int8),
        "mult7": (c % 7 == 0).sum(1).astype(np.int8),
        "max_decade_count": dec_counts.max(1).astype(np.int8),
        "decades_used": (dec_counts > 0).sum(1).astype(np.int8),
        "decade_entropy": ent,
        "min_number": c[:, 0].astype(np.int8),
        "max_number": c[:, 5].astype(np.int8),
        "le31": (c <= 31).sum(1).astype(np.int8),   # "birthday" numbers (player-popularity analysis)
        "le12": (c <= 12).sum(1).astype(np.int8),
    })
    for k, lab in enumerate(DECADE_LABELS):
        out[f"dec_{lab}"] = dec_counts[:, k].astype(np.int8)
    return out


@lru_cache(maxsize=1)
def all_combinations() -> np.ndarray:
    """Every valid ticket: (5,245,786, 6) int8 array, cached to disk."""
    cache = ROOT / "artifacts" / "all_combinations.npy"
    if cache.exists():
        return np.load(cache)
    log.info("enumerating all C(42,6) combinations ...")
    arr = np.fromiter(combinations(range(1, POOL + 1), 6), dtype=np.dtype((np.int8, 6)), count=5245786)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    return arr


@lru_cache(maxsize=1)
def exact_feature_table() -> pd.DataFrame:
    """Structural features for all 5,245,786 combinations (cached as parquet)."""
    cache = ROOT / "artifacts" / "all_combination_features.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    combos = all_combinations()
    parts = [structural_features(combos[i:i + 500_000]) for i in range(0, len(combos), 500_000)]
    df = pd.concat(parts, ignore_index=True)
    keep = ["sum", "range", "odd", "low", "consecutive_pairs", "longest_run", "min_gap", "max_gap", "primes",
            "distinct_endings", "max_same_ending", "mult3", "mult5", "mult7", "max_decade_count", "decades_used",
            "min_number", "max_number", "le31", "le12"] + [f"dec_{l}" for l in DECADE_LABELS]
    df = df[keep]
    df.to_parquet(cache, index=False)
    return df


def exact_distribution(feature: str) -> pd.Series:
    """Exact probability mass function of a discrete structural feature under a fair draw."""
    t = exact_feature_table()[feature]
    return t.value_counts(normalize=True).sort_index()


def overlap_with_previous(nums: np.ndarray, lag: int = 1) -> np.ndarray:
    """Number of main numbers shared between draw t and draw t-lag (NaN-free, length n-lag)."""
    ind = np.zeros((len(nums), POOL + 1), dtype=np.int8)
    ind[np.repeat(np.arange(len(nums)), 6), np.asarray(nums).ravel()] = 1
    return (ind[lag:] * ind[:-lag]).sum(1)
