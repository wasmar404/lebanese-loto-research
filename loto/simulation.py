"""Monte Carlo simulation of fair lottery histories.

Two kinds of simulation:

1. *History-level*: thousands of synthetic histories, each exactly as long as the real one.
   For each we compute the same summary statistics as for the real data (frequency
   chi-square, extreme frequency deviations, longest absences, extreme pair/triplet counts,
   consecutive-draw overlap, ...).  This gives a calibrated null distribution for
   "the most extreme thing we found anywhere" - the right answer to multiple comparisons
   when someone points at the hottest number or the most frequent pair.

2. *Bulk*: millions of independent fair draws for structural-property frequencies
   (cross-checked against the exact enumeration in features.py).
"""
from __future__ import annotations

import time
from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import N_JOBS, load_config, setup_logging
from .features import structural_features

log = setup_logging("loto.simulation")
POOL, PICK = 42, 6

# triplet encoding helpers
_TRIP_IDX = np.array(list(combinations(range(6), 3)))  # 20 triplets per draw


def fair_draws(rng: np.random.Generator, n: int) -> np.ndarray:
    """n fair draws, sorted, shape (n, 6), values 1..42."""
    keys = rng.random((n, POOL))
    return np.sort(np.argpartition(keys, PICK, axis=1)[:, :PICK] + 1, axis=1).astype(np.int16)


def indicator(nums: np.ndarray) -> np.ndarray:
    ind = np.zeros((len(nums), POOL + 1), dtype=np.int8)
    ind[np.repeat(np.arange(len(nums)), PICK), np.asarray(nums).ravel()] = 1
    return ind[:, 1:]


def history_statistics(nums: np.ndarray) -> dict[str, float]:
    """Summary statistics of one history (real or simulated). nums: (N, 6) sorted."""
    nums = np.asarray(nums, dtype=np.int64)
    N = len(nums)
    ind = indicator(nums).astype(np.int32)
    p = PICK / POOL
    counts = ind.sum(0)
    exp = N * p
    sd = np.sqrt(N * p * (1 - p))
    z = (counts - exp) / sd
    chi2 = float(((counts - exp) ** 2 / exp).sum())

    # absences: longest gap between appearances for each number (including start/end)
    max_gaps = np.empty(POOL)
    gap_sd = np.empty(POOL)
    for k in range(POOL):
        idx = np.flatnonzero(ind[:, k])
        g = np.diff(np.concatenate(([-1], idx, [N])))
        max_gaps[k] = g.max()
        inner = np.diff(idx)
        gap_sd[k] = inner.std() if len(inner) > 1 else np.nan

    # pairs
    pair = ind.T @ ind
    iu = np.triu_indices(POOL, 1)
    pc = pair[iu]
    pexp = N * (PICK * (PICK - 1)) / (POOL * (POOL - 1))

    # triplets (encode a<b<c as a*43*43+b*43+c)
    t = nums[:, _TRIP_IDX]  # (N, 20, 3)
    codes = (t[..., 0] * 43 + t[..., 1]) * 43 + t[..., 2]
    tc = np.bincount(codes.ravel(), minlength=43 ** 3)
    max_trip = int(tc.max())

    # consecutive-draw overlap at lag 1
    ov1 = (ind[1:] * ind[:-1]).sum(1)
    # repeated pairs across consecutive draws
    rep_pairs = (ov1 * (ov1 - 1) // 2).sum()

    feats = structural_features(nums)
    # full-combination repeats
    combo_codes = np.zeros(N, dtype=np.int64)
    for j in range(6):
        combo_codes = combo_codes * 43 + nums[:, j]
    _, rep = np.unique(combo_codes, return_counts=True)

    # lag-1 cross dependence: chi-square-like statistic of transition matrix
    trans = ind[:-1].T @ ind[1:]
    texp = (N - 1) * p * p
    trans_stat = float(((trans - texp) ** 2 / texp).sum())

    # serial correlation of indicator series at lag 1, averaged
    x = ind - p
    ac1 = (x[1:] * x[:-1]).sum(0) / (x * x).sum(0)

    return {
        "freq_chi2": chi2,
        "freq_max_z": float(z.max()),
        "freq_min_z": float(z.min()),
        "freq_max_abs_z": float(np.abs(z).max()),
        "freq_range": float(counts.max() - counts.min()),
        "max_absence": float(max_gaps.max()),
        "min_of_max_absence": float(max_gaps.min()),
        "mean_gap_sd": float(np.nanmean(gap_sd)),
        "pair_max": float(pc.max()),
        "pair_min": float(pc.min()),
        "pair_chi2": float(((pc - pexp) ** 2 / pexp).sum()),
        "triplet_max": float(max_trip),
        "triplets_seen_3plus": float((tc >= 3).sum()),
        "overlap_lag1_mean": float(ov1.mean()),
        "overlap_lag1_max": float(ov1.max()),
        "overlap_lag1_ge3_frac": float((ov1 >= 3).mean()),
        "repeated_pairs_consecutive": float(rep_pairs),
        "repeated_full_combos": float((rep > 1).sum()),
        "sum_mean": float(feats["sum"].mean()),
        "sum_sd": float(feats["sum"].std()),
        "consecutive_any_frac": float((feats["consecutive_pairs"] > 0).mean()),
        "longest_run_max": float(feats["longest_run"].max()),
        "max_decade_count_max": float(feats["max_decade_count"].max()),
        "all_odd_or_even_count": float(((feats["odd"] == 0) | (feats["odd"] == 6)).sum()),
        "single_decade_draws": float((feats["decades_used"] == 1).sum()),
        "transition_chi2": trans_stat,
        "ac1_max_abs": float(np.abs(ac1).max()),
        "ac1_mean": float(ac1.mean()),
    }


def _sim_batch(seed: int, n_hist: int, N: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [history_statistics(fair_draws(rng, N)) for _ in range(n_hist)]


def simulate_histories(N: int, n_histories: int | None = None, seed: int | None = None) -> pd.DataFrame:
    cfg = load_config()["simulation"]
    n_histories = n_histories or cfg["n_histories"]
    seed = cfg["seed"] if seed is None else seed
    n_batches = max(N_JOBS * 4, 1)
    sizes = [n_histories // n_batches + (i < n_histories % n_batches) for i in range(n_batches)]
    seeds = np.random.SeedSequence(seed).spawn(n_batches)
    t0 = time.time()
    res = Parallel(n_jobs=N_JOBS)(
        delayed(_sim_batch)(int(s.generate_state(1)[0]), k, N) for s, k in zip(seeds, sizes) if k
    )
    df = pd.DataFrame([r for batch in res for r in batch])
    log.info("simulated %d fair histories of %d draws in %.1fs", len(df), N, time.time() - t0)
    return df


def compare_to_null(real: dict[str, float], sims: pd.DataFrame) -> pd.DataFrame:
    """Two-sided and one-sided empirical p-values (with +1 correction) for each statistic."""
    rows = []
    n = len(sims)
    for k, v in real.items():
        s = sims[k].to_numpy()
        ge = (np.sum(s >= v) + 1) / (n + 1)
        le = (np.sum(s <= v) + 1) / (n + 1)
        rows.append({
            "statistic": k, "observed": v, "sim_mean": s.mean(), "sim_sd": s.std(ddof=1),
            "sim_p05": np.quantile(s, 0.05), "sim_p50": np.quantile(s, 0.5), "sim_p95": np.quantile(s, 0.95),
            "p_upper": ge, "p_lower": le, "p_two_sided": min(1.0, 2 * min(ge, le)),
            "z_vs_sim": (v - s.mean()) / s.std(ddof=1) if s.std(ddof=1) > 0 else 0.0,
            "n_simulations": n,
        })
    return pd.DataFrame(rows)


def bulk_structural_frequencies(n_draws: int | None = None, seed: int | None = None) -> pd.DataFrame:
    """Monte Carlo frequencies of structural properties over millions of fair draws."""
    cfg = load_config()["simulation"]
    n_draws = n_draws or cfg["n_draws_bulk"]
    bs = cfg["batch_size"]
    rng = np.random.default_rng(cfg["seed"] + 1 if seed is None else seed)
    acc: dict[str, dict] = {}
    done = 0
    t0 = time.time()
    while done < n_draws:
        b = min(bs, n_draws - done)
        f = structural_features(fair_draws(rng, b))
        for col in ["sum", "odd", "low", "consecutive_pairs", "longest_run", "primes", "max_decade_count",
                    "decades_used", "range", "max_same_ending"]:
            vc = f[col].value_counts()
            d = acc.setdefault(col, {})
            for val, c in vc.items():
                d[val] = d.get(val, 0) + int(c)
        done += b
    log.info("bulk simulation: %d fair draws in %.1fs", n_draws, time.time() - t0)
    rows = [{"feature": col, "value": val, "mc_count": c, "mc_prob": c / n_draws}
            for col, d in acc.items() for val, c in sorted(d.items())]
    return pd.DataFrame(rows)
