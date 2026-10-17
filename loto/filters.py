"""Phase 7: optional structural plausibility filters.

A filter *excludes* combinations with an unusual structure (e.g. all six numbers in one
decade). None of these combinations is invalid: every one of the 5,245,786 combinations
has exactly the same 1-in-5,245,786 chance in a fair draw.  A filter can only ever change
*which* tickets you play, never their probability of winning.  This module quantifies that:

* exact share of the combination space each filter removes;
* how often historical winning combinations would have been removed;
* whether that differs from the exact share (binomial test);
* on the untouched holdout: winners retained / space retained ("efficiency"; 1.0 for a
  fair lottery) and the exact expected hits of a random ticket from the filtered space.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats

from .features import all_combinations, exact_feature_table, structural_features

POOL, PICK = 42, 6


@lru_cache(maxsize=None)
def _sum_q(q):
    return float(exact_feature_table()["sum"].quantile(q))


@lru_cache(maxsize=1)
def filter_definitions() -> dict[str, tuple[str, callable]]:
    s005, s995 = _sum_q(0.005), _sum_q(0.995)
    s01, s99 = _sum_q(0.01), _sum_q(0.99)
    s05, s95 = _sum_q(0.05), _sum_q(0.95)
    return {
        "single_decade": ("all six numbers in one decade", lambda f: f["decades_used"] == 1),
        "run_ge5": ("five or six consecutive numbers", lambda f: f["longest_run"] >= 5),
        "arithmetic_progression": ("constant gap between all numbers", lambda f: f["min_gap"] == f["max_gap"]),
        "all_odd_or_even": ("all odd or all even", lambda f: f["odd"].isin([0, 6])),
        "all_low_or_high": ("all six <= 21 or all six >= 22", lambda f: f["low"].isin([0, 6])),
        "decade_ge5": ("five or more numbers in one decade", lambda f: f["max_decade_count"] >= 5),
        "sum_extreme_1pct": (f"sum outside exact 1%-99% range [{s01:.0f}, {s99:.0f}]", lambda f: (f["sum"] < s01) | (f["sum"] > s99)),
        "sum_extreme_0.5pct": (f"sum outside exact 0.5%-99.5% range [{s005:.0f}, {s995:.0f}]", lambda f: (f["sum"] < s005) | (f["sum"] > s995)),
        "run_ge4": ("four or more consecutive numbers", lambda f: f["longest_run"] >= 4),
        "same_ending_ge4": ("four or more numbers with the same last digit", lambda f: f["max_same_ending"] >= 4),
        "sum_outside_5_95": (f"sum outside exact 5%-95% range [{s05:.0f}, {s95:.0f}]", lambda f: (f["sum"] < s05) | (f["sum"] > s95)),
        "odd_1_or_5": ("exactly 1 or 5 odd numbers", lambda f: f["odd"].isin([1, 5])),
        "narrow_range": ("max - min < 20", lambda f: f["range"] < 20),
    }


MODES = {
    "none": [],
    "conservative": ["single_decade", "run_ge5", "arithmetic_progression"],
    "moderate": ["single_decade", "run_ge5", "arithmetic_progression", "all_odd_or_even", "all_low_or_high", "decade_ge5", "sum_extreme_1pct"],
    "aggressive": ["single_decade", "run_ge5", "arithmetic_progression", "all_odd_or_even", "all_low_or_high", "decade_ge5",
                   "sum_outside_5_95", "run_ge4", "same_ending_ge4", "odd_1_or_5", "narrow_range"],
}


def excluded_mask(features: pd.DataFrame, names: list[str]) -> np.ndarray:
    defs = filter_definitions()
    m = np.zeros(len(features), dtype=bool)
    for n in names:
        m |= defs[n][1](features).to_numpy()
    return m


def retained_marginals(names: list[str]) -> np.ndarray:
    """P(number i is in a uniformly random ticket from the filtered space), i=1..42."""
    combos = all_combinations()
    keep = ~excluded_mask(exact_feature_table(), names)
    kept = combos[keep].astype(np.int64)
    return np.bincount(kept.ravel(), minlength=POOL + 1)[1:] / keep.sum()


def evaluate_filters(df: pd.DataFrame, holdout_start: int, sim_features: pd.DataFrame | None = None) -> pd.DataFrame:
    nums = df[[f"n{i}" for i in range(1, 7)]].to_numpy()
    hist_f = structural_features(nums)
    exact_f = exact_feature_table()
    defs = filter_definitions()
    ind = np.zeros((len(nums), POOL), dtype=np.int8)
    ind[np.repeat(np.arange(len(nums)), 6), nums.ravel() - 1] = 1
    N = len(nums)
    rows = []
    items = [(n, [n], defs[n][0]) for n in defs] + [(f"MODE:{m}", names, f"mode '{m}' ({len(names)} filters)") for m, names in MODES.items() if names]
    for key, names, desc in items:
        ex_space = excluded_mask(exact_f, names)
        p_ex = float(ex_space.mean())
        ex_hist = excluded_mask(hist_f, names)
        k = int(ex_hist.sum())
        pv = stats.binomtest(k, N, p_ex).pvalue if 0 < p_ex < 1 else 1.0
        sim_rate = float(excluded_mask(sim_features, names).mean()) if sim_features is not None else np.nan
        # holdout
        h_ex = ex_hist[holdout_start:]
        n_h = len(h_ex)
        retained_space = 1 - p_ex
        retained_win = 1 - h_ex.mean()
        eff = retained_win / retained_space if retained_space > 0 else np.nan
        q = retained_marginals(names)
        exp_hits = float((ind[holdout_start:] * q).sum(1).mean())
        rows.append({
            "filter": key, "description": desc,
            "exact_share_of_space_excluded": p_ex, "combinations_excluded": int(ex_space.sum()),
            "historical_winners_excluded": k, "historical_share_excluded": k / N,
            "expected_excluded_if_fair": p_ex * N, "binomial_p_hist_vs_exact": pv,
            "monte_carlo_share_excluded": sim_rate,
            "holdout_draws": n_h, "holdout_winners_excluded": int(h_ex.sum()),
            "holdout_efficiency_ratio": eff,
            "holdout_expected_hits_random_filtered_ticket": exp_hits,
            "holdout_expected_hits_random_unfiltered_ticket": float((ind[holdout_start:] * (PICK / POOL)).sum(1).mean()),
            "jackpot_chance_per_ticket": "unchanged: 1 / 5,245,786",
        })
    return pd.DataFrame(rows)
