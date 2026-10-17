"""Statistical model family (vectorised, leakage-safe by construction).

Row t of every output depends only on rows < t of the indicator matrix ``ind`` (N, 42).
"""
from __future__ import annotations

import numpy as np

POOL, PICK = 42, 6
P0 = PICK / POOL
EPS = 1e-4


def _normalise(scores: np.ndarray, floor: float = 0.01, cap: float = 0.6) -> np.ndarray:
    """Rescale non-negative scores to probabilities summing to 6 per row, then clip."""
    s = np.clip(scores, 1e-12, None)
    p = PICK * s / s.sum(1, keepdims=True)
    return np.clip(p, floor, cap)


def _cs(ind: np.ndarray) -> np.ndarray:
    return np.vstack([np.zeros((1, ind.shape[1])), np.cumsum(ind, 0)])


def uniform(ind: np.ndarray, **_) -> np.ndarray:
    return np.full(ind.shape, P0)


def frequency(ind: np.ndarray, alpha: float = 0.0, **_) -> np.ndarray:
    """Cumulative frequency ('hot numbers') with optional Dirichlet shrinkage alpha."""
    cs = _cs(ind)[:-1]                         # counts before t
    t = np.arange(len(ind))[:, None]
    f = (cs + alpha) / (PICK * t + POOL * alpha + 1e-12)
    f[0] = 1 / POOL
    return _normalise(f)


def bayes_shrinkage(ind: np.ndarray, alpha: float = 50.0, **_) -> np.ndarray:
    return frequency(ind, alpha=alpha)


def cold(ind: np.ndarray, **_) -> np.ndarray:
    """'Overdue / cold numbers': mirror image of the frequency forecast."""
    pf = frequency(ind, alpha=10.0)
    return _normalise(np.clip(2 * P0 - pf, 1e-6, None))


def window(ind: np.ndarray, w: int = 50, alpha: float = 1.0, **_) -> np.ndarray:
    cs = _cs(ind)
    t = np.arange(len(ind))
    lo = np.maximum(0, t - w)
    c = cs[t] - cs[lo]
    n = (t - lo)[:, None]
    f = (c + alpha) / (PICK * n + POOL * alpha)
    return _normalise(f)


def ewma(ind: np.ndarray, half_life: float = 30.0, **_) -> np.ndarray:
    lam = 0.5 ** (1.0 / half_life)
    out = np.empty(ind.shape, float)
    e = np.full(POOL, P0)
    for t in range(len(ind)):
        out[t] = e
        e = lam * e + (1 - lam) * ind[t]
    return _normalise(out)


def gap_hazard(ind: np.ndarray, kappa: float = 0.3, **_) -> np.ndarray:
    """Overdue model: boost numbers whose current absence is long relative to their mean gap."""
    N = len(ind)
    out = np.empty((N, POOL))
    last = np.full(POOL, -1)
    gsum = np.zeros(POOL); gcnt = np.zeros(POOL)
    for t in range(N):
        cur = t - last
        mean_gap = np.where(gcnt > 0, gsum / np.maximum(gcnt, 1), 1 / P0)
        ratio = cur / mean_gap
        z = (ratio - ratio.mean()) / (ratio.std() + 1e-9)
        out[t] = P0 * (1 + kappa * z)
        seen = ind[t] == 1
        gap = t - last[seen]
        valid = last[seen] >= 0
        gsum[seen] += np.where(valid, gap, 0); gcnt[seen] += valid
        last[seen] = t
    return _normalise(np.clip(out, 1e-6, None))


def markov(ind: np.ndarray, alpha: float = 20.0, **_) -> np.ndarray:
    """First-order Markov: P(i at t | numbers at t-1) from transitions observed before t."""
    N = len(ind)
    T = np.zeros((POOL, POOL))
    nj = np.zeros(POOL)
    out = np.full((N, POOL), P0)
    for t in range(2, N):
        prev2, prev1 = ind[t - 2], ind[t - 1]
        T += np.outer(prev2, prev1)          # transition (t-2) -> (t-1), both before t
        nj += prev2
        js = np.flatnonzero(prev1)
        cond = (T[js] + alpha * P0) / (nj[js, None] + alpha)
        out[t] = cond.mean(0)
    return _normalise(out)


def pair_association(ind: np.ndarray, alpha: float = 5.0, **_) -> np.ndarray:
    """Numbers that historically co-occur with the numbers of the previous draw."""
    N = len(ind)
    C = np.zeros((POOL, POOL))
    cnt = np.zeros(POOL)
    out = np.full((N, POOL), P0)
    q = (PICK - 1) / (POOL - 1)   # P(i in draw | j in draw) under a fair lottery
    for t in range(1, N):
        x = ind[t - 1]
        C += np.outer(x, x)       # co-occurrence counts from draws < t
        cnt += x
        js = np.flatnonzero(x)
        cond = (C[js] + alpha * q) / (cnt[js, None] + alpha)   # shrunk P(i | j co-drawn)
        lift = cond / q
        lift[np.arange(len(js)), js] = np.nan
        out[t] = np.nanmean(lift, 0)
    return _normalise(np.clip(out, 1e-6, None))


STAT_MODELS = {
    "uniform_random": (uniform, {}),
    "frequency_hot": (frequency, {"alpha": 0.0}),
    "bayes_shrinkage": (bayes_shrinkage, {"alpha": 50.0}),
    "cold_overdue": (cold, {}),
    "recency_window": (window, {"w": 50, "alpha": 1.0}),
    "ewma_frequency": (ewma, {"half_life": 30.0}),
    "gap_hazard": (gap_hazard, {"kappa": 0.3}),
    "markov_lag1": (markov, {"alpha": 20.0}),
    "pair_association": (pair_association, {"alpha": 5.0}),
}

# hyper-parameter search spaces used for tuning (on the tuning period only)
SEARCH_SPACES = {
    "bayes_shrinkage": {"alpha": [1, 5, 20, 50, 100, 250, 1000]},
    "recency_window": {"w": [10, 20, 50, 100, 250, 500]},
    "ewma_frequency": {"half_life": [5, 10, 20, 50, 100, 250]},
    "gap_hazard": {"kappa": [0.1, 0.3, 0.6, 1.0]},
    "markov_lag1": {"alpha": [5, 20, 100, 500]},
    "pair_association": {"alpha": [1, 5, 20, 100]},
}
