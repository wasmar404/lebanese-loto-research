"""Phase 4: explicit hypotheses about an 'underlying algorithm'.

For each hypothesis we state what the data would have to show to distinguish it from a fair
random process, run a calibrated test (null distribution from shuffled or simulated
histories), and record a verdict.  Predictive hypotheses (Markov, recency, regimes, ML) are
additionally judged out-of-sample in the walk-forward backtest.
"""
from __future__ import annotations

import warnings
from math import gcd

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import load_config, setup_logging
from .features import structural_features
from .simulation import fair_draws, indicator

log = setup_logging("loto.hypotheses")
POOL, PICK = 42, 6


def _affine_maps():
    """All bijections x -> (a*x + b) mod 42 on {0..41} (a coprime with 42)."""
    maps = []
    for a in range(1, 42):
        if gcd(a, 42) != 1:
            continue
        for b in range(42):
            maps.append(((a * np.arange(42) + b) % 42))
    return np.array(maps)  # (504, 42)


def affine_map_statistics(ind: np.ndarray, maps: np.ndarray):
    """For every transition t->t+1, the best overlap achieved by any affine map applied to draw t,
    and how often the single most common best-map wins."""
    A, B = ind[:-1].astype(np.int16), ind[1:].astype(np.int16)
    best = np.zeros(len(A), dtype=np.int16)
    best_map = np.zeros(len(A), dtype=np.int32)
    for m_i, perm in enumerate(maps):
        mapped = np.zeros_like(A)
        mapped[:, perm] = A  # number x goes to perm[x]
        ov = (mapped * B).sum(1)
        upd = ov > best
        best = np.where(upd, ov, best)
        best_map = np.where(upd, m_i, best_map)
    top_share = np.bincount(best_map, minlength=len(maps)).max() / len(A)
    return float(best.mean()), float((best >= 4).mean()), float(top_share)


def _hmm_bic_gain(X: np.ndarray, k: int = 2, seed: int = 0) -> float:
    from hmmlearn.hmm import GaussianHMM
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ll1 = GaussianHMM(1, covariance_type="diag", random_state=seed, n_iter=50).fit(X).score(X)
        best = -np.inf
        for s in range(3):
            try:
                mk = GaussianHMM(k, covariance_type="diag", random_state=seed + s, n_iter=200).fit(X)
                best = max(best, mk.score(X))
            except Exception:
                pass
    d = X.shape[1]
    params = lambda kk: kk * (kk - 1) + (kk - 1) + kk * 2 * d
    n = len(X)
    bic1 = -2 * ll1 + params(1) * np.log(n)
    bick = -2 * best + params(k) * np.log(n)
    return float(bic1 - bick)  # >0 means k states preferred


def rolling_chi2_max(ind: np.ndarray, w: int = 250) -> float:
    cs = np.vstack([np.zeros((1, ind.shape[1])), np.cumsum(ind, 0)])
    win = cs[w:] - cs[:-w]
    e = w * PICK / POOL
    return float((((win - e) ** 2) / e).sum(1).max())


def ar_f_test(y: np.ndarray, p: int = 10) -> tuple[float, float]:
    Y = y[p:]
    X = np.column_stack([y[p - k:-k] for k in range(1, p + 1)])
    res = sm.OLS(Y, sm.add_constant(X)).fit()
    return float(res.fvalue), float(res.f_pvalue)


def run_hypotheses(df: pd.DataFrame, n_null: int = 200) -> pd.DataFrame:
    cfg = load_config()
    rng = np.random.default_rng(cfg["simulation"]["seed"] + 1234)
    nums = df[[f"n{i}" for i in range(1, 7)]].to_numpy()
    N = len(nums)
    ind = indicator(nums).astype(np.int8)
    feats = structural_features(nums)
    rows = []

    def add(h, test, stat, p, evidence, extra=""):
        rows.append(dict(hypothesis=h, test=test, statistic=stat, p_value=p, evidence_required=evidence, details=extra))

    # H1 pseudorandom affine generator (x_{t+1} = a x_t + b mod 42 on the drawn set)
    maps = _affine_maps()
    obs = affine_map_statistics(ind, maps)
    null = np.array([affine_map_statistics(indicator(fair_draws(rng, N)), maps) for _ in range(n_null)])
    for j, name in enumerate(["mean best-map overlap", "share of transitions with best overlap>=4", "share won by the single most common map"]):
        p = (np.sum(null[:, j] >= obs[j]) + 1) / (n_null + 1)
        add("H1 affine/LCG-style generator maps draw t to draw t+1", f"{name} over 504 affine maps (vs {n_null} fair histories)",
            obs[j], p, "A consistent map (same a,b) that predicts the next draw far more often than in fair histories",
            f"null mean {null[:, j].mean():.4f}, 99th pct {np.quantile(null[:, j], 0.99):.4f}")

    # H2 hidden regimes in draw structure (HMM)
    X = feats[["sum", "odd", "low", "range", "consecutive_pairs"]].to_numpy(float)
    X = (X - X.mean(0)) / X.std(0)
    gain = _hmm_bic_gain(X)
    null_g = []
    for i in range(min(n_null, 40)):
        Xs = structural_features(fair_draws(rng, N))[["sum", "odd", "low", "range", "consecutive_pairs"]].to_numpy(float)
        Xs = (Xs - Xs.mean(0)) / Xs.std(0)
        null_g.append(_hmm_bic_gain(Xs, seed=i))
    null_g = np.array(null_g)
    add("H2 hidden states / regimes in draw structure", "BIC gain of 2-state Gaussian HMM over 1 state (vs fair histories)", gain,
        (np.sum(null_g >= gain) + 1) / (len(null_g) + 1),
        "A positive BIC gain clearly beyond what fair histories produce, and regimes that predict out of sample",
        f"null mean {null_g.mean():.2f}, max {null_g.max():.2f}")

    # H3 frequency drift: some window of 250 draws far from uniform
    obs_r = rolling_chi2_max(ind)
    null_r = np.array([rolling_chi2_max(indicator(fair_draws(rng, N))) for _ in range(n_null)])
    add("H3 frequency drift (machine/ball bias in some period)", "max rolling-250-draw chi-square (vs fair histories)", obs_r,
        (np.sum(null_r >= obs_r) + 1) / (n_null + 1),
        "A window whose deviation exceeds the fair-history maximum, followed by persistence out of sample",
        f"null mean {null_r.mean():.1f}, 99th pct {np.quantile(null_r, 0.99):.1f}")

    # H4 linear predictability of draw-level properties from the past
    for col in ["sum", "odd", "low", "range"]:
        f, p = ar_f_test(feats[col].to_numpy(float))
        add(f"H4 AR(10) predictability of draw {col}", "OLS F-test that 10 lags jointly have no effect", f, p,
            "Jointly significant lag coefficients that also improve out-of-sample forecasts")

    # H5 Markov dependence of number appearance (aggregate)
    A, B = ind[:-1].astype(float), ind[1:].astype(float)
    p0 = PICK / POOL
    T = A.T @ B
    e = (N - 1) * p0 * p0
    stat = float(((T - e) ** 2 / e).sum())
    nullm = []
    for _ in range(n_null):
        s = ind[rng.permutation(N)].astype(float)
        Tm = s[:-1].T @ s[1:]
        nullm.append(float(((Tm - e) ** 2 / e).sum()))
    nullm = np.array(nullm)
    add("H5 first-order Markov dependence between consecutive draws", "transition-matrix chi-square (vs shuffled draw order)", stat,
        (np.sum(nullm >= stat) + 1) / (n_null + 1),
        "Transition counts outside the shuffled-order distribution, and a Markov model that beats uniform in walk-forward tests")

    out = pd.DataFrame(rows)
    from statsmodels.stats.multitest import multipletests
    out["p_fdr"] = multipletests(out["p_value"], method="fdr_bh")[1]
    out["verdict"] = np.where(out["p_fdr"] < 0.05,
                              "EVIDENCE OF STRUCTURE (survives FDR) - must also predict out of sample",
                              "REJECTED - no evidence beyond what a fair random process produces")
    log.info("hypotheses: %d tests, %d survive FDR", len(out), int((out.p_fdr < 0.05).sum()))
    return out
