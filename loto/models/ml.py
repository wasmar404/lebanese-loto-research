"""Machine-learning model family: per-number inclusion classifiers on a long (draw x number)
table of history-only features, refit on an expanding window every ``refit_every`` draws.

Features for draw t use draws < t only; preprocessing (scaling) lives inside the sklearn
pipeline and is fitted on training rows only.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from ..config import setup_logging

log = setup_logging("loto.ml")
POOL, PICK = 42, 6
P0 = PICK / POOL
WINDOWS = (10, 20, 50, 100, 250)


def build_features(ind: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Returns X of shape (N, 42, F): features of number i for draw t from draws < t."""
    N = len(ind)
    cs = np.vstack([np.zeros((1, POOL)), np.cumsum(ind, 0)])
    t = np.arange(N)
    feats, names = [], []
    for w in WINDOWS:
        lo = np.maximum(0, t - w)
        n = np.maximum(t - lo, 1)[:, None]
        feats.append((cs[t] - cs[lo]) / n); names.append(f"freq_{w}")
    feats.append(cs[t] / np.maximum(t, 1)[:, None]); names.append("freq_all")

    gap = np.zeros((N, POOL)); mean_gap = np.zeros((N, POOL))
    ew10 = np.zeros((N, POOL)); ew50 = np.zeros((N, POOL))
    lag = np.zeros((3, N, POOL))
    mk = np.zeros((N, POOL)); pa = np.zeros((N, POOL))
    last = np.full(POOL, -1); gsum = np.zeros(POOL); gcnt = np.zeros(POOL)
    e10 = np.full(POOL, P0); e50 = np.full(POOL, P0)
    l10, l50 = 0.5 ** (1 / 10), 0.5 ** (1 / 50)
    T = np.zeros((POOL, POOL)); nj = np.zeros(POOL); C = np.zeros((POOL, POOL)); cnt = np.zeros(POOL)
    q = (PICK - 1) / (POOL - 1)
    for k in range(N):
        cur = np.where(last >= 0, k - last, k + 1)
        gap[k] = np.minimum(cur, 300)
        mean_gap[k] = np.where(gcnt > 0, gsum / np.maximum(gcnt, 1), 1 / P0)
        ew10[k], ew50[k] = e10, e50
        for j in range(3):
            if k - 1 - j >= 0:
                lag[j, k] = ind[k - 1 - j]
        if k >= 1:
            js = np.flatnonzero(ind[k - 1])
            if k >= 2:
                mk[k] = ((T[js] + 20 * P0) / (nj[js, None] + 20)).mean(0) / P0
            pr = (C[js] + 5 * q) / (cnt[js, None] + 5) / q
            pr[np.arange(len(js)), js] = np.nan
            pa[k] = np.nanmean(pr, 0) if k >= 2 else 1.0
        # update state with draw k (becomes available for k+1)
        x = ind[k]
        seen = x == 1
        valid = last[seen] >= 0
        gsum[seen] += np.where(valid, k - last[seen], 0); gcnt[seen] += valid
        last[seen] = k
        e10 = l10 * e10 + (1 - l10) * x
        e50 = l50 * e50 + (1 - l50) * x
        if k >= 1:
            T += np.outer(ind[k - 1], x); nj += ind[k - 1]
        C += np.outer(x, x); cnt += x
    feats += [np.log1p(gap), gap / np.maximum(mean_gap, 1), ew10, ew50, lag[0], lag[1], lag[2], mk, np.nan_to_num(pa, nan=1.0)]
    names += ["log_gap", "overdue_ratio", "ewma_10", "ewma_50", "in_t-1", "in_t-2", "in_t-3", "markov_lift", "pair_lift"]
    # number identity (lets a global model learn per-number bias) and previous-draw structure
    num = np.broadcast_to(np.arange(1, POOL + 1, dtype=float), (N, POOL))
    feats += [num / POOL, (num % 2), np.floor(num / 10)]
    names += ["number_scaled", "number_odd", "number_decade"]
    prev_sum = np.zeros(N); prev_sum[1:] = (ind[:-1] * np.arange(1, POOL + 1)).sum(1)
    feats.append(np.broadcast_to(prev_sum[:, None] / 129.0, (N, POOL))); names.append("prev_draw_sum_scaled")
    X = np.stack(feats, axis=-1).astype(np.float32)
    return X, names


def make_estimator(name: str, params: dict | None = None, seed: int = 0):
    params = params or {}
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == "logistic_l2":
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(StandardScaler(), LogisticRegression(C=params.get("C", 0.05), max_iter=500))
    if name == "logistic_l1":
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(StandardScaler(), LogisticRegression(C=params.get("C", 0.01), penalty="l1", solver="liblinear", max_iter=500))
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=params.get("n_estimators", 150), max_depth=params.get("max_depth", 6),
                                      min_samples_leaf=params.get("min_samples_leaf", 200), n_jobs=-1, random_state=seed)
    if name == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=params.get("max_iter", 150), learning_rate=params.get("learning_rate", 0.03),
                                              max_depth=params.get("max_depth", 3), l2_regularization=1.0, random_state=seed)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=params.get("n_estimators", 200), max_depth=params.get("max_depth", 3),
                             learning_rate=params.get("learning_rate", 0.03), subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=params.get("reg_lambda", 5.0), n_jobs=-1, random_state=seed, verbosity=0)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=params.get("n_estimators", 200), num_leaves=params.get("num_leaves", 15),
                              learning_rate=params.get("learning_rate", 0.03), min_child_samples=params.get("min_child_samples", 200),
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=params.get("reg_lambda", 5.0),
                              n_jobs=-1, random_state=seed, verbose=-1)
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=params.get("iterations", 200), depth=params.get("depth", 4),
                                  learning_rate=params.get("learning_rate", 0.05), l2_leaf_reg=5, random_seed=seed,
                                  verbose=False, thread_count=-1)
    raise ValueError(name)


ML_MODELS = ["logistic_l2", "logistic_l1", "random_forest", "hist_gradient_boosting", "xgboost", "lightgbm", "catboost"]


def walk_forward_ml(name: str, ind: np.ndarray, X: np.ndarray, start: int, end: int, refit_every: int,
                    params: dict | None = None, seed: int = 0, train_from: int = 50) -> tuple[np.ndarray, float, int]:
    """Probabilities for draws [start, end). Refit on draws [train_from, t) every refit_every draws."""
    N = len(ind)
    P = np.full((N, POOL), P0)
    t0 = time.time()
    n_fits = 0
    F = X.shape[-1]
    for block in range(start, end, refit_every):
        tr = np.arange(train_from, block)
        Xtr = X[tr].reshape(-1, F)
        ytr = ind[tr].reshape(-1)
        est = make_estimator(name, params, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(Xtr, ytr)
        n_fits += 1
        hi = min(block + refit_every, end)
        Xte = X[block:hi].reshape(-1, F)
        P[block:hi] = est.predict_proba(Xte)[:, 1].reshape(hi - block, POOL)
    return np.clip(P, 1e-4, 1 - 1e-4), time.time() - t0, n_fits


def feature_importance_snapshot(name: str, ind: np.ndarray, X: np.ndarray, names: list[str], end: int, params=None) -> pd.Series | None:
    est = make_estimator(name, params)
    F = X.shape[-1]
    est.fit(X[50:end].reshape(-1, F), ind[50:end].reshape(-1))
    model = est[-1] if hasattr(est, "__getitem__") else est
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=names).sort_values(ascending=False)
    if hasattr(model, "coef_"):
        return pd.Series(np.abs(model.coef_[0]), index=names).sort_values(ascending=False)
    return None
