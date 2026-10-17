"""Phase 8: leakage-safe walk-forward backtesting.

Chronology (indices into the draw history, N draws):

    [0, start)                  warm-up only (never scored)
    [start, tune_end)           TUNING     - hyper-parameters chosen here
    [tune_end, hold_start)      SELECTION  - model families compared, final model chosen here
    [hold_start, N)             HOLDOUT    - untouched until the final model is fixed

Every forecast for draw t uses draws < t only (statistical models: by construction;
ML models: history-only features + expanding-window refits).  Hyper-parameters and the
ensemble's members are chosen on TUNING only; the final model is chosen on SELECTION only.
"""
from __future__ import annotations

import json
import time
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats

from .config import N_JOBS, load_config, p, setup_logging
from .models.advanced import walk_forward_gru, walk_forward_regime
from .models.ml import ML_MODELS, build_features, walk_forward_ml
from .models.statistical import SEARCH_SPACES, STAT_MODELS
from .simulation import fair_draws, indicator

log = setup_logging("loto.backtest")
POOL, PICK = 42, 6
P0 = PICK / POOL
HYP = stats.hypergeom(POOL, PICK, PICK)
HIT_PMF = HYP.pmf(np.arange(7))
HIT_MEAN = PICK * PICK / POOL                    # 0.857
HIT_VAR = float((HIT_PMF * (np.arange(7) - HIT_MEAN) ** 2).sum())


# ----------------------------------------------------------------------------- helpers
def periods(N: int) -> dict[str, tuple[int, int]]:
    cfg = load_config()["backtest"]
    start = int(cfg["min_train_draws"])
    tune_end = int(round(N * cfg["tuning_end_fraction"]))
    hold_start = int(round(N * (1 - cfg["holdout_fraction"])))
    return {"tuning": (start, tune_end), "selection": (tune_end, hold_start), "holdout": (hold_start, N),
            "all_oos": (start, N)}


def tickets_from_probs(P: np.ndarray, seed: int = 0) -> np.ndarray:
    """Top-6 numbers per row (ties broken by tiny seeded noise). Returns (N, 6) 1-based sorted."""
    rng = np.random.default_rng(seed)
    noisy = P + rng.random(P.shape) * 1e-9
    top = np.argsort(-noisy, axis=1)[:, :PICK] + 1
    return np.sort(top, axis=1)


def hits_of(tickets: np.ndarray, ind: np.ndarray) -> np.ndarray:
    return ind[np.arange(len(ind))[:, None], tickets - 1].sum(1)


def log_score(P: np.ndarray, ind: np.ndarray) -> np.ndarray:
    Pc = np.clip(P, 1e-4, 1 - 1e-4)
    return (ind * np.log(Pc) + (1 - ind) * np.log(1 - Pc)).sum(1)


UNIFORM_LOGSCORE = float(PICK * np.log(P0) + (POOL - PICK) * np.log(1 - P0))


def brier(P: np.ndarray, ind: np.ndarray) -> np.ndarray:
    return ((P - ind) ** 2).mean(1)


@lru_cache(maxsize=64)
def total_hits_null_pmf(n: int) -> np.ndarray:
    """Exact pmf of the total hits of n independent tickets under a fair lottery."""
    pmf = np.array([1.0])
    base = HIT_PMF
    k = n
    acc = np.array([1.0])
    while k:
        if k & 1:
            acc = np.convolve(acc, base)
        base = np.convolve(base, base)
        k >>= 1
    return acc


def hac_mean_test(d: np.ndarray, lags: int = 5):
    """Mean of d with Newey-West standard error; one-sided p for mean > 0."""
    d = np.asarray(d, float)
    n = len(d)
    m = d.mean()
    x = d - m
    gamma0 = (x @ x) / n
    s = gamma0
    for L in range(1, lags + 1):
        g = (x[L:] @ x[:-L]) / n
        s += 2 * (1 - L / (lags + 1)) * g
    se = np.sqrt(max(s, 1e-18) / n)
    t = m / se
    return m, se, t, stats.norm.sf(t)


def calibration(P: np.ndarray, ind: np.ndarray, bins: int = 10):
    p = P.ravel(); y = ind.ravel()
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    ece = 0.0
    for k in range(bins):
        s = b == k
        if s.any():
            ece += s.mean() * abs(p[s].mean() - y[s].mean())
    # calibration slope: logistic regression of y on logit(p) (slope 1 = calibrated)
    lp = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    if lp.std() < 1e-9:
        return float(ece), float("nan")
    import statsmodels.api as sm
    try:
        res = sm.Logit(y, sm.add_constant(lp)).fit(disp=0)
        slope = float(res.params[1])
    except Exception:
        slope = float("nan")
    return float(ece), slope


def evaluate(P: np.ndarray, ind: np.ndarray, lo: int, hi: int, seed: int = 0) -> dict:
    Ps, ys = P[lo:hi], ind[lo:hi]
    n = hi - lo
    tk = tickets_from_probs(Ps, seed)
    h = hits_of(tk, ys)
    tot = int(h.sum())
    pmf = total_hits_null_pmf(n)
    p_hits = float(pmf[tot:].sum())
    ls = log_score(Ps, ys)
    m, se, t, p_ls = hac_mean_test(ls - UNIFORM_LOGSCORE)
    ece, slope = calibration(Ps, ys)
    dist = np.bincount(h, minlength=7)[:7]
    sd = h.std(ddof=1)
    return {
        "n_draws": n,
        "mean_hits": float(h.mean()),
        "hits_ci_low": float(h.mean() - 1.96 * sd / np.sqrt(n)),
        "hits_ci_high": float(h.mean() + 1.96 * sd / np.sqrt(n)),
        "random_expected_hits": HIT_MEAN,
        "hits_vs_random": float(h.mean() - HIT_MEAN),
        "p_hits_better_than_random": p_hits,
        **{f"hits_{k}": int(dist[k]) for k in range(7)},
        **{f"expected_hits_{k}": float(HIT_PMF[k] * n) for k in range(7)},
        "precision_at_6": float(h.mean() / 6),
        "mean_log_score": float(ls.mean()),
        "uniform_log_score": UNIFORM_LOGSCORE,
        "log_score_gain": float(m),
        "log_score_gain_se": float(se),
        "p_log_score_better_than_uniform": float(p_ls),
        "brier": float(brier(Ps, ys).mean()),
        "uniform_brier": float(P0 * (1 - P0)),
        "ece": ece,
        "calibration_slope": slope,
    }


# ----------------------------------------------------------------------------- tuning
def tune_statistical(ind: np.ndarray, lo: int, hi: int) -> tuple[dict, list[dict]]:
    best_params, trials = {}, []
    for name, (fn, defaults) in STAT_MODELS.items():
        space = SEARCH_SPACES.get(name)
        if not space:
            best_params[name] = defaults
            continue
        (key, values), = space.items()
        scores = []
        for v in values:
            params = {**defaults, key: v}
            s = float(log_score(fn(ind, **params)[lo:hi], ind[lo:hi]).mean())
            scores.append(s)
            trials.append({"model": name, "params": json.dumps(params), "tuning_log_score": s})
        best_params[name] = {**defaults, key: values[int(np.argmax(scores))]}
    return best_params, trials


def tune_ml_optuna(name: str, ind: np.ndarray, X: np.ndarray, lo: int, hi: int, n_trials: int, seed: int):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        if name == "lightgbm":
            params = {"num_leaves": trial.suggest_int("num_leaves", 4, 31), "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                      "n_estimators": trial.suggest_int("n_estimators", 50, 300), "min_child_samples": trial.suggest_int("min_child_samples", 50, 2000, log=True),
                      "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 50, log=True)}
        else:  # xgboost
            params = {"max_depth": trial.suggest_int("max_depth", 2, 6), "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                      "n_estimators": trial.suggest_int("n_estimators", 50, 300), "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 50, log=True)}
        P, _, _ = walk_forward_ml(name, ind, X, lo, hi, refit_every=max(100, (hi - lo) // 6), params=params, seed=seed)
        return float(log_score(P[lo:hi], ind[lo:hi]).mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    trials = [{"model": name, "params": json.dumps(t.params), "tuning_log_score": t.value} for t in study.trials]
    return study.best_params, trials


# ----------------------------------------------------------------------------- main run
def run_backtest(df: pd.DataFrame, quick: bool = False) -> dict:
    cfg = load_config()["backtest"]
    seed = cfg["seed"]
    nums = df[[f"n{i}" for i in range(1, 7)]].to_numpy()
    ind = indicator(nums).astype(np.int8)
    N = len(ind)
    per = periods(N)
    lo_t, hi_t = per["tuning"]
    start = per["all_oos"][0]
    refit = int(cfg["refit_every"])
    log.info("periods: %s", per)

    probs: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}
    all_trials: list[dict] = []

    # statistical family -------------------------------------------------------
    t0 = time.time()
    best_stat, trials = tune_statistical(ind, lo_t, hi_t)
    all_trials += trials
    for name, (fn, _) in STAT_MODELS.items():
        ts = time.time()
        probs[name] = fn(ind, **best_stat[name])
        meta[name] = {"family": "baseline" if name == "uniform_random" else "statistical", "params": best_stat[name],
                      "compute_seconds": time.time() - ts, "n_fits": 1}
    log.info("statistical family done in %.1fs", time.time() - t0)

    # ML family ------------------------------------------------------------------
    X, fnames = build_features(ind)
    ml_list = ["logistic_l2", "lightgbm"] if quick else ML_MODELS
    ml_params = {}
    for name in ("lightgbm", "xgboost"):
        if name in ml_list:
            bp, tr = tune_ml_optuna(name, ind, X, lo_t, hi_t, n_trials=4 if quick else 15, seed=seed)
            ml_params[name] = bp
            all_trials += tr
    for name in ml_list:
        r = refit * (2 if name in ("random_forest", "catboost") else 1)
        P, secs, nf = walk_forward_ml(name, ind, X, start, N, r, ml_params.get(name), seed)
        probs[name] = P
        meta[name] = {"family": "machine_learning", "params": ml_params.get(name, "defaults (regularised)"),
                      "compute_seconds": secs, "n_fits": nf, "refit_every": r}
        log.info("%s done in %.1fs (%d fits)", name, secs, nf)

    # advanced family ------------------------------------------------------------
    try:
        P, secs, nf = walk_forward_gru(ind, start, N, refit_every=250 if quick else 100, seed=seed)
        probs["gru_sequence"] = P
        meta["gru_sequence"] = {"family": "advanced", "params": {"seq_len": 20, "hidden": 32}, "compute_seconds": secs, "n_fits": nf}
        log.info("gru done in %.1fs", secs)
    except ImportError:
        log.warning("PyTorch not installed - GRU model skipped")
    P, secs, nf = walk_forward_regime(ind, start, N, refit_every=refit, seed=seed)
    probs["kmeans_regime"] = P
    meta["kmeans_regime"] = {"family": "advanced", "params": {"k": 3, "window": 50}, "compute_seconds": secs, "n_fits": nf}

    # ensemble: members chosen on the TUNING period only ---------------------------
    tune_scores = {m: float(log_score(Pm[lo_t:hi_t], ind[lo_t:hi_t]).mean()) for m, Pm in probs.items() if m != "uniform_random"}
    members = sorted(tune_scores, key=tune_scores.get, reverse=True)[:3]
    probs["ensemble_top3"] = np.mean([probs[m] for m in members], axis=0)
    meta["ensemble_top3"] = {"family": "ensemble", "params": {"members": members}, "compute_seconds": 0.0, "n_fits": 0}

    # evaluation -------------------------------------------------------------------
    rows, per_draw = [], []
    for name, Pm in probs.items():
        for pname, (lo, hi) in per.items():
            ev = evaluate(Pm, ind, lo, hi, seed)
            rows.append({"model": name, "family": meta[name]["family"], "period": pname, **ev,
                         "compute_seconds": meta[name]["compute_seconds"], "params": json.dumps(meta[name]["params"], default=str)})
        tk = tickets_from_probs(Pm[start:], seed)
        h = hits_of(tk, ind[start:])
        ls = log_score(Pm[start:], ind[start:])
        for k in range(N - start):
            t = start + k
            pname = next(n for n, (a, b) in per.items() if n != "all_oos" and a <= t < b)
            per_draw.append({"model": name, "draw": int(df["draw"].iloc[t]), "draw_date": str(df["draw_date"].iloc[t])[:10],
                             "period": pname, "ticket": " ".join(map(str, tk[k])), "actual": " ".join(map(str, nums[t])),
                             "hits": int(h[k]), "log_score": float(ls[k]), "log_score_gain": float(ls[k] - UNIFORM_LOGSCORE)})
    comp = pd.DataFrame(rows)
    # multiple-comparison correction across models within each period
    from statsmodels.stats.multitest import multipletests
    for pname in per:
        m = comp["period"] == pname
        comp.loc[m, "p_hits_fdr_across_models"] = multipletests(comp.loc[m, "p_hits_better_than_random"], method="fdr_bh")[1]
        comp.loc[m, "p_logscore_fdr_across_models"] = multipletests(comp.loc[m, "p_log_score_better_than_uniform"], method="fdr_bh")[1]

    # final model chosen on SELECTION only --------------------------------------------
    sel = comp[comp.period == "selection"].set_index("model")
    cand = sel.drop(index="uniform_random")
    best = cand["log_score_gain"].idxmax()
    best_sig = bool(cand.loc[best, "p_logscore_fdr_across_models"] < 0.05 or cand.loc[best, "p_hits_fdr_across_models"] < 0.05)
    final_model = best if (best_sig and cand.loc[best, "log_score_gain"] > 0) else "uniform_random"
    hold = comp[comp.period == "holdout"].set_index("model")
    verdicts = {}
    for mname in probs:
        if mname == "uniform_random":
            verdicts[mname] = "BASELINE"
            continue
        s_ok = sel.loc[mname, "log_score_gain"] > 0 and (sel.loc[mname, "p_logscore_fdr_across_models"] < 0.05 or sel.loc[mname, "p_hits_fdr_across_models"] < 0.05)
        h_ok = hold.loc[mname, "p_hits_fdr_across_models"] < 0.05 or hold.loc[mname, "p_logscore_fdr_across_models"] < 0.05
        verdicts[mname] = "KEEP (validated)" if (s_ok and h_ok and hold.loc[mname, "log_score_gain"] > 0) else "REJECT (no edge over random on unseen draws)"
    comp["verdict"] = comp["model"].map(verdicts)

    # multi-ticket comparison at equal ticket counts (10 tickets / draw) ---------------
    portfolio = portfolio_comparison(probs, ind, per["holdout"], ["uniform_random", final_model, "ensemble_top3", best], seed)

    summary = {
        "periods": per, "n_draws": N, "final_model": final_model, "selection_best_model": best,
        "selection_best_significant": best_sig, "ensemble_members": members, "verdicts": verdicts,
        "n_tuning_configurations": len(all_trials), "tuned_params": {k: v["params"] for k, v in meta.items()},
        "total_compute_seconds": float(sum(v["compute_seconds"] for v in meta.values())),
        "total_model_fits": int(sum(v["n_fits"] for v in meta.values())),
        "hits_null": {"mean": HIT_MEAN, "var": HIT_VAR, "pmf": HIT_PMF.tolist()},
        "min_detectable_hits_effect_holdout": float(2.8 * np.sqrt(HIT_VAR / (per["holdout"][1] - per["holdout"][0]))),
        "feature_names": fnames,
    }
    return {"comparison": comp, "per_draw": pd.DataFrame(per_draw), "trials": pd.DataFrame(all_trials),
            "probs": probs, "meta": meta, "summary": summary, "portfolio": portfolio}


def gumbel_tickets(P: np.ndarray, n_tickets: int, rng) -> np.ndarray:
    """Sample n_tickets distinct-number tickets per row, without replacement, weighted by P."""
    logp = np.log(np.clip(P, 1e-9, None))
    out = np.empty((len(P), n_tickets, PICK), dtype=np.int16)
    for k in range(n_tickets):
        g = logp - np.log(-np.log(rng.random(P.shape)))
        out[:, k] = np.sort(np.argsort(-g, axis=1)[:, :PICK] + 1, axis=1)
    return out


def portfolio_comparison(probs, ind, rng_period, models, seed, n_tickets: int = 10) -> pd.DataFrame:
    lo, hi = rng_period
    rows = []
    for mname in dict.fromkeys(models):
        rng = np.random.default_rng(seed + 10)
        T = gumbel_tickets(probs[mname][lo:hi], n_tickets, rng)
        y = ind[lo:hi]
        h = np.stack([y[np.arange(hi - lo)[:, None], T[:, k] - 1].sum(1) for k in range(n_tickets)], 1)
        rows.append({"model": mname, "n_tickets": n_tickets, "period": "holdout", "n_draws": hi - lo,
                     "mean_hits_per_ticket": float(h.mean()), "mean_best_ticket_hits": float(h.max(1).mean()),
                     "draws_with_any_3plus": int((h.max(1) >= 3).sum()), "draws_with_any_4plus": int((h.max(1) >= 4).sum())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- bonus experiment
def bonus_experiment(df: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward: can past bonus numbers predict the next bonus number?

    Strategies predict one bonus number per draw using only earlier draws. Unconditional
    chance of guessing the bonus is 1/42. (If the six main numbers were known in advance, a
    guess among the remaining 36 would succeed 1/36 of the time; the main numbers are not
    known before the draw, so 1/42 is the relevant baseline.)
    """
    b = df["bonus"].to_numpy()
    N = len(b)
    per = periods(N)
    rng = np.random.default_rng(load_config()["backtest"]["seed"] + 77)
    counts = np.zeros(43)
    preds = {"hot_bonus": [], "cold_bonus": [], "repeat_last_bonus": [], "random": []}
    for t in range(N):
        if t >= per["all_oos"][0]:
            c = counts[1:]
            preds["hot_bonus"].append(int(np.argmax(c + rng.random(42) * 1e-6) + 1))
            preds["cold_bonus"].append(int(np.argmin(c + rng.random(42) * 1e-6) + 1))
            preds["repeat_last_bonus"].append(int(b[t - 1]))
            preds["random"].append(int(rng.integers(1, 43)))
        counts[b[t]] += 1
    actual = b[per["all_oos"][0]:]
    rows = []
    for name, pr in preds.items():
        pr = np.array(pr)
        for pname in ("selection", "holdout", "all_oos"):
            lo, hi = per[pname]
            sl = slice(lo - per["all_oos"][0], hi - per["all_oos"][0])
            hits = int((pr[sl] == actual[sl]).sum())
            n = hi - lo
            rows.append({"strategy": name, "period": pname, "n_draws": n, "correct": hits, "hit_rate": hits / n,
                         "chance_rate": 1 / 42, "p_better_than_chance": float(stats.binom.sf(hits - 1, n, 1 / 42))})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- negative controls
def _stat_edges(ind: np.ndarray, params: dict, per: dict) -> dict:
    out = {}
    for name, (fn, _) in STAT_MODELS.items():
        if name == "uniform_random":
            continue
        P = fn(ind, **params[name])
        for pname in ("holdout", "all_oos"):
            lo, hi = per[pname]
            tk = tickets_from_probs(P[lo:hi])
            out[(name, pname, "hits")] = float(hits_of(tk, ind[lo:hi]).mean() - HIT_MEAN)
            out[(name, pname, "logscore")] = float(log_score(P[lo:hi], ind[lo:hi]).mean() - UNIFORM_LOGSCORE)
    return out


def negative_controls(df: pd.DataFrame, params: dict, n_reps: int = 20) -> pd.DataFrame:
    """Run the statistical models on shuffled-order and fully simulated histories."""
    from joblib import Parallel, delayed
    nums = df[[f"n{i}" for i in range(1, 7)]].to_numpy()
    ind = indicator(nums).astype(np.int8)
    N = len(ind)
    per = periods(N)
    seed = load_config()["backtest"]["seed"]

    def one(kind, r):
        rng = np.random.default_rng(seed * 1000 + r + (0 if kind == "shuffled" else 50000))
        x = ind[rng.permutation(N)] if kind == "shuffled" else indicator(fair_draws(rng, N)).astype(np.int8)
        return kind, r, _stat_edges(x, params, per)

    res = Parallel(n_jobs=N_JOBS)(delayed(one)(k, r) for k in ("shuffled", "simulated") for r in range(n_reps))
    real = _stat_edges(ind, params, per)
    rows = []
    for (name, pname, metric), v in real.items():
        ctrl = {k: np.array([e[(name, pname, metric)] for kk, _, e in res if kk == k]) for k in ("shuffled", "simulated")}
        rows.append({"model": name, "period": pname, "metric": metric, "real_edge": v,
                     **{f"{k}_mean": c.mean() for k, c in ctrl.items()},
                     **{f"{k}_sd": c.std(ddof=1) for k, c in ctrl.items()},
                     **{f"{k}_max": c.max() for k, c in ctrl.items()},
                     "real_rank_vs_shuffled": float((ctrl["shuffled"] >= v).mean()),
                     "real_rank_vs_simulated": float((ctrl["simulated"] >= v).mean()),
                     "n_reps": n_reps})
    return pd.DataFrame(rows)
