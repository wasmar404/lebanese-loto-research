"""Phase 10: candidate tickets for the next draw.

Candidates come from several methods.  Each is reported with its structural profile and the
out-of-sample backtest record of the method that produced it.  No candidate is assigned a
winning probability other than the exact 1 / 5,245,786 that every valid ticket has.
"""
from __future__ import annotations

import json
import re
from datetime import datetime

import numpy as np
import pandas as pd

from .config import load_config, p, setup_logging
from .features import exact_feature_table, structural_features
from .filters import MODES, excluded_mask
from .models.statistical import STAT_MODELS
from .simulation import indicator

log = setup_logging("loto.predict")
POOL, PICK = 42, 6
N_COMB = 5_245_786


def next_draw_info(df: pd.DataFrame) -> dict:
    """Next draw number and date: from the live results page when reachable, else the schedule."""
    latest = int(df["draw"].max())
    info = {"draw": latest + 1, "date": None, "source": "schedule"}
    try:
        import httpx
        cfg = load_config()["scraper"]
        html = httpx.get(cfg["latest_url"], timeout=20, headers={"User-Agent": cfg["user_agent"]}).text
        m = re.search(r'id="footercountdowntime"[^>]*value="(\d{1,2})/(\d{1,2})/(\d{4})\s+(\d{1,2}):(\d{2})', html)
        lm = re.search(r'id="resultdateinput"[^>]*value="(\d+)"', html)
        if lm and int(lm.group(1)) > latest:
            info["warning"] = f"site shows draw {lm.group(1)} which is newer than the dataset - run `update`"
        if m:
            mo, d, y, hh, mm = map(int, m.groups())
            info.update(date=f"{y:04d}-{mo:02d}-{d:02d}", time=f"{hh:02d}:{mm:02d}", source="lldj.com countdown")
            return info
    except Exception as exc:  # offline
        info["note"] = f"live page unavailable ({type(exc).__name__}); date inferred from schedule"
    last = pd.Timestamp(df["draw_date"].max())
    wd = df["draw_date"].tail(100).dt.dayofweek.value_counts().index.tolist()[:2]
    d = last + pd.Timedelta(days=1)
    while d.dayofweek not in wd:
        d += pd.Timedelta(days=1)
    info["date"] = d.strftime("%Y-%m-%d")
    return info


def _profile(t: list[int]) -> dict:
    f = structural_features(np.array([t])).iloc[0]
    dec = "/".join(str(int(f[f"dec_{l}"])) for l in ["1-9", "10-19", "20-29", "30-39", "40-42"])
    return {"sum": int(f["sum"]), "odd_even": f"{int(f['odd'])}/{6 - int(f['odd'])}", "low_high": f"{int(f['low'])}/{6 - int(f['low'])}",
            "decades_1-9/10s/20s/30s/40s": dec, "consecutive_pairs": int(f["consecutive_pairs"]),
            "_odd": int(f["odd"]), "_low": int(f["low"]), "_decades_used": int(f["decades_used"]), "_le31": int(f["le31"])}


def _structural_frequency(prof: dict, hist_feats: pd.DataFrame) -> tuple[float, float]:
    """Share of historical draws (and of all combinations) with the same odd count, low count
    and number of decades used - a coarse 'how typical does this look' measure."""
    m = (hist_feats["odd"] == prof["_odd"]) & (hist_feats["low"] == prof["_low"]) & (hist_feats["decades_used"] == prof["_decades_used"])
    ex = exact_feature_table()
    me = (ex["odd"] == prof["_odd"]) & (ex["low"] == prof["_low"]) & (ex["decades_used"] == prof["_decades_used"])
    return float(m.mean()), float(me.mean())


def _top6(prob: np.ndarray) -> list[int]:
    return sorted((np.argsort(-prob)[:PICK] + 1).tolist())


def _sample(prob: np.ndarray, rng, filter_mode: str = "none", tries: int = 10000) -> list[int]:
    for _ in range(tries):
        g = np.log(np.clip(prob, 1e-12, None)) - np.log(-np.log(rng.random(POOL)))
        t = sorted((np.argsort(-g)[:PICK] + 1).tolist())
        if filter_mode == "none" or not excluded_mask(structural_features(np.array([t])), MODES[filter_mode])[0]:
            return t
    raise RuntimeError("could not sample a ticket satisfying the filter")


def _low_overlap_portfolio(rng, n: int, existing: list[list[int]], prob: np.ndarray, filter_mode: str) -> list[list[int]]:
    """Simulated annealing: choose n tickets minimising overlap with each other and with existing
    tickets (maximises coverage of distinct numbers - it does NOT raise any ticket's odds)."""
    tickets = [_sample(prob, rng, filter_mode) for _ in range(n)]

    def cost(ts):
        allt = ts + existing
        c = 0
        for i in range(len(ts)):
            for j in range(len(allt)):
                if j != i:
                    c += len(set(ts[i]) & set(allt[j])) ** 2
        return c

    cur = cost(tickets)
    T = 2.0
    for it in range(4000):
        i = rng.integers(n)
        cand = [t[:] for t in tickets]
        out_num = rng.choice(cand[i])
        choices = [x for x in range(1, POOL + 1) if x not in cand[i]]
        cand[i] = sorted([x for x in cand[i] if x != out_num] + [int(rng.choice(choices))])
        if filter_mode != "none" and excluded_mask(structural_features(np.array([cand[i]])), MODES[filter_mode])[0]:
            continue
        c = cost(cand)
        if c <= cur or rng.random() < np.exp((cur - c) / T):
            tickets, cur = cand, c
        T *= 0.999
    return tickets


def generate_candidates(df: pd.DataFrame, backtest_summary: dict, comparison: pd.DataFrame, final_probs: dict[str, np.ndarray],
                        n_candidates: int = 10, filter_mode: str = "none", seed: int | None = None,
                        popularity: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    cfg = load_config()["prediction"]
    seed = cfg["seed"] if seed is None else seed
    rng = np.random.default_rng(seed)
    nums = df[[f"n{i}" for i in range(1, 7)]].to_numpy()
    hist_feats = structural_features(nums)
    final_model = backtest_summary["final_model"]
    no_signal = final_model == "uniform_random"
    hold = comparison[comparison.period == "holdout"].set_index("model")

    def record(model):
        if model not in hold.index:
            return "n/a"
        r = hold.loc[model]
        return (f"holdout {int(r.n_draws)} draws: {r.mean_hits:.3f} hits/ticket vs 0.857 random "
                f"(p={r.p_hits_better_than_random:.3f}); log-score gain {r.log_score_gain:+.4f}; {r.verdict}")

    uni = np.full(POOL, PICK / POOL)
    p_final = final_probs.get(final_model, uni)
    p_ens = final_probs.get("ensemble_top3", uni)
    sel_best = backtest_summary.get("selection_best_model", "ensemble_top3")
    p_alt = final_probs.get(sel_best, uni)

    plan: list[tuple[str, str, list[int]]] = []
    if no_signal:
        lab = "Random (no model beat random selection out of sample)"
        plan.append((lab, "uniform_random", _sample(uni, rng, filter_mode)))
    else:
        plan.append(("Best validated model (top-6 probabilities)", final_model, _top6(p_final)))
    plan.append(("Ensemble of top-3 tuning-period models (top-6)" + (" - NOT validated" if no_signal else ""), "ensemble_top3", _top6(p_ens)))
    plan.append((f"Probability-weighted sample from {final_model if not no_signal else 'uniform'}", final_model, _sample(p_final, rng, filter_mode)))
    plan.append(("Conservative structural filter, random within filtered space", "uniform_random", _sample(uni, rng, "conservative" if filter_mode == "none" else filter_mode)))
    plan.append((f"Alternative model: best on selection period ({sel_best}) top-6" + (" - NOT validated" if no_signal else ""), sel_best, _top6(p_alt)))
    # unpopular-pattern candidate: only if the winner-count data show numbers <= 31 are over-played
    pop_ok = False
    if popularity is not None and len(popularity) and {"feature", "coef_log_winners", "p_value", "prize_rank"} <= set(popularity.columns):
        r5 = popularity[(popularity.prize_rank == 5) & (popularity.feature == "le31")]
        pop_ok = bool(len(r5) and r5.coef_log_winners.iloc[0] > 0 and r5.p_value.iloc[0] < 0.05)
    if pop_ok:
        best_t, best_s = None, np.inf
        for _ in range(3000):
            t = _sample(uni, rng, "conservative")
            pr = _profile(t)
            s = pr["_le31"] * 1.0 + 0.5 * pr["consecutive_pairs"]
            if s < best_s:
                best_t, best_s = t, s
        plan.append(("Low-popularity pattern (random, avoids widely played patterns: few numbers ≤31) - same odds, larger share if it wins",
                     "uniform_random", best_t))
    existing = [t for _, _, t in plan]
    n_left = n_candidates - len(plan)
    n_div = max(0, n_left - 1)
    for t in _low_overlap_portfolio(rng, n_div, existing, p_final if not no_signal else uni, filter_mode):
        plan.append(("Diverse portfolio / low-overlap (simulated annealing for number coverage)", final_model, t))
    while len(plan) < n_candidates:
        plan.append(("Additional random valid ticket", "uniform_random", _sample(uni, rng, filter_mode)))

    # de-duplicate
    seen, rows = set(), []
    for method, model, t in plan:
        while tuple(t) in seen:
            t = _sample(uni, rng, filter_mode)
        seen.add(tuple(t))
        prof = _profile(t)
        hf, ef = _structural_frequency(prof, hist_feats)
        rows.append({"numbers": " ".join(f"{x:02d}" for x in t), **{f"n{i + 1}": t[i] for i in range(6)}, "method": method,
                     "source_model": model, **{k: v for k, v in prof.items() if not k.startswith("_")},
                     "structural_class_hist_share": round(hf, 4), "structural_class_exact_share": round(ef, 4),
                     "method_backtest": record(model), "win_probability_jackpot": f"1 / {N_COMB:,}"})
    cand = pd.DataFrame(rows)
    sets = [set(map(int, r.split())) for r in cand["numbers"]]
    cand["max_overlap_with_other_tickets"] = [max(len(s & o) for j, o in enumerate(sets) if j != i) for i, s in enumerate(sets)]
    cand.insert(0, "rank", range(1, len(cand) + 1))
    info = {"no_signal": no_signal, "final_model": final_model, "filter_mode": filter_mode, "seed": seed,
            "distinct_numbers_covered": len(set().union(*sets))}
    return cand, info


def explanation_markdown(target: dict, cand: pd.DataFrame, info: dict, summary: dict, comparison: pd.DataFrame) -> str:
    hold = comparison[comparison.period == "holdout"].sort_values("log_score_gain", ascending=False)
    lines = [f"# Candidate tickets for Lebanese Loto draw #{target['draw']}",
             "",
             f"Target draw: **#{target['draw']}** on **{target.get('date')}** {target.get('time', '')} (source: {target.get('source')}).",
             f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} from {summary['n_draws']} validated historical draws.",
             ""]
    if info["no_signal"]:
        lines += ["## Bottom line",
                  "",
                  "**No model beat random selection on unseen draws.** After tuning on the early history, choosing between "
                  f"{len(summary['verdicts']) - 1} models on a separate selection period, and testing on an untouched holdout of "
                  f"{summary['periods']['holdout'][1] - summary['periods']['holdout'][0]} draws, no method's advantage was larger than "
                  "what chance produces. The final method is therefore **uniform random selection**.",
                  "",
                  "Every ticket below has exactly the same chance of winning the jackpot as any other valid combination: "
                  "**1 in 5,245,786**. The tickets labelled with model names are shown for transparency only; their method failed validation.",
                  ""]
    else:
        lines += ["## Bottom line", "", f"The selected model is **{info['final_model']}**. See its holdout record in the table.", ""]
    show = cand[["rank", "numbers", "method", "sum", "odd_even", "decades_1-9/10s/20s/30s/40s", "structural_class_hist_share",
                 "max_overlap_with_other_tickets"]]
    lines += ["## Candidates", "", show.to_markdown(index=False), "",
              f"The 10 tickets cover {info['distinct_numbers_covered']} distinct numbers. Diversifying tickets changes *which* outcomes you "
              "cover, not the probability of any individual ticket.", "",
              "## Backtest record of each method (untouched holdout)", "",
              hold[["model", "family", "n_draws", "mean_hits", "p_hits_better_than_random", "log_score_gain", "p_log_score_better_than_uniform", "verdict"]]
              .to_markdown(index=False, floatfmt=".4f"), "",
              "Random selection averages 0.857 matching main numbers per ticket.", "",
              "## How to read this", "",
              "* `structural_class_hist_share` is the share of past draws with the same odd count, low count and number of decades used. "
              "It describes how typical a ticket looks; it is **not** a probability of winning.",
              "* The low-popularity candidate avoids patterns many players choose (mostly numbers ≤ 31). It has the same odds but, "
              "in prize tiers shared between winners, it would be shared with fewer people.",
              "* Lotteries are designed to return less than is staked. Play only amounts you are comfortable losing."]
    return "\n".join(lines)
