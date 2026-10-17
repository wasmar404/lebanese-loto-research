"""End-to-end pipeline steps used by the CLI and the dashboard."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import ROOT, load_config, p, setup_logging

log = setup_logging("loto.pipeline")


def hardware() -> dict:
    info = {"cpus": os.cpu_count(), "platform": platform.platform(), "python": platform.python_version()}
    try:
        with open("/proc/meminfo") as fh:
            mem = {l.split(":")[0]: int(l.split()[1]) for l in fh if ":" in l}
        info["ram_gb"] = round(mem.get("MemTotal", 0) / 1024 ** 2, 1)
    except OSError:
        pass
    try:
        import torch
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        info["torch"] = torch.__version__
    except ImportError:
        info["gpu"] = None
    return info


def track(command: str, started: float, **results) -> None:
    cfg_hash = hashlib.sha256((ROOT / "config.yaml").read_bytes()).hexdigest()[:12]
    rec = {"timestamp": datetime.now(timezone.utc).isoformat(), "command": command, "seconds": round(time.time() - started, 1),
           "config_sha": cfg_hash, **results}
    with open(p("artifacts_dir") / "experiments.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


# ----------------------------------------------------------------------------- steps
def step_scrape(refresh_latest: int = 3) -> dict:
    from .scraper import scrape
    t0 = time.time()
    s = scrape(refresh_latest=refresh_latest)
    track("scrape", t0, latest=s["latest_draw"], downloaded=s["downloaded"], failed=len(s["failed"]))
    return s


def step_validate() -> dict:
    from .validate import run_validation
    t0 = time.time()
    r = run_validation()
    track("validate", t0, passed=r["passed"], n_draws=r["n_draws"])
    return r


def _load():
    from .storage import load_draws
    return load_draws(validated_only=True)


def _prizes() -> pd.DataFrame:
    import sqlite3
    con = sqlite3.connect(p("sqlite"))
    try:
        return pd.read_sql("select * from prizes", con)
    finally:
        con.close()


def step_simulate() -> dict:
    from .simulation import bulk_structural_frequencies, compare_to_null, history_statistics, simulate_histories
    from .features import exact_distribution
    t0 = time.time()
    df = _load()
    nums = df[[f"n{i}" for i in range(1, 7)]].to_numpy()
    real = history_statistics(nums)
    sims = simulate_histories(len(df))
    sims.to_parquet(p("artifacts_dir") / "simulated_histories.parquet", index=False)
    cmp = compare_to_null(real, sims)
    bulk = bulk_structural_frequencies()
    ex = []
    for feat in bulk["feature"].unique():
        pmf = exact_distribution(feat)
        for v, pr in pmf.items():
            ex.append({"feature": feat, "value": v, "exact_prob": pr})
    bulk = bulk.merge(pd.DataFrame(ex), on=["feature", "value"], how="outer").fillna({"mc_count": 0, "mc_prob": 0})
    bulk["mc_minus_exact"] = bulk["mc_prob"] - bulk["exact_prob"]
    out = pd.concat([cmp.assign(section="history_level_vs_fair_histories"),
                     bulk.assign(section="bulk_structural_mc_vs_exact")], ignore_index=True)
    out.to_csv(p("reports_dir") / "simulation_results.csv", index=False)
    bulk.to_csv(p("artifacts_dir") / "bulk_vs_exact.csv", index=False)
    cmp.to_csv(p("artifacts_dir") / "history_level_comparison.csv", index=False)
    n_sim_draws = int(len(sims) * len(df) + load_config()["simulation"]["n_draws_bulk"])
    track("simulate", t0, n_histories=len(sims), simulated_draws=n_sim_draws,
          extreme_stats=int((cmp.p_two_sided < 0.01).sum()))
    return {"n_histories": len(sims), "simulated_draws": n_sim_draws}


def step_analyze() -> dict:
    from .analysis import number_table, pair_table, pattern_report, popularity_analysis, statistical_report, triplet_table
    from .filters import evaluate_filters
    from .hypotheses import run_hypotheses
    from .randomness import run_randomness_tests, sim_number_counts
    from .backtest import periods
    from .simulation import fair_draws
    from .features import structural_features
    t0 = time.time()
    df = _load()
    R = p("reports_dir")
    numbers = number_table(df); numbers.to_csv(R / "number_statistics.csv", index=False)
    pairs = pair_table(df); pairs.to_csv(R / "pair_statistics.csv", index=False)
    trips = triplet_table(df); trips.to_csv(R / "triplet_statistics.csv", index=False)
    sc = sim_number_counts(len(df), 5000, load_config()["simulation"]["seed"] + 5)
    tests = run_randomness_tests(df, sim_counts=sc)
    tests.to_csv(R / "randomness_tests.csv", index=False)
    hyp = run_hypotheses(df)
    hyp.to_csv(R / "hypothesis_tests.csv", index=False)
    rng = np.random.default_rng(load_config()["simulation"]["seed"] + 7)
    sim_feats = structural_features(fair_draws(rng, 1_000_000))
    filt = evaluate_filters(df, periods(len(df))["holdout"][0], sim_features=sim_feats)
    filt.to_csv(R / "filter_evaluation.csv", index=False)
    try:
        pop_coef, pop_desc = popularity_analysis(df, _prizes())
        pop_coef.to_csv(R / "popularity_analysis.csv", index=False)
        pop_desc.to_csv(p("artifacts_dir") / "popularity_by_le31.csv", index=False)
    except Exception as exc:
        log.warning("popularity analysis failed: %s", exc)
        pop_coef = pop_desc = None
    sim_path = R / "simulation_results.csv"
    if not sim_path.exists():
        step_simulate()
    cmp = pd.read_csv(p("artifacts_dir") / "history_level_comparison.csv")
    bulk = pd.read_csv(p("artifacts_dir") / "bulk_vs_exact.csv")
    sims = pd.read_parquet(p("artifacts_dir") / "simulated_histories.parquet")
    bulk_show = bulk[bulk.feature.isin(["odd", "low", "longest_run", "decades_used"])]
    statistical_report(df, numbers, tests, cmp, bulk_show, R / "statistical_analysis.html", sims=sims)
    pattern_report(df, pairs, trips, hyp, filt, pop_coef, pop_desc, R / "pattern_analysis.html")
    track("analyze", t0, n_tests=len(tests), raw_sig=int(tests.significant_raw.sum()), fdr_sig=int(tests.significant_fdr.sum()),
          hypotheses_surviving_fdr=int((hyp.p_fdr < 0.05).sum()))
    return {"tests": len(tests), "fdr_significant": int(tests.significant_fdr.sum())}


def step_backtest(quick: bool = False, controls: int = 20) -> dict:
    from .backtest import negative_controls, run_backtest
    t0 = time.time()
    df = _load()
    res = run_backtest(df, quick=quick)
    R, A = p("reports_dir"), p("artifacts_dir")
    res["comparison"].to_csv(R / "model_comparison.csv", index=False)
    res["per_draw"].to_csv(R / "backtesting_results.csv", index=False)
    res["trials"].to_csv(R / "tuning_trials.csv", index=False)
    res["portfolio"].to_csv(R / "portfolio_comparison.csv", index=False)
    np.savez_compressed(A / "backtest_probs.npz", draws=df["draw"].to_numpy(), **{k: v.astype(np.float32) for k, v in res["probs"].items()})
    params = {k: v["params"] for k, v in res["meta"].items() if res["meta"][k]["family"] in ("statistical", "baseline")}
    if controls:
        nc = negative_controls(df, params, n_reps=controls)
        nc.to_csv(R / "negative_controls.csv", index=False)
    summary = res["summary"] | {"hardware": hardware(), "elapsed_seconds": time.time() - t0, "stat_params": params}
    (A / "backtest_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    track("backtest", t0, final_model=summary["final_model"], fits=summary["total_model_fits"],
          tuning_configs=summary["n_tuning_configurations"])
    return summary


def next_draw_probabilities(df: pd.DataFrame, summary: dict, models: list[str]) -> dict[str, np.ndarray]:
    """Refit the requested models on the complete history and forecast the next draw."""
    from .models.advanced import walk_forward_gru, walk_forward_regime
    from .models.ml import ML_MODELS, build_features, walk_forward_ml
    from .models.statistical import STAT_MODELS
    from .simulation import indicator
    nums = df[[f"n{i}" for i in range(1, 7)]].to_numpy()
    ind = indicator(nums).astype(np.int8)
    ext = np.vstack([ind, np.zeros((1, 42), dtype=np.int8)])  # placeholder row N (never used as input)
    N = len(ind)
    tuned = summary.get("tuned_params", {})
    out: dict[str, np.ndarray] = {}
    X = None
    want = set(models)
    if "ensemble_top3" in want:
        want |= set(summary.get("ensemble_members", []))
    for m in want:
        if m in STAT_MODELS:
            fn = STAT_MODELS[m][0]
            out[m] = fn(ext, **(tuned.get(m) or {}))[N]
        elif m in ML_MODELS:
            if X is None:
                X, _ = build_features(ext)
            params = tuned.get(m)
            params = params if isinstance(params, dict) else None
            P, _, _ = walk_forward_ml(m, ext, X, N, N + 1, 1, params)
            out[m] = P[N]
        elif m == "gru_sequence":
            P, _, _ = walk_forward_gru(ext, N, N + 1, refit_every=1)
            out[m] = P[N]
        elif m == "kmeans_regime":
            P, _, _ = walk_forward_regime(ext, N, N + 1, refit_every=1)
            out[m] = P[N]
    if "ensemble_top3" in models:
        out["ensemble_top3"] = np.mean([out[m] for m in summary["ensemble_members"]], axis=0)
    return out


def step_predict(n: int | None = None, filter_mode: str | None = None, seed: int | None = None) -> dict:
    from .predict import explanation_markdown, generate_candidates, next_draw_info
    t0 = time.time()
    cfg = load_config()["prediction"]
    df = _load()
    A, R, PR = p("artifacts_dir"), p("reports_dir"), p("predictions_dir")
    summary = json.loads((A / "backtest_summary.json").read_text())
    comp = pd.read_csv(R / "model_comparison.csv")
    models = list(dict.fromkeys([summary["final_model"], "ensemble_top3", summary["selection_best_model"]]))
    probs = next_draw_probabilities(df, summary, models)
    pop = pd.read_csv(R / "popularity_analysis.csv") if (R / "popularity_analysis.csv").exists() else None
    target = next_draw_info(df)
    cand, info = generate_candidates(df, summary, comp, probs, n or cfg["n_candidates"], filter_mode or cfg["default_filter_mode"], seed, pop)
    cand.insert(1, "target_draw", target["draw"])
    cand.insert(2, "target_date", target["date"])
    # optional bonus-number experiment: the bonus is uniform over the 36 numbers not drawn
    cand["bonus_forecast_note"] = "bonus: no evidence of predictability; any of the 36 remaining numbers is equally likely"
    cand.to_csv(PR / "next_draw_predictions.csv", index=False)
    md = explanation_markdown(target, cand, info, summary, comp)
    (PR / "prediction_explanation.md").write_text(md, encoding="utf-8")
    np.save(A / "next_draw_probs.npy", np.array([probs[m] for m in models]))
    track("predict", t0, target=target, no_signal=info["no_signal"])
    return {"target": target, "info": info, "candidates": cand}
