"""Dataset validation and completeness report.

Modelling commands refuse to run unless ``data_quality_report.json`` says ``passed: true``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import ROOT, load_config, p, setup_logging
from .storage import DISPLAYED, MAIN, build_dataset, save_dataset

log = setup_logging("loto.validate")


def validate_frame(df: pd.DataFrame, index: dict[int, str] | None = None, expected_latest: int | None = None,
                   date_rows: pd.Series | None = None) -> dict:
    """date_rows: boolean mask of rows whose dates enter the chronology checks (default all)."""
    cfg = load_config()["game"]
    pool, pick = cfg["pool_size"], cfg["pick"]
    issues: dict[str, list] = {k: [] for k in (
        "out_of_range", "duplicate_numbers_within_draw", "bonus_out_of_range", "bonus_in_main",
        "duplicate_draw_ids", "missing_draw_ids", "non_increasing_dates", "duplicate_dates",
        "dropdown_date_mismatch", "index_date_mismatch", "index_missing", "displayed_vs_sorted_mismatch")}

    nums = df[MAIN].to_numpy()
    disp = df[DISPLAYED].to_numpy()
    for i, row in df.iterrows():
        ns = nums[i]
        if ((ns < 1) | (ns > pool)).any():
            issues["out_of_range"].append(int(row.draw))
        if len(set(ns.tolist())) != pick:
            issues["duplicate_numbers_within_draw"].append(int(row.draw))
        if not 1 <= row.bonus <= pool:
            issues["bonus_out_of_range"].append(int(row.draw))
        if row.bonus in set(ns.tolist()):
            issues["bonus_in_main"].append(int(row.draw))
        if sorted(disp[i].tolist()) != ns.tolist():
            issues["displayed_vs_sorted_mismatch"].append(int(row.draw))
        page_date = row["draw_date_raw"] if "draw_date_raw" in row and isinstance(row.get("draw_date_raw"), str) else str(row["draw_date"])[:10]
        if row.get("dropdown_date") and row["dropdown_date"] != page_date:
            issues["dropdown_date_mismatch"].append(int(row.draw))

    ids = df["draw"].astype(int)
    issues["duplicate_draw_ids"] = ids[ids.duplicated()].tolist()
    latest = int(expected_latest or ids.max())
    issues["missing_draw_ids"] = sorted(set(range(1, latest + 1)) - set(ids.tolist()))
    dates = pd.to_datetime(df["draw_date"])
    order = df.assign(_d=dates)
    if date_rows is not None:
        order = order[date_rows.to_numpy()]
    order = order.sort_values("draw")
    nd = order["_d"].diff().dt.days
    issues["non_increasing_dates"] = order.loc[nd <= 0, "draw"].astype(int).tolist()
    issues["duplicate_dates"] = order.loc[order["_d"].duplicated(), "draw"].astype(int).tolist()

    if index:
        src = df.sort_values("draw")
        page_dates = src["draw_date_raw"] if "draw_date_raw" in src else pd.to_datetime(src["draw_date"]).dt.strftime("%Y-%m-%d")
        for d, dstr in zip(src["draw"].astype(int), page_dates):
            if d not in index:
                issues["index_missing"].append(d)
            elif index[d] != dstr:
                issues["index_date_mismatch"].append({"draw": d, "page": dstr, "index": index[d]})

    gaps = nd.dropna()
    weekday_counts = order["_d"].dt.day_name().value_counts().to_dict()
    by_year = order.groupby(order["_d"].dt.year)["draw"].count().to_dict()
    gap_hist = gaps.value_counts().sort_index().to_dict()
    long_gaps = order.loc[nd > 14, ["draw", "_d"]].assign(gap_days=nd[nd > 14])
    long_gap_list = [{"draw": int(d), "date": dt.strftime("%Y-%m-%d"), "gap_days": int(g)}
                     for d, dt, g in zip(long_gaps["draw"], long_gaps["_d"], long_gaps["gap_days"])]

    critical = ["out_of_range", "duplicate_numbers_within_draw", "bonus_out_of_range", "bonus_in_main",
                "duplicate_draw_ids", "missing_draw_ids", "non_increasing_dates", "duplicate_dates",
                "displayed_vs_sorted_mismatch"]
    passed = all(len(issues[k]) == 0 for k in critical) and len(df) > 0
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "n_draws": int(len(df)),
        "first_draw": {"draw": int(order["draw"].iloc[0]), "date": order["_d"].iloc[0].strftime("%Y-%m-%d")} if len(df) else None,
        "last_draw": {"draw": int(order["draw"].iloc[-1]), "date": order["_d"].iloc[-1].strftime("%Y-%m-%d")} if len(df) else None,
        "expected_draw_ids": latest,
        "completeness_pct": round(100 * len(set(ids)) / latest, 4) if latest else 0,
        "issue_counts": {k: len(v) for k, v in issues.items()},
        "issues": issues,
        "critical_checks": critical,
        "displayed_order": {
            "draws_displayed_unsorted": int((~df["displayed_sorted"].astype(bool)).sum()),
            "draws_displayed_sorted": int(df["displayed_sorted"].astype(bool).sum()),
            "last_unsorted_draw": int(df.loc[~df["displayed_sorted"].astype(bool), "draw"].max()) if (~df["displayed_sorted"].astype(bool)).any() else None,
            "note": "Where the page lists balls unsorted we assume this is the order drawn; sorted pages give no draw-order information.",
        },
        "schedule": {
            "weekday_counts": weekday_counts,
            "draws_per_year": {int(k): int(v) for k, v in by_year.items()},
            "gap_days_histogram": {int(k): int(v) for k, v in gap_hist.items()},
            "gaps_over_14_days": long_gap_list,
        },
        "rules_checked": {
            "pool": pool, "pick": pick,
            "main_numbers_distinct_in_range": True,
            "bonus_in_range_and_not_in_main": True,
        },
    }


def run_validation() -> dict:
    from .clean import clean_frame
    draws, prizes, parse_errors = build_dataset()
    if draws.empty:
        raise SystemExit("No cached draws found. Run `python -m loto scrape` first.")
    idx_path = ROOT / load_config()["paths"]["data_dir"] / "site_draw_index.json"
    index = {int(k): v for k, v in json.loads(idx_path.read_text()).items()} if idx_path.exists() else None
    summary_path = ROOT / load_config()["paths"]["data_dir"] / "scrape_summary.json"
    expected_latest = json.loads(summary_path.read_text()).get("latest_draw") if summary_path.exists() else None

    # 1. raw data exactly as published
    raw = validate_frame(draws, index=index, expected_latest=expected_latest)
    # 2. documented cleaning (see loto/clean.py), then validation of what the analysis uses
    clean, clog = clean_frame(draws)
    save_dataset(clean, prizes)
    dated = clean["analysis_include"] & ~clean["date_uncertain"]
    report = validate_frame(clean, index=index, expected_latest=expected_latest, date_rows=dated)
    ana = clean[clean["analysis_include"]]
    key = ana[[f"n{i}" for i in range(1, 7)] + ["bonus"]].astype(str).agg("-".join, axis=1)
    adj_dups = int((key.to_numpy()[1:] == key.to_numpy()[:-1]).sum())
    report["issues"]["adjacent_identical_draws_in_analysis_set"] = [adj_dups] if adj_dups else []
    report["issue_counts"]["adjacent_identical_draws_in_analysis_set"] = adj_dups
    # any repeated full combination anywhere in the analysis set (expected ~0.6 by chance)
    vc = key.value_counts()
    report["repeated_combinations_in_analysis_set"] = {k: int(v) for k, v in vc[vc > 1].items()}
    report["passed"] = report["passed"] and adj_dups == 0
    report["raw_published_data"] = {"issue_counts": raw["issue_counts"], "issues": raw["issues"]}
    report["cleaning"] = clog
    report["n_draw_ids"] = int(len(clean))
    report["n_analysis_draws"] = int(clean["analysis_include"].sum())
    report["note"] = ("Draw IDs 1..latest are all present. Placeholder duplicate records in the official history are "
                      "excluded from analysis and date day/month swaps corrected, as documented in 'cleaning'.")
    report["parse_errors"] = parse_errors
    if parse_errors:
        report["passed"] = False
    report["site_index_entries"] = len(index) if index else 0
    if summary_path.exists():
        report["scrape_summary"] = json.loads(summary_path.read_text())
    p("quality_report").write_text(json.dumps(report, indent=2, default=str))
    log.info("validation %s: %d draws, issues=%s", "PASSED" if report["passed"] else "FAILED",
             report["n_draws"], {k: v for k, v in report["issue_counts"].items() if v})
    return report
