"""Command-line interface.

    python -m loto scrape        # download / resume the full history (polite, cached)
    python -m loto update        # fetch new draws, rebuild and re-validate the dataset
    python -m loto validate      # build SQLite/CSV/Parquet and the data-quality report
    python -m loto simulate      # Monte Carlo fair-lottery simulations
    python -m loto analyze       # statistics, randomness tests, hypotheses, filters, HTML reports
    python -m loto backtest      # tune, walk-forward backtest, negative controls
    python -m loto predict       # candidate tickets for the next draw
    python -m loto all           # everything above, in order
    python -m loto dashboard     # launch the Streamlit dashboard
    python -m loto hardware      # show detected CPU / RAM / GPU
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

from .config import ROOT


def main(argv=None):
    ap = argparse.ArgumentParser(prog="loto", description="Lebanese Loto research system")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scrape")
    sub.add_parser("update")
    sub.add_parser("validate")
    sub.add_parser("simulate")
    sub.add_parser("analyze")
    b = sub.add_parser("backtest")
    b.add_argument("--quick", action="store_true", help="fewer models / tuning trials (smoke test)")
    b.add_argument("--controls", type=int, default=20, help="negative-control repetitions (0 to skip)")
    pr = sub.add_parser("predict")
    pr.add_argument("-n", type=int, default=None)
    pr.add_argument("--filter", choices=["none", "conservative", "moderate", "aggressive"], default=None)
    pr.add_argument("--seed", type=int, default=None)
    a = sub.add_parser("all")
    a.add_argument("--quick", action="store_true")
    sub.add_parser("dashboard")
    sub.add_parser("hardware")
    args = ap.parse_args(argv)

    from . import pipeline as pl
    if args.cmd == "scrape":
        print(json.dumps({k: v for k, v in pl.step_scrape(refresh_latest=0).items() if k != "failed"}, indent=2, default=str))
    elif args.cmd == "update":
        pl.step_scrape(refresh_latest=3)
        r = pl.step_validate()
        print(f"validation {'PASSED' if r['passed'] else 'FAILED'}: {r['n_draws']} draws, last {r['last_draw']}")
    elif args.cmd == "validate":
        r = pl.step_validate()
        print(f"validation {'PASSED' if r['passed'] else 'FAILED'}: {r['n_draws']} draws, last {r['last_draw']}")
        print(json.dumps({k: v for k, v in r["issue_counts"].items() if v}, indent=2))
        sys.exit(0 if r["passed"] else 1)
    elif args.cmd == "simulate":
        print(pl.step_simulate())
    elif args.cmd == "analyze":
        print(pl.step_analyze())
    elif args.cmd == "backtest":
        s = pl.step_backtest(quick=args.quick, controls=args.controls)
        print(json.dumps({k: s[k] for k in ("final_model", "selection_best_model", "verdicts")}, indent=2))
    elif args.cmd == "predict":
        r = pl.step_predict(n=args.n, filter_mode=args.filter, seed=args.seed)
        print(f"target draw #{r['target']['draw']} on {r['target']['date']}")
        print(r["candidates"][["rank", "numbers", "method"]].to_string(index=False))
    elif args.cmd == "all":
        pl.step_scrape(refresh_latest=3)
        r = pl.step_validate()
        if not r["passed"]:
            sys.exit("dataset failed validation - stopping before modelling")
        pl.step_simulate()
        pl.step_analyze()
        pl.step_backtest(quick=args.quick)
        r = pl.step_predict()
        print(r["candidates"][["rank", "numbers", "method"]].to_string(index=False))
    elif args.cmd == "dashboard":
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard" / "app.py")], check=False)
    elif args.cmd == "hardware":
        print(json.dumps(pl.hardware(), indent=2))


if __name__ == "__main__":
    main()
