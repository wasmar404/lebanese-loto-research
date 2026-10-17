"""Build the clean dataset (SQLite + CSV + Parquet) from the raw page cache."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import pandas as pd

from .config import ROOT, load_config, p, setup_logging
from .scraper import load_cached_records

log = setup_logging("loto.storage")

MAIN = [f"n{i}" for i in range(1, 7)]
DISPLAYED = [f"d{i}" for i in range(1, 7)]


def build_dataset() -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    records, errors = load_cached_records()
    rows, prize_rows = [], []
    for rec, sha in records:
        row = rec.to_row()
        row["weekday"] = rec.draw_date.strftime("%A")
        row["source_url"] = load_config()["scraper"]["draw_url"].format(draw=rec.draw)
        row["html_sha256"] = sha
        rows.append(row)
        for pr in rec.prizes:
            prize_rows.append({"draw": rec.draw, **pr.__dict__})
    draws = pd.DataFrame(rows).sort_values("draw").reset_index(drop=True) if rows else pd.DataFrame()
    prizes = pd.DataFrame(prize_rows)
    return draws, prizes, errors


def save_dataset(draws: pd.DataFrame, prizes: pd.DataFrame) -> None:
    draws.to_csv(p("csv"), index=False)
    draws.to_parquet(p("parquet"), index=False)
    db = p("sqlite")
    if db.exists():
        db.unlink()
    con = sqlite3.connect(db)
    try:
        draws.to_sql("draws", con, index=False)
        prizes.to_sql("prizes", con, index=False)
        idx_path = ROOT / load_config()["paths"]["data_dir"] / "site_draw_index.json"
        if idx_path.exists():
            idx = json.loads(idx_path.read_text())
            pd.DataFrame({"draw": [int(k) for k in idx], "index_date": list(idx.values())}).to_sql("site_index", con, index=False)
        con.execute("CREATE UNIQUE INDEX ix_draws_draw ON draws(draw)")
        con.execute("CREATE INDEX ix_prizes_draw ON prizes(draw)")
        pd.DataFrame([{"built_at": datetime.now(timezone.utc).isoformat(), "n_draws": len(draws)}]).to_sql("meta", con, index=False)
        con.commit()
    finally:
        con.close()
    log.info("saved %d draws to CSV, Parquet and SQLite", len(draws))


def load_draws(validated_only: bool = True, include_excluded: bool = False) -> pd.DataFrame:
    """Load the clean dataset (genuine draws only unless include_excluded=True).
    Refuses unvalidated data unless validated_only=False."""
    if validated_only:
        qr = p("quality_report")
        if not qr.exists() or not json.loads(qr.read_text()).get("passed"):
            raise SystemExit("Dataset has not passed validation. Run `python -m loto validate` first.")
    df = pd.read_parquet(p("parquet"))
    df["draw_date"] = pd.to_datetime(df["draw_date"])
    if include_excluded is False and "analysis_include" in df:
        df = df[df["analysis_include"].astype(bool)]
    return df.sort_values("draw").reset_index(drop=True)


def draws_matrix(df: pd.DataFrame):
    """(n_draws, 6) int array of sorted main numbers and (n_draws, 43) 0/1 indicator matrix."""
    import numpy as np

    nums = df[MAIN].to_numpy(dtype=np.int16)
    ind = np.zeros((len(df), 43), dtype=np.int8)
    rows = np.repeat(np.arange(len(df)), 6)
    ind[rows, nums.ravel()] = 1
    return nums, ind
