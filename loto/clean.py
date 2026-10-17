"""Documented cleaning of known defects in the official LLDJ history.

Every rule is conservative and every change is recorded in its own column; the raw values
are kept.  Nothing is invented: a draw whose genuine result is not available is excluded,
never filled in.

1. Placeholder duplicates.  Runs of consecutive draws with an *identical* six numbers and
   bonus are record errors, not outcomes: a fair draw repeats the previous combination with
   probability 1 / 5,245,786.  In each run we keep the **last** record, whose date fits its
   slot in the schedule (e.g. draws 379-387 occupy the nine twice-weekly slots of the July-
   August 2006 war and carry the 17 August result that belongs to draw 388), and mark the
   others ``duplicate_of``.  They are excluded from analysis.
2. Day/month swaps.  A date that breaks chronology is corrected by swapping day and month
   **only if** the swapped date lies strictly between its neighbours *and* falls on a regular
   draw weekday.  Otherwise the date is flagged ``date_uncertain`` (numbers are kept; the draw
   is excluded only from calendar-based tests).
"""
from __future__ import annotations

from datetime import date

import pandas as pd

MAIN = [f"n{i}" for i in range(1, 7)]


def _swap(d: pd.Timestamp):
    try:
        return pd.Timestamp(date(d.year, d.day, d.month))
    except ValueError:
        return None


def _lis(seq) -> list[int]:
    """Indices of one longest strictly increasing subsequence (O(n log n))."""
    import bisect
    tails, tails_idx, prev = [], [], [-1] * len(seq)
    for i, x in enumerate(seq):
        j = bisect.bisect_left(tails, x)
        if j == len(tails):
            tails.append(x); tails_idx.append(i)
        else:
            tails[j] = x; tails_idx[j] = i
        prev[i] = tails_idx[j - 1] if j > 0 else -1
    out, k = [], tails_idx[-1] if tails_idx else -1
    while k >= 0:
        out.append(k); k = prev[k]
    return out[::-1]


def clean_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.sort_values("draw").reset_index(drop=True).copy()
    df["draw_date_raw"] = df["draw_date"].astype(str).str[:10]
    dates = pd.to_datetime(df["draw_date_raw"])
    sched = set(dates.dt.dayofweek.value_counts().index[:2])   # the two regular draw weekdays

    # 1. placeholder duplicates -------------------------------------------------------
    key = df[MAIN + ["bonus"]].astype(int).astype(str).agg("-".join, axis=1)
    df["duplicate_of"] = pd.array([pd.NA] * len(df), dtype="Int64")
    runs = []
    i = 0
    while i < len(df):
        j = i
        while j + 1 < len(df) and key[j + 1] == key[i]:
            j += 1
        if j > i:
            keep = int(df.loc[j, "draw"])
            for k in range(i, j):
                df.loc[k, "duplicate_of"] = keep
            runs.append({"draws": [int(x) for x in df.loc[i:j, "draw"]], "kept": keep,
                         "combination": key[i], "date": df.loc[j, "draw_date_raw"]})
        i = j + 1
    dup = df["duplicate_of"].notna().to_numpy()

    # 2. dates --------------------------------------------------------------------------
    # Anchor = longest strictly increasing subsequence of dates (in draw order). Draws outside
    # it are the outliers; each is corrected by a day/month swap only if the swapped date lies
    # strictly between its nearest anchors (or already-corrected neighbours) and falls on a
    # regular draw weekday.
    df["date_corrected"] = False
    df["date_uncertain"] = False
    d = dates.copy()
    corrections = []
    valid = [k for k in range(len(df)) if not dup[k]]
    anchors = set(_lis([d[k] for k in valid]))
    anchor_idx = {valid[i] for i in anchors}
    ok = {k: (k in anchor_idx) for k in valid}
    # resolve outliers left to right so consecutive outliers can use corrected neighbours
    for pos, k in enumerate(valid):
        if ok[k]:
            continue
        prev_d = next((d[valid[q]] for q in range(pos - 1, -1, -1) if ok[valid[q]]), pd.Timestamp.min)
        next_d = next((d[valid[q]] for q in range(pos + 1, len(valid)) if valid[q] in anchor_idx), pd.Timestamp.max)
        s = _swap(d[k])
        if s is not None and prev_d < s < next_d and s.dayofweek in sched:
            corrections.append({"draw": int(df.loc[k, "draw"]), "raw": d[k].strftime("%Y-%m-%d"), "corrected": s.strftime("%Y-%m-%d"),
                                "weekday": s.day_name()})
            d[k] = s
            df.loc[k, "date_corrected"] = True
            ok[k] = True
        else:
            df.loc[k, "date_uncertain"] = True
    # consecutive genuine draws sharing one date (different numbers): we cannot tell which
    # date is wrong, so both are uncertain. (Non-adjacent ties are outliers handled above.)
    for a, b in zip(valid, valid[1:]):
        if d[a] == d[b]:
            df.loc[[a, b], "date_uncertain"] = True
    df["draw_date"] = d.dt.strftime("%Y-%m-%d")
    df["weekday"] = d.dt.day_name()
    df.loc[dup, "date_uncertain"] = False
    df["off_schedule_weekday"] = ~d.dt.dayofweek.isin(sched) & ~dup
    df["analysis_include"] = ~dup

    log = {
        "regular_draw_weekdays": sorted(pd.Timestamp(2024, 1, 1 + w).day_name() for w in sched),
        "duplicate_runs": runs,
        "duplicate_records_excluded": int(dup.sum()),
        "date_corrections": corrections,
        "date_uncertain_draws": [int(x) for x in df.loc[df["date_uncertain"], "draw"]],
        "off_schedule_weekday_draws": [{"draw": int(r.draw), "date": r.draw_date, "weekday": r.weekday}
                                       for r in df[df["off_schedule_weekday"]].itertuples()],
        "analysis_draws": int((~dup).sum()),
    }
    return df, log
