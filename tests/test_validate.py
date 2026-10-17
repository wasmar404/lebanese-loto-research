import numpy as np
import pandas as pd

from loto.validate import validate_frame


def frame(rows):
    out = []
    for i, (d, dt, nums, bonus) in enumerate(rows):
        s = sorted(nums)
        out.append({"draw": d, "draw_date": dt, **{f"n{k + 1}": s[k] for k in range(6)}, "bonus": bonus,
                    **{f"d{k + 1}": nums[k] for k in range(6)}, "displayed_sorted": nums == s, "dropdown_date": dt})
    return pd.DataFrame(out)


GOOD = [(1, "2002-09-12", [7, 25, 11, 8, 20, 32], 30), (2, "2002-09-16", [1, 2, 3, 4, 5, 6], 7), (3, "2002-09-19", [10, 20, 30, 40, 41, 42], 1)]


def test_clean_frame_passes():
    r = validate_frame(frame(GOOD), expected_latest=3)
    assert r["passed"], r["issue_counts"]
    assert r["completeness_pct"] == 100


def test_detects_defects():
    bad = GOOD + [(3, "2002-09-19", [10, 20, 30, 40, 41, 42], 1),          # duplicate id + date
                  (5, "2002-09-26", [1, 2, 3, 4, 5, 43], 9),               # out of range, missing 4
                  (6, "2002-09-25", [1, 2, 3, 4, 5, 6], 6)]                # bonus in main, date goes back
    r = validate_frame(frame(bad), expected_latest=6)
    c = r["issue_counts"]
    assert not r["passed"]
    assert c["duplicate_draw_ids"] >= 1 and c["missing_draw_ids"] == 1
    assert c["out_of_range"] == 1 and c["bonus_in_main"] == 1 and c["non_increasing_dates"] >= 1


def test_long_gaps_reported():
    rows = GOOD + [(4, "2002-10-30", [3, 9, 15, 22, 31, 40], 2)]
    r = validate_frame(frame(rows), expected_latest=4)
    assert r["schedule"]["gaps_over_14_days"] == [{"draw": 4, "date": "2002-10-30", "gap_days": 41}]


def test_index_cross_check():
    r = validate_frame(frame(GOOD), index={1: "2002-09-12", 2: "2002-09-17"}, expected_latest=3)
    assert r["issue_counts"]["index_date_mismatch"] == 1
    assert r["issue_counts"]["index_missing"] == 1
