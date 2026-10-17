"""Checks on the real scraped dataset (skipped until it exists)."""
import json

import numpy as np
import pandas as pd
import pytest

from loto.config import p

pytestmark = pytest.mark.skipif(not p("parquet").exists(), reason="dataset not built yet")


@pytest.fixture(scope="module")
def df():
    return pd.read_parquet(p("parquet"))


def test_quality_report_passed():
    r = json.loads(p("quality_report").read_text())
    assert r["passed"], r["issue_counts"]


def test_rules(df):
    nums = df[[f"n{i}" for i in range(1, 7)]].to_numpy()
    assert ((nums >= 1) & (nums <= 42)).all()
    assert (np.diff(nums, axis=1) > 0).all()                # six distinct, sorted
    assert all(b not in set(r) for b, r in zip(df.bonus, nums.tolist()))
    assert df.bonus.between(1, 42).all()


def test_contiguous_and_chronological(df):
    assert df.draw.tolist() == list(range(1, len(df) + 1))
    assert pd.to_datetime(df.draw_date).is_monotonic_increasing


def test_known_draws(df):
    d = df.set_index("draw")
    assert d.loc[1, ["n1", "n2", "n3", "n4", "n5", "n6", "bonus"]].tolist() == [7, 8, 11, 20, 25, 32, 30]
    assert d.loc[1000, ["n1", "n2", "n3", "n4", "n5", "n6", "bonus"]].tolist() == [5, 12, 13, 29, 34, 39, 22]
