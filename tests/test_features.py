from math import comb

import numpy as np
from scipy import stats

from loto.features import all_combinations, exact_distribution, structural_features


def test_enumeration_is_complete():
    c = all_combinations()
    assert len(c) == comb(42, 6) == 5_245_786
    assert c.min() == 1 and c.max() == 42
    assert (np.diff(c.astype(int), axis=1) > 0).all()


def test_exact_odd_and_low_are_hypergeometric():
    h = stats.hypergeom(42, 21, 6).pmf(np.arange(7))
    assert np.allclose(exact_distribution("odd").values, h)
    assert np.allclose(exact_distribution("low").values, h)


def test_exact_sum_mean():
    pmf = exact_distribution("sum")
    assert abs((np.asarray(pmf.index) * pmf.to_numpy()).sum() - 6 * 43 / 2) < 1e-9
    assert abs(pmf.sum() - 1) < 1e-12


def test_structural_features_examples():
    f = structural_features(np.array([[1, 2, 3, 4, 5, 6], [5, 12, 13, 29, 34, 39]]))
    assert f.loc[0, "longest_run"] == 6 and f.loc[0, "consecutive_pairs"] == 5 and f.loc[0, "decades_used"] == 1
    assert f.loc[1, "sum"] == 132 and f.loc[1, "odd"] == 4 and f.loc[1, "longest_run"] == 2
