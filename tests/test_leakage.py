"""No forecast may depend on the draw it predicts or on any later draw."""
import numpy as np
import pytest

from loto.models.ml import build_features, walk_forward_ml
from loto.models.statistical import STAT_MODELS
from loto.simulation import fair_draws, indicator


@pytest.fixture(scope="module")
def two_histories():
    rng = np.random.default_rng(0)
    a = indicator(fair_draws(rng, 400)).astype(np.int8)
    b = a.copy()
    cut = 300
    b[cut:] = indicator(fair_draws(rng, 100)).astype(np.int8)   # different future from draw 300 on
    return a, b, cut


@pytest.mark.parametrize("name", list(STAT_MODELS))
def test_statistical_models_no_leakage(two_histories, name):
    a, b, cut = two_histories
    fn, params = STAT_MODELS[name]
    Pa, Pb = fn(a, **params), fn(b, **params)
    # forecasts for draws 0..cut (inclusive) only use draws < cut, so must be identical
    assert np.allclose(Pa[: cut + 1], Pb[: cut + 1])
    # after warm-up (never scored), forecasts are proper: about six expected numbers per draw
    assert np.allclose(Pa[200:].sum(1), 6, atol=0.5)


def test_ml_features_no_leakage(two_histories):
    a, b, cut = two_histories
    Xa, _ = build_features(a)
    Xb, _ = build_features(b)
    assert np.allclose(Xa[: cut + 1], Xb[: cut + 1])
    assert not np.allclose(Xa[cut + 1:], Xb[cut + 1:])


def test_ml_walk_forward_no_leakage(two_histories):
    a, b, cut = two_histories
    Xa, _ = build_features(a)
    Xb, _ = build_features(b)
    Pa, _, _ = walk_forward_ml("logistic_l2", a, Xa, 250, 400, 50)
    Pb, _, _ = walk_forward_ml("logistic_l2", b, Xb, 250, 400, 50)
    assert np.allclose(Pa[250:cut + 1], Pb[250:cut + 1])
