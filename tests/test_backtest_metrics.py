import numpy as np

from loto.backtest import HIT_MEAN, UNIFORM_LOGSCORE, evaluate, hits_of, tickets_from_probs, total_hits_null_pmf
from loto.simulation import fair_draws, indicator


def test_hit_null():
    assert abs(HIT_MEAN - 36 / 42) < 1e-12
    pmf = total_hits_null_pmf(300)
    assert abs(pmf.sum() - 1) < 1e-9
    assert abs((np.arange(len(pmf)) * pmf).sum() - 300 * HIT_MEAN) < 1e-6


def test_tickets_are_six_distinct_numbers():
    P = np.random.default_rng(1).random((50, 42))
    t = tickets_from_probs(P)
    assert t.shape == (50, 6)
    assert all(len(set(r)) == 6 and r.min() >= 1 and r.max() <= 42 for r in t)


def test_uniform_model_scores_as_random():
    rng = np.random.default_rng(2)
    ind = indicator(fair_draws(rng, 2000))
    P = np.full(ind.shape, 6 / 42)
    ev = evaluate(P, ind, 0, 2000)
    assert abs(ev["mean_log_score"] - UNIFORM_LOGSCORE) < 1e-9
    assert abs(ev["mean_hits"] - HIT_MEAN) < 0.06       # random tickets average 0.857 hits


def test_perfect_foresight_is_detected():
    rng = np.random.default_rng(3)
    ind = indicator(fair_draws(rng, 200))
    P = np.where(ind == 1, 0.9, 0.02)
    ev = evaluate(P, ind, 0, 200)
    assert ev["mean_hits"] == 6
    assert ev["p_hits_better_than_random"] < 1e-12
    assert ev["p_log_score_better_than_uniform"] < 1e-6
