"""Phase 3D / 4 randomness test battery.

Every test is recorded as one row with its family, subject, statistic, p-value and the
method used to get the p-value (exact, asymptotic, Monte Carlo or permutation).  At the end,
p-values are corrected with Benjamini-Hochberg (FDR) across *all* tests and within each
family, plus Bonferroni, so isolated small p-values are not mistaken for discoveries.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests

from .config import load_config, setup_logging
from .features import exact_distribution, overlap_with_previous, structural_features
from .simulation import fair_draws, indicator

log = setup_logging("loto.randomness")
POOL, PICK = 42, 6
P = PICK / POOL


class TestLog:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, family, test, subject, statistic, p_value, method, n=None, df=None, note=""):
        self.rows.append(dict(family=family, test=test, subject=str(subject), statistic=float(statistic),
                              df=df, p_value=float(np.clip(p_value, 0, 1)), method=method, n=n, note=note))

    def frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.rows)
        ok = df["p_value"].notna()
        df.loc[ok, "p_fdr_global"] = multipletests(df.loc[ok, "p_value"], method="fdr_bh")[1]
        df.loc[ok, "p_bonferroni"] = np.minimum(1, df.loc[ok, "p_value"] * ok.sum())
        df["p_fdr_family"] = np.nan
        for fam, g in df[ok].groupby("family"):
            df.loc[g.index, "p_fdr_family"] = multipletests(g["p_value"], method="fdr_bh")[1]
        alpha = load_config()["analysis"]["alpha"]
        df["significant_raw"] = df["p_value"] < alpha
        df["significant_fdr"] = df["p_fdr_global"] < alpha
        return df


def _merge_bins(obs: np.ndarray, exp: np.ndarray, min_exp: float = 5.0):
    """Merge adjacent bins until every expected count >= min_exp."""
    o, e = list(obs.astype(float)), list(exp.astype(float))
    i = 0
    while i < len(e):
        if e[i] < min_exp and len(e) > 1:
            j = i + 1 if i + 1 < len(e) else i - 1
            e[j] += e[i]; o[j] += o[i]
            del e[i]; del o[i]
            i = max(0, min(i, j) - 0)
        else:
            i += 1
    return np.array(o), np.array(e)


def gof_vs_exact(values: pd.Series, pmf: pd.Series):
    support = pmf.index.to_numpy()
    obs = values.value_counts().reindex(support, fill_value=0).to_numpy()
    extra = int(len(values) - obs.sum())
    exp = pmf.to_numpy() * len(values)
    o, e = _merge_bins(obs, exp)
    stat = ((o - e) ** 2 / e).sum()
    dof = len(o) - 1
    return stat, dof, stats.chi2.sf(stat, dof), extra


def sim_number_counts(N: int, n_hist: int, seed: int) -> np.ndarray:
    """Number-frequency vectors of n_hist fair histories (n_hist, 42). Used to calibrate any
    statistic that depends only on the 42 number counts (handles within-draw dependence)."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_hist, POOL), dtype=np.int32)
    for h in range(n_hist):
        out[h] = indicator(fair_draws(rng, N)).sum(0)
    return out


def cusum_test(x: np.ndarray, p: float):
    """Brownian-bridge CUSUM for a change in Bernoulli rate. Returns (stat, asymptotic p)."""
    n = len(x)
    s = np.cumsum(x - x.mean())
    stat = np.abs(s).max() / np.sqrt(n * p * (1 - p))
    return stat, stats.kstwobign.sf(stat)


def _fisher_g(x: np.ndarray):
    """Fisher's exact g test for a hidden periodic component (Fisher 1929)."""
    from math import comb, floor, log, exp
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    I = np.abs(np.fft.rfft(x)[1:(n - 1) // 2 + 1]) ** 2
    m = len(I)
    g = I.max() / I.sum()
    # The exact alternating series is numerically unstable for m ~ 1000; its first term
    # m(1-g)^(m-1) is an upper bound that is essentially exact whenever p is small.
    pv = m * exp((m - 1) * log(max(1e-300, 1 - g)))
    period = n / (np.argmax(I) + 1)
    return g, float(np.clip(pv, 0, 1)), period


def permutation_contingency(labels: np.ndarray, ind: np.ndarray, n_perm: int, rng) -> tuple[float, float]:
    """Number x label contingency chi-square, p-value by permuting labels across draws."""
    cats = np.unique(labels)
    L = (labels[:, None] == cats[None, :]).astype(np.int32)  # (N, C)

    def stat_of(Lm):
        tab = Lm.T @ ind  # (C, 42)
        exp = Lm.sum(0)[:, None] * P
        return float(((tab - exp) ** 2 / exp).sum())

    obs = stat_of(L)
    ge = 1
    for _ in range(n_perm):
        ge += stat_of(L[rng.permutation(len(L))]) >= obs
    return obs, ge / (n_perm + 1)


def run_randomness_tests(df: pd.DataFrame, sim_counts: np.ndarray | None = None) -> pd.DataFrame:
    cfg = load_config()
    rng = np.random.default_rng(cfg["simulation"]["seed"] + 99)
    n_perm = cfg["backtest"]["n_permutations"]
    T = TestLog()
    nums = df[[f"n{i}" for i in range(1, 7)]].to_numpy()
    N = len(nums)
    ind = indicator(nums).astype(np.int32)
    counts = ind.sum(0)
    exp = N * P
    if sim_counts is None:
        sim_counts = sim_number_counts(N, 5000, cfg["simulation"]["seed"] + 5)

    # ---------- A. frequencies -------------------------------------------------------
    chi2 = ((counts - exp) ** 2 / exp).sum()
    sim_chi2 = ((sim_counts - exp) ** 2 / exp).sum(1)
    T.add("A_frequency", "chi-square GOF (main numbers, all draws)", "all", chi2,
          (np.sum(sim_chi2 >= chi2) + 1) / (len(sim_chi2) + 1), "monte_carlo", N, 41,
          f"asymptotic p with 41/36 dependence correction = {stats.chi2.sf(chi2 * 41 / 36, 41):.4f}")
    for k in range(POOL):
        pv = stats.binomtest(int(counts[k]), N, P).pvalue
        T.add("A_frequency_per_number", "exact binomial (two-sided)", k + 1, (counts[k] - exp) / np.sqrt(N * P * (1 - P)), pv, "exact", N)
    # partitions of the number space: decades, low/high, odd/even, primes, residues mod m
    partitions = {
        "decades": np.array([0] * 9 + [1] * 10 + [2] * 10 + [3] * 10 + [4] * 3),
        "low_high": np.array([0] * 21 + [1] * 21),
        "odd_even": np.array([i % 2 for i in range(1, 43)]),
        "prime": np.array([int(i in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)) for i in range(1, 43)]),
    }
    for m in range(3, 13):   # mod 10 = last digit (ending); mod 2 is odd_even above
        partitions[f"mod_{m}" if m != 10 else "mod_10 (ending digit)"] = np.array([i % m for i in range(1, 43)])
    for name, lab in partitions.items():
        G = (lab[:, None] == np.unique(lab)[None, :]).astype(float)
        o = counts @ G
        e = exp * G.sum(0)
        s = ((o - e) ** 2 / e).sum()
        so = sim_counts @ G
        ss = ((so - e) ** 2 / e).sum(1)
        fam = "H_modular" if name.startswith("mod_") or name == "ending_digit" else "A_frequency_partition"
        T.add(fam, "number-class frequency (chi-square, MC calibrated)", name, s,
              (np.sum(ss >= s) + 1) / (len(ss) + 1), "monte_carlo", N, len(e) - 1)

    # bonus number
    bc = df["bonus"].value_counts().reindex(range(1, 43), fill_value=0).to_numpy()
    s, pv = stats.chisquare(bc)
    T.add("A_frequency", "chi-square GOF (bonus number)", "bonus", s, pv, "asymptotic", N, 41)
    # bonus vs remaining pool: bonus should be uniform over the 36 numbers not drawn
    # (it cannot be a main number). check rank of bonus among remaining numbers
    rem_rank = np.array([sorted(set(range(1, 43)) - set(r)).index(b) for r, b in zip(nums.tolist(), df["bonus"])])
    s, pv = stats.chisquare(np.bincount(rem_rank, minlength=36))
    T.add("A_frequency", "bonus uniform over remaining 36 (rank chi-square)", "bonus", s, pv, "asymptotic", N, 35)

    # draw order (only where the page displays balls unsorted = draw order)
    uns = ~df["displayed_sorted"].astype(bool).to_numpy()
    if uns.sum() >= 200:
        disp = df.loc[uns, [f"d{i}" for i in range(1, 7)]].to_numpy()
        for j in range(6):
            s, pv = stats.chisquare(np.bincount(disp[:, j], minlength=43)[1:])
            T.add("A_draw_order", "ball-position uniformity (chi-square)", f"position {j + 1}", s, pv, "asymptotic", int(uns.sum()), 41,
                  "only draws whose page lists balls unsorted (assumed draw order)")
        # is the k-th ball ever systematically smaller? compare mean of each position to 21.5
        for j in range(6):
            t_, pv = stats.ttest_1samp(disp[:, j], 21.5)
            T.add("A_draw_order", "position mean = 21.5 (t-test)", f"position {j + 1}", t_, pv, "asymptotic", int(uns.sum()))

    # era homogeneity: quarters of the history, permutation over draw order
    era = np.minimum((np.arange(N) * 4) // N, 3)
    s, pv = permutation_contingency(era, ind, n_perm, rng)
    T.add("E_drift", "era homogeneity (4 chronological quarters, permutation)", "number x era", s, pv, "permutation", N)
    years = pd.to_datetime(df["draw_date"]).dt.year.to_numpy()
    s, pv = permutation_contingency(years, ind, n_perm, rng)
    T.add("E_drift", "year homogeneity (permutation)", "number x year", s, pv, "permutation", N)

    # ---------- B. structural vs exact distributions --------------------------------
    feats = structural_features(nums)
    for col in ["sum", "odd", "low", "consecutive_pairs", "longest_run", "primes", "max_decade_count", "decades_used",
                "max_same_ending", "distinct_endings", "mult3", "mult5", "mult7", "min_number", "max_number", "range",
                "min_gap", "max_gap", "le31", "le12"] + [f"dec_{l}" for l in ["1-9", "10-19", "20-29", "30-39", "40-42"]]:
        pmf = exact_distribution(col)
        s, dof, pv, _ = gof_vs_exact(feats[col], pmf)
        T.add("B_structure", "chi-square GOF vs exact enumeration", col, s, pv, "exact_pmf+asymptotic", N, dof)
    # sum mod m
    for m in (2, 3, 4, 5, 6, 7, 10):
        from .features import exact_feature_table
        pmf = (exact_feature_table()["sum"] % m).value_counts(normalize=True).sort_index()
        s, dof, pv, _ = gof_vs_exact(feats["sum"] % m, pmf)
        T.add("H_modular", "draw sum mod m vs exact", f"sum mod {m}", s, pv, "exact_pmf+asymptotic", N, dof)

    # ---------- C. serial dependence ---------------------------------------------
    hyp = stats.hypergeom(POOL, PICK, PICK)
    for lag in cfg["analysis"]["lags"]:
        ov = overlap_with_previous(nums, lag)
        pmf = pd.Series(hyp.pmf(np.arange(7)), index=np.arange(7))
        s, dof, pv, _ = gof_vs_exact(pd.Series(ov), pmf)
        T.add("C_serial", "overlap with draw t-lag vs hypergeometric", f"lag {lag}", s, pv, "exact_pmf+asymptotic", len(ov), dof,
              f"mean overlap {ov.mean():.4f} vs expected {36 / 42:.4f}")
    for k in range(POOL):
        lb = acorr_ljungbox(ind[:, k], lags=[20], return_df=True)
        T.add("C_serial_per_number", "Ljung-Box (20 lags)", k + 1, lb["lb_stat"].iloc[0], lb["lb_pvalue"].iloc[0], "asymptotic", N, 20)
        z, pv = runstest_1samp(ind[:, k].astype(float), cutoff=0.5, correction=False)
        T.add("C_runs_per_number", "Wald-Wolfowitz runs (appear / not appear)", k + 1, z, pv, "asymptotic", N)
    for col in ["sum", "odd", "low", "consecutive_pairs", "range"]:
        lb = acorr_ljungbox(feats[col].astype(float), lags=[20], return_df=True)
        T.add("C_serial", "Ljung-Box (20 lags)", col, lb["lb_stat"].iloc[0], lb["lb_pvalue"].iloc[0], "asymptotic", N, 20)
        z, pv = runstest_1samp(feats[col].astype(float).to_numpy(), cutoff="median", correction=False)
        T.add("C_serial", "runs test above/below median", col, z, pv, "asymptotic", N)

    # pairwise conditional dependence: number i at t  -> number j at t+1  (42 x 42 2x2 tables)
    A, B = ind[:-1], ind[1:]
    n11 = A.T @ B
    a1 = A.sum(0)[:, None]; b1 = B.sum(0)[None, :]
    n = len(A)
    e11 = a1 * b1 / n
    # chi-square (1 df) for 2x2 with margins: (n11-e11)^2 * n^3 / (a1 (n-a1) b1 (n-b1))
    chi = (n11 - e11) ** 2 * n ** 3 / (a1 * (n - a1) * b1 * (n - b1))
    pv = stats.chi2.sf(chi, 1)
    for i in range(POOL):
        for j in range(POOL):
            T.add("C_conditional_lag1", "2x2 chi-square: i at t -> j at t+1", f"{i + 1}->{j + 1}", chi[i, j], pv[i, j], "asymptotic", n, 1)

    # ---------- D. pairs ---------------------------------------------------------
    pair = ind.T @ ind
    pexp_p = PICK * (PICK - 1) / (POOL * (POOL - 1))
    iu = np.triu_indices(POOL, 1)
    for a, b in zip(*iu):
        c = int(pair[a, b])
        T.add("D_pairs", "exact binomial pair frequency", f"{a + 1}-{b + 1}", c, stats.binomtest(c, N, pexp_p).pvalue, "exact", N)

    # ---------- E. change points --------------------------------------------------
    for k in range(POOL):
        s, pv = cusum_test(ind[:, k], P)
        T.add("E_changepoint", "CUSUM (Brownian bridge) rate change", k + 1, s, pv, "asymptotic", N)
    for col in ["sum", "odd", "low"]:
        s, pv = cusum_test((feats[col] - feats[col].mean()).to_numpy() / feats[col].std() * np.sqrt(0.25) + 0.5, 0.5)
        T.add("E_changepoint", "CUSUM mean change (standardised)", col, s, pv, "asymptotic", N)
    try:
        import ruptures as rpt
        sig = feats["sum"].to_numpy(float)
        sig = (sig - sig.mean()) / sig.std()
        pen = 3 * np.log(N)
        n_cp = len(rpt.Pelt(model="l2", min_size=50).fit(sig).predict(pen=pen)) - 1
        null = []
        for _ in range(200):
            sh = rng.permutation(sig)
            null.append(len(rpt.Pelt(model="l2", min_size=50).fit(sh).predict(pen=pen)) - 1)
        null = np.array(null)
        T.add("E_changepoint", "PELT change points in draw sum (vs 200 shuffles)", "sum", n_cp,
              (np.sum(null >= n_cp) + 1) / (len(null) + 1), "permutation", N, note=f"shuffled mean {null.mean():.2f}")
    except Exception as exc:  # pragma: no cover
        log.warning("ruptures unavailable: %s", exc)

    # ---------- F. calendar (draws with uncertain dates excluded) -----------------
    cal = ~df["date_uncertain"].astype(bool).to_numpy() if "date_uncertain" in df else np.ones(N, bool)
    dates = pd.to_datetime(df["draw_date"])[cal]
    ind_c, nums_c, Nc = ind[cal], nums[cal], int(cal.sum())
    wd = dates.dt.dayofweek.to_numpy()
    wd = np.where(np.isin(wd, [0, 3]), wd, 7)   # Monday / Thursday / other (rescheduled)
    for name, lab in [("weekday (Mon/Thu/other)", wd), ("month", dates.dt.month.to_numpy()),
                      ("day_of_month_bucket", (dates.dt.day.to_numpy() - 1) // 8)]:
        if len(np.unique(lab)) > 1:
            s, pv = permutation_contingency(lab, ind_c, n_perm, rng)
            T.add("F_calendar", "number x calendar label (permutation)", name, s, pv, "permutation", Nc)
    dom = dates.dt.day.to_numpy()
    hits = sum(int(d in set(r)) for d, r in zip(dom, nums_c.tolist()))
    T.add("F_calendar", "day-of-month number drawn (exact binomial)", "day_of_month in draw", hits,
          stats.binomtest(hits, Nc, P).pvalue, "exact", Nc, note=f"{hits} vs expected {Nc * P:.1f}")
    mon = dates.dt.month.to_numpy()
    hits = sum(int(d in set(r)) for d, r in zip(mon, nums_c.tolist()))
    T.add("F_calendar", "month number drawn (exact binomial)", "month in draw", hits,
          stats.binomtest(hits, Nc, P).pvalue, "exact", Nc, note=f"{hits} vs expected {Nc * P:.1f}")

    # ---------- G. periodicity -----------------------------------------------------
    for k in range(POOL):
        g, pv, per = _fisher_g(ind[:, k])
        T.add("G_periodicity", "Fisher g (hidden periodicity)", k + 1, g, pv, "exact", N, note=f"peak period {per:.1f} draws")
    for col in ["sum", "odd", "low"]:
        g, pv, per = _fisher_g(feats[col].to_numpy(float))
        T.add("G_periodicity", "Fisher g (hidden periodicity)", col, g, pv, "exact", N, note=f"peak period {per:.1f} draws")

    out = T.frame()
    log.info("randomness tests: %d tests, %d raw p<0.05, %d significant after global FDR",
             len(out), int(out["significant_raw"].sum()), int(out["significant_fdr"].sum()))
    return out
