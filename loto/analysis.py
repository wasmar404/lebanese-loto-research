"""Phase 3 exploratory analysis, Phase 6 simulation comparison, player-popularity analysis,
and the two HTML research reports."""
from __future__ import annotations

import html
import json
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

from .config import load_config, p, setup_logging
from .features import DECADE_LABELS, exact_distribution, overlap_with_previous, structural_features
from .simulation import indicator

log = setup_logging("loto.analysis")
POOL, PICK = 42, 6
P0 = PICK / POOL


# ----------------------------------------------------------------------------- tables
def number_table(df: pd.DataFrame) -> pd.DataFrame:
    nums = df[[f"n{i}" for i in range(1, 7)]].to_numpy()
    ind = indicator(nums)
    N = len(ind)
    counts = ind.sum(0)
    exp = N * P0
    sd = np.sqrt(N * P0 * (1 - P0))
    rows = []
    for k in range(POOL):
        idx = np.flatnonzero(ind[:, k])
        gaps = np.diff(idx)
        full = np.diff(np.concatenate(([-1], idx, [N])))
        r = {"number": k + 1, "count": int(counts[k]), "expected": exp, "z": (counts[k] - exp) / sd,
             "p_binomial": stats.binomtest(int(counts[k]), N, P0).pvalue,
             "draws_since_last": int(N - 1 - idx[-1]) if len(idx) else N,
             "max_absence": int(full.max()), "mean_gap": float(gaps.mean()) if len(gaps) else np.nan,
             "gap_sd": float(gaps.std(ddof=1)) if len(gaps) > 1 else np.nan,
             "expected_mean_gap": 1 / P0, "expected_gap_sd": float(np.sqrt(1 - P0) / P0),
             "bonus_count": int((df["bonus"] == k + 1).sum())}
        for w in load_config()["analysis"]["rolling_windows"]:
            r[f"freq_last_{w}"] = int(ind[-w:, k].sum())
        rows.append(r)
    out = pd.DataFrame(rows)
    from statsmodels.stats.multitest import multipletests
    out["p_fdr"] = multipletests(out["p_binomial"], method="fdr_bh")[1]
    return out


def pair_table(df: pd.DataFrame) -> pd.DataFrame:
    nums = df[[f"n{i}" for i in range(1, 7)]].to_numpy()
    ind = indicator(nums).astype(np.int32)
    N = len(ind)
    pair = ind.T @ ind
    q = PICK * (PICK - 1) / (POOL * (POOL - 1))
    rows = []
    for a, b in combinations(range(POOL), 2):
        c = int(pair[a, b])
        rows.append({"pair": f"{a + 1}-{b + 1}", "a": a + 1, "b": b + 1, "count": c, "expected": N * q,
                     "lift": c / (N * q), "p_binomial": stats.binomtest(c, N, q).pvalue})
    out = pd.DataFrame(rows)
    from statsmodels.stats.multitest import multipletests
    out["p_fdr"] = multipletests(out["p_binomial"], method="fdr_bh")[1]
    return out.sort_values("count", ascending=False).reset_index(drop=True)


def triplet_table(df: pd.DataFrame, top: int = 50) -> pd.DataFrame:
    nums = df[[f"n{i}" for i in range(1, 7)]].to_numpy()
    cnt: dict[tuple, int] = {}
    for r in nums.tolist():
        for t in combinations(r, 3):
            cnt[t] = cnt.get(t, 0) + 1
    N = len(nums)
    q = (6 * 5 * 4) / (42 * 41 * 40)
    s = sorted(cnt.items(), key=lambda kv: -kv[1])[:top]
    return pd.DataFrame([{"triplet": "-".join(map(str, t)), "count": c, "expected": N * q,
                          "p_at_least_this_for_one_triplet": stats.binom.sf(c - 1, N, q)} for t, c in s])


def popularity_analysis(df: pd.DataFrame, prizes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """How the number of winning grids depends on the winning combination's structure.

    Uses rank 5 (3 good numbers) and rank 4 (4 good numbers) winner counts.  Sales levels
    change over time (and LBP inflated massively after 2019), so each draw's count is
    compared with the median of its 20 neighbouring draws: log(count / local median).
    A positive coefficient means combinations with that feature are *played more often*,
    so their prizes are shared by more winners (lower payout per winner if you win).
    """
    import statsmodels.api as sm

    feats = structural_features(df[[f"n{i}" for i in range(1, 7)]].to_numpy())
    feats["draw"] = df["draw"].to_numpy()
    out = []
    coefs = []
    for rank, label in [(5, "3 good numbers"), (4, "4 good numbers")]:
        g = prizes[prizes["rank"] == rank][["draw", "winning_grids"]].dropna()
        d = feats.merge(g, on="draw")
        d = d[d["winning_grids"] > 0].sort_values("draw")
        lg = np.log(d["winning_grids"].astype(float))
        med = lg.rolling(21, center=True, min_periods=11).median()
        # exclude the draw itself from its local baseline
        d["rel_log_winners"] = lg - med
        d = d.dropna(subset=["rel_log_winners"])
        X = d[["le31", "le12", "consecutive_pairs", "max_same_ending", "mult7", "odd"]].astype(float)
        X = sm.add_constant(X)
        res = sm.OLS(d["rel_log_winners"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        for name in X.columns[1:]:
            coefs.append({"prize_rank": rank, "tier": label, "feature": name, "coef_log_winners": res.params[name],
                          "pct_more_winners_per_unit": 100 * (np.exp(res.params[name]) - 1),
                          "se": res.bse[name], "p_value": res.pvalues[name], "n_draws": int(res.nobs), "r2": res.rsquared})
        # descriptive: by count of numbers <= 31
        out.append(d.groupby("le31")["rel_log_winners"].agg(["count", "mean"]).reset_index().assign(prize_rank=rank))
    return pd.DataFrame(coefs), pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------------- HTML
class Report:
    def __init__(self, title: str, subtitle: str = ""):
        self.title, self.subtitle = title, subtitle
        self.parts: list[str] = []
        self._first_fig = True

    def h2(self, t):
        self.parts.append(f"<h2>{html.escape(t)}</h2>")

    def text(self, t):
        self.parts.append(f"<p>{t}</p>")

    def table(self, df: pd.DataFrame, n: int = 50, floatfmt: str = "{:.4g}"):
        self.parts.append(df.head(n).to_html(index=False, float_format=lambda x: floatfmt.format(x), classes="tbl", border=0))

    def fig(self, f):
        f.update_layout(template="plotly_white", margin=dict(l=40, r=20, t=50, b=40), height=420)
        self.parts.append(f.to_html(full_html=False, include_plotlyjs="cdn" if self._first_fig else False))
        self._first_fig = False

    def save(self, path):
        css = """body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:1150px;margin:24px auto;padding:0 16px;color:#1d1d1f;line-height:1.5}
        h1{margin-bottom:0}h2{margin-top:40px;border-bottom:1px solid #ddd;padding-bottom:4px}.sub{color:#666}
        table.tbl{border-collapse:collapse;font-size:13px;margin:8px 0;display:block;overflow-x:auto}
        table.tbl th,table.tbl td{padding:4px 8px;border-bottom:1px solid #eee;text-align:right}table.tbl th{background:#f6f6f6}
        .note{background:#fff8e1;border-left:4px solid #f0b400;padding:8px 12px}"""
        doc = (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
               f"<title>{html.escape(self.title)}</title><style>{css}</style></head><body><h1>{html.escape(self.title)}</h1>"
               f"<p class='sub'>{html.escape(self.subtitle)}</p>{''.join(self.parts)}</body></html>")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(doc)


def statistical_report(df: pd.DataFrame, numbers: pd.DataFrame, tests: pd.DataFrame, sim_cmp: pd.DataFrame,
                       bulk: pd.DataFrame, path, sims: pd.DataFrame | None = None) -> None:
    nums = df[[f"n{i}" for i in range(1, 7)]].to_numpy()
    N = len(nums)
    feats = structural_features(nums)
    R = Report("Lebanese Loto - Statistical Analysis",
               f"Draws {int(df.draw.min())}-{int(df.draw.max())} ({N} draws, {str(df.draw_date.min())[:10]} to {str(df.draw_date.max())[:10]})")
    R.text("<span class='note'>Every chart compares the real history against the exact or simulated behaviour of a fair 6/42 lottery. "
           "Deviations are only meaningful if they exceed what fair draws produce <i>and</i> survive multiple-testing correction.</span>")

    R.h2("1. Number frequencies")
    fig = go.Figure(go.Bar(x=numbers.number, y=numbers["count"], marker_color=np.where(numbers.p_fdr < 0.05, "#d62728", "#4c78a8")))
    fig.add_hline(y=N * P0, line_dash="dash", annotation_text="expected")
    sd = np.sqrt(N * P0 * (1 - P0))
    fig.add_hrect(y0=N * P0 - 1.96 * sd, y1=N * P0 + 1.96 * sd, fillcolor="gray", opacity=0.12, line_width=0)
    fig.update_layout(title="Main-number frequency (band = 95% range for a single number)", xaxis_title="number", yaxis_title="count")
    R.fig(fig)
    R.table(numbers.sort_values("z", ascending=False)[["number", "count", "expected", "z", "p_binomial", "p_fdr", "draws_since_last",
                                                      "max_absence", "mean_gap", "gap_sd", "bonus_count"]], n=42)
    years = pd.to_datetime(df.draw_date).dt.year
    ind = indicator(nums)
    heat = pd.DataFrame(ind, columns=range(1, 43)).groupby(years.values).mean() / P0
    fig = px.imshow(heat.T, aspect="auto", color_continuous_scale="RdBu_r", origin="lower", zmin=0.5, zmax=1.5,
                    labels=dict(x="year", y="number", color="freq / expected"), title="Frequency by year relative to expectation")
    R.fig(fig)
    roll = pd.DataFrame(ind, columns=range(1, 43)).rolling(250).mean() / P0
    fig = go.Figure()
    for k in [1, 7, 13, 21, 29, 37, 42]:
        fig.add_trace(go.Scatter(x=df.draw, y=roll[k], name=str(k), mode="lines"))
    fig.update_layout(title="Rolling 250-draw frequency / expected (selected numbers)", xaxis_title="draw")
    R.fig(fig)
    fig = go.Figure(go.Bar(x=numbers.number, y=numbers.bonus_count))
    fig.add_hline(y=N / 42, line_dash="dash")
    fig.update_layout(title="Bonus (additional) number frequency", xaxis_title="number")
    R.fig(fig)

    R.h2("2. Structure of each draw vs exact fair-lottery distribution")
    for col, title in [("sum", "Sum of the six numbers"), ("odd", "Odd numbers per draw"), ("low", "Low numbers (1-21) per draw"),
                       ("consecutive_pairs", "Consecutive pairs per draw"), ("primes", "Primes per draw"),
                       ("max_decade_count", "Largest count in one decade"), ("range", "Range (max - min)")]:
        pmf = exact_distribution(col)
        obs = feats[col].value_counts(normalize=True).reindex(pmf.index, fill_value=0)
        fig = go.Figure([go.Bar(x=pmf.index, y=obs.values, name="observed", opacity=0.7),
                         go.Scatter(x=pmf.index, y=pmf.values, name="exact (fair)", mode="lines+markers")])
        fig.update_layout(title=title, yaxis_title="share of draws")
        R.fig(fig)
    ov = overlap_with_previous(nums, 1)
    hyp = stats.hypergeom(42, 6, 6).pmf(np.arange(7))
    fig = go.Figure([go.Bar(x=list(range(7)), y=np.bincount(ov, minlength=7)[:7] / len(ov), name="observed"),
                     go.Scatter(x=list(range(7)), y=hyp, name="exact", mode="lines+markers")])
    fig.update_layout(title="Numbers shared with the previous draw")
    R.fig(fig)

    R.h2("3. Randomness test battery")
    fam = tests.groupby("family").agg(tests=("p_value", "size"), raw_p_below_005=("significant_raw", "sum"),
                                      fdr_significant=("significant_fdr", "sum"), min_p=("p_value", "min")).reset_index()
    fam["expected_raw_by_chance"] = fam["tests"] * 0.05
    R.text(f"{len(tests)} tests in total. Under a fair lottery about 5% of tests ({0.05 * len(tests):.0f}) are expected below p=0.05 by chance alone. "
           f"Observed: {int(tests.significant_raw.sum())} below 0.05; <b>{int(tests.significant_fdr.sum())}</b> remain significant after "
           "Benjamini-Hochberg correction across all tests.")
    R.table(fam, n=40)
    pv = tests["p_value"].sort_values().to_numpy()
    fig = go.Figure([go.Scatter(x=np.arange(1, len(pv) + 1) / len(pv), y=pv, mode="lines", name="observed p-values"),
                     go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="uniform (all nulls true)", line=dict(dash="dash"))])
    fig.update_layout(title="P-value distribution (a fair lottery gives the diagonal)", xaxis_title="quantile", yaxis_title="p-value")
    R.fig(fig)
    R.text("Smallest 25 p-values (with corrections):")
    R.table(tests.sort_values("p_value")[["family", "test", "subject", "statistic", "p_value", "p_fdr_global", "p_bonferroni", "method", "note"]], n=25)

    R.h2("4. Monte Carlo comparison: real history vs fair histories of the same length")
    R.text(f"Each row compares a statistic of the real history with its distribution across {int(sim_cmp.n_simulations.iloc[0]):,} simulated fair histories "
           "of exactly the same length. Extremes such as the hottest number, the longest absence or the most frequent pair are judged against the "
           "<i>extremes</i> of fair histories, which accounts for having searched across all numbers and pairs.")
    R.table(sim_cmp[["statistic", "observed", "sim_mean", "sim_p05", "sim_p95", "p_two_sided", "z_vs_sim"]], n=40)
    if sims is not None:
        obs = sim_cmp.set_index("statistic")["observed"]
        for st, label in [("freq_chi2", "Frequency chi-square"), ("freq_max_abs_z", "Largest |z| of any number (hottest/coldest)"),
                          ("max_absence", "Longest absence of any number (draws)"), ("pair_max", "Count of the most frequent pair"),
                          ("triplet_max", "Count of the most frequent triplet"), ("overlap_lag1_mean", "Mean overlap with previous draw")]:
            fig = go.Figure(go.Histogram(x=sims[st], nbinsx=60, name="fair histories", marker_color="#9ecae1"))
            fig.add_vline(x=float(obs[st]), line_color="#d62728", line_width=3, annotation_text="real history")
            fig.update_layout(title=f"{label}: real vs {len(sims):,} fair histories", showlegend=False, height=320)
            R.fig(fig)
    R.text("Bulk simulation vs exact enumeration (validates the simulator):")
    R.table(bulk, n=30)
    R.save(path)


def pattern_report(df: pd.DataFrame, pairs: pd.DataFrame, triplets: pd.DataFrame, hyp: pd.DataFrame, filt: pd.DataFrame,
                   pop_coef: pd.DataFrame | None, pop_desc: pd.DataFrame | None, path) -> None:
    nums = df[[f"n{i}" for i in range(1, 7)]].to_numpy()
    N = len(nums)
    ind = indicator(nums).astype(np.int32)
    R = Report("Lebanese Loto - Pattern Analysis", f"{N} draws; pairs, triplets, lags, hypotheses, structural filters, player popularity")
    R.h2("1. Pairs and triplets")
    pm = np.zeros((42, 42))
    for r in pairs.itertuples():
        pm[r.a - 1, r.b - 1] = pm[r.b - 1, r.a - 1] = r.lift
    np.fill_diagonal(pm, np.nan)
    R.fig(px.imshow(pm, x=list(range(1, 43)), y=list(range(1, 43)), color_continuous_scale="RdBu_r", zmin=0.7, zmax=1.3,
                    title="Pair co-occurrence lift (observed / expected)"))
    R.text(f"Pairs: {len(pairs)} tested; {int((pairs.p_binomial < 0.05).sum())} with raw p&lt;0.05 (expected by chance ≈ {0.05 * len(pairs):.0f}); "
           f"{int((pairs.p_fdr < 0.05).sum())} after FDR correction.")
    R.text("Most frequent pairs:")
    R.table(pairs.head(15)[["pair", "count", "expected", "lift", "p_binomial", "p_fdr"]])
    R.text("Least frequent pairs:")
    R.table(pairs.tail(10)[["pair", "count", "expected", "lift", "p_binomial", "p_fdr"]])
    R.text("Most frequent triplets (a single fixed triplet's binomial tail; with 11,480 triplets, extremes are expected - see Monte Carlo 'triplet_max'):")
    R.table(triplets.head(15))

    R.h2("2. Lagged relationships")
    rows = []
    for lag in load_config()["analysis"]["lags"]:
        ov = overlap_with_previous(nums, lag)
        rows.append({"lag": lag, "mean_overlap": ov.mean(), "expected": 36 / 42,
                     "share_with_3plus": (ov >= 3).mean(), "expected_3plus": stats.hypergeom(42, 6, 6).sf(2)})
    R.table(pd.DataFrame(rows))
    x = ind - P0
    ac = np.array([[(x[l:, k] * x[:-l, k]).sum() / (x[:, k] ** 2).sum() for l in range(1, 31)] for k in range(42)])
    R.fig(px.imshow(ac, x=list(range(1, 31)), y=list(range(1, 43)), color_continuous_scale="RdBu_r", zmin=-0.08, zmax=0.08,
                    labels=dict(x="lag (draws)", y="number", color="autocorr"),
                    title=f"Autocorrelation of each number's appearance series (±{2 / np.sqrt(N):.3f} ≈ 95% band)"))

    R.h2("3. Hypotheses about an underlying algorithm (Phase 4)")
    R.table(hyp, n=30)

    R.h2("4. Structural plausibility filters (Phase 7)")
    R.text("<span class='note'>No combination is impossible except those violating the rules. Every valid ticket has the same 1 in 5,245,786 "
           "chance. Filters change which tickets you play, never their chance of winning. An efficiency ratio of 1.0 is what a fair lottery gives.</span>")
    R.table(filt, n=30)

    if pop_coef is not None and len(pop_coef):
        R.h2("5. Player popularity (pari-mutuel prize sharing)")
        R.text("Winner counts per prize tier reveal which combinations other players choose. Positive coefficients = more winners than usual "
               "when the winning combination has that feature, i.e. those patterns are popular and prizes are shared more widely. "
               "This does <b>not</b> change your chance of winning; it changes how much you would receive if you won a shared tier.")
        R.table(pop_coef)
        if pop_desc is not None:
            d = pop_desc[pop_desc.prize_rank == 5]
            fig = go.Figure(go.Bar(x=d.le31, y=100 * (np.exp(d["mean"]) - 1), text=d["count"], textposition="outside"))
            fig.update_layout(title="3-number winners vs local median, by how many winning numbers are ≤ 31 (labels: draws)",
                              xaxis_title="winning numbers ≤ 31", yaxis_title="% more winners than usual")
            R.fig(fig)
    R.save(path)
