"""Lebanese Loto research dashboard.  Run:  python -m loto dashboard   (or streamlit run dashboard/app.py)"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loto.config import p  # noqa: E402
from loto.features import DECADE_LABELS, exact_distribution, structural_features  # noqa: E402
from loto.simulation import indicator  # noqa: E402

st.set_page_config(page_title="Lebanese Loto Research", layout="wide")
P0 = 6 / 42


@st.cache_data
def load_draws():
    df = pd.read_parquet(p("parquet"))
    df["draw_date"] = pd.to_datetime(df["draw_date"])
    return df.sort_values("draw").reset_index(drop=True)


@st.cache_data
def load_csv(name):
    path = p("reports_dir") / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_probs():
    path = p("artifacts_dir") / "backtest_probs.npz"
    if not path.exists():
        return None
    z = np.load(path)
    return {k: z[k] for k in z.files}


def summary():
    path = p("artifacts_dir") / "backtest_summary.json"
    return json.loads(path.read_text()) if path.exists() else None


st.sidebar.title("Lebanese Loto Research")
page = st.sidebar.radio("Section", ["Historical Results", "Statistical Analysis", "Model Research", "Prediction Generator", "Backtest Explorer"])
st.sidebar.caption("Every valid ticket has a 1 in 5,245,786 chance of winning the jackpot. This tool tests that claim rigorously; it cannot change it.")

if not p("parquet").exists():
    st.error("No dataset yet. Run `python -m loto scrape` then `python -m loto validate`.")
    st.stop()
df = load_draws()
nums = df[[f"n{i}" for i in range(1, 7)]].to_numpy()
ind = indicator(nums)
N = len(df)

# ------------------------------------------------------------------------------------------
if page == "Historical Results":
    st.header("Historical results")
    qr = p("quality_report")
    if qr.exists():
        q = json.loads(qr.read_text())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Draws", q["n_draws"])
        c2.metric("Completeness", f"{q['completeness_pct']:.2f}%")
        c3.metric("Validation", "PASSED" if q["passed"] else "FAILED")
        c4.metric("Latest", f"#{q['last_draw']['draw']} ({q['last_draw']['date']})")
    if st.button("Update results from lldj.com"):
        from loto import pipeline as pl
        with st.spinner("Fetching new draws (polite rate limit) and re-validating..."):
            pl.step_scrape(refresh_latest=3)
            r = pl.step_validate()
        st.cache_data.clear()
        st.success(f"Validation {'passed' if r['passed'] else 'FAILED'} - {r['n_draws']} draws")
    c1, c2, c3 = st.columns(3)
    draw_q = c1.text_input("Draw number")
    date_rng = c2.date_input("Date range", (df.draw_date.min().date(), df.draw_date.max().date()))
    contains = c3.multiselect("Contains numbers", list(range(1, 43)))
    view = df.copy()
    if draw_q.strip().isdigit():
        view = view[view.draw == int(draw_q)]
    if isinstance(date_rng, tuple) and len(date_rng) == 2:
        view = view[(view.draw_date.dt.date >= date_rng[0]) & (view.draw_date.dt.date <= date_rng[1])]
    for x in contains:
        view = view[(view[[f"n{i}" for i in range(1, 7)]] == x).any(axis=1)]
    cols = ["draw", "draw_date", "n1", "n2", "n3", "n4", "n5", "n6", "bonus", "weekday", "jackpot_winners"]
    st.dataframe(view[cols].sort_values("draw", ascending=False), use_container_width=True, hide_index=True)
    st.download_button("Download CSV", view.to_csv(index=False), "loto_history_filtered.csv", "text/csv")

# ------------------------------------------------------------------------------------------
elif page == "Statistical Analysis":
    st.header("Statistical analysis")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Numbers", "Pairs", "Structure", "Monte Carlo", "Randomness tests"])
    with tab1:
        w = st.select_slider("Window (most recent draws)", [10, 20, 50, 100, 250, 500, N], value=N)
        cnt = ind[-w:].sum(0)
        exp = w * P0
        sd = np.sqrt(w * P0 * (1 - P0))
        fig = go.Figure(go.Bar(x=list(range(1, 43)), y=cnt))
        fig.add_hline(y=exp, line_dash="dash")
        fig.add_hrect(y0=exp - 1.96 * sd, y1=exp + 1.96 * sd, fillcolor="gray", opacity=0.15, line_width=0)
        fig.update_layout(title=f"Frequency in last {w} draws (band: 95% range for one number in a fair lottery)")
        st.plotly_chart(fig, use_container_width=True)
        hc = pd.DataFrame({"number": range(1, 43), "count": cnt, "z": (cnt - exp) / sd}).sort_values("z", ascending=False)
        c1, c2 = st.columns(2)
        c1.write("Hottest in window"); c1.dataframe(hc.head(10), hide_index=True)
        c2.write("Coldest in window"); c2.dataframe(hc.tail(10), hide_index=True)
        st.caption("Hot/cold lists describe the past. The backtest shows whether they predict the future.")
        years = df.draw_date.dt.year
        heat = pd.DataFrame(ind, columns=range(1, 43)).groupby(years.values).mean() / P0
        st.plotly_chart(px.imshow(heat.T, aspect="auto", origin="lower", color_continuous_scale="RdBu_r", zmin=0.5, zmax=1.5,
                                  labels=dict(x="year", y="number", color="freq / expected"), title="Number heatmap by year"),
                        use_container_width=True)
        sel = st.multiselect("Trend: rolling frequency of numbers", list(range(1, 43)), default=[7, 21, 42])
        rw = st.slider("Rolling window", 50, 500, 250, 50)
        roll = pd.DataFrame(ind, columns=range(1, 43)).rolling(rw).mean() / P0
        st.plotly_chart(px.line(roll[sel].assign(draw=df.draw), x="draw", y=sel, title="Rolling frequency / expected"), use_container_width=True)
    with tab2:
        pairs = load_csv("pair_statistics.csv")
        if pairs is not None:
            st.dataframe(pairs.head(30), hide_index=True)
            st.plotly_chart(px.histogram(pairs, x="count", nbins=40, title="Distribution of pair counts (861 pairs)"), use_container_width=True)
        trips = load_csv("triplet_statistics.csv")
        if trips is not None:
            st.write("Most frequent triplets"); st.dataframe(trips.head(20), hide_index=True)
    with tab3:
        feats = structural_features(nums)
        col = st.selectbox("Property", ["sum", "odd", "low", "consecutive_pairs", "longest_run", "primes", "max_decade_count", "range", "decades_used"])
        pmf = exact_distribution(col)
        obs = feats[col].value_counts(normalize=True).reindex(pmf.index, fill_value=0)
        fig = go.Figure([go.Bar(x=pmf.index, y=obs, name="observed", opacity=0.7), go.Scatter(x=pmf.index, y=pmf, name="exact fair", mode="lines+markers")])
        st.plotly_chart(fig, use_container_width=True)
    with tab4:
        cmp = pd.read_csv(p("artifacts_dir") / "history_level_comparison.csv") if (p("artifacts_dir") / "history_level_comparison.csv").exists() else None
        sims_path = p("artifacts_dir") / "simulated_histories.parquet"
        if cmp is not None:
            st.dataframe(cmp[["statistic", "observed", "sim_mean", "sim_p05", "sim_p95", "p_two_sided"]], hide_index=True)
            if sims_path.exists():
                sims = pd.read_parquet(sims_path)
                stat = st.selectbox("Statistic", cmp.statistic.tolist())
                fig = go.Figure(go.Histogram(x=sims[stat], nbinsx=60))
                fig.add_vline(x=float(cmp.set_index("statistic").loc[stat, "observed"]), line_color="red", line_width=3)
                fig.update_layout(title=f"{stat}: real (red) vs {len(sims):,} fair histories")
                st.plotly_chart(fig, use_container_width=True)
    with tab5:
        tests = load_csv("randomness_tests.csv")
        if tests is not None:
            st.metric("Tests run", len(tests))
            c1, c2, c3 = st.columns(3)
            c1.metric("Raw p < 0.05", int(tests.significant_raw.sum()))
            c2.metric("Expected by chance", f"{0.05 * len(tests):.0f}")
            c3.metric("Significant after FDR", int(tests.significant_fdr.sum()))
            fam = st.selectbox("Family", ["(all)"] + sorted(tests.family.unique()))
            t = tests if fam == "(all)" else tests[tests.family == fam]
            st.dataframe(t.sort_values("p_value").head(200), hide_index=True)

# ------------------------------------------------------------------------------------------
elif page == "Model Research":
    st.header("Model research")
    comp = load_csv("model_comparison.csv")
    s = summary()
    if comp is None or s is None:
        st.info("Run `python -m loto backtest` first.")
        st.stop()
    c1, c2, c3 = st.columns(3)
    c1.metric("Final model", s["final_model"])
    c2.metric("Models tested", len(s["verdicts"]))
    c3.metric("Validated", sum(v.startswith("KEEP") for v in s["verdicts"].values()))
    period = st.radio("Period", ["holdout", "selection", "tuning", "all_oos"], horizontal=True)
    c = comp[comp.period == period].sort_values("log_score_gain", ascending=False)
    fig = go.Figure(go.Bar(x=c.model, y=c.mean_hits, error_y=dict(type="data", symmetric=False, array=c.hits_ci_high - c.mean_hits,
                                                                  arrayminus=c.mean_hits - c.hits_ci_low)))
    fig.add_hline(y=36 / 42, line_dash="dash", annotation_text="random 0.857")
    fig.update_layout(title=f"Mean matching numbers per ticket ({period}) with 95% CI")
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(px.bar(c, x="model", y="log_score_gain", error_y=1.96 * c.log_score_gain_se,
                           title="Log-score gain vs uniform (positive = better calibrated forecasts than random)"), use_container_width=True)
    st.dataframe(c[["model", "family", "n_draws", "mean_hits", "hits_ci_low", "hits_ci_high", "p_hits_better_than_random", "p_hits_fdr_across_models",
                    "log_score_gain", "p_log_score_better_than_uniform", "brier", "ece", "calibration_slope", "compute_seconds", "verdict"]], hide_index=True)
    st.subheader("Training configuration")
    st.json({"periods": s["periods"], "tuned_params": s["tuned_params"], "ensemble_members": s["ensemble_members"],
             "tuning_configurations_tried": s["n_tuning_configurations"], "model_fits": s["total_model_fits"]})
    nc = load_csv("negative_controls.csv")
    if nc is not None:
        st.subheader("Negative controls (models run on shuffled and simulated fair histories)")
        st.dataframe(nc, hide_index=True)

# ------------------------------------------------------------------------------------------
elif page == "Prediction Generator":
    st.header("Prediction generator")
    s = summary()
    if s is None:
        st.info("Run `python -m loto backtest` first.")
        st.stop()
    from loto.models.statistical import STAT_MODELS
    from loto.predict import _low_overlap_portfolio, _profile, _sample, next_draw_info
    target = next_draw_info(df)
    st.subheader(f"Target: draw #{target['draw']} on {target['date']}")
    if s["final_model"] == "uniform_random":
        st.warning("No model beat random selection on unseen draws. Model-based tickets below are shown for exploration only; "
                   "every ticket has the same 1 in 5,245,786 chance.")
    c1, c2, c3, c4 = st.columns(4)
    n = c1.number_input("Tickets", 1, 50, 10)
    model = c2.selectbox("Model", ["uniform_random"] + [m for m in STAT_MODELS if m != "uniform_random"])
    filt = c3.selectbox("Structural filter", ["none", "conservative", "moderate", "aggressive"])
    diverse = c4.checkbox("Maximise diversity (low overlap)", True)
    seed = st.number_input("Random seed", 0, 10 ** 6, 2459)
    if st.button("Generate"):
        tuned = s.get("tuned_params", {})
        ext = np.vstack([ind, np.zeros((1, 42))])
        fn = STAT_MODELS[model][0]
        prob = fn(ext, **(tuned.get(model) or {}))[N]
        rng = np.random.default_rng(int(seed))
        tickets = _low_overlap_portfolio(rng, int(n), [], prob, filt) if diverse else [_sample(prob, rng, filt) for _ in range(int(n))]
        rows = []
        for t in tickets:
            pr = _profile(t)
            rows.append({"numbers": " ".join(f"{x:02d}" for x in t), **{k: v for k, v in pr.items() if not k.startswith("_")}})
        out = pd.DataFrame(rows)
        st.dataframe(out, hide_index=True)
        st.download_button("Export CSV", out.to_csv(index=False), f"tickets_draw_{target['draw']}.csv", "text/csv")
    pred = p("predictions_dir") / "next_draw_predictions.csv"
    if pred.exists():
        st.subheader("Pipeline candidates (from `python -m loto predict`)")
        st.dataframe(pd.read_csv(pred), hide_index=True)

# ------------------------------------------------------------------------------------------
elif page == "Backtest Explorer":
    st.header("Backtest explorer")
    probs = load_probs()
    s = summary()
    if probs is None or s is None:
        st.info("Run `python -m loto backtest` first.")
        st.stop()
    start = s["periods"]["all_oos"][0]
    draws = probs["draws"]
    d = st.select_slider("Draw", options=draws[start:].tolist(), value=int(draws[-1]))
    t = int(np.flatnonzero(draws == d)[0])
    actual = set(nums[t].tolist())
    st.subheader(f"Draw #{d} on {df.draw_date.iloc[t].date()}  ·  actual: {' '.join(f'{x:02d}' for x in sorted(actual))}  + bonus {df.bonus.iloc[t]}")
    st.caption("Each model's prediction below was made using only draws before this one.")
    rows = []
    for m, P in probs.items():
        if m == "draws":
            continue
        tk = sorted((np.argsort(-P[t])[:6] + 1).tolist()) if m != "uniform_random" else None
        if tk is None:
            rng = np.random.default_rng(int(d))
            tk = sorted(rng.choice(np.arange(1, 43), 6, replace=False).tolist())
        rows.append({"model": m, "ticket": " ".join(f"{x:02d}" for x in tk), "hits": len(actual & set(tk)),
                     "prob_assigned_to_actual_numbers": float(P[t][[x - 1 for x in actual]].sum())})
    st.dataframe(pd.DataFrame(rows).sort_values("hits", ascending=False), hide_index=True)
    st.caption("A fair lottery gives 0.857 hits per ticket on average and 6 x 1/7 = 0.857 total probability on the six actual numbers.")
