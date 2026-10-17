# Lebanese Loto Research System

A reproducible scientific pipeline that asks one question: **do more than two decades of Lebanese Loto (LLDJ) results contain any statistically defensible, exploitable pattern that helps predict future draws?**

It scrapes the complete official history, validates it, measures every draw against the exact behaviour of a fair 6/42 lottery, runs thousands of Monte Carlo fair histories, tests explicit "hidden algorithm" hypotheses, backtests statistical, machine-learning and sequence models with strict walk-forward validation, and then produces clearly labelled candidate tickets for the next draw.

The findings are in **[`reports/RESEARCH_REPORT.md`](reports/RESEARCH_REPORT.md)**.

## Findings in brief

* **Data:** all 2,458 published draws (#1, 12 Sep 2002 – #2458, 15 Oct 2026) were retrieved. LLDJ's own history contains 11 placeholder duplicates (draws 379–387 around the 2006 war, plus 462 and 569) and 9 day/month-swapped dates. Excluding the placeholders leaves 2,447 genuine draws.
* **Randomness:** of 2,915 tests, 163 fall below p = 0.05 (≈146 expected by chance) and none survives FDR correction. All 28 history-level statistics sit inside the range of 10,000 simulated fair histories.
* **Prediction:** 18 models (hot/cold/overdue, Bayesian, Markov, logistic, random forest, XGBoost, LightGBM, CatBoost, GRU, regimes, ensemble) were walk-forward backtested on an untouched 367-draw holdout. **None beat random selection.** The candidate tickets are therefore random, and labelled that way.
* **The one real pattern is in the players:** numbers ≤ 31 are heavily over-played and consecutive numbers under-played, so unpopular tickets share pari-mutuel prizes with fewer winners. This does not change the odds of winning.

> Every valid six-number ticket has exactly the same 1 in 5,245,786 chance of winning the jackpot in a fair draw. This project tests that rigorously; it cannot change it.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.12+
pip install -r requirements.txt

python -m loto all          # scrape -> validate -> simulate -> analyze -> backtest -> predict
python -m loto dashboard    # interactive dashboard at http://localhost:8501
```

Individual steps:

| Command | What it does |
|---|---|
| `python -m loto scrape` | Download (or resume) every draw page from lldj.com at ≤1 request/second, caching raw HTML |
| `python -m loto update` | Fetch new draws, rebuild the dataset and re-validate |
| `python -m loto validate` | Build SQLite / CSV / Parquet and `data/data_quality_report.json`. **Modelling refuses to run until this passes.** |
| `python -m loto simulate` | Monte Carlo: 10,000 fair histories of the same length + 5 million bulk fair draws |
| `python -m loto analyze` | Number / pair / structure statistics, 2,915 randomness tests with FDR correction, Phase 4 hypotheses, structural filters, player-popularity analysis, HTML reports |
| `python -m loto backtest` | Tune → select → untouched-holdout walk-forward backtest of 19 models + negative controls |
| `python -m loto predict [-n 10] [--filter conservative]` | Candidate tickets for the next scheduled draw |
| `python -m loto hardware` | Detected CPU / RAM / GPU |
| `pytest` | Parser, validation, exact-enumeration, metric and **leakage** tests |

## Deliverables

| File | Contents |
|---|---|
| `data/loto_history.csv`, `.parquet`, `loto_database.sqlite` | Every draw: number, date, six main numbers (sorted and as displayed), bonus, weekday, source URL, page hash; SQLite also holds prize tables and the site's own draw index |
| `data/data_quality_report.json` | Completeness, rule checks, duplicates, gaps, date cross-checks, schedule changes |
| `reports/statistical_analysis.html` | Frequencies, heatmaps, structure vs exact distributions, randomness battery, Monte Carlo comparisons |
| `reports/pattern_analysis.html` | Pairs, triplets, lags, Phase 4 hypotheses, structural filters, player popularity |
| `reports/randomness_tests.csv` | Every test with raw, FDR (global and per family) and Bonferroni p-values |
| `reports/simulation_results.csv` | Real history vs fair-history distributions; bulk simulation vs exact enumeration |
| `reports/model_comparison.csv` | Every model × period: hits, CIs, exact p-values, log score, Brier, calibration, compute, verdict |
| `reports/backtesting_results.csv` | Every out-of-sample forecast: draw, model, ticket, actual numbers, hits, log score |
| `predictions/next_draw_predictions.csv`, `prediction_explanation.md` | 10 candidate tickets with method, structure and the method's backtest record |
| `reports/RESEARCH_REPORT.md` | The final research report |

Supporting outputs: `reports/hypothesis_tests.csv`, `filter_evaluation.csv`, `popularity_analysis.csv`, `negative_controls.csv`, `portfolio_comparison.csv`, `tuning_trials.csv`, `number_statistics.csv`, `pair_statistics.csv`, `triplet_statistics.csv`, and `artifacts/experiments.jsonl` (experiment log).

## Method in brief

**Data.** `lldj.com/en/LatestResults/Loto?draw=N` for N = 1..latest. Only `#resultcontainer` is parsed; the page footer (which shows the *next* jackpot on every historical page) is deliberately ignored. The site's draw-selector endpoint provides an independent list of draw numbers and dates for cross-checking.

**Exact null.** All 5,245,786 combinations are enumerated once, so any structural property (sum, odd/even, decades, runs, endings, …) has an *exact* fair-lottery probability rather than an estimate.

**Multiple testing.** ~2,900 tests are corrected with Benjamini-Hochberg across all tests and within families; extremes (hottest number, most frequent pair, longest absence) are judged against the extremes of 10,000 simulated fair histories of the same length.

**Backtesting.** Draws `[0, 500)` warm-up · `[500, 70%)` tuning (hyper-parameters, ensemble members) · `[70%, 85%)` selection (final model) · `[85%, 100%)` untouched holdout. Every forecast for draw *t* uses draws < *t* only, which `tests/test_leakage.py` verifies for every model. Primary metrics: matching numbers per ticket (random = 6×6/42 = 0.857, exact null distribution) and the log score of per-number probabilities against uniform 1/7. Negative controls rerun the models on shuffled-order and fully simulated fair histories.

**Configuration.** `config.yaml` (rate limits, simulation sizes, backtest periods, seeds). All randomness is seeded.

## Project layout

```
loto/            scraper, parser, storage, validation, features, simulation, randomness tests,
                 hypotheses, filters, analysis + HTML, backtest, predict, pipeline, CLI
loto/models/     statistical.py · ml.py · advanced.py (GRU, KMeans regimes)
dashboard/app.py Streamlit dashboard (history, statistics, models, generator, backtest explorer)
tests/           pytest suite with real page fixtures
```

## Responsible use

Lotteries return less to players than they take in. This repository is a statistics project; nothing here is a reason to spend more than you can comfortably lose.
