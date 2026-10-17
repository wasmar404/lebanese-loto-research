# Does the Lebanese Loto contain a predictable signal?

**A complete statistical and machine-learning investigation of 24 years of LLDJ Loto draws (#1, 12 Sep 2002 – #2458, 15 Oct 2026)**

---

## Summary

**No.** Across the full official history there is no statistically defensible pattern that predicts future draws.

* **Data.** All 2,458 published draws were retrieved, with zero failures. 11 of them are placeholder copies in LLDJ's own records, which leaves 2,447 genuine results (§1).
* **Randomness.** About 2,900 randomness tests found nothing that survives correction for multiple testing. The number of raw "hits" is what chance alone produces (§3).
* **Simulation.** On every one of 28 history-level statistics, the real history sits inside the range of 10,000 simulated fair histories of the same length (§4).
* **Hidden-algorithm hypotheses.** All were rejected: generator-style maps, hidden regimes, drift, autoregression and Markov dependence (§5).
* **Models.** Eighteen forecasting models were backtested with strict walk-forward validation: statistical, machine learning (including XGBoost, LightGBM and CatBoost), a GRU sequence network, a regime model and an ensemble. **None beat random selection on unseen draws** (§6).
* **Candidates.** The 10 candidate tickets for draw #2459 (§8) are therefore labelled for what they are. Each has the same 1-in-5,245,786 chance as any other valid combination.

The study produced one useful, practical finding, and it concerns **other players, not the balls**. Lebanese players heavily favour numbers ≤ 31 (birthdays) and avoid consecutive numbers. Prize tiers 1–4 are shared among winners, so a ticket built from unpopular numbers has the same chance of winning but would be shared with fewer people (§7).

| Evidence class | Finding |
|---|---|
| **Confirmed** | Results are consistent with a fair 6/42 draw: uniform numbers, independent draws, exact structural distributions, stable over time. |
| **Confirmed** | The published history contains 11 placeholder duplicates and 9 day/month-swapped dates (§1). |
| **Confirmed** | Player choices are strongly non-uniform (numbers ≤ 31 over-played; consecutive numbers under-played), p < 10⁻²⁰. |
| **Exploratory, not significant** | The overall frequency spread (p ≈ 0.05); the hottest number 20 and coldest 26; residue classes mod 5/10; the "day-of-month number" effect; a small hot-number edge over the whole out-of-sample period. None survives multiple-testing correction, and the one that could be tested on fresh data (hot numbers) disappeared there. |
| **Unsupported / rejected** | Hot/cold/overdue numbers, recurring pairs or triplets, cycles, modular rules, Markov or LCG-style generators, hidden regimes, calendar effects, structural "plausibility" filters, bonus-number prediction, and every ML or sequence model tested. |

---

## 1. Data: what was retrieved and how complete it is

**Source.** `https://www.lldj.com/en/LatestResults/Loto?draw=N` for N = 1…2458. The site publishes no `robots.txt` (HTTP 404), so no crawl restrictions apply. The scraper still ran at ≤ 1 request per second: 2,583 requests in 41 minutes, with caching and resumable progress. Only the result block of each page is parsed. Every historical page also shows the *next* draw's jackpot (LBP 106 billion) in its footer, and that footer is ignored. A unit test confirms it never leaks into the data.

**Cross-checks.**
* The site's own draw-selector endpoint lists all 2,458 draw numbers and dates, and its dates agree with all 2,458 result pages.
* A random sample of 15 draws (plus the first, the latest and one placeholder draw) was re-downloaded and matched the stored dataset exactly.
* All 2,458 records obey the game rules: six distinct numbers in 1–42, and a bonus in 1–42 that is never one of the main numbers.

**Defects in the official history** (documented in `data/data_quality_report.json` and `loto/clean.py`; raw values are kept):

| Defect | Draws | Treatment |
|---|---|---|
| **Placeholder duplicates.** Ten consecutive draws show the identical result 1-4-7-11-12-32 (bonus 15), all dated 17 Aug 2006. | 379–388 | Keep 388, exclude 379–387 |
| Identical consecutive pair | 462 = 463 (7 May 2007) | Keep 463, exclude 462 |
| Identical consecutive pair | 569 = 570 (15 May 2008) | Keep 570, exclude 569 |
| **Day and month swapped** in the published date | 229, 246, 264, 271, 273, 279, 280, 290, 342 (2005–06) | Corrected. Each swapped date falls exactly between its neighbours on a Monday or Thursday draw day |
| Date cannot be resolved | 231, 232 (both "20 Feb 2005", a Sunday), 244 ("31 Jan 2005" between March and April draws) | Kept, flagged; excluded only from calendar tests |

A fair draw repeats the previous combination with probability 1 in 5,245,786, so the identical runs are record errors, not outcomes. The war-period run fits the calendar precisely. Draw 378 is dated 13 July 2006, the day after the July 2006 war began. Nine twice-weekly slots (17 Jul – 14 Aug) then pass before the 17 August draw, and the nine placeholders (379–387) carry its result. In 2007 and 2008 one scheduled draw is missing just before each duplicated pair (Thursday 3 May 2007; Monday 12 May 2008), so one genuine result is lost in each case.

**This matters.** A [2020 statistics post](https://dataviz.etublogs.usj.edu.lb/2020/11/17/monday-or-thursday-loto-at-first-sight/) about the Lebanese Loto reported 4-12-32 as the most frequent triplet and found that the top triplets all come from {1, 4, 7, 11, 12, 32}. That is the duplicated placeholder combination counted ten times. In the cleaned data the pair 4-12 appears 62 times, which is unremarkable.

**Completeness.** 100% of draw IDs are present. **2,447 genuine draws** are used for analysis, and every critical validation check passes on this set. Draw order is unavailable: only 6 pages (all before #1286) list the balls unsorted, so ball-position tests could not be run.

**Schedule** (consistent with LLDJ's history: weekly in 2002, Monday and Thursday from 2003). There are 1,160 Monday draws, 1,219 Thursday draws and 65 rescheduled draws, mostly Tuesdays after Easter and Christmas. Four long interruptions appear: July–Aug 2006 (war, 35 days), Mar–Apr 2020 (46 days) and Jan–Feb 2021 (35 days), which match the COVID lockdowns, and Sep–Oct 2024 (18 days), during the escalation.

## 2. The game, mathematically

Six distinct numbers are drawn from 1–42, plus an additional number from the remaining 36. There are C(42,6) = **5,245,786** combinations. The rules have not changed over the whole history; see `docs/GAME_RULES.md` for sources, the draw procedure and the 2021–2023 price changes.

To have an exact reference, **all 5,245,786 combinations were enumerated**. Every structural property (sum, odd/even, decades, consecutive runs, endings, primes, multiples, range, gaps) therefore has an exact fair-lottery distribution, not an estimate. Under a fair draw, every ticket matches on average 6 × 6/42 = **0.857** numbers, whichever ticket it is and whatever happened before.

## 3. Randomness tests

**2,915 tests** were run, covering every area named in the brief. Under a fair lottery about 146 of them fall below p = 0.05 by chance alone; **163 did**. After Benjamini–Hochberg correction across all tests, **0 remain significant**. The smallest raw p-value (0.00035, pair 2-35) becomes 0.62 after correction. Small raw p-values cluster in the largest families (1,764 conditional-dependence and 861 pair tests), exactly as chance predicts. Every test, with raw, family-FDR, global-FDR and Bonferroni p-values, is in `reports/randomness_tests.csv`.

| Area | What was tested | Result |
|---|---|---|
| Number frequency | Chi-square over 42 numbers, calibrated by Monte Carlo for the within-draw dependence | χ² = 49.2, p = 0.053, not significant. Hottest 20 (396 vs 349.6, z = 2.68), coldest 26 (309, z = −2.34); FDR p = 0.27 for both |
| Bonus number | Uniform over 1–42, and uniform over the 36 numbers left after the main draw | p = 0.23 and p = 0.22 |
| Draw structure | 25 properties vs exact enumeration | Smallest p = 0.03 (repeated last digits), FDR 0.85. Sum p = 0.94 |
| Consecutive draws | Overlap with the draw 1, 2, 3, 5, 10, 20 and 50 back vs the exact hypergeometric | Mean overlap 0.851 vs 0.857 expected (p = 0.21); all lags p > 0.06 |
| Serial dependence | Ljung-Box and runs tests for each number and for sums, odd and low counts | Raw rejections at the chance rate; none after FDR |
| Conditional dependence | "i in this draw → j in the next", 1,764 tables | 96 raw p < 0.05 (88 expected); smallest FDR p = 0.62 |
| Pairs | 861 exact binomial tests | 50 raw (43 expected), none after FDR. The rarest pair, 2-35 (21 vs 42.6), has FDR p = 0.30 across pairs (0.62 across all tests) |
| Drift and change points | Eras, years, CUSUM per number, PELT on sums | Era p = 0.91, year p = 0.52; PELT finds 0 change points |
| Calendar | Number × weekday, month, day-of-month block; "number = day of month" | p = 0.80, 0.48, 0.11; the day-of-month number appeared 396 times vs 349 expected (p = 0.008, FDR 0.78) |
| Periodicity | Fisher's g for each number and for sums | 1 of 45 raw |
| Modular arithmetic | Residue classes mod 3–12, sum mod m | Classes mod 5 and mod 10 (last digit) have p ≈ 0.006 (FDR 0.054 within the family, 0.80 across all tests). These are regroupings of the same 42 counts, driven by 20, 14 and 26 |

**Calibration check.** The same battery run on one *simulated fair* history gave 158 of 2,928 tests below 0.05 and 0 after FDR. The real history behaves exactly like a fair one.

## 4. Monte Carlo: is anything more extreme than fair draws produce?

Some pattern always looks remarkable when you search 42 numbers, 861 pairs and 11,480 triplets. The right comparison is the *extremes* of fair histories. **10,000 simulated fair histories, each 2,447 draws long** (24.5 million draws), plus 5 million bulk draws to verify the simulator against the exact enumeration:

| Statistic | Real | Fair mean | Fair 5–95% | p (two-sided) |
|---|---|---|---|---|
| Frequency chi-square | 49.2 | 36.1 | 24.0–50.2 | 0.12 |
| Largest \|z\| of any number | 2.68 | 2.45 | 1.82–3.26 | 0.57 |
| Longest absence of any number | 66 | 66.3 | 56–82 | 0.94 |
| Most frequent pair | 66 | 64.7 | 61–70 | 0.66 |
| Least frequent pair | 21 | 23.5 | 20–26 | 0.28 |
| Most frequent triplet | 14 | 14.3 | 13–16 | 1.00 |
| Mean overlap with previous draw | 0.851 | 0.857 | 0.831–0.884 | 0.70 |
| Transition chi-square (draw → next) | 1,906 | 1,729 | 1,564–1,914 | 0.11 |
| Repeated complete combinations | 0 | 0.56 | 0–2 | 1.00 |

All 28 statistics are in `reports/simulation_results.csv`; the smallest p is 0.088. The most frequent triplet (8-25-32 and 4-16-39, 14 times each against 4.3 expected) would look like p = 0.0001 if it were the only triplet. With 11,480 of them, it is exactly what fair histories produce.

## 5. Hypotheses about an underlying algorithm

Each hypothesis was stated together with the evidence that would distinguish it from randomness, then tested against simulated fair or shuffled histories.

| Hypothesis | Test | p | FDR p | Verdict |
|---|---|---|---|---|
| H1 LCG/affine generator maps each draw to the next | Best of 504 maps x→ax+b (mod 42): mean overlap; share ≥ 4; consistency of the winning map | 0.99 / 0.93 / 0.62 | ≥ 0.93 | Rejected |
| H2 Hidden regimes in draw structure | BIC gain of a 2-state HMM vs fair histories | 0.20 | 0.90 | Rejected |
| H3 Frequency drift (biased balls in some period) | Max chi-square over rolling 250-draw windows | 0.22 | 0.90 | Rejected |
| H4 Autoregression of sum, odd, low, range | AR(10) F-tests | 0.45–0.88 | ≥ 0.93 | Rejected |
| H5 First-order Markov dependence | Transition chi-square vs shuffled order | 0.27 | 0.90 | Rejected |

Observing outputs cannot in general reverse-engineer a random-number generator, and these tests do not claim to. They rule out the simple generator-like structures that would be exploitable. Public descriptions say the Loto is drawn mechanically from 42 balls under National Lottery supervision. A mechanical bias would show up as persistent per-number deviations, drift or change points, and none were found.

## 6. Prediction models and backtesting

**Protocol.** Strict walk-forward validation: the forecast for draw *t* uses only draws before *t*, and `tests/test_leakage.py` verifies this for every model.

| Period | Draws | Purpose |
|---|---|---|
| Warm-up | #1–#510 (2002–2007; 500 genuine draws) | Training only |
| Tuning | #511–#1724 (Oct 2007 – Jun 2019; 1,213 draws) | Hyper-parameters (61 configurations, Optuna for XGBoost and LightGBM) and ensemble members |
| Selection | #1725–#2091 (Jun 2019 – Mar 2023; 367 draws) | Choose the final model |
| Holdout | #2092–#2458 (Mar 2023 – Oct 2026; 367 draws) | Untouched until the final model was fixed |

**Metrics.** Matching numbers per top-6 ticket, with exact p-values from the hypergeometric distribution. Also the log score and Brier score of each model's 42 per-number probabilities against uniform 1/7, with Newey-West errors, plus calibration. P-values are corrected across models. The study ran 575 model fits.

**Models.** Statistical: uniform, hot (frequency), Bayesian shrinkage, cold/overdue, recency window, exponentially weighted frequency, gap hazard, Markov, pair association. ML: L2 and L1 logistic regression, random forest, histogram gradient boosting, XGBoost, LightGBM, CatBoost; 19 history-only features per number (recent and long-run frequencies, absence length, overdue ratio, exponentially weighted frequencies, appearance in the last three draws, Markov and pair lifts, number identity, previous-draw sum). Advanced: a GRU sequence network over the last 20 draws, a k-means regime model, and an ensemble of the three best tuning-period models.

**Results on the untouched holdout** (random = 0.857 matches per ticket; log-score gain > 0 would mean better forecasts than uniform):

| Model | Family | Matches per ticket (95% CI) | p vs random | p (FDR, 19 models) | Log-score gain ± SE | Verdict |
|---|---|---|---|---|---|---|
| XGBoost | machine learning | 0.891 (0.808–0.974) | 0.22 | 0.47 | +0.0004 ± 0.0013 | Reject |
| Logistic (L1) | machine learning | 0.902 (0.818–0.986) | 0.15 | 0.47 | +0.0002 ± 0.0014 | Reject |
| Uniform random (baseline) | baseline | 0.875 (0.789–0.960) | 0.35 | 0.47 | +0.0000 ± 0.0000 | baseline |
| Hist. gradient boosting | machine learning | 0.880 (0.796–0.964) | 0.30 | 0.47 | -0.0009 ± 0.0014 | Reject |
| Ensemble (top 3) | ensemble | 0.913 (0.830–0.995) | 0.10 | 0.47 | -0.0019 ± 0.0022 | Reject |
| Bayesian shrinkage | statistical | 0.866 (0.787–0.946) | 0.42 | 0.47 | -0.0022 ± 0.0020 | Reject |
| LightGBM | machine learning | 0.880 (0.797–0.963) | 0.30 | 0.47 | -0.0024 ± 0.0038 | Reject |
| Logistic (L2) | machine learning | 0.886 (0.801–0.970) | 0.26 | 0.47 | -0.0039 ± 0.0043 | Reject |
| Markov (lag 1) | statistical | 0.902 (0.823–0.981) | 0.15 | 0.47 | -0.0053 ± 0.0039 | Reject |
| Random forest | machine learning | 0.888 (0.805–0.972) | 0.24 | 0.47 | -0.0064 ± 0.0037 | Reject |
| GRU sequence network | advanced | 0.875 (0.794–0.955) | 0.35 | 0.47 | -0.0068 ± 0.0049 | Reject |
| Cold / overdue | statistical | 0.921 (0.835–1.007) | 0.07 | 0.47 | -0.0074 ± 0.0079 | Reject |
| CatBoost | machine learning | 0.886 (0.803–0.969) | 0.26 | 0.47 | -0.0095 ± 0.0062 | Reject |
| Pair association | statistical | 0.913 (0.829–0.996) | 0.10 | 0.47 | -0.0169 ± 0.0079 | Reject |
| Hot numbers | statistical | 0.866 (0.787–0.946) | 0.42 | 0.47 | -0.0198 ± 0.0081 | Reject |
| EWMA frequency | statistical | 0.869 (0.787–0.951) | 0.40 | 0.47 | -0.0338 ± 0.0151 | Reject |
| K-means regimes | advanced | 0.894 (0.810–0.977) | 0.20 | 0.47 | -0.0351 ± 0.0143 | Reject |
| Gap hazard | statistical | 0.815 (0.736–0.894) | 0.85 | 0.89 | -0.0438 ± 0.0123 | Reject |
| Recency window | statistical | 0.807 (0.725–0.889) | 0.89 | 0.89 | -0.0457 ± 0.0177 | Reject |

All 19 rows for all four periods, with Brier score, calibration error, calibration slope and compute time, are in `reports/model_comparison.csv`.

On the selection period, the best model (Markov) averaged 0.902 matches per ticket (FDR p = 0.41) with log-score gain p = 0.10. That is not significant, so **the final method is uniform random selection**. No model passes on either period. Every verdict is **REJECT**.

**Telling signs of no signal.**
* Tuning drove every statistical model toward uniform. It chose the strongest shrinkage (α = 1,000), the longest window (500 draws), the longest half-life (250 draws) and the weakest Markov and overdue effects available.
* Most models score *worse* than uniform on log score. They learn noise and pay for it on new draws.
* The GRU's forecasts stayed within 0.10–0.19 per number. It found nothing to learn.

**The closest thing to a signal: "hot numbers".** Over all 1,947 out-of-sample draws, picking the six historically most frequent numbers averaged 0.895 matches vs 0.857 (raw p = 0.019). That does not survive correction across 18 models (FDR p = 0.13). On the untouched holdout it fell to 0.867 (p = 0.42). The negative controls explain it. Applied to the *same draws in shuffled order*, the strategy shows the same edge (+0.029 vs +0.034). So the effect is the history-wide frequency spread (§3, p ≈ 0.05), not something that persists through time. On fully simulated fair histories its edge reaches +0.034 by chance. Even taking the edge at face value, it would mean each chosen number had a 14.9% instead of a 14.3% chance of being drawn. That changes no practical outcome and does not offset the ~55% of stakes that are not returned as prizes.

**Multiple tickets.** At equal ticket counts (10 per draw on the holdout), model-weighted portfolios and random portfolios gave the same results: 2.16 best-ticket matches, and a 3-match or better in 91–92 of 367 draws. The model probabilities are so close to uniform that they choose essentially the same tickets.

**Bonus number.** Hot, cold, repeat-last and random bonus guesses all hit at the 1-in-42 chance rate (all p ≥ 0.38 on the holdout).

**Power.** On the 367-draw holdout, the smallest detectable improvement is about 0.12 matches per ticket (14%). Over all 1,947 out-of-sample draws it is about 0.05. Smaller biases cannot be ruled out with 24 years of data, but a bias that small would also be useless.

## 7. Structural filters and player popularity

**Filters (Phase 7).** None of the "unusual" combinations is impossible. All six in one decade, five consecutive numbers, all odd, extreme sums and the rest are rare only because there are few such combinations, and each one has the same 1/5,245,786 chance. Measured on the history, every filter removed winners at its exact share of the combination space. On the holdout, efficiency (winners kept ÷ space kept) ranged from 0.96 to 1.03, against 1.00 for a fair lottery. The aggressive mode removes 29.7% of combinations and would have discarded 30% of holdout winners. **The default is no filtering**, because no filter improved anything.

| Mode | Share of combinations excluded | Historical winners excluded (expected if fair) |
|---|---|---|
| Conservative (single decade, 5+ consecutive, arithmetic progressions) | 0.04% | 1 (1.0) |
| Moderate (+ all odd/even, all low/high, 5+ in a decade, extreme 1% sums) | 5.4% | 130 (132) |
| Aggressive (+ 5–95% sums, 4+ consecutive, 4+ same ending, 1 or 5 odd, narrow range) | 29.7% | 701 (727) |

**Player popularity: the one actionable finding.** LLDJ publishes how many tickets won each tier, and tiers 1–4 are shared among winners. Each draw's winner count was compared with the median of its 20 neighbouring draws, which removes sales trends and the 2019+ currency collapse, and regressed on the winning numbers' structure with HAC standard errors:

| Feature of the winning combination | Effect on 4-match winners | Effect on 3-match winners |
|---|---|---|
| Each number ≤ 31 | **+9.5%** (p < 10⁻²⁰) | +5.4% (p < 10⁻²⁰) |
| Each pair of consecutive numbers | −2.1% (p = 0.002) | −1.3% (p = 0.04) |
| Each repeated last digit (max same ending) | +4.9% (p < 10⁻⁵) | +2.0% (p = 0.008) |
| Each multiple of 7 | +1.2% (p = 0.02) | +1.4% (p = 0.001) |

When all six winning numbers were ≤ 31, there were about 20% more 4-match winners than usual. With only two numbers ≤ 31, there were about 18% fewer. In plain terms: players over-choose birthday numbers and multiples of 7 and avoid consecutive numbers. **This cannot change your chance of winning.** It changes how many people share a prize if you do. It is the only practical "edge" the data support.

## 8. Candidates for draw #2459 (Monday 19 October 2026, 18:45)

Because no model validated, the system's final method is **uniform random selection**. The candidates come from the pipeline (`predictions/next_draw_predictions.csv`, seed 2459). Tickets 2 and 5 are shown for transparency; their methods **failed** validation.

| # | Numbers | Method | Sum | Odd/even | Decades 1-9/10s/20s/30s/40s | Typical structure |
|---|---|---|---|---|---|---|
| 1 | **11 · 15 · 25 · 28 · 29 · 42** | Random selection (final method: nothing beat it) | 150 | 4/2 | 0/2/3/0/1 | 2.2% |
| 2 | **12 · 14 · 18 · 20 · 22 · 28** | Ensemble top-6 (failed validation) | 114 | 0/6 | 0/3/3/0/0 | 0.04% |
| 3 | **04 · 06 · 09 · 27 · 37 · 40** | Random, probability-weighted (uniform) | 123 | 3/3 | 3/0/1/1/1 | 6.0% |
| 4 | **13 · 21 · 22 · 35 · 36 · 39** | Random within conservative filter | 166 | 4/2 | 0/1/2/3/0 | 2.2% |
| 5 | **12 · 14 · 18 · 20 · 28 · 35** | Markov top-6 (failed validation) | 127 | 1/5 | 0/3/2/1/0 | 0.7% |
| 6 | **32 · 36 · 37 · 38 · 41 · 42** | Low-popularity numbers (§7) | 226 | 2/4 | 0/0/0/4/2 | 0.2% |
| 7 | **02 · 03 · 08 · 21 · 24 · 34** | Diverse portfolio (low overlap) | 92 | 2/4 | 3/0/2/1/0 | 2.0% |
| 8 | **01 · 05 · 07 · 26 · 30 · 33** | Diverse portfolio (low overlap) | 102 | 4/2 | 3/0/1/2/0 | 2.7% |
| 9 | **10 · 16 · 17 · 19 · 23 · 31** | Diverse portfolio (low overlap) | 116 | 4/2 | 0/4/1/1/0 | 1.8% |
| 10 | **04 · 13 · 17 · 26 · 27 · 39** | Random | 126 | 4/2 | 1/2/2/1/0 | 4.2% |

* **Every ticket has a 1 in 5,245,786 chance of the jackpot.** Ranks are an ordering of methods, not of odds.
* *Typical structure* is the share of past draws with the same odd count, low count and number of decades used. It describes appearance, not probability.
* Ticket 6 is the low-popularity construction from §7: numbers above 31, consecutive numbers allowed.
* The ten tickets cover 42 distinct numbers. Diversifying changes *which* outcomes you cover, not the odds of any single ticket.
* Bonus number: no evidence of predictability; every number left after the main draw is equally likely.

## 9. Limitations

* **Single source.** All results come from lldj.com. They are cross-checked against the site's own index and a live re-check, but no third-party archive was used. The 11 placeholder slots have no genuine result available, and none was invented.
* **No draw order or equipment data.** Ball sequence, machine and ball-set changes are not published, so tests specific to a machine or ball set were not possible.
* **Power.** See §6. Very small biases are undetectable, and equally unusable.
* **Popularity analysis is observational.** The local-median normalisation removes sales trends but not every draw-specific factor, such as jackpot rollovers. The direction and size of the birthday effect are consistent with lotteries elsewhere.
* **Model scope.** The networks were kept small on purpose; 2,447 draws cannot support large models. Testing more model families would not fix this, because the problem is the absence of a signal, which §§3–5 establish without any model.

## 10. Reproducibility and computational effort

* `python -m loto all` reproduces everything from scratch. All randomness is seeded (`config.yaml`).
* Hardware: 2 CPUs, 7.8 GB RAM, no GPU; independent simulations run in parallel.
* Simulated fair draws: about 44 million (10,000 full histories, 5,000 histories for count calibration, 5 million bulk draws, 1 million for filter checks, plus null histories for the hypothesis tests and 20 shuffled and 20 simulated negative-control histories).
* Exact enumeration of 5,245,786 combinations. 2,915 randomness tests and 10 hypothesis tests. 18 models, 61 tuning configurations, 575 walk-forward fits.
* Wall time for the full analysis: about 15 minutes after the 41-minute scrape. Every run is logged in `artifacts/experiments.jsonl`.

---

*Lotteries are designed to pay out less than they take in; under the 2016 contract, 45% of stakes go to prizes. Nothing in this report is a reason to play more than you can comfortably lose.*
