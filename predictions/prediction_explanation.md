# Candidate tickets for Lebanese Loto draw #2459

Target draw: **#2459** on **2026-10-19** 18:45 (source: lldj.com countdown).
Generated 2026-10-17 21:16 from 2447 validated historical draws.

## Bottom line

**No model beat random selection on unseen draws.** After tuning on the early history, choosing between 18 models on a separate selection period, and testing on an untouched holdout of 367 draws, no method's advantage was larger than what chance produces. The final method is therefore **uniform random selection**.

Every ticket below has exactly the same chance of winning the jackpot as any other valid combination: **1 in 5,245,786**. The tickets labelled with model names are shown for transparency only; their method failed validation.

## Candidates

|   rank | numbers           | method                                                                                                                                    |   sum | odd_even   | decades_1-9/10s/20s/30s/40s   |   structural_class_hist_share |   max_overlap_with_other_tickets |
|-------:|:------------------|:------------------------------------------------------------------------------------------------------------------------------------------|------:|:-----------|:------------------------------|------------------------------:|---------------------------------:|
|      1 | 11 15 25 28 29 42 | Random (no model beat random selection out of sample)                                                                                     |   150 | 4/2        | 0/2/3/0/1                     |                        0.0225 |                                1 |
|      2 | 12 14 18 20 22 28 | Ensemble of top-3 tuning-period models (top-6) - NOT validated                                                                            |   114 | 0/6        | 0/3/3/0/0                     |                        0.0004 |                                5 |
|      3 | 04 06 09 27 37 40 | Probability-weighted sample from uniform                                                                                                  |   123 | 3/3        | 3/0/1/1/1                     |                        0.0597 |                                2 |
|      4 | 13 21 22 35 36 39 | Conservative structural filter, random within filtered space                                                                              |   166 | 4/2        | 0/1/2/3/0                     |                        0.0225 |                                2 |
|      5 | 12 14 18 20 28 35 | Alternative model: best on selection period (markov_lag1) top-6 - NOT validated                                                           |   127 | 1/5        | 0/3/2/1/0                     |                        0.0065 |                                5 |
|      6 | 32 36 37 38 41 42 | Low-popularity pattern (structure other players choose least: numbers above 31; same odds, fewer people to share a prize with if it wins) |   226 | 2/4        | 0/0/0/4/2                     |                        0.002  |                                1 |
|      7 | 02 03 08 21 24 34 | Diverse portfolio / low-overlap (simulated annealing for number coverage)                                                                 |    92 | 2/4        | 3/0/2/1/0                     |                        0.02   |                                1 |
|      8 | 01 05 07 26 30 33 | Diverse portfolio / low-overlap (simulated annealing for number coverage)                                                                 |   102 | 4/2        | 3/0/1/2/0                     |                        0.0274 |                                1 |
|      9 | 10 16 17 19 23 31 | Diverse portfolio / low-overlap (simulated annealing for number coverage)                                                                 |   116 | 4/2        | 0/4/1/1/0                     |                        0.018  |                                1 |
|     10 | 04 13 17 26 27 39 | Additional random valid ticket                                                                                                            |   126 | 4/2        | 1/2/2/1/0                     |                        0.0425 |                                2 |

The 10 tickets cover 42 distinct numbers. Diversifying tickets changes *which* outcomes you cover, not the probability of any individual ticket.

## Backtest record of each method (untouched holdout)

| model                  | family           |   n_draws |   mean_hits |   p_hits_better_than_random |   log_score_gain |   p_log_score_better_than_uniform | verdict                                      |
|:-----------------------|:-----------------|----------:|------------:|----------------------------:|-----------------:|----------------------------------:|:---------------------------------------------|
| xgboost                | machine_learning |       367 |      0.8910 |                      0.2184 |           0.0004 |                            0.3808 | REJECT (no edge over random on unseen draws) |
| logistic_l1            | machine_learning |       367 |      0.9019 |                      0.1504 |           0.0002 |                            0.4402 | REJECT (no edge over random on unseen draws) |
| uniform_random         | baseline         |       367 |      0.8747 |                      0.3482 |           0.0000 |                            0.5000 | BASELINE                                     |
| hist_gradient_boosting | machine_learning |       367 |      0.8801 |                      0.3017 |          -0.0009 |                            0.7423 | REJECT (no edge over random on unseen draws) |
| ensemble_top3          | ensemble         |       367 |      0.9128 |                      0.0983 |          -0.0019 |                            0.8102 | REJECT (no edge over random on unseen draws) |
| bayes_shrinkage        | statistical      |       367 |      0.8665 |                      0.4223 |          -0.0022 |                            0.8719 | REJECT (no edge over random on unseen draws) |
| lightgbm               | machine_learning |       367 |      0.8801 |                      0.3017 |          -0.0024 |                            0.7351 | REJECT (no edge over random on unseen draws) |
| logistic_l2            | machine_learning |       367 |      0.8856 |                      0.2583 |          -0.0039 |                            0.8171 | REJECT (no edge over random on unseen draws) |
| markov_lag1            | statistical      |       367 |      0.9019 |                      0.1504 |          -0.0053 |                            0.9156 | REJECT (no edge over random on unseen draws) |
| random_forest          | machine_learning |       367 |      0.8883 |                      0.2379 |          -0.0064 |                            0.9588 | REJECT (no edge over random on unseen draws) |
| gru_sequence           | advanced         |       367 |      0.8747 |                      0.3482 |          -0.0068 |                            0.9179 | REJECT (no edge over random on unseen draws) |
| cold_overdue           | statistical      |       367 |      0.9210 |                      0.0690 |          -0.0074 |                            0.8242 | REJECT (no edge over random on unseen draws) |
| catboost               | machine_learning |       367 |      0.8856 |                      0.2583 |          -0.0095 |                            0.9350 | REJECT (no edge over random on unseen draws) |
| pair_association       | statistical      |       367 |      0.9128 |                      0.0983 |          -0.0169 |                            0.9840 | REJECT (no edge over random on unseen draws) |
| frequency_hot          | statistical      |       367 |      0.8665 |                      0.4223 |          -0.0198 |                            0.9925 | REJECT (no edge over random on unseen draws) |
| ewma_frequency         | statistical      |       367 |      0.8692 |                      0.3971 |          -0.0338 |                            0.9872 | REJECT (no edge over random on unseen draws) |
| kmeans_regime          | advanced         |       367 |      0.8937 |                      0.1999 |          -0.0351 |                            0.9929 | REJECT (no edge over random on unseen draws) |
| gap_hazard             | statistical      |       367 |      0.8147 |                      0.8520 |          -0.0438 |                            0.9998 | REJECT (no edge over random on unseen draws) |
| recency_window         | statistical      |       367 |      0.8065 |                      0.8930 |          -0.0457 |                            0.9951 | REJECT (no edge over random on unseen draws) |

Random selection averages 0.857 matching main numbers per ticket.

## How to read this

* `structural_class_hist_share` is the share of past draws with the same odd count, low count and number of decades used. It describes how typical a ticket looks; it is **not** a probability of winning.
* The low-popularity candidate avoids patterns many players choose (mostly numbers ≤ 31). It has the same odds but, in prize tiers shared between winners, it would be shared with fewer people.
* Lotteries are designed to return less than is staked. Play only amounts you are comfortable losing.