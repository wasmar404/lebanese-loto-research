# Lebanese Loto: game rules, mechanism and history (Phase 2)

## Rules (as published by LLDJ, October 2026)

* A basic grid is **6 distinct numbers from 1 to 42**. Advanced grids of 7–10 (online up to 12) numbers are bundles of 6-number combinations and do not change the draw.
* Each draw selects **six main numbers and one additional (bonus) number** from the same 42 balls. The bonus can never equal a main number, so it is drawn from the 36 balls left.
* Prize ranks: 6 numbers (jackpot); 5 + additional; 5; 4; 3. LLDJ states that the jackpot and ranks 2–4 depend on the number of players and winners (pari-mutuel). Rank 5 (3 numbers) is a fixed amount (currently LBP 400,000).
* Number of possible tickets: C(42, 6) = **5,245,786**.

Source: lldj.com, "How To Play" and "Prizes & Ranking" pages (fetched 17 Oct 2026).

## Draw mechanism and supervision

* Draws are held under the supervision of the National Lebanese Lottery and broadcast live on television (LBC International, historically at 7:30 pm). The current countdown on lldj.com targets 18:45 local time.
* Secondary sources describe the draw as six numbers drawn from 42 balls plus one additional number. LLDJ's pages do not document the machine model or any equipment changes. **No public record of a change in machinery or procedure was found.** Phase 3 drift and change-point tests are the empirical check.

## History and schedule

* LLDJ has operated the Loto since 2002, succeeding SOLEI (operator since 1986). Its contract was renewed in 2016 for ten years. **Draw #1 took place on 12 September 2002.**
* At launch, draws were **weekly**. With the new Loto identity in 2003, draws doubled to **Monday and Thursday**, which is the current schedule. `data/data_quality_report.json` records the empirical weekday counts and gaps.
* Grid prices changed with the currency collapse: LL2,000 (2010–2021), LL3,000 (Apr 2021), LL8,000 (Apr 2022), LL20,000 (Feb 2023), LL50,000 (Jul 2023). The site now lists LBP 100,000 for a ticket. Each increase cut sales by roughly 30% for months. This changes **how many people play**, not how numbers are drawn, but it matters for the winner-count (popularity) analysis, which therefore compares each draw with its local median.
* Revenue split under the 2016 contract: 45% prizes, 41.5% state, 5% retailers, 8.5% LLDJ. On average a player gets back less than half of what is staked.

## Consequences for the analysis

* The 6/42 + additional format is the same throughout the 2002–2026 data (verified per draw by the validation step). No source reports a change in the number pool, so the dataset is analysed as one game.
* The weekly → twice-weekly change (2003) and the price changes (2021–2023) are candidate regime boundaries. They are tested explicitly through era and year homogeneity tests, CUSUM / PELT change-point detection and the rolling-window drift hypothesis.
* **Prediction vs fluctuation.** In a fair draw, every combination is equally likely and draws are independent. Any "pattern" in past results is then a description of noise, and with 42 numbers, 861 pairs and 11,480 triplets, *some* will always look extreme. The research question is whether any deviation is (a) larger than the extremes fair histories produce, (b) significant after correcting for the number of tests, and (c) **persistent enough to improve forecasts of draws the model has never seen**.

## Sources

* [LLDJ – Loto: About / How To Play / Prizes](https://www.lldj.com/en/DrawGames/Loto/About)
* [L'Orient Today – Do Lebanese still dream of lottery jackpot wins? (Jan 2024)](https://today.lorientlejour.com/article/1363602/do-lebanese-still-dream-of-lottery-jackpot-wins.html)
* [USJ dataviz blog – Monday or Thursday? Loto at first sight (Nov 2020)](https://dataviz.etublogs.usj.edu.lb/2020/11/17/monday-or-thursday-loto-at-first-sight/)
