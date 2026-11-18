# Lab notes — first iteration (5 Nov 2026)

Split: train cutoff 2025-01-01 · graph = fills Jul–Dec 2024 · labels = bets Jan–Jun 2025.

## Data facts

| | |
|---|---|
| Trades kept (graph / label window) | 30.4M / 31.0M (self-trades ≈ 0; >99.99% binary) |
| Wallets (graph / label window) | 633k / 863k |
| Graph | 633k nodes, 13.6M directed taker→maker pairs (26M undirected nnz) |
| Wallets with ≥30 resolved bets in label window | 37,129 (bets = (wallet, market), |net dir| ≥ 0.1) |
| …of which have a node in the pre-cutoff graph | 17,514 (47%) |
| Mean \|last price − outcome\| per market | 0.05 (reference-token mapping is correct) |
| Mean neighbour maker share | 0.76 → most counterparties are market makers |

## Labels

| type | count | share |
|---|---|---|
| unskilled loser | 20,299 | 54.7% |
| lucky winner | 16,736 | 45.1% |
| skilled winner (BH α=0.1) | 86 | 0.23% |
| unlucky skilled | 8 | 0.02% |

* p < 0.05 for 4.2% of wallets with ≥30 bets (≈ null), but 16.6% of wallets with ≥250 bets.
  Skill exists; it is just invisible at 30 bets. Skilled winners: median 390 bets.
* z-scores are over-dispersed (sd 1.4 vs 1.0): sibling markets in one event (e.g. election
  sub-markets) are not independent → group bets by `event_id`.
* Analytic vs Monte-Carlo p-values: corr 0.91 on a 400-wallet sample (MC with 2k draws is coarse).

## Experiment 1 — transductive wallet split, 5 seeds (test n = 3,503)

| method | profitable AUROC | skill (z>2) AUROC | skill AUPRC (base 0.022) |
|---|---|---|---|
| history-only test | 0.557 ± .010 | 0.652 ± .028 | 0.116 ± .030 |
| XGBoost, own features | **0.690 ± .007** | 0.678 ± .022 | 0.098 ± .026 |
| XGBoost, own + 1-hop nbr aggregates | 0.682 ± .007 | 0.686 ± .014 | 0.102 ± .022 |
| spectral(32) + LR | 0.510 | 0.660 ± .029 | 0.070 ± .014 |
| label propagation | 0.560 ± .012 | 0.657 ± .018 | 0.059 ± .011 |
| GraphSAGE (2-layer, mean, multitask) | 0.651 ± .005 | **0.699 ± .017** | **0.146 ± .036** |

Paired over seeds, skill AUPRC: SAGE − hist = +0.030 ± 0.013, SAGE − XGB(own) = +0.048 ± 0.023 (5/5 wins each).
By pre-cutoff fills (figure `outputs/figures/exp1_train_detection.png`): the SAGE gain is in the
>50-fill bucket (0.235 vs 0.21 hist). For 1–10 fills every method is near the 2% base rate.

**Reading.** (1) Profitability is mostly *style* (top feature: `frac_sureshot`); counterparty
structure alone carries almost nothing (spectral 0.51). (2) For skill, the GNN adds signal on top of
history, but not yet where the proposal claims it matters (new wallets). (3) The proposal's graph claim
(GraphSAGE beats both baselines at k ≤ 10 on new wallets) is **not** met yet.

## Next directions (ordered)

1. **New-wallet setting (core claim).** 53% of labelled wallets have no pre-cutoff node. Build an
   inductive "first-k" snapshot: attach each new wallet to the pre-cutoff graph through its first k
   post-cutoff fills (counterparties + markets); label only with bets *after* those k fills.
2. **Wallet–market bipartite / hetero graph.** Counterparties are 76% market makers, so the
   taker→maker graph is mostly "which MM filled you". Market nodes + co-trading (same market, same
   side, short lag; the proposal's *moved-first* edge) should carry the informed-flow signal → HGT.
3. **Label target.** BH-skilled is 0.25% → near-empty classes. Options for the group: (a) rank /
   regress the z-score; (b) "skill-leaning" z > 2 (current); (c) 12-month label window for more bets.
4. **Event-level bets** to fix over-dispersion; **fees** in profit (`markets.has_fee`, `taker_base_fee`).
5. **Out-of-time test** (cutoff 2025-07-01): `python scripts/download.py --trades 2025-07 2025-12`.
6. Wash-trade community filter, Leiden clusters + ARI/NMI vs types, TGN/DyGFormer on a GPU box.

## Compute notes

2 cores / 7 GB: whole pipeline minus GNN ≈ 5 min; GraphSAGE ≈ 6.5 s/epoch full-batch (sparse matmul),
≈ 6 min/seed. node2vec needs `torch_cluster` (GPU box); spectral embedding is the CPU stand-in.

---

# Iteration 2 — first-k inductive setting (5 Nov 2026)

Question: does the graph help classify **new** wallets from their first k fills (proposal §5 claim)?
Same split as iteration 1 (cutoff 2025-01-01). Machine: 12-core Mac, 48 GB; DuckDB pinned to 2 threads.

## What was run

| step | script | wall | note |
|---|---|---|---|
| setup | `uv venv` + install, `pytest` | 1.2 min | download 6.9 min (3 GB); 01/02/03/04 = 13 s / 7 s / 39 s / 29 s; 04 reproduces iteration 1 (633,082 nodes, 26.1M nnz) |
| 2 | `03_labels.py train all` | 33 s | event-level bet option (`labels.bet_unit`, default `market`) |
| 3 | `07_first_k.py` | 87 s | first-k populations, features, per-k leakage asserts |
| 4 | `08_hetero_graph.py` | 3.8 min | frozen wallet–market HeteroData + moved-first edges |
| 5 | `09_first_k_experiments.py` + `10_first_k_report.py` | 14 min (3 procs × 2 threads; ≈35 min sequential) | 6 k × 5 seeds × 5 methods; no subsampling needed (≈2.5 GB RSS per process) |

Two environment fixes: at 12 DuckDB threads the wallet-feature query spilled 70 GB and failed
(`connect()` now defaults to 2 threads, `PMGRAPH_THREADS` overrides); on macOS xgboost and torch
segfault in one process (two `libomp` copies, either import order), so 09 runs XGBoost in a
torch-free subprocess (`pmgraph/tabular.py`). 05 on macOS: run `xgb_*` and `sage` in separate calls.

## Event-level bets (step 2)

Event bet = mean of the wallet's market bets in one `event_id`; null couples siblings so the wallet's
payoffs are comonotone (largest variance any dependence allows → valid for mutually-exclusive
siblings too), exact analytic variance = what the MC simulates (tests simulate correlated siblings:
market-level z sd 1.71, event-level 1.06).

| | market (default) | event |
|---|---|---|
| labelled wallets (≥ 30 bets) | 37,129 | 27,082 |
| z mean / sd | −0.23 / **1.40** | −0.29 / **1.45** |
| p < 0.05 overall | 4.2% | 3.9% |
| p < 0.05 by bets 30–49 / 50–99 / 100–249 / ≥ 250 | 3.5 / 3.7 / 6.3 / 16.8% | 3.0 / 3.9 / 7.4 / 13.8% |
| z sd by the same buckets | 1.10 / 1.33 / 2.02 / 3.30 | 1.14 / 1.39 / 2.14 / 3.34 |
| skilled winner / lucky winner / unlucky skilled / unskilled loser | 86 / 16,736 / 8 / 20,299 | 54 / 12,031 / 5 / 14,992 |
| analytic vs MC p corr (400 wallets) | 0.91 | 0.92 |

**Reading.** Event grouping does *not* remove the over-dispersion: the median event bet has 1.04
markets, so sibling markets are not the cause. A placebo with the calibration refitted in-sample
on the label window gives the same sd (1.40), so it is not calibration drift either. The excess
grows with activity (sd 1.1 → 3.3), i.e. it lives in active wallets: real heterogeneity in skill
(both tails) and/or within-wallet dependence across events (e.g. one view expressed in many
markets). Labels stay market-level for steps 3–5.

## First-k dataset (step 3)

Observed(w, k) = first k post-cutoff fills (+ pre-cutoff history for existing wallets); label =
skill test / profit on bets strictly **after** the k-th fill in markets **not** in the observed set,
≥ 30 bets (`first_k.label_after: k`; `kmax` = one fixed population, but only 10.7k wallets).
Asserts per k: max observed ts < min labelled ts; no labelled (wallet, market) observed. "New" =
no fill in the Jul–Dec 2024 graph window (earlier activity is not checked).

| k | 1 | 3 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|
| wallets (new) | 34,980 (18,984) | 32,691 (17,981) | 30,732 (17,068) | 26,895 (15,261) | 21,313 (12,322) | 10,712 (5,461) |
| skill z > 2, new / existing | 1.8 / 2.2% | 1.9 / 2.2% | 1.8 / 2.2% | 1.9 / 2.4% | 2.0 / 2.6% | 2.7 / 4.2% |
| new: fills with a pre-cutoff counterparty | 63% | 63% | 64% | 64% | 63% | 66% |
| new: fills in a pre-cutoff market | 39% | 35% | 33% | 29% | 23% | 22% |

## Hetero graph (step 4)

Frozen at the cutoff (asserted max ts 2024-12-31 23:59:59): 633k wallets, 13.3k markets
(category, log volume / fills / takers, age, last price, time to scheduled expiry, outcome if
closed before the cutoff), 7.46M (wallet, held, market) edges (net direction, log $, VWAP, log
hours to close, maker share), 26.1M counterparty nnz, 13.3M moved-first pairs (≤ 10 preceding
entries within 60 min, ≥ 2 co-moves; raw pairs within 60 min: 497M). Scored wallets are
*query nodes* that only receive messages from the frozen graph through their observed edges;
10.5M query moved-first edges (follower edges only if before the k-th fill). Model: 2 ×
HeteroConv of an edge-attribute GraphSAGE conv, numpy neighbour sampler (fan-out 10/5, no
pyg-lib), multitask, early stopping on val AUPRC. `sage` = same model, counterparty edges only.

## Experiment 2 — detection curve (step 5), 5 seeds, 60/20/20 split per k

Figure: `outputs/figures/exp2_train_detection.png`. Tables: `outputs/tables/exp2_train_{summary,paired}.csv`.
Trained on new + existing jointly (`is_new` feature), scored per population. Mean over seeds (sd in the CSV).

Skill (z > 2) AUPRC, **new** wallets (test n ≈ 3,800 at k=1 → 1,100 at k=50)

| k | 1 | 3 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|
| base rate | 0.018 | 0.019 | 0.018 | 0.019 | 0.021 | 0.027 |
| history-only test | 0.018 | 0.021 | 0.019 | 0.024 | 0.022 | 0.053 |
| XGBoost, first-k features | 0.026 | 0.030 | 0.030 | 0.031 | 0.044 | 0.068 |
| XGBoost + first-k nbr aggregates | 0.026 | 0.033 | 0.028 | 0.033 | 0.041 | 0.075 |
| GraphSAGE (counterparty) | 0.025 | 0.029 | 0.035 | 0.033 | 0.050 | **0.088** |
| hetero GraphSAGE | 0.027 | 0.028 | 0.032 | 0.035 | **0.058** | 0.073 |

Paired vs best non-graph baseline (xgb_own at every k), new wallets, skill AUPRC, mean ± sd (wins/5):

| k | 1 | 3 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|
| hetero SAGE | +0.000 ± .007 (2) | −0.002 ± .006 (2) | +0.002 ± .002 (4) | +0.003 ± .010 (4) | +0.014 ± .017 (4) | +0.005 ± .035 (3) |
| SAGE | −0.001 ± .006 (2) | −0.002 ± .007 (2) | +0.005 ± .011 (3) | +0.001 ± .004 (3) | +0.006 ± .012 (3) | +0.020 ± .018 (4) |
| XGB + nbr | +0.000 ± .002 (2) | +0.002 ± .004 (4) | −0.002 ± .005 (1) | +0.002 ± .007 (3) | −0.003 ± .006 (2) | +0.006 ± .016 (2) |

Profitable AUROC, new wallets: XGBoost 0.70 → 0.74 (k = 1 → 10), +nbr identical (±0.005),
GraphSAGE / hetero 0.66–0.72 (−0.02 to −0.04, 0/5 wins at every k ≤ 20), history-only 0.50–0.54.

Existing wallets (pre-cutoff history + first k), skill AUPRC (base 0.022–0.042):

| k | 1 | 3 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|
| history-only test | 0.121 | 0.134 | 0.156 | 0.143 | 0.154 | 0.211 |
| XGBoost, own | 0.121 | 0.146 | 0.140 | 0.165 | 0.136 | 0.174 |
| XGBoost + nbr | 0.124 | 0.135 | 0.148 | 0.166 | 0.164 | 0.214 |
| GraphSAGE | **0.152** | **0.177** | **0.186** | **0.183** | 0.153 | **0.232** |
| hetero GraphSAGE | 0.145 | 0.169 | 0.180 | 0.177 | **0.178** | **0.232** |

GraphSAGE − best baseline: +0.031 (4/5), +0.031 (5/5), +0.030 (5/5), +0.018 (3/5) at k = 1, 3, 5, 10.
Profitable AUROC, existing: XGBoost 0.70–0.75 again best; GNNs 0.65–0.74.

**Claim check (proposal §5): NOT MET.** No graph model beats both baselines on new wallets at
k ≤ 10 (criterion: mean paired diff > 0 and ≥ 4/5 wins vs each); at k ≤ 10 every model is within
~0.015 AUPRC of the 1.9% base rate.

**Reading.**
1. For new wallets, ≤ 10 fills carry almost no skill signal by any route: features, counterparties,
   markets or moved-first. The graph only starts to separate at k = 20–50 (hetero +0.014 at k=20,
   SAGE +0.020 at k=50, 3–4/5 wins: suggestive, not significant at n_pos ≈ 30 per test fold).
2. The iteration-1 result reproduces inductively: with a pre-cutoff node, message passing adds
   ~+0.03 AUPRC over history-only / XGBoost for k ≤ 5. The graph helps where the wallet's *own*
   neighbourhood exists, not through the neighbourhood its first trades attach to.
3. Market and moved-first edges add nothing measurable over counterparties (hetero ≈ SAGE). For
   new wallets, 61–78% of first-k fills are in markets with no pre-cutoff trades, which have no
   frozen market node, so the market channel is thin exactly where it was meant to help.
4. Profitability is style (price level, size, maker share) and is best handled by trees; GNNs
   lose 0.02–0.04 AUROC (they also train on 60% of wallets vs 80% for XGBoost, which needs no val).
5. Caveats: populations shrink with k (per-k label), so compare methods within k, not across k;
   no hyper-parameter search for the GNNs (fixed fan-out 10/5, hidden 64, 40 epochs, patience 6);
   `hours_to_close` uses the market's realised `close_time` (inherited from iteration 1).

## Next steps (ordered)

1. **Rare-class power.** ~70 skilled new wallets per test fold → AUPRC sd ≈ 0.005–0.03. Use the
   z-score as a ranking/regression target (Spearman, precision@k) and/or a 12-month label window
   *(changes the label definition / windows: needs the group's OK)*.
2. **Post-cutoff markets.** Add market nodes for markets created after the cutoff, using only
   information available by each query wallet's k-th fill (category, expiry, cumulative volume),
   so new wallets attach through markets at all.
3. **Out-of-time test** at cutoff 2025-07-01 (`download.py --trades 2025-07 2025-12`; run 01–10 with `test`).
4. Temporal models on the event stream (TGN / DyGFormer) on a GPU box (`gnn.gpu_scale`); GNN
   hyper-parameter search (fan-out, depth, class weighting) before drawing a final null.
5. Over-dispersion: random-effects model of z on activity / category to separate skill
   heterogeneity from within-wallet dependence across events.

---

# Iteration 3 — rare-class power, labels unchanged (12 Nov 2026)

Problem: ~70 skilled new wallets per 20% test fold → per-seed AUPRC sd 0.005–0.03, as large as
the effects being tested. Step 1 of the list above, the part that needs no label change:

* `09 ... cv` (or `first_k.split_mode: cv`): 5-fold CV (fold f test, f+1 val, rest train: same
  60/20/20 sizes as iteration 2) so every wallet is scored once; test scores of every wallet are now
  saved to `data/processed/{exp2,exp3}_preds/`. Random mode (iteration 2) is unchanged and the default.
* `11_rank_report.py`: pools the out-of-fold scores (within-fold percentile ranks) → all positives
  (~5× one fold), and adds **precision@100** and **Spearman(score, label z)**, which uses every
  wallet, not only the 2% positives. 95% wallet-bootstrap CIs; methods compared by paired bootstrap.
  Claim check: graph model's paired CI > 0 vs both baselines on new wallets at every k ≤ 10.
* `pmgraph/rank_eval.py` + `tests/test_rank_eval.py` (null: Spearman ≈ 0, P@100 ≈ base rate,
  paired-CI coverage of a zero difference).
* Synthetic smoke test (4k wallets, 50 positives per population): a scorer with a real signal had
  AUPRC and P@100 differences with CIs covering 0, while the Spearman difference was clearly > 0.
  So Spearman on z should be the most sensitive check of the graph claim.

Caveat: z grows with √(bets) under real skill, so Spearman on z partly rewards predicting activity.
11 therefore also reports **spearman_z_strat** (size-weighted within label bet buckets 30–49 / 50–99 /
≥ 100, with CI) and per-bucket point estimates (`exp3_{split}_buckets.csv`); read skill from the
stratified one (test: an activity-only score has overall Spearman > 0.1, stratified ≈ 0).
Run: `bash run_mvp.sh` (last two lines).

## Results (train split; 16 GB / 10-core Mac; 09 cv ≈ 50 min wall with 2–4 processes, 11 ≈ 7 min)

Re-running 01–08 from a fresh download reproduced the iteration-1/2 inputs up to float noise (label z sd
1.3977; 13,307,234 vs 13,307,223 moved-first pairs). Figure `outputs/figures/exp3_train_detection.png`;
tables `outputs/tables/exp3_train_{pooled,paired,buckets}.csv`.

Pooled skill AUPRC, **new** wallets (n 19.0k → 5.5k, positives ≈ 340 → 150), 95% bootstrap CI:

| k | 1 | 3 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|
| base rate | .018 | .019 | .018 | .019 | .020 | .027 |
| history-only | .018 | .019 | .019 | .022 | .021 | .034 |
| XGBoost own | .023 [.020, .027] | .026 | .028 | .034 | .039 | .049 |
| XGBoost + nbr | .025 | .028 | .026 | .032 | .039 | .048 |
| GraphSAGE | .024 | .027 | .026 | .031 | .040 | .061 |
| hetero GraphSAGE | .025 [.021, .031] | .026 | .028 | .033 | .037 | .048 |

Graph model − best baseline, new wallets, AUPRC (paired CI): hgnn +.002 [−.002, .007], −.000 [−.006, .005],
.000 [−.005, .005], −.001 [−.018, .015] at k = 1, 3, 5, 10; sage and xgb_nbr the same within ±.003.
Existing wallets: hgnn +.038 [.009, ·] at k=10 and +.041 [.010, ·] at k=20, sage +.048 [.012, ·] at k=20;
CIs cover 0 at k ≤ 5. Precision@100: CIs ±0.05 (≈ 2 expected hits at base rate), not informative.

Spearman(score, label z), new wallets: history-only .006 → .18 (k = 1 → 50); every learned model is
**negative** at k ≤ 20 (−.03 to −.14; bet-bucket-stratified the same), positive only at k=50.
Score deciles (xgb_own, k=10, new): P(z > 2) rises 0.5% → 3.5% bottom → top, but mean z *falls*
0.20 → −0.31, sd(z) rises 0.99 → 1.60 and P(z < −2) rises 3.3% → 11%.

**Reading.**
1. The new-wallet null is now tight, not underpowered: at k ≤ 5 the graph adds at most ≈ 0.005–0.007
   AUPRC (upper CI) over XGBoost. The §5 claim fails on every metric (`11` claim check: NOT MET).
2. The learned models beat the base rate (≈ 1.3–1.5×) by finding wallets whose z is **dispersed**,
   both tails, not wallets with high z: the z > 2 target rewards variance. Likely the over-dispersion
   (within-wallet dependence, e.g. one view expressed in many markets) inflates |z| for some trading
   styles, and the classifier learns those styles. So the binary target, not only power, is the
   problem → the label-change half of step 1 (train/evaluate on z or edge with a dependence-robust
   variance, e.g. clustered by day or topic) is now the priority. *Needs the group's OK.*
3. Existing wallets: the GNN gain over history/XGBoost is significant at k = 10–20 (≈ +0.04 AUPRC).
4. Only the pre-cutoff history z ranks wallets by z positively everywhere: skill signal lives in the
   wallet's own resolved history, not in first-k features or the neighbourhood they attach to.

Compute notes: on this Mac hgnn is the bottleneck (≈ 100 s per fold at k=1 with 2 threads, slower when
4 processes share the CPU); with fewer processes raise `gnn.threads`. Jul–Dec 2025 trades are downloaded
(4.5 GB; Oct–Dec alone 3.6 GB), so the test split's label window is ~2–3× the train one.

---

# Iteration 4 — what the skill label measures (14 Nov 2026)

Question from iteration 3: the z > 2 target rewards dispersed z, so what should replace it? No GNN was
trained; labels and all exp1–3 outputs are unchanged. Train split, 16 GB / 10-core Mac.

* `12_label_diagnostics.py` (≈ 10 min) + `pmgraph/label_diag.py`: edge per fill by role / price band,
  split halves of every wallet's label bets, persistence, variance function, saved exp3 scores
  re-scored against realised edge.
* `13_target_bakeoff.py` (≈ 4 min, XGBoost only): candidate targets built from one half of the label
  bets, judged on the other half of held-out wallets. `tabular.py` gained a regression objective and
  sample weights (classifier path unchanged); `firstk.label_bets` was split out of `population`.
* `tests/test_label_diag.py` (8 tests, simulated nulls and planted effects): independent bets → no
  shared variance and flat persistence; planted edge sd recovered; dependence inside a week shows in
  the market split and not in the week / time splits; a centred null gives the blind score nothing.

Notation: x = T − mu0 (excess edge per bet, in cents below), s = sd0, z = x / s. Tables:
`outputs/tables/exp4_train_*.csv`; figure `outputs/figures/exp4_train_label_diag.png`.

## Results

**1. The over-dispersion is persistent wallet edge, not dependence between a wallet's bets.**
Covariance across wallets of the edges of two halves of the label (cents², 95% wallet-bootstrap CI), k=1:

| split of the label bets | new (n ≈ 19.0k) | existing (n ≈ 16.0k) |
|---|---|---|
| random market | 18.8 [17.2, 20.5] | 14.1 [12.7, 15.4] |
| random event | 18.3 [16.8, 20.1] | 14.1 [12.7, 15.6] |
| alternating close weeks | 18.5 [16.9, 20.0] | 12.8 [11.4, 14.3] |
| earlier vs later closes | 16.7 [15.3, 18.3] | 11.5 [10.1, 12.9] |

The halves share an edge sd of ≈ 4 cents per bet however they are cut, and what is left inside a half
is the null variance (ratio 0.95–1.07). Dependence between sibling markets or inside a week is at
most ~10% of the shared part. Half-to-half correlation (time split): 0.25 new / 0.22 existing at k=1,
0.47 / 0.36 at k=50; for wallets with ≥ 100 bets 0.52. Spearman–Brown reliability of the whole label
≈ 0.40 at k=1. So the edge label is a real, moderately reliable wallet trait. This replaces the
"within-wallet dependence" explanation of iterations 2–3. (The Gaussian fit x ~ N(mu, tau² + phi s²)
prefers phi ≈ 2 with tau ≈ 1c, but that model is wrong for these data: see 2. Do not read its phi as
dependence.)

**2. The persistence is one-sided: losing repeats, winning barely does.** Wallets sorted by the z of
the earlier half of their label bets; outcome on the later half (k=1, new / existing):

| decile of earlier z | mean later edge | P(later z < −2) | P(later z > 2) |
|---|---|---|---|
| bottom (z ≈ −2.5) | −7.6c / −6.3c | 30% / 28% | 0.7% / 1.1% |
| 2nd | −2.9c / −2.4c | 8% / 8% | 1.1% / 0.8% |
| 5th–9th | −1.6c … −0.1c | 3–5% | 0.6–1.8% |
| top (z ≈ +1.6) | −0.2c / −0.4c | 3.9% / 4.0% | 3.9% / 6.5% |
| all wallets | −1.4c / −1.1c on the whole label (mean z −0.25 / −0.22) | 7.3% / 6.9% | 1.4% / 1.8% |

The null is not centred (average labelled wallet: −1.3c per bet, and z drifts down with √bets), the
left tail is 4–5× the right one, and nearly all the shared variance of result 1 sits in the losers.
A small group of repeat winners exists (top decile: about 3× the base rate of z > 2) but its mean later
edge is not above zero.

**3. Role and price.** Per fill, takers lose 0.23c (SE 0.06, market clusters) and makers gain it:
the role gap is small (weather −0.94c, sports −0.32c, crypto 0.0c for takers). The wallet-level gap
by maker share seen in the first-k features is therefore who the wallets are, not the spread. By buy
price the label window has a band effect the pre-cutoff calibration misses: buying at 0.3–0.4 made
+4.9c (SE 1.5), buying at 0.6–0.7 lost 5.6c (SE 1.3), both roles alike. Wallets whose first fill is
above 0.99 (20% of new wallets at k=1) later earn +0.6c per bet (raw, uncalibrated: +0.5c) against
−1.2 to −2.8c for first fills below 0.97: a small steady premium for holding near-certain outcomes.

**4. Models trained on z > 2 rank new wallets backwards.** Saved exp3 scores against realised
edge (inverse-variance weighted): top minus bottom score decile, cents per bet, new wallets:

| k | 1 | 3 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|
| history-only | +0.1 | +0.5 | +0.5 | +0.2 | +1.0 | +2.9 |
| XGBoost own | −1.2 | −1.5 | −1.4 | −1.3 | −0.8 | +0.5 |
| GraphSAGE | −1.3 | −1.4 | −1.4 | −1.5 | −1.1 | +1.0 |
| hetero GraphSAGE | −1.1 | −1.7 | −1.5 | −1.6 | −1.0 | +0.6 |

All CIs of the learned models exclude 0 at k ≤ 20 (e.g. hgnn k=10: −1.6 [−1.9, −1.4]). Their weighted
correlation with later-half edge (−0.08 to −0.11) is what a score that knows only the wallet's noise
scale gets (`blind`: −0.12). Existing wallets: history-only +4.2 to +7.5c; XGBoost +0.7 to +1.2c and
the GNNs +0.2 to +0.8c at k ≤ 20. So the iteration-3 GNN gain for existing wallets is a gain in
right-tail precision only (later-half z > 2 AUPRC at k=10: hgnn 0.119, history 0.108, XGBoost 0.087,
CIs overlap); on ranking by edge the GNNs trained on z > 2 are below XGBoost and far below history.
Reference for the right tail: the wallet's own earlier-half z reaches AUPRC 0.08 / 0.11 / 0.22 (new,
k = 1 / 10 / 50, base 0.014 / 0.014 / 0.020); first-k models reach 0.02–0.045.

**5. Target bake-off (XGBoost on first-k features; trained on even-close-week bets, judged on
odd-week realised edge of held-out wallets; 5 CV folds of 09).** Weighted correlation with held-out edge:

| target | new k=1 | new k=10 | new k=50 | existing k=1 | existing k=10 | existing k=50 |
|---|---|---|---|---|---|---|
| z > 2, whole label (iter. 1–3) | −0.11 | −0.13 | +0.02 | +0.09 | +0.08 | +0.21 |
| z > 2, training half | −0.15 | −0.15 | −0.01 | +0.06 | +0.05 | +0.15 |
| regression on z | +0.21 | +0.28 | +0.30 | +0.27 | +0.32 | +0.39 |
| regression on edge, weights 1/(tau² + phi s²) | +0.22 | +0.28 | +0.32 | +0.28 | +0.34 | +0.42 |
| regression on empirical-Bayes edge | +0.21 | +0.26 | +0.32 | +0.27 | +0.30 | +0.41 |
| regression on centred z | +0.22 | +0.28 | +0.33 | +0.28 | +0.33 | +0.42 |
| history-only (no training) | +0.01 | +0.07 | +0.17 | +0.19 | +0.23 | +0.31 |

CI half-widths ≈ 0.015–0.03. Top minus bottom decile for the weighted edge regression: +3.3 / +4.5 /
+7.7c (new) and +4.9 / +6.1 / +9.7c (existing), against −1.7 / −1.6 / +0.3c and +0.9 / +0.9 / +2.6c
for the z > 2 classifier. The four regression targets are indistinguishable; all beat the classifier
(paired difference, new, k=1: +0.36 [0.34, 0.39]) and sit above history-only with CIs that do not overlap.
Without sure-shot wallets (≥ half of the observed fills above 0.90; 22–28% of new wallets at
k ≤ 10): new +0.16 / +0.22 / +0.31, existing +0.29 / +0.35 / +0.41, so the signal is not only that
one style. For the rare class nothing changes: held-out-half z > 2 AUPRC of every target stays at
0.013–0.038 for new wallets; the classifiers are, if anything, slightly ahead there.

## Reading

1. The label problem is the threshold, not the statistic. The edge behind z is a persistent trait;
   z > 2 keeps its noisiest 2% and throws away the well-populated, strongly persistent losing side.
2. "≤ 10 fills carry almost no skill signal" (iteration 2) was the target. Trained on edge, XGBoost
   ranks **new wallets from one fill** with correlation 0.22 to held-out edge. What it finds is how a
   wallet trades (price level, category, size) and hence who will lose, not who is informed.
3. The proposal's four types should be read through this: "unskilled loser" is detectable early and
   persistent; "skilled winner" persists only for a small group and no first-k model finds it.
4. Up to here the graph claim had only been tested on z > 2; the section below tests it on edge.

Recommendation for the group (label change, needs agreement): train on a continuous target (centred z
or precision-weighted edge), report the held-out-half edge metrics (weighted correlation, top minus
bottom decile) as the main result and keep z > 2 AUPRC as the rare-class metric. Decide whether the
sure-shot premium counts as skill; report with and without those wallets.

Caveats. (a) Everything is inside one six-month window. The alternating-week split keeps test-week
outcomes out of training, but a style that happened to pay in this window (the 0.3–0.4 / 0.6–0.7
band effect) would look persistent here: the out-of-time test decides. (b) CIs are wallet bootstraps
and ignore that wallets share markets, so they are too narrow. (c) Same XGBoost settings for every
target, no tuning. (d) A half needs ≥ 10 bets, so targets come from 15–25 bets for most wallets.
(e) The plan's simulated "noise ceiling" was dropped: the Gaussian model it needs does not reproduce
the 2% base rate; the earlier-half z and the blind score are the empirical replacements.

## Graph claim on the edge target (same day)

`09 ... cv edge` + `14_graph_on_edge.py`: all five methods fit on the centred z of the even-week half
of the label bets (GNNs: regression head with a Huber loss in `train_hetero`, early stopping on the
validation correlation; XGBoost: regression), judged on the odd-week realised edge of the held-out
wallets. 5 CV folds, pooled, 500 wallet bootstraps. Wall ≈ 50 min (2–3 processes; hgnn 9–25 min per k).
Tables `exp4_train_graph{,_paired}.csv`, figure `outputs/figures/exp4_train_graph.png`.

Weighted correlation with held-out edge, **new** wallets (CI half-width ≈ 0.015):

| k | 1 | 3 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|
| history-only | .009 | .069 | .069 | .070 | .108 | .172 |
| XGBoost own | .216 | .242 | .255 | .283 | .292 | .328 |
| XGBoost + nbr | .220 | .239 | .257 | .288 | .298 | .339 |
| GraphSAGE | .202 | .239 | .261 | .282 | .291 | .340 |
| hetero GraphSAGE | .211 | .248 | .265 | .279 | .295 | .342 |

Graph model − XGBoost own, new wallets (paired 95% CI): hgnn −.005 [−.012, .003], +.006 [−.003, .015],
+.010 [.001, .019], −.004 [−.015, .007] at k = 1, 3, 5, 10; sage −.014 [−.022, −.006], −.004, +.006,
−.001; xgb_nbr +.004, −.003, +.002, +.005 (CIs cover 0 unless shown). Existing wallets: XGBoost own
.278 → .419 (k = 1 → 50), every graph model within −.016 … +.012 of it; the only CIs excluding 0 are
hgnn −.016 at k=5 and xgb_nbr +.012 at k=1. Top minus bottom decile: 2.9–4.9c for every trained
model on new wallets at k ≤ 10 (history-only ≤ 0.6c).

**Claim check (proposal §5, edge target): NOT MET.** One paired CI of twelve is above 0 for new
wallets at k ≤ 10 (hgnn, k=5, +.010) and one is below (sage, k=1).

**Reading.**
1. This is the first test of the graph claim on a target the models can learn (0.2–0.3 against
   0.01–0.07 for history-only), and the null is tight: for new wallets at k ≤ 10 a graph model adds
   at most ≈ 0.02 correlation over XGBoost on the wallet's own first-k features.
2. The GNN advantage for existing wallets (iterations 1–3) does not survive the change of target:
   trees and GNNs are level, with the GNNs slightly behind.
3. Counterparty, market and moved-first neighbourhoods carry no information about a wallet's edge
   beyond its own trading style, at least as mean-aggregated 2-hop GraphSAGE inputs, and with
   61–78% of new wallets' first fills in markets that have no node.
4. Caveats as above (one window, wallet bootstrap, untuned GNNs: fan-out 10/5, hidden 64).

## Next steps (ordered)

1. **Decide the label with the group** using results 1–5 and the graph table above.
2. **Out-of-time test** (fit on train, score the 2025-07-01 split): the main threat to result 5.
3. **Give the graph its best chance before calling the null final:** nodes for post-cutoff markets
   (information as of each wallet's k-th fill), then one GNN hyper-parameter pass on the edge target.
4. Split the edge into a style part (price band × category × period mean) and a residual, and repeat
   12–14 on the residual: is anything beyond style predictable early, and does the graph help there?
5. Markout (price move shortly after a fill) as a dense auxiliary target for the informed tail.
6. Proposal deliverables still missing: Leiden + ARI/NMI, fees, wash-trade filter, macro-F1.

## Server handoff (15 Nov 2026) — code written, not run

A teammate will run the heavy jobs on a server with 8 x RTX 4090. Added for that, tested on CPU only:
* `train_hetero` moves the model and batches to CUDA when available (checked on a non-CPU device
  locally; sampling stays on the CPU, so the GPU pays off through larger `batch` / `hidden`, and
  through 8 processes at once, not at the default settings).
* `09`: GNN settings from env `PMGRAPH_GNN` (JSON), tagged outputs with `PMGRAPH_TAG`; `14` takes the tag.
* `15_out_of_time.py`: XGBoost fit on one split, scored on a later one (edge regression, z > 2
  classifier, history-only), 5 seeds, judged on the later split's realised edge. Only an in-sample
  plumbing check has been run; **there are no out-of-time numbers yet.**
* `run_server.sh`: stages `data prep diag edge oot` (+ optional `grid`), one GPU per process.

What the server run answers: (1) does the edge signal of result 5 survive out of time; (2) does the
tie between graph models and XGBoost repeat on the second period (test split, cv) and under larger /
slower GNN settings. Report every grid setting: the CV folds that would rank them are the test folds.

## Repo refresh (16 Nov 2026) — no new experiment

Nothing was trained and no number changed. Added for readers and for the group:
* `scripts/16_overview_figures.py` + `pmgraph/plotstyle.py`: two overview figures drawn from the committed
  tables only (`overview_train_setup.png`: the task and how rare skill is; `overview_train_results.png`:
  persistence, target bake-off, graph minus XGBoost). Sources: `prepare_stats.json`, `labels_train.json`,
  `eda_train.json`, `exp4_train_{persistence,targets,graph,graph_paired}.csv`.
* `docs/RESULTS.md` (every figure in order, how to read it, what it changed) and `docs/ONBOARDING.md`
  (setup check without data, repo map, rules, open tasks). README links both and shows the two figures.

## Server interface audit (16 Nov 2026) — no new full-data result

Reviewed the fresh-clone path for the eight-4090 server and added `docs/SERVER.md`.
Changes preserve the task, graph architecture and historical result tables:

* `run_server.sh`: dependency/data/GPU preflight before work; `check` and data-free `smoke` stages;
  respects allocated CUDA ids/UUIDs and validates NGPU; no silent CPU fallback for GPU stages;
  configurable interpreter, k values, splits and methods; tagged graph outputs (`server` by default).
* `server_check.py`: actual CUDA operation/backward check, tiny homogeneous/heterogeneous GNN
  classification/regression fits on selected devices, checkpoint reload and separate XGBoost worker;
  package/hardware/config manifests, Parquet/DuckDB storage check and stage prerequisite checks.
* `train_hetero`: saves best-validation model weights and fit metadata (parameters, epochs, time,
  process peak RSS, peak allocated/reserved CUDA bytes). `09` saves preprocessing scalers and completed
  method manifests; PMGRAPH_RESUME skips matching completed k x method jobs. Incomplete methods restart
  their five folds. Checkpoints are not mid-epoch optimizer/RNG snapshots.
* Inference batches now follow the training batch unless `eval_batch` is set, replacing the hardcoded
  4096; this changes neighbour-sampling sequences, so do not expect old scores to reproduce bit-for-bit.
  Handles empty graph relations and fully isolated query batches, constant validation predictions,
  degenerate regression targets and nonfinite losses explicitly. Data preparation rejects missing
  daily trade partitions instead of silently shortening a requested window.
* `14` refuses missing k/method/fold/wallet predictions, and does not count a pilot subset as the full
  graph claim. Existing baseline result CSVs/figures were not regenerated.

Validation: original statistical/sampler tests, new routing/failure/resume/empty-neighbour tests,
real `09 cv edge` -> saved weights/predictions -> resume -> `14` integration on a tiny prepared
fixture. CPU smoke (both GNNs, binary/regression, checkpoint reload, XGBoost worker) passes. This Mac
needed its existing bundled OpenMP library on DYLD_LIBRARY_PATH for the XGBoost subprocess; no
system library was installed. Linux dependencies/native libraries and actual CUDA kernels are still
checked on the server by `check`/`smoke`. No remote GPU or full downloaded-data run was performed.

Scientific caveats found during review, not silently redefined here: `hours_to_close` derives from
realised close_time; first-k label construction bounds fill times but does not require every label
market to resolve before label_end. Hence script 15's statement that all training labels are known
before the scoring period is not enforced. Resolve as-of timing/label availability before a strict
out-of-time claim. The frozen graph's poor new-market coverage and unequal GNN/tree fit-set sizes
also remain. Hidden 64 / two layers is a reasonable baseline; use the declared width-128/fanout grid
and a self-only MLP control before attributing a null result to graph information rather than capacity.

Final local validation: `python -m pytest -q tests` — 30 passed; Python error/undefined-name lint,
`bash -n run_server.sh`, and `git diff --check` pass. The isolated-query regression test also covers
an entire heterogeneous batch with no usable wallet/market neighbours. No server/GPU result is implied.

## First-k label reconstruction fix (17 Nov 2026)

Server logs: train diagnostics (12) and the train target bake-off (13) completed; test diagnostics
failed at k=1 when reconstructed T did not match fk_pop. The main GNN stage was not reached.
No full server data are available locally, so the log alone does not identify the discrepant bets.

Reproduced a concrete numerical cause locally with the actual bet/first-k functions: long quantities
0.1 + 0.2 + 0.8 and short quantities 0.3 + 0.6 have net/gross direction 0.1 mathematically. Ordinary
floating-point SUM with two fill orders kept this bet in one calculation and dropped it in another.
A first-k fixture then contained 31 versus 30 label bets and different T values, matching the failed
consistency check. Both new regression tests failed before the fix.

`bets_from_sides` now uses compensated FSUM aggregates and a 1e-12 comparison tolerance on the bounded
net-direction ratio. This preserves the intended inclusive 0.1 eligibility rule at machine precision;
a genuinely below-threshold bet (tested at ~5e-11 below it) remains excluded. This affects all callers,
so regenerate labels/features/first-k/graph artifacts together from existing raw data, rather than
reusing old populations with the new aggregation. Historical committed results were not regenerated.

The diagnostic check keeps its original T comparison tolerance (atol 1e-9, rtol 1e-5), also verifies
bet counts, rejects missing/nonfinite values, and writes per-wallet saved/rebuilt values to
outputs/tables/exp4_<split>_label_mismatch_k<k>.csv on failure. The message includes mismatch counts
and maximum absolute delta. It does not replace saved labels or skip an inconsistency.

Validation: `python -m pytest -q tests` — 38 passed; error/undefined-name lint and diff checks pass.
New tests cover all 120 fill permutations at the threshold, true below-threshold exclusion, first-k
label reconstruction, wallet reordering, roundoff, count/value/missing-wallet mismatches and NaN/inf.
Existing PyTorch/Pandas dependency deprecation warnings remain. See docs/SERVER.md for recovery.
The full server run remains the confirmation that this was the original failure's cause; if it still
fails, retain the assertion and inspect the new mismatch report for additional data/artifact issues.

## Experiment 6 interface — GNN-only tuning (18 Nov 2026; no full-data run)

Added scripts/17_gnn_only.py and docs/GNN_ONLY.md for training/evaluating only sage/hgnn on the
already prepared graphs. It schedules model x k x period jobs over allocated GPUs, has CLI/YAML
hyperparameters, saves per-trial manifests/logs/checkpoints, resumes matching completed jobs, and
produces its own validation/test metrics and paired hgnn-minus-sage intervals without XGBoost.

For tuning, the new `holdout` mode of 09 uses one fixed, stratified 60/20/20 wallet partition across
models/settings/initialization seeds. Half-A target variance normalization fits training wallets
only. Early stopping uses validation half-A correlation; configuration selection can use the
validation-wallet half-B metrics. --stage train withholds the test report; --stage test evaluates
saved test-wallet predictions without fitting. --stage train-test runs both. This is a new evaluation
protocol (exp6), not the previous five-fold experiment or an earlier-period-to-later-period transfer.
Old exp4 result tables/checkpoints remain separate. Any later baseline claim needs matching splits.

Stability changes: missing market features are imputed to zero before scaling, matching wallet
missing-input handling. A NaN market feature reproduces an invalid heterogeneous input path in a
test fixture; the fixed path trains and evaluates both models successfully. Other nonfinite node/
edge inputs fail with their source/index before sampling; gradients are checked before optimizer
updates; weight decay and gradient clipping are configurable (new CLI default clip 1). Logs now name
the actual validation metric and save epoch histories. The old k=20/50 epoch-3 failures cannot be
attributed conclusively without the server data; the new checks preserve failures rather than hide them.

Artifacts: outputs/logs/gnn/<tag>/ for logs/manifests/readable reports,
outputs/tables/exp6_<tag>_{val,test}_{metrics,paired}.csv for numerical results,
data/processed/{exp6_preds_<tag>,checkpoints/exp6_<tag>} for predictions/weights/scalers/partitions.
Changing settings/code/prepared inputs/package versions under an existing trial name requires a
new tag. Data preparation is not run automatically and no baseline model is fit by the new script.

Validation: 45 tests pass, including real CPU training of both models through the CLI, separate
validation/test evaluation, resume, held-out target normalization, fixed/disjoint partitions,
missing-market inputs, nonfinite-gradient detection, job failure propagation and tag isolation.
The prior CV/report integration test still passes. Error/undefined-name lint and diff checks pass.
CUDA scheduling reuses allocation ids/UUIDs and checks kernels, but no remote GPU/full-data tuning
run has been performed locally. Existing dependency deprecation warnings remain.
