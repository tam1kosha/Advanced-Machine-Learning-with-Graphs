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
