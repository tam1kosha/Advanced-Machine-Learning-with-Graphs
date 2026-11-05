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
