Read CLAUDE.md, NOTES.md and proposal.pdf first. We are on iteration 2 of the "Skilled or Lucky?" project.
Iteration 1 (NOTES.md) showed GraphSAGE helps on the rare skill class only for active wallets, and that 53% of
labelled wallets have no node in the pre-cutoff graph. This iteration tests the proposal's core claim:
does the graph help classify NEW wallets from their first k trades?

Work in this order, running tests and committing after each step (small commits, clear messages):

1. Setup check. Create the venv, run `pytest`, and if data/ is empty run `bash run_mvp.sh train` up to
   `04_build_graph.py` (skip 05 — GraphSAGE is slow on CPU). Report wall-clock times.

2. Event-level bets. In `labels.py`, add an option to aggregate bets at `event_id` level (sibling markets of one
   event share information, which over-disperses the z-scores: sd 1.4 in iteration 1). The trades table has
   `event_id`; carry it through `prep.py`. Keep market-level as the default, make it a config flag, add a test,
   re-run 03_labels and report: z sd, share with p<0.05 per bet-count bucket, type counts, for both variants.

3. First-k inductive dataset (new script `07_first_k.py`). For each wallet whose FIRST fill is after the cutoff
   and that has >= 30 resolved bets in the label window:
   - its first k fills (k in configs eval.k_buckets) define its observed history; attach it to the frozen
     pre-cutoff graph through those fills' counterparties (and, in step 4, markets);
   - features: the same wallet features computed on the k fills only;
   - label: the skill test and profit on bets in markets NOT touched by the first k fills, and only on fills
     after the k-th fill (no overlap between what we observe and what we label).
   Write it so the same code can also be used for pre-existing wallets (pre-cutoff history + first k).
   Leakage checks as asserts: max observed ts < min labelled ts per wallet; no labelled market in observed set.

4. Wallet–market heterogeneous graph. Counterparties are ~76% market makers, so add market nodes (category,
   log volume, time-to-close features) with wallet–market edges (net direction, size, price, hours to close),
   plus the proposal's moved-first edge (u -> v if v takes the same side in the same market within
   `graph.moved_first_window_min` after u), all built strictly before the cutoff. Use PyG HeteroData;
   implement hetero GraphSAGE (to_hetero or HeteroConv) with sparse/neighbour-sampled training that fits in
   7 GB RAM on CPU (use NeighborLoader if pyg-lib is available, else restrict to the 2-hop neighbourhood of
   labelled nodes). New wallets are inserted at inference time using only their first-k edges.

5. Evaluation. For k in {1, 3, 5, 10, 20, 50}: history-only test on the first k fills, XGBoost on first-k
   features, XGBoost + first-k neighbour aggregates, homogeneous GraphSAGE, hetero GraphSAGE. Targets: skill
   (z > 2) AUPRC and profitable AUROC; 5 seeds; mean ± sd and paired seed-wise differences vs the best
   non-graph baseline. Produce the detection-curve figure (metric vs k) separately for new and existing
   wallets. The graph claim holds only if a graph model beats both baselines at k <= 10 on new wallets.

6. Update NOTES.md with a dated "Iteration 2" section (what was run, tables, reading, next steps) and README if
   commands changed. Do not overwrite iteration-1 results.

Constraints: DuckDB for all fill-level work via `pmgraph.prep.connect()`; never pull address strings for 10M+
rows into pandas; no lifetime user tables as features; keep each script runnable on its own. If something
takes
longer than ~20 min on CPU, subsample (state the fraction) and leave a GPU-scale flag in the config.
Stop and ask me before changing the label definition beyond step 2, or the train/test windows.
