#!/usr/bin/env bash
# End-to-end MVP pipeline for one split (default: train = cutoff 2025-01-01).
# ~3 GB download, ~5 min of processing on a laptop, GraphSAGE adds ~10-30 min on CPU.
set -euo pipefail
SPLIT=${1:-train}
cd "$(dirname "$0")"

python scripts/download.py --meta
if [ "$SPLIT" = "train" ]; then
  python scripts/download.py --trades 2024-07 2025-06
else
  python scripts/download.py --trades 2025-01 2025-12
fi
python scripts/01_prepare.py mvp "$SPLIT"
python scripts/02_eda.py "$SPLIT"
python scripts/03_labels.py "$SPLIT"
python scripts/04_build_graph.py "$SPLIT"
python scripts/05_experiments.py "$SPLIT"
python scripts/06_plot_results.py "$SPLIT"
# iteration 2: first-k inductive setting (09 ~35 min on CPU)
python scripts/07_first_k.py "$SPLIT"
python scripts/08_hetero_graph.py "$SPLIT"
python scripts/09_first_k_experiments.py "$SPLIT"
python scripts/10_first_k_report.py "$SPLIT"
# iteration 3: pooled 5-fold CV scores, precision@100 + Spearman with label z, bootstrap CIs (09 cv ~35 min)
python scripts/09_first_k_experiments.py "$SPLIT" hist,xgb_own,xgb_nbr,sage,hgnn 1,3,5,10,20,50 cv
python scripts/11_rank_report.py "$SPLIT"
