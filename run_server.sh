#!/usr/bin/env bash
# GPU-server run: everything that is too heavy for a laptop (written for 8 x RTX 4090, 24 GB each).
# Assumes the environment of README.md is installed and active, on Linux with CUDA.
#   bash run_server.sh                  # data prep diag edge oot
#   bash run_server.sh edge oot         # only these stages
#   bash run_server.sh grid             # optional: GNN settings on the edge target (24 jobs)
# Stages
#   data  download meta + trades 2024-07 .. 2025-12 (both splits, ~8 GB)
#   prep  01, 03, 04, 07, 08 for train and test (DuckDB, CPU; 2 threads by default, see CLAUDE.md)
#   diag  12 + 13 for train and test (CPU, XGBoost): label diagnostics, split halves, target bake-off
#   edge  09 cv edge for train and test, one k per GPU, then 14: graph models vs XGBoost on the edge target
#   oot   15: fit on train, score test (XGBoost): the out-of-time test
#   grid  09 cv edge on train under 4 GNN settings, tagged; 14 per tag
#   asof  Gate C: 17 (as-of market nodes) + 09 cv edge with asof_markets, tag asof; 14; train and test
# GPU jobs: one process per GPU (CUDA_VISIBLE_DEVICES), each loads the frozen graph (~3 GB RAM for train,
# more for test). NGPU overrides the detected GPU count. Logs: outputs/logs/. Small outputs (outputs/tables,
# outputs/figures) are what to commit back; data/ stays on the server.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs/logs
NGPU=${NGPU:-$( (nvidia-smi -L 2>/dev/null || true) | wc -l | tr -d ' ')}
[ "$NGPU" -gt 0 ] || NGPU=1
KS="1 3 5 10 20 50"
METHODS=hist,xgb_own,xgb_nbr,sage,hgnn
STAGES=${*:-data prep diag edge oot}

pool() {  # stdin: one command per line. Worker g runs commands g, g + NGPU, ... in order on GPU g.
  local jobs=() line pids=() fail=0 g
  while IFS= read -r line; do [ -n "$line" ] && jobs+=("$line"); done
  for g in $(seq 0 $((NGPU - 1))); do
    ( i=$g; while [ "$i" -lt "${#jobs[@]}" ]; do
        CUDA_VISIBLE_DEVICES=$g bash -c "${jobs[$i]}" || { echo "FAILED on GPU $g: ${jobs[$i]}" >&2; exit 1; }
        i=$((i + NGPU)); done ) &
    pids+=($!)
  done
  for g in "${pids[@]}"; do wait "$g" || fail=1; done
  return $fail
}

for stage in $STAGES; do
  echo "== $stage $(date -u +%H:%M:%S)"
  case $stage in
    data)
      python scripts/download.py --meta
      python scripts/download.py --trades 2024-07 2025-12 ;;
    prep)
      python scripts/01_prepare.py mvp train test
      for S in train test; do
        python scripts/03_labels.py "$S"
        python scripts/04_build_graph.py "$S"
        python scripts/07_first_k.py "$S"
        python scripts/08_hetero_graph.py "$S"
      done ;;
    diag)
      for S in train test; do
        python scripts/12_label_diagnostics.py "$S" > "outputs/logs/12_$S.log" 2>&1
        python scripts/13_target_bakeoff.py "$S" 1,3,5,10,20,50 > "outputs/logs/13_$S.log" 2>&1
      done ;;
    edge)
      for S in train test; do for k in $KS; do
        echo "python scripts/09_first_k_experiments.py $S $METHODS $k cv edge > outputs/logs/09_edge_${S}_k$k.log 2>&1"
      done; done | pool
      for S in train test; do python scripts/14_graph_on_edge.py "$S" > "outputs/logs/14_$S.log" 2>&1; done ;;
    oot)
      python scripts/15_out_of_time.py train test > outputs/logs/15_train_to_test.log 2>&1 ;;
    grid)  # report every setting, not the best one: the folds that rank the settings are the test folds
      while read -r tag json; do for k in $KS; do
        echo "PMGRAPH_TAG=$tag PMGRAPH_GNN='$json' python scripts/09_first_k_experiments.py train hist,xgb_own,sage,hgnn $k cv edge > outputs/logs/09_edge_${tag}_k$k.log 2>&1"
      done; done <<'GRID' | pool
wide     {"hidden": 128, "batch": 2048}
fan      {"fanout": [20, 10]}
widefan  {"hidden": 128, "fanout": [20, 10], "batch": 2048, "epochs": 60, "patience": 10}
slow     {"lr": 0.001, "epochs": 80, "patience": 12}
GRID
      for tag in wide fan widefan slow; do python scripts/14_graph_on_edge.py train 500 "$tag" > "outputs/logs/14_train_$tag.log" 2>&1; done ;;
    asof)  # Gate C (iteration 5); train was run on a laptop, this repeats it on the second period too
      for S in train test; do python scripts/17_asof_markets.py "$S" 1,3,5,10 30 > "outputs/logs/17_$S.log" 2>&1; done
      for S in train test; do for k in 1 3 5 10; do
        echo "PMGRAPH_TAG=asof PMGRAPH_GNN='{\"asof_markets\": true}' python scripts/09_first_k_experiments.py $S $METHODS $k cv edge > outputs/logs/09_edge_asof_${S}_k$k.log 2>&1"
      done; done | pool
      for S in train test; do python scripts/14_graph_on_edge.py "$S" 500 asof > "outputs/logs/14_${S}_asof.log" 2>&1; done ;;
    *) echo "unknown stage: $stage" >&2; exit 2 ;;
  esac
done
echo "== done $(date -u +%H:%M:%S)"
