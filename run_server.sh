#!/usr/bin/env bash
# See docs/SERVER.md. Independent jobs, one per allocated GPU (not distributed training).
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}
export KS=${KS:-"1 3 5 10 20 50"}
KS=${KS//,/ }
read -r -a K_VALUES <<< "$KS"
export KS="${K_VALUES[*]}"
export SPLITS=${SPLITS:-"train test"}
export METHODS=${METHODS:-hist,xgb_own,xgb_nbr,sage,hgnn}
export PMGRAPH_TAG=${PMGRAPH_TAG:-server}
export PMGRAPH_RESUME=${PMGRAPH_RESUME:-1}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
STAGES=("$@")
if [ "${#STAGES[@]}" -eq 0 ]; then STAGES=(data prep diag edge oot); fi
if [ "${1:-}" = "--help" ]; then
  echo 'Usage: bash run_server.sh [check|smoke|data|prep|diag|edge|oot|grid|asof ...]'
  echo 'Settings: PYTHON, CUDA_VISIBLE_DEVICES, NGPU, KS (space-separated), SPLITS, METHODS, PMGRAPH_TAG, PMGRAPH_GNN'
  echo 'PMGRAPH_RESUME=1 skips matching completed methods; PMGRAPH_ALLOW_CPU=1 enables CPU debugging.'
  exit 0
fi
for stage in "${STAGES[@]}"; do
  case "$stage" in check|smoke|data|prep|diag|edge|oot|grid|asof) ;; *) echo "Unknown stage: $stage" >&2; exit 2 ;; esac
done
"$PYTHON" scripts/server_check.py --stages "${STAGES[@]}"
mkdir -p outputs/logs
GPU_TOKENS=()
for stage in "${STAGES[@]}"; do
  case "$stage" in
    edge|grid|asof)
      tokens=$("$PYTHON" scripts/server_check.py --gpu-tokens)
      read -r -a GPU_TOKENS <<< "$tokens"
      [ "${#GPU_TOKENS[@]}" -gt 0 ] || { echo 'No worker devices' >&2; exit 1; }
      if [ "${PMGRAPH_ALLOW_CPU:-0}" != 1 ]; then export PMGRAPH_REQUIRE_CUDA=1; fi
      ;;
  esac
done

pool() {
  local jobs=() line pids=() fail=0 g i token
  while IFS= read -r line; do [ -z "$line" ] || jobs+=("$line"); done
  for ((g=0; g<${#GPU_TOKENS[@]} && g<${#jobs[@]}; g++)); do
    token=${GPU_TOKENS[$g]}
    (
      # A worker waits for each command and propagates its failure to the launcher.
      for ((i=g; i<${#jobs[@]}; i+=${#GPU_TOKENS[@]})); do
        echo "Starting worker=$g device=$token: ${jobs[$i]}"
        if [ "$token" = cpu ]; then
          CUDA_VISIBLE_DEVICES='' bash -c "${jobs[$i]}"
        else
          CUDA_VISIBLE_DEVICES=$token bash -c "${jobs[$i]}"
        fi
      done
    ) &
    pids+=("$!")
  done
  for g in "${pids[@]}"; do wait "$g" || fail=1; done
  return "$fail"
}

job() {
  local logfile=$1
  shift
  printf '%q ' "$@"
  printf '> %q 2>&1\n' "$logfile"
}

for stage in "${STAGES[@]}"; do
  echo "== $stage $(date -u +%H:%M:%S)"
  case "$stage" in
    check) ;;
    smoke) "$PYTHON" scripts/server_check.py --smoke ;;
    data)
      "$PYTHON" scripts/download.py --meta
      "$PYTHON" scripts/download.py --trades 2024-07 2025-12 ;;
    prep)
      for S in $SPLITS; do
        "$PYTHON" scripts/01_prepare.py mvp "$S"
        "$PYTHON" scripts/03_labels.py "$S"
        "$PYTHON" scripts/04_build_graph.py "$S"
        "$PYTHON" scripts/07_first_k.py "$S"
        "$PYTHON" scripts/08_hetero_graph.py "$S"
      done ;;
    diag)
      for S in $SPLITS; do
        "$PYTHON" scripts/12_label_diagnostics.py "$S" > "outputs/logs/12_$S.log" 2>&1
        "$PYTHON" scripts/13_target_bakeoff.py "$S" "${KS// /,}" > "outputs/logs/13_$S.log" 2>&1
      done ;;
    edge)
      for S in $SPLITS; do for k in $KS; do
        job "outputs/logs/09_edge_${PMGRAPH_TAG}_${S}_k$k.log" "$PYTHON" scripts/09_first_k_experiments.py "$S" "$METHODS" "$k" cv edge
      done; done | pool
      for S in $SPLITS; do
        "$PYTHON" scripts/14_graph_on_edge.py "$S" 500 "$PMGRAPH_TAG" > "outputs/logs/14_${S}_${PMGRAPH_TAG}.log" 2>&1
      done ;;
    oot)
      "$PYTHON" scripts/15_out_of_time.py train test "${KS// /,}" > outputs/logs/15_train_to_test.log 2>&1 ;;
    grid)
      while read -r tag json; do for k in $KS; do
        job "outputs/logs/09_edge_${PMGRAPH_TAG}_${tag}_k$k.log" env "PMGRAPH_TAG=${PMGRAPH_TAG}_$tag" "PMGRAPH_GNN=$json" \
          "$PYTHON" scripts/09_first_k_experiments.py train "$METHODS" "$k" cv edge
      done; done <<'GRID' | pool
wide     {"hidden": 128, "batch": 2048}
fan      {"fanout": [20, 10]}
widefan  {"hidden": 128, "fanout": [20, 10], "batch": 2048, "epochs": 60, "patience": 10}
slow     {"lr": 0.001, "epochs": 80, "patience": 12}
GRID
      for tag in wide fan widefan slow; do
        "$PYTHON" scripts/14_graph_on_edge.py train 500 "${PMGRAPH_TAG}_$tag" > "outputs/logs/14_train_${PMGRAPH_TAG}_$tag.log" 2>&1
      done ;;
    asof)  # Gate C (iteration 5): 17 builds as-of market nodes, then 09 cv edge with them; Gate C needs KS="1 3 5 10"
      for S in $SPLITS; do
        "$PYTHON" scripts/17_asof_markets.py "$S" "${KS// /,}" 30 > "outputs/logs/17_$S.log" 2>&1
      done
      for S in $SPLITS; do for k in $KS; do
        job "outputs/logs/09_edge_${PMGRAPH_TAG}_asof_${S}_k$k.log" env "PMGRAPH_TAG=${PMGRAPH_TAG}_asof" \
          'PMGRAPH_GNN={"asof_markets": true}' "$PYTHON" scripts/09_first_k_experiments.py "$S" "$METHODS" "$k" cv edge
      done; done | pool
      for S in $SPLITS; do
        "$PYTHON" scripts/14_graph_on_edge.py "$S" 500 "${PMGRAPH_TAG}_asof" > "outputs/logs/14_${S}_${PMGRAPH_TAG}_asof.log" 2>&1
      done ;;
  esac
done
echo "== done $(date -u +%H:%M:%S)"
