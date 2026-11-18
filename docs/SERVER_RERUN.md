# Server rerun after the NaN fix (18 Nov 2026)

A copy-paste runbook for the 8 x RTX 4090 server. Run the steps in order. Each step lists what to check
before going on. If a check fails, stop and send the log back. Do not patch the code on the server.
Background: `docs/SERVER.md` (setup), `docs/GNN_ONLY.md` (script 17), `NOTES.md` (18 Nov entry).

## What went wrong and what changed

Heterogeneous GraphSAGE (`hgnn`) crashed on the later period (`test`) at k = 20 and 50 in every run:
* the old runs: `nonfinite loss at epoch 3` (`outputs/logs/09_edge_server_test_k20.log`, `..._k50.log`);
* trial001 and trial002: `query.rev_held.edge_attr: nonfinite input at index (..., 3)`.

Both have one cause. Some markets have no close time. For query wallets, the code computed
"log hours to close" as NaN for those markets, while the frozen graph used 0. The fix is one line in
`src/pmgraph/hetero.py` (`held_edges_pd`: `fillna(0)` before `clip`). It also adds a test. Lowering the
learning rate could never fix this, because the problem is in the input data, not in training.

Consequences:
* **Use new tags.** The code changed, so old tags are rejected on purpose. Do not delete trial001/trial002.
* **No data re-prep is needed.** The fix runs at training time. Do not rerun `data`, `prep` or `diag`.

## Step 0: get the fix and check it (2 minutes)

```bash
cd ~/projects/Advanced-Machine-Learning-with-Graphs
git fetch origin
git checkout main && git pull origin main      # after PR #7 is merged
# before the merge, use the PR branch instead:
# git checkout claude/gpu-processing-results-7c8279 && git pull
source .venv/bin/activate
grep -n "fillna(0).clip" src/pmgraph/hetero.py
python -m pytest -q tests/test_hetero.py
```

Check: `grep` prints one line, and pytest ends with `passed` and no `failed`.

## Step 1: pilot only the job that used to crash (≈ 10 minutes, 1 GPU)

```bash
tmux new -s rerun          # detach: Ctrl+B then D; reattach: tmux attach -t rerun
python scripts/17_gnn_only.py --tag pilot_nanfix --stage train \
  --gpus 0 --models hgnn --splits test --ks 20 --seeds 0 \
  --hidden 128 --lr 0.001 --dropout 0.2 --weight-decay 0.0001 --grad-clip 1.0 \
  --fanout 10 5 --batch 1024 --eval-batch 1024 --epochs 80 --patience 10 \
  2>&1 | tee -a outputs/logs/gnn_pilot_nanfix.log
```

Check: `outputs/logs/gnn/pilot_nanfix/test_k20_hgnn.log` contains `completed:` and `edge scores saved`.
It must not contain `nonfinite`. If it still fails, stop here and send that log.

## Step 2: full GNN trial with the trial001 settings (≈ 20–40 minutes, 8 GPUs)

These are the same settings as trial001, now with every job able to finish. This gives the first
complete val report.

```bash
python scripts/17_gnn_only.py --tag trial003 --stage train \
  --gpus 0 1 2 3 4 5 6 7 \
  --hidden 128 --lr 0.001 --dropout 0.2 --weight-decay 0.0001 --grad-clip 1.0 \
  --fanout 10 5 --batch 1024 --eval-batch 1024 --epochs 80 --patience 10 \
  2>&1 | tee -a outputs/logs/gnn_trial003.log
```

Check:
* the launcher log does not end with `Some jobs failed`;
* `outputs/logs/gnn/trial003/val_report.log` and `outputs/tables/exp6_trial003_val_metrics.csv` exist.

Do **not** run `--stage test` yet (see step 5).

## Step 3: later-period graph vs XGBoost, all k (≈ 30–60 minutes)

This is the main scientific result: do graph models beat XGBoost on new wallets? The later period
only has k = 1–10 so far, because hgnn crashed at k = 20/50. This step reruns all five methods with
the fixed code.

```bash
SPLITS=test PMGRAPH_TAG=server2 NGPU=8 bash run_server.sh edge
```

Check:
* `outputs/logs/14_test_server2.log` exists and has tables with columns `1 3 5 10 20 50`;
* no `outputs/logs/09_edge_server2_test_k*.log` contains `Traceback`.

You may run steps 2 and 3 at the same time on separate cards, for example `--gpus 0 1 2 3` in
step 2 and `CUDA_VISIBLE_DEVICES=4,5,6,7 NGPU=4` in step 3. Do this only if `free -g` shows at
least 60 GB of free RAM. Each process uses up to about 6 GB of host RAM.

## Step 4: out-of-time test (CPU, XGBoost only, ≈ 10–30 minutes)

This has never been run. It fits on the earlier period and scores the later one.

```bash
bash run_server.sh oot
```

Check: `outputs/logs/15_train_to_test.log` ends without a `Traceback`.

## Step 5: optional capacity trials, because the GPUs are almost idle

Your observation is right: the 2.4 GB came from a small model, not from a hardware limit.
* hgnn has 339 k parameters and sage has 70 k.
* Peak PyTorch memory was 1.2 GB (hgnn) and 0.16 GB (sage), about 5% of a 24 GB card.
* Neighbour sampling runs on the CPU, so the GPU waits for most of each epoch.

A bigger model is a fair "best chance" check for the graph before we call the null result final.
It is not a promise of better scores: so far XGBoost ties or beats the GNNs.

Run these three predeclared settings, one tag each, all on the train stage only. Report all of them,
not just the best.

```bash
COMMON="--stage train --gpus 0 1 2 3 4 5 6 7 --dropout 0.2 --weight-decay 0.0001 --grad-clip 1.0 --epochs 80 --patience 10"
python scripts/17_gnn_only.py --tag trial004 $COMMON --hidden 256 --lr 0.001  --fanout 10 5  --batch 1024 --eval-batch 1024 2>&1 | tee -a outputs/logs/gnn_trial004.log
python scripts/17_gnn_only.py --tag trial005 $COMMON --hidden 128 --lr 0.001  --fanout 25 10 --batch 1024 --eval-batch 1024 2>&1 | tee -a outputs/logs/gnn_trial005.log
python scripts/17_gnn_only.py --tag trial006 $COMMON --hidden 256 --lr 0.002  --fanout 15 10 --batch 4096 --eval-batch 4096 2>&1 | tee -a outputs/logs/gnn_trial006.log
```

* Larger fanout makes CPU sampling slower and uses more RAM. Watch `free -g` during trial005.
* If RAM gets tight, add `--workers 4`.
* Larger batch means fewer optimizer steps per epoch, which is why trial006 also raises lr.

**Choosing a setting.** Compare trials 003–006 using `wcorr_edge` for new wallets in
`outputs/tables/exp6_<tag>_val_metrics.csv`. Then run the test stage **once**, on the chosen tag only:

```bash
python scripts/17_gnn_only.py --tag <chosen> --stage test
```

## Step 6: send the results back

Logs and tables are git-ignored, so add them explicitly. Do not add `data/`.

```bash
git checkout -b server-results-$(date +%Y%m%d)
git add -f outputs/logs/gnn/pilot_nanfix outputs/logs/gnn/trial00[3-6] outputs/logs/gnn_*.log \
  outputs/logs/09_edge_server2_* outputs/logs/14_test_server2.log outputs/logs/15_train_to_test.log \
  outputs/tables/exp6_trial00* outputs/tables/*server2* outputs/tables/exp5_* 2>/dev/null
git status --short          # check: only logs and CSV tables, nothing under data/
git commit -m "Server results after NaN fix: trial003-006, later-period graph test, out-of-time"
git push -u origin HEAD
```

If a step failed, push its log anyway and say which step failed. A failed log is still useful.

## If something goes wrong

| Message in the log | Meaning | What to do |
|---|---|---|
| `signature` / `changed settings` for an existing tag | tag reused after a code or setting change | use a new tag; keep the old one |
| `nonfinite input at index (...)` | another bad input value | stop and send the log; do not skip or zero it |
| `CUDA is unavailable` | wrong venv or driver | `source .venv/bin/activate`, then `python scripts/server_check.py` |
| `CUDA out of memory` | only possible with larger batch/hidden | add `--workers 4` or go back to the step-2 settings |
| process `Killed` with no Python error | host RAM ran out | fewer workers (`--workers 4`, `NGPU=4`) |
| `label bets do not reproduce fk_pop` | label artifacts are stale | see "Recovering from ..." in `docs/SERVER.md` |
