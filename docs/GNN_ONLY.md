# Training and tuning the two GNNs

`scripts/17_gnn_only.py` runs **only** `sage` (counterparty GraphSAGE) and `hgnn` (heterogeneous
GraphSAGE). It reuses the processed graphs/first-k data from steps 07, 08 and 12. It does not download
data, run XGBoost, rerun diagnostics, or call the baseline-dependent report in step 14.

This is experiment 6 with its own holdout protocol and output directories. It is not a replacement
for the historical five-fold CV experiment. Two message-passing layers are retained because the
sampler constructs two-hop batches. The script exposes supported hyperparameters, not arbitrary depth.

## Server commands

Use the existing CUDA-enabled virtual environment. From the repository directory, start tmux:

```bash
tmux new -s gnns
```

Inside tmux, run a tuning trial (the settings below are an example, not a claim they perform better):

```bash
source .venv/bin/activate
mkdir -p outputs/logs
set -o pipefail

python scripts/17_gnn_only.py \
  --tag trial001 --stage train \
  --gpus 0 1 2 3 4 5 6 7 \
  --hidden 128 --lr 0.001 --dropout 0.2 --weight-decay 0.0001 \
  --grad-clip 1.0 --fanout 10 5 --batch 1024 --eval-batch 1024 \
  --epochs 80 --patience 10 \
  2>&1 | tee -a outputs/logs/gnn_trial001.log
```

This schedules 24 model × k × period jobs, with one job per GPU and a dynamic queue. Each job uses
five initialization seeds. Selected GPU ids/UUIDs must belong to `CUDA_VISIBLE_DEVICES` if an
allocation is set. `--workers 4` caps concurrency to reduce host RAM/CPU pressure. The full graph is
stored separately in each process's host RAM; the GPUs are not pooled into one model.

Detach with Ctrl+B then D; reconnect with `tmux attach -t gnns`. The main `tee` log contains launcher
progress. Detailed epochs/resources are already redirected to per-job files, for example:

```bash
tail -f outputs/logs/gnn/trial001/train_k1_hgnn.log
```

For a one-card pilot before the full run:

```bash
python scripts/17_gnn_only.py --tag pilot --stage train \
  --gpus 0 --splits train --ks 1 --seeds 0
```

For a CPU debugging run, use `--device cpu --workers 1` and omit `--gpus`. CUDA is required by
default and actually exercised before GPU jobs are launched; it never silently switches to CPU.
Use `--dry-run` to inspect the settings/jobs without data or GPUs.

## Validation and test are different stages

Within each dataset period/k, wallets receive one stratified **60% train / 20% validation / 20% test**
partition. It is identical across both models, initialization seeds, and trials that retain the same
`--split-seed` and prepared inputs. `--seeds` are initialization/sampling repetitions, not CV folds.

Targets are centred continuous edge scores from even-week (half-A) label bets. The target variance
normalization is fit on training wallets only. GNN early stopping uses half-A validation correlation.
After fitting, `exp6_<tag>_val_metrics.csv` evaluates the same validation wallets against separate odd-week
(half-B) realised edge. Use that file's `wcorr_edge` on new wallets to compare tuning settings.
The checkpoint's `best_val_score` is the early-stopping correlation, which differs from this half-B
evaluation. Scores are correlations, not classification accuracies.

For each new setting, choose a new tag and keep `--split-seed` fixed. Select settings using validation
results, then evaluate the chosen trial on its untouched test wallets:

```bash
python scripts/17_gnn_only.py --tag trial001 --stage test
```

This reads the saved trial configuration and predictions, does not retrain, needs no GPU, and writes
`exp6_<tag>_test_metrics.csv`. Do not pass training flags to this stage. If you want one-off training followed
immediately by test reporting, use `--stage train-test` (also the default). Do not choose a winning
configuration from repeated test results and report that same score as an unbiased final evaluation.

`--splits train test` refers to the two calendar periods, not the wallet partitions above. Each period
gets a separate model/holdout. This is within-period evaluation, not training in the earlier period
and transferring a fitted GNN to the later period. Existing label-availability/realised-closing-time
caveats in SERVER.md remain. Existing exp4 baseline numbers use different evaluation wallets and
normalization; a later formal XGBoost comparison must use this same holdout protocol.
These wallets already appeared in earlier project experiments; the fixed split separates tuning
and testing for these new fits, but does not create a new independent dataset.

## Hyperparameters and artifacts

`--settings configs/gnn_only.yaml` loads an editable YAML/JSON profile. Explicit CLI flags override it.
Supported settings: `--hidden`, `--lr`, `--dropout`, `--weight-decay`, `--grad-clip`, `--fanout H1 H2`,
`--batch`, `--eval-batch`, `--epochs`, `--patience`, `--eval-reps`, and `--threads`. Missing eval_batch
defaults to the training batch. `--grad-clip 0` disables clipping but retains nonfinite-gradient checks.
Use `--models sage` or `--models hgnn` to restrict the model, `--ks 1 3 5 10` for early-trade settings,
and `--splits train` / `--splits test` to restrict the calendar period.

Outputs for tag `trial001`:

| Path | Contents |
|---|---|
| `outputs/logs/gnn/trial001/run.json` | Exact settings, package versions, source/input fingerprint, device allocation |
| `outputs/logs/gnn/trial001/jobs.json` | Per-job return codes and log paths; failed jobs cause a nonzero launcher exit |
| `outputs/logs/gnn/trial001/*_k*_*.log` | Training epochs and resource records |
| `outputs/tables/exp6_trial001_{val,test}_metrics.csv` | Ensemble metrics by period/k/model/new-or-existing population |
| `outputs/tables/exp6_trial001_{val,test}_paired.csv` | Paired differences, heterogeneous GraphSAGE minus GraphSAGE |
| `outputs/logs/gnn/trial001/{val,test}_report.log` | Readable evaluation tables, also printed to the launcher terminal |
| `data/processed/exp6_preds_trial001/` | Separate validation/test scores, seeds and saved wallet partition |
| `data/processed/checkpoints/exp6_trial001/` | Best-validation weights per seed, scalers and fit metadata |

Metrics include weighted correlation, Spearman correlation, top-minus-bottom-decile edge, and rare
winner/loser AUPRC. Evaluation requires at least ten half-B bets, so evaluated wallet counts may be
smaller than the prediction holdout. Predictions are converted to percentile ranks per initialization
and averaged into a seed ensemble. Bootstrap intervals describe wallet uncertainty for that ensemble;
`sd_seeds` separately measures score variation between initializations. Wallet bootstraps do not
fully account for shared-market dependence. `--bootstrap 0` gives faster point estimates with no CIs;
the default is 200, and `--bootstrap 500` increases repetitions for final reporting.
The CSV `unit` column identifies edge units per share for top-minus-bottom; readable logs convert
that metric to cents per share. Neither quantity is a guaranteed account profit/trading return.

Repeated identical training commands skip completed model/k/period jobs when their artifacts match.
An interrupted job restarts its seeds; it does not resume an optimizer mid-epoch. Changing settings,
source, or prepared data under an existing tag is rejected: choose a new tag to preserve the prior
trial. Keep `data/` and checkpoints on persistent server storage (Git ignores them). Ctrl+C/SIGTERM
stops the launcher and its child process groups; tmux protects against SSH disconnection.

## Numerical failures

Missing market feature values are now imputed to zero before scaling, consistently with wallet
features. This fixes a reproducible route for a NaN market input to corrupt heterogeneous message
passing. Logs record the imputation count. Infinities and invalid graph/edge inputs are rejected
with a source/index before stochastic batches hide the location. Gradient clipping and a finite
gradient check protect optimizer updates; epoch history and the actual validation metric name are
saved. No loss is replaced with zero and no failed batch is silently discarded.

The full later-period k=20/50 server data are not available locally. These changes address concrete
input/stability failure modes but do not prove the exact cause of the old epoch-3 errors. A server
pilot is still needed; any remaining failure is retained in the per-job log for diagnosis.
