# Server training logs

This folder contains a committed snapshot of the server logs shared with the group. Start with
the [learning-rate comparison](gnn/learning_rate_comparison.md) for the latest GNN experiments.
The next server run is described step by step in [`docs/SERVER_RERUN.md`](../../docs/SERVER_RERUN.md).

## Latest GNN-only trials

| Trial | Learning rate | Settings and job status | Launcher log |
|---|---:|---|---|
| trial001 | 0.001 | [Settings](gnn/trial001/run.json), [jobs](gnn/trial001/jobs.json) | [gnn_trial001.log](gnn_trial001.log) |
| trial002 | 0.0005 | [Settings](gnn/trial002/run.json), [jobs](gnn/trial002/jobs.json) | [gnn_trial002.log](gnn_trial002.log) |

Detailed epoch histories, checkpoints' validation scores and resource measurements are in the
individual model logs in [trial001](gnn/trial001/) and [trial002](gnn/trial002/).

Each trial completed 22 of 24 jobs (110 individual model fits). Heterogeneous GraphSAGE failed
before fitting at k=20 and k=50 in the later calendar period because a time-to-close edge feature
was nonfinite. Neither trial generated the final validation/test reports. The comparison describes
early-stopping validation correlations, not final held-out test performance.

`train` and `test` in these filenames identify calendar datasets. Each dataset separately has
training, validation and test wallet partitions. See the [GNN training guide](../../docs/GNN_ONLY.md)
for the evaluation protocol and commands.

## Earlier experiments

- `server_train.log`: historical launcher output, including earlier errors and attempts.
- `preflight_*.json`: recorded server hardware, package versions and run settings.
- `09_edge_server_*.log`: earlier per-k training logs.
- `12_train.log` / `12_test.log`: label diagnostics.
- `13_train.log` / `13_test.log`: earlier baseline experiments.
- `14_train_server.log` / `14_test_server_early.log`: earlier edge-evaluation reports.

Historical runs use different settings/protocols; their numbers are not direct comparisons with
the new fixed-holdout trials. Server paths in raw logs are preserved as recorded. Model weights,
prediction files and prepared data are not included in this snapshot.

New generated logs remain ignored by Git. To share another reviewed snapshot, explicitly add
the intended files with `git add -f`.
