# CLAUDE.md — guidance for Claude Code in this repo

Project: "Skilled or Lucky?" — early trader typing on Polymarket trading graphs (HKUST COMP 5222).
Read `README.md` (project map, decision tree, gates), `docs/RESULTS.md` (what each figure showed) and the
latest entry of `NOTES.md` (numbers, ordered next steps) before starting. `proposal.pdf` has the original goals;
`docs/ONBOARDING.md` has the open tasks.

## Run
```bash
uv venv --python 3.11 && source .venv/bin/activate && uv pip install -e ".[community,dev]"
python -m pytest -q tests
bash run_mvp.sh train          # download (~3 GB) -> prepare -> eda -> labels -> graph -> experiments -> plot
python scripts/05_experiments.py train hist,xgb_own      # subset of methods; results merge into exp1_*_raw.csv
python scripts/09_first_k_experiments.py train sage 1,3  # iteration 2: methods x k; parts in outputs/tables/exp2_parts/
python scripts/09_first_k_experiments.py train xgb_own,hgnn 1,3 cv edge   # iteration 4: 5-fold CV on the edge target
python scripts/16_overview_figures.py train              # overview figures from outputs/tables only (no data/)
python scripts/17_asof_markets.py train && PMGRAPH_TAG=asof PMGRAPH_GNN='{"asof_markets": true}' \
  python scripts/09_first_k_experiments.py train hgnn 1 cv edge  # iteration 5, Gate C; 18_gate_report.py compares tags
bash run_server.sh                                       # GPU-server stages (both splits, out-of-time, GNN grid)
```
DuckDB runs with 2 threads by default (`PMGRAPH_THREADS`); more threads under the 5 GB cap spill tens of GB.
On macOS xgboost and torch segfault in one process: keep XGBoost in `pmgraph.tabular` (subprocess).
Data lives in `data/` (git-ignored; re-create with `scripts/download.py`). Numbered scripts are the pipeline;
library code is in `src/pmgraph/`. Settings in `configs/mvp.yaml`.

## Invariants (do not break)
- **No leakage.** Features/graph use only fills with `ts < cutoff`. Anything using outcomes (`w`) before the
  cutoff must be restricted to markets with `close_time < cutoff` (see `bets(..., closed_before=)`).
  Labels use only fills in `[cutoff, label_end)`. Never use `user_features.parquet` / `user_pnl_summary.parquet`
  as model inputs: they are lifetime aggregates.
- **UTC.** Always get DuckDB connections from `pmgraph.prep.connect()` (pins `TimeZone='UTC'`; the machine is HKT).
- **Reference token.** Fills are signed positions on the market's outcome-0 token (`price`, `taker_long`, `w`).
- **Bet = (wallet, market)**, |net direction| >= 0.1. Calibration is market-weighted, isotonic, pre-cutoff markets only.
- Memory: DuckDB for anything over fills (30M rows/6 months); pull only integer ids / aggregates into pandas.
  Address strings into pandas at 10M+ rows will OOM on small machines.

## Conventions
- Report mean ± sd over seeds in `configs/mvp.yaml: eval.seeds`; break results down by pre-cutoff activity bucket.
- New experiments: new numbered script + rows in a CSV under `outputs/tables/`, figure under `outputs/figures/`,
  and a short dated entry in `NOTES.md` (what was run, numbers, reading).
- Keep `tests/` passing; add a test for any new statistical procedure (simulate the null).
- Figures: use `pmgraph.plotstyle` (method names and colours). When the story changes, update the README
  tables and add a section to `docs/RESULTS.md`. Do not rewrite the outputs of earlier experiments.
- Concise code, no notebooks for core logic (notebooks/ is for exploration only).
