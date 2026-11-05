# CLAUDE.md — guidance for Claude Code in this repo

Project: "Skilled or Lucky?" — early trader typing on Polymarket trading graphs (HKUST COMP 5222).
Read `proposal.pdf` (goals) and `NOTES.md` (current results + ordered next steps) before starting.

## Run
```bash
uv venv --python 3.11 && source .venv/bin/activate && uv pip install -e ".[community,dev]"
python -m pytest -q tests
bash run_mvp.sh train          # download (~3 GB) -> prepare -> eda -> labels -> graph -> experiments -> plot
python scripts/05_experiments.py train hist,xgb_own      # subset of methods; results merge into exp1_*_raw.csv
```
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
- Concise code, no notebooks for core logic (notebooks/ is for exploration only).
