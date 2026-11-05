# Skilled or Lucky? — Early trader typing on Polymarket graphs

COMP 5222 group project (Mukhanov, Lin, Bayraktar). Proposal: `proposal.pdf`.

Given the Polymarket trading graph up to a cutoff, predict each wallet's **type**
(skilled winner / lucky winner / unlucky skilled / unskilled loser), where types come
from how the wallet's *later* bets resolve.

## Setup

```bash
# Python 3.10–3.12. uv is fastest; plain pip works too.
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[community,dev]"
# CPU torch is fine for the MVP. On a CUDA box install the matching torch wheel first.
python -m pytest -q tests          # sanity checks for the skill test
```

## Pipeline

```bash
bash run_mvp.sh train      # cutoff 2025-01-01 (graph Jul–Dec 2024, labels Jan–Jun 2025)
bash run_mvp.sh test       # cutoff 2025-07-01 (needs Jan–Dec 2025 trades, ~4.5 GB more)
```

| Step | Script | Output (`data/processed`, `outputs/`) | Time* |
|---|---|---|---|
| 0 | `scripts/download.py` | `data/raw/` (HF parquet, day-partitioned trades) | ~1 min / 3 GB |
| 1 | `01_prepare.py` | `data/interim/{split}_{graph,label}_fills.parquet` | ~45 s |
| 2 | `02_eda.py` | `outputs/figures/eda_*.png`, `outputs/tables/eda_*.json` | ~20 s |
| 3 | `03_labels.py` | `{split}_labels.parquet`, `{split}_calibration.parquet` | ~35 s |
| 4 | `04_build_graph.py` | `{split}_features.parquet`, `{split}_adj.npz` | ~1 min |
| 5 | `05_experiments.py` | `outputs/tables/exp1_*` | 30 s (no GNN) |
| 6 | `06_plot_results.py` | `outputs/figures/exp1_*_detection.png` | 5 s |
| 7 | `07_first_k.py` | `{split}_fk_{pop,obs,feat}.parquet`: first-k populations, per-k leakage asserts | ~1.5 min |
| 8 | `08_hetero_graph.py` | `{split}_hetero.pt` (PyG HeteroData, frozen at the cutoff), `{split}_fk_mf.parquet` | ~4 min |
| 9 | `09_first_k_experiments.py` | `outputs/tables/exp2_parts/` (one CSV per k × method) | ~35 min (GNNs) |
| 10 | `10_first_k_report.py` | `exp2_*_{raw,summary,paired}.csv`, `outputs/figures/exp2_*_detection.png` | 5 s |

\*2-core / 7 GB machine (steps 7–10 timed on a 12-core Mac). DuckDB does the heavy lifting out-of-core;
memory and threads are set in `prep.connect()` (2 threads by default, `PMGRAPH_THREADS=n` overrides: more
threads under the 5 GB cap spill heavily).

```bash
# iteration 2 (first-k inductive setting); 09 takes methods and k lists, so k values can run in parallel
python scripts/03_labels.py train all                      # market- and event-level labels
python scripts/07_first_k.py train && python scripts/08_hetero_graph.py train
python scripts/09_first_k_experiments.py train hist,xgb_own,xgb_nbr,sage,hgnn 1,3,5,10,20,50
python scripts/10_first_k_report.py train
```
On macOS, xgboost and torch cannot share a process (two `libomp`, segfault); 09 runs XGBoost in a
subprocess, and for 05 run `xgb_*` and `sage` in separate calls (results merge per method).

Config: `configs/mvp.yaml` (windows, label thresholds, FDR level, seeds).

## Layout

```
configs/mvp.yaml        windows, label + cleaning settings
src/pmgraph/prep.py     raw trades -> canonical fills (one signed position on the reference token)
src/pmgraph/labels.py   calibration, bets, skill test, BH, four types
src/pmgraph/features.py wallet features + counterparty graph (pre-cutoff only)
src/pmgraph/models.py   XGBoost, LR, spectral embedding, label propagation, sparse GraphSAGE
src/pmgraph/firstk.py   first-k populations: observed fills, post-k labels, leakage asserts
src/pmgraph/hetero.py   frozen wallet-market graph, query-node sampler, hetero GraphSAGE (HeteroConv)
src/pmgraph/tabular.py  torch-free XGBoost worker (subprocess)
scripts/                numbered pipeline steps
tests/                  simulated-wallet checks of the skill test
```

## Data

`vgregoire/polymarket-users` (Hugging Face, CC-BY 4.0). Tables used:
`trades/` (day-partitioned: `price, quantity, maker_address, taker_address, taker_bought, winner, …`),
`predictions.parquet` (token → market, outcome index), `markets.parquet` (`close_time`, fees).
`user_features` / `user_pnl_summary` are **lifetime** aggregates: never use them as features (leakage).

## Design decisions (beyond the proposal)

* **Reference token.** Every fill is mapped to the market's outcome-0 token: buying token 1 at *p*
  = shorting token 0 at 1 − *p*. Only binary markets (>99.99% of tokens).
* **Bet = (wallet, market) position**, not a fill. Fills in the same market share one outcome, so
  counting them separately inflates significance. Pure round-trips (|net direction| < 0.1) are dropped.
* **Skill statistic.** Per bet, *d* = A·W − C (signed outcome minus *calibrated* price, per share); per
  wallet, *T* = mean over bets. Null: W ~ Bernoulli(calibrated price), independent across markets.
  Analytic z-test (MC available; tests in `tests/`). BH at α = 0.10.
* **Favourite–longshot calibration** is fitted on markets that **closed before the cutoff**, with every
  market weighted equally. A per-fill fit is dominated by the 2024 election and bakes its outcome into
  the null.
* **Timezones.** DuckDB defaults to the machine's zone (HKT here); `connect()` pins UTC.
* **No fees** in profit yet (trades table has no fee column; fees only on some markets from Q4 2024).

## First results (train cutoff 2025-01-01, transductive wallet split, 5 seeds)

GraphSAGE leads on the rare skill class (AUPRC 0.146 vs 0.116 history-only, 0.098 XGBoost; 5/5 seeds),
XGBoost leads on profitability (AUROC 0.69). Gains are in active wallets, not yet new ones.
Details, tables and next steps: `NOTES.md`; figure: `outputs/figures/exp1_train_detection.png`.

## Iteration 2 (first k post-cutoff fills, new vs existing wallets)

The proposal's graph claim is **not met**: for new wallets at k ≤ 10 every method's skill AUPRC is
within ~0.015 of the 1.9% base rate, and no graph model (XGBoost + neighbour aggregates, GraphSAGE,
wallet–market hetero GraphSAGE with moved-first edges) beats both baselines. For existing wallets
GraphSAGE adds ~+0.03 AUPRC at k ≤ 5 (4–5/5 seeds). Event-level bets do not fix the z over-dispersion.
Details: `NOTES.md` (Iteration 2); figure: `outputs/figures/exp2_train_detection.png`.
