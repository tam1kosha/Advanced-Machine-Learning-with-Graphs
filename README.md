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

\*2-core / 7 GB machine. DuckDB does the heavy lifting out-of-core; memory is set in `prep.connect()`.

Config: `configs/mvp.yaml` (windows, label thresholds, FDR level, seeds).

## Layout

```
configs/mvp.yaml        windows, label + cleaning settings
src/pmgraph/prep.py     raw trades -> canonical fills (one signed position on the reference token)
src/pmgraph/labels.py   calibration, bets, skill test, BH, four types
src/pmgraph/features.py wallet features + counterparty graph (pre-cutoff only)
src/pmgraph/models.py   XGBoost, LR, spectral embedding, label propagation, sparse GraphSAGE
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

See `outputs/tables/exp1_train_summary.csv` and `NOTES.md`.
