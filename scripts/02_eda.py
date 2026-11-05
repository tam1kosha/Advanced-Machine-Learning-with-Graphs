"""Step 2: EDA + sanity checks on the cleaned fills (prints tables, saves figures)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pmgraph import ROOT, load_config  # noqa: E402
from pmgraph.prep import connect  # noqa: E402

cfg = load_config()
split = sys.argv[1] if len(sys.argv) > 1 else "train"
I = ROOT / cfg["data"]["interim_dir"]
G, L = I / f"{split}_graph_fills.parquet", I / f"{split}_label_fills.parquet"
con = connect()
q = lambda s: con.execute(s).fetchdf()  # noqa: E731
res = {}

for name, f in [("graph", G), ("label", L)]:
    con.execute(f"""CREATE OR REPLACE TEMP VIEW sides_{name} AS
        SELECT maker AS wallet, qty, usd FROM '{f}' UNION ALL SELECT taker, qty, usd FROM '{f}'""")
    w = q(f"""SELECT count(*) n_wallets, median(n) med_fills, quantile_cont(n, 0.99) p99_fills, max(n) max_fills
              FROM (SELECT wallet, count(*) n FROM sides_{name} GROUP BY 1)""")
    res[f"{name}_wallets"] = w.to_dict("records")[0]
    print(name, res[f"{name}_wallets"])

# wallet-level bets in the label window: (wallet, market) positions on resolved markets
bets = q(f"""
  WITH s AS (SELECT maker wallet, market_id, -1 sgn, qty FROM '{L}' WHERE w IS NOT NULL
             UNION ALL SELECT taker, market_id, 1, qty FROM '{L}' WHERE w IS NOT NULL)
  SELECT n_bets, count(*) n_wallets FROM (SELECT wallet, count(DISTINCT market_id) n_bets FROM s GROUP BY 1)
  GROUP BY 1 ORDER BY 1""")
nb = bets.set_index("n_bets")["n_wallets"]
for k in [1, 10, 30, 100, 250]:
    res[f"label_wallets_with_>={k}_bets"] = int(nb[nb.index >= k].sum())
print({k: v for k, v in res.items() if "bets" in k})

# overlap: labelable wallets that are also in the graph window
ov = q(f"""
  WITH lb AS (SELECT wallet FROM (SELECT maker wallet, market_id FROM '{L}' WHERE w IS NOT NULL
                                 UNION ALL SELECT taker, market_id FROM '{L}' WHERE w IS NOT NULL)
              GROUP BY 1 HAVING count(DISTINCT market_id) >= {cfg['labels']['min_resolved_bets']}),
       gw AS (SELECT DISTINCT wallet FROM sides_graph)
  SELECT count(*) labelable, count(gw.wallet) in_graph FROM lb LEFT JOIN gw USING (wallet)""")
res["labelable_overlap"] = ov.to_dict("records")[0]
print("labelable overlap", res["labelable_overlap"])

# concentration: share of fills touched by the top-10 wallets (market makers / hubs)
top = q("SELECT wallet, count(*) n FROM sides_graph GROUP BY 1 ORDER BY 2 DESC LIMIT 10")
tot = q("SELECT count(*) n FROM sides_graph")["n"][0]
res["top10_wallet_fill_share"] = float(top["n"].sum() / tot)
print("top10 share of fill-sides", res["top10_wallet_fill_share"])

# calibration / favourite-longshot curve on graph-window fills (symmetrised)
cal = q(f"""
  WITH x AS (SELECT price p, w FROM '{G}' WHERE w IS NOT NULL
             UNION ALL SELECT 1 - price, 1 - w FROM '{G}' WHERE w IS NOT NULL)
  SELECT floor(p * 50) / 50 + 0.01 AS bin, avg(w) win_rate, count(*) n FROM x GROUP BY 1 ORDER BY 1""")
# mapping sanity: last fill before resolution should sit near the realised outcome
last = q(f"""SELECT avg(abs(price - w)) mae FROM (
  SELECT market_id, arg_max(price, ts) price, any_value(w) w FROM '{G}' WHERE w IS NOT NULL GROUP BY 1)""")
res["last_price_abs_err"] = float(last["mae"][0])
print("mean |last price - outcome|", res["last_price_abs_err"])

cats = q(f"SELECT category, count(*) n, sum(usd)/1e6 usd_m FROM '{G}' GROUP BY 1 ORDER BY 3 DESC")
print(cats.to_string(index=False))
res["categories_graph"] = cats.to_dict("records")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot([0, 1], [0, 1], color="#999", lw=1, ls="--")
ax[0].plot(cal["bin"], cal["win_rate"], marker="o", ms=3, color="#2a6fdb")
ax[0].set(xlabel="price (implied probability)", ylabel="empirical win rate",
          title="Calibration of fills, Jul–Dec 2024")
ax[1].loglog(nb.index, nb.values, ".", color="#2a6fdb")
ax[1].axvline(30, color="#d9534f", lw=1)
ax[1].set(xlabel="resolved bets per wallet (label window)", ylabel="# wallets",
          title="Bets per wallet; red = label threshold")
fig.tight_layout()
(ROOT / "outputs/figures").mkdir(parents=True, exist_ok=True)
fig.savefig(ROOT / f"outputs/figures/eda_{split}.png", dpi=150)
with open(ROOT / f"outputs/tables/eda_{split}.json", "w") as f:
    json.dump(res, f, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
