"""Step 17 (iteration 5, Gate C): as-of nodes for markets without a pre-cutoff node (pmgraph/asof_markets.py).

For every k-population wallet (07) and every market of its first k fills that has no node in the frozen
graph (08), one private market node with features and trader in-edges as of the wallet's k-th fill.
Used by 09 when the GNN settings contain "asof_markets": true (env PMGRAPH_GNN), e.g.
  PMGRAPH_TAG=asof PMGRAPH_GNN='{"asof_markets": true}' python scripts/09_first_k_experiments.py train ... cv edge
Outputs (data/processed): {split}_fk_asof_mkt.parquet (iid, k, nid, market_id, market feature columns),
{split}_fk_asof_mkt_edges.parquet (iid, src, HELD_ATTRS); outputs/tables/asof_{split}_markets.json.
usage: python scripts/17_asof_markets.py [split] [ks, comma (default 1,3,5,10)] [max_traders (30)]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pmgraph import ROOT, load_config  # noqa: E402
from pmgraph.asof_markets import build  # noqa: E402
from pmgraph.firstk import int_sides, register_ids  # noqa: E402
from pmgraph.prep import connect  # noqa: E402

cfg = load_config()
split = sys.argv[1] if len(sys.argv) > 1 else "train"
ks = [int(k) for k in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 3, 5, 10]
M = int(sys.argv[3]) if len(sys.argv) > 3 else 30
I, P = ROOT / cfg["data"]["interim_dir"], ROOT / cfg["data"]["processed_dir"]
G, L = I / f"{split}_graph_fills.parquet", I / f"{split}_label_fills.parquet"
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)  # noqa: E731
con = connect()

gindex = pd.read_parquet(P / f"{split}_features.parquet", columns=["log_usd"]).index
N = register_ids(con, gindex, L)
frozen_mk = set(con.execute(f"SELECT DISTINCT market_id FROM '{G}'").fetchdf()["market_id"])
Pop = pd.read_parquet(P / f"{split}_fk_pop.parquet", columns=["nid", "k", "t_k", "new"])
obs = pd.read_parquet(P / f"{split}_fk_obs.parquet", columns=["nid", "rn", "market_id"])

inst = []
for k in ks:
    pk = Pop[Pop.k == k]
    o = obs[(obs.rn <= k) & obs.nid.isin(pk.nid) & ~obs.market_id.isin(frozen_mk)]
    inst.append(o[["nid", "market_id"]].drop_duplicates().merge(pk[["nid", "t_k", "new"]], on="nid").assign(k=k))
inst = pd.concat(inst, ignore_index=True)
inst["iid"] = np.arange(len(inst))
log("instances", len(inst), inst.groupby("k").size().to_dict())

int_sides(con, L, "ls")
log("label-window sides ready")
F, E = build(con, inst, "ls", N, M, ROOT / cfg["data"]["raw_dir"] / "markets.parquet")
log("features", F.shape, "edges", len(E), "| as-of asserts passed (every fill used is before t_k)")

out = inst[["iid", "k", "nid", "market_id"]].join(F, on="iid")
out.to_parquet(P / f"{split}_fk_asof_mkt.parquet", index=False)
E.to_parquet(P / f"{split}_fk_asof_mkt_edges.parquet", index=False)

deg = E.groupby("iid").size().reindex(inst.iid).fillna(0)
summ = {"max_traders": M, "minutes": round((time.time() - t0) / 60, 1), "by_k": {}}
for k in ks:
    m = (inst.k == k).to_numpy()
    o = obs[(obs.rn <= k) & obs.nid.isin(Pop[Pop.k == k].nid)]
    nw = o.nid >= N
    summ["by_k"][k] = {
        "instances": int(m.sum()), "instances_new_wallets": int((m & inst.new.to_numpy()).sum()),
        "new_fills_in_frozen_markets": round(float(o.market_id[nw].isin(frozen_mk).mean()), 3),
        "mean_traders": round(float(deg[m].mean()), 1), "share_no_trader": round(float((deg[m] == 0).mean()), 3),
        "median_log_fills_before_tk": round(float(F.loc[inst.iid[m], "log_fills"].median()), 2)}
print(json.dumps(summ, indent=1))
with open(ROOT / f"outputs/tables/asof_{split}_markets.json", "w") as f:
    json.dump(summ, f, indent=2)
