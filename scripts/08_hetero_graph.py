"""Step 8 (iteration 2): wallet-market heterogeneous graph frozen at the cutoff + query moved-first edges.

Outputs (data/processed):
  {split}_hetero.pt     torch_geometric HeteroData: wallet / market nodes, (wallet, held, market),
                        (wallet, cp, wallet), (wallet, mf, wallet); all from fills with ts < cutoff.
                        wallet.x = iteration-1 own features (raw; standardised at training time)
  {split}_fk_mf.parquet moved-first edges of the first-k query wallets (nid, rn, other, t_other, dir),
                        against frozen wallets' entries at any time; filtered by k in 09.
Needs 04 (features, adjacency) and 07 (first-k observed fills).
usage: python scripts/08_hetero_graph.py [split]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch_geometric.data import HeteroData

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pmgraph import ROOT, load_config  # noqa: E402
from pmgraph.firstk import int_sides, register_ids  # noqa: E402
from pmgraph.hetero import (HELD_ATTRS, entries_sql, held_edges, market_features,  # noqa: E402
                            moved_first_frozen, moved_first_query)
from pmgraph.models import own_cols  # noqa: E402
from pmgraph.prep import connect  # noqa: E402

cfg = load_config()
split = sys.argv[1] if len(sys.argv) > 1 else "train"
I, P = ROOT / cfg["data"]["interim_dir"], ROOT / cfg["data"]["processed_dir"]
G, L = I / f"{split}_graph_fills.parquet", I / f"{split}_label_fills.parquet"
cut, gc = cfg["splits"][split]["cutoff"], cfg["graph"]
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)  # noqa: E731
con = connect()

X = pd.read_parquet(P / f"{split}_features.parquet")
N = register_ids(con, X.index, L)
int_sides(con, G, "gsall")
mx = pd.Timestamp(con.execute("SELECT max(ts)::VARCHAR FROM gsall").fetchall()[0][0])
assert mx < pd.Timestamp(cut, tz="UTC"), "frozen graph uses a fill at/after the cutoff"
log("pre-cutoff sides", con.execute("SELECT count(*) FROM gsall").fetchall()[0][0], "max ts", mx)

# ---- market nodes
M = market_features(con, G, ROOT / cfg["data"]["raw_dir"] / "markets.parquet", cut)
midx = pd.Series(np.arange(len(M)), index=M.index)
log("markets", M.shape)

# ---- (wallet, held, market)
H = held_edges(con, "gsall")
log("held edges", len(H))

# ---- (wallet, cp, wallet): iteration-1 counterparty adjacency (symmetric, log weights)
A = sp.load_npz(P / f"{split}_adj.npz").tocoo()

# ---- (wallet, mf, wallet): moved first, pre-cutoff entries only
con.execute(f"CREATE OR REPLACE TEMP TABLE ent_g AS {entries_sql('gsall')}")
MF = moved_first_frozen(con, "ent_g", gc["moved_first_window_min"], gc["moved_first_max_leaders"],
                        gc["moved_first_min_count"])
log("moved-first pairs", len(MF), "entries", con.execute("SELECT count(*) FROM ent_g").fetchall()[0][0])

data = HeteroData()
oc = own_cols(X)
data["wallet"].x = torch.from_numpy(X[oc].to_numpy(np.float32))
data["wallet"].cols = oc
data["market"].x = torch.from_numpy(M.to_numpy(np.float32))
data["market"].cols = list(M.columns)
data["market"].market_id = torch.from_numpy(M.index.to_numpy(np.int64))
data["wallet", "held", "market"].edge_index = torch.from_numpy(
    np.stack([H.nid.to_numpy(np.int64), midx.loc[H.market_id].to_numpy(np.int64)]))
data["wallet", "held", "market"].edge_attr = torch.from_numpy(H[HELD_ATTRS].to_numpy(np.float32))
data["wallet", "cp", "wallet"].edge_index = torch.from_numpy(np.stack([A.row, A.col]).astype(np.int64))
data["wallet", "cp", "wallet"].edge_attr = torch.from_numpy(A.data.astype(np.float32)[:, None])
data["wallet", "mf", "wallet"].edge_index = torch.from_numpy(MF[["src", "dst"]].to_numpy(np.int64).T.copy())
data["wallet", "mf", "wallet"].edge_attr = torch.from_numpy(np.log1p(MF["n"].to_numpy(np.float32))[:, None])
data.validate()
torch.save(data, P / f"{split}_hetero.pt")
log("saved", data)

# ---- query moved-first edges for the first-k wallets
obs = pd.read_parquet(P / f"{split}_fk_obs.parquet", columns=["nid", "rn", "is_maker", "market_id", "long_", "ts"])
q = (obs[obs.is_maker == 0].sort_values("rn").groupby(["nid", "market_id", "long_"], as_index=False).first())
con.register("q_ent", q[["nid", "rn", "market_id", "long_", "ts"]])
int_sides(con, L, "ls_g", keep=f"(SELECT nid FROM ids WHERE nid < {N})")
con.execute(f"""CREATE OR REPLACE TEMP TABLE ent_all AS
                SELECT * FROM ent_g UNION ALL {entries_sql('ls_g')}""")
QMF = moved_first_query(con, "q_ent", "ent_all", gc["moved_first_window_min"], gc["moved_first_max_leaders"])
QMF.to_parquet(P / f"{split}_fk_mf.parquet")
log("query moved-first edges", len(QMF), QMF["dir"].value_counts().to_dict())

stats = {"wallets": N, "markets": len(M), "held_edges": len(H), "cp_nnz": int(A.nnz), "mf_edges_count_ge_min": len(MF),
         "query_entries": len(q), "query_mf_edges": len(QMF),
         "frozen_max_ts": str(mx), "minutes": round((time.time() - t0) / 60, 1)}
print(json.dumps(stats, indent=1))
with open(ROOT / f"outputs/tables/hetero_{split}_graph.json", "w") as f:
    json.dump(stats, f, indent=2)
