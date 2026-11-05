"""Step 4: wallet features + counterparty graph as of the cutoff; attach labels."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pmgraph import ROOT, load_config  # noqa: E402
from pmgraph.features import counterparty_graph, neighbour_features, wallet_features  # noqa: E402
from pmgraph.labels import labels_file  # noqa: E402
from pmgraph.prep import connect  # noqa: E402

cfg = load_config()
split = sys.argv[1] if len(sys.argv) > 1 else "train"
sp_ = cfg["splits"][split]
I, P = ROOT / cfg["data"]["interim_dir"], ROOT / cfg["data"]["processed_dir"]
con = connect()
t0 = time.time()
G = I / f"{split}_graph_fills.parquet"
calib = pd.read_parquet(P / f"{split}_calibration.parquet")

X = wallet_features(con, G, sp_["cutoff"], calib)
print("features", X.shape, f"{time.time()-t0:.0f}s", flush=True)
A = counterparty_graph(con, G, X.index)
print("graph nnz", A.nnz, f"{time.time()-t0:.0f}s", flush=True)
N = neighbour_features(A, X, ["hist_z", "hist_raw_edge", "hist_profit_slog", "log_usd",
                              "maker_share", "frac_longshot", "log_n_fills"])
X = X.join(N)

lab = pd.read_parquet(labels_file(P, split, cfg["labels"].get("bet_unit", "market")))
X["in_label"] = X.index.isin(lab.index)
X.to_parquet(P / f"{split}_features.parquet")
sp.save_npz(P / f"{split}_adj.npz", A)
print("labelled wallets in graph:", int(X["in_label"].sum()), "of", len(lab))
print(X.describe().T[["mean", "std", "min", "max"]].round(3).to_string())
