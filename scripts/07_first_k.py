"""Step 7 (iteration 2): inductive first-k dataset for new and existing wallets.

For every k in eval.k_buckets: wallets with >= k post-cutoff fills whose label (bets after the k-th
fill, in markets not observed; see pmgraph/firstk.py) has >= min_resolved_bets bets.
`new` = no fill in the graph window. Leakage asserts run per k.
Outputs (data/processed, integer wallet ids only):
  {split}_fk_pop.parquet    (nid, k): new, t_k, label columns (n_bets, T, z, p, profit, skill, profitable)
  {split}_fk_obs.parquet    first max(k) post-cutoff fills of every wallet in any population
  {split}_fk_feat.parquet   (nid, k): wallet features on observed(w, k), as of the k-th fill
usage: python scripts/07_first_k.py [split]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pmgraph import ROOT, load_config  # noqa: E402
from pmgraph.firstk import first_k_features, int_sides, population, prepare, register_ids  # noqa: E402
from pmgraph.prep import connect  # noqa: E402

cfg = load_config()
split = sys.argv[1] if len(sys.argv) > 1 else "train"
I, P = ROOT / cfg["data"]["interim_dir"], ROOT / cfg["data"]["processed_dir"]
G, L = I / f"{split}_graph_fills.parquet", I / f"{split}_label_fills.parquet"
lc, fk = cfg["labels"], cfg["first_k"]
ks = cfg["eval"]["k_buckets"]
K = max(ks)
mode = fk.get("label_after", "k")
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)  # noqa: E731
con = connect()

calib = pd.read_parquet(P / f"{split}_calibration.parquet")
gindex = pd.read_parquet(P / f"{split}_features.parquet", columns=["log_usd"]).index
N = register_ids(con, gindex, L)
log("ids", con.execute("SELECT count(*) FROM ids").fetchall()[0][0], "graph wallets", N)
int_sides(con, L, "ls")
log("label-window sides", con.execute("SELECT count(*) FROM ls").fetchall()[0][0])
prepare(con, N, K, lc["min_resolved_bets"], G)
log("candidates", con.execute("SELECT count(*) FROM cand").fetchall()[0][0])

pops, feats = [], []
fixed = population(con, N, K, lc["min_resolved_bets"], lc.get("bet_unit", "market"), calib) if mode == "kmax" else None
for k in ks:
    p = fixed.copy() if fixed is not None else population(con, N, k, lc["min_resolved_bets"],
                                                           lc.get("bet_unit", "market"), calib)
    if fk.get("subsample", 1.0) < 1.0:
        p = p[(p.index.to_series() * 2654435761 % 1000) < fk["subsample"] * 1000]   # same wallets for every k
    p["k"] = k
    f = first_k_features(con, k, calib, p.index.to_numpy())
    f["k"] = k
    pops.append(p.drop(columns=["t_j"], errors="ignore").reset_index())
    feats.append(f.reset_index())
    log(f"k={k}: population {len(p)} (new {int(p.new.sum())}), skill {p.skill.mean():.4f}, "
        f"profitable {p.profitable.mean():.3f}, hist_z non-null new {f.loc[p.index[p.new], 'hist_z'].notna().mean():.3f}"
        " | leakage asserts passed")
Pop = pd.concat(pops, ignore_index=True)
F = pd.concat(feats, ignore_index=True)
keep = Pop.nid.unique()
obs = con.execute("SELECT * EXCLUDE (close_time) FROM obs ORDER BY nid, rn").fetchdf()
obs = obs[obs.nid.isin(keep)]
tk = obs[obs.rn.isin(ks)][["nid", "rn", "ts"]].rename(columns={"rn": "k", "ts": "t_k"})
Pop = Pop.merge(tk, on=["nid", "k"], how="left")

# attachment of the first k fills to the frozen pre-cutoff graph
mk = set(con.execute(f"SELECT DISTINCT market_id FROM '{G}'").fetchdf()["market_id"])
attach = {}
for k in ks:
    w = Pop[Pop.k == k]
    o = obs[(obs.rn <= k) & obs.nid.isin(w.nid)]
    nw = o.nid >= N
    attach[k] = {pop: {"cp_in_graph": float((o.cp[m] < N).mean()), "market_in_graph": float(o.market_id[m].isin(mk).mean())}
                 for pop, m in [("new", nw), ("existing", ~nw)]}

Pop.to_parquet(P / f"{split}_fk_pop.parquet")
obs.to_parquet(P / f"{split}_fk_obs.parquet")
F.to_parquet(P / f"{split}_fk_feat.parquet")

summ = {"label_after": mode, "bet_unit": lc.get("bet_unit", "market"), "subsample": fk.get("subsample", 1.0),
        "by_k": {int(k): {("new" if n else "existing"): {
            "n": len(g), "skill_z2": round(float(g.skill.mean()), 4), "n_skill": int(g.skill.sum()),
            "profitable": round(float(g.profitable.mean()), 4), "median_label_bets": float(g.n_bets.median()),
            "z_sd": round(float(g.z.std()), 3)} for n, g in Pop[Pop.k == k].groupby("new")} for k in ks},
        "attachment": attach}
print(json.dumps(summ, indent=1))
with open(ROOT / f"outputs/tables/fk_{split}_population.json", "w") as f:
    json.dump(summ, f, indent=2)
log("done")
