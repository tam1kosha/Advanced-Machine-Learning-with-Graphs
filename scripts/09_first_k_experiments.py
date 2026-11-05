"""Step 9 (iteration 2): detection curve over k for new and existing wallets (inductive first-k setting).

For each k and seed, one stratified 60/20/20 wallet split of the k-population (07), shared by all
methods (paired comparisons). Models are trained on new + existing wallets together (is_new is a
feature) and evaluated per population. Query wallets are inserted into the frozen pre-cutoff graph
(08) through their observed edges only (pmgraph/hetero.py).
Methods
  hist      history-only skill test on the observed fills (markets closed before the k-th fill)
  xgb_own   XGBoost on first-k wallet features
  xgb_nbr   + aggregates over the first-k neighbours (counterparties, markets, moved-first)
  sage      sampled GraphSAGE, counterparty edges only (homogeneous)
  hgnn      sampled hetero GraphSAGE: counterparty + moved-first + wallet-market edges
Targets: skill = label z > 2 (AUPRC), profitable = label profit > 0 (AUROC).
Writes one CSV per (k, method) to outputs/tables/exp2_parts/ (re-runs overwrite; 10 aggregates).
usage: python scripts/09_first_k_experiments.py [split] [methods,comma] [ks,comma]
"""
import sys
import time

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pmgraph import ROOT, load_config  # noqa: E402
from pmgraph.hetero import CSR, HELD_ATTRS, Graph, frozen_csrs, held_edges_pd, train_hetero  # noqa: E402
from pmgraph.tabular import run_in_subprocess  # noqa: E402  (XGBoost cannot share a process with torch on macOS)

cfg = load_config()
split = sys.argv[1] if len(sys.argv) > 1 else "train"
ALL = ["hist", "xgb_own", "xgb_nbr", "sage", "hgnn"]
methods = sys.argv[2].split(",") if len(sys.argv) > 2 else ALL
ks = [int(k) for k in sys.argv[3].split(",")] if len(sys.argv) > 3 else cfg["eval"]["k_buckets"]
P = ROOT / cfg["data"]["processed_dir"]
parts = ROOT / "outputs/tables/exp2_parts"
parts.mkdir(parents=True, exist_ok=True)
hp = cfg["gnn"]
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)  # noqa: E731
TARGETS = ["profitable", "skill"]
NB = ["hist_z", "hist_raw_edge", "hist_profit_slog", "log_usd", "maker_share", "frac_longshot", "log_n_fills"]

# ---------------------------------------------------------------- frozen graph
data = torch.load(P / f"{split}_hetero.pt", weights_only=False)
oc = list(data["wallet"].cols)
N = data["wallet"].num_nodes
Xraw = pd.DataFrame(data["wallet"].x.numpy(), columns=oc)


def wallet_matrix(df: pd.DataFrame, is_new: np.ndarray) -> np.ndarray:
    M = df[oc].copy()
    M["hist_missing"] = M["hist_z"].isna().astype(float)
    M = M.fillna(0.0)
    M["is_new"] = is_new.astype(float)
    return M.to_numpy(np.float32)


xsc = StandardScaler().fit(wallet_matrix(Xraw, np.zeros(N)))
Xf = xsc.transform(wallet_matrix(Xraw, np.zeros(N))).astype(np.float32)
Mraw = data["market"].x.numpy()
Xm = StandardScaler().fit_transform(Mraw).astype(np.float32)
midx = pd.Series(np.arange(len(Mraw)), index=data["market"].market_id.numpy())
fz, scalers = frozen_csrs(data)
del data
log("frozen graph", {r: len(c.src) for r, c in fz.items()})

Pop = pd.read_parquet(P / f"{split}_fk_pop.parquet")
Feat = pd.read_parquet(P / f"{split}_fk_feat.parquet")
obs = pd.read_parquet(P / f"{split}_fk_obs.parquet")
QMF = pd.read_parquet(P / f"{split}_fk_mf.parquet")


# ---------------------------------------------------------------- query graph for one k
def query_edges(k: int, nids: np.ndarray, t_k: pd.Series) -> dict:
    """Relation -> (dst query idx, src frozen id, standardised attr) for the wallets `nids` at k."""
    qi = pd.Series(np.arange(len(nids)), index=nids)
    o = obs[(obs.rn <= k) & obs.nid.isin(nids)]
    E = {}
    c = o[o.cp < N].groupby(["nid", "cp"]).size()
    E["cp"] = [(c.index.get_level_values(0), c.index.get_level_values(1), np.log1p(c.to_numpy())[:, None])]
    h = held_edges_pd(o[o.market_id.isin(midx.index)])
    E["rev_held"] = [(h.nid, midx.loc[h.market_id].to_numpy(), h[HELD_ATTRS].to_numpy())]
    m = QMF[(QMF.rn <= k) & QMF.nid.isin(nids)]
    m = m[(m["dir"] == "lead") | (m.t_other <= t_k.reindex(m.nid).to_numpy())]
    for r, d in [("mf", "lead"), ("rev_mf", "follow")]:
        g = m[m["dir"] == d].groupby(["nid", "other"]).size()
        E[r] = [(g.index.get_level_values(0), g.index.get_level_values(1), np.log1p(g.to_numpy())[:, None])]
    out = {}
    ex = nids[nids < N]
    for r, lst in E.items():
        dst, src, a = lst[0]
        a = (np.asarray(a, np.float32) - scalers[r][0]) / scalers[r][1]
        dst, src = qi.loc[np.asarray(dst)].to_numpy(), np.asarray(src, np.int64)
        if len(ex):                                  # existing wallets keep their pre-cutoff edges
            pos, vi = fz[r].rows(ex)
            dst = np.r_[dst, qi.loc[ex[vi]].to_numpy()]
            src = np.r_[src, fz[r].src[pos]]
            a = np.r_[a, fz[r].attr[pos]]
        out[r] = (dst.astype(np.int64), src, a.astype(np.float32))
    return out


def nbr_aggregates(E: dict, n: int, o: pd.DataFrame, nids: np.ndarray) -> pd.DataFrame:
    """Unweighted means of frozen neighbours' features per relation, degrees, attachment shares."""
    cols = {}
    for r, V, names in [("cp", Xraw[NB].fillna(0).to_numpy(), NB), ("mf", Xraw[["hist_z"]].fillna(0).to_numpy(), ["hist_z"]),
                        ("rev_mf", Xraw[["hist_z"]].fillna(0).to_numpy(), ["hist_z"]), ("rev_held", Mraw, None)]:
        dst, src, _ = E[r]
        S = sp.csr_matrix((np.ones(len(dst), np.float32), (dst, src)), shape=(n, V.shape[0]))
        deg = np.asarray(S.sum(1)).ravel()
        agg = sp.diags(1 / np.maximum(deg, 1)) @ S @ V
        names = names or [f"m{j}" for j in range(V.shape[1])]
        for j, c in enumerate(names):
            cols[f"nbr_{r}_{c}"] = agg[:, j]
        cols[f"deg_{r}"] = np.log1p(deg)
    qi = pd.Series(np.arange(n), index=nids)
    g = o.assign(cp_in=o.cp < N, mk_in=o.market_id.isin(midx.index)).groupby("nid")[["cp_in", "mk_in"]].mean()
    cols["share_cp_in_graph"] = g["cp_in"].reindex(nids).fillna(0).to_numpy()
    cols["share_mk_in_graph"] = g["mk_in"].reindex(nids).fillna(0).to_numpy()
    return pd.DataFrame(cols)


def metrics_rows(s, Y, te, new, **kw):
    rows = []
    for pop, mask in [("all", np.ones(len(te), bool)), ("new", new[te]), ("existing", ~new[te])]:
        for j, t in enumerate(TARGETS):
            yt = Y[te, j][mask]
            if yt.sum() < 3 or yt.sum() == len(yt):
                continue
            rows.append(dict(**kw, pop=pop, target=t, n=int(mask.sum()), pos=float(yt.mean()),
                             auroc=roc_auc_score(yt, s[te][mask, j]), auprc=average_precision_score(yt, s[te][mask, j])))
    return rows


# ---------------------------------------------------------------- main loop
for k in ks:
    pk = Pop[Pop.k == k].reset_index(drop=True)
    nids = pk.nid.to_numpy()
    n = len(pk)
    new = pk.new.to_numpy()
    Y = pk[TARGETS].to_numpy()
    fk = Feat[Feat.k == k].set_index("nid").reindex(nids)
    Fq_raw = wallet_matrix(fk, new)
    Fq = xsc.transform(Fq_raw).astype(np.float32)
    E = query_edges(k, nids, pk.set_index("nid")["t_k"])
    o = obs[(obs.rn <= k) & obs.nid.isin(nids)]
    NBR = nbr_aggregates(E, n, o, nids)
    g = Graph(x={"wallet": Xf, "market": Xm}, frozen=fz, xq=Fq,
              query={r: CSR.build(d, s, a, n) for r, (d, s, a) in E.items()})
    log(f"k={k}: n={n} (new {new.sum()}), query edges", {r: len(v[0]) for r, v in E.items()})
    strat = Y[:, 1] * 4 + Y[:, 0] * 2 + new
    splits = {}
    for seed in cfg["eval"]["seeds"]:
        tr, te = train_test_split(np.arange(n), test_size=0.2, random_state=seed, stratify=strat)
        tr, va = train_test_split(tr, test_size=0.25, random_state=seed, stratify=strat[tr])
        splits[seed] = (tr, va, te)
    for m in methods:
        rows = []
        tm = time.time()
        if m in ("xgb_own", "xgb_nbr"):    # trees need no early stopping: fit on train + val
            Fm = Fq_raw if m == "xgb_own" else np.c_[Fq_raw, NBR.to_numpy(np.float32)]
            xs = run_in_subprocess(Fm, Y, {sd: (np.r_[tr, va], te) for sd, (tr, va, te) in splits.items()})
        for seed, (tr, va, te) in splits.items():
            if m == "hist":
                s = np.stack([fk["hist_profit_slog"].fillna(0).to_numpy(), fk["hist_z"].fillna(0).to_numpy()], 1)
            elif m in ("xgb_own", "xgb_nbr"):
                s = xs[seed]
            else:
                rels = ["cp"] if m == "sage" else ["cp", "mf", "rev_mf", "rev_held", "held"]
                s = train_hetero(g, rels, Y, {"train": tr, "val": va}, seed, hp,
                                 log=lambda *a: log(f"k={k}", *a))
            rows += metrics_rows(s, Y, te, new, seed=seed, k=k, method=m)
        R = pd.DataFrame(rows)
        R.to_csv(parts / f"{split}_k{k}_{m}.csv", index=False)
        sk = R[(R.target == "skill")].groupby("pop").auprc.mean().round(3).to_dict()
        pr = R[(R.target == "profitable")].groupby("pop").auroc.mean().round(3).to_dict()
        log(f"k={k} {m:8s} {time.time()-tm:5.0f}s  skill AUPRC {sk}  profit AUROC {pr}")
