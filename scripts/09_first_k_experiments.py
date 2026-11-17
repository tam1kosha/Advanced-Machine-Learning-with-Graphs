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
Split modes (4th argument, default configs first_k.split_mode):
  random  iteration 2: 5 seeds x one 60/20/20 split -> outputs/tables/exp2_parts/ (10 aggregates)
  cv      5-fold CV: fold f = test, fold f+1 = val, rest = train (same 60/20/20), so every wallet is
          scored once -> outputs/tables/exp3_parts/ (11 pools the out-of-fold scores)
Writes one CSV of per-split metrics per (k, method) (re-runs overwrite) and the test scores of every
wallet to data/processed/{exp2,exp3}_preds/.
Target (5th argument, iteration 4): `label` (default, the two binary targets above) or `edge`: one
regression target, the centred z of the wallet's label bets in markets closing in even weeks (half A
of the `week` split of 12; wallets with < 10 such bets are not trained on). cv mode only; only the
scores are written, to data/processed/exp4_preds/, and 14 judges them on the odd-week half.
GNN settings: configs gnn, overridden by the JSON in env PMGRAPH_GNN (e.g. '{"hidden": 128, "fanout": [20, 10]}');
"asof_markets": true (iteration 5, Gate C; needs 17) adds as-of nodes for markets without a frozen node, for
every method that uses the graph (xgb_nbr, sage, hgnn);
env PMGRAPH_TAG=name writes the scores to {exp}_preds_name/ instead, so several settings can run side by
side (one process per GPU: CUDA_VISIBLE_DEVICES). The device is CUDA when available (pmgraph.hetero).
usage: python scripts/09_first_k_experiments.py [split] [methods,comma] [ks,comma] [random|cv] [label|edge]
"""
import json
import os
import sys
import time

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pmgraph import ROOT, load_config  # noqa: E402
from pmgraph.hetero import CSR, HELD_ATTRS, Graph, frozen_csrs, held_edges_pd, train_hetero  # noqa: E402
from pmgraph.label_diag import variance_function  # noqa: E402
from pmgraph.tabular import run_in_subprocess  # noqa: E402  (XGBoost cannot share a process with torch on macOS)

cfg = load_config()
split = sys.argv[1] if len(sys.argv) > 1 else "train"
ALL = ["hist", "xgb_own", "xgb_nbr", "sage", "hgnn"]
methods = sys.argv[2].split(",") if len(sys.argv) > 2 else ALL
ks = [int(k) for k in sys.argv[3].split(",")] if len(sys.argv) > 3 else cfg["eval"]["k_buckets"]
mode = sys.argv[4] if len(sys.argv) > 4 else cfg["first_k"].get("split_mode", "random")
assert mode in ("random", "cv"), mode
target = sys.argv[5] if len(sys.argv) > 5 else "label"
assert target in ("label", "edge") and (target == "label" or mode == "cv"), (mode, target)
tag = {"random": "exp2", "cv": "exp3"}[mode] if target == "label" else "exp4"
P = ROOT / cfg["data"]["processed_dir"]
run_tag = os.environ.get("PMGRAPH_TAG", "")
parts, preds = ROOT / f"outputs/tables/{tag}_parts", P / (f"{tag}_preds" + (f"_{run_tag}" if run_tag else ""))
if target == "label":
    parts.mkdir(parents=True, exist_ok=True)
preds.mkdir(parents=True, exist_ok=True)
hp = {**cfg["gnn"], **json.loads(os.environ.get("PMGRAPH_GNN", "{}"))}
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
mcols = list(data["market"].cols)
msc = StandardScaler().fit(Mraw)
Xm = msc.transform(Mraw).astype(np.float32)
midx = pd.Series(np.arange(len(Mraw)), index=data["market"].market_id.numpy())
fz, scalers = frozen_csrs(data)
del data
log("frozen graph", {r: len(c.src) for r, c in fz.items()})

Pop = pd.read_parquet(P / f"{split}_fk_pop.parquet")
Feat = pd.read_parquet(P / f"{split}_fk_feat.parquet")
obs = pd.read_parquet(P / f"{split}_fk_obs.parquet")
QMF = pd.read_parquet(P / f"{split}_fk_mf.parquet")
HALF = pd.read_parquet(P / f"{split}_fk_halves.parquet").query("scheme == 'week'") if target == "edge" else None
ASOF = hp.get("asof_markets", False)        # Gate C (17): as-of nodes for markets without a frozen node
if ASOF:
    AM = pd.read_parquet(P / f"{split}_fk_asof_mkt.parquet")
    AME = pd.read_parquet(P / f"{split}_fk_asof_mkt_edges.parquet")
    log("as-of market nodes", len(AM), "edges", len(AME))


def asof_graph(k: int):
    """Market features, frozen `held` CSR and (nid, market_id) -> node index, extended with k's as-of nodes."""
    if not ASOF:
        return Mraw, Xm, fz, None
    am = AM[AM.k == k].reset_index(drop=True)
    idx = len(Mraw) + np.arange(len(am))
    imap = pd.Series(idx, index=pd.MultiIndex.from_arrays([am.nid, am.market_id]))
    e = AME[AME.iid.isin(am.iid)]
    dst = pd.Series(idx, index=am.iid).loc[e.iid].to_numpy()
    a = (e[HELD_ATTRS].to_numpy(np.float32) - scalers["held"][0]) / scalers["held"][1]
    h = fz["held"]
    old_dst = np.repeat(np.arange(len(Mraw)), np.diff(h.indptr))
    fzk = dict(fz)
    fzk["held"] = CSR.build(np.r_[old_dst, dst], np.r_[h.src, e.src.to_numpy(np.int64)], np.r_[h.attr, a],
                            len(Mraw) + len(am))
    Mk = np.r_[Mraw, am[mcols].to_numpy(np.float32)]
    return Mk, msc.transform(Mk).astype(np.float32), fzk, imap


def edge_target(k: int, nids: np.ndarray) -> np.ndarray:
    """Centred z of half A, (x - mu) / sqrt(tau2 + phi s^2); NaN for wallets with < 10 bets in half A."""
    H = HALF[HALF.k == k].set_index("nid").reindex(nids)
    ok = (H["n0"] >= 10).to_numpy()
    assert ok.sum() > 100, f"no split halves for k={k}: run scripts/12_label_diagnostics.py {split} first"
    x, s = H["x0"].to_numpy(), H["s0"].to_numpy()
    v = variance_function(x[ok], s[ok])
    return np.where(ok, (x - v["mu"]) / np.sqrt(v["tau2"] + v["phi"] * s ** 2), np.nan)[:, None]


# ---------------------------------------------------------------- query graph for one k
def query_edges(k: int, nids: np.ndarray, t_k: pd.Series, imap=None) -> dict:
    """Relation -> (dst query idx, src frozen id, standardised attr) for the wallets `nids` at k.
    imap: (nid, market_id) -> as-of market node (Gate C) for markets without a frozen node."""
    qi = pd.Series(np.arange(len(nids)), index=nids)
    o = obs[(obs.rn <= k) & obs.nid.isin(nids)]
    E = {}
    c = o[o.cp < N].groupby(["nid", "cp"]).size()
    E["cp"] = [(c.index.get_level_values(0), c.index.get_level_values(1), np.log1p(c.to_numpy())[:, None])]
    h = held_edges_pd(o[o.market_id.isin(midx.index)])
    E["rev_held"] = [(h.nid, midx.loc[h.market_id].to_numpy(), h[HELD_ATTRS].to_numpy())]
    if imap is not None:
        h2 = held_edges_pd(o[~o.market_id.isin(midx.index)])
        h2 = h2[pd.MultiIndex.from_arrays([h2.nid, h2.market_id]).isin(imap.index)]
        src2 = imap.loc[pd.MultiIndex.from_arrays([h2.nid, h2.market_id])].to_numpy()
        d0, s0, a0 = E["rev_held"][0]
        E["rev_held"] = [(np.r_[np.asarray(d0), h2.nid.to_numpy()], np.r_[s0, src2],
                          np.r_[a0, h2[HELD_ATTRS].to_numpy()])]
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


def nbr_aggregates(E: dict, n: int, o: pd.DataFrame, nids: np.ndarray, Mk: np.ndarray) -> pd.DataFrame:
    """Unweighted means of frozen neighbours' features per relation, degrees, attachment shares."""
    cols = {}
    for r, V, names in [("cp", Xraw[NB].fillna(0).to_numpy(), NB), ("mf", Xraw[["hist_z"]].fillna(0).to_numpy(), ["hist_z"]),
                        ("rev_mf", Xraw[["hist_z"]].fillna(0).to_numpy(), ["hist_z"]), ("rev_held", Mk, None)]:
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
    Mk, Xmk, fzk, imap = asof_graph(k)
    E = query_edges(k, nids, pk.set_index("nid")["t_k"], imap)
    o = obs[(obs.rn <= k) & obs.nid.isin(nids)]
    NBR = nbr_aggregates(E, n, o, nids, Mk)
    g = Graph(x={"wallet": Xf, "market": Xmk}, frozen=fzk, xq=Fq,
              query={r: CSR.build(d, s, a, n) for r, (d, s, a) in E.items()})
    log(f"k={k}: n={n} (new {new.sum()}), query edges", {r: len(v[0]) for r, v in E.items()})
    strat = Y[:, 1] * 4 + Y[:, 0] * 2 + new
    Yt, obj = (Y, None) if target == "label" else (edge_target(k, nids), ["reg"])     # what the models are fit on
    splits = {}
    if mode == "random":
        for seed in cfg["eval"]["seeds"]:
            tr, te = train_test_split(np.arange(n), test_size=0.2, random_state=seed, stratify=strat)
            tr, va = train_test_split(tr, test_size=0.25, random_state=seed, stratify=strat[tr])
            splits[seed] = (tr, va, te)
    else:                                  # "seed" = fold index (also seeds the GNN)
        folds = [te for _, te in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(n), strat)]
        for f in range(5):
            va = folds[(f + 1) % 5]
            splits[f] = (np.setdiff1d(np.arange(n), np.r_[folds[f], va]), va, folds[f])
    for m in methods:
        rows, pr_rows = [], []
        tm = time.time()
        if m in ("xgb_own", "xgb_nbr"):    # trees need no early stopping: fit on train + val
            Fm = Fq_raw if m == "xgb_own" else np.c_[Fq_raw, NBR.to_numpy(np.float32)]
            xs = run_in_subprocess(Fm, Yt, {sd: (np.r_[tr, va], te) for sd, (tr, va, te) in splits.items()}, objectives=obj)
        for seed, (tr, va, te) in splits.items():
            if m == "hist":
                s = np.stack([fk["hist_profit_slog"].fillna(0).to_numpy(), fk["hist_z"].fillna(0).to_numpy()], 1)
                s = s if target == "label" else s[:, 1:]
            elif m in ("xgb_own", "xgb_nbr"):
                s = xs[seed]
            else:
                rels = ["cp"] if m == "sage" else ["cp", "mf", "rev_mf", "rev_held", "held"]
                s = train_hetero(g, rels, Yt, {"train": tr, "val": va}, seed, hp,
                                 log=lambda *a: log(f"k={k}", *a), objectives=obj)
            if target == "edge":
                pr_rows.append(pd.DataFrame({"seed": seed, "nid": nids[te], "new": new[te], "s_edge": s[te, 0]}))
                continue
            rows += metrics_rows(s, Y, te, new, seed=seed, k=k, method=m)
            pr_rows.append(pd.DataFrame({"seed": seed, "nid": nids[te], "new": new[te], "z": pk.z.to_numpy()[te],
                                         "n_bets": pk.n_bets.to_numpy()[te], "skill": Y[te, 1], "profitable": Y[te, 0],
                                         "s_skill": s[te, 1], "s_profitable": s[te, 0]}))
        pd.concat(pr_rows).assign(k=k, method=m).to_parquet(preds / f"{split}_k{k}_{m}.parquet", index=False)
        if target == "edge":
            log(f"k={k} {m:8s} {time.time()-tm:5.0f}s  edge scores saved")
            continue
        R = pd.DataFrame(rows)
        R.to_csv(parts / f"{split}_k{k}_{m}.csv", index=False)
        sk = R[(R.target == "skill")].groupby("pop").auprc.mean().round(3).to_dict()
        pr = R[(R.target == "profitable")].groupby("pop").auroc.mean().round(3).to_dict()
        log(f"k={k} {m:8s} {time.time()-tm:5.0f}s  skill AUPRC {sk}  profit AUROC {pr}")
