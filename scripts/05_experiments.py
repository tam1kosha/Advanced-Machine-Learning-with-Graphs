"""Step 5: first experiments (single cutoff, transductive wallet split, 5 seeds).

Targets (label window, wallets with >= min_resolved_bets):
  profitable : profit > 0                       -> AUROC
  skill      : skill z > 2 (top ~2%)             -> AUPRC (main rare-class metric)
Methods: history-only test, XGBoost (own), XGBoost (own + 1-hop neighbour aggregates),
spectral embedding + LR, label propagation, GraphSAGE.
Results are also broken down by pre-cutoff activity (number of fills before the cutoff).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pmgraph import ROOT, load_config  # noqa: E402
from pmgraph.features import sym_norm  # noqa: E402
from pmgraph.models import (label_propagation, lr_scores, matrix, own_cols,  # noqa: E402
                            spectral_embedding, train_sage, xgb_scores)

cfg = load_config()
split = sys.argv[1] if len(sys.argv) > 1 else "train"
methods = sys.argv[2].split(",") if len(sys.argv) > 2 else ["hist", "xgb_own", "xgb_nbr", "spectral_lr", "labelprop", "sage"]
P_ = ROOT / cfg["data"]["processed_dir"]
out_dir = ROOT / "outputs/tables"
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)  # noqa: E731

X = pd.read_parquet(P_ / f"{split}_features.parquet")
A = sp.load_npz(P_ / f"{split}_adj.npz")
lab = pd.read_parquet(P_ / f"{split}_labels.parquet")
TARGETS = {"profitable": lab["profitable"].astype(int), "skill": (lab["z"] > 2).astype(int)}

nodes = np.flatnonzero(X["in_label"].to_numpy())
wal = X.index[nodes]
Y = np.stack([TARGETS[t].loc[wal].to_numpy() for t in TARGETS], 1)
n_fills_pre = np.expm1(X["log_n_fills"].to_numpy()[nodes])
bucket = pd.cut(n_fills_pre, [0, 3, 10, 50, np.inf], labels=["1-3", "4-10", "11-50", ">50"]).astype(str)
log("labelled nodes", len(nodes), "pos rates", Y.mean(0).round(4), "buckets", pd.Series(bucket).value_counts().to_dict())

cols_own = own_cols(X)
cols_nbr = cols_own + [c for c in X.columns if c.startswith(("nbr_", "deg"))]
F_own, F_nbr = matrix(X, cols_own), matrix(X, cols_nbr)
S = sym_norm(A)
Prow = sp.diags(1 / np.maximum(np.asarray(A.sum(1)).ravel(), 1e-12)) @ A
emb = None
if "spectral_lr" in methods:
    emb = spectral_embedding(S, 32)
    log("spectral embedding done")

rows = []
for seed in cfg["eval"]["seeds"]:
    strat = Y[:, 1] * 2 + Y[:, 0]
    tr, te = train_test_split(np.arange(len(nodes)), test_size=0.2, random_state=seed, stratify=strat)
    tr, va = train_test_split(tr, test_size=0.2, random_state=seed, stratify=strat[tr])
    g_tr, g_va, g_te = nodes[tr], nodes[va], nodes[te]
    scores = {}
    if "hist" in methods:  # no training: history-only test statistic / realised profit
        hz = X["hist_z"].fillna(0).to_numpy()[g_te]
        hp = X["hist_profit_slog"].fillna(0).to_numpy()[g_te]
        scores["hist"] = np.stack([hp, hz], 1)
    for name, Fm in [("xgb_own", F_own), ("xgb_nbr", F_nbr)]:
        if name in methods:
            scores[name] = np.stack([xgb_scores(Fm[g_tr], Y[tr, j], Fm[g_te], seed) for j in range(2)], 1)
    if "spectral_lr" in methods:
        scores["spectral_lr"] = np.stack([lr_scores(emb[g_tr], Y[tr, j], emb[g_te]) for j in range(2)], 1)
    if "labelprop" in methods:
        scores["labelprop"] = np.stack([label_propagation(S, g_tr, Y[tr, j].astype(np.float32))[g_te]
                                        for j in range(2)], 1)
    if "sage" in methods:
        Yfull = np.zeros((len(X), 2), np.float32)
        Yfull[nodes] = Y
        s = train_sage(F_own, Prow, {"train": g_tr, "val": g_va}, Yfull, seed, log=log)
        scores["sage"] = s[g_te]
    for m, s in scores.items():
        for j, t in enumerate(TARGETS):
            for b in ["all"] + sorted(set(bucket)):
                mask = np.ones(len(te), bool) if b == "all" else (bucket[te] == b)
                yt = Y[te, j][mask]
                if yt.sum() < 3 or yt.sum() == len(yt):
                    continue
                rows.append(dict(seed=seed, method=m, target=t, bucket=b, n=int(mask.sum()),
                                 pos=float(yt.mean()), auroc=roc_auc_score(yt, s[mask, j]),
                                 auprc=average_precision_score(yt, s[mask, j])))
    log(f"seed {seed} done:", {m: round(r["auprc"], 3) for r in rows[-200:] for m in [r["method"]]
                               if r["seed"] == seed and r["target"] == "skill" and r["bucket"] == "all"})

R = pd.DataFrame(rows)
R.to_csv(out_dir / f"exp1_{split}_raw.csv", index=False)
agg = (R.groupby(["target", "bucket", "method"])
         .agg(n=("n", "first"), pos=("pos", "mean"), auroc=("auroc", "mean"), auroc_sd=("auroc", "std"),
              auprc=("auprc", "mean"), auprc_sd=("auprc", "std")).round(4).reset_index())
agg.to_csv(out_dir / f"exp1_{split}_summary.csv", index=False)
with pd.option_context("display.width", 200, "display.max_rows", 200):
    print(agg[agg.bucket == "all"].to_string(index=False))
    print(agg[agg.target == "skill"].pivot(index="method", columns="bucket", values="auprc").round(3))
    print(agg[agg.target == "profitable"].pivot(index="method", columns="bucket", values="auroc").round(3))
