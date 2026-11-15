"""Step 13 (iteration 4): which skill target should the models be trained on? XGBoost only.

Every candidate target is built from half A of a wallet's label bets and judged on the realised
excess edge of half B of held-out wallets, which no candidate is trained on. Halves come from 12
(default scheme `week`: markets closing in alternating weeks, so no outcome of a B week enters any
training target: a cheap stand-in for an out-of-time test). Same 5 CV folds as 09 cv; XGBoost on
the first-k features (fk_feat), fit on the 4 non-test folds.
x = excess edge per bet, s = null sd, z = x / s, (mu, tau2, phi) = variance function of half A.
Targets
  cls_z2_full  z > 2 on the whole label (iterations 1-3; sees B-week outcomes of training wallets)
  cls_z2       z_A > 2
  reg_z        z_A
  reg_edge_w   x_A, weights 1 / (tau2 + phi s_A^2)
  reg_eb       empirical-Bayes edge  mu + tau2 / (tau2 + phi s_A^2) * (x_A - mu)
  reg_zc       centred z  (x_A - mu) / sqrt(tau2 + phi s_A^2)
  hist         no training: history-only z on the observed fills (reference)
Metrics (pmgraph.label_diag.EDGE_METRICS) on half B, pooled over folds, wallet bootstrap; paired
difference of every target to cls_z2.  -> outputs/tables/exp4_{split}_targets.csv
Populations: new, existing, and the same without "sure-shot" wallets (new_nosure, existing_nosure:
under half of the observed fills bought above 0.90). Buying near-certain outcomes earns a small
steady premium, and that one style is most of what the first fills predict.
usage: python scripts/13_target_bakeoff.py [split] [ks,comma] [scheme] [n_boot]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pmgraph import ROOT, load_config  # noqa: E402
from pmgraph.label_diag import EDGE_METRICS, boot_edge, variance_function  # noqa: E402
from pmgraph.rank_eval import fold_ranks  # noqa: E402
from pmgraph.tabular import run_in_subprocess  # noqa: E402

cfg = load_config()
split = sys.argv[1] if len(sys.argv) > 1 else "train"
ks = [int(k) for k in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 10, 50]
scheme = sys.argv[3] if len(sys.argv) > 3 else "week"
B = int(sys.argv[4]) if len(sys.argv) > 4 else 300
MIN_HALF = 10                                # bets a half needs to give a target / an evaluation
P = ROOT / cfg["data"]["processed_dir"]
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)  # noqa: E731
Pop = pd.read_parquet(P / f"{split}_fk_pop.parquet")
Feat = pd.read_parquet(P / f"{split}_fk_feat.parquet")
Hv = pd.read_parquet(P / f"{split}_fk_halves.parquet")
Hv = Hv[Hv.scheme == scheme]
REF = "cls_z2"
rows = []
for k in ks:
    pk = Pop[Pop.k == k].reset_index(drop=True)
    nids, new, n = pk.nid.to_numpy(), pk.new.to_numpy(), len(pk)
    f = Feat[Feat.k == k].set_index("nid").reindex(nids).drop(columns="k")
    X = f.assign(hist_missing=f["hist_z"].isna().astype(float)).fillna(0.0).assign(is_new=new.astype(float))
    H = Hv[Hv.k == k].set_index("nid").reindex(nids)
    okA, okB = (H["n0"] >= MIN_HALF).to_numpy(), (H["n1"] >= MIN_HALF).to_numpy()
    xA, sA, xB, sB = (H[c].to_numpy() for c in ("x0", "s0", "x1", "s1"))
    vA, vB = variance_function(xA[okA], sA[okA]), variance_function(xB[okB], sB[okB])
    totA = vA["tau2"] + vA["phi"] * sA ** 2
    nanA = lambda y: np.where(okA, y, np.nan)  # noqa: E731
    targets = {                              # name: (y, objective, weights)
        "cls_z2_full": (pk["skill"].to_numpy(float), "binary", None),
        "cls_z2": (nanA((xA / sA > 2).astype(float)), "binary", None),
        "reg_z": (nanA(xA / sA), "reg", None),
        "reg_edge_w": (nanA(xA), "reg", 1 / totA),
        "reg_eb": (nanA(vA["mu"] + vA["tau2"] / totA * (xA - vA["mu"])), "reg", None),
        "reg_zc": (nanA((xA - vA["mu"]) / np.sqrt(totA)), "reg", None),
    }
    Y = np.stack([t[0] for t in targets.values()], 1)
    W = np.stack([np.ones(n) if t[2] is None else np.nan_to_num(t[2]) for t in targets.values()], 1)
    strat = pk["skill"].to_numpy() * 4 + pk["profitable"].to_numpy() * 2 + new      # the folds of 09 cv
    folds = [te for _, te in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(n), strat)]
    fold = np.empty(n, int)
    for i, te in enumerate(folds):
        fold[te] = i
    S = run_in_subprocess(X.to_numpy(np.float32), Y, {i: (np.setdiff1d(np.arange(n), te), te) for i, te in enumerate(folds)},
                          objectives=[t[1] for t in targets.values()], W=W)
    score = np.zeros(Y.shape, np.float32)
    for i, te in enumerate(folds):
        score[te] = S[i][te]
    sc = {name: fold_ranks(score[:, j], fold) for j, name in enumerate(targets)}
    sc["hist"] = fold_ranks(f["hist_z"].fillna(0).to_numpy(), fold)
    wB = 1 / (vB["tau2"] + vB["phi"] * sB ** 2)
    log(f"k={k}: n={n}, target on {okA.sum()}, evaluated on {okB.sum()} | half A mu {100*vA['mu']:+.2f}c "
        f"tau {100*np.sqrt(vA['tau2']):.2f}c phi {vA['phi']:.2f}")
    sure = (f["frac_sureshot"].fillna(0) >= 0.5).to_numpy()
    for pop, m in [("new", new & okB), ("existing", ~new & okB), ("new_nosure", new & okB & ~sure),
                   ("existing_nosure", ~new & okB & ~sure)]:
        out = boot_edge({a: r[m] for a, r in sc.items()}, xB[m], wB[m], (xB / sB)[m], B=B, seed=k, reps=True)
        for name, (est, lo, hi, R) in out.items():
            d = R - out[REF][3]
            dlo, dhi = np.nanquantile(d, .025, 0), np.nanquantile(d, .975, 0)
            for j, met in enumerate(EDGE_METRICS):
                rows.append(dict(k=k, scheme=scheme, pop=pop, target=name, metric=met, n=int(m.sum()), est=est[j],
                                 lo=lo[j], hi=hi[j], diff_ref=est[j] - out[REF][0][j], diff_lo=dlo[j], diff_hi=dhi[j]))
R = pd.DataFrame(rows)
R.to_csv(ROOT / f"outputs/tables/exp4_{split}_targets.csv", index=False)
print(R.drop_duplicates(["k", "pop"])[["k", "pop", "n"]].to_string(index=False))
for met, scale in [("top_minus_bottom", 100), ("wcorr_edge", 1), ("auprc_hi", 1)]:
    for pop in ("new", "new_nosure", "existing", "existing_nosure"):
        q = R[(R.metric == met) & (R["pop"] == pop)]
        tab = (scale * q.pivot(index="target", columns="k", values="est")).round(3).astype(str) + " [" + \
              (scale * q.pivot(index="target", columns="k", values="lo")).round(3).astype(str) + ", " + \
              (scale * q.pivot(index="target", columns="k", values="hi")).round(3).astype(str) + "]"
        log(f"{met}{' (cents)' if scale == 100 else ''}, {pop} wallets, half B\n" + tab.loc[list(q.target.unique())].to_string())
log("done")
