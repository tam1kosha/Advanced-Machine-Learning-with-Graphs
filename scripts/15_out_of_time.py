"""Step 15 (iteration 4, next step 2): out-of-time test of the edge target. XGBoost only.

Fit on every labelled wallet of one split, score the wallets of a later split. With fit = train
(cutoff 2025-01-01, labels Jan-Jun 2025) and score = test (cutoff 2025-07-01, labels Jul-Dec 2025)
every training label is resolved before the first scored fill, so nothing of the scored period,
neither outcomes nor a period-specific style premium, can enter the model.
Models (first-k features of 07, per k):
  reg_zc    regression on the centred z of the whole label, (x - mu) / sqrt(tau2 + phi s^2)
  cls_z2    classifier for z > 2 (iterations 1-3)
  hist      no training: history-only z of the scored wallet
Scores are averaged over eval.seeds as percentile ranks and judged on the scored split's realised
excess edge (pmgraph.label_diag.EDGE_METRICS, wallet bootstrap; sd over seeds of each estimate).
Compare with the in-period numbers of 13 (exp4_{split}_targets.csv): a large drop means the
in-period signal was a style that paid in that window, not a wallet trait.
Needs 01, 03, 04, 07 for both splits.  -> outputs/tables/exp5_{fit}_to_{score}_oot.csv
usage: python scripts/15_out_of_time.py [fit split] [score split] [ks,comma] [n_boot]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pmgraph import ROOT, load_config  # noqa: E402
from pmgraph.label_diag import EDGE_METRICS, boot_edge, edge_metrics, variance_function  # noqa: E402
from pmgraph.tabular import run_in_subprocess  # noqa: E402

cfg = load_config()
fit_split = sys.argv[1] if len(sys.argv) > 1 else "train"
score_split = sys.argv[2] if len(sys.argv) > 2 else "test"
ks = [int(k) for k in sys.argv[3].split(",")] if len(sys.argv) > 3 else cfg["eval"]["k_buckets"]
B = int(sys.argv[4]) if len(sys.argv) > 4 else 300
P = ROOT / cfg["data"]["processed_dir"]
seeds = cfg["eval"]["seeds"]
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)  # noqa: E731
if fit_split == score_split:
    log("WARNING: fit and score split are the same: in-sample numbers, a plumbing check only")


def load(split: str, k: int):
    """Labelled wallets of one split at k: feature matrix, excess edge x, null sd s, new flag, hist z."""
    pk = pd.read_parquet(P / f"{split}_fk_pop.parquet").query("k == @k").reset_index(drop=True)
    f = pd.read_parquet(P / f"{split}_fk_feat.parquet").query("k == @k").set_index("nid").reindex(pk.nid).drop(columns="k")
    X = f.assign(hist_missing=f["hist_z"].isna().astype(float)).fillna(0.0).assign(is_new=pk.new.to_numpy(float))
    return X, (pk["T"] - pk["mu0"]).to_numpy(), pk["sd0"].to_numpy(), pk.new.to_numpy(), f["hist_z"].fillna(0).to_numpy()


rows = []
for k in ks:
    Xa, xa, sa, _, _ = load(fit_split, k)
    Xb, xb, sb, new, hist = load(score_split, k)
    assert list(Xa.columns) == list(Xb.columns), "the two splits have different feature columns"
    va, vb = variance_function(xa, sa), variance_function(xb, sb)
    na, nb = len(xa), len(xb)
    ya = np.stack([(xa - va["mu"]) / np.sqrt(va["tau2"] + va["phi"] * sa ** 2), (xa / sa > 2).astype(float)], 1)
    Y = np.r_[ya, np.full((nb, 2), np.nan)]                       # the scored split has no targets
    S = run_in_subprocess(np.r_[Xa.to_numpy(np.float32), Xb.to_numpy(np.float32)], Y,
                          {sd: (np.arange(na), na + np.arange(nb)) for sd in seeds}, objectives=["reg", "binary"])
    per_seed = {name: [S[sd][na:, j] for sd in seeds] for j, name in enumerate(["reg_zc", "cls_z2"])}
    wb, zb = 1 / (vb["tau2"] + vb["phi"] * sb ** 2), xb / sb
    log(f"k={k}: fit on {na} {fit_split} wallets (mu {100*va['mu']:+.2f}c), scored {nb} {score_split} wallets "
        f"(mu {100*vb['mu']:+.2f}c, new {int(new.sum())})")
    for pop, m in [("new", new), ("existing", ~new)]:
        sc = {name: np.mean([rankdata(s[m]) / m.sum() for s in ss], 0) for name, ss in per_seed.items()}
        sc["hist"] = hist[m]
        out = boot_edge(sc, xb[m], wb[m], zb[m], B=B, seed=k)
        for name, (est, lo, hi) in out.items():
            sd = (np.std([edge_metrics(s[m], xb[m], wb[m], zb[m]) for s in per_seed[name]], 0)
                  if name in per_seed else np.zeros(len(EDGE_METRICS)))
            rows += [dict(k=k, fit=fit_split, score=score_split, pop=pop, model=name, metric=met, n=int(m.sum()),
                          est=est[j], lo=lo[j], hi=hi[j], sd_seeds=sd[j]) for j, met in enumerate(EDGE_METRICS)]
R = pd.DataFrame(rows)
R.to_csv(ROOT / f"outputs/tables/exp5_{fit_split}_to_{score_split}_oot.csv", index=False)
for met, scale in [("wcorr_edge", 1), ("top_minus_bottom", 100), ("auprc_hi", 1)]:
    for pop in ("new", "existing"):
        q = R[(R.metric == met) & (R["pop"] == pop)]
        c = lambda v: (scale * q.pivot(index="model", columns="k", values=v)).round(3).astype(str)  # noqa: E731
        log(f"{met}{' (cents)' if scale == 100 else ''}, {pop} wallets of {score_split}, model fit on {fit_split}\n" +
            (c("est") + " [" + c("lo") + ", " + c("hi") + "]").to_string())
ref = ROOT / f"outputs/tables/exp4_{fit_split}_targets.csv"
if ref.exists():
    q = pd.read_csv(ref).query("metric == 'wcorr_edge' and target in ['reg_zc', 'cls_z2', 'hist'] and pop in ['new', 'existing']")
    log(f"for comparison, in-period ({fit_split}, held-out half, script 13): wcorr_edge\n" +
        q.pivot(index=["pop", "target"], columns="k", values="est").round(3).to_string())
log("done")
