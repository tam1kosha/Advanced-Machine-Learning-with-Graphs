"""Step 14 (iteration 4): the graph claim on the edge target.

Needs `09 ... cv edge`: every method is fit on the centred z of half A of the label bets (markets
closing in even weeks) and scores every wallet once. Here the scores are judged on the realised
excess edge of half B (odd weeks) of the same held-out wallets, pooled over the 5 folds as
within-fold percentile ranks (metrics: pmgraph.label_diag.EDGE_METRICS, wallet bootstrap).
Paired differences: every graph model minus XGBoost on own features and minus the better baseline.
Claim check (proposal §5 on this target): a graph model beats both baselines (history-only, XGBoost
own) on new wallets at every k <= 10, paired 95% CI of the weighted edge correlation above 0.
Outputs: outputs/tables/exp4_{split}_graph.csv, exp4_{split}_graph_paired.csv,
outputs/figures/exp4_{split}_graph.png.
A third argument names a tagged run of 09 (env PMGRAPH_TAG): scores are read from exp4_preds_<tag>/
and the outputs get the suffix _<tag>.
usage: python scripts/14_graph_on_edge.py [split] [n_boot] [tag]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pmgraph import ROOT, load_config  # noqa: E402
from pmgraph.label_diag import EDGE_METRICS, boot_edge, variance_function  # noqa: E402
from pmgraph.rank_eval import fold_ranks  # noqa: E402

cfg = load_config()
split = sys.argv[1] if len(sys.argv) > 1 else "train"
B = int(sys.argv[2]) if len(sys.argv) > 2 else 500
P, T = ROOT / cfg["data"]["processed_dir"], ROOT / "outputs/tables"
sfx = f"_{sys.argv[3]}" if len(sys.argv) > 3 else ""
files = sorted((P / f"exp4_preds{sfx}").glob(f"{split}_k*_*.parquet"))
assert files, "no edge-target scores: run scripts/09_first_k_experiments.py <split> <methods> <ks> cv edge"
D = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
H = pd.read_parquet(P / f"{split}_fk_halves.parquet").query("scheme == 'week'")
BASE, GRAPH = ["hist", "xgb_own"], ["xgb_nbr", "sage", "hgnn"]
ORDER = [m for m in BASE + GRAPH if m in set(D.method)]
MAIN = EDGE_METRICS.index("wcorr_edge")
rows, paired = [], []
for k, g in D.groupby("k"):
    S = g.pivot(index="nid", columns="method", values="s_edge").dropna()      # wallets scored by every method run at k
    meth = [m for m in ORDER if m in S]
    one = g[g.method == meth[0]].set_index("nid").loc[S.index]
    h = H[H.k == k].set_index("nid").reindex(S.index)
    ok = (h["n1"] >= 10).to_numpy()
    xB, sB = h["x1"].to_numpy(), h["s1"].to_numpy()
    v = variance_function(xB[ok], sB[ok])
    w = 1 / (v["tau2"] + v["phi"] * sB ** 2)
    for pop, m_ in [("new", one["new"].to_numpy() & ok), ("existing", ~one["new"].to_numpy() & ok)]:
        sc = {m: fold_ranks(S[m].to_numpy(), one["seed"].to_numpy())[m_] for m in meth}
        out = boot_edge(sc, xB[m_], w[m_], (xB / sB)[m_], B=B, seed=int(k), reps=True)
        for m, (est, lo, hi, _) in out.items():
            rows += [dict(k=k, pop=pop, method=m, metric=met, n=int(m_.sum()), est=est[j], lo=lo[j], hi=hi[j])
                     for j, met in enumerate(EDGE_METRICS)]
        best = max([b for b in BASE if b in out], key=lambda b: out[b][0][MAIN])
        for m in [m for m in GRAPH if m in out]:
            for ref in dict.fromkeys(["xgb_own", best]):
                if ref not in out:
                    continue
                d = out[m][3] - out[ref][3]
                lo, hi = np.nanquantile(d, [.025, .975], axis=0)
                paired += [dict(k=k, pop=pop, method=m, vs=ref, best_baseline=best, metric=met,
                                diff=out[m][0][j] - out[ref][0][j], lo=lo[j], hi=hi[j]) for j, met in enumerate(EDGE_METRICS)]
R, Q = pd.DataFrame(rows), pd.DataFrame(paired)
R.to_csv(T / f"exp4_{split}_graph{sfx}.csv", index=False)
Q.to_csv(T / f"exp4_{split}_graph_paired{sfx}.csv", index=False)

cell = lambda d, v, sc=1: (sc * d.pivot(index="method", columns="k", values=v)).round(3)  # noqa: E731
for met, sc in [("wcorr_edge", 1), ("top_minus_bottom", 100), ("auprc_hi", 1)]:
    for pop in ("new", "existing"):
        q = R[(R.metric == met) & (R["pop"] == pop)]
        tab = cell(q, "est", sc).astype(str) + " [" + cell(q, "lo", sc).astype(str) + ", " + cell(q, "hi", sc).astype(str) + "]"
        print(f"\n{met}{' (cents)' if sc == 100 else ''}, {pop} wallets, held-out half\n" +
              tab.loc[[m for m in ORDER if m in tab.index]].to_string())
if len(Q):
    q = Q[(Q.metric == "wcorr_edge") & (Q.vs == "xgb_own")]
    for pop in ("new", "existing"):
        d = q[q["pop"] == pop]
        tab = cell(d, "diff").astype(str) + " [" + cell(d, "lo").astype(str) + ", " + cell(d, "hi").astype(str) + "]"
        print(f"\nweighted edge correlation, graph model minus XGBoost own, {pop} wallets (paired 95% CI)\n" + tab.to_string())
    c = Q[(Q.metric == "wcorr_edge") & (Q["pop"] == "new") & (Q.k <= 10)]
    for m in [m for m in GRAPH if m in set(c.method)]:
        cm = c[c.method == m]
        wins = cm.groupby("k")["lo"].min() > 0            # above xgb_own and the better baseline
        print(f"claim check {m}: beats both baselines on new wallets at k = {list(wins.index[wins])} of {list(wins.index)}"
              f" -> {'MET' if wins.all() else 'NOT MET'}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
COL = {"hist": "grey", "xgb_own": "#4c72b0", "xgb_nbr": "#8172b3", "sage": "#dd8452", "hgnn": "#55a868"}
for a, pop in zip(ax, ("new", "existing")):
    for i, m in enumerate(ORDER):
        q = R[(R.metric == "wcorr_edge") & (R["pop"] == pop) & (R.method == m)].sort_values("k")
        a.errorbar(q.k * (1 + .03 * (i - 2)), q.est, yerr=[q.est - q.lo, q.hi - q.est], color=COL[m], marker="o",
                   capsize=2, label=m)
    a.axhline(0, color="k", lw=.8)
    a.set_xscale("log")
    a.set_xticks(sorted(R.k.unique()))
    a.set_xticklabels(sorted(R.k.unique()))
    a.set_xlabel("k observed fills")
    a.set_title(f"{pop} wallets")
ax[0].set_ylabel("weighted correlation with held-out edge (95% CI)")
ax[0].legend(fontsize=8)
fig.suptitle("Edge target: graph models vs baselines, judged on the held-out half of the label")
fig.tight_layout()
fig.savefig(ROOT / f"outputs/figures/exp4_{split}_graph{sfx}.png", dpi=130)
