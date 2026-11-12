"""Step 11 (iteration 3): pooled out-of-fold skill metrics with bootstrap CIs (NOTES: rare-class power).

Needs 09 run in cv mode (every wallet scored once). Labels are unchanged (skill = label z > 2); the
added metrics use more of the label: precision@100 and the Spearman correlation with the label z
(see pmgraph/rank_eval.py), also within label bet-count buckets (spearman_z_strat), since z grows with
activity under real skill. Scores are within-fold percentile ranks, pooled over the 5 folds.
Outputs: outputs/tables/exp3_{split}_pooled.csv (estimate + 95% CI per method), exp3_{split}_paired.csv
(graph model minus baseline, paired bootstrap), exp3_{split}_buckets.csv (Spearman per bet bucket,
point estimates), outputs/figures/exp3_{split}_detection.png.
Claim check (stricter than 10): a graph model beats both baselines on new wallets at every k <= 10
with the paired 95% CI above 0.
usage: python scripts/11_rank_report.py [split] [n_boot]
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
from pmgraph.rank_eval import BET_BUCKETS, METRICS, bet_bucket, bootstrap, fold_ranks, spearman, summarise  # noqa: E402

cfg = load_config()
split = sys.argv[1] if len(sys.argv) > 1 else "train"
B = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
T, F = ROOT / "outputs/tables", ROOT / "outputs/figures"
files = sorted((ROOT / cfg["data"]["processed_dir"] / "exp3_preds").glob(f"{split}_k*_*.parquet"))
assert files, "no exp3 predictions: run scripts/09_first_k_experiments.py <split> all <ks> cv"
D = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
assert not D.duplicated(["k", "method", "nid"]).any(), "a wallet is scored twice: 09 was not run in cv mode"

BASE, GRAPH = ["hist", "xgb_own"], ["xgb_nbr", "sage", "hgnn"]
ORDER = [m for m in BASE + GRAPH if m in set(D.method)]
pooled, paired, buckets = [], [], []
for (k, pop), g in D.assign(pop=np.where(D.new, "new", "existing")).groupby(["k", "pop"]):
    # same wallets for every method: align on nid
    W = g.pivot(index="nid", columns="method", values="s_skill").dropna()
    lab = g.drop_duplicates("nid").set_index("nid").loc[W.index]
    fold = g[g.method == ORDER[0]].set_index("nid").loc[W.index, "seed"].to_numpy()
    scores = {m: fold_ranks(W[m].to_numpy(), fold) for m in ORDER if m in W}
    y, z = lab["skill"].to_numpy(), lab["z"].to_numpy()
    gb = bet_bucket(lab["n_bets"].to_numpy())
    bs = bootstrap(y, z, scores, B=B, seed=int(k), g=gb)
    names = [f"{a}-{c - 1}" for a, c in zip(BET_BUCKETS, BET_BUCKETS[1:])] + [f">={BET_BUCKETS[-1]}"]
    for b_i, name in enumerate(names, start=1):
        mk = gb == b_i
        for m, s in scores.items():
            buckets.append(dict(k=k, pop=pop, bucket=name, method=m, n=int(mk.sum()), pos=y[mk].mean(),
                                spearman_z=spearman(z[mk], s[mk]) if mk.sum() >= 20 else np.nan))
    for m, (est, reps) in bs.items():
        for met, r in summarise(est, reps).items():
            pooled.append(dict(k=k, pop=pop, method=m, metric=met, n=len(y), pos=y.mean(), **r))
    for met_j, met in enumerate(METRICS):
        best = max([b for b in BASE if b in bs], key=lambda b: bs[b][0][met_j])
        for m in [m for m in GRAPH if m in bs]:
            for b in [b for b in BASE if b in bs]:
                r = summarise(bs[m][0] - bs[b][0], bs[m][1] - bs[b][1])[met]
                paired.append(dict(k=k, pop=pop, method=m, baseline=b, is_best=b == best, metric=met, **r))
    print(f"k={k} {pop}: n={len(y)} pos={y.sum()}", flush=True)

S, Pd = pd.DataFrame(pooled), pd.DataFrame(paired)
S.round(4).to_csv(T / f"exp3_{split}_pooled.csv", index=False)
Pd.round(4).to_csv(T / f"exp3_{split}_paired.csv", index=False)
Bk = pd.DataFrame(buckets)
Bk.round(4).to_csv(T / f"exp3_{split}_buckets.csv", index=False)

fmt = lambda d: d.est.map("{:.3f}".format) + " [" + d.lo.map("{:.3f}".format) + ", " + d.hi.map("{:.3f}".format) + "]"  # noqa: E731
with pd.option_context("display.width", 250, "display.max_rows", 400):
    for met in METRICS:
        for pop in ["new", "existing"]:
            s = S[(S.metric == met) & (S["pop"] == pop)]
            print(f"\n{met}, {pop} wallets, pooled, 95% CI (n = {s.groupby('k').n.first().to_dict()}, "
                  f"base rate {s.groupby('k').pos.first().round(4).to_dict()})")
            print(s.assign(v=fmt(s)).pivot(index="method", columns="k", values="v").reindex(ORDER).to_string())
    d = Pd[(Pd["pop"] == "new") & Pd.is_best]
    print("\nnew wallets: graph model minus best baseline, paired 95% CI:")
    print(d.assign(v=fmt(d)).pivot_table(index=["metric", "method"], columns="k", values="v", aggfunc="first").to_string())
    small = Pd[(Pd["pop"] == "new") & (Pd.k <= 10)]
    for met in METRICS:
        ok = small[small.metric == met].groupby("method").lo.apply(lambda x: bool((x > 0).all()))
        print(f"claim ({met}: CI > 0 vs both baselines at every k <= 10):", ok.to_dict(), "->",
              "HOLDS" if ok.any() else "NOT MET")
    print("\nSpearman(score, z) per label bet bucket, new wallets (point estimates):")
    bn = Bk[Bk["pop"] == "new"]
    print(bn.pivot_table(index=["bucket", "method"], columns="k", values="spearman_z").round(3).to_string())

style = {"hist": ("#888", "--"), "xgb_own": ("#f0ad4e", "-"), "xgb_nbr": ("#d9534f", "-"),
         "sage": ("#5cb85c", "-"), "hgnn": ("#2a6fdb", "-")}
fig, axes = plt.subplots(2, len(METRICS), figsize=(5 * len(METRICS), 7.5), sharex=True)
for i, pop in enumerate(["new", "existing"]):
    for j, met in enumerate(METRICS):
        ax, s = axes[i, j], S[(S["pop"] == pop) & (S.metric == met)]
        for m, (c, ls) in style.items():
            d = s[s.method == m].sort_values("k")
            if len(d):
                ax.errorbar(d.k, d.est, yerr=[d.est - d.lo, d.hi - d.est], color=c, ls=ls, marker="o", ms=4,
                            capsize=2, label=m)
        if not met.startswith("spearman"):
            base = s.groupby("k").pos.first()
            ax.plot(base.index, base.values, color="#bbb", lw=1, ls=":", label="base rate")
        else:
            ax.axhline(0, color="#bbb", lw=1, ls=":")
        ax.set(xscale="log", title=f"{pop} wallets: {met}", xlabel="k (observed fills)" if i else None)
        ax.set_xticks(sorted(S.k.unique()), [str(x) for x in sorted(S.k.unique())])
axes[0, 0].legend(frameon=False, fontsize=8)
fig.suptitle("Skill (label z > 2): pooled out-of-fold, 5-fold CV, 95% wallet-bootstrap CI")
fig.tight_layout()
fig.savefig(F / f"exp3_{split}_detection.png", dpi=150)
print("wrote", F / f"exp3_{split}_detection.png")
