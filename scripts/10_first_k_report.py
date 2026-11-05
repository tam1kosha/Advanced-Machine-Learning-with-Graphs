"""Step 10 (iteration 2): aggregate 09's parts into tables, paired differences and the detection curve.

Outputs: outputs/tables/exp2_{split}_raw.csv, exp2_{split}_summary.csv, exp2_{split}_paired.csv,
         outputs/figures/exp2_{split}_detection.png (rows: new / existing wallets).
Claim check (proposal §5): a graph model (xgb_nbr, sage, hgnn) beats BOTH baselines (hist, xgb_own)
on new wallets at every k <= 10: mean paired difference > 0 and >= 4/5 seed wins against each.
usage: python scripts/10_first_k_report.py [split]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
split = sys.argv[1] if len(sys.argv) > 1 else "train"
T = ROOT / "outputs/tables"
R = pd.concat([pd.read_csv(f) for f in sorted((T / "exp2_parts").glob(f"{split}_k*_*.csv"))], ignore_index=True)
R["metric"] = R.target.map({"skill": "auprc", "profitable": "auroc"})
R["value"] = [r.auprc if r.metric == "auprc" else r.auroc for r in R.itertuples()]
R.to_csv(T / f"exp2_{split}_raw.csv", index=False)

BASE, GRAPH = ["hist", "xgb_own"], ["xgb_nbr", "sage", "hgnn"]
ORDER = BASE + GRAPH
S = (R.groupby(["pop", "target", "k", "method"])
       .agg(n=("n", "mean"), pos=("pos", "mean"), mean=("value", "mean"), sd=("value", "std"), seeds=("seed", "nunique"))
       .round(4).reset_index())
S.to_csv(T / f"exp2_{split}_summary.csv", index=False)

rows = []
for (pop, tgt, k), g in R.groupby(["pop", "target", "k"]):
    w = g.pivot(index="seed", columns="method", values="value")
    have_b = [b for b in BASE if b in w]
    if not have_b:
        continue
    best = max(have_b, key=lambda b: w[b].mean())
    for m in [m for m in GRAPH if m in w]:
        r = dict(pop=pop, target=tgt, k=k, method=m, best_baseline=best)
        for b in have_b + ["best"]:
            d = w[m] - w[best if b == "best" else b]
            r[f"d_{b}"], r[f"d_{b}_sd"], r[f"wins_{b}"] = d.mean(), d.std(), int((d > 0).sum())
        r["beats_both"] = all(r[f"d_{b}"] > 0 and r[f"wins_{b}"] >= 4 for b in BASE if b in w)
        rows.append(r)
D = pd.DataFrame(rows).round(4)
D.to_csv(T / f"exp2_{split}_paired.csv", index=False)

with pd.option_context("display.width", 220, "display.max_rows", 400):
    for tgt, metric in [("skill", "AUPRC"), ("profitable", "AUROC")]:
        for pop in ["new", "existing"]:
            s = S[(S.target == tgt) & (S["pop"] == pop)]
            tab = s.assign(v=s["mean"].map("{:.3f}".format) + " ± " + s["sd"].map("{:.3f}".format)) \
                   .pivot(index="method", columns="k", values="v").reindex([m for m in ORDER if m in set(s.method)])
            base = s.groupby("k")["pos"].mean().round(4)
            print(f"\n{tgt} {metric}, {pop} wallets (n test ≈ {s.groupby('k')['n'].mean().round(0).to_dict()}, "
                  f"base rate {base.to_dict()})")
            print(tab.to_string())
    print("\npaired differences vs best non-graph baseline (mean ± sd, wins / 5):")
    d = D[D["pop"] == "new"].copy()
    d["vs_best"] = d.d_best.map("{:+.3f}".format) + " ± " + d.d_best_sd.map("{:.3f}".format) + " (" + d.wins_best.astype(str) + ")"
    print(d.pivot_table(index=["target", "method"], columns="k", values="vs_best", aggfunc="first").to_string())
    small = D[(D["pop"] == "new") & (D.k <= 10) & (D.target == "skill")]
    ok = small.groupby("method")["beats_both"].all()
    print("\nclaim (graph model beats both baselines on NEW wallets at every k <= 10, skill AUPRC):",
          {m: bool(v) for m, v in ok.items()}, "->", "HOLDS" if ok.any() else "NOT MET")
    smallp = D[(D["pop"] == "new") & (D.k <= 10) & (D.target == "profitable")]
    print("same for profitable AUROC:", {m: bool(v) for m, v in smallp.groupby("method")["beats_both"].all().items()})

style = {"hist": ("#888", "--"), "xgb_own": ("#f0ad4e", "-"), "xgb_nbr": ("#d9534f", "-"),
         "sage": ("#5cb85c", "-"), "hgnn": ("#2a6fdb", "-")}
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
for i, pop in enumerate(["new", "existing"]):
    for j, (tgt, metric) in enumerate([("skill", "AUPRC"), ("profitable", "AUROC")]):
        ax = axes[i, j]
        s = S[(S["pop"] == pop) & (S.target == tgt)]
        for m, (c, ls) in style.items():
            d = s[s.method == m].sort_values("k")
            if len(d):
                ax.errorbar(d.k, d["mean"], yerr=d["sd"], color=c, ls=ls, marker="o", ms=4, capsize=2, label=m)
        if tgt == "skill":
            b = s.groupby("k")["pos"].mean()
            ax.plot(b.index, b.values, color="#ccc", lw=4, alpha=.6, zorder=0, label="base rate")
        else:
            ax.axhline(0.5, color="#ccc", lw=1)
        ax.set_xscale("log")
        ks = sorted(s.k.unique())
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.minorticks_off()
        ax.set(ylabel=metric, title=f"{pop} wallets — {'skill: z > 2' if tgt == 'skill' else 'profit > 0'}")
        if i == 1:
            ax.set_xlabel("k = observed post-cutoff fills")
axes[0, 0].legend(frameon=False, fontsize=8)
fig.suptitle(f"Detection curve, first-k inductive setting ({split}; mean ± sd over 5 seeds)", fontsize=11)
fig.tight_layout()
fig.savefig(ROOT / f"outputs/figures/exp2_{split}_detection.png", dpi=150)
