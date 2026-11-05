"""Step 6: detection-curve style figure from exp1 results (metric by pre-cutoff activity bucket)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
split = sys.argv[1] if len(sys.argv) > 1 else "train"
R = pd.read_csv(ROOT / f"outputs/tables/exp1_{split}_raw.csv")
order = ["1-3", "4-10", "11-50", ">50"]
style = {"hist": ("#888", "--"), "xgb_own": ("#f0ad4e", "-"), "xgb_nbr": ("#d9534f", "-"),
         "spectral_lr": ("#9b59b6", ":"), "labelprop": ("#5cb85c", ":"), "sage": ("#2a6fdb", "-")}

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (target, metric) in zip(axes, [("skill", "auprc"), ("profitable", "auroc")]):
    sub = R[(R.target == target) & (R.bucket.isin(order))]
    g = sub.groupby(["method", "bucket"])[metric].agg(["mean", "std"]).reset_index()
    for m, (c, ls) in style.items():
        d = g[g.method == m].set_index("bucket").reindex(order)
        ax.errorbar(range(4), d["mean"], yerr=d["std"], color=c, ls=ls, marker="o", ms=4, capsize=2, label=m)
    if target == "skill":
        base = sub.groupby("bucket")["pos"].mean().reindex(order)
        ax.plot(range(4), base, color="#ccc", lw=4, alpha=.6, label="base rate", zorder=0)
    else:
        ax.axhline(0.5, color="#ccc", lw=1)
    ax.set_xticks(range(4), order)
    ax.set(xlabel="fills before cutoff", ylabel=metric.upper(),
           title=f"{target}: {'z > 2 (≈2%)' if target == 'skill' else 'profit > 0'}  (mean ± sd, 5 seeds)")
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / f"outputs/figures/exp1_{split}_detection.png", dpi=150)

# paired seed-wise comparison: GraphSAGE vs. best non-graph baseline on skill AUPRC
a = R[(R.target == "skill") & (R.bucket == "all")].pivot(index="seed", columns="method", values="auprc")
for base in ["hist", "xgb_own", "xgb_nbr"]:
    d = a["sage"] - a[base]
    print(f"sage - {base}: {d.mean():+.4f} ± {d.std():.4f}  (wins {int((d > 0).sum())}/{len(d)})")
