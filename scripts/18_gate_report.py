"""Step 18 (iteration 5): Gates B and C side by side, from 14's per-tag tables.

Gate B: GNN settings (README; `grid` stage of run_server.sh) — every setting is reported, none is picked.
Gate C: as-of nodes for markets without a pre-cutoff node (17, tag `asof`, default GNN settings).
Pass rule (README, fixed before the run): a graph model beats both baselines (history-only, XGBoost own)
on new wallets at every k <= 10: paired 95% CI of the weighted edge correlation above 0 against each.
(The README asks for both periods; this is the train period only.)
Inputs: outputs/tables/exp4_{split}_graph{,_paired}_{tag}.csv. Outputs: outputs/tables/exp5_{split}_gates.csv,
outputs/figures/exp5_{split}_gates.png.
usage: python scripts/18_gate_report.py [split] [tags, comma (default repro,wide,fan,widefan,slow,asof)]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pmgraph.plotstyle import METHOD_COLOR, METHOD_LABEL, apply_style  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
split = sys.argv[1] if len(sys.argv) > 1 else "train"
tags = sys.argv[2].split(",") if len(sys.argv) > 2 else ["repro", "wide", "fan", "widefan", "slow", "asof"]
T = ROOT / "outputs/tables"
GATE = {"repro": "B (default settings, = iteration 4)", "wide": "B", "fan": "B", "widefan": "B", "slow": "B",
        "asof": "C (as-of market nodes)"}

rows = []
for tag in tags:
    f, fp = T / f"exp4_{split}_graph_{tag}.csv", T / f"exp4_{split}_graph_paired_{tag}.csv"
    if not f.exists():
        print("missing", f.name)
        continue
    R, Q = pd.read_csv(f), pd.read_csv(fp)
    R = R[(R.metric == "wcorr_edge") & (R.k <= 10)]
    Q = Q[(Q.metric == "wcorr_edge") & (Q.k <= 10)]
    for (pop, k, m), q in Q.groupby(["pop", "k", "method"]):
        x = q[q.vs == "xgb_own"].iloc[0]
        est = R[(R["pop"] == pop) & (R.k == k) & (R.method == m)].iloc[0]
        ref = R[(R["pop"] == pop) & (R.k == k) & (R.method == "xgb_own")].iloc[0]
        rows.append(dict(tag=tag, gate=GATE.get(tag, "?"), pop=pop, k=k, method=m, wcorr=est.est, xgb_own=ref.est,
                         d_xgb=x["diff"], lo_xgb=x.lo, hi_xgb=x.hi, beats_both=bool((q.lo > 0).all())))
D = pd.DataFrame(rows).round(4)
D.to_csv(T / f"exp5_{split}_gates.csv", index=False)

with pd.option_context("display.width", 200, "display.max_rows", 200):
    for pop in ("new", "existing"):
        d = D[D["pop"] == pop]
        cell = d.d_xgb.map("{:+.3f}".format) + " [" + d.lo_xgb.map("{:+.3f}".format) + ", " + d.hi_xgb.map("{:+.3f}".format) + "]"
        print(f"\n{pop} wallets: weighted edge correlation, graph model minus XGBoost own (paired 95% CI)")
        print(d.assign(c=cell).pivot_table(index=["tag", "method"], columns="k", values="c", aggfunc="first")
              .reindex(tags, level=0).to_string())
    n = D[D["pop"] == "new"]
    verdict = n.groupby(["tag", "method"]).beats_both.all()
    print("\npass rule (new wallets, every k <= 10, CI > 0 vs both baselines):")
    for (tag, m), ok in verdict.items():
        print(f"  {tag:8s} {m:8s} {'PASS' if ok else 'fail'}  (k passing: "
              f"{list(n[(n.tag == tag) & (n.method == m) & n.beats_both].k)})")
    print("->", "a graph model PASSES on the train period" if verdict.any() else "NO graph model passes (train period)")

apply_style()
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
for a, pop in zip(axes, ("new", "existing")):
    d = D[D["pop"] == pop]
    combos = [(t, m) for t in tags for m in ("sage", "hgnn", "xgb_nbr") if len(d[(d.tag == t) & (d.method == m)])]
    w = 0.8 / max(len(combos), 1)
    for i, (t, m) in enumerate(combos):
        q = d[(d.tag == t) & (d.method == m)].sort_values("k")
        x = [list(sorted(d.k.unique())).index(k) + (i - len(combos) / 2) * w for k in q.k]
        a.errorbar(x, q.d_xgb, yerr=[q.d_xgb - q.lo_xgb, q.hi_xgb - q.d_xgb], fmt="o" if t != "asof" else "s",
                   ms=3.5, lw=1, color=METHOD_COLOR[m], alpha=1 if t in ("repro", "asof") else 0.55,
                   label=f"{m} · {t}" if pop == "new" else None)
    a.axhline(0, color="#52514e", lw=1)
    ks = sorted(d.k.unique())
    a.set_xticks(range(len(ks)), [str(k) for k in ks])
    a.set(xlabel="k = observed post-cutoff fills", title=f"{pop} wallets")
axes[0].set_ylabel("graph model − XGBoost own\n(weighted edge correlation, 95% CI)")
axes[0].legend(fontsize=6.5, ncol=2, frameon=False)
fig.suptitle(f"Gates B (GNN settings) and C (as-of market nodes), {split} period: "
             f"{', '.join(METHOD_LABEL[m].split(' (')[0] for m in ('sage', 'hgnn'))}", fontsize=10)
fig.tight_layout()
fig.savefig(ROOT / f"outputs/figures/exp5_{split}_gates.png", dpi=150)
