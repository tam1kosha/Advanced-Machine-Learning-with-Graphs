"""Step 16: the two overview figures used in README.md and docs/.

Reads only the tables committed under outputs/tables/ (no data/ needed, a few seconds), so anyone can
re-draw them right after cloning.
  overview_{split}_setup.png    what a model may see and what it must predict, and how rare skill is
  overview_{split}_results.png  the three findings so far: (A) losing repeats, (B) the training target
                                decides everything, (C) graph models tie XGBoost
Needs the outputs of 01, 03, 07, 12, 13 and 14 for the split.
usage: python scripts/16_overview_figures.py [split]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pmgraph import ROOT, load_config  # noqa: E402
from pmgraph.plotstyle import (ACCENT, GRAPH_METHODS, GRID, INK, INK2, METHOD_COLOR, METHOD_LABEL,  # noqa: E402
                               MUTED, WARM, apply_style)

split = sys.argv[1] if len(sys.argv) > 1 else "train"
T, F = ROOT / "outputs/tables", ROOT / "outputs/figures"
win = load_config()["splits"][split]
apply_style()
LABEL_C = "#eb6834"   # "what is predicted" in the schematic


def js(name):
    return json.loads((T / name).read_text())


def span(start, end):
    """'Jul–Dec 2024' for the half-open window [start, end)."""
    a, b = pd.Timestamp(start), pd.Timestamp(end) - pd.Timedelta(days=1)
    return f"{a.strftime('%b') if a.year == b.year else a.strftime('%b %Y')}–{b.strftime('%b %Y')}"


# ---------------------------------------------------------------- figure 1: the task and its scale
prep, lab, eda, fk = js("prepare_stats.json"), js(f"labels_{split}.json"), js(f"eda_{split}.json"), \
    js(f"fk_{split}_population.json")
g, lw = prep[f"{split}/graph"], prep[f"{split}/label"]
fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[2.15, 1], wspace=0.08))

ax.set(xlim=(-2.7, 12.25), ylim=(-0.45, 5.0))
ax.axis("off")
for x0, col, head, sub in [
        (0, "#efeeea", f"graph window · {span(win['graph_start'], win['cutoff'])}",
         f"{g['kept'] / 1e6:.1f}M trades · {eda['graph_wallets']['n_wallets'] / 1e3:.0f}k wallets"),
        (6, "#e3eefb", f"label window · {span(win['cutoff'], win['label_end'])}",
         f"{lw['kept'] / 1e6:.1f}M trades · {eda['label_wallets']['n_wallets'] / 1e3:.0f}k wallets")]:
    ax.add_patch(Rectangle((x0 + 0.04, 3.45), 5.92, 0.95, color=col, lw=0))
    ax.text(x0 + 0.25, 4.07, head, fontsize=10, fontweight="bold", va="center")
    ax.text(x0 + 0.25, 3.70, sub, fontsize=9, color=INK2, va="center")
ax.plot([6, 6], [0.55, 4.72], color=INK, lw=1.2)
ax.text(6.0, 4.84, f"cutoff {pd.Timestamp(win['cutoff']).strftime('%-d %b %Y')}: the graph is frozen here", ha="center",
        fontsize=9, fontweight="bold")

rng = np.random.default_rng(3)
hist_x = np.sort(rng.uniform(0.4, 5.7, 9))
obs_x = np.array([6.35, 6.62, 6.89])
lab_x = np.linspace(7.45, 11.8, 14) + rng.uniform(-0.09, 0.09, 14)
for y, name, note, has_hist in [(2.45, "existing wallet", "traded before the cutoff:\nalready a node in the graph", True),
                                (1.05, "new wallet", "first trade is after the cutoff:\nno history, no node", False)]:
    ax.plot([0.05, 11.95], [y, y], color=GRID, lw=1, zorder=1)
    ax.text(-0.15, y + 0.13, name, ha="right", va="center", fontsize=10, fontweight="bold")
    ax.text(-0.15, y - 0.33, note, ha="right", va="center", fontsize=8, color=INK2, linespacing=1.25)
    if has_hist:
        ax.scatter(hist_x, np.full(len(hist_x), y), s=62, color=MUTED, edgecolor="white", lw=1.5, zorder=3)
    else:
        ax.text(3.0, y + 0.17, "nothing to read here", ha="center", fontsize=8.5, color=MUTED, style="italic")
    ax.scatter(obs_x, np.full(3, y), s=74, color=ACCENT, edgecolor="white", lw=1.5, zorder=3)
    ax.scatter(lab_x, np.full(len(lab_x), y), s=62, color=LABEL_C, edgecolor="white", lw=1.5, zorder=3)
ax.annotate("", xy=(7.05, 1.75), xytext=(6.2, 1.75), arrowprops=dict(arrowstyle="-", color=ACCENT, lw=1.4))
ax.text(6.2, 1.86, "first k trades", ha="left", fontsize=8.5, color=INK)
ax.text(9.95, 1.86, "later bets, in other markets", ha="center", fontsize=8.5, color=INK)
ax.annotate("", xy=(11.9, 1.75), xytext=(7.4, 1.75), arrowprops=dict(arrowstyle="-", color=LABEL_C, lw=1.4))
ks = load_config()["eval"]["k_buckets"]
ax.scatter([-2.45], [0.2], s=62, color=MUTED, edgecolor="white", lw=1.5)
ax.scatter([-2.15], [0.2], s=74, color=ACCENT, edgecolor="white", lw=1.5)
ax.text(-1.85, 0.2, f"a model may read: the frozen graph, the wallet's own history, its first k trades (k = {ks[0]} to {ks[-1]})",
        va="center", fontsize=9)
ax.scatter([-2.3], [-0.2], s=62, color=LABEL_C, edgecolor="white", lw=1.5)
ax.text(-1.85, -0.2, f"it must predict: how well the wallet bets afterwards (needs ≥ {load_config()['labels']['min_resolved_bets']} resolved later bets)",
        va="center", fontsize=9)
ax.set_title("A  What a model sees and what it must predict", pad=10)

rows = [("wallets trading in the label window", eda["label_wallets"]["n_wallets"]),
        (f"… with ≥ {load_config()['labels']['min_resolved_bets']} resolved bets (get a label)", lab["labelled_wallets"]),
        ("… that look better than luck (z > 2)", round(lab["share_z_gt_2"] * lab["labelled_wallets"])),
        ("… that pass the strict multiple-testing check", lab["type_counts"]["skilled_winner"])]
bx.set_xscale("log")
bx.set(xlim=(20, 6e6), ylim=(len(rows) - 0.45, -0.75))
for i, (name, n) in enumerate(rows):
    bx.plot([20, n], [i, i], color=GRID, lw=1.2, zorder=1)
    bx.scatter([n], [i], s=95, color=ACCENT, edgecolor="white", lw=1.5, zorder=3)
    bx.text(n * 1.45, i, f"{n:,}", va="center", fontsize=10, fontweight="bold")
    bx.text(22, i - 0.33, name, va="center", fontsize=8.8, color=INK2)
bx.set_yticks([])
bx.grid(axis="y", visible=False)
bx.spines["left"].set_visible(False)
bx.set_xlabel("wallets (log scale)")
bx.set_title("B  Skill is rare", pad=10)
fig.savefig(F / f"overview_{split}_setup.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- figure 2: the three findings
P = pd.read_csv(T / f"exp4_{split}_persistence.csv").query("k == 1 and pop == 'new'").sort_values("group")
TG = pd.read_csv(T / f"exp4_{split}_targets.csv").query("metric == 'wcorr_edge' and pop == 'new'")
GP = pd.read_csv(T / f"exp4_{split}_graph_paired.csv").query("metric == 'wcorr_edge' and pop == 'new' and vs == 'xgb_own'")
G = pd.read_csv(T / f"exp4_{split}_graph.csv").query("metric == 'wcorr_edge' and pop == 'new' and method == 'xgb_own'")
fig, (a, b, c) = plt.subplots(1, 3, figsize=(15.5, 4.7), gridspec_kw=dict(wspace=0.3))

# A: persistence of the edge between the earlier and the later half of a wallet's label bets
x = np.arange(1, len(P) + 1)
a.bar(x, P.xB_c, width=0.62, color=ACCENT, zorder=3)
a.errorbar(x, P.xB_c, yerr=2 * P.xB_c_se, fmt="none", ecolor=INK, elinewidth=1, capsize=2, zorder=4)
a.axhline(0, color=INK2, lw=1)
a.set_xticks(x, [str(i) for i in x])
a.set(xlabel="wallets sorted by result on the EARLIER half of their bets\n(1 = worst tenth, 10 = best tenth)",
      ylabel="result on the LATER half, cents per bet (± 2 SE)")
a.grid(axis="x", visible=False)
lo, hi = P.xB_c.iloc[0], P.xB_c.iloc[-1]
a.text(1.55, lo, f"{lo:+.1f}c: the worst tenth\nkeeps losing", va="center", fontsize=9)
a.annotate(f"{hi:+.1f}c: the best tenth\nends up about even", xy=(10, hi - 2 * P.xB_c_se.iloc[-1] - 0.1),
           xytext=(10.35, -3.6), ha="right", va="top", fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
a.set_title("A  Losing repeats, winning barely does")

# B: same model, same inputs, three training targets
for tgt, col, name, dy in [("reg_edge_w", ACCENT, "XGBoost trained\non the edge", 0.0),
                           ("hist", METHOD_COLOR["hist"], "history-only\n(no training)", 0.0),
                           ("cls_z2_full", WARM, "XGBoost trained\non “z > 2”\n(iterations 1–3)", -0.03)]:
    d = TG[TG.target == tgt].sort_values("k")
    b.errorbar(d.k, d.est, yerr=[d.est - d.lo, d.hi - d.est], color=col, marker="o", capsize=2, elinewidth=1)
    b.text(60, d.est.iloc[-1] + dy, name, va="center", fontsize=8.8, linespacing=1.2)
b.axhline(0, color=INK2, lw=1)
b.set_xscale("log")
kb = sorted(TG.k.unique())
b.set_xlim(0.85, 330)
b.set_xticks(kb, [str(k) for k in kb])
b.minorticks_off()
b.set(xlabel="k = trades of a new wallet the model has seen",
      ylabel="correlation of the score with the wallet's later result")
b.text(0.9, -0.03, "below 0 = ranks wallets backwards", fontsize=8.5, color=INK2, va="top")
b.set_title("B  New wallets can be ranked from one trade,\n    if the model learns the right target")

# C: what the graph adds on top of XGBoost
off = {m: o for m, o in zip(GRAPH_METHODS, (-0.22, 0.0, 0.22))}
for m in GRAPH_METHODS:
    d = GP[GP.method == m].set_index("k").reindex(ks)
    c.errorbar(np.arange(len(ks)) + off[m], d["diff"], yerr=[d["diff"] - d.lo, d.hi - d["diff"]], fmt="o",
               color=METHOD_COLOR[m], capsize=2, elinewidth=1.2, label=METHOD_LABEL[m])
c.axhline(0, color=INK, lw=1.2)
base = G.set_index("k").est
c.text(-0.45, -0.047, f"0 = XGBoost on the wallet's own trades,\nwhich itself scores {base.min():.2f} to {base.max():.2f}",
       va="center", fontsize=8.5, color=INK2, linespacing=1.25)
c.set_xticks(range(len(ks)), [str(k) for k in ks])
c.set_ylim(-0.055, 0.055)
c.grid(axis="x", visible=False)
c.set(xlabel="k = trades of a new wallet the model has seen", ylabel="graph model − XGBoost, same correlation (95% CI)")
c.legend(loc="upper left", handletextpad=0.3, borderaxespad=0.2)
c.set_title("C  The graph adds nothing so far")
fig.savefig(F / f"overview_{split}_results.png", bbox_inches="tight")
print("wrote", F / f"overview_{split}_setup.png", "and", F / f"overview_{split}_results.png")
