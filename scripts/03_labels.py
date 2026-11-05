"""Step 3: calibrate prices, build label-window bets, run the skill test, assign four types."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pmgraph import ROOT, load_config  # noqa: E402
from pmgraph.labels import assign_types, bets, fit_calibration, skill_test  # noqa: E402
from pmgraph.prep import connect  # noqa: E402

cfg = load_config()
split = sys.argv[1] if len(sys.argv) > 1 else "train"
sp, lc = cfg["splits"][split], cfg["labels"]
I, P = ROOT / cfg["data"]["interim_dir"], ROOT / cfg["data"]["processed_dir"]
P.mkdir(parents=True, exist_ok=True)
con = connect()

calib = fit_calibration(con, I / f"{split}_graph_fills.parquet", sp["cutoff"], lc["calib_bins"])
calib.to_parquet(P / f"{split}_calibration.parquet")

b = bets(con, I / f"{split}_label_fills.parquet", calib)
print("bets", len(b), "wallets", b["wallet"].nunique(), flush=True)
w = skill_test(b)
lab = assign_types(w, lc["min_resolved_bets"], lc["fdr_alpha"])

# Monte Carlo check of the analytic p-values on a random subsample
sub = lab.sample(min(400, len(lab)), random_state=0).index
mc = skill_test(b[b.wallet.isin(sub)], n_sim=lc["n_sim"])
agree = float(np.corrcoef(mc["p"], mc["p_mc"])[0, 1])
print("analytic vs MC p-value corr:", round(agree, 4))

lab.to_parquet(P / f"{split}_labels.parquet")
summary = {
    "labelled_wallets": len(lab),
    "type_counts": lab["type"].value_counts().to_dict(),
    "type_share": lab["type"].value_counts(normalize=True).round(4).to_dict(),
    "share_profitable": float(lab.profitable.mean()),
    "share_skilled": float(lab.skilled.mean()),
    "median_bets": float(lab.n_bets.median()),
    "p_analytic_vs_mc_corr": agree,
}
print(json.dumps(summary, indent=2))
with open(ROOT / f"outputs/tables/labels_{split}.json", "w") as f:
    json.dump(summary, f, indent=2)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(calib["mid"], calib["rate"], ".", color="#bbb", label="binned (market-weighted)")
ax[0].plot(calib["mid"], calib["pi"], color="#2a6fdb", label="isotonic fit")
ax[0].plot([0, 1], [0, 1], ls="--", lw=1, color="#999")
ax[0].set(xlabel="price", ylabel="P(win)", title="Favourite–longshot calibration")
ax[0].legend(frameon=False)
colors = dict(zip(["skilled_winner", "lucky_winner", "unlucky_skilled", "unskilled_loser"],
                  ["#2a6fdb", "#f0ad4e", "#5cb85c", "#bbbbbb"]))
for t, g in lab.groupby("type"):
    ax[1].scatter(g["z"].clip(-10, 15), np.sign(g.profit) * np.log10(1 + g.profit.abs()),
                  s=3, alpha=.4, color=colors[t], label=f"{t} ({len(g)})")
ax[1].set(xlabel="skill z-score", ylabel="sign·log10(1+|profit $|)", title="Four trader types")
ax[1].legend(frameon=False, fontsize=7, markerscale=3)
fig.tight_layout()
fig.savefig(ROOT / f"outputs/figures/labels_{split}.png", dpi=150)
