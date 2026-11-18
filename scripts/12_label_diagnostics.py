"""Step 12 (iteration 4): what does the skill label measure? No model is trained, labels are unchanged.

Model and split schemes: pmgraph/label_diag.py. x = T - mu0 is a wallet's excess edge per bet, s its
null sd (sd0), z = x / s the label. Parts:
  A  edge per fill by role (taker / maker), buy-price band and category, label-window fills,
     SE clustered by market                                    -> exp4_{split}_edge_role.csv
  B  split halves of every wallet's label bets (market / event / week / time): variance of the edge
     shared by the two halves, per population and bet bucket   -> exp4_{split}_reliability.csv,
     data/processed/{split}_fk_halves.parquet (per-wallet halves, input of 13)
     persistence: wallets sorted by the z of their earlier half, edge of the later half per decile
                                                               -> exp4_{split}_persistence.csv
  C  variance function x ~ N(mu, tau2 + phi s^2) per k         -> exp4_{split}_variance.csv
  D  saved exp3 scores (09 cv) against realised excess edge    -> exp4_{split}_rescored.csv
     and, on the later half only, next to two reference scores: the wallet's own earlier-half z
     (`other_half`: what knowing the wallet buys) and its noise-scale bin (`blind`: what knowing
     nothing about its skill buys)                             -> exp4_{split}_reference.csv
Figure: outputs/figures/exp4_{split}_label_diag.png.
usage: python scripts/12_label_diagnostics.py [split] [ks for part B, comma] [n_boot]
"""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pmgraph import ROOT, load_config  # noqa: E402
from pmgraph.features import sides_sql  # noqa: E402
from pmgraph.firstk import check_population_labels, int_sides, label_bets, prepare, register_ids  # noqa: E402
from pmgraph.label_diag import (EDGE_METRICS, SCHEMES, blind_score, boot_edge, half_table,  # noqa: E402
                                persistence, split_half, variance_function)
from pmgraph.labels import skill_test  # noqa: E402
from pmgraph.prep import connect  # noqa: E402
from pmgraph.rank_eval import fold_ranks  # noqa: E402

cfg = load_config()
split = sys.argv[1] if len(sys.argv) > 1 else "train"
ks_half = [int(k) for k in sys.argv[2].split(",")] if len(sys.argv) > 2 else cfg["eval"]["k_buckets"]
B = int(sys.argv[3]) if len(sys.argv) > 3 else 300
I, P = ROOT / cfg["data"]["interim_dir"], ROOT / cfg["data"]["processed_dir"]
G, L = I / f"{split}_graph_fills.parquet", I / f"{split}_label_fills.parquet"
T, F = ROOT / "outputs/tables", ROOT / "outputs/figures"
lc, ks = cfg["labels"], cfg["eval"]["k_buckets"]
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)  # noqa: E731
con = connect()
calib = pd.read_parquet(P / f"{split}_calibration.parquet")
Pop = pd.read_parquet(P / f"{split}_fk_pop.parquet")
Pop["x"] = Pop["T"] - Pop["mu0"]
POPS = [("all", lambda d: np.ones(len(d), bool)), ("new", lambda d: d["new"].to_numpy()),
        ("existing", lambda d: ~d["new"].to_numpy())]

# ---------------------------------------------------------------- A: edge by role and price band
bins = len(calib)
con.register("calib_a", calib[["b", "pi"]])
E = con.execute(f"""
  WITH s AS ({sides_sql(L, "price, w")}),
       j AS (SELECT s.*, c.pi FROM s JOIN calib_a c ON c.b = least(floor(s.price * {bins}), {bins - 1})::INT
             WHERE s.w IS NOT NULL)
  SELECT market_id, any_value(coalesce(category, 'none')) category, is_maker,
         least(floor(bp * 10), 9)::INT band, count(*) n,
         sum(CASE WHEN long_ THEN w - pi ELSE pi - w END) edge,
         sum(CASE WHEN long_ THEN w - price ELSE price - w END) raw
  FROM j GROUP BY market_id, is_maker, band""").fetchdf()


def clustered(d: pd.DataFrame) -> dict:
    """Mean edge per fill in cents; SE treats markets as independent clusters."""
    m = d.groupby("market_id")[["n", "edge", "raw"]].sum()
    mean = m["edge"].sum() / m["n"].sum()
    se = np.sqrt(((m["edge"] - mean * m["n"]) ** 2).sum()) / m["n"].sum()
    return dict(n_fills=int(m["n"].sum()), n_markets=len(m), edge_c=100 * mean, se_c=100 * se,
                raw_c=100 * m["raw"].sum() / m["n"].sum())


role = []
for mk, d in E.groupby("is_maker"):
    role.append(dict(by="all", level="all", role="maker" if mk else "taker", **clustered(d)))
    for b_, dd in d.groupby("band"):
        role.append(dict(by="buy_price", level=f"{b_/10:.1f}-{(b_+1)/10:.1f}", role="maker" if mk else "taker",
                         **clustered(dd)))
    for c_, dd in d.groupby("category"):
        role.append(dict(by="category", level=c_, role="maker" if mk else "taker", **clustered(dd)))
role = pd.DataFrame(role)
role.to_csv(T / f"exp4_{split}_edge_role.csv", index=False)
log("A: edge per fill (cents)\n" + role[role.by != "category"].round(3).to_string(index=False))
del E

# ---------------------------------------------------------------- B: split halves
gindex = pd.read_parquet(P / f"{split}_features.parquet", columns=["log_usd"]).index
N = register_ids(con, gindex, L)
int_sides(con, L, "ls")
prepare(con, N, max(ks), lc["min_resolved_bets"], G)
close = con.execute(f"SELECT market_id, any_value(close_time) close_time FROM '{L}' WHERE w IS NOT NULL GROUP BY 1").fetchdf()
unit = lc.get("bet_unit", "market")
rel, halves, pers = [], [], []
BUCKETS = [("all", 0, 10 ** 9), ("30-49", 30, 49), ("50-99", 50, 99), (">=100", 100, 10 ** 9)]
for k in ks_half:
    pk = Pop[Pop.k == k].set_index("nid")
    b = label_bets(con, k, calib)
    b = b[b["wallet"].isin(pk.index)].merge(close, on="market_id", how="left")
    rebuilt = skill_test(b, unit=unit)
    check_population_labels(rebuilt, pk, k, T / f"exp4_{split}_label_mismatch_k{k}.csv")
    log(f"B k={k}: {len(b)} label bets of {len(pk)} wallets reproduce fk_pop")
    for scheme in SCHEMES:
        H = half_table(b, scheme, unit=unit).join(pk[["new", "n_bets"]])
        halves.append(H.reset_index(names="nid").assign(k=k, scheme=scheme))
        for pop, mask in POPS:
            for name, lo, hi in BUCKETS:
                Hs = H[mask(H) & H["n_bets"].between(lo, hi)]
                rel.append(dict(k=k, scheme=scheme, pop=pop, bets=name, **split_half(Hs, B=B, seed=k)))
            if scheme == "time" and pop != "all":
                pers.append(persistence(H[mask(H)]).assign(k=k, pop=pop))
    del b
rel = pd.DataFrame(rel)
rel.to_csv(T / f"exp4_{split}_reliability.csv", index=False)
halves, pers = pd.concat(halves, ignore_index=True), pd.concat(pers, ignore_index=True)
halves.to_parquet(P / f"{split}_fk_halves.parquet", index=False)
pers.to_csv(T / f"exp4_{split}_persistence.csv", index=False)
show = rel[rel.bets == "all"].assign(shared_sd_c=lambda d: 100 * np.sign(d["cov"]) * np.sqrt(d["cov"].abs()))
log("B: variance shared by the two halves\n" + show[["k", "scheme", "pop", "n", "cov", "cov_lo", "cov_hi",
                                                     "shared_sd_c", "corr", "spearman", "phi"]].round(5).to_string(index=False))

log("B: later-half outcome by decile of earlier-half z (time split)\n" +
    pers[pers.k == ks_half[0]].round(3).to_string(index=False))

# ---------------------------------------------------------------- C: variance function
var, wts = [], {}
for k in ks:
    pk = Pop[Pop.k == k]
    for pop, mask in POPS:
        d = pk[mask(pk)]
        x, s = d["x"].to_numpy(), d["sd0"].to_numpy()
        fits = {"free": variance_function(x, s), "edge_only": variance_function(x, s, phi=1.0),
                "dependence_only": variance_function(x, s, tau2=0.0)}
        for name, f in fits.items():
            var.append(dict(k=k, pop=pop, model=name, **f, tau_c=100 * np.sqrt(f["tau2"]), mu_c=100 * f["mu"],
                            mean_x_c=100 * x.mean(), sd_z=d["z"].std(), mean_z=d["z"].mean()))
        if pop == "all":
            wts[k] = fits["free"]
var = pd.DataFrame(var)
var.to_csv(T / f"exp4_{split}_variance.csv", index=False)
log("C: variance function\n" + var[var["pop"] != "all"][["k", "pop", "model", "mu_c", "tau_c", "phi", "nll", "mean_x_c",
                                                         "mean_z", "sd_z"]].round(3).to_string(index=False))

# ---------------------------------------------------------------- D: saved exp3 scores vs realised edge
files = sorted((P / "exp3_preds").glob(f"{split}_k*_*.parquet"))
res, ref = [], []
if files:
    D = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    ORDER = [m for m in ["hist", "xgb_own", "xgb_nbr", "sage", "hgnn"] if m in set(D.method)]
    for k, g in D.groupby("k"):
        pk = Pop[Pop.k == k].set_index("nid")
        S = g.pivot(index="nid", columns="method", values="s_skill").dropna()
        fold = g[g.method == ORDER[0]].set_index("nid").loc[S.index, "seed"].to_numpy()
        lab = pk.loc[S.index]
        w = 1 / (wts[k]["tau2"] + wts[k]["phi"] * lab["sd0"].to_numpy() ** 2)
        for pop, mask in POPS[1:]:
            m_ = mask(lab)
            sc = {m: fold_ranks(S[m].to_numpy(), fold)[m_] for m in ORDER}
            out = boot_edge(sc, lab["x"].to_numpy()[m_], w[m_], lab["z"].to_numpy()[m_], B=B, seed=int(k))
            for m, (est, lo, hi) in out.items():
                for j, met in enumerate(EDGE_METRICS):
                    res.append(dict(k=k, pop=pop, method=m, metric=met, n=int(m_.sum()), est=est[j], lo=lo[j], hi=hi[j]))
        if k in ks_half:                    # the same scores on the later half, next to the reference scores
            Hk = halves[(halves.k == k) & (halves.scheme == "time")].set_index("nid").reindex(S.index)
            ok = ((Hk.n0 >= 10) & (Hk.n1 >= 10)).to_numpy()
            xB, sB, zA = Hk["x1"].to_numpy(), Hk["s1"].to_numpy(), (Hk["x0"] / Hk["s0"]).to_numpy()
            vB = variance_function(xB[ok], sB[ok])
            for pop, mask in POPS[1:]:
                m_ = mask(lab) & ok
                zB = xB[m_] / sB[m_]
                sc = {"other_half": zA[m_], "blind": blind_score(sB[m_], zB > 2, seed=int(k)),
                      **{m: fold_ranks(S[m].to_numpy(), fold)[m_] for m in ORDER}}
                out = boot_edge(sc, xB[m_], 1 / (vB["tau2"] + vB["phi"] * sB[m_] ** 2), zB, B=B, seed=int(k))
                for m, (est, lo, hi) in out.items():
                    for j, met in enumerate(EDGE_METRICS):
                        ref.append(dict(k=k, pop=pop, score=m, metric=met, n=int(m_.sum()), base_hi=(zB > 2).mean(),
                                        base_lo=(zB < -2).mean(), est=est[j], lo=lo[j], hi=hi[j]))
        log(f"D k={k} rescored")
    res = pd.DataFrame(res)
    res.to_csv(T / f"exp4_{split}_rescored.csv", index=False)
    for met in ("top_minus_bottom", "spearman_edge"):
        tab = res[(res["pop"] == "new") & (res.metric == met)].pivot(index="method", columns="k", values="est")
        log(f"D: new wallets, {met}\n" + tab.loc[ORDER].round(4).to_string())
    ref = pd.DataFrame(ref)
    ref.to_csv(T / f"exp4_{split}_reference.csv", index=False)
    for met in ("auprc_hi", "auprc_lo", "spearman_edge"):
        tab = ref[ref.metric == met].pivot(index="score", columns=["pop", "k"], values="est")
        log(f"D: later half, {met} (base hi/lo in the CSV)\n" + tab.loc[["other_half", "blind"] + ORDER].round(3).to_string())
else:
    log("D skipped: no exp3 predictions (run 09 in cv mode)")

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
a = ax[0, 0]
r = role[(role.by == "buy_price") & (role.role == "taker")]
a.bar(range(len(r)), r.edge_c, yerr=2 * r.se_c, color="#4c72b0", capsize=2)
a.axhline(0, color="k", lw=.8)
a.set_xticks(range(len(r)))
a.set_xticklabels(r.level, rotation=45, ha="right", fontsize=8)
tk = role[(role.by == "all") & (role.role == "taker")].iloc[0]
a.set_title(f"A  Taker edge per fill by buy price (all fills: {tk.edge_c:+.2f}c; maker = mirror)")
a.set_ylabel("outcome - calibrated price, cents (±2 SE, market clusters)")

a = ax[0, 1]
k0 = ks_half[0]
for i, pop in enumerate(["new", "existing"]):
    r = rel[(rel.k == k0) & (rel["pop"] == pop) & (rel.bets == "all")].set_index("scheme").loc[list(SCHEMES)]
    xs = np.arange(len(r)) + (i - .5) * .35
    a.bar(xs, 1e4 * r["cov"], .35, yerr=[1e4 * (r["cov"] - r.cov_lo), 1e4 * (r.cov_hi - r["cov"])], capsize=2,
          label=pop, color=["#4c72b0", "#dd8452"][i])
a.axhline(0, color="k", lw=.8)
a.set_xticks(range(len(SCHEMES)))
a.set_xticklabels(SCHEMES)
a.set_title(f"B  Edge variance shared by two halves of the label (k={k0})")
a.set_ylabel("covariance of half edges, cents² (95% CI)")
a.legend()

a = ax[1, 0]
for pop, col in [("new", "#4c72b0"), ("existing", "#dd8452")]:
    q = pers[(pers.k == k0) & (pers["pop"] == pop)]
    a.errorbar(q.zA, q.xB_c, yerr=2 * q.xB_c_se, color=col, marker="o", capsize=2, label=f"{pop}: mean")
    a.plot(q.zA, q.xB_c_median, color=col, ls=":", marker="x", label=f"{pop}: median")
a.axhline(0, color="k", lw=.8)
a.set_xlabel("z of the earlier half of the label bets (decile means)")
a.set_ylabel("edge on the later half, cents per bet (±2 SE)")
a.set_title(f"C  Persistence: losing repeats, winning barely (k={k0})")
a.legend(fontsize=8)

a = ax[1, 1]
if len(res):
    for m, col in [("hist", "grey"), ("xgb_own", "#4c72b0"), ("sage", "#dd8452"), ("hgnn", "#55a868")]:
        q = res[(res["pop"] == "new") & (res.metric == "top_minus_bottom") & (res.method == m)].set_index("k")
        a.errorbar(q.index, 100 * q.est, yerr=[100 * (q.est - q.lo), 100 * (q.hi - q.est)], color=col, marker="o",
                   capsize=2, label=m)
a.axhline(0, color="k", lw=.8)
a.set_xscale("log")
a.set_xticks(ks)
a.set_xticklabels(ks)
a.set_xlabel("k observed fills")
a.set_ylabel("cents per bet (95% CI)")
a.set_title("D  Realised edge, top minus bottom score decile, new wallets")
a.legend(fontsize=8)
fig.tight_layout()
fig.savefig(F / f"exp4_{split}_label_diag.png", dpi=130)
log("done")
