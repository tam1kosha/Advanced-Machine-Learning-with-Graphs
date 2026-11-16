"""Exercise real training/report entry points on a tiny isolated prepared dataset."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_training_entrypoint_checkpoints_resume_and_complete_report(tmp_path):
    for directory in ("src", "scripts", "configs"):
        shutil.copytree(ROOT / directory, tmp_path / directory, ignore=shutil.ignore_patterns("__pycache__"))
    for d in ("outputs/tables", "outputs/figures", "data/processed"):
        (tmp_path / d).mkdir(parents=True)
    cfg = yaml.safe_load((tmp_path / "configs/mvp.yaml").read_text())
    cfg["eval"]["k_buckets"] = [1]
    (tmp_path / "configs/mvp.yaml").write_text(yaml.safe_dump(cfg))
    p = tmp_path / "data/processed"
    rng, nw, nm, nq = np.random.default_rng(4), 40, 6, 180
    cols = ["hist_z", "hist_profit_slog", "hist_raw_edge", "log_usd", "maker_share", "frac_longshot", "log_n_fills"]
    data = HeteroData()
    data["wallet"].x = torch.from_numpy(rng.normal(size=(nw, len(cols))).astype("float32"))
    data["wallet"].cols = cols
    data["market"].x = torch.from_numpy(rng.normal(size=(nm, 3)).astype("float32"))
    data["market"].market_id = torch.arange(nm)
    for relation, src, dst, width in [("cp", nw, nw, 1), ("mf", nw, nw, 1), ("held", nw, nm, 5)]:
        key = ("wallet", relation, "market" if relation == "held" else "wallet")
        data[key].edge_index = torch.from_numpy(np.stack([rng.integers(0, src, 120), rng.integers(0, dst, 120)]))
        data[key].edge_attr = torch.from_numpy(rng.normal(size=(120, width)).astype("float32"))
    torch.save(data, p / "train_hetero.pt")
    ids = np.arange(nq)
    pop = pd.DataFrame(dict(nid=ids, k=1, new=ids >= nw, skill=ids % 2, profitable=(ids // 2) % 2,
                            t_k=pd.Timestamp("2025-01-02", tz="UTC"), z=rng.normal(size=nq), n_bets=40))
    pop.to_parquet(p / "train_fk_pop.parquet")
    pd.DataFrame(rng.normal(size=(nq, len(cols))), columns=cols).assign(nid=ids, k=1).to_parquet(p / "train_fk_feat.parquet")
    pd.DataFrame(dict(nid=ids, rn=1, cp=rng.integers(0, nw, nq), market_id=rng.integers(0, nm, nq),
                      long_=True, qty=1., price=.5, usd=.5, hours_to_close=24., is_maker=0)).to_parquet(p / "train_fk_obs.parquet")
    pd.DataFrame(dict(nid=ids, rn=1, other=rng.integers(0, nw, nq), t_other=pop.t_k, dir="lead")).to_parquet(p / "train_fk_mf.parquet")
    pd.DataFrame(dict(nid=ids, k=1, scheme="week", n0=20, n1=20,
                      x0=rng.normal(size=nq), x1=rng.normal(size=nq), s0=.1, s1=.1)).to_parquet(p / "train_fk_halves.parquet")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PMGRAPH_REQUIRE_CUDA": "0", "PMGRAPH_RESUME": "1",
           "PMGRAPH_TAG": "integration", "KS": "1", "METHODS": "hist,sage,hgnn", "MPLCONFIGDIR": str(tmp_path / "mpl"),
           "PMGRAPH_GNN": json.dumps(dict(hidden=8, fanout=[2, 2], batch=32, eval_batch=16,
                                         epochs=2, patience=2, eval_reps=1, threads=1))}
    command = [sys.executable, "scripts/09_first_k_experiments.py", "train", "hist,sage,hgnn", "1", "cv", "edge"]
    result = subprocess.run(command, cwd=tmp_path, env=env, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    ckpts = list((p / "checkpoints/exp4_integration").glob("*_fold*.pt"))
    assert len(ckpts) == 10
    times = [c.stat().st_mtime_ns for c in ckpts]
    for method in ("hist", "sage", "hgnn"):
        pred = pd.read_parquet(p / f"exp4_preds_integration/train_k1_{method}.parquet")
        assert len(pred) == nq and not pred.nid.duplicated().any() and np.isfinite(pred.s_edge).all()
    result = subprocess.run(command, cwd=tmp_path, env=env, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0 and result.stdout.count("skipped") == 3, result.stdout + result.stderr
    assert times == [c.stat().st_mtime_ns for c in ckpts]
    report = [sys.executable, "scripts/14_graph_on_edge.py", "train", "10", "integration"]
    result = subprocess.run(report, cwd=tmp_path, env=env, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "outputs/tables/exp4_train_graph_integration.csv").is_file()
    env["KS"] = "1 3"
    result = subprocess.run(report, cwd=tmp_path, env=env, text=True, capture_output=True, timeout=120)
    assert result.returncode != 0 and "incomplete CV predictions" in result.stderr


def test_launcher_routes_gpu_subset_quotes_paths_and_propagates_failure(tmp_path):
    shutil.copy(ROOT / "run_server.sh", tmp_path)
    fake = tmp_path / "python with spaces"
    fake.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
if sys.argv[1].endswith('server_check.py'):
    if '--gpu-tokens' in sys.argv: print('3 7')
    sys.exit(0)
if sys.argv[1].endswith('09_first_k_experiments.py'):
    print(json.dumps({'gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'k': sys.argv[4]}))
    if sys.argv[4] == os.environ.get('FAIL_K'): sys.exit(5)
''')
    fake.chmod(0o755)
    env = {**os.environ, "PYTHON": str(fake), "SPLITS": "train", "KS": "1 3", "PMGRAPH_TAG": "test"}
    result = subprocess.run(["bash", "run_server.sh", "edge"], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(p.read_text()) for p in (tmp_path / "outputs/logs").glob("09*.log")]
    assert {r["gpu"] for r in rows} == {"3", "7"}
    env["FAIL_K"] = "3"
    result = subprocess.run(["bash", "run_server.sh", "edge"], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode != 0
