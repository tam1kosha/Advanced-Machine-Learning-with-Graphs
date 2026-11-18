"""GNN-only tuning uses fixed disjoint wallets, validation selection and independent test metrics."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from pmgraph.gnn_only import centered_edge_target, wallet_holdout
from pmgraph.hetero import require_finite
from pmgraph.hetero import train_hetero
from test_hetero import _reg_graph
from test_server_integration import prepared_graph

ROOT = Path(__file__).resolve().parents[1]


def test_holdout_is_disjoint_exhaustive_and_invariant_to_row_order():
    nids = np.arange(400)
    strat = nids % 8
    split = wallet_holdout(nids, strat, 7)
    assert [len(x) for x in split] == [240, 80, 80]
    assert len(set(np.concatenate(split))) == 400
    shuffled = np.random.default_rng(9).permutation(400)
    other = wallet_holdout(nids[shuffled], strat[shuffled], 7)
    for a, b in zip(split, other):
        assert set(nids[a]) == set(nids[shuffled][b])


def test_target_scaling_does_not_fit_validation_or_test_labels():
    rng = np.random.default_rng(2)
    H = pd.DataFrame(dict(n0=20, x0=rng.normal(size=120), s0=rng.uniform(.05, .1, 120)))
    fit = np.arange(80)
    a = centered_edge_target(H, fit)
    H.loc[80:, "x0"] += 1000
    b = centered_edge_target(H, fit)
    np.testing.assert_array_equal(a[fit], b[fit])


def test_nonfinite_graph_inputs_report_the_source():
    values = np.ones((3, 5), np.float32)
    values[2, 4] = np.nan
    with pytest.raises(ValueError, match=r"frozen.market.x: nonfinite input at index \(2, 4\)"):
        require_finite(values, "frozen.market.x")


def test_nonfinite_gradient_is_caught_before_optimizer_update(monkeypatch, tmp_path):
    import torch
    import pmgraph.hetero as hetero
    original = hetero.HeteroSAGE
    def bad_gradient_model(*args, **kwargs):
        model = original(*args, **kwargs)
        next(model.parameters()).register_hook(lambda grad: torch.full_like(grad, float("nan")))
        return model
    monkeypatch.setattr(hetero, "HeteroSAGE", bad_gradient_model)
    graph, target = _reg_graph(5, 1.0)
    path = tmp_path / "bad.pt"
    with pytest.raises(FloatingPointError, match="nonfinite gradient at epoch 1"):
        train_hetero(graph, ["cp"], target, {"train": np.arange(360), "val": np.arange(360, 480)}, 0,
                     dict(device="cpu", hidden=8, fanout=[2, 2], batch=64, epochs=2, grad_clip=1),
                     objectives=["reg"], checkpoint_path=path, log=lambda *a: None)
    assert not path.exists()


def test_cli_dry_run_only_plans_the_requested_gnns(tmp_path):
    p = subprocess.run([sys.executable, "scripts/17_gnn_only.py", "--tag", "plan", "--ks", "1", "--splits", "train",
                        "--gpus", "0", "1", "--lr", ".001", "--hidden", "128", "--dry-run"],
                       cwd=ROOT, text=True, capture_output=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert '"hidden": 128' in p.stdout and '"lr": 0.001' in p.stdout
    assert "train sage 1 holdout edge" in p.stdout and "train hgnn 1 holdout edge" in p.stdout
    assert "xgb" not in p.stdout and "hist " not in p.stdout


def test_gnn_only_train_validate_test_resume_and_missing_market_inputs(tmp_path):
    processed, env, _ = prepared_graph(tmp_path, missing_market=True)
    # No XGBoost worker/native library should be required anywhere in this interface.
    env.pop("PMGRAPH_GNN", None)
    script = [sys.executable, "scripts/17_gnn_only.py", "--tag", "trial"]
    fit_args = ["--stage", "train", "--device", "cpu", "--splits", "train", "--ks", "1", "--seeds", "0", "1",
                "--epochs", "2", "--patience", "2", "--hidden", "8", "--fanout", "2", "2", "--batch", "32",
                "--eval-batch", "16", "--eval-reps", "1", "--threads", "1", "--workers", "2", "--bootstrap", "10"]
    def run(args):
        return subprocess.run(script + args, cwd=tmp_path, env=env, text=True, capture_output=True, timeout=120)
    result = run(fit_args)
    assert result.returncode == 0, result.stdout + result.stderr
    out = tmp_path / "outputs/logs/gnn/trial"
    tables = tmp_path / "outputs/tables"
    assert (tables / "exp6_trial_val_metrics.csv").is_file() and not (tables / "exp6_trial_test_metrics.csv").exists()
    validation = pd.read_csv(tables / "exp6_trial_val_metrics.csv")
    assert set(validation.model) == {"sage", "hgnn"} and set(validation.phase) == {"val"}
    assert (validation.loc[validation.metric == "wcorr_edge", "lo"].notna()).all()
    paired = pd.read_csv(tables / "exp6_trial_val_paired.csv")
    assert set(paired.comparison) == {"hgnn-minus-sage"}
    preds = processed / "exp6_preds_trial"
    partitions = pd.read_parquet(preds / "train_k1_wallet_split.parquet").set_index("nid").partition
    for model in ("sage", "hgnn"):
        v = pd.read_parquet(preds / f"train_k1_{model}_val.parquet")
        t = pd.read_parquet(preds / f"train_k1_{model}.parquet")
        assert set(v.seed) == set(t.seed) == {0, 1}
        assert not set(v.nid) & set(t.nid)
        assert (partitions.loc[v.nid] == "val").all() and (partitions.loc[t.nid] == "test").all()
    assert "imputed 1 missing market" in (out / "train_k1_hgnn.log").read_text()
    models = sorted((processed / "checkpoints/exp6_trial").glob("*_seed*.pt"))
    assert len(models) == 4
    times = [p.stat().st_mtime_ns for p in models]
    result = run(fit_args)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "skipped" in (out / "train_k1_hgnn.log").read_text()
    assert times == [p.stat().st_mtime_ns for p in models]
    result = run(["--stage", "test", "--bootstrap", "10"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert set(pd.read_csv(tables / "exp6_trial_test_metrics.csv").phase) == {"test"}
    assert times == [p.stat().st_mtime_ns for p in models]
    assert all(r["returncode"] == 0 for r in json.loads((out / "jobs.json").read_text()))
    result = run(fit_args + ["--hidden", "16"])
    assert result.returncode != 0 and "use a new --tag" in result.stderr
    # The reporter must reject swapped validation/test files rather than score the wrong wallets.
    data = pd.read_parquet(preds / "train_k1_sage_val.parquet")
    data.to_parquet(preds / "train_k1_sage.parquet")
    result = run(["--stage", "test", "--bootstrap", "0"])
    assert result.returncode != 0 and "wrong test wallet set" in result.stderr


def test_failed_gnn_job_returns_nonzero_and_does_not_report_test_success(tmp_path):
    import torch
    processed, env, _ = prepared_graph(tmp_path)
    path = processed / "train_hetero.pt"
    graph = torch.load(path, weights_only=False)
    graph["wallet", "cp", "wallet"].edge_attr[0, 0] = float("nan")
    torch.save(graph, path)
    cmd = [sys.executable, "scripts/17_gnn_only.py", "--tag", "failed", "--device", "cpu", "--splits", "train",
           "--ks", "1", "--seeds", "0", "--models", "hgnn", "--bootstrap", "0"]
    result = subprocess.run(cmd, cwd=tmp_path, env=env, text=True, capture_output=True, timeout=120)
    assert result.returncode != 0 and "Some jobs failed" in result.stdout
    out = tmp_path / "outputs/logs/gnn/failed"
    assert not (tmp_path / "outputs/tables/exp6_failed_test_metrics.csv").exists()
    assert "frozen.cp.edge_attr: nonfinite input" in (out / "train_k1_hgnn.log").read_text()
    assert json.loads((out / "jobs.json").read_text())[0]["returncode"] != 0
