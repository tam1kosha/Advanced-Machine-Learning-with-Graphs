"""Server routing, completed-job invalidation, and missing-data failure cases."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from pmgraph.hetero import CSR, Graph, HeteroSAGE, sample_batch, train_hetero
from pmgraph.prep import trade_files
from pmgraph.runinfo import completed_job, finish_job, job_signature

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("server_check", ROOT / "scripts/server_check.py")
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


def test_allocated_gpus_are_preserved_and_invalid_counts_rejected():
    assert server.gpu_tokens(2, "3,7", None) == ["3", "7"]
    assert server.gpu_tokens(2, "GPU-a,GPU-b", "1") == ["GPU-a"]
    for n in ("0", "3", "-1"):
        with pytest.raises(ValueError):
            server.gpu_tokens(2, "3,7", n)
    with pytest.raises(ValueError):
        server.gpu_tokens(1, "3,7", None)


def test_gpu_stage_refuses_silent_cpu_fallback(monkeypatch):
    import sys
    import torch
    monkeypatch.delenv("PMGRAPH_ALLOW_CPU", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", ["server_check.py", "--stages", "edge"])
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        server.main()


def test_empty_relations_allow_an_isolated_query_and_checkpoint(tmp_path):
    empty = np.array([], dtype=np.int64)
    csr = CSR.build(empty, empty, np.zeros((0, 1)), 12)
    rng = np.random.default_rng(3)
    g = Graph(x={"wallet": rng.normal(size=(12, 3)).astype("float32")}, frozen={"cp": csr},
              xq=rng.normal(size=(12, 3)).astype("float32"), query={"cp": csr})
    ckpt = tmp_path / "model.pt"
    result = train_hetero(g, ["cp"], rng.normal(size=(12, 1)),
                          {"train": np.arange(8), "val": np.arange(8, 12)}, 0,
                          dict(device="cpu", hidden=8, batch=2, eval_batch=2, epochs=2, eval_reps=1),
                          objectives=["reg"], checkpoint_path=ckpt, log=lambda *a: None)
    assert result.shape == (12, 1) and np.isfinite(result).all()
    info = json.loads(ckpt.with_suffix(".json").read_text())
    assert info["parameters"] > 0 and info["elapsed_seconds"] > 0 and info["peak_cuda_allocated_bytes"] == 0


def test_completed_job_invalidated_by_code_inputs_settings_or_checkpoint(tmp_path):
    src, data, pred, checkpoint = [tmp_path / p for p in ("src.py", "input", "pred.parquet", "model.pt")]
    for p in (src, data, pred, checkpoint):
        p.write_text("a")
    sig = job_signature({"hidden": 64}, [src], [data])
    finish_job(pred, sig, [checkpoint])
    assert completed_job(pred, sig)
    assert not completed_job(pred, job_signature({"hidden": 128}, [src], [data]))
    src.write_text("b")
    assert not completed_job(pred, job_signature({"hidden": 64}, [src], [data]))
    data.write_text("new input")
    assert not completed_job(pred, job_signature({"hidden": 64}, [src], [data]))
    checkpoint.unlink()
    assert not completed_job(pred, sig)


def test_heterogeneous_batch_with_no_market_or_wallet_neighbours():
    empty = np.array([], dtype=np.int64)
    wc = CSR.build(empty, empty, np.zeros((0, 1)), 3)
    wh = CSR.build(empty, empty, np.zeros((0, 5)), 3)
    mh = CSR.build(empty, empty, np.zeros((0, 5)), 2)
    g = Graph(x={"wallet": np.zeros((3, 4), np.float32), "market": np.zeros((2, 2), np.float32)},
              frozen={"cp": wc, "held": mh, "rev_held": wh},
              xq=np.ones((3, 4), np.float32), query={"cp": wc, "rev_held": wh})
    b = sample_batch(g, np.arange(3), ["cp", "held", "rev_held"], (2, 2), np.random.default_rng(1))
    model = HeteroSAGE({"wallet": 4, "market": 2}, ["cp", "held", "rev_held"], d_h=8, n_out=1)
    result = model(b).detach().numpy()
    assert result.shape == (3, 1) and np.isfinite(result).all()


def test_partial_download_is_rejected(tmp_path):
    p = tmp_path / "trades/year=2025/month=01/day=01/data.parquet"
    p.parent.mkdir(parents=True)
    p.touch()
    assert trade_files(tmp_path, "2025-01-01", "2025-01-02") == [str(p)]
    with pytest.raises(FileNotFoundError, match="2025-01-02"):
        trade_files(tmp_path, "2025-01-01", "2025-01-03")
