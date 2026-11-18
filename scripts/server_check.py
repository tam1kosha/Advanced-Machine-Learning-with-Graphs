"""Server preflight; --smoke trains tiny models on each selected device without trade data."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def gpu_tokens(count: int, visible: str | None, requested: str | None) -> list[str]:
    """Preserve ids/UUIDs assigned by CUDA_VISIBLE_DEVICES when creating workers."""
    tokens = visible.split(",") if visible is not None else [str(i) for i in range(count)]
    tokens = [s.strip() for s in tokens if s.strip() and s.strip() != "-1"]
    if len(tokens) != count:
        raise ValueError(f"torch sees {count} GPUs but CUDA_VISIBLE_DEVICES describes {len(tokens)}; check allocation")
    n = int(requested) if requested is not None else count
    if n < 1 or n > count:
        raise ValueError(f"NGPU must be between 1 and the {count} visible GPUs, got {n}")
    return tokens[:n]


def smoke(device: str, directory: Path) -> None:
    import numpy as np
    import torch
    from pmgraph.hetero import CSR, Graph, HeteroSAGE, train_hetero
    from pmgraph.tabular import run_in_subprocess

    rng = np.random.default_rng(12)
    nw, nm, nq = 32, 6, 48
    cp = rng.integers(0, nw, (2, 120))
    held = np.stack([rng.integers(0, nw, 70), rng.integers(0, nm, 70)])
    mf = rng.integers(0, nw, (2, 60))
    frozen = {"cp": CSR.build(cp[1], cp[0], np.ones(120), nw),
              "held": CSR.build(held[1], held[0], np.ones((70, 5)), nm),
              "rev_held": CSR.build(held[0], held[1], np.ones((70, 5)), nw),
              "mf": CSR.build(mf[1], mf[0], np.ones(60), nw),
              "rev_mf": CSR.build(mf[0], mf[1], np.ones(60), nw)}
    dst = np.repeat(np.arange(nq), 2)
    query = {r: CSR.build(dst, rng.integers(0, nm if r == "rev_held" else nw, len(dst)),
                         np.ones((len(dst), 5 if r == "rev_held" else 1)), nq)
             for r in ["cp", "mf", "rev_mf", "rev_held"]}
    g = Graph(x={"wallet": rng.normal(size=(nw, 4)).astype("float32"),
                 "market": rng.normal(size=(nm, 3)).astype("float32")},
              frozen=frozen, xq=rng.normal(size=(nq, 4)).astype("float32"), query=query)
    hp = dict(device=device, hidden=8, fanout=[2, 2], batch=8, eval_batch=8, epochs=2, patience=2,
              eval_reps=1, threads=1)
    idx = {"train": np.arange(32), "val": np.arange(32, 40)}
    for method, rels in [("sage", ["cp"]), ("hgnn", ["cp", "mf", "rev_mf", "held", "rev_held"])]:
        for objective in ("binary", "reg"):
            y = (np.arange(nq) % 2)[:, None].astype("float32") if objective == "binary" else g.xq[:, :1].copy()
            path = directory / f"{device.replace(':', '_')}_{method}_{objective}.pt"
            s = train_hetero(g, rels, y, idx, 0, hp, objectives=[objective], checkpoint_path=path)
            if s.shape != (nq, 1) or not np.isfinite(s).all():
                raise RuntimeError(f"{method} {objective}: invalid smoke predictions")
            c = torch.load(path, map_location="cpu", weights_only=False)
            model = HeteroSAGE(c["in_dims"], c["rels"], c["hidden"], c["n_layers"], c["n_out"], c["dropout"])
            model.load_state_dict(c["state_dict"])
    y = np.c_[np.arange(nq) % 2, g.xq[:, 0]]
    result = run_in_subprocess(g.xq, y, {0: (np.arange(32), np.arange(32, nq))}, objectives=["binary", "reg"])
    if not np.isfinite(result[0]).all():
        raise RuntimeError("XGBoost smoke predictions are nonfinite")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-tokens", action="store_true", help="print worker CUDA device ids only")
    parser.add_argument("--smoke", action="store_true", help="train tiny models on every selected device")
    parser.add_argument("--stages", nargs="*", default=[])
    args = parser.parse_args()
    if not (3, 10) <= sys.version_info[:2] < (3, 13):
        raise RuntimeError("pmgraph needs Python 3.10–3.12; use a Python 3.11 virtual environment")
    import torch
    cpu = os.environ.get("PMGRAPH_ALLOW_CPU") == "1"
    need_gpu = bool(set(args.stages) & {"check", "edge", "grid", "smoke"}) or args.smoke or args.gpu_tokens
    if need_gpu and not cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install a CUDA-enabled torch build matching the driver; "
                           "PMGRAPH_ALLOW_CPU=1 is an explicit CPU-only debugging override")
    tokens = (["cpu"] if cpu else gpu_tokens(torch.cuda.device_count(), os.environ.get("CUDA_VISIBLE_DEVICES"),
                                            os.environ.get("NGPU"))) if need_gpu else []
    if args.gpu_tokens:
        print(" ".join(tokens))
        return
    packages = {p: importlib.metadata.version(p) for p in ["torch", "torch-geometric", "numpy", "pandas",
                "pyarrow", "duckdb", "scipy", "scikit-learn", "xgboost", "matplotlib", "pyyaml", "huggingface-hub"]}
    from pmgraph.hetero import HeteroSAGE  # noqa: F401
    import duckdb
    import pandas as pd
    import pyarrow  # noqa: F401
    subprocess.run([sys.executable, "-c", "from xgboost import XGBRegressor; print('XGBoost import OK')"], check=True)
    from pmgraph import load_config
    cfg = load_config()
    ks = [int(k) for k in os.environ.get("KS", " ".join(map(str, cfg["eval"]["k_buckets"]))).replace(",", " ").split()]
    splits = os.environ.get("SPLITS", "train test").split()
    methods = os.environ.get("METHODS", "hist,xgb_own,xgb_nbr,sage,hgnn").split(",")
    if not ks or len(ks) != len(set(ks)) or set(ks) - set(cfg["eval"]["k_buckets"]):
        raise ValueError("KS must be a nonempty, unique subset of eval.k_buckets")
    if not splits or len(splits) != len(set(splits)) or set(splits) - set(cfg["splits"]):
        raise ValueError("SPLITS must contain train and/or test once each")
    if not methods or set(methods) - {"hist", "xgb_own", "xgb_nbr", "sage", "hgnn"}:
        raise ValueError("invalid METHODS list")
    if os.environ.get("PMGRAPH_TAG") and not re.fullmatch(r"[A-Za-z0-9_-]+", os.environ["PMGRAPH_TAG"]):
        raise ValueError("PMGRAPH_TAG must contain only letters, numbers, underscores and hyphens")
    hp = {**cfg["gnn"], **json.loads(os.environ.get("PMGRAPH_GNN", "{}"))}
    allowed = {"hidden", "fanout", "batch", "eval_batch", "epochs", "patience", "lr", "eval_reps",
               "threads", "dropout", "device", "gpu_scale", "weight_decay", "grad_clip"}
    if set(hp) - allowed:
        raise ValueError(f"unknown GNN settings: {set(hp) - allowed}")
    if len(hp["fanout"]) != 2 or any(int(v) <= 0 for v in [*hp["fanout"], hp["hidden"], hp["batch"],
                 hp.get("eval_batch", hp["batch"]), hp["epochs"], hp["patience"], hp["threads"], hp["eval_reps"]]):
        raise ValueError("GNN sizes/counts must be positive; fanout must have two entries")
    if not 0 <= hp.get("dropout", 0.2) < 1 or hp["lr"] <= 0:
        raise ValueError("invalid dropout or learning rate")
    if hp.get("weight_decay", 1e-4) < 0 or hp.get("grad_clip", 0) < 0:
        raise ValueError("weight_decay and grad_clip must be nonnegative")
    if need_gpu and not cpu and hp.get("device") not in (None, "cuda", "cuda:0"):
        raise ValueError("GPU workers must use device=cuda:0 (or automatic); each sees one allocated GPU")
    seen = set()
    for stage in args.stages:
        needed = []
        if stage == "prep" and "data" not in seen:
            from pmgraph.prep import trade_files
            raw = ROOT / cfg["data"]["raw_dir"]
            for split in splits:
                sp = cfg["splits"][split]
                trade_files(raw, sp["graph_start"], sp["label_end"])
            needed = [raw / n for n in ["predictions.parquet", "markets.parquet"]]
        if stage in ("diag", "edge", "grid", "oot") and "prep" not in seen:
            check_splits = ["train", "test"] if stage == "oot" else (["train"] if stage == "grid" else splits)
            suffixes = ["fk_pop.parquet", "fk_feat.parquet"]
            if stage == "diag":
                suffixes += ["features.parquet", "calibration.parquet"]
                needed += [ROOT / cfg["data"]["interim_dir"] / f"{s}_{p}_fills.parquet"
                           for s in check_splits for p in ("graph", "label")]
            if stage in ("edge", "grid"):
                suffixes += ["hetero.pt", "fk_obs.parquet", "fk_mf.parquet"]
            needed += [ROOT / cfg["data"]["processed_dir"] / f"{s}_{p}" for s in check_splits for p in suffixes]
        if stage in ("edge", "grid") and "diag" not in seen:
            needed += [ROOT / cfg["data"]["processed_dir"] / f"{s}_fk_halves.parquet"
                       for s in (["train"] if stage == "grid" else splits)]
        missing = [str(p.relative_to(ROOT)) for p in needed if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"stage {stage}: missing inputs {missing}; run prerequisite stages first")
        seen.add(stage)
    for directory in ("outputs/logs", "outputs/tables", "outputs/figures", "data/processed", "data/tmp"):
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "data/tmp") as tmp:
        p = Path(tmp) / "check.parquet"
        pd.DataFrame({"x": [1, 2]}).to_parquet(p)
        with duckdb.connect() as con:
            if con.execute("SELECT sum(x) FROM read_parquet(?)", [str(p)]).fetchone()[0] != 3:
                raise RuntimeError("DuckDB/Parquet round trip failed")
    devices = []
    if tokens and tokens != ["cpu"]:
        for i, token in enumerate(tokens):
            with torch.cuda.device(i):
                x = torch.ones(16, device=f"cuda:{i}", requires_grad=True)
                x.square().sum().backward()
                torch.cuda.synchronize(i)
            devices.append(dict(token=token, name=torch.cuda.get_device_name(i),
                                vram_bytes=torch.cuda.get_device_properties(i).total_memory))
    manifest = dict(time=datetime.now(timezone.utc).isoformat(), python=sys.version, executable=sys.executable,
                    platform=platform.platform(), cpu_count=os.cpu_count(), packages=packages,
                    cuda_build=torch.version.cuda, devices=devices, disk_free_bytes=shutil.disk_usage(ROOT).free,
                    stages=args.stages, ks=ks, splits=splits, methods=methods, gnn=hp,
                    settings={k: os.environ[k] for k in ["PMGRAPH_TAG", "PMGRAPH_GNN", "PMGRAPH_RESUME",
                              "PMGRAPH_REQUIRE_CUDA", "PMGRAPH_ALLOW_CPU", "PMGRAPH_THREADS", "CUDA_VISIBLE_DEVICES",
                              "NGPU", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"] if k in os.environ})
    if Path("/proc/meminfo").is_file():
        mem = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
               if line.startswith(("MemTotal:", "MemAvailable:"))}
        manifest["host_memory_bytes"] = mem
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    manifest["git_commit"] = git.stdout.strip() if git.returncode == 0 else None
    manifest["cpu_model"] = platform.processor()
    if Path("/proc/cpuinfo").is_file():
        manifest["cpu_model"] = next((s.split(":", 1)[1].strip() for s in Path("/proc/cpuinfo").read_text().splitlines()
                                      if s.startswith("model name")), manifest["cpu_model"])
    if shutil.which("nvidia-smi"):
        driver = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"],
                                text=True, capture_output=True)
        manifest["nvidia_smi"] = driver.stdout.strip() if driver.returncode == 0 else driver.stderr.strip()
    print(json.dumps(manifest, indent=2), flush=True)
    record = ROOT / "outputs/logs" / f"preflight_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}.json"
    record.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Preflight OK: {record}", flush=True)
    if args.smoke:
        with tempfile.TemporaryDirectory(dir=ROOT / "data/tmp") as tmp:
            for device in (["cpu"] if cpu else [f"cuda:{i}" for i in range(len(tokens))]):
                smoke(device, Path(tmp))
        print("Smoke passed: GraphSAGE + hetero GraphSAGE, classification + regression, checkpoint reload, XGBoost")


if __name__ == "__main__":
    main()
