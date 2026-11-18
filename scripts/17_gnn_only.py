"""Train/evaluate only GraphSAGE and heterogeneous GraphSAGE on prepared first-k graphs.

Example: python scripts/17_gnn_only.py --tag trial01 --gpus 0 1 2 3 4 5 6 7 --hidden 128 --lr .001
Use --stage train while tuning; --stage test evaluates the stored fixed test wallets without fitting.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pmgraph import load_config
from pmgraph.gnn_only import report_holdout, validate_hparams
from pmgraph.runinfo import job_signature
from server_check import gpu_tokens

HP_FLAGS = {"hidden": int, "lr": float, "dropout": float, "weight_decay": float, "grad_clip": float,
            "batch": int, "eval_batch": int, "epochs": int, "patience": int, "eval_reps": int, "threads": int}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True, help="unique experiment name; identical settings can resume")
    p.add_argument("--stage", choices=["train", "test", "train-test"], default="train-test")
    p.add_argument("--models", nargs="+", choices=["sage", "hgnn"], default=["sage", "hgnn"])
    p.add_argument("--splits", "--periods", nargs="+", choices=["train", "test"], default=["train", "test"],
                   help="dataset periods; each gets its own wallet holdout (not out-of-time transfer)")
    p.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10, 20, 50])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="initialization seeds, not CV folds")
    p.add_argument("--split-seed", type=int, default=0, help="keep fixed across hyperparameter trials")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--gpus", nargs="+", help="physical ids/UUIDs from your visible allocation; default all visible")
    p.add_argument("--workers", type=int, help="parallel jobs; at most one per selected GPU (CPU default 1)")
    p.add_argument("--settings", type=Path, help="optional YAML/JSON mapping of hyperparameters")
    for name, typ in HP_FLAGS.items():
        p.add_argument("--" + name.replace("_", "-"), type=typ)
    p.add_argument("--fanout", nargs=2, type=int, help="neighbours per relation at hops 1/2")
    p.add_argument("--bootstrap", type=int, default=200, help="wallet bootstrap repetitions; 0 gives point estimates only")
    p.add_argument("--no-resume", action="store_true", help="refit even matching completed jobs")
    p.add_argument("--dry-run", action="store_true", help="show plan/commands without requiring data or GPUs")
    return p


def make_plan(args, cfg: dict) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        raise ValueError("tag may contain only letters, numbers, underscores and hyphens")
    for name in ("models", "splits", "ks", "seeds"):
        values = getattr(args, name)
        if not values or len(values) != len(set(values)):
            raise ValueError(f"{name} must be nonempty and unique")
    if set(args.ks) - set(cfg["eval"]["k_buckets"]) or min(args.seeds) < 0 or args.split_seed < 0:
        raise ValueError("ks must be prepared k_buckets; seeds must be nonnegative")
    hp = {k: v for k, v in cfg["gnn"].items() if k in HP_FLAGS or k == "fanout"}
    hp.update(dropout=.2, weight_decay=1e-4, grad_clip=1.0)
    if args.settings:
        supplied = yaml.safe_load(args.settings.read_text())
        if not isinstance(supplied, dict) or set(supplied) - (set(HP_FLAGS) | {"fanout"}):
            raise ValueError("settings must be a mapping of supported hyperparameters; see --help")
        hp.update(supplied)
    for key in [*HP_FLAGS, "fanout"]:
        value = getattr(args, key)
        if value is not None:
            hp[key] = value
    validate_hparams(hp)
    hp["device"] = args.device
    versions = {p: importlib.metadata.version(p) for p in ["torch", "torch-geometric", "numpy", "pandas", "scipy",
                                                          "scikit-learn", "pyarrow"]}
    return dict(tag=args.tag, protocol="fixed_wallet_holdout_60_20_20", models=args.models,
                splits=args.splits, ks=args.ks, seeds=args.seeds, split_seed=args.split_seed, hp=hp, packages=versions)


def devices_for(args) -> list[str]:
    if args.device == "cpu":
        if args.gpus:
            raise ValueError("--gpus cannot be used with --device cpu")
        return ["cpu"] * (args.workers or 1)
    if args.dry_run:
        return (args.gpus or ["auto"])[ :args.workers]
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Install matching CUDA torch/driver, or explicitly use --device cpu.")
    available = gpu_tokens(torch.cuda.device_count(), os.environ.get("CUDA_VISIBLE_DEVICES"), None)
    chosen = args.gpus or available
    if len(chosen) != len(set(chosen)) or set(chosen) - set(available):
        raise ValueError(f"--gpus must be unique ids/UUIDs within CUDA_VISIBLE_DEVICES allocation {available}")
    if args.workers and args.workers > len(chosen):
        raise ValueError("--workers exceeds selected GPUs; each worker needs its own card")
    chosen = chosen[:args.workers]
    for token in chosen:
        i = available.index(token)
        x = torch.ones(4, device=f"cuda:{i}", requires_grad=True)
        x.square().sum().backward()
        torch.cuda.synchronize(i)
    return chosen


def atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def run_jobs(jobs: list, devices: list, plan: dict, out: Path, resume: bool) -> bool:
    pending = queue.Queue()
    for job in jobs:
        pending.put(job)
    active, results = {}, []
    lock, stop = threading.Lock(), threading.Event()

    def worker(token):
        while not stop.is_set():
            try:
                split, k, model = pending.get_nowait()
            except queue.Empty:
                return
            logfile = out / f"{split}_k{k}_{model}.log"
            cmd = [sys.executable, "-u", str(ROOT / "scripts/09_first_k_experiments.py"), split, model, str(k), "holdout", "edge"]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "" if token == "cpu" else token,
                   "PMGRAPH_REQUIRE_CUDA": "0" if token == "cpu" else "1", "PMGRAPH_TAG": plan["tag"],
                   "PMGRAPH_GNN": json.dumps(plan["hp"]), "PMGRAPH_SEEDS": json.dumps(plan["seeds"]),
                   "PMGRAPH_SPLIT_SEED": str(plan["split_seed"]), "PMGRAPH_RESUME": "1" if resume else "0",
                   "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": str(plan["hp"]["threads"]),
                   "OPENBLAS_NUM_THREADS": str(plan["hp"]["threads"])}
            print(f"START {split} k={k} {model} device={token}; log={logfile}", flush=True)
            try:
                with logfile.open("a") as stream:
                    with lock:
                        if stop.is_set():
                            return
                        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                                start_new_session=True)
                        active[proc.pid] = proc
                    code = proc.wait()
                    with lock:
                        active.pop(proc.pid, None)
            except OSError as exc:
                code = -1
                with logfile.open("a") as stream:
                    stream.write(f"Launcher error: {exc}\n")
            with lock:
                results.append(dict(split=split, k=k, model=model, device=token, returncode=code, log=str(logfile)))
                atomic_json(out / "jobs.json", results)
            print(f"{'DONE' if code == 0 else 'FAILED'} {split} k={k} {model}; log={logfile}", flush=True)

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    old_term = signal.signal(signal.SIGTERM, interrupted)
    pool = ThreadPoolExecutor(max_workers=len(devices))
    futures = [pool.submit(worker, token) for token in devices]
    try:
        for future in futures:
            future.result()
    except BaseException:
        stop.set()
        with lock:
            running = list(active.values())
        for proc in running:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for proc in running:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
        raise
    finally:
        pool.shutdown(wait=True)
        signal.signal(signal.SIGTERM, old_term)
    return all(r["returncode"] == 0 for r in results) and len(results) == len(jobs)


def main() -> int:
    cli = parser()
    args = cli.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        raise ValueError("tag may contain only letters, numbers, underscores and hyphens")
    if args.stage == "test":
        extras = {word.split("=", 1)[0] for word in sys.argv[1:] if word.startswith("--")} - {
            "--stage", "--tag", "--bootstrap", "--dry-run"}
        if extras:
            cli.error(f"--stage test uses the saved run settings; omit training options {sorted(extras)}")
    if args.bootstrap < 0 or (args.workers is not None and args.workers < 1):
        raise ValueError("bootstrap must be nonnegative and workers positive")
    cfg = load_config()
    out = ROOT / "outputs/logs/gnn" / args.tag
    manifest_path = out / "run.json"
    if args.stage == "test":
        # Report-only reads the exact saved settings, not today's CLI defaults, and never fits.
        saved = json.loads(manifest_path.read_text())
        plan = saved["plan"]
    else:
        plan = make_plan(args, cfg)
    processed = ROOT / cfg["data"]["processed_dir"]
    inputs = [processed / f"{split}_{suffix}" for split in plan["splits"] for suffix in
              ["hetero.pt", "fk_pop.parquet", "fk_feat.parquet", "fk_obs.parquet", "fk_mf.parquet", "fk_halves.parquet"]]
    jobs = [(s, k, m) for s in plan["splits"] for k in plan["ks"] for m in plan["models"]]
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        print(f"Planned {len(jobs)} independent model/k/period jobs; devices={devices_for(args)}")
        for s, k, m in jobs:
            print(f"09_first_k_experiments.py {s} {m} {k} holdout edge")
        return 0
    missing = [str(p) for p in inputs if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"missing prepared inputs: {missing}. Prepare steps 07/08/12 before this GNN-only run.")
    sources = [Path(__file__), ROOT / "scripts/09_first_k_experiments.py", ROOT / "configs/mvp.yaml",
               *sorted((ROOT / "src/pmgraph").glob("*.py"))]
    signature = job_signature(plan, sources, inputs)
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old["signature"] != signature:
            raise ValueError("this tag belongs to different settings/code/data; use a new --tag to preserve the previous trial")
    out.mkdir(parents=True, exist_ok=True)
    if args.stage != "test":
        devices = devices_for(args)
        atomic_json(manifest_path, dict(plan=plan, signature=signature, devices=devices,
                                       python=sys.version, executable=sys.executable))
        if not run_jobs(jobs, devices, plan, out, resume=not args.no_resume):
            print(f"Some jobs failed. Inspect {out / 'jobs.json'} and the per-job logs; no test report generated.", flush=True)
            return 1
        report_holdout(processed, out, plan, "val", args.bootstrap, ROOT / "outputs/tables")
    if args.stage in ("test", "train-test"):
        report_holdout(processed, out, plan, "test", args.bootstrap, ROOT / "outputs/tables")
    print(f"Completed {args.stage}: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
