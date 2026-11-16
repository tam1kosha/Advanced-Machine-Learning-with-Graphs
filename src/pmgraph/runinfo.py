"""Small, torch-free helpers for reproducible, restartable training jobs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def job_signature(settings: dict, sources: list[Path], inputs: list[Path]) -> str:
    """Hash code/config and input identity. A changed input invalidates completed-job reuse."""
    h = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for p in sorted(sources):
        h.update(str(p).encode())
        h.update(p.read_bytes())
    for p in sorted(inputs):
        st = p.stat()
        h.update(f"{p}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()


def completed_job(prediction: Path, signature: str) -> bool:
    manifest = prediction.with_suffix(".json")
    if not prediction.is_file() or not manifest.is_file():
        return False
    try:
        data = json.loads(manifest.read_text())
        return (data["signature"] == signature and data["prediction_bytes"] == prediction.stat().st_size
                and all(Path(p).is_file() for p in data.get("checkpoints", [])))
    except (OSError, ValueError, KeyError):
        return False


def finish_job(prediction: Path, signature: str, checkpoints: list[Path]) -> None:
    manifest = prediction.with_suffix(".json")
    tmp = manifest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(signature=signature, prediction_bytes=prediction.stat().st_size,
                                  checkpoints=[str(p.resolve()) for p in checkpoints]), indent=2) + "\n")
    tmp.replace(manifest)
