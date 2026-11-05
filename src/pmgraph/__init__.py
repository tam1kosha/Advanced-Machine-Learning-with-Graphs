"""pmgraph: early trader typing on Polymarket trading graphs."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(name: str = "mvp") -> dict:
    with open(ROOT / "configs" / f"{name}.yaml") as f:
        return yaml.safe_load(f)
