"""evo-math: evolutionary optimization tracks for Llama-3.2-1B math ability.

Shared infrastructure lives in `harness/`; per-track code in `track*/`.
See CLAUDE.md (constraints) and DESIGN.md (research design) at repo root.
"""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "configs/base.yaml") -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(p)
    return cfg
