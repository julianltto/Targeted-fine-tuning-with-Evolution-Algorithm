"""evo-math: evolutionary optimization tracks for Llama-3.2-1B math ability.

Shared infrastructure lives in `harness/`; per-track code in `track*/`.
See CLAUDE.md (constraints) and DESIGN.md (research design) at repo root.
"""

import os
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TMP_DIR = REPO_ROOT.parent / "tmp" / "evo-math"


def configure_tmpdir(path: str | Path | None = None) -> Path:
    """Keep Python/library temporary files under the iailab71 home tree."""
    tmp_dir = Path(path or os.environ.get("EVOMATH_TMPDIR") or DEFAULT_TMP_DIR)
    tmp_dir = tmp_dir.expanduser().resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["EVOMATH_TMPDIR"] = str(tmp_dir)
    os.environ["TMPDIR"] = str(tmp_dir)
    os.environ["TEMP"] = str(tmp_dir)
    os.environ["TMP"] = str(tmp_dir)
    tempfile.tempdir = str(tmp_dir)
    return tmp_dir


TMP_DIR = configure_tmpdir()


def load_config(path: str | Path = "configs/base.yaml") -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["tmp_dir"] = str(configure_tmpdir(cfg.get("tmp_dir")))
    cfg["_config_path"] = str(p)
    return cfg
