"""W&B wrapper: run name = {exp}-s{seed}-{git7}; project from configs/base.yaml."""

from __future__ import annotations

import subprocess

from evomath import REPO_ROOT


def git_short() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "nogit"


def init_run(cfg: dict, exp: str, seed: int, extra_config: dict | None = None,
             offline: bool = False):
    import wandb
    run = wandb.init(
        project=cfg["wandb_project"],
        name=f"{exp}-s{seed}-{git_short()}",
        config={"seed": seed, "exp": exp, **(extra_config or {})},
        mode="offline" if offline else "online",
    )
    return run
