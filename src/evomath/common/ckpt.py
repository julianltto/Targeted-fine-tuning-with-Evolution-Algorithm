"""Checkpoint IO: individuals are deltas on disk too (design constraint 3)."""

from __future__ import annotations

import pickle
from pathlib import Path

import torch

from evomath import REPO_ROOT


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_delta(obj, path: str | Path) -> Path:
    p = _resolve(path)
    torch.save(obj, p)
    return p


def load_delta(path: str | Path):
    return torch.load(_resolve(path), weights_only=False)


def save_pickle(obj, path: str | Path) -> Path:
    p = _resolve(path)
    with open(p, "wb") as f:
        pickle.dump(obj, f)
    return p


def load_pickle(path: str | Path):
    with open(_resolve(path), "rb") as f:
        return pickle.load(f)
