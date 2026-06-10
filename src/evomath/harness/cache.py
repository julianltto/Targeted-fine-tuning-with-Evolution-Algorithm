"""Fitness cache: individual-parameter hash -> score (M0).

Key = sha256 over (sorted tensor names, shapes, raw bytes) + fitness kind +
eval-indices file hash. Same individual evaluated twice -> second call is a
dict lookup (<1s acceptance requirement). JSON on local NVMe (never /tmp).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from evomath import REPO_ROOT


def individual_key(named_tensors, kind: str, indices_hash: str) -> str:
    h = hashlib.sha256()
    h.update(kind.encode())
    h.update(indices_hash.encode())
    for name, t in named_tensors:
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().to(torch.float32).cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def file_hash(path: str | Path) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


class FitnessCache:
    def __init__(self, cache_path: str | Path):
        self.path = Path(cache_path)
        if not self.path.is_absolute():
            self.path = REPO_ROOT / self.path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, float] = {}
        if self.path.exists():
            self._data = json.loads(self.path.read_text())
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> float | None:
        v = self._data.get(key)
        if v is not None:
            self.hits += 1
        return v

    def put(self, key: str, score: float) -> None:
        self.misses += 1
        self._data[key] = score
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data))
        tmp.replace(self.path)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
