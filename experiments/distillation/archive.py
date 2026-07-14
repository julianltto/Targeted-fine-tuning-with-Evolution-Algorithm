"""Pick the intervention archive used for mining (spec §3).

Two flavours:
- ``safe_archive``  : maximise rescue coverage while penalising hurt (test-time use)
- ``mining_archive``: maximise rescue coverage with small hurt penalty (data collection)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.overlap.io import OverlapData


@dataclass
class ArchiveResult:
    selected: list[str]
    rescue_union: list[int]
    hurt_union: list[int]

    def net(self, lam: float) -> list[float]:
        return [r - lam * h for r, h in zip(self.rescue_union, self.hurt_union)]


def _greedy(
    data: OverlapData,
    K: int,
    lam: float,
    candidate_indices: list[int] | None = None,
    drop_zero_effect: bool = True,
) -> ArchiveResult:
    R = data.rescue_matrix()
    H = data.hurt_matrix()
    n_iv, n_x = R.shape
    pool = set(range(n_iv)) if candidate_indices is None else set(candidate_indices)
    if drop_zero_effect:
        sizes = R.sum(axis=1) + H.sum(axis=1)
        pool = {i for i in pool if sizes[i] > 0}

    covered_R = np.zeros(n_x, dtype=bool)
    covered_H = np.zeros(n_x, dtype=bool)
    selected_ids: list[str] = []
    rescue_union: list[int] = []
    hurt_union: list[int] = []
    while pool and len(selected_ids) < K:
        best, best_score = None, -np.inf
        for idx in pool:
            novel_R = int(np.logical_and(R[idx], ~covered_R).sum())
            novel_H = int(np.logical_and(H[idx], ~covered_H).sum())
            score = novel_R - lam * novel_H
            if score > best_score:
                best, best_score = idx, score
        if best is None:
            break
        covered_R |= R[best]
        covered_H |= H[best]
        selected_ids.append(data.intervention_ids[best])
        rescue_union.append(int(covered_R.sum()))
        hurt_union.append(int(covered_H.sum()))
        pool.discard(best)
    return ArchiveResult(selected_ids, rescue_union, hurt_union)


def safe_archive(data: OverlapData, K: int = 5, lam: float = 1.0,
                 drop_zero_effect: bool = True) -> ArchiveResult:
    """Conservative archive for test-time intervention (high hurt penalty)."""
    return _greedy(data, K, lam, drop_zero_effect=drop_zero_effect)


def mining_archive(data: OverlapData, K: int = 10, lam: float = 0.1,
                   drop_zero_effect: bool = True) -> ArchiveResult:
    """Permissive archive for data collection (low hurt penalty: hurt examples are
    filtered downstream by the verifier, so we trade hurt for coverage)."""
    return _greedy(data, K, lam, drop_zero_effect=drop_zero_effect)


def composite_archive(
    data: OverlapData,
    must_include: list[str],
    K: int = 10,
    lam: float = 0.3,
) -> ArchiveResult:
    """Seed with a fixed must-include list (e.g. global_s1.050) and greedily extend."""
    name_to_row = {iid: i for i, iid in enumerate(data.intervention_ids)}
    seeds = [name_to_row[i] for i in must_include if i in name_to_row]

    R = data.rescue_matrix()
    H = data.hurt_matrix()
    n_iv, n_x = R.shape
    covered_R = np.zeros(n_x, dtype=bool)
    covered_H = np.zeros(n_x, dtype=bool)
    selected_ids: list[str] = []
    rescue_union: list[int] = []
    hurt_union: list[int] = []
    for s in seeds:
        covered_R |= R[s]
        covered_H |= H[s]
        selected_ids.append(data.intervention_ids[s])
        rescue_union.append(int(covered_R.sum()))
        hurt_union.append(int(covered_H.sum()))

    pool = set(range(n_iv)) - set(seeds)
    while pool and len(selected_ids) < K:
        best, best_score = None, -np.inf
        for idx in pool:
            novel_R = int(np.logical_and(R[idx], ~covered_R).sum())
            novel_H = int(np.logical_and(H[idx], ~covered_H).sum())
            score = novel_R - lam * novel_H
            if score > best_score:
                best, best_score = idx, score
        if best is None:
            break
        covered_R |= R[best]
        covered_H |= H[best]
        selected_ids.append(data.intervention_ids[best])
        rescue_union.append(int(covered_R.sum()))
        hurt_union.append(int(covered_H.sum()))
        pool.discard(best)
    return ArchiveResult(selected_ids, rescue_union, hurt_union)
