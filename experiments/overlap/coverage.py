"""Archive selection and coverage curves (core experiment 2)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import OverlapData


@dataclass
class CoverageCurve:
    method: str
    selected_ids: list[str]
    rescue_union: list[int]
    hurt_union: list[int]

    def net(self, lambda_harm: float) -> list[float]:
        return [r - lambda_harm * h for r, h in zip(self.rescue_union, self.hurt_union)]


def _union_size(boolean_rows: np.ndarray, indices: list[int]) -> int:
    if not indices:
        return 0
    return int(np.any(boolean_rows[indices], axis=0).sum())


def topk_accuracy(data: OverlapData, K: int) -> list[int]:
    acc = data.correct.mean(axis=1)
    return list(np.argsort(-acc)[:K])


def topk_rescue(data: OverlapData, K: int) -> list[int]:
    R = data.rescue_matrix()
    sizes = R.sum(axis=1)
    return list(np.argsort(-sizes)[:K])


def greedy_coverage(
    data: OverlapData,
    K: int,
    lambda_harm: float = 1.0,
    candidate_indices: list[int] | None = None,
) -> list[int]:
    R = data.rescue_matrix()
    H = data.hurt_matrix()
    n = R.shape[0]
    pool = list(range(n)) if candidate_indices is None else list(candidate_indices)
    covered_R = np.zeros(R.shape[1], dtype=bool)
    covered_H = np.zeros(H.shape[1], dtype=bool)
    selected: list[int] = []
    remaining = set(pool)
    while remaining and len(selected) < K:
        best, best_score = None, -np.inf
        for idx in remaining:
            novel_R = int(np.logical_and(R[idx], ~covered_R).sum())
            novel_H = int(np.logical_and(H[idx], ~covered_H).sum())
            score = novel_R - lambda_harm * novel_H
            if score > best_score:
                best_score = score
                best = idx
        if best is None:
            break
        selected.append(best)
        covered_R |= R[best]
        covered_H |= H[best]
        remaining.discard(best)
    return selected


def random_archive(data: OverlapData, K: int, seed: int = 0) -> list[int]:
    rng = np.random.default_rng(seed)
    n = data.correct.shape[0]
    perm = rng.permutation(n)
    return list(perm[:K])


def diversity_archive(data: OverlapData, K: int) -> list[int]:
    """Farthest-point traversal on effect vectors."""
    E = data.effect_matrix().astype(np.float32)
    sizes = np.linalg.norm(E, axis=1)
    if sizes.max() == 0:
        return list(range(min(K, E.shape[0])))
    seed = int(np.argmax(sizes))
    selected = [seed]
    min_dist = np.linalg.norm(E - E[seed], axis=1)
    while len(selected) < K:
        nxt = int(np.argmax(min_dist))
        if min_dist[nxt] == 0:
            break
        selected.append(nxt)
        min_dist = np.minimum(min_dist, np.linalg.norm(E - E[nxt], axis=1))
    return selected


def coverage_curve(
    data: OverlapData,
    method: str,
    K: int,
    lambda_harm: float = 1.0,
    seed: int = 0,
) -> CoverageCurve:
    if method == "topk_accuracy":
        order = topk_accuracy(data, K)
    elif method == "topk_rescue":
        order = topk_rescue(data, K)
    elif method == "greedy":
        order = greedy_coverage(data, K, lambda_harm=lambda_harm)
    elif method == "random":
        order = random_archive(data, K, seed=seed)
    elif method == "diversity":
        order = diversity_archive(data, K)
    else:
        raise ValueError(f"unknown method {method!r}")

    R = data.rescue_matrix()
    H = data.hurt_matrix()
    rescue_union = []
    hurt_union = []
    for k in range(1, len(order) + 1):
        idx = order[:k]
        rescue_union.append(_union_size(R, idx))
        hurt_union.append(_union_size(H, idx))

    selected_ids = [data.intervention_ids[i] for i in order]
    return CoverageCurve(
        method=method,
        selected_ids=selected_ids,
        rescue_union=rescue_union,
        hurt_union=hurt_union,
    )


def permutation_null_coverage(
    data: OverlapData,
    K: int,
    n_perms: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """Greedy coverage under shuffled rescue labels (sizes preserved).

    Returns an array of shape (n_perms, K) with rescue-union counts.
    """
    rng = np.random.default_rng(seed)
    R = data.rescue_matrix()
    wrong_idx = np.where(data.wrong_universe)[0]
    M = wrong_idx.size
    out = np.zeros((n_perms, K), dtype=int)
    sizes = R[:, wrong_idx].sum(axis=1)
    I = R.shape[0]
    for t in range(n_perms):
        # shuffle each intervention's rescue positions within wrong universe
        Rp = np.zeros_like(R)
        for i in range(I):
            if sizes[i] == 0:
                continue
            picks = rng.choice(wrong_idx, size=int(sizes[i]), replace=False)
            Rp[i, picks] = True
        # greedy coverage on Rp ignoring hurt (null is rescue-only)
        covered = np.zeros(R.shape[1], dtype=bool)
        remaining = set(range(I))
        for k in range(K):
            best, best_gain = None, -1
            for idx in remaining:
                gain = int(np.logical_and(Rp[idx], ~covered).sum())
                if gain > best_gain:
                    best_gain = gain
                    best = idx
            if best is None:
                break
            covered |= Rp[best]
            remaining.discard(best)
            out[t, k] = int(covered.sum())
        # carry forward the last value if loop broke early
        for k in range(1, K):
            if out[t, k] == 0:
                out[t, k] = out[t, k - 1]
    return out
