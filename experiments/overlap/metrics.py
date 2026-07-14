"""Pairwise overlap metrics.

All matrices below are intervention × intervention. They are computed
vectorised over the per-sample boolean rescue / hurt matrices from
``io.OverlapData``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from .io import OverlapData


def _jaccard_from_bool(A: np.ndarray) -> np.ndarray:
    """Pairwise Jaccard over rows of boolean matrix A (shape (I, N))."""
    Af = A.astype(np.float32)
    inter = Af @ Af.T                            # (I, I)
    sizes = Af.sum(axis=1)                       # (I,)
    union = sizes[:, None] + sizes[None, :] - inter
    out = np.zeros_like(inter)
    nz = union > 0
    out[nz] = inter[nz] / union[nz]
    return out


def rescue_jaccard(data: OverlapData) -> np.ndarray:
    return _jaccard_from_bool(data.rescue_matrix())


def hurt_jaccard(data: OverlapData) -> np.ndarray:
    return _jaccard_from_bool(data.hurt_matrix())


def changed_jaccard(data: OverlapData) -> np.ndarray:
    R = data.rescue_matrix()
    H = data.hurt_matrix()
    return _jaccard_from_bool(R | H)


def rescue_intersection_size(data: OverlapData) -> np.ndarray:
    R = data.rescue_matrix().astype(np.int32)
    return R @ R.T


def rescue_lift(data: OverlapData) -> np.ndarray:
    """Null-adjusted overlap.

    Expected intersection under random sampling within baseline-wrong universe:

        E[|R_i ∩ R_j|] = |R_i| * |R_j| / M

    Lift = observed / expected. Values are NaN where either rescue set is empty.
    """
    R = data.rescue_matrix()
    M = int(data.wrong_universe.sum())
    inter = R.astype(np.int32) @ R.astype(np.int32).T
    sizes = R.sum(axis=1).astype(np.float64)
    expected = (sizes[:, None] * sizes[None, :]) / max(M, 1)
    lift = np.full(expected.shape, np.nan, dtype=np.float64)
    mask = expected > 0
    lift[mask] = inter[mask] / expected[mask]
    return lift


def conditional_overlap(data: OverlapData) -> np.ndarray:
    """O[i, j] = |R_i ∩ R_j| / |R_i|."""
    R = data.rescue_matrix().astype(np.int32)
    inter = R @ R.T
    sizes = R.sum(axis=1).astype(np.float64)
    out = np.full_like(inter, fill_value=np.nan, dtype=np.float64)
    nz = sizes > 0
    out[nz, :] = inter[nz, :] / sizes[nz, None]
    return out


def effect_cosine(data: OverlapData) -> np.ndarray:
    """Cosine similarity between signed effect vectors."""
    E = data.effect_matrix().astype(np.float32)
    norms = np.linalg.norm(E, axis=1)
    dot = E @ E.T
    denom = norms[:, None] * norms[None, :]
    out = np.zeros_like(dot)
    nz = denom > 0
    out[nz] = dot[nz] / denom[nz]
    return out


def hypergeometric_pvalues(data: OverlapData) -> np.ndarray:
    """One-sided survival p-value for pairwise rescue intersection."""
    R = data.rescue_matrix()
    inter = R.astype(np.int32) @ R.astype(np.int32).T
    sizes = R.sum(axis=1).astype(int)
    M = int(data.wrong_universe.sum())
    I = data.I
    pvals = np.ones((I, I), dtype=np.float64)
    for i in range(I):
        for j in range(I):
            a = sizes[i]
            b = sizes[j]
            k = int(inter[i, j])
            if a == 0 or b == 0 or M == 0:
                continue
            pvals[i, j] = hypergeom.sf(k - 1, M, a, b)
    return pvals


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR on the off-diagonal upper triangle.

    Returns a matrix of q-values with the same shape; diagonal entries are 0.
    """
    I = pvals.shape[0]
    iu = np.triu_indices(I, k=1)
    p_flat = pvals[iu]
    n = p_flat.size
    order = np.argsort(p_flat)
    ranked = p_flat[order]
    q_ranked = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity from the largest rank down
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_flat = np.empty_like(q_ranked)
    q_flat[order] = np.clip(q_ranked, 0, 1)
    Q = np.zeros_like(pvals)
    Q[iu] = q_flat
    Q = Q + Q.T
    return Q


@dataclass
class OverlapMatrices:
    intervention_ids: list[str]
    rescue_jaccard: np.ndarray
    hurt_jaccard: np.ndarray
    changed_jaccard: np.ndarray
    rescue_intersection: np.ndarray
    rescue_lift: np.ndarray
    conditional_overlap: np.ndarray
    effect_cosine: np.ndarray
    hypergeom_p: np.ndarray
    hypergeom_q: np.ndarray

    def to_long_dataframe(self) -> pd.DataFrame:
        ids = self.intervention_ids
        rows = []
        for i, a in enumerate(ids):
            for j, b in enumerate(ids):
                if i >= j:
                    continue
                rows.append({
                    "i": a, "j": b,
                    "J_R": float(self.rescue_jaccard[i, j]),
                    "J_H": float(self.hurt_jaccard[i, j]),
                    "J_S": float(self.changed_jaccard[i, j]),
                    "inter_R": int(self.rescue_intersection[i, j]),
                    "lift_R": float(self.rescue_lift[i, j]),
                    "cond_i_to_j": float(self.conditional_overlap[i, j]),
                    "cond_j_to_i": float(self.conditional_overlap[j, i]),
                    "cos_E": float(self.effect_cosine[i, j]),
                    "p_hyper": float(self.hypergeom_p[i, j]),
                    "q_hyper": float(self.hypergeom_q[i, j]),
                })
        return pd.DataFrame(rows)


def compute_overlap_matrices(data: OverlapData) -> OverlapMatrices:
    p = hypergeometric_pvalues(data)
    return OverlapMatrices(
        intervention_ids=list(data.intervention_ids),
        rescue_jaccard=rescue_jaccard(data),
        hurt_jaccard=hurt_jaccard(data),
        changed_jaccard=changed_jaccard(data),
        rescue_intersection=rescue_intersection_size(data),
        rescue_lift=rescue_lift(data),
        conditional_overlap=conditional_overlap(data),
        effect_cosine=effect_cosine(data),
        hypergeom_p=p,
        hypergeom_q=bh_fdr(p),
    )
