"""Family-level overlap analysis (core experiment 3)."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .io import OverlapData
from .coverage import greedy_coverage


def family_indices(data: OverlapData) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for i, iid in enumerate(data.intervention_ids):
        out[data.meta_by_id[iid].family].append(i)
    return dict(out)


def family_union_rescue(data: OverlapData) -> dict[str, np.ndarray]:
    R = data.rescue_matrix()
    out: dict[str, np.ndarray] = {}
    for fam, idxs in family_indices(data).items():
        out[fam] = np.any(R[idxs], axis=0) if idxs else np.zeros(R.shape[1], dtype=bool)
    return out


def family_union_hurt(data: OverlapData) -> dict[str, np.ndarray]:
    H = data.hurt_matrix()
    out: dict[str, np.ndarray] = {}
    for fam, idxs in family_indices(data).items():
        out[fam] = np.any(H[idxs], axis=0) if idxs else np.zeros(H.shape[1], dtype=bool)
    return out


def family_overlap_matrix(data: OverlapData) -> pd.DataFrame:
    fams = list(family_indices(data).keys())
    unions = family_union_rescue(data)
    M = np.zeros((len(fams), len(fams)))
    for i, a in enumerate(fams):
        for j, b in enumerate(fams):
            inter = int(np.logical_and(unions[a], unions[b]).sum())
            union = int(np.logical_or(unions[a], unions[b]).sum())
            M[i, j] = inter / union if union > 0 else 0.0
    return pd.DataFrame(M, index=fams, columns=fams)


def family_summary(
    data: OverlapData,
    K: int = 10,
    lambda_harm: float = 1.0,
    global_family: str = "global",
) -> pd.DataFrame:
    R = data.rescue_matrix()
    H = data.hurt_matrix()
    fam_idx = family_indices(data)
    fam_union_R = family_union_rescue(data)
    fam_union_H = family_union_hurt(data)

    global_rescue = fam_union_R.get(global_family, np.zeros(R.shape[1], dtype=bool))

    rows = []
    for fam, idxs in fam_idx.items():
        # best single
        sizes_R = R[idxs].sum(axis=1)
        sizes_H = H[idxs].sum(axis=1)
        best_single = int((sizes_R - lambda_harm * sizes_H).max()) if idxs else 0
        best_single_rescue = int(sizes_R.max()) if idxs else 0
        # greedy union within family
        sel = greedy_coverage(data, K=K, lambda_harm=lambda_harm, candidate_indices=idxs)
        greedy_R = int(np.any(R[sel], axis=0).sum()) if sel else 0
        greedy_H = int(np.any(H[sel], axis=0).sum()) if sel else 0
        # vs global
        f_union = fam_union_R[fam]
        overlap_global = (
            float(np.logical_and(f_union, global_rescue).sum())
            / max(int(np.logical_or(f_union, global_rescue).sum()), 1)
        )
        unique_over_global = int(np.logical_and(f_union, ~global_rescue).sum())
        rows.append({
            "family": fam,
            "n_interventions": len(idxs),
            "best_single_rescue": best_single_rescue,
            "best_single_net": best_single,
            f"greedy_union_rescue_K{K}": greedy_R,
            f"greedy_union_hurt_K{K}": greedy_H,
            "union_rescue_total": int(f_union.sum()),
            "union_hurt_total": int(fam_union_H[fam].sum()),
            "jaccard_vs_global": overlap_global,
            "unique_over_global": unique_over_global,
        })
    return pd.DataFrame(rows).set_index("family")
