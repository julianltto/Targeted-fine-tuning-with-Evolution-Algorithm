"""Pairwise combination / synergy analysis (core experiment 4).

A "combined" intervention I+J is expected to have been *re-evaluated* by the
runner — the resulting per-sample correctness must be saved under an
``intervention_meta`` entry whose ``extra`` field declares the two component
IDs::

    InterventionMeta(
        intervention_id="combo:abc+def",
        family="combo",
        scale=...,
        parameter_group="cluster_A+cluster_B",
        extra={"combo_of": ["abc", "def"]},
    )

This module just compares the per-sample sets; it does not itself fuse masks.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from .io import OverlapData


@dataclass
class SynergyRecord:
    i: str
    j: str
    combo: str
    syn_rescue: int        # |R_combo \ (R_i ∪ R_j)|
    lost_rescue: int       # |(R_i ∪ R_j) \ R_combo|
    new_hurt: int          # |H_combo \ (H_i ∪ H_j)|
    net_synergy: float
    rescue_union: int
    rescue_combo: int


def compute_synergy(
    data: OverlapData,
    lambda_harm: float = 1.0,
) -> pd.DataFrame:
    """Find every meta record whose `extra.combo_of` references two known interventions.

    For each such combo, compare ``R_combo`` against ``R_i ∪ R_j``.
    """
    id_to_row = {iid: row for row, iid in enumerate(data.intervention_ids)}
    R = data.rescue_matrix()
    H = data.hurt_matrix()

    records: list[SynergyRecord] = []
    for combo_id, meta in data.meta_by_id.items():
        combo_of = meta.extra.get("combo_of") if isinstance(meta.extra, dict) else None
        if not combo_of or len(combo_of) != 2:
            continue
        i_id, j_id = combo_of
        if i_id not in id_to_row or j_id not in id_to_row or combo_id not in id_to_row:
            continue
        i = id_to_row[i_id]
        j = id_to_row[j_id]
        c = id_to_row[combo_id]

        R_union = R[i] | R[j]
        H_union = H[i] | H[j]

        syn = int(np.logical_and(R[c], ~R_union).sum())
        lost = int(np.logical_and(R_union, ~R[c]).sum())
        new_hurt = int(np.logical_and(H[c], ~H_union).sum())
        net = syn - lost - lambda_harm * new_hurt

        records.append(SynergyRecord(
            i=i_id, j=j_id, combo=combo_id,
            syn_rescue=syn, lost_rescue=lost, new_hurt=new_hurt,
            net_synergy=float(net),
            rescue_union=int(R_union.sum()),
            rescue_combo=int(R[c].sum()),
        ))
    return pd.DataFrame([r.__dict__ for r in records])


def select_top_pairs_to_test(
    data: OverlapData,
    top_n: int = 20,
    strategy: str = "unique_rescue",
    reference_id: str | None = None,
) -> list[tuple[str, str]]:
    """Pick which pairs are worth evaluating as combos.

    Strategies:
      'unique_rescue' — pick interventions whose rescue set has the most
        elements not covered by any other intervention, then form pairs.
      'low_overlap_to_global' — pick interventions with lowest Jaccard to
        ``reference_id`` (typically global scale 1.05).
      'diverse_families' — round-robin across families.
    """
    R = data.rescue_matrix()
    ids = data.intervention_ids
    if strategy == "unique_rescue":
        # how much of an intervention's rescue is unique within the pool?
        any_other = np.zeros_like(R)
        for i in range(len(ids)):
            mask = np.ones(len(ids), dtype=bool)
            mask[i] = False
            any_other[i] = np.any(R[mask], axis=0)
        unique = (R & ~any_other).sum(axis=1)
        picked = np.argsort(-unique)[:top_n]
        return [(ids[a], ids[b]) for a, b in combinations(picked.tolist(), 2)]

    if strategy == "low_overlap_to_global":
        if reference_id is None or reference_id not in ids:
            raise ValueError("low_overlap_to_global requires a valid reference_id")
        ref = ids.index(reference_id)
        # raw Jaccard to reference
        from .metrics import _jaccard_from_bool
        J = _jaccard_from_bool(R)
        scores = J[ref]
        scores[ref] = np.inf  # exclude self
        picked = np.argsort(scores)[:top_n]
        return [(ids[ref], ids[int(b)]) for b in picked if int(b) != ref]

    if strategy == "diverse_families":
        from collections import defaultdict
        buckets: dict[str, list[str]] = defaultdict(list)
        for iid in ids:
            buckets[data.meta_by_id[iid].family].append(iid)
        out: list[tuple[str, str]] = []
        fams = list(buckets.keys())
        for a, b in combinations(fams, 2):
            for x in buckets[a][:3]:
                for y in buckets[b][:3]:
                    out.append((x, y))
                    if len(out) >= top_n:
                        return out
        return out

    raise ValueError(f"unknown strategy {strategy!r}")
