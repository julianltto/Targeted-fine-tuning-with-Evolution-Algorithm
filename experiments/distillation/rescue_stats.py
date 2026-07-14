"""Per-example rescue statistics built from the overlap experiment's CSVs.

Definitions follow the CID spec, sections 1 and 11:

    r(x) = #{ I in archive : intervention I rescues x }

This module is pure pandas/numpy so it runs without torch — every figure below
section 11.1 is computable from the existing
``results/overlap_gsm8k/{baseline,interventions/*}.csv`` files.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.overlap.io import OverlapData, load_overlap_data


@dataclass
class BucketDef:
    name: str
    lo: int                # inclusive
    hi: int | float        # inclusive (use float('inf') for open upper)


DEFAULT_BUCKETS: tuple[BucketDef, ...] = (
    BucketDef("never_rescued", 0, 0),
    BucketDef("niche", 1, 1),
    BucketDef("medium", 2, 4),
    BucketDef("high_consensus", 5, float("inf")),
)


def rescue_frequency(data: OverlapData, archive_ids: list[str] | None = None) -> pd.DataFrame:
    """For every baseline-wrong problem, count how many archive interventions rescue it.

    Returns a DataFrame indexed by problem_id with columns:
        baseline_wrong : bool (always True for the returned rows)
        r              : int, rescue count
        rescuing_ids   : list[str], which interventions rescued this problem
        any_hurt       : int, how many archive interventions hurt this problem
                          (intervention correct on baseline-correct → wrong; for a
                           baseline-wrong row this is always 0, so we also record
                           per-problem hurt over the *complement*)
    """
    if archive_ids is None:
        archive_ids = list(data.intervention_ids)
    name_to_row = {iid: i for i, iid in enumerate(data.intervention_ids)}
    rows = [name_to_row[i] for i in archive_ids if i in name_to_row]

    R = data.rescue_matrix()[rows]      # (|A|, N)
    H = data.hurt_matrix()[rows]        # (|A|, N)

    wrong_mask = data.wrong_universe
    wrong_idx = np.where(wrong_mask)[0]
    r_counts = R[:, wrong_idx].sum(axis=0).astype(int)

    rescuing: list[list[str]] = []
    for j_local, j_global in enumerate(wrong_idx):
        rescued_by = [archive_ids[i] for i in range(R.shape[0]) if R[i, j_global]]
        rescuing.append(rescued_by)

    # hurt counts are defined on baseline-correct examples; for wrong rows we
    # record 0 here and compute the complementary table separately below.
    df = pd.DataFrame({
        "problem_id": [data.problem_ids[j] for j in wrong_idx],
        "baseline_correct": False,
        "r": r_counts,
        "rescuing_ids": rescuing,
    }).set_index("problem_id")
    return df


def hurt_frequency(data: OverlapData, archive_ids: list[str] | None = None) -> pd.DataFrame:
    """Complement of `rescue_frequency`: per baseline-correct problem, how many archive interventions hurt it."""
    if archive_ids is None:
        archive_ids = list(data.intervention_ids)
    name_to_row = {iid: i for i, iid in enumerate(data.intervention_ids)}
    rows = [name_to_row[i] for i in archive_ids if i in name_to_row]
    H = data.hurt_matrix()[rows]

    correct_idx = np.where(data.baseline)[0]
    h_counts = H[:, correct_idx].sum(axis=0).astype(int)
    hurting: list[list[str]] = []
    for j_global in correct_idx:
        hurting.append([archive_ids[i] for i in range(H.shape[0]) if H[i, j_global]])
    return pd.DataFrame({
        "problem_id": [data.problem_ids[j] for j in correct_idx],
        "baseline_correct": True,
        "h": h_counts,
        "hurting_ids": hurting,
    }).set_index("problem_id")


def assign_buckets(r_series: pd.Series, buckets: tuple[BucketDef, ...] = DEFAULT_BUCKETS) -> pd.Series:
    """Map a rescue-count series to a bucket name per spec §1 / §11.2."""
    out = pd.Series(index=r_series.index, dtype=object)
    for b in buckets:
        mask = (r_series >= b.lo) & (r_series <= b.hi)
        out.loc[mask] = b.name
    return out


def bucket_table(rescue_df: pd.DataFrame,
                 buckets: tuple[BucketDef, ...] = DEFAULT_BUCKETS) -> pd.DataFrame:
    df = rescue_df.copy()
    df["bucket"] = assign_buckets(df["r"], buckets)
    agg = (
        df.groupby("bucket")
        .agg(
            n_examples=("r", "size"),
            mean_rescue_count=("r", "mean"),
            max_rescue_count=("r", "max"),
        )
        .reindex([b.name for b in buckets], fill_value=0)
    )
    agg["fraction"] = agg["n_examples"] / max(int(agg["n_examples"].sum()), 1)
    return agg


def family_composition_by_bucket(
    rescue_df: pd.DataFrame,
    data: OverlapData,
    buckets: tuple[BucketDef, ...] = DEFAULT_BUCKETS,
) -> pd.DataFrame:
    """How often does each intervention family appear in each bucket's rescues?

    For every example, expand the ``rescuing_ids`` list into family votes; then
    aggregate per bucket. Useful to see whether high-consensus rescues are
    driven by one family or by structurally-diverse interventions.
    """
    df = rescue_df.copy()
    df["bucket"] = assign_buckets(df["r"], buckets)

    fam_counts: dict[tuple[str, str], int] = defaultdict(int)
    for _, row in df.iterrows():
        if not row["rescuing_ids"]:
            continue
        families_in_row = {
            data.meta_by_id[iid].family for iid in row["rescuing_ids"]
            if iid in data.meta_by_id
        }
        for fam in families_in_row:
            fam_counts[(row["bucket"], fam)] += 1

    all_fams = sorted({data.meta_by_id[i].family for i in data.intervention_ids})
    bucket_names = [b.name for b in buckets]
    out = pd.DataFrame(0, index=bucket_names, columns=all_fams, dtype=int)
    for (bucket_name, fam), c in fam_counts.items():
        out.at[bucket_name, fam] = c
    return out


def family_diversity_by_bucket(
    rescue_df: pd.DataFrame,
    data: OverlapData,
    buckets: tuple[BucketDef, ...] = DEFAULT_BUCKETS,
) -> pd.DataFrame:
    """Per-example: how many distinct families rescued this problem? Aggregated per bucket."""
    df = rescue_df.copy()
    df["bucket"] = assign_buckets(df["r"], buckets)

    def n_fams(ids: list[str]) -> int:
        return len({data.meta_by_id[i].family for i in ids if i in data.meta_by_id})

    df["n_families"] = df["rescuing_ids"].apply(n_fams)
    return (
        df.groupby("bucket")["n_families"]
        .agg(["mean", "median", "max", "count"])
        .reindex([b.name for b in buckets], fill_value=0)
    )


def jaccard_to_global(
    rescue_df: pd.DataFrame,
    data: OverlapData,
    global_ids: list[str],
    buckets: tuple[BucketDef, ...] = DEFAULT_BUCKETS,
) -> pd.DataFrame:
    """For each bucket: fraction of its problems also rescued by any 'global' intervention.

    A useful proxy for spec §11.2's "uniqueness over global" axis: high-consensus
    problems that are *not* solved by plain global scaling are the most valuable
    distillation targets.
    """
    name_to_row = {iid: i for i, iid in enumerate(data.intervention_ids)}
    g_rows = [name_to_row[g] for g in global_ids if g in name_to_row]
    if not g_rows:
        return pd.DataFrame()
    global_rescue = np.any(data.rescue_matrix()[g_rows], axis=0)
    global_rescued_pids = {
        data.problem_ids[j] for j in np.where(global_rescue)[0]
    }
    df = rescue_df.copy()
    df["bucket"] = assign_buckets(df["r"], buckets)
    df["rescued_by_global"] = df.index.isin(global_rescued_pids)
    out = (
        df.groupby("bucket")
        .agg(
            n=("rescued_by_global", "size"),
            n_rescued_by_global=("rescued_by_global", "sum"),
        )
        .reindex([b.name for b in buckets], fill_value=0)
    )
    out["fraction_rescued_by_global"] = out["n_rescued_by_global"] / out["n"].replace(0, np.nan)
    out["unique_over_global"] = out["n"] - out["n_rescued_by_global"]
    return out


def hurt_to_rescue_ratio_by_bucket(
    rescue_df: pd.DataFrame,
    data: OverlapData,
    archive_ids: list[str] | None = None,
    buckets: tuple[BucketDef, ...] = DEFAULT_BUCKETS,
) -> pd.DataFrame:
    """For each bucket: average per-rescuing-intervention hurt-to-rescue ratio.

    For every (problem, rescuing intervention) pair, look at |H_I| / |R_I| of
    the intervention itself. A bucket whose rescues come from low-hurt
    interventions is safer to distill from.
    """
    if archive_ids is None:
        archive_ids = list(data.intervention_ids)
    name_to_row = {iid: i for i, iid in enumerate(data.intervention_ids)}
    R = data.rescue_matrix()
    H = data.hurt_matrix()
    sizes_R = {iid: int(R[name_to_row[iid]].sum()) for iid in archive_ids if iid in name_to_row}
    sizes_H = {iid: int(H[name_to_row[iid]].sum()) for iid in archive_ids if iid in name_to_row}

    df = rescue_df.copy()
    df["bucket"] = assign_buckets(df["r"], buckets)

    def ratio(ids: list[str]) -> float:
        vals = []
        for iid in ids:
            if iid not in sizes_R or sizes_R[iid] == 0:
                continue
            vals.append(sizes_H[iid] / sizes_R[iid])
        return float(np.mean(vals)) if vals else float("nan")

    df["hurt_to_rescue"] = df["rescuing_ids"].apply(ratio)
    return (
        df.groupby("bucket")["hurt_to_rescue"]
        .agg(["mean", "median", "count"])
        .reindex([b.name for b in buckets], fill_value=np.nan)
    )


def save_rescue_metadata(
    rescue_df: pd.DataFrame,
    out_path: Path,
    buckets: tuple[BucketDef, ...] = DEFAULT_BUCKETS,
) -> Path:
    """Persist per-example rescue metadata (spec §4.3) as CSV.

    Columns: problem_id, r, bucket, rescuing_ids (semicolon-joined).
    """
    df = rescue_df.copy()
    df["bucket"] = assign_buckets(df["r"], buckets)
    df["rescuing_ids"] = df["rescuing_ids"].apply(lambda ids: ";".join(ids))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_csv(out_path, index=False)
    return out_path


def load_overlap_root(root: Path) -> OverlapData:
    return load_overlap_data(Path(root))
