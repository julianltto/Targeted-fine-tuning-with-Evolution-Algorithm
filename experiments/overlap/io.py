"""I/O for per-sample correctness records.

A single experiment directory looks like:

    <root>/
      baseline.csv                           # columns: problem_id, correct
      interventions/
        <intervention_id>.csv                # columns: problem_id, correct
        ...
      intervention_meta.json                 # list of dicts, one per intervention

All overlap analysis works off these files; the runner is the only piece that
touches the model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class InterventionMeta:
    intervention_id: str
    family: str                    # 'global' | 'layer' | 'layer_window' | 'cluster' | 'causality_bucket' | 'module_type' | 'random_math' | 'nonmath'
    scale: float
    parameter_group: str           # human-readable description
    extra: dict[str, Any] = field(default_factory=dict)


def per_sample_path(root: Path, intervention_id: str) -> Path:
    return Path(root) / "interventions" / f"{intervention_id}.csv"


def baseline_path(root: Path) -> Path:
    return Path(root) / "baseline.csv"


def meta_path(root: Path) -> Path:
    return Path(root) / "intervention_meta.json"


def save_per_sample(path: Path, problem_ids: list[str], correct: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "problem_id": [str(p) for p in problem_ids],
        "correct": correct.astype(int),
    })
    df.to_csv(path, index=False)


def load_per_sample(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["correct"] = df["correct"].astype(bool)
    df["problem_id"] = df["problem_id"].astype(str)
    # Defensive: older runs emitted one row per (doc, filter); keep only the
    # first occurrence per problem_id so downstream reindex never collides.
    if df["problem_id"].duplicated().any():
        df = df.drop_duplicates(subset="problem_id", keep="first").reset_index(drop=True)
    return df


def save_meta(root: Path, metas: list[InterventionMeta]) -> None:
    meta_path(root).parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path(root), "w") as f:
        json.dump([asdict(m) for m in metas], f, indent=2)


def load_meta(root: Path) -> list[InterventionMeta]:
    with open(meta_path(root)) as f:
        raw = json.load(f)
    return [InterventionMeta(**r) for r in raw]


@dataclass
class OverlapData:
    """In-memory representation suitable for vectorised overlap math.

    Attributes
    ----------
    problem_ids : list[str]
        Order of samples corresponds to columns of `correct`.
    baseline : np.ndarray, shape (N,), dtype=bool
        Baseline correctness.
    intervention_ids : list[str]
    correct : np.ndarray, shape (I, N), dtype=bool
        correct[i, x] = intervention i is correct on sample x.
    meta_by_id : dict[str, InterventionMeta]
    """
    problem_ids: list[str]
    baseline: np.ndarray
    intervention_ids: list[str]
    correct: np.ndarray
    meta_by_id: dict[str, InterventionMeta]

    @property
    def N(self) -> int:
        return len(self.problem_ids)

    @property
    def I(self) -> int:
        return len(self.intervention_ids)

    @property
    def wrong_universe(self) -> np.ndarray:
        """Boolean mask of baseline-wrong samples, shape (N,)."""
        return ~self.baseline

    def rescue_matrix(self) -> np.ndarray:
        """R[i, x] = True iff baseline wrong and intervention correct."""
        return self.correct & (~self.baseline)[None, :]

    def hurt_matrix(self) -> np.ndarray:
        """H[i, x] = True iff baseline correct and intervention wrong."""
        return (~self.correct) & self.baseline[None, :]

    def effect_matrix(self) -> np.ndarray:
        """E[i, x] in {-1, 0, +1} as defined in section 4.3."""
        E = np.zeros(self.correct.shape, dtype=np.int8)
        E[self.rescue_matrix()] = 1
        E[self.hurt_matrix()] = -1
        return E


def load_overlap_data(root: Path) -> OverlapData:
    root = Path(root)
    base = load_per_sample(baseline_path(root))
    base = base.sort_values("problem_id").reset_index(drop=True)
    problem_ids = base["problem_id"].astype(str).tolist()
    baseline = base["correct"].to_numpy(dtype=bool)
    baseline_pid_set = set(problem_ids)

    metas = load_meta(root)
    intervention_ids: list[str] = []
    rows = []
    for m in metas:
        p = per_sample_path(root, m.intervention_id)
        if not p.exists():
            continue
        df_raw = load_per_sample(p)
        # Hard-fail on intervention vs baseline size mismatch — silently
        # truncating the intervention to the baseline's problem_ids would mask
        # config drift between runs (e.g., baseline collected at limit=256
        # while interventions ran at limit=1024).
        iv_pid_set = set(df_raw["problem_id"].astype(str).tolist())
        if iv_pid_set != baseline_pid_set:
            extra = iv_pid_set - baseline_pid_set
            missing = baseline_pid_set - iv_pid_set
            raise ValueError(
                f"{m.intervention_id}: problem_id set differs from baseline "
                f"(baseline={len(baseline_pid_set)}, intervention={len(iv_pid_set)}, "
                f"only-in-intervention={len(extra)}, only-in-baseline={len(missing)}); "
                f"re-run baseline at the same eval_dataset_subset as the interventions"
            )
        df = df_raw.set_index("problem_id").reindex(problem_ids)
        intervention_ids.append(m.intervention_id)
        rows.append(df["correct"].to_numpy(dtype=bool))

    correct = np.stack(rows, axis=0) if rows else np.zeros((0, len(problem_ids)), dtype=bool)
    meta_by_id = {m.intervention_id: m for m in metas if m.intervention_id in intervention_ids}
    return OverlapData(
        problem_ids=problem_ids,
        baseline=baseline,
        intervention_ids=intervention_ids,
        correct=correct,
        meta_by_id=meta_by_id,
    )
