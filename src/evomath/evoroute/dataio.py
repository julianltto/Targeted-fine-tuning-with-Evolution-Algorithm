"""EvoRoute-LoRA data IO: load the 50-adapter cached test outputs, build the
prediction/correctness matrices, and manage the frozen D_ea/D_select/D_final split.

All members were trained on GSM8K train and never saw test, so carving the 1319
test set into ea/select/final leaks nothing to the adapters; the aggregator never
touches `final`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from evomath.harness.parse import parse_gold
from scripts.optimize_track4_ensemble import load_bundle

SEED_SPLIT = 2026
TEST_DIRS = [
    "results/evomath/track4_sgd_ens_eval_test_8shot",       # seeds 0-24
    "results/evomath/track4_sgd_ens_more_eval_test_8shot",  # seeds 25-49
    "results/evomath/track4_sgd_ens_50_100_eval_test_8shot_chunk0",   # seeds 50-59
    "results/evomath/track4_sgd_ens_50_100_eval_test_8shot_chunk10",  # seeds 60-69
    "results/evomath/track4_sgd_ens_50_100_eval_test_8shot_chunk20",  # seeds 70-79
    "results/evomath/track4_sgd_ens_50_100_eval_test_8shot_chunk30",  # seeds 80-89
    "results/evomath/track4_sgd_ens_50_100_eval_test_8shot_chunk40",  # seeds 90-99
]
SPLIT_PATH = "data/evoroute_split.json"
TOL = 1e-4
_SGD_RE = re.compile(r"_sgd_(\d+)$")


def _seed_of(name: str) -> int:
    m = _SGD_RE.search(name)
    return int(m.group(1)) if m else 10 ** 9


def load_population(cfg) -> tuple[list[str], list[dict], np.ndarray, np.ndarray, np.ndarray]:
    """Return (members, docs, preds[M,N], gold[N], base_preds[N]).

    preds/base_preds hold the parsed final answer (rounded to 4dp) or NaN when the
    member produced no parseable number. docs/columns are aligned to the fixed
    1319-doc test order shared by both bundles.
    """
    docs = None
    gold = None
    base = None
    member_vals: dict[str, list] = {}
    for root in TEST_DIRS:
        _, _, _, dcs, vals = load_bundle(cfg, [root])
        if docs is None:
            docs = dcs
            gold = np.array([parse_gold(d["gold_text"]) for d in dcs], dtype=float)
        else:
            assert len(dcs) == len(docs), "test bundles disagree on doc count"
        for k, v in vals.items():
            if k == "base":
                base = v if base is None else base
            else:
                member_vals[k] = v
    members = sorted(member_vals, key=_seed_of)
    N = len(docs)
    M = len(members)
    preds = np.full((M, N), np.nan)
    for i, k in enumerate(members):
        for q, v in enumerate(member_vals[k]):
            if v is not None:
                preds[i, q] = round(float(v), 4)
    base_preds = np.full(N, np.nan)
    if base is not None:
        for q, v in enumerate(base):
            if v is not None:
                base_preds[q] = round(float(v), 4)
    return members, docs, preds, gold, base_preds


def correct_matrix(preds: np.ndarray, gold: np.ndarray, tol: float = TOL) -> np.ndarray:
    """C[M,N] bool: member i produced the gold answer on question q."""
    return (~np.isnan(preds)) & (np.abs(preds - gold[None, :]) <= tol)


def make_split(n: int, seed: int = SEED_SPLIT, fracs=(0.5, 0.25, 0.25)) -> dict:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_ea = int(round(fracs[0] * n))
    n_sel = int(round(fracs[1] * n))
    return {
        "ea_pos": sorted(order[:n_ea].tolist()),
        "select_pos": sorted(order[n_ea:n_ea + n_sel].tolist()),
        "final_pos": sorted(order[n_ea + n_sel:].tolist()),
    }


def build_split(docs: list[dict], path: str = SPLIT_PATH, seed: int = SEED_SPLIT) -> dict:
    N = len(docs)
    sp = make_split(N, seed)
    out = {"seed": seed, "n": N, "fracs": [0.5, 0.25, 0.25], **sp}
    for name in ("ea", "select", "final"):
        out[f"{name}_index"] = [int(docs[p]["index"]) for p in sp[f"{name}_pos"]]
    # sanity: disjoint and covering
    allpos = out["ea_pos"] + out["select_pos"] + out["final_pos"]
    assert len(allpos) == N and len(set(allpos)) == N, "split not a partition"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out))
    return out


def load_split(path: str = SPLIT_PATH) -> dict:
    return json.loads(Path(path).read_text())


def clean_correct(preds: np.ndarray, gold: np.ndarray, positions, members_idx=None,
                  tol: float = TOL) -> np.ndarray:
    """Uniform plurality vote (ties -> lowest value) correctness per position, in the
    given position order. Used for hard-case flags and majority-retention metrics."""
    if members_idx is None:
        members_idx = np.arange(preds.shape[0])
    out = np.zeros(len(positions), dtype=bool)
    for t, q in enumerate(positions):
        cands = candidates_for(preds[members_idx, q], tol)
        if not cands:
            continue
        best = max(cands.items(), key=lambda kv: (int(kv[1].sum()), -kv[0]))
        out[t] = abs(best[0] - gold[q]) <= tol
    return out


def candidates_for(preds_col: np.ndarray, tol: float = TOL) -> dict[float, np.ndarray]:
    """Map each distinct (rounded) candidate answer -> boolean supporter mask over members."""
    out: dict[float, np.ndarray] = {}
    valid = ~np.isnan(preds_col)
    vals = np.round(preds_col[valid], 4)
    idx = np.where(valid)[0]
    for v in np.unique(vals):
        out[float(v)] = np.zeros(len(preds_col), dtype=bool)
    for j, v in zip(idx, vals):
        out[float(v)][j] = True
    return out
