"""Driver that turns InterventionConfig objects into per-sample correctness files.

It reuses the existing math/calibration masks and lm-eval pipeline; only the
selector logic (which math-only positions to scale) and the post-evaluation
correctness extraction are new.
"""
from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from mathneuro.core import apply_mask_to_model
from mathneuro.ea_search import backup_weights, restore_weights

from .intervention_configs import (
    InterventionConfig,
    layer_of,
    proj_type_of,
)
from .io import (
    InterventionMeta,
    baseline_path,
    load_per_sample,
    meta_path,
    per_sample_path,
    save_meta,
    save_per_sample,
)


# Selectors -----------------------------------------------------------------


def _math_only_masks(
    math_important: dict[str, torch.Tensor],
    calib_important: dict[str, torch.Tensor],
    exclude_substring: str = "embed",
) -> dict[str, torch.Tensor]:
    """Per-layer boolean mask of (math & ~calib)."""
    out: dict[str, torch.Tensor] = {}
    for name, m in math_important.items():
        if exclude_substring in name:
            out[name] = torch.zeros_like(m, dtype=torch.bool)
            continue
        out[name] = m & (~calib_important[name])
    return out


def _selector_for_config(
    config: InterventionConfig,
    math_only: dict[str, torch.Tensor],
    math_important: dict[str, torch.Tensor],
    importance_scores: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return a dict layer_name -> bool tensor, selecting positions to scale.

    `math_only` is (math & ~calib) per layer; that is the "intervention
    universe" for every selector kind except ``nonmath_random_subset``, which
    samples from the complement of ``math_important`` so the negative control
    is *entirely outside* the math-neuron set.
    """
    kind = config.selector_kind
    args = config.selector_args
    sel: dict[str, torch.Tensor] = {n: torch.zeros_like(t, dtype=torch.bool) for n, t in math_only.items()}

    if kind == "all":
        for n, t in math_only.items():
            sel[n] = t.clone()
        return sel

    if kind == "layer_set":
        layers = set(int(l) for l in args["layers"])
        for n, t in math_only.items():
            li = layer_of(n)
            if li is not None and li in layers:
                sel[n] = t.clone()
        return sel

    if kind == "proj_type_set":
        proj_types = set(args["proj_types"])
        for n, t in math_only.items():
            pt = proj_type_of(n)
            if pt is not None and pt in proj_types:
                sel[n] = t.clone()
        return sel

    if kind == "causality_quantile":
        # Vectorised: concatenate |importance| for every math-only position,
        # then take a single global top-k across layers. Avoids the per-position
        # Python loop, which would OOM on ~50M positions for a 1B model.
        top_fraction = float(args["top_fraction"])
        per_layer_scores: list[torch.Tensor] = []
        per_layer_local_idx: list[torch.Tensor] = []
        per_layer_names: list[str] = []
        for n, t in math_only.items():
            if not t.any():
                continue
            score = importance_scores.get(n)
            if score is None:
                continue
            flat_imp = score.abs().reshape(-1)
            flat_mask = t.reshape(-1)
            local_idx = torch.nonzero(flat_mask, as_tuple=False).flatten()
            per_layer_local_idx.append(local_idx)
            per_layer_scores.append(flat_imp[local_idx])
            per_layer_names.append(n)
        if not per_layer_scores:
            return sel
        all_scores = torch.cat(per_layer_scores)
        total = int(all_scores.numel())
        keep = int(total * top_fraction)
        if keep <= 0:
            return sel
        keep = min(keep, total)
        _, topk_global = torch.topk(all_scores, keep, largest=True)
        chosen = torch.zeros(total, dtype=torch.bool)
        chosen[topk_global] = True
        offset = 0
        for n, local_idx, scores_chunk in zip(per_layer_names, per_layer_local_idx, per_layer_scores):
            size = int(scores_chunk.numel())
            picked = chosen[offset:offset + size]
            offset += size
            if not picked.any():
                continue
            sel[n].view(-1)[local_idx[picked]] = True
        return sel

    if kind == "random_math_subset":
        fraction = float(args["fraction"])
        seed = int(args["seed"])
        rng = np.random.default_rng(seed)
        for n, t in math_only.items():
            if not t.any():
                continue
            flat = t.reshape(-1)
            idx = torch.nonzero(flat, as_tuple=False).flatten().cpu().numpy()
            keep = int(len(idx) * fraction)
            if keep <= 0:
                continue
            picks = rng.choice(idx, size=keep, replace=False)
            flat_sel = sel[n].reshape(-1)
            flat_sel[picks.tolist()] = True
        return sel

    if kind == "nonmath_random_subset":
        # Sample from positions strictly outside math_important so the
        # negative control shares no parameters with any math-cluster.
        fraction = float(args["fraction"])
        seed = int(args["seed"])
        rng = np.random.default_rng(seed)
        for n, t in math_only.items():
            math_imp = math_important.get(n)
            if math_imp is None:
                continue
            flat_imp = math_imp.reshape(-1)
            non_math_idx = torch.nonzero(~flat_imp, as_tuple=False).flatten().cpu().numpy()
            # Match the size of the corresponding math-only group on this layer
            # so the control is comparable across layers.
            target_size = int(t.sum().item())
            keep = int(target_size or len(non_math_idx) * fraction)
            keep = min(keep, len(non_math_idx))
            if keep <= 0:
                continue
            picks = rng.choice(non_math_idx, size=keep, replace=False)
            flat_sel = sel[n].reshape(-1)
            flat_sel[picks.tolist()] = True
        return sel

    if kind == "combo":
        a = InterventionConfig(**args["a"])
        b = InterventionConfig(**args["b"])
        sel_a = _selector_for_config(a, math_only, math_important, importance_scores)
        sel_b = _selector_for_config(b, math_only, math_important, importance_scores)
        for n in sel:
            sel[n] = sel_a[n] | sel_b[n]
        return sel

    raise ValueError(f"unknown selector kind {kind!r}")


# Evaluation glue -----------------------------------------------------------


def _sample_exact_match(sample: dict, prefer_filter: str = "strict-match") -> float:
    """Robust extraction of a 0/1 exact-match flag from one lm_eval sample.

    lm_eval encodes metrics either as a flat field ``exact_match`` or, when a
    task declares multiple filters (e.g. gsm8k_cot has strict-match and
    flexible-extract), as ``exact_match,<filter_name>``. Fall back to a
    nested ``metrics`` dict for older versions.
    """
    if "exact_match" in sample and not isinstance(sample["exact_match"], (list, dict)):
        return float(sample["exact_match"])
    preferred_key = f"exact_match,{prefer_filter}"
    if preferred_key in sample:
        return float(sample[preferred_key])
    for key in sorted(sample.keys()):
        if key.startswith("exact_match,"):
            try:
                return float(sample[key])
            except (TypeError, ValueError):
                continue
    metrics = sample.get("metrics") or {}
    if "exact_match" in metrics:
        return float(metrics["exact_match"])
    raise KeyError(
        f"no exact_match field found in sample; available keys={list(sample.keys())}"
    )


def _extract_per_sample_correctness(
    lm_eval_results: dict,
    task: str,
    prefer_filter: str = "strict-match",
) -> tuple[list[str], np.ndarray]:
    """Pull (problem_id, correct) from lm_eval's per-sample log.

    Tasks with multiple filters (e.g. gsm8k_cot has strict-match and
    flexible-extract) emit one sample dict per (doc, filter) — so a 256-doc
    evaluation produces 512 entries. We dedup by doc_id and keep the entry
    whose ``filter`` field matches ``prefer_filter``; if no entry advertises
    a filter we fall back to first-occurrence.
    """
    samples = lm_eval_results.get("samples", {}).get(task)
    if samples is None:
        raise ValueError(f"task {task!r} produced no per-sample log; pass log_samples=True")

    by_doc: dict[str, tuple[bool, bool]] = {}  # doc_id -> (correct, is_preferred)
    for k, s in enumerate(samples):
        pid = str(s.get("doc_id", k))
        em = _sample_exact_match(s, prefer_filter=prefer_filter)
        is_preferred = (s.get("filter") == prefer_filter)
        correct_bit = bool(int(round(em)))
        prev = by_doc.get(pid)
        if prev is None or (is_preferred and not prev[1]):
            by_doc[pid] = (correct_bit, is_preferred)

    ids = list(by_doc.keys())
    correct = np.array([by_doc[i][0] for i in ids], dtype=bool)
    return ids, correct


def run_lm_eval_per_sample(
    model: nn.Module,
    tokenizer,
    task: str,
    eval_subset: int,
    random_state: int,
    batch_size: int | str = 1,
) -> tuple[list[str], np.ndarray]:
    """Wrapper around lm_eval.simple_evaluate that always returns per-sample marks."""
    from lm_eval import simple_evaluate
    from lm_eval.tasks import TaskManager

    results = simple_evaluate(
        model="hf",
        model_args={"pretrained": model, "dtype": "bfloat16", "tokenizer": tokenizer, "max_length": 2048},
        tasks=[task],
        task_manager=TaskManager(),
        log_samples=True,
        batch_size=batch_size,
        limit=eval_subset,
        random_seed=random_state,
    )
    if results is None:
        raise RuntimeError("lm_eval.simple_evaluate returned None")
    return _extract_per_sample_correctness(results, task)


# Main loop -----------------------------------------------------------------


def run_baseline(
    out_root: Path,
    model: nn.Module,
    tokenizer,
    task: str,
    eval_subset: int,
    random_state: int,
    batch_size: int | str = 1,
    overwrite: bool = False,
) -> tuple[list[str], np.ndarray]:
    out_root = Path(out_root)
    bp = baseline_path(out_root)
    if bp.exists() and not overwrite:
        df = load_per_sample(bp)
        return df["problem_id"].astype(str).tolist(), df["correct"].to_numpy(dtype=bool)
    pids, correct = run_lm_eval_per_sample(
        model, tokenizer, task, eval_subset, random_state, batch_size
    )
    save_per_sample(bp, pids, correct)
    return pids, correct


def run_intervention(
    out_root: Path,
    config: InterventionConfig,
    model: nn.Module,
    tokenizer,
    math_important: dict[str, torch.Tensor],
    calib_important: dict[str, torch.Tensor],
    importance_scores: dict[str, torch.Tensor],
    task: str,
    eval_subset: int,
    random_state: int,
    batch_size: int | str = 1,
    weight_backup: dict[str, torch.Tensor] | None = None,
    overwrite: bool = False,
) -> tuple[list[str], np.ndarray]:
    out_root = Path(out_root)
    target = per_sample_path(out_root, config.intervention_id)
    if target.exists() and not overwrite:
        df = load_per_sample(target)
        return df["problem_id"].astype(str).tolist(), df["correct"].to_numpy(dtype=bool)

    math_only = _math_only_masks(math_important, calib_important)
    sel = _selector_for_config(config, math_only, math_important, importance_scores)

    if weight_backup is None:
        weight_backup = backup_weights(model)

    apply_mask_to_model(model, sel, factor=config.scale)
    try:
        pids, correct = run_lm_eval_per_sample(
            model, tokenizer, task, eval_subset, random_state, batch_size
        )
    finally:
        restore_weights(model, weight_backup)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    save_per_sample(target, pids, correct)
    return pids, correct


def run_pool(
    out_root: Path,
    configs: list[InterventionConfig],
    model: nn.Module,
    tokenizer,
    math_important: dict[str, torch.Tensor],
    calib_important: dict[str, torch.Tensor],
    importance_scores: dict[str, torch.Tensor],
    task: str,
    eval_subset: int,
    random_state: int,
    batch_size: int | str = 1,
    overwrite: bool = False,
    progress_log: Callable[[str], None] | None = None,
) -> None:
    out_root = Path(out_root)
    weight_backup = backup_weights(model)
    metas: list[InterventionMeta] = []
    if meta_path(out_root).exists() and not overwrite:
        from .io import load_meta
        existing = {m.intervention_id: m for m in load_meta(out_root)}
    else:
        existing = {}

    log = progress_log or print
    for k, cfg in enumerate(configs):
        t0 = time.time()
        run_intervention(
            out_root, cfg, model, tokenizer,
            math_important, calib_important, importance_scores,
            task, eval_subset, random_state, batch_size,
            weight_backup=weight_backup, overwrite=overwrite,
        )
        existing[cfg.intervention_id] = cfg.meta()
        save_meta(out_root, list(existing.values()))
        log(f"[{k + 1}/{len(configs)}] {cfg.intervention_id} done in {time.time() - t0:.1f}s")

    metas = list(existing.values())
    save_meta(out_root, metas)
