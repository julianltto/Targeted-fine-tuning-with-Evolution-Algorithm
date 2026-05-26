"""Phase-2 trajectory mining.

For each (baseline-wrong) problem and each archive intervention, generate a
solution, verify it, and persist the verified-correct trajectories as the
distillation training pool.

Output schema (one JSONL line per rescued (problem, intervention) pair):

    {
      "problem_id": str,
      "intervention_id": str,
      "prompt": str,
      "gold": str,
      "baseline_solution": str,
      "baseline_correct": false,
      "intervention_solution": str,
      "intervention_correct": true,
      "answer_extracted": str
    }
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.overlap.io import baseline_path, load_per_sample, per_sample_path
from experiments.overlap.runner import (
    _math_only_masks,
    _selector_for_config,
    _sample_exact_match,
)
from mathneuro.core import apply_mask_to_model
from mathneuro.ea_search import backup_weights, restore_weights


def _samples_with_text(lm_eval_results: dict, task: str, prefer_filter: str = "strict-match"):
    """Return list of dicts with text fields, deduplicated per doc_id by preferred filter."""
    samples = lm_eval_results.get("samples", {}).get(task)
    if samples is None:
        raise RuntimeError(f"task {task!r} returned no per-sample log (set log_samples=True)")

    by_doc: dict[str, dict] = {}
    for k, s in enumerate(samples):
        pid = str(s.get("doc_id", k))
        em = _sample_exact_match(s, prefer_filter=prefer_filter)
        is_pref = (s.get("filter") == prefer_filter)
        record = {
            "doc_id": pid,
            "prompt": _stringify(s.get("arguments")),
            "gold": str(s.get("target", "")),
            "raw_response": _first_resp(s.get("resps")),
            "filtered_response": _first_resp(s.get("filtered_resps")),
            "exact_match": bool(int(round(em))),
            "filter": s.get("filter"),
        }
        prev = by_doc.get(pid)
        if prev is None or (is_pref and prev.get("filter") != prefer_filter):
            by_doc[pid] = record
    return by_doc


def _stringify(arguments) -> str:
    if arguments is None:
        return ""
    if isinstance(arguments, (list, tuple)) and arguments:
        first = arguments[0]
        return first[0] if isinstance(first, (list, tuple)) else str(first)
    return str(arguments)


def _first_resp(resps) -> str:
    if not resps:
        return ""
    cur = resps[0]
    while isinstance(cur, (list, tuple)) and cur:
        cur = cur[0]
    return str(cur) if cur is not None else ""


def run_lm_eval_with_text(model, tokenizer, task: str, eval_subset: int, random_state: int,
                          batch_size: int | str = 1) -> dict[str, dict]:
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
        raise RuntimeError("simple_evaluate returned None")
    return _samples_with_text(results, task)


def mine_trajectories(
    out_path: Path,
    model,
    tokenizer,
    archive_meta: list[dict],
    math_important: dict[str, torch.Tensor],
    calib_important: dict[str, torch.Tensor],
    importance_scores: dict[str, torch.Tensor],
    task: str,
    eval_subset: int,
    random_state: int,
    batch_size: int | str = 1,
    baseline_csv: Path | None = None,
) -> Path:
    """Run baseline + each archive intervention, persisting verified-correct
    (problem, intervention) trajectories as JSONL."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[mining] baseline pass…")
    baseline_records = run_lm_eval_with_text(model, tokenizer, task, eval_subset, random_state, batch_size)

    wrong_pids = {pid for pid, r in baseline_records.items() if not r["exact_match"]}
    print(f"[mining] baseline: {len(baseline_records)} problems, {len(wrong_pids)} wrong")

    if baseline_csv is not None:
        # Sanity check against overlap experiment if provided
        b = load_per_sample(baseline_csv)
        b_pid_set = set(b["problem_id"].astype(str))
        if b_pid_set != set(baseline_records):
            print(f"[mining] WARN: baseline pids differ from overlap csv "
                  f"(overlap={len(b_pid_set)}, current={len(baseline_records)})")

    weight_backup = backup_weights(model)
    math_only = _math_only_masks(math_important, calib_important)
    n_lines = 0
    with open(out_path, "w") as f:
        for k, cfg in enumerate(archive_meta):
            from experiments.overlap.intervention_configs import InterventionConfig
            iv_cfg = InterventionConfig(**cfg)
            sel = _selector_for_config(iv_cfg, math_only, math_important, importance_scores)
            t0 = time.time()
            apply_mask_to_model(model, sel, factor=iv_cfg.scale)
            try:
                iv_records = run_lm_eval_with_text(model, tokenizer, task, eval_subset, random_state, batch_size)
            finally:
                restore_weights(model, weight_backup)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            n_rescued_here = 0
            for pid, iv_rec in iv_records.items():
                if pid not in wrong_pids:
                    continue
                if not iv_rec["exact_match"]:
                    continue
                base_rec = baseline_records[pid]
                f.write(json.dumps({
                    "problem_id": pid,
                    "intervention_id": iv_cfg.intervention_id,
                    "prompt": base_rec["prompt"],
                    "gold": base_rec["gold"],
                    "baseline_solution": base_rec["filtered_response"] or base_rec["raw_response"],
                    "baseline_correct": False,
                    "intervention_solution": iv_rec["filtered_response"] or iv_rec["raw_response"],
                    "intervention_correct": True,
                    "answer_extracted": iv_rec["filtered_response"],
                }) + "\n")
                n_lines += 1
                n_rescued_here += 1
            print(f"[mining] [{k + 1}/{len(archive_meta)}] {iv_cfg.intervention_id}: "
                  f"+{n_rescued_here} rescued in {time.time() - t0:.1f}s")

    print(f"[mining] wrote {n_lines} rescued (problem, intervention) lines → {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="mathneuro yaml")
    p.add_argument("--overlap-root", required=True, help="dir with mask cache + intervention metas")
    p.add_argument("--archive", required=True, help="archive json file (from analyze_rescue)")
    p.add_argument("--out", required=True)
    p.add_argument("--task", default="gsm8k_cot")
    args = p.parse_args()

    import pickle
    from experiments.overlap.run_pool import _load_model_and_tokenizer
    from mathneuro.config import MathNeuroConfig

    cfg = MathNeuroConfig.from_yaml(args.config)
    overlap_root = Path(args.overlap_root)
    with open(overlap_root / "masks_cache.pkl", "rb") as fp:
        bundle = pickle.load(fp)
    archive = json.loads(Path(args.archive).read_text())

    from experiments.overlap.io import load_meta
    metas = {m.intervention_id: m for m in load_meta(overlap_root)}
    archive_meta = []
    for iid in archive["selected"]:
        m = metas[iid]
        archive_meta.append({
            "intervention_id": m.intervention_id,
            "family": m.family,
            "scale": m.scale,
            "selector_kind": m.extra.get("selector_kind"),
            "selector_args": m.extra.get("selector_args", {}),
            "parameter_group": m.parameter_group,
        })

    model, tokenizer = _load_model_and_tokenizer(cfg.model)
    mine_trajectories(
        Path(args.out),
        model, tokenizer,
        archive_meta=archive_meta,
        math_important=bundle["math_important"],
        calib_important=bundle["calib_important"],
        importance_scores=bundle["math_scores"],
        task=args.task,
        eval_subset=cfg.eval_dataset_subset,
        random_state=cfg.random_state,
        batch_size=cfg.batch_size,
        baseline_csv=overlap_root / "baseline.csv",
    )


if __name__ == "__main__":
    main()
