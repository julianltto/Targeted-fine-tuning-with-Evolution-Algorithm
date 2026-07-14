"""Held-out evaluation of distilled checkpoints (spec §9).

Evaluates a base model with optional LoRA adapter against:
  - math tasks (gsm8k_cot, MATH-style, etc.)
  - retention tasks (mmlu_high_school_world_history, race, ...)

For each task we record overall accuracy and the per-sample correctness so we
can compute § post-distillation recovery rate (spec §9.4).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.overlap.runner import _extract_per_sample_correctness


def _load_model(model_name: str, adapter_dir: Path | None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda")
    if adapter_dir is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        model = model.merge_and_unload()
    model.generation_config.do_sample = False
    return model, tokenizer


def evaluate(model, tokenizer, tasks: list[str], limit: int, random_state: int,
             batch_size: int | str = 1) -> dict:
    from lm_eval import simple_evaluate
    from lm_eval.tasks import TaskManager
    results = simple_evaluate(
        model="hf",
        model_args={"pretrained": model, "dtype": "bfloat16", "tokenizer": tokenizer, "max_length": 2048},
        tasks=tasks,
        task_manager=TaskManager(),
        log_samples=True,
        batch_size=batch_size,
        limit=limit,
        random_seed=random_state,
    )
    if results is None:
        raise RuntimeError("simple_evaluate returned None")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--math-tasks", nargs="+", default=["gsm8k_cot"])
    p.add_argument("--retention-tasks", nargs="+", default=["mmlu_high_school_world_history"])
    p.add_argument("--limit", type=int, default=1024)
    p.add_argument("--retention-limit", type=int, default=256)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--batch-size", default="1")
    p.add_argument("--baseline-csv", default=None,
                   help="optional baseline.csv from overlap experiment — enables post-distillation recovery rate")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = _load_model(args.model, Path(args.adapter) if args.adapter else None)

    summary: dict = {"model": args.model, "adapter": args.adapter, "tasks": {}}

    bs = int(args.batch_size) if args.batch_size.isdigit() else args.batch_size

    # Math tasks
    math_results = evaluate(model, tokenizer, args.math_tasks, args.limit, args.random_state, bs)
    for task in args.math_tasks:
        summary["tasks"][task] = math_results["results"][task]
        ids, correct = _extract_per_sample_correctness(math_results, task)
        import pandas as pd
        pd.DataFrame({"problem_id": ids, "correct": correct.astype(int)}) \
            .to_csv(out_dir / f"{task}_per_sample.csv", index=False)

    # Retention tasks
    if args.retention_tasks:
        ret_results = evaluate(model, tokenizer, args.retention_tasks, args.retention_limit, args.random_state, bs)
        for task in args.retention_tasks:
            summary["tasks"][task] = ret_results["results"][task]

    # Post-distillation recovery rate (spec §9.4)
    if args.baseline_csv:
        import pandas as pd
        base = pd.read_csv(args.baseline_csv)
        base["problem_id"] = base["problem_id"].astype(str)
        base["correct"] = base["correct"].astype(bool)
        wrong_pids = set(base.loc[~base["correct"], "problem_id"])
        for task in args.math_tasks:
            iv = pd.read_csv(out_dir / f"{task}_per_sample.csv")
            iv["problem_id"] = iv["problem_id"].astype(str)
            iv["correct"] = iv["correct"].astype(bool)
            recovered = int(iv.loc[iv["problem_id"].isin(wrong_pids), "correct"].sum())
            new_hurt = int((
                iv["problem_id"].isin(set(base.loc[base["correct"], "problem_id"]))
                & ~iv["correct"]
            ).sum())
            summary["tasks"][task]["post_distill_recovered_from_baseline_wrong"] = recovered
            summary["tasks"][task]["post_distill_new_hurt"] = new_hurt

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
