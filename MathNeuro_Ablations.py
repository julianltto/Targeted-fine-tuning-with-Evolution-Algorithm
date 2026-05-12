"""
Refactored MathNeuro ablation pipeline.

This is the ablation counterpart to MathNeuro.py: same data loading and evaluation, but with two
control conditions for the pruning mask:

    `top_good` : Prune ALL math-important positions (no intersection with the calibration mask).
                 Tests whether the *math-specific* set matters, or whether pruning any math-active
                 weight is enough.
    `random`   : Compute the math-specific positions as in MathNeuro.py, count how many there are
                 per layer, then prune the same *number* of weights chosen uniformly at random.
                 Tests whether the specific positions matter, or whether any random subset of the
                 same size would hurt math just as much.

Both ablations reuse mathneuro.core for hooks / importance / top-k. Heavy IO / SGSM eval / lm_eval
helpers are imported from MathNeuro.py to avoid duplication; this file only adds the ablation-
specific mask construction and orchestration.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from mathneuro_config import load_config
from mathneuro.core import (
    apply_mask_to_model,
    compute_importance,
    make_calibration_prompt_fn,
    make_math_prompt_fn,
    register_activation_hooks,
    remove_hooks,
    top_k_mask,
)
from MathNeuro import (
    append_text,
    evaluate_sgsm_few_shot,
    load_calibration_datasets,
    load_model,
    load_train_dataset,
    make_results_root,
    run_lm_eval,
    save_json,
)


# ---------------------------------------------------------------------------
# Ablation-specific mask construction
# ---------------------------------------------------------------------------

def build_top_good_mask(
    math_important: dict[str, torch.Tensor],
    factor: float,
    exclude_substring: str = 'embed',
) -> dict[str, torch.Tensor]:
    """
    `top_good` ablation: prune ALL math-important positions (ignore the calibration mask).

    Returns a multiplicative mask where math-important positions are set to `factor` (0 prunes
    them) and every other position is 1.
    """
    masks: dict[str, torch.Tensor] = {}
    for name, math_mask in math_important.items():
        if exclude_substring in name:
            masks[name] = torch.ones_like(math_mask, dtype=torch.float32)
            continue

        mask = torch.ones_like(math_mask, dtype=torch.float32)
        mask[math_mask] = factor
        masks[name] = mask
    return masks


def build_random_mask(
    math_important: dict[str, torch.Tensor],
    calib_important: dict[str, torch.Tensor],
    factor: float,
    exclude_substring: str = 'embed',
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """
    `random` ablation: for each layer, count the number of math-specific positions
    (`math & ~calib`) and prune the same number of *randomly chosen* positions instead.

    Returns a multiplicative mask shaped like each weight, with that many entries set to `factor`
    (uniformly at random) and the rest set to 1.
    """
    masks: dict[str, torch.Tensor] = {}
    for name, math_mask in math_important.items():
        if exclude_substring in name:
            masks[name] = torch.ones_like(math_mask, dtype=torch.float32)
            continue

        calib_mask = calib_important[name]
        num_to_prune = int((math_mask & ~calib_mask).sum().item())

        flat_mask = torch.ones(math_mask.numel(), dtype=torch.float32)
        if num_to_prune > 0:
            random_indices = torch.randperm(flat_mask.numel(), generator=generator)[:num_to_prune]
            flat_mask[random_indices] = factor

        masks[name] = flat_mask.view(math_mask.shape)
    return masks


# ---------------------------------------------------------------------------
# Pruning pipelines (uses mathneuro.core)
# ---------------------------------------------------------------------------

def _pick_calib_prompt_fn(df: pd.DataFrame, dataset_name: str) -> Callable[[pd.Series], str]:
    """Same convention as MathNeuro.py: datasets named "Bad…" use column '0', others use 'qa'."""
    if 'Bad' in dataset_name:
        return make_calibration_prompt_fn()
    return make_math_prompt_fn(df)


def _compute_math_and_calib_importance(
    model: nn.Module,
    tokenizer,
    train_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    calib_name: str,
    num_samples: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    Run forward passes on math and calibration data with hooks attached, return both importance
    score dicts. Hooks are always removed before returning, even on error.
    """
    magnitude, handles = register_activation_hooks(model)
    try:
        math_scores = compute_importance(
            model, tokenizer, train_df,
            make_math_prompt_fn(train_df), magnitude, num_samples,
        )
        calib_scores = compute_importance(
            model, tokenizer, calib_df,
            _pick_calib_prompt_fn(calib_df, calib_name), magnitude, num_samples,
        )
    finally:
        remove_hooks(handles)
    return math_scores, calib_scores


def prune_ablation(
    model: nn.Module,
    tokenizer,
    train_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    calib_name: str,
    keep_ratio: float,
    num_samples: int,
    factor: float,
    method: str,
    generator: torch.Generator | None = None,
) -> None:
    """
    Run the ablation pruning end-to-end. The `method` argument selects which baseline mask to
    apply:
        - 'top_good': prune all math-important positions.
        - 'random'  : prune the same number of randomly chosen positions as math-specific.

    All other dataset / hooking logic mirrors MathNeuro.prune_math_specific.
    """
    math_scores, calib_scores = _compute_math_and_calib_importance(
        model, tokenizer, train_df, calib_df, calib_name, num_samples,
    )
    math_mask = top_k_mask(math_scores, keep_ratio)

    if method == 'top_good':
        mask = build_top_good_mask(math_mask, factor=factor)
    elif method == 'random':
        calib_mask = top_k_mask(calib_scores, keep_ratio)
        mask = build_random_mask(math_mask, calib_mask, factor=factor, generator=generator)
    else:
        raise ValueError(f"Unknown ablation method: {method!r}. Expected 'top_good' or 'random'.")

    apply_mask_to_model(model, mask)


# ---------------------------------------------------------------------------
# Evaluation orchestration
# ---------------------------------------------------------------------------

# lm_eval batch size used by the ablation script. Matches the original argparse-based code.
_LM_EVAL_BATCH_SIZE: int | str = 'auto:4'


def run_pre_train_eval_ablation(
    args,
    model: nn.Module,
    tokenizer,
    train: pd.DataFrame,
    val: pd.DataFrame | None,
    results_root: str,
    output_file: str,
) -> None:
    """Optional baseline evaluation before any ablation pruning runs."""
    if 'sgsm' in args.train_dataset:
        assert val is not None, "SGSM training set must produce a validation split."
        acc = evaluate_sgsm_few_shot(model, tokenizer, train, val, args.eval_dataset_subset)
        n = min(args.eval_dataset_subset, len(val))
        append_text(
            output_file,
            f"Average eval accuracy on {n} questions before training with greedy decoding "
            f"(few-shot): {acc}",
        )
        results = run_lm_eval(
            model, tokenizer, args.eval_datasets,
            args.eval_dataset_subset, args.random_state, batch_size=_LM_EVAL_BATCH_SIZE,
        )
        save_json(f"{results_root}pre_results.json", results)

    if args.train_lm_eval_task is not None:
        train_results = run_lm_eval(
            model, tokenizer, args.train_lm_eval_task,
            args.eval_dataset_subset, args.random_state, batch_size=_LM_EVAL_BATCH_SIZE,
        )
        save_json(f"{results_root}pre_results_train_task.json", train_results)

        eval_results = run_lm_eval(
            model, tokenizer, args.eval_datasets,
            args.eval_dataset_subset, args.random_state, batch_size=_LM_EVAL_BATCH_SIZE,
        )
        save_json(f"{results_root}pre_results.json", eval_results)


def run_post_prune_eval_ablation(
    args,
    model: nn.Module,
    tokenizer,
    train: pd.DataFrame,
    val: pd.DataFrame | None,
    results_root: str,
    output_file: str,
    method: str,
    dataset_name: str,
    good_percent: float,
    repeat: int,
    num_samples: int,
) -> None:
    """
    Evaluate the ablation-pruned model and dump per-run JSONs / accuracy lines. Output paths
    include a `/{method}/` subdirectory so the two ablations don't overwrite each other.
    """
    method_root = f"{results_root}{method}/"

    if 'sgsm' in args.train_dataset:
        assert val is not None
        acc = evaluate_sgsm_few_shot(model, tokenizer, train, val, args.eval_dataset_subset)
        n = min(args.eval_dataset_subset, len(val))
        append_text(
            output_file,
            f"Average eval accuracy on {n} questions for pruning top {good_percent}% good "
            f"parameters based on not being activated by {dataset_name} based on {num_samples} "
            f"training samples and greedy decoding (few-shot) [method={method}]: {acc}",
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        results = run_lm_eval(
            model, tokenizer, args.eval_datasets,
            args.eval_dataset_subset, args.random_state, batch_size=_LM_EVAL_BATCH_SIZE,
        )
        save_json(
            f"{method_root}{dataset_name}_calculate{good_percent}_run{repeat}.json",
            results,
        )

    if args.train_lm_eval_task is not None:
        train_results = run_lm_eval(
            model, tokenizer, args.train_lm_eval_task,
            args.eval_dataset_subset, args.random_state, batch_size=_LM_EVAL_BATCH_SIZE,
        )
        save_json(
            f"{method_root}{dataset_name}_calculate{good_percent}_run{repeat}_train_task.json",
            train_results,
        )
        eval_results = run_lm_eval(
            model, tokenizer, args.eval_datasets,
            args.eval_dataset_subset, args.random_state, batch_size=_LM_EVAL_BATCH_SIZE,
        )
        save_json(
            f"{method_root}{dataset_name}_calculate{good_percent}_run{repeat}.json",
            eval_results,
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def default_keep_ratios() -> list[float]:
    """Top-k ratios swept when `proportion` is not specified in the config."""
    return [0.0001, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15]


def default_ablation_methods() -> list[str]:
    """Ablation methods compared in the original script."""
    return ['random', 'top_good']


def main() -> None:
    args = load_config()

    train, val = load_train_dataset(args)
    calibration_datasets = load_calibration_datasets(args)

    results_root = make_results_root(args)
    output_file = f"{args.save_path}/eval_results/{args.model}/{args.text_file}"

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.pre_train_eval:
        model = load_model(args.model)
        run_pre_train_eval_ablation(
            args, model, tokenizer, train, val, results_root, output_file,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    keep_ratios = [args.proportion] if args.proportion is not None else default_keep_ratios()
    methods = default_ablation_methods()
    num_samples = args.num_samples

    # Seed a generator for the `random` ablation so reruns are reproducible per config seed.
    generator = torch.Generator()
    generator.manual_seed(args.random_state)

    for calibration_dataset in calibration_datasets:
        dataset_name = calibration_dataset.name
        calib_df = calibration_dataset.data
        for repeat in range(args.num_repeats):
            sampled_train = train.sample(n=num_samples, replace=True)
            sampled_calib = calib_df.sample(n=num_samples, replace=True)

            for keep_ratio in keep_ratios:
                for method in methods:
                    # Reload a fresh model so ablations never accumulate across runs.
                    model = load_model(args.model)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    prune_ablation(
                        model=model,
                        tokenizer=tokenizer,
                        train_df=sampled_train,
                        calib_df=sampled_calib,
                        calib_name=dataset_name,
                        keep_ratio=keep_ratio,
                        num_samples=num_samples,
                        factor=args.scalar,
                        method=method,
                        generator=generator,
                    )

                    run_post_prune_eval_ablation(
                        args, model, tokenizer, train, val, results_root, output_file,
                        method=method, dataset_name=dataset_name,
                        good_percent=keep_ratio, repeat=repeat, num_samples=num_samples,
                    )

                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
