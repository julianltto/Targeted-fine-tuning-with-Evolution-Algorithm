"""
Refactored MathNeuro pruning pipeline.

Stages:
    1) Load the YAML config, training data, calibration datasets, model, and tokenizer.
    2) Optional pre-pruning evaluation (SGSM few-shot accuracy + lm_eval baselines).
    3) For each (calibration dataset, repeat, keep_ratio):
       - Reload a fresh copy of the model.
       - Compute math vs. calibration importance using mathneuro.core.
       - Build the math-specific multiplicative mask and apply it in place.
       - Run post-pruning evaluation and save results.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, cast

import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_eval import simple_evaluate
from lm_eval.tasks import TaskManager

from mathneuro_config import load_config
from mathneuro.core import (
    apply_mask_to_model,
    build_prune_mask,
    compute_importance,
    make_calibration_prompt_fn,
    make_math_prompt_fn,
    register_activation_hooks,
    remove_hooks,
    top_k_mask,
)


@dataclass(frozen=True)
class CalibrationDataset:
    name: str
    data: pd.DataFrame


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_train_dataset(args) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Load and shuffle the training dataset. For SGSM we also split off a validation set used by
    the SGSM-specific few-shot Python-program evaluation.

    Returns:
        (train_df, val_df) — val_df is None when the dataset is not SGSM.
    """
    if 'sgsm' in args.train_dataset:
        df = pd.read_csv(args.train_dataset)
        df = df[df['subset'] == 'sgsm_train']
        df = df.sample(frac=1, random_state=args.random_state)

        # Drop rows whose answer cannot be parsed as a float (SGSM cleanup).
        numeric = pd.to_numeric(df['answer'], errors='coerce')
        df = df[numeric.notna()].copy()
        df['answer'] = numeric[numeric.notna()].astype(float)

        train = df.iloc[0:1500]
        val = df.iloc[1500:].sample(frac=1, random_state=args.random_state)
        return train, val

    train = pd.read_csv(args.train_dataset).sample(frac=1, random_state=args.random_state)
    return train, None


def load_calibration_datasets(args) -> list[CalibrationDataset]:
    """
    Load every calibration dataset listed in the config and keep its display name for file naming
    and prompt-fn selection.
    """
    datasets: list[CalibrationDataset] = []
    for path, display_name in zip(args.calibration_datasets, args.calibration_dataset_names):
        df = pd.read_csv(path).sample(frac=1, random_state=args.random_state)
        datasets.append(CalibrationDataset(name=str(display_name), data=df))
    return datasets


# ---------------------------------------------------------------------------
# Model / tokenizer
# ---------------------------------------------------------------------------

def load_model(model_name: str) -> nn.Module:
    """
    Load the model in bf16 with HF auto device map and disable sampling so generation is greedy
    and deterministic.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map='auto', torch_dtype=torch.bfloat16,
    )
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    return model


# ---------------------------------------------------------------------------
# SGSM few-shot Python-program evaluation
# ---------------------------------------------------------------------------

# Markers that delimit a stray second turn / extra output inside a generated solution.
_SOLUTION_TRUNCATE_MARKERS: tuple[str, ...] = (
    'Instruct:', 'print', 'Student:', 'Output:', '#TODO',
)


def run_solution_code(solution_text: str) -> Any:
    """
    Execute a generated Python program that defines `solution()` and return its result.
    The text is run in a fresh namespace so it cannot accidentally see our globals.
    """
    exec_namespace: dict[str, Any] = {}
    exec(solution_text, exec_namespace)
    solution_fn = exec_namespace.get('solution')
    if not callable(solution_fn):
        raise ValueError('Generated code did not define a callable solution().')
    return cast(Callable[[], Any], solution_fn)()


def clean_solution_text(solution_text: str) -> str:
    """
    Trim everything after the first occurrence of any truncation marker so we only execute the
    first solution() body, then keep `return result` if present.
    """
    for marker in _SOLUTION_TRUNCATE_MARKERS:
        if marker in solution_text:
            solution_text = solution_text.split(marker)[0]
    if 'return result' in solution_text:
        parts = re.split(r'(return result)', solution_text)
        solution_text = parts[0] + parts[1]
    return solution_text


def build_few_shot_prompt(train: pd.DataFrame, final_question: str, k: int = 8) -> str:
    """Build a k-shot prompt: k (question, solution) demonstrations followed by the final question."""
    demos: list[str] = []
    for j in range(k):
        question = train['question'].iloc[j]
        answer = train['solution'].iloc[j]
        demo = f"Instruct: {question} Let's write a Python program.\nOutput:\n{answer}"
        if demo not in demos:
            demos.append(demo)
    demos.append(f"Instruct: {final_question} Let's write a Python program.\nOutput:")
    return "\n\n".join(demos)


def evaluate_sgsm_few_shot(
    model: nn.Module,
    tokenizer,
    train: pd.DataFrame,
    val: pd.DataFrame,
    eval_subset: int,
) -> float:
    """
    Run k-shot evaluation on a subset of the SGSM validation set: generate a Python program for
    each question, execute it, and compare to the gold numeric answer.

    Returns:
        Mean accuracy in [0, 1] over the evaluated subset (0 if the subset is empty).
    """
    correct: list[int] = []
    n = min(eval_subset, len(val))
    for i in range(n):
        final_question = str(val.iloc[i]['question'])
        final_answer = float(val.iloc[i]['answer'])
        formatted_prompt = build_few_shot_prompt(train, final_question)
        final_prompt = f"Instruct: {final_question} Let's write a Python program.\nOutput:"

        inputs = tokenizer.encode(formatted_prompt, return_tensors='pt').to(model.device)
        output = cast(Any, model).generate(inputs, max_new_tokens=150)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        solution_text = clean_solution_text(generated_text.split(final_prompt)[-1].strip())

        try:
            model_answer = float(run_solution_code(solution_text))
            correct.append(1 if model_answer == final_answer else 0)
        except Exception:
            correct.append(0)

    return sum(correct) / len(correct) if correct else 0.0


# ---------------------------------------------------------------------------
# lm_eval harness wrapper
# ---------------------------------------------------------------------------

def run_lm_eval(
    model: nn.Module,
    tokenizer,
    tasks,
    eval_subset: int,
    random_state: int,
    batch_size: int | str = 1,
) -> dict:
    """Run `lm_eval.simple_evaluate` on the given tasks and return its `results` dict."""
    task_manager = TaskManager()
    results = simple_evaluate(
        model='hf',
        model_args={'pretrained': model, 'dtype': 'bfloat16', 'tokenizer': tokenizer},
        tasks=tasks,
        task_manager=task_manager,
        log_samples=False,
        batch_size=batch_size,
        limit=eval_subset,
        random_seed=random_state,
    )
    if results is None:
        raise RuntimeError('lm_eval.simple_evaluate returned None.')
    return results['results']


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def append_text(output_file: str, line: str) -> None:
    """Append a single line to the human-readable text results file (creates parent dirs)."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'a') as f:
        f.write(line if line.endswith('\n') else line + '\n')


def save_json(path: str, payload: Any) -> None:
    """Write `payload` as JSON to `path`, creating parent directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(payload, f)


def make_results_root(args) -> str:
    """Per-model results directory; everything else is named relative to this."""
    root = f"{args.save_path}/eval_results/{args.model}/"
    os.makedirs(root, exist_ok=True)
    return root


# ---------------------------------------------------------------------------
# Pruning pipeline (uses mathneuro.core)
# ---------------------------------------------------------------------------

def _pick_calib_prompt_fn(df: pd.DataFrame, dataset_name: str) -> Callable[[pd.Series], str]:
    """
    Pick the prompt extractor for a calibration dataframe. Preserves the original convention:
      - If `dataset_name` contains "Bad", treat the dataset as plain text (column '0').
      - Otherwise treat it as math-like (use 'qa' if present, else 'question' + 'solution').
    """
    if 'Bad' in dataset_name:
        return make_calibration_prompt_fn()
    return make_math_prompt_fn(df)


def prune_math_specific(
    model: nn.Module,
    tokenizer,
    train_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    calib_name: str,
    keep_ratio: float,
    num_samples: int,
    factor: float,
) -> None:
    """
    End-to-end pruning for one (train, calibration, keep_ratio) combination. Registers hooks,
    runs forward passes on both datasets to score importance, builds the math-specific mask, and
    multiplies it into the model parameters in place. Hooks are removed before returning.

    Inputs:
        train_df    : math training samples used to identify math-important weights.
        calib_df    : calibration samples used to identify general-purpose important weights.
        calib_name  : display name used to preserve the "Bad" calibration-dataset convention.
        keep_ratio  : per-layer top-k ratio used by top_k_mask().
        num_samples : number of forward passes per dataset.
        factor      : multiplier applied to math-specific positions (0 = prune them out).
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

    math_mask = top_k_mask(math_scores, keep_ratio)
    calib_mask = top_k_mask(calib_scores, keep_ratio)
    pruning_mask = build_prune_mask(math_mask, calib_mask, factor=factor)
    apply_mask_to_model(model, pruning_mask)


# ---------------------------------------------------------------------------
# Evaluation orchestration
# ---------------------------------------------------------------------------

def run_pre_train_eval(
    args,
    model: nn.Module,
    tokenizer,
    train: pd.DataFrame,
    val: pd.DataFrame | None,
    results_root: str,
    output_file: str,
) -> None:
    """Optional baseline evaluation before any pruning happens."""
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
            model, tokenizer, args.eval_datasets, args.eval_dataset_subset, args.random_state,
        )
        save_json(f"{results_root}pre_results.json", results)

    if args.train_lm_eval_task is not None:
        train_results = run_lm_eval(
            model, tokenizer, args.train_lm_eval_task,
            args.eval_dataset_subset, args.random_state,
        )
        save_json(f"{results_root}pre_results_train_task.json", train_results)

        eval_results = run_lm_eval(
            model, tokenizer, args.eval_datasets,
            args.eval_dataset_subset, args.random_state,
        )
        save_json(f"{results_root}pre_results.json", eval_results)


def run_post_prune_eval(
    args,
    model: nn.Module,
    tokenizer,
    train: pd.DataFrame,
    val: pd.DataFrame | None,
    results_root: str,
    output_file: str,
    dataset_name: str,
    good_percent: float,
    repeat: int,
    num_samples: int,
) -> None:
    """Evaluate the pruned model and dump per-run JSONs / accuracy lines."""
    if 'sgsm' in args.train_dataset:
        assert val is not None
        acc = evaluate_sgsm_few_shot(model, tokenizer, train, val, args.eval_dataset_subset)
        n = min(args.eval_dataset_subset, len(val))
        append_text(
            output_file,
            f"Average eval accuracy on {n} questions for pruning top {good_percent}% good "
            f"parameters based on not being activated by {dataset_name} based on {num_samples} "
            f"training samples and greedy decoding (few-shot): {acc}",
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        results = run_lm_eval(
            model, tokenizer, args.eval_datasets, args.eval_dataset_subset, args.random_state,
        )
        save_json(
            f"{results_root}{dataset_name}_calculate{good_percent}_run{repeat}.json",
            results,
        )

    if args.train_lm_eval_task is not None:
        train_results = run_lm_eval(
            model, tokenizer, args.train_lm_eval_task,
            args.eval_dataset_subset, args.random_state,
        )
        save_json(
            f"{results_root}{dataset_name}_calculate{good_percent}_run{repeat}_train_task.json",
            train_results,
        )
        eval_results = run_lm_eval(
            model, tokenizer, args.eval_datasets,
            args.eval_dataset_subset, args.random_state,
        )
        save_json(
            f"{results_root}{dataset_name}_calculate{good_percent}_run{repeat}.json",
            eval_results,
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def default_keep_ratios() -> list[float]:
    """Top-k ratios swept when `proportion` is not specified in the config."""
    return [0.0001, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15]


def main() -> None:
    args = load_config()

    train, val = load_train_dataset(args)
    calibration_datasets = load_calibration_datasets(args)

    results_root = make_results_root(args)
    output_file = f"{args.save_path}/eval_results/{args.model}/{args.text_file}"

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.pre_train_eval:
        model = load_model(args.model)
        run_pre_train_eval(args, model, tokenizer, train, val, results_root, output_file)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    keep_ratios = [args.proportion] if args.proportion is not None else default_keep_ratios()
    num_samples = args.num_samples

    for calibration_dataset in calibration_datasets:
        dataset_name = calibration_dataset.name
        calib_df = calibration_dataset.data
        for repeat in range(args.num_repeats):
            sampled_train = train.sample(n=num_samples, replace=True)
            sampled_calib = calib_df.sample(n=num_samples, replace=True)

            for keep_ratio in keep_ratios:
                # Reload a fresh model for every combo so pruning never accumulates across runs.
                model = load_model(args.model)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                prune_math_specific(
                    model=model,
                    tokenizer=tokenizer,
                    train_df=sampled_train,
                    calib_df=sampled_calib,
                    calib_name=dataset_name,
                    keep_ratio=keep_ratio,
                    num_samples=num_samples,
                    factor=args.scalar,
                )

                run_post_prune_eval(
                    args, model, tokenizer, train, val, results_root, output_file,
                    dataset_name=dataset_name, good_percent=keep_ratio,
                    repeat=repeat, num_samples=num_samples,
                )

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
