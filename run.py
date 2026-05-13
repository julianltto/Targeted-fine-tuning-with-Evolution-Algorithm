from __future__ import annotations

import gc
import json
import os
import pickle
import re
from dataclasses import dataclass
from typing import Any, Callable, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_eval import simple_evaluate
from lm_eval.tasks import TaskManager

from mathneuro.config import load_config
from mathneuro import (
    apply_mask_to_model,
    backup_weights,
    build_intervention_mask_per_layer,
    build_prune_mask,
    build_prune_mask_per_layer,
    compute_importance,
    make_calibration_prompt_fn,
    make_math_prompt_fn,
    register_activation_hooks,
    remove_hooks,
    restore_weights,
    top_k_mask,
    run_ea_search,
    format_pareto_front,
)


@dataclass(frozen=True)
class CalibrationDataset:
    name: str
    data: pd.DataFrame


def load_train_dataset(args) -> tuple[pd.DataFrame, pd.DataFrame | None]:
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
    datasets: list[CalibrationDataset] = []
    for path, display_name in zip(args.calibration_datasets, args.calibration_dataset_names):
        df = pd.read_csv(path).sample(frac=1, random_state=args.random_state)
        datasets.append(CalibrationDataset(name=str(display_name), data=df))
    return datasets


def load_model(model_name: str) -> nn.Module:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Install a CUDA-enabled PyTorch build, "
            "or revert this function to device_map='auto' to allow CPU fallback."
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
    ).to('cuda')
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    return model


# Markers that delimit a stray second turn / extra output inside a generated solution.
_SOLUTION_TRUNCATE_MARKERS: tuple[str, ...] = (
    'Instruct:', 'print', 'Student:', 'Output:', '#TODO',
)


def run_solution_code(solution_text: str) -> Any:
    exec_namespace: dict[str, Any] = {}
    exec(solution_text, exec_namespace)
    solution_fn = exec_namespace.get('solution')
    if not callable(solution_fn):
        raise ValueError('Generated code did not define a callable solution().')
    return cast(Callable[[], Any], solution_fn)()


def clean_solution_text(solution_text: str) -> str:
    for marker in _SOLUTION_TRUNCATE_MARKERS:
        if marker in solution_text:
            solution_text = solution_text.split(marker)[0]
    if 'return result' in solution_text:
        parts = re.split(r'(return result)', solution_text)
        solution_text = parts[0] + parts[1]
    return solution_text


def build_few_shot_prompt(train: pd.DataFrame, final_question: str, k: int = 8) -> str:
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


def run_lm_eval(
    model: nn.Module,
    tokenizer,
    tasks,
    eval_subset: int,
    random_state: int,
    batch_size: int | str = 1,
    sample_dump_path: str | None = None,
    n_samples_to_dump: int = 5,
) -> dict:
    task_manager = TaskManager()
    results = simple_evaluate(
        model='hf',
        model_args={'pretrained': model, 'dtype': 'bfloat16', 'tokenizer': tokenizer},
        tasks=tasks,
        task_manager=task_manager,
        log_samples=sample_dump_path is not None,
        batch_size=batch_size,
        limit=eval_subset,
        random_seed=random_state,
    )
    if results is None:
        raise RuntimeError('lm_eval.simple_evaluate returned None.')

    if sample_dump_path and 'samples' in results:
        dump: dict[str, list[dict[str, Any]]] = {}
        for task_name, samples in results['samples'].items():
            picked = []
            for s in samples[:n_samples_to_dump]:
                prompt = ''
                if s.get('arguments'):
                    arg0 = s['arguments'][0]
                    prompt = arg0[0] if isinstance(arg0, (list, tuple)) else str(arg0)
                picked.append({
                    'prompt_tail': prompt[-600:],
                    'gold': s.get('target', ''),
                    'generated': s.get('resps', [[None]])[0][0] if s.get('resps') else None,
                    'filtered': s.get('filtered_resps', [None])[0] if s.get('filtered_resps') else None,
                    'exact_match': s.get('exact_match'),
                })
            dump[task_name] = picked
        save_json(sample_dump_path, dump)

    return results['results']


def ea_checkpoint_path(
    results_root: str,
    dataset_name: str,
    keep_ratio: float,
    repeat: int,
) -> str:
    return f"{results_root}{dataset_name}_calculate{keep_ratio}_run{repeat}_ea_ckpt.pkl"


def save_ea_checkpoint(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(payload, f)


def try_load_ea_checkpoint(path: str, expected: dict[str, Any]) -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    cfg = payload.get('config', {})
    for k, v in expected.items():
        if cfg.get(k) != v:
            print(f"[EA ckpt] {path} skipped: {k}={cfg.get(k)} != {v}")
            return None
    return payload


def append_text(output_file: str, line: str) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'a') as f:
        f.write(line if line.endswith('\n') else line + '\n')


def save_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(payload, f)


def make_results_root(args) -> str:
    root = f"{args.save_path}/eval_results/{args.model}/"
    os.makedirs(root, exist_ok=True)
    return root


def _pick_calib_prompt_fn(df: pd.DataFrame, dataset_name: str) -> Callable[[pd.Series], str]:
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
    magnitude, handles = register_activation_hooks(model)
    try:
        math_scores = compute_importance(
            model, tokenizer, train_df,
            make_math_prompt_fn(train_df), magnitude, num_samples,
        )
        math_mask = top_k_mask(math_scores, keep_ratio)
        del math_scores

        calib_scores = compute_importance(
            model, tokenizer, calib_df,
            _pick_calib_prompt_fn(calib_df, calib_name), magnitude, num_samples,
        )
        calib_mask = top_k_mask(calib_scores, keep_ratio)
        del calib_scores
    finally:
        remove_hooks(handles)

    pruning_mask = build_prune_mask(math_mask, calib_mask)
    del math_mask, calib_mask
    apply_mask_to_model(model, pruning_mask, factor=factor)


def compute_prune_stats(
    math_mask: dict[str, torch.Tensor],
    calib_mask: dict[str, torch.Tensor],
    strengths: dict[str, float],
    exclude_substring: str = 'embed',
) -> dict[str, Any]:
    total_params = 0
    weighted_pruned = 0.0
    total_math_only = 0
    per_layer: dict[str, dict[str, float]] = {}

    for name, m_mask in math_mask.items():
        layer_size = int(m_mask.numel())
        total_params += layer_size

        if exclude_substring in name:
            continue

        math_only = m_mask & (~calib_mask[name])
        math_only_count = int(math_only.sum().item())
        total_math_only += math_only_count

        strength = float(strengths.get(name, 0.0))
        weighted_pruned += math_only_count * strength

        per_layer[name] = {
            "strength": strength,
            "math_only_count": math_only_count,
            "layer_size": layer_size,
        }

    return {
        "effective_intervention_strength": weighted_pruned / max(total_params, 1),
        "math_only_ratio": total_math_only / max(total_params, 1),
        "total_params": total_params,
        "per_layer": per_layer,
    }


def search_pareto_front(
    model: nn.Module,
    tokenizer,
    train_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    calib_name: str,
    keep_ratio: float,
    num_samples: int,
    seed: int = 42,
    pop_size: int = 20,
    n_gen: int = 15,
    eval_samples: int = 30,
    eval_batch_size: int = 16,
    mode: str = 'prune',
    max_scale: float = 0.1,
) -> tuple[
    list[dict[str, float]],
    np.ndarray,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    magnitude, handles = register_activation_hooks(model)
    try:
        math_scores = compute_importance(
            model, tokenizer, train_df,
            make_math_prompt_fn(train_df), magnitude, num_samples,
        )
        math_mask = top_k_mask(math_scores, keep_ratio)
        del math_scores
        calib_scores = compute_importance(
            model, tokenizer, calib_df,
            _pick_calib_prompt_fn(calib_df, calib_name), magnitude, num_samples,
        )
        calib_mask = top_k_mask(calib_scores, keep_ratio)
        del calib_scores
    finally:
        remove_hooks(handles)
    gc.collect()

    result, layer_names = run_ea_search(
        model, math_mask, calib_mask,
        make_eval_fn(tokenizer, train_df, calib_df, n=eval_samples, batch_size=eval_batch_size),
        pop_size=pop_size, n_gen=n_gen, seed=seed,
        mode=mode, max_scale=max_scale,
    )
    if result.F is None or result.X is None:
        raise RuntimeError("EA search returned no Pareto front.")

    F = -result.F                     
    X = result.X                       
    strengths_list = [
        {layer_names[j]: float(X[i, j]) for j in range(len(layer_names))}
        for i in range(X.shape[0])
    ]
    return strengths_list, F, math_mask, calib_mask

_QA_SPLIT_MARKERS: tuple[str, ...] = (
    "A: Let's think step by step.\n",
    "\n\nAnswer: ",   
    "\n\nSolution: ",                
)


def split_qa(qa: str) -> tuple[str, str]:
    for marker in _QA_SPLIT_MARKERS:
        idx = qa.rfind(marker)
        if idx != -1:
            end = idx + len(marker)
            return qa[:end], qa[end:]
    raise ValueError(f"No QA marker matched: {qa[:120]!r}...")


@torch.no_grad()
def loglike(model, tokenizer, df, n=50, batch_size=16):
    n = min(n, len(df))
    if n == 0:
        return 0.0

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    examples = []
    for i in range(n):
        row = df.iloc[i]
        if 'qa' in df.columns:
            prompt, target = split_qa(str(row['qa']))
        else:
            prompt = f"Question: {row['question']}\nAnswer: "
            target = str(row['solution'])
        full_ids = tokenizer.encode(prompt + target)
        prompt_ids = tokenizer.encode(prompt)
        examples.append((full_ids, len(prompt_ids)))

    order = sorted(range(n), key=lambda i: len(examples[i][0]))

    total_loglike, total_tokens = 0.0, 0
    for start in range(0, n, batch_size):
        idxs = order[start:start + batch_size]
        chunk = [examples[i] for i in idxs]
        max_len = max(len(ids) for ids, _ in chunk)
        B = len(chunk)

        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
        attn_mask = torch.zeros((B, max_len), dtype=torch.long)
        labels = torch.full((B, max_len), -100, dtype=torch.long)
        for b, (ids, plen) in enumerate(chunk):
            L = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            input_ids[b, :L] = ids_tensor
            attn_mask[b, :L] = 1
            if L > plen:
                labels[b, plen:L] = ids_tensor[plen:]

        input_ids = input_ids.to(model.device)
        attn_mask = attn_mask.to(model.device)
        labels = labels.to(model.device)

        logits = model(input_ids, attention_mask=attn_mask).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss_sum = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='sum',
        )
        total_loglike += -loss_sum.item()
        total_tokens += shift_labels.ne(-100).sum().item()
    return total_loglike / max(total_tokens, 1)


_LM_EVAL_TASK_MANAGER: Any = None


def _get_task_manager():
    global _LM_EVAL_TASK_MANAGER
    if _LM_EVAL_TASK_MANAGER is None:
        _LM_EVAL_TASK_MANAGER = TaskManager()
    return _LM_EVAL_TASK_MANAGER


@torch.no_grad()
def gsm8k_cot_acc(
    model: nn.Module,
    tokenizer,
    limit: int = 32,
    batch_size: int = 8,
    seed: int = 42,
) -> float:
    results = simple_evaluate(
        model='hf',
        model_args={'pretrained': model, 'dtype': 'bfloat16', 'tokenizer': tokenizer},
        tasks='gsm8k_cot',
        task_manager=_get_task_manager(),
        log_samples=False,
        batch_size=batch_size,
        limit=limit,
        random_seed=seed,
        verbosity='ERROR',
    )
    if results is None:
        return 0.0
    return float(results['results']['gsm8k_cot'].get('exact_match,strict-match', 0.0))


def dump_sample_generations(
    model: nn.Module,
    tokenizer,
    n: int = 5,
    task: str = 'gsm8k_cot',
    out_path: str | None = None,
    batch_size: int = 4,
    seed: int = 42,
) -> list[dict]:
    results = simple_evaluate(
        model='hf',
        model_args={'pretrained': model, 'dtype': 'bfloat16', 'tokenizer': tokenizer},
        tasks=task,
        task_manager=_get_task_manager(),
        log_samples=True,
        batch_size=batch_size,
        limit=n,
        random_seed=seed,
        verbosity='ERROR',
    )
    if results is None or 'samples' not in results:
        print("[dump] no samples returned")
        return []

    samples = results['samples'].get(task, [])
    for i, s in enumerate(samples):
        prompt = ''
        if s.get('arguments'):
            arg0 = s['arguments'][0]
            prompt = arg0[0] if isinstance(arg0, (list, tuple)) else str(arg0)
        gen = s.get('resps', [[None]])[0][0]
        gold = s.get('target', '?')
        print(f"\n=== {task} sample {i} ===")
        print(f"--- PROMPT (last 400 chars) ---\n{prompt[-400:]}")
        print(f"--- GOLD ---\n{gold}")
        print(f"--- GENERATED ---\n{gen}")

    if out_path:
        save_json(out_path, samples)
        print(f"\n[dump] {len(samples)} samples saved to {out_path}")
    return samples


def make_eval_fn(tokenizer, math_eval_df, general_eval_df, n: int = 50, batch_size: int = 16):
    def eval_fn(model):
        math_score = gsm8k_cot_acc(
            model, tokenizer, limit=n, batch_size=min(batch_size, 8),
        )
        general_score = loglike(
            model, tokenizer, general_eval_df,
            n=n, batch_size=batch_size,
        )
        return math_score, general_score
    return eval_fn

def run_pre_train_eval(
    args,
    model: nn.Module,
    tokenizer,
    train: pd.DataFrame,
    val: pd.DataFrame | None,
    results_root: str,
    output_file: str,
) -> None:
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
            batch_size=args.batch_size,
        )
        save_json(f"{results_root}pre_results.json", results)

    if args.train_lm_eval_task is not None:
        train_results = run_lm_eval(
            model, tokenizer, args.train_lm_eval_task,
            args.eval_dataset_subset, args.random_state,
            batch_size=args.batch_size,
            sample_dump_path=f"{results_root}pre_results_train_task_samples.json",
        )
        save_json(f"{results_root}pre_results_train_task.json", train_results)

        eval_results = run_lm_eval(
            model, tokenizer, args.eval_datasets,
            args.eval_dataset_subset, args.random_state,
            batch_size=args.batch_size,
            sample_dump_path=f"{results_root}pre_results_eval_samples.json",
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
            batch_size=args.batch_size,
        )
        save_json(
            f"{results_root}{dataset_name}_calculate{good_percent}_run{repeat}.json",
            results,
        )

    if args.train_lm_eval_task is not None:
        tag = f"{dataset_name}_calculate{good_percent}_run{repeat}"
        train_results = run_lm_eval(
            model, tokenizer, args.train_lm_eval_task,
            args.eval_dataset_subset, args.random_state,
            batch_size=args.batch_size,
            sample_dump_path=f"{results_root}{tag}_train_task_samples.json",
        )
        save_json(f"{results_root}{tag}_train_task.json", train_results)

        eval_results = run_lm_eval(
            model, tokenizer, args.eval_datasets,
            args.eval_dataset_subset, args.random_state,
            batch_size=args.batch_size,
            sample_dump_path=f"{results_root}{tag}_eval_samples.json",
        )
        save_json(f"{results_root}{tag}.json", eval_results)


def main():
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

    if args.proportion is None:
        keep_ratios = [0.0001, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15]
    elif isinstance(args.proportion, (list, tuple)):
        keep_ratios = list(args.proportion)
    else:
        keep_ratios = [args.proportion]
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

                if args.with_ea:
                    ckpt_path = ea_checkpoint_path(results_root, dataset_name, keep_ratio, repeat)
                    ckpt_config = {
                        'model': args.model,
                        'keep_ratio': keep_ratio,
                        'num_samples': num_samples,
                        'seed': args.random_state,
                        'pop_size': args.ea_pop_size,
                        'n_gen': args.ea_n_gen,
                        'eval_samples': args.ea_eval_samples,
                        'mode': args.ea_mode,
                        'max_scale': args.ea_max_scale,
                        'fitness_version': args.ea_fitness_version,
                    }
                    cached = try_load_ea_checkpoint(ckpt_path, ckpt_config)
                    if cached is not None:
                        print(f"[EA ckpt] loaded {ckpt_path}")
                        strengths_list = cached['strengths_list']
                        scores = cached['scores']
                        math_mask = cached['math_mask']
                        calib_mask = cached['calib_mask']
                    else:
                        strengths_list, scores, math_mask, calib_mask = search_pareto_front(
                            model=model,
                            tokenizer=tokenizer,
                            train_df=sampled_train,
                            calib_df=sampled_calib,
                            calib_name=dataset_name,
                            keep_ratio=keep_ratio,
                            num_samples=num_samples,
                            seed=args.random_state,
                            pop_size=args.ea_pop_size,
                            n_gen=args.ea_n_gen,
                            eval_samples=args.ea_eval_samples,
                            eval_batch_size=int(args.batch_size) if isinstance(args.batch_size, int) or (isinstance(args.batch_size, str) and args.batch_size.isdigit()) else 16,
                            mode=args.ea_mode,
                            max_scale=args.ea_max_scale,
                        )
                        save_ea_checkpoint(ckpt_path, {
                            'config': ckpt_config,
                            'strengths_list': strengths_list,
                            'scores': scores,
                            'math_mask': math_mask,
                            'calib_mask': calib_mask,
                        })
                        print(f"[EA ckpt] saved {ckpt_path}")
                    weight_snapshot = backup_weights(model)
                    for point_idx, strengths in enumerate(strengths_list):
                        math_score, general_score = scores[point_idx]
                        stats = compute_prune_stats(math_mask, calib_mask, strengths)

                        append_text(
                            output_file,
                            f"[EA point {point_idx}] mode={args.ea_mode} "
                            f"keep_ratio={keep_ratio} calib={dataset_name} "
                            f"math_acc={math_score:.4f} general_loglike={general_score:.4f} "
                            f"intervention_strength={stats['effective_intervention_strength']:.4f}",
                        )

                        save_json(
                            f"{results_root}{dataset_name}_ea{point_idx}_calculate{keep_ratio}_run{repeat}_strengths.json",
                            {
                                "point_idx": point_idx,
                                "mode": args.ea_mode,
                                "max_scale": args.ea_max_scale,
                                "keep_ratio": keep_ratio,
                                "calib_dataset": dataset_name,
                                "math_acc": float(math_score),
                                "general_loglike": float(general_score),
                                "effective_intervention_strength": stats["effective_intervention_strength"],
                                "math_only_ratio": stats["math_only_ratio"],
                                "total_params": stats["total_params"],
                                "strengths": strengths,
                                "per_layer": stats["per_layer"],
                            },
                        )

                        intervention_mask = build_intervention_mask_per_layer(
                            math_mask, calib_mask, strengths,
                            mode=args.ea_mode,
                            max_scale=args.ea_max_scale,
                        )
                        apply_mask_to_model(model, intervention_mask)

                        run_post_prune_eval(
                            args, model, tokenizer, train, val, results_root, output_file,
                            dataset_name=f"{dataset_name}_ea{point_idx}",
                            good_percent=keep_ratio,
                            repeat=repeat, num_samples=num_samples,
                        )

                        restore_weights(model, weight_snapshot)
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                else:
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
