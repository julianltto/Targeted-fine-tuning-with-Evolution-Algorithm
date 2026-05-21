from __future__ import annotations
import os
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

import gc
import json

import pickle
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_eval import simple_evaluate
from lm_eval.tasks import TaskManager
from lm_eval.models.huggingface import HFLM   # ← add this

from mathneuro.config import load_config
from mathneuro import *


@dataclass(frozen=True)
class CalibrationDataset:
    name: str
    data: pd.DataFrame


def _wrap_lm(model: nn.Module, tokenizer, batch_size: int | str = 1,
             max_length: int | None = None) -> HFLM:
    """Wrap an already-loaded HF model + tokenizer as an lm-eval LM instance.

    Passing this object to simple_evaluate(model=...) avoids the SHA-lookup
    warning that fires when `pretrained` is given a model object instead of
    a string repo id.
    """
    kwargs = dict(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    if max_length is not None:
        kwargs['max_length'] = max_length
    return HFLM(**kwargs)

# Data loading

def load_train_dataset(args) -> pd.DataFrame:
    return pd.read_csv(args.train_dataset).sample(frac=1, random_state=args.random_state)


def load_calibration_datasets(args) -> list[CalibrationDataset]:
    datasets: list[CalibrationDataset] = []
    for path, display_name in zip(args.calibration_datasets, args.calibration_dataset_names):
        df = pd.read_csv(path).sample(frac=1, random_state=args.random_state)
        datasets.append(CalibrationDataset(name=str(display_name), data=df))
    return datasets


def load_model(model_name: str) -> nn.Module:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
    ).to('cuda')
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.generation_config.cache_implementation = None
    model.generation_config.max_length = 4096
    return model


# lm-eval evaluation

def run_lm_eval(
    model: nn.Module,
    tokenizer,
    tasks: str | list[str],
    eval_subset: int,
    random_state: int,
    batch_size: int | str = 1,
    sample_dump_path: str | None = None,
    n_samples_to_dump: int = 5,
) -> dict:
    """
    Run lm-eval on one or more tasks and return the per-task metric dict.

    When ``sample_dump_path`` is set, also writes the first
    ``n_samples_to_dump`` prompt/gold/generation triples per task to that
    path for qualitative inspection.
    """
    task_manager = TaskManager()
    task_list: list[Any] = [tasks] if isinstance(tasks, str) else list(tasks)
    results = simple_evaluate(
        model=_wrap_lm(model, tokenizer, batch_size=batch_size, max_length=2048),
        tasks=task_list,
        task_manager=task_manager,
        log_samples=sample_dump_path is not None,
        batch_size=batch_size,
        limit=eval_subset,
        random_seed=random_state,
    )
    if results is None:
        raise RuntimeError('lm_eval.simple_evaluate returned None.')

    # Dump a few results
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


# EA checkpointing

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


# IO helpers

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


# Pruning

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
    """
    Summarize how aggressively an EA solution intervenes on the model.

    "math-only" params are those the math mask keeps but the calibration mask
    drops (i.e. specific to the math task). ``effective_intervention_strength``
    is the per-layer strength weighted by each layer's math-only count and
    normalized by total params, giving a single scalar for the whole model.
    Layers whose name contains ``exclude_substring`` (e.g. embeddings) still
    count toward ``total_params`` but are never intervened on.
    """
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


# wandb monitoring

# Strictness controls (env-driven):
#   WANDB_DISABLED=1         -> skip wandb entirely, log_safe is a no-op
#   WANDB_REQUIRE_ONLINE=1   -> raise if online init fails (do NOT silently
#                               fall back to offline) -- recommended for long runs
#   WANDB_MODE=offline       -> standard wandb env knob, respected by wandb.init
def init_wandb(
    project: str,
    run_name: str,
    config: dict[str, Any],
    *,
    group: str | None = None,
    job_type: str | None = None,
    tags: list[str] | None = None,
    resume_id: str | None = None,
):
    """Initialize a wandb run, loudly.

    - If WANDB_DISABLED=1, returns None silently (true opt-out).
    - Otherwise attempts online init; on failure either raises (if
      WANDB_REQUIRE_ONLINE=1) or falls back to offline with a loud warning
      so you know you'll need `wandb sync` later.
    - On success, prints the run URL so it's visible in nohup logs.
    """
    if os.environ.get("WANDB_DISABLED") == "1":
        print("[wandb] WANDB_DISABLED=1 set; skipping wandb init")
        return None

    try:
        import wandb
    except Exception as e:
        msg = f"[wandb] import failed: {e}"
        if os.environ.get("WANDB_REQUIRE_ONLINE") == "1":
            raise RuntimeError(msg) from e
        print(msg + "; continuing without tracking")
        return None

    requested_mode = os.environ.get("WANDB_MODE", "online")
    init_kwargs = dict(
        project=project,
        name=run_name,
        config=config,
        reinit=True,
        group=group,
        job_type=job_type,
        tags=tags,
    )
    if resume_id is not None:
        init_kwargs["id"] = resume_id
        init_kwargs["resume"] = "allow"

    # Try the requested mode first.
    try:
        run = wandb.init(mode=requested_mode, **init_kwargs)
    except Exception as e:
        print(f"[wandb] init mode={requested_mode!r} failed: {e}", flush=True)
        if requested_mode == "online" and os.environ.get("WANDB_REQUIRE_ONLINE") != "1":
            print("[wandb] falling back to mode='offline' -- "
                  "run `wandb sync wandb/offline-run-*` after the job to upload.",
                  flush=True)
            try:
                run = wandb.init(mode="offline", **init_kwargs)
            except Exception as e2:
                print(f"[wandb] offline fallback also failed: {e2}", flush=True)
                return None
        else:
            if os.environ.get("WANDB_REQUIRE_ONLINE") == "1":
                raise
            return None

    # Print the run URL so it shows up in nohup logs.
    try:
        url = getattr(run, "url", None) or getattr(run, "get_url", lambda: None)()
        rid = getattr(run, "id", "?")
        mode_actual = getattr(run.settings, "_mode", "?") if hasattr(run, "settings") else "?"
        print(f"[wandb] init ok id={rid} mode={mode_actual} url={url}", flush=True)
    except Exception:
        pass
    return run


def finish_wandb(run) -> None:
    if run is None:
        return
    try:
        import wandb
        wandb.finish()
    except Exception as e:
        print(f"[wandb] finish failed: {e}", flush=True)


def wandb_log_safe(metrics: dict[str, Any], step: int | None = None) -> None:
    """Log to the currently-active wandb run if one exists; otherwise no-op.

    Use this everywhere you want to log a metric. It will not crash if wandb
    is disabled, not installed, or no run is active.
    """
    try:
        import wandb
    except Exception:
        return
    if wandb.run is None:
        return
    try:
        if step is not None:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)
    except Exception as e:
        print(f"[wandb] log failed: {e}", flush=True)


def _wandb_summary_update(summary: dict[str, Any]) -> None:
    """Write key=value pairs to wandb.run.summary if a run is active."""
    try:
        import wandb
    except Exception:
        return
    if wandb.run is None:
        return
    try:
        for k, v in summary.items():
            wandb.run.summary[k] = v
    except Exception as e:
        print(f"[wandb] summary update failed: {e}", flush=True)


def _flatten_lm_eval_results(prefix: str, results: dict[str, Any]) -> dict[str, float]:
    """Turn lm-eval's nested per-task dict into a flat {prefix/task/metric: value} dict."""
    flat: dict[str, float] = {}
    if not isinstance(results, dict):
        return flat
    for task, task_result in results.items():
        if not isinstance(task_result, dict):
            continue
        for k, v in task_result.items():
            if k == "alias" or "stderr" in k:
                continue
            try:
                flat[f"{prefix}/{task}/{k}"] = float(v)
            except (TypeError, ValueError):
                continue
    return flat


# Pareto-front EA search

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
    max_prune: float = 0.1,
    math_task: str = 'gsm8k_cot',
    general_task: str = 'mmlu_high_school_world_history',
) -> tuple[
    list[dict[str, float]],
    np.ndarray,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    """
    Find a Pareto front trading off math accuracy vs. general ability.

    Computes activation-magnitude importance masks for the math (train) and
    calibration data, then runs a multi-objective EA over per-layer
    intervention strengths. Returns one strength dict per Pareto point, the
    (math, general) scores for each, and the two masks.
    """
    # Importance is measured from activation magnitudes recorded by hooks;
    # the hooks must be removed before the EA re-runs the model many times.
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
        make_eval_fn(tokenizer, math_task, general_task, n=eval_samples, batch_size=eval_batch_size),
        pop_size=pop_size, n_gen=n_gen, seed=seed,
        mode=mode, max_scale=max_scale, max_prune=max_prune,
    )
    if result.F is None or result.X is None:
        raise RuntimeError("EA search returned no Pareto front.")

    # pymoo minimizes, so objectives were negated during search; flip back.
    pareto_F = -result.F
    pareto_X = result.X
    strengths_list = [
        {layer_names[j]: float(pareto_X[i, j]) for j in range(len(layer_names))}
        for i in range(pareto_X.shape[0])
    ]
    return strengths_list, pareto_F, math_mask, calib_mask

# lm-eval scoring

_LM_EVAL_TASK_MANAGER: Any = None


def _get_task_manager():
    global _LM_EVAL_TASK_MANAGER
    if _LM_EVAL_TASK_MANAGER is None:
        _LM_EVAL_TASK_MANAGER = TaskManager()
    return _LM_EVAL_TASK_MANAGER


# Preferred scalar metric per lm_eval task, tried in order. Falls back to the
# first non-stderr numeric entry so new tasks still yield a usable scalar.
_LM_EVAL_METRIC_PRIORITY: tuple[str, ...] = (
    'exact_match,strict-match',
    'exact_match,flexible-extract',
    'exact_match,none',
    'acc_norm,none',
    'acc,none',
)


def _extract_scalar_metric(task_result: dict[str, Any]) -> float:
    for key in _LM_EVAL_METRIC_PRIORITY:
        if key in task_result:
            return float(task_result[key])
    for k, v in task_result.items():
        if k == 'alias' or 'stderr' in k:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return 0.0


@torch.no_grad()
def lm_eval_score(
    model: nn.Module,
    tokenizer,
    task: str,
    limit: int = 32,
    batch_size: int = 8,
    seed: int = 42,
) -> float:
    results = simple_evaluate(
        model=_wrap_lm(model, tokenizer, batch_size=batch_size),
        tasks=[task],
        task_manager=_get_task_manager(),
        log_samples=False,
        batch_size=batch_size,
        limit=limit,
        random_seed=seed,
        verbosity='ERROR',
    )
    if results is None:
        return 0.0
    res = results['results']
    task_result = res.get(task)
    if task_result is None and res:
        task_result = next(iter(res.values()))
    return _extract_scalar_metric(task_result or {})


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
        model=_wrap_lm(model, tokenizer, batch_size=batch_size),
        tasks=[task],
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


def make_eval_fn(
    tokenizer,
    math_task: str,
    general_task: str,
    n: int = 32,
    batch_size: int = 8,
):
    def eval_fn(model):
        math_score = lm_eval_score(
            model, tokenizer, math_task, limit=n, batch_size=min(batch_size, 8),
        )
        general_score = lm_eval_score(
            model, tokenizer, general_task, limit=n, batch_size=min(batch_size, 8),
        )
        return math_score, general_score
    return eval_fn

# Evaluation pipeline

def run_pre_train_eval(
    args,
    model: nn.Module,
    tokenizer,
    results_root: str,
) -> None:
    if args.train_lm_eval_task is not None:
        train_results = run_lm_eval(
            model, tokenizer, args.train_lm_eval_task,
            args.eval_dataset_subset, args.random_state,
            batch_size=args.batch_size,
            sample_dump_path=f"{results_root}pre_results_train_task_samples.json",
        )
        save_json(f"{results_root}pre_results_train_task.json", train_results)
        wandb_log_safe(_flatten_lm_eval_results("pre/train_task", train_results))

        eval_results = run_lm_eval(
            model, tokenizer, args.eval_datasets,
            args.eval_dataset_subset, args.random_state,
            batch_size=args.batch_size,
            sample_dump_path=f"{results_root}pre_results_eval_samples.json",
        )
        save_json(f"{results_root}pre_results.json", eval_results)
        wandb_log_safe(_flatten_lm_eval_results("pre/eval", eval_results))


def run_post_prune_eval(
    args,
    model: nn.Module,
    tokenizer,
    results_root: str,
    dataset_name: str,
    good_percent: float,
    repeat: int,
    wandb_prefix: str | None = None,
) -> None:
    if args.train_lm_eval_task is not None:
        tag = f"{dataset_name}_calculate{good_percent}_run{repeat}"
        train_results = run_lm_eval(
            model, tokenizer, args.train_lm_eval_task,
            args.eval_dataset_subset, args.random_state,
            batch_size=args.batch_size,
            sample_dump_path=f"{results_root}{tag}_train_task_samples.json",
        )
        save_json(f"{results_root}{tag}_train_task.json", train_results)
        if wandb_prefix:
            wandb_log_safe(_flatten_lm_eval_results(f"{wandb_prefix}/train_task", train_results))

        eval_results = run_lm_eval(
            model, tokenizer, args.eval_datasets,
            args.eval_dataset_subset, args.random_state,
            batch_size=args.batch_size,
            sample_dump_path=f"{results_root}{tag}_eval_samples.json",
        )
        save_json(f"{results_root}{tag}.json", eval_results)
        if wandb_prefix:
            wandb_log_safe(_flatten_lm_eval_results(f"{wandb_prefix}/eval", eval_results))


# Entry point

def main():
    """
    Drive the full sweep: for each calibration dataset x repeat x keep-ratio,
    either run the EA Pareto search (``--with_ea``) or the fixed math-specific
    prune, then evaluate the resulting model.
    """
    args = load_config()

    train = load_train_dataset(args)
    calibration_datasets = load_calibration_datasets(args)

    results_root = make_results_root(args)
    output_file = f"{args.save_path}/eval_results/{args.model}/{args.text_file}"

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    wandb_project = f"mathneuro-{args.model.split('/')[-1]}"

    if args.pre_train_eval:
        pre_run = init_wandb(
            project=wandb_project,
            run_name="pre_train_eval",
            config=args.as_dict() if hasattr(args, "as_dict") else {"phase": "pre"},
            job_type="pre_eval",
        )
        try:
            model = load_model(args.model)
            run_pre_train_eval(args, model, tokenizer, results_root)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            finish_wandb(pre_run)

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
                model = load_model(args.model)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Build the per-iteration wandb config up front so cached and
                # uncached paths get the same metadata.
                iter_run_name = f"{dataset_name}_keep{keep_ratio}_run{repeat}"
                iter_config = {
                    'model': args.model,
                    'calib_dataset': dataset_name,
                    'keep_ratio': keep_ratio,
                    'repeat': repeat,
                    'num_samples': num_samples,
                    'seed': args.random_state,
                    'with_ea': bool(args.with_ea),
                    'ea_mode': args.ea_mode,
                    'ea_pop_size': args.ea_pop_size,
                    'ea_n_gen': args.ea_n_gen,
                    'ea_eval_samples': args.ea_eval_samples,
                    'ea_max_scale': args.ea_max_scale,
                    'ea_max_prune': args.ea_max_prune,
                    'ea_fitness_version': args.ea_fitness_version,
                    'batch_size': args.batch_size,
                }

                # One wandb run per outer iteration, wrapping EVERYTHING --
                # cache check, search, all per-Pareto-point evaluations.
                wandb_run = init_wandb(
                    project=wandb_project,
                    run_name=iter_run_name,
                    config=iter_config,
                    group=dataset_name,
                    job_type="ea" if args.with_ea else "fixed_prune",
                    tags=[f"keep={keep_ratio}", f"mode={args.ea_mode}"],
                )

                try:
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
                            'max_prune': args.ea_max_prune,
                            'fitness_version': args.ea_fitness_version,
                        }
                        # Reuse a prior EA run only if its config matches exactly;
                        # the search is expensive so checkpoints save full reruns.
                        cached = try_load_ea_checkpoint(ckpt_path, ckpt_config)
                        if cached is not None:
                            print(f"[EA ckpt] loaded {ckpt_path}")
                            strengths_list = cached['strengths_list']
                            scores = cached['scores']
                            math_mask = cached['math_mask']
                            calib_mask = cached['calib_mask']
                            wandb_log_safe({"ea/from_cache": 1, "ea/pareto_size_final": len(strengths_list)})
                        else:
                            wandb_log_safe({"ea/from_cache": 0})
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
                                max_prune=args.ea_max_prune,
                                math_task=args.train_lm_eval_task or 'gsm8k_cot',
                                general_task=getattr(args, 'ea_general_task', 'mmlu_high_school_world_history'),
                            )
                            save_ea_checkpoint(ckpt_path, {
                                'config': ckpt_config,
                                'strengths_list': strengths_list,
                                'scores': scores,
                                'math_mask': math_mask,
                                'calib_mask': calib_mask,
                            })
                            print(f"[EA ckpt] saved {ckpt_path}")

                        # Each Pareto point is applied to the same base weights,
                        # evaluated, then rolled back via this snapshot.
                        weight_snapshot = backup_weights(model)
                        for point_idx, strengths in enumerate(strengths_list):
                            math_score, general_score = scores[point_idx]
                            stats = compute_prune_stats(math_mask, calib_mask, strengths)

                            append_text(
                                output_file,
                                f"[EA point {point_idx}] mode={args.ea_mode} "
                                f"keep_ratio={keep_ratio} calib={dataset_name} "
                                f"math_acc={math_score:.4f} general_score={general_score:.4f} "
                                f"intervention_strength={stats['effective_intervention_strength']:.4f}",
                            )
                            wandb_log_safe({
                                "pareto/point_idx": point_idx,
                                "pareto/math_acc_search": float(math_score),
                                "pareto/general_search": float(general_score),
                                "pareto/intervention_strength": float(stats['effective_intervention_strength']),
                            })

                            intervention_mask = build_intervention_mask_per_layer(
                                math_mask, calib_mask, strengths,
                                mode=args.ea_mode,
                                max_scale=args.ea_max_scale,
                                max_prune=args.ea_max_prune,
                            )
                            apply_mask_to_model(model, intervention_mask)

                            run_post_prune_eval(
                                args, model, tokenizer, results_root,
                                dataset_name=f"{dataset_name}_ea{point_idx}",
                                good_percent=keep_ratio,
                                repeat=repeat,
                                wandb_prefix=f"post/ea_point_{point_idx}",
                            )

                            restore_weights(model, weight_snapshot)
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                        _wandb_summary_update({
                            "pareto/n_points": len(strengths_list),
                            "pareto/best_math_acc_search": float(max(s[0] for s in scores)) if len(scores) else 0.0,
                            "pareto/best_general_search": float(max(s[1] for s in scores)) if len(scores) else 0.0,
                        })
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
                            args, model, tokenizer, results_root,
                            dataset_name=dataset_name, good_percent=keep_ratio,
                            repeat=repeat,
                            wandb_prefix="post/fixed",
                        )
                finally:
                    finish_wandb(wandb_run)

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()