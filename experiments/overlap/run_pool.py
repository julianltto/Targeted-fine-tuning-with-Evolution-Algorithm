"""Drive the intervention pool end-to-end.

Usage:

    python -m experiments.overlap.run_pool \
        --config configs/mathneuro_gsm8k.yaml \
        --out results/overlap \
        --pool minimal \
        --task gsm8k_cot

The script computes (or loads) math_important / calib_important, then loops
over every InterventionConfig in the chosen pool, scaling the corresponding
mask positions and dumping per-sample correctness as parquet.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
import torch

from mathneuro.config import MathNeuroConfig
from mathneuro.core import (
    compute_importance,
    make_calibration_prompt_fn,
    make_math_prompt_fn,
    register_activation_hooks,
    remove_hooks,
    top_k_mask,
)

from .intervention_configs import (
    full_pool,
    minimal_viable_pool,
)
from .runner import run_baseline, run_pool


def _load_model_and_tokenizer(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda")
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.generation_config.cache_implementation = None
    model.generation_config.max_length = 4096
    return model, tokenizer


def _load_or_compute_masks(
    model,
    tokenizer,
    train_csv: str,
    calib_csvs: list[str],
    calib_names: list[str],
    keep_ratio: float,
    num_samples: int,
    random_state: int,
    cache_path: Path,
):
    """Return (math_important, calib_important, math_scores). Cached to a pickle file."""
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    train_df = pd.read_csv(train_csv).sample(frac=1, random_state=random_state)
    # Take the first declared calibration set; identical to run.py default behavior.
    calib_df = pd.read_csv(calib_csvs[0]).sample(frac=1, random_state=random_state)
    calib_name = str(calib_names[0])

    magnitude, handles = register_activation_hooks(model)
    try:
        math_scores = compute_importance(
            model, tokenizer, train_df,
            make_math_prompt_fn(train_df), magnitude, num_samples,
        )
        math_important = top_k_mask(math_scores, keep_ratio)

        # Mirror run.py: 'Bad*' calibration sets use the calibration prompt
        # ("0" column); everything else (Race, MMLU, ...) uses the same prompt
        # template as the math task so the magnitudes are comparable.
        calib_prompt_fn = (
            make_calibration_prompt_fn() if "Bad" in calib_name
            else make_math_prompt_fn(calib_df)
        )
        calib_scores = compute_importance(
            model, tokenizer, calib_df,
            calib_prompt_fn, magnitude, num_samples,
        )
        calib_important = top_k_mask(calib_scores, keep_ratio)
    finally:
        remove_hooks(handles)

    # math_scores aligned to parameter names; rename ".weight" suffix away
    out = {
        "math_important": math_important,
        "calib_important": calib_important,
        "math_scores": math_scores,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(out, f)
    return out


def _num_transformer_layers(model) -> int:
    cfg = getattr(model, "config", None)
    if cfg is not None and getattr(cfg, "num_hidden_layers", None) is not None:
        return int(cfg.num_hidden_layers)
    # fallback: count .layers.<i>.
    import re
    rng = re.compile(r"\.layers\.(\d+)\.")
    seen: set[int] = set()
    for n, _ in model.named_parameters():
        m = rng.search(n)
        if m:
            seen.add(int(m.group(1)))
    return max(seen) + 1 if seen else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="path to mathneuro_*.yaml")
    p.add_argument("--out", required=True, help="output root for per-sample parquets")
    p.add_argument("--pool", choices=["minimal", "full"], default="minimal")
    p.add_argument("--task", default="gsm8k_cot")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--mask-cache",
        default=None,
        help="path to a pickle cache of (math_important, calib_important, math_scores)",
    )
    args = p.parse_args()

    cfg = MathNeuroConfig.from_yaml(args.config)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    model, tokenizer = _load_model_and_tokenizer(cfg.model)

    proportion = cfg.proportion[0] if isinstance(cfg.proportion, list) else cfg.proportion
    keep_ratio = float(proportion)

    mask_cache = Path(args.mask_cache or (out_root / "masks_cache.pkl"))
    bundle = _load_or_compute_masks(
        model, tokenizer,
        train_csv=cfg.train_dataset,
        calib_csvs=cfg.calibration_datasets,
        calib_names=cfg.calibration_dataset_names,
        keep_ratio=keep_ratio,
        num_samples=cfg.num_samples,
        random_state=cfg.random_state,
        cache_path=mask_cache,
    )
    math_important = bundle["math_important"]
    calib_important = bundle["calib_important"]
    math_scores = bundle["math_scores"]

    num_layers = _num_transformer_layers(model)
    if args.pool == "minimal":
        configs = minimal_viable_pool(num_layers)
    else:
        configs = full_pool(num_layers)

    print(f"[overlap] model={cfg.model}  num_layers={num_layers}  pool_size={len(configs)}")

    run_baseline(
        out_root, model, tokenizer,
        task=args.task,
        eval_subset=cfg.eval_dataset_subset,
        random_state=cfg.random_state,
        batch_size=cfg.batch_size,
        overwrite=args.overwrite,
    )
    run_pool(
        out_root, configs, model, tokenizer,
        math_important, calib_important, math_scores,
        task=args.task,
        eval_subset=cfg.eval_dataset_subset,
        random_state=cfg.random_state,
        batch_size=cfg.batch_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
