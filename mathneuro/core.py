"""
The core realization of the MathNeuro pruning:

    1. Register forward hooks to all nn.Linear layers to record "activation × |weight|" during forward passes.
         —— register_activation_hooks()

    2. Run forward on a batch of math training samples to accumulate "math importance scores" for each parameter position. 
    Then do the same with non-math (calibration) samples to get "calibration importance scores".
         —— compute_importance()

    3. For each layer, select the top-k positions with the largest |score| to get a boolean mask:
    True = this position is important for the task.

    4. Combine math_important and calib_important: positions that are important for math but not for general tasks are "math-specific neurons". 
    These form the final pruning target.
         —— build_prune_mask()

    5. Multiply this pruning mask back to the model parameters to complete pruning.
        —— apply_mask_to_model()
"""
from __future__ import annotations

from typing import Callable, Iterable

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle


# ---------------------------------------------------------------------------
# Forward hook
# ---------------------------------------------------------------------------

def _make_hook(magnitude: dict[str, torch.Tensor], name: str):
    """
    Generate a forward hook that records the "activation × |weight|" for this Linear layer during the forward pass.
    """
    def hook(module: nn.Linear, inputs, output):
        activations = inputs[0]  
        activations_norm = activations.norm(p=2, dim=1).to(torch.bfloat16)
        magnitude[name] = (activations_norm * torch.abs(module.weight.data)).detach()
    return hook


def register_activation_hooks(
    model: nn.Module,
) -> tuple[dict[str, torch.Tensor], list[RemovableHandle]]:
    """
    Regsiter hooks to all nn.Linear layers in the model to record their activation magnitudes during forward passes.
    Usage:
        magnitude, handles = register_activation_hooks(model)
        # ... forward ...
        remove_hooks(handles)
    """
    magnitude: dict[str, torch.Tensor] = {}
    handles: list[RemovableHandle] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(_make_hook(magnitude, name)))
    return magnitude, handles


def remove_hooks(handles: Iterable[RemovableHandle]) -> None:
    """Remove the registered hooks to clean up."""
    for handle in handles:
        handle.remove()


# ---------------------------------------------------------------------------
# Prompt extraction
# ---------------------------------------------------------------------------

def make_calibration_prompt_fn() -> Callable[[pd.Series], str]:
    """Returning a function that extracts the prompt from a row of the calibration dataset."""
    return lambda row: row['0']


def make_math_prompt_fn(df: pd.DataFrame) -> Callable[[pd.Series], str]:
    """
    There are two formats of math dataset:
        1) Some datasets already have a 'qa' column that contains the full prompt (question + solution).
       In this case, we can directly use that column as the prompt.
        2) Other datasets only have separate 'question' and 'solution' columns.
       In this case, we need to construct the prompt by concatenating the question and solution in a specific format.
    """
    if 'qa' in df.columns:
        return lambda row: row['qa']

    def fn(row: pd.Series) -> str:
        return (
            f"Instruct: {row['question']} Let's write a Python program.\n"
            f"Output:\n{row['solution']}"
        )
    return fn


# ---------------------------------------------------------------------------
# Importance score computation
# ---------------------------------------------------------------------------

def compute_importance(
    model: nn.Module,
    tokenizer,
    df: pd.DataFrame,
    prompt_fn: Callable[[pd.Series], str],
    magnitude: dict[str, torch.Tensor],
    num_samples: int,
) -> dict[str, torch.Tensor]:
    """
    Run forward passes on the given dataset to accumulate importance scores for each parameter position.

    Inputs:
        model       : model to analyze, should already have hooks registered to record activations.
        tokenizer   : corresponding tokenizer for encoding prompts into model inputs.
        df          : dataframe(math or calibration)
        prompt_fn   : function to extract prompt from a row of df
        magnitude   : dict that hooks write to.
        num_samples : number of samples to run for importance estimation.

    Returns:
        {param_name -> importance tensor}, only for Linear weight parameters that are actually monitored by hooks.

    """
    # Here the accumulation is done on CPU to avoid GPU OOM on my own laptop, but it can be later changed to model.device
    # 1) Initialize the importance scores dict.
    scores: dict[str, torch.Tensor] = {
        name: torch.zeros_like(param, device='cpu') 
        for name, param in model.named_parameters()
    }

    # 2) Track which layers are actually monitored by hooks.
    seen_layers: set[str] = set()

    n = min(num_samples, len(df))
    for i in range(n):
        prompt = prompt_fn(df.iloc[i])
        inputs = tokenizer.encode(prompt, return_tensors='pt').to(model.device)

        with torch.no_grad():
            model(inputs) # Now hooks have written the "activation × |weight|" for each monitored layer into magnitude.

        for layer_name, tensor in magnitude.items():
            seen_layers.add(layer_name)
            key = f"{layer_name}.weight"
            if key in scores:
                scores[key] += tensor.detach().cpu()

        magnitude.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 3) only return the scores for the weight parameters of the layers that are monitored by hooks.
    return {
        name: tensor for name, tensor in scores.items()
        if name.endswith('.weight') and name[: -len('.weight')] in seen_layers
    }


# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------

def top_k_mask(
    scores: dict[str, torch.Tensor],
    keep_ratio: float,
    exclude_substring: str = 'embed',
) -> dict[str, torch.Tensor]:
    """
    Pick the top-k positions with the largest |score| in each layer to get a boolean mask.
    True = this position is important for the task, False = not important.

    Inputs:
        scores            : return value of compute_importance(), {param_name -> importance tensor}
        keep_ratio        : percentage of positions to keep in each layer.
        exclude_substring : layer names containing this substring will be excluded from pruning, i.e., all False.

    Returns:
        {param_name -> bool tensor}
    """
    masks: dict[str, torch.Tensor] = {}
    for name, score in scores.items():
        if exclude_substring in name:
            masks[name] = torch.zeros_like(score, dtype=torch.bool)
            continue

        flat = score.view(-1)
        keep_num = int(flat.numel() * keep_ratio)

        top_positions = torch.topk(flat.abs(), keep_num, largest=True).indices

        mask_flat = torch.zeros_like(flat, dtype=torch.bool)
        mask_flat[top_positions] = True
        masks[name] = mask_flat.view(score.shape)
    return masks


def build_prune_mask(
    math_important: dict[str, torch.Tensor],
    calib_important: dict[str, torch.Tensor],
    factor: float = 0.0,
    exclude_substring: str = 'embed',
) -> dict[str, torch.Tensor]:
    """
    Combine the two boolean masks to get the final "multiplicative pruning mask".
    Keep positions that are important for both math and calibration ( general importance ), 
    prune positions that are important for math but not for calibration ( math-specific importance ).

    Inputs:
        math_important / calib_important : output of top_k_mask()
        factor                           : factor to multiply to the math-specific positions.
                                           0 = prune, 1 = keep, (0, 1) = partially prune.
        exclude_substring                : same as in top_k_mask(), layer names containing this substring will be excluded from pruning, i.e., all 1.
    """
    pruning_masks: dict[str, torch.Tensor] = {}
    for name, math_mask in math_important.items():
        if exclude_substring in name:
            pruning_masks[name] = torch.ones_like(math_mask, dtype=torch.float32)
            continue

        calib_mask = calib_important[name]
        math_only = math_mask & (~calib_mask)

        mask = torch.ones_like(math_mask, dtype=torch.float32)
        mask[math_only] = factor
        pruning_masks[name] = mask
    return pruning_masks


# ---------------------------------------------------------------------------
# Apply mask to model
# ---------------------------------------------------------------------------

def apply_mask_to_model(model: nn.Module, mask: dict[str, torch.Tensor]) -> None:
    """
    Apply the pruning mask back to the model parameters by multiplying the mask to the corresponding weight tensors.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask:
                param.mul_(mask[name].to(device=param.device, dtype=param.dtype))
