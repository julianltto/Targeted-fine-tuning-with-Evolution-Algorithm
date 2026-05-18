from __future__ import annotations

from typing import Callable, Iterable

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle


def _make_hook(magnitude: dict[str, torch.Tensor], name: str):
    def hook(module: nn.Linear, inputs, output):
        activations = inputs[0]  
        activations_norm = activations.norm(p=2, dim=1).to(torch.bfloat16)
        magnitude[name] = (activations_norm * torch.abs(module.weight.data)).detach()
    return hook


def register_activation_hooks(
    model: nn.Module,
) -> tuple[dict[str, torch.Tensor], list[RemovableHandle]]:
    magnitude: dict[str, torch.Tensor] = {}
    handles: list[RemovableHandle] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(_make_hook(magnitude, name)))
    return magnitude, handles


def remove_hooks(handles: Iterable[RemovableHandle]) -> None:
    for handle in handles:
        handle.remove()


def make_calibration_prompt_fn() -> Callable[[pd.Series], str]:
    return lambda row: row['0']


def make_math_prompt_fn(df: pd.DataFrame) -> Callable[[pd.Series], str]:
    if 'qa' in df.columns:
        return lambda row: row['qa']

    def fn(row: pd.Series) -> str:
        return (
            f"Instruct: {row['question']} Let's write a Python program.\n"
            f"Output:\n{row['solution']}"
        )
    return fn


def compute_importance(
    model: nn.Module,
    tokenizer,
    df: pd.DataFrame,
    prompt_fn: Callable[[pd.Series], str],
    magnitude: dict[str, torch.Tensor],
    num_samples: int,
) -> dict[str, torch.Tensor]:
    scores: dict[str, torch.Tensor] = {
        f"{name}.weight": torch.zeros_like(module.weight, device='cpu')
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    }

    n = min(num_samples, len(df))
    for i in range(n):
        prompt = prompt_fn(df.iloc[i])
        inputs = tokenizer.encode(prompt, return_tensors='pt').to(model.device)

        with torch.no_grad():
            model(inputs)

        for layer_name, tensor in magnitude.items():
            key = f"{layer_name}.weight"
            if key in scores:
                scores[key] += tensor.detach().cpu()

        magnitude.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return scores


def top_k_mask(
    scores: dict[str, torch.Tensor],
    keep_ratio: float,
    exclude_substring: str = 'embed',
) -> dict[str, torch.Tensor]:
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
    exclude_substring: str = 'embed',
) -> dict[str, torch.Tensor]:
    # Returns a bool mask: True at positions to be scaled by `factor` in apply_mask_to_model.
    # Bool is 8× smaller than the previous fp32 mask.
    pruning_masks: dict[str, torch.Tensor] = {}
    for name, math_mask in math_important.items():
        if exclude_substring in name:
            pruning_masks[name] = torch.zeros_like(math_mask, dtype=torch.bool)
            continue

        calib_mask = calib_important[name]
        pruning_masks[name] = math_mask & (~calib_mask)
    return pruning_masks


def apply_mask_to_model(
    model: nn.Module,
    mask: dict[str, torch.Tensor],
    factor: float = 0.0,
) -> None:
    # Two mask formats are supported:
    #   - bool tensor: True positions are scaled by `factor` (memory-efficient path).
    #   - float tensor: elementwise multiplier applied to the parameter (legacy / per-layer EA path).
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in mask:
                continue
            m = mask[name].to(device=param.device)
            if m.dtype == torch.bool:
                if not m.any():
                    continue
                if factor == 0:
                    param.masked_fill_(m, 0)
                else:
                    param[m] = param[m] * factor
            else:
                param.mul_(m.to(dtype=param.dtype))
