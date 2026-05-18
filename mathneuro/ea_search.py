from __future__ import annotations

import math as _math
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from mathneuro.core import apply_mask_to_model


def list_prunable_layer_names(
    math_important: dict[str, torch.Tensor],
    exclude_substring: str = 'embed',
) -> list[str]:
    return sorted(
        name for name in math_important.keys()
        if exclude_substring not in name
    )


def build_intervention_mask_per_layer(
    math_important: dict[str, torch.Tensor],
    calib_important: dict[str, torch.Tensor],
    strengths: dict[str, float],
    mode: str = 'prune',
    max_scale: float = 0.1,
    exclude_substring: str = 'embed',
) -> dict[str, torch.Tensor]:
    if mode not in {'prune', 'scale'}:
        raise ValueError(f"mode must be 'prune' or 'scale', got {mode!r}")

    masks: dict[str, torch.Tensor] = {}
    for name, math_mask in math_important.items():
        if exclude_substring in name:
            masks[name] = torch.ones_like(math_mask, dtype=torch.float32)
            continue

        calib_mask = calib_important[name]
        math_only = math_mask & (~calib_mask)

        strength = float(strengths.get(name, 0.0))
        if mode == 'prune':
            target_value = 1.0 - strength
        else:
            target_value = 1.0 + max_scale * strength

        mask = torch.ones_like(math_mask, dtype=torch.float32)
        mask[math_only] = target_value
        masks[name] = mask
    return masks


def build_prune_mask_per_layer(
    math_important: dict[str, torch.Tensor],
    calib_important: dict[str, torch.Tensor],
    strengths: dict[str, float],
    exclude_substring: str = 'embed',
) -> dict[str, torch.Tensor]:
    return build_intervention_mask_per_layer(
        math_important, calib_important, strengths,
        mode='prune', exclude_substring=exclude_substring,
    )


def backup_weights(model: nn.Module, device: str = 'cpu') -> dict[str, torch.Tensor]:
    return {n: p.detach().to(device, copy=True) for n, p in model.named_parameters()}


def restore_weights(model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in backup:
                param.copy_(backup[name].to(param.device, non_blocking=True))


def _build_problem_class():
    from pymoo.core.problem import ElementwiseProblem

    class PerLayerFactorProblem(ElementwiseProblem):
        def __init__(
            self,
            model: nn.Module,
            math_important: dict[str, torch.Tensor],
            calib_important: dict[str, torch.Tensor],
            layer_names: list[str],
            weight_backup: dict[str, torch.Tensor],
            eval_fn: Callable[[nn.Module], tuple[float, float]],
            mode: str = 'prune',
            max_scale: float = 0.1,
            exclude_substring: str = 'embed',
        ):
            if mode not in {'prune', 'scale'}:
                raise ValueError(f"mode must be 'prune' or 'scale', got {mode!r}")
            super().__init__(
                n_var=len(layer_names),
                n_obj=2,
                n_constr=0,
                xl=np.zeros(len(layer_names)),
                xu=np.ones(len(layer_names)),
            )
            self.model = model
            self.math_important = math_important
            self.calib_important = calib_important
            self.layer_names = layer_names
            self.weight_backup = weight_backup
            self.eval_fn = eval_fn
            self.mode = mode
            self.max_scale = max_scale
            self.exclude_substring = exclude_substring

        def _evaluate(self, x, out, *args, **kwargs):
            strengths = {
                self.layer_names[i]: float(x[i])
                for i in range(len(self.layer_names))
            }
            params = dict(self.model.named_parameters())
            with torch.no_grad():
                for name, math_mask in self.math_important.items():
                    if self.exclude_substring in name or name not in params:
                        continue
                    calib_mask = self.calib_important[name]
                    math_only = (math_mask & (~calib_mask)).to(params[name].device)
                    if not math_only.any():
                        continue
                    strength = strengths.get(name, 0.0)
                    if self.mode == 'prune':
                        target = 1.0 - strength
                    else:
                        target = 1.0 + self.max_scale * strength
                    params[name][math_only] = params[name][math_only] * target
            try:
                math_acc, general_acc = self.eval_fn(self.model)
            finally:
                restore_weights(self.model, self.weight_backup)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if not (_math.isfinite(math_acc) and _math.isfinite(general_acc)):
                math_acc, general_acc = 0.0, -1e3

            out["F"] = [-float(math_acc), -float(general_acc)]

    return PerLayerFactorProblem


def _active_wandb():
    """Return the wandb module iff it is importable and has a live run, else None."""
    try:
        import wandb
    except Exception:
        return None
    return wandb if wandb.run is not None else None


def _build_wandb_callback():
    from pymoo.core.callback import Callback

    class WandbCallback(Callback):
        def notify(self, algorithm):
            wandb = _active_wandb()
            if wandb is None:
                return
            
            acc = -algorithm.pop.get("F")
            opt = -algorithm.opt.get("F")
            wandb.log({
                "ea/gen": int(algorithm.n_gen),
                "ea/math_acc_best": float(acc[:, 0].max()),
                "ea/math_acc_mean": float(acc[:, 0].mean()),
                "ea/general_best": float(acc[:, 1].max()),
                "ea/general_mean": float(acc[:, 1].mean()),
                "ea/pareto_size": int(len(opt)),
            })

    return WandbCallback


def _log_pareto_scatter(result) -> None:
    wandb = _active_wandb()
    if wandb is None or result.F is None:
        return
    pareto = -result.F  # [P, 2] = (math_acc, general)
    table = wandb.Table(
        columns=["math_acc", "general"],
        data=[[float(m), float(g)] for m, g in pareto],
    )
    wandb.log({
        "ea/pareto_front": wandb.plot.scatter(
            table, "math_acc", "general", title="EA Pareto front",
        ),
        "ea/pareto_size_final": int(pareto.shape[0]),
    })


def run_ea_search(
    model: nn.Module,
    math_important: dict[str, torch.Tensor],
    calib_important: dict[str, torch.Tensor],
    eval_fn: Callable[[nn.Module], tuple[float, float]],
    pop_size: int = 30,
    n_gen: int = 30,
    mode: str = 'prune',
    max_scale: float = 0.1,
    exclude_substring: str = 'embed',
    seed: int = 42,
    verbose: bool = True,
):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.optimize import minimize

    layer_names = list_prunable_layer_names(math_important, exclude_substring)

    weight_backup = backup_weights(model)

    ProblemClass = _build_problem_class()
    problem = ProblemClass(
        model=model,
        math_important=math_important,
        calib_important=calib_important,
        layer_names=layer_names,
        weight_backup=weight_backup,
        eval_fn=eval_fn,
        mode=mode,
        max_scale=max_scale,
        exclude_substring=exclude_substring,
    )

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    result = minimize(
        problem,
        algorithm,
        ("n_gen", n_gen),
        seed=seed,
        verbose=verbose,
        callback=_build_wandb_callback()(),
    )

    _log_pareto_scatter(result)
    return result, layer_names


def format_pareto_front(
    result,
    layer_names: list[str],
    top_k_layers: int = 5,
) -> str:
    lines = []
    X = result.X
    F = -result.F
    order = np.argsort(-F[:, 0])

    for rank, idx in enumerate(order):
        math_acc, gen_acc = F[idx]
        strengths = X[idx]
        top_layers = np.argsort(-strengths)[:top_k_layers]
        layers_str = ", ".join(
            f"{layer_names[i].split('.')[-2]}:{strengths[i]:.2f}"
            for i in top_layers
        )
        lines.append(
            f"[{rank:2d}] math={math_acc:.4f}  general={gen_acc:.4f}  "
            f"top-pruned: {layers_str}"
        )
    return "\n".join(lines)
