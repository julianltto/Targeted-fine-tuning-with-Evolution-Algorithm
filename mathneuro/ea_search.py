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


def _group_key(layer_name: str, group_by: str) -> str:
    """
    Map a layer's parameter name to the key of the EA group it belongs to.

    'none'      -> the layer name itself (one variable per layer, ~112 vars).
    'proj_type' -> the projection name, e.g. 'q_proj', shared across all blocks.
    'block'     -> 'block.<i>' for every layer inside transformer block i.

    Names that don't match the expected pattern (e.g. 'lm_head.weight') fall
    back to their own name and simply become singleton groups.
    """
    if group_by == 'none':
        return layer_name

    parts = layer_name.split('.')
    if group_by == 'proj_type':
        for part in parts:
            if part.endswith('_proj'):
                return part
        return layer_name
    if group_by == 'block':
        for i, part in enumerate(parts):
            if part == 'layers' and i + 1 < len(parts):
                return f'block.{parts[i + 1]}'
        return layer_name

    raise ValueError(
        f"group_by must be 'none', 'proj_type' or 'block', got {group_by!r}"
    )


def build_layer_groups(
    layer_names: list[str],
    group_by: str = 'none',
) -> tuple[list[str], dict[str, int]]:
    """
    Partition prunable layers into EA optimization groups.

    Returns ``(group_names, layer_to_group)``: ``group_names`` is the ordered
    list of group keys whose length is the EA ``n_var``; ``layer_to_group``
    maps every layer name to the index of its group. Every layer in a group
    shares one optimized strength, which is what lets a small evaluation
    budget cover the whole model instead of degenerating into random search.
    """
    group_names: list[str] = []
    index_of: dict[str, int] = {}
    layer_to_group: dict[str, int] = {}
    for name in layer_names:
        key = _group_key(name, group_by)
        if key not in index_of:
            index_of[key] = len(group_names)
            group_names.append(key)
        layer_to_group[name] = index_of[key]
    return group_names, layer_to_group


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


def _build_problem_class(n_obj: int = 2):
    from pymoo.core.problem import ElementwiseProblem

    class PerLayerFactorProblem(ElementwiseProblem):
        def __init__(
            self,
            model: nn.Module,
            math_important: dict[str, torch.Tensor],
            calib_important: dict[str, torch.Tensor],
            layer_to_group: dict[str, int],
            n_groups: int,
            weight_backup: dict[str, torch.Tensor],
            eval_fn: Callable[[nn.Module], tuple[float, ...]],
            mode: str = 'prune',
            max_scale: float = 0.1,
            exclude_substring: str = 'embed',
        ):
            if mode not in {'prune', 'scale'}:
                raise ValueError(f"mode must be 'prune' or 'scale', got {mode!r}")
            super().__init__(
                n_var=n_groups,
                n_obj=n_obj,
                n_constr=0,
                xl=np.zeros(n_groups),
                xu=np.ones(n_groups),
            )
            self.model = model
            self.math_important = math_important
            self.calib_important = calib_important
            self.layer_to_group = layer_to_group
            self.weight_backup = weight_backup
            self.eval_fn = eval_fn
            self.mode = mode
            self.max_scale = max_scale
            self.exclude_substring = exclude_substring
            # running ideal/nadir for online normalization (higher acc = better)
            self._ideal = np.zeros(n_obj)
            self._nadir = np.ones(n_obj)
            self._n_seen = 0

        def _evaluate(self, x, out, *args, **kwargs):
            strengths = {
                name: float(x[group_idx])
                for name, group_idx in self.layer_to_group.items()
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
                    target = (1.0 - strength) if self.mode == 'prune' else (1.0 + self.max_scale * strength)
                    params[name][math_only] = params[name][math_only] * target
            try:
                raw = self.eval_fn(self.model)
            finally:
                restore_weights(self.model, self.weight_backup)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            accs = np.array([s if _math.isfinite(s) else 0.0 for s in raw])

            self._n_seen += 1

            _labels = (["math", "holdout", "general"] if n_obj == 3
                       else ["math", "general"] if n_obj == 2
                       else ["math"])
            wandb = _active_wandb()
            if wandb is not None:
                wandb.log({"eval/n": self._n_seen,
                           **{f"eval/{_labels[i]}": float(accs[i]) for i in range(len(accs))}})

            if n_obj > 1:
                # normalize multi-objective scores so each axis is comparable
                self._ideal = np.maximum(self._ideal, accs)
                self._nadir = np.minimum(self._nadir, accs)
                if self._n_seen >= n_obj * 5:  # warm-up: 5 evals per objective
                    scale = self._ideal - self._nadir
                    scale[scale < 1e-6] = 1.0
                    accs = (accs - self._nadir) / scale

            out["F"] = -accs  # negate: pymoo minimizes

    return PerLayerFactorProblem


def _active_wandb():
    """Return the wandb module iff it is importable and has a live run, else None."""
    try:
        import wandb
    except Exception:
        return None
    return wandb if wandb.run is not None else None


def _build_wandb_callback(
    group_names: list[str] | None = None,
    seed_ref: list | None = None,
    n_obj: int = 2,
):
    from pymoo.core.callback import Callback

    obj_labels = (["math", "holdout", "general"] if n_obj == 3
                  else ["math", "general"] if n_obj == 2
                  else ["math"])

    class WandbCallback(Callback):
        def notify(self, algorithm):
            if seed_ref is not None:
                seed_ref[0] = int(algorithm.n_gen)

            acc = -algorithm.pop.get("F")   # (pop_size, n_obj), normalized scores
            opt = -algorithm.opt.get("F")
            X = algorithm.pop.get("X")

            names = group_names or [str(i) for i in range(X.shape[1])]

            obj_str = "  ".join(
                f"{lbl}_best={acc[:, i].max():.4f}  {lbl}_mean={acc[:, i].mean():.4f}"
                for i, lbl in enumerate(obj_labels)
            )
            print(f"[EA gen {int(algorithm.n_gen):>3}]  {obj_str}  pareto={len(opt)}")
            col_w = max(len(n) for n in names)
            for i, name in enumerate(names):
                vals = X[:, i]
                print(f"  {name:<{col_w}}  mean={vals.mean():.4f}  std={vals.std():.4f}  "
                      f"min={vals.min():.4f}  max={vals.max():.4f}")

            wandb = _active_wandb()
            if wandb is None:
                return
            gen = int(algorithm.n_gen)
            log = {"ea/gen": gen, "ea/pareto_size": int(len(opt))}
            for i, lbl in enumerate(obj_labels):
                log[f"ea/{lbl}_acc_best"] = float(acc[:, i].max())
                log[f"ea/{lbl}_acc_mean"] = float(acc[:, i].mean())
            for i, name in enumerate(names):
                vals = X[:, i]
                log[f"scales/{name}/mean"] = float(vals.mean())
                log[f"scales/{name}/std"]  = float(vals.std())
                log[f"scales/{name}/max"]  = float(vals.max())
                log[f"scales/{name}/min"]  = float(vals.min())
            wandb.log(log)

    return WandbCallback


def _log_pareto_scatter(result) -> None:
    wandb = _active_wandb()
    if wandb is None or result.F is None or result.F.shape[1] < 2:
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
    eval_fn: Callable[[nn.Module], tuple[float, ...]],
    pop_size: int = 30,
    n_gen: int = 30,
    mode: str = 'prune',
    max_scale: float = 0.1,
    exclude_substring: str = 'embed',
    seed: int = 42,
    verbose: bool = True,
    group_by: str = 'none',
    seed_ref: list | None = None,
    n_obj: int = 2,
):
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.optimize import minimize

    layer_names = list_prunable_layer_names(math_important, exclude_substring)
    group_names, layer_to_group = build_layer_groups(layer_names, group_by)
    print(
        f"[EA] group_by={group_by!r}: {len(layer_names)} prunable layers "
        f"-> {len(group_names)} search variables  n_obj={n_obj}"
    )

    weight_backup = backup_weights(model)

    ProblemClass = _build_problem_class(n_obj)
    problem = ProblemClass(
        model=model,
        math_important=math_important,
        calib_important=calib_important,
        layer_to_group=layer_to_group,
        n_groups=len(group_names),
        weight_backup=weight_backup,
        eval_fn=eval_fn,
        mode=mode,
        max_scale=max_scale,
        exclude_substring=exclude_substring,
    )

    if n_obj == 1:
        from pymoo.algorithms.soo.nonconvex.ga import GA
        algorithm = GA(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
        )
    else:
        from pymoo.algorithms.moo.nsga2 import NSGA2
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
        callback=_build_wandb_callback(group_names, seed_ref, n_obj=n_obj)(),
    )

    if n_obj == 1:
        # GA returns a single best; take top-k from final population as candidates
        pop_F = result.pop.get("F")   # (pop_size, 1)
        pop_X = result.pop.get("X")   # (pop_size, n_var)
        top_k = max(1, min(10, pop_size // 4))
        idx = np.argsort(pop_F.flatten())[:top_k]
        result.F = pop_F[idx]
        result.X = pop_X[idx]

    _log_pareto_scatter(result)
    return result, layer_names, layer_to_group


def format_pareto_front(
    result,
    group_names: list[str],
    top_k: int = 5,
) -> str:
    lines = []
    X = result.X
    F = -result.F
    order = np.argsort(-F[:, 0])

    for rank, idx in enumerate(order):
        scores = F[idx]
        strengths = X[idx]
        top = np.argsort(-strengths)[:top_k]
        groups_str = ", ".join(
            f"{group_names[i]}:{strengths[i]:.2f}" for i in top
        )
        scores_str = "  ".join(f"obj{i}={scores[i]:.4f}" for i in range(len(scores)))
        lines.append(f"[{rank:2d}] {scores_str}  top-pruned: {groups_str}")
    return "\n".join(lines)
