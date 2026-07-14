"""Generate intervention configurations.

Each config describes a *boolean mask family*: which positions of which
parameter tensors to scale, plus the scale value ``s``. The runner converts
``InterventionConfig`` -> a dict[str, torch.Tensor (bool)] ``selector`` that
selects positions of the math-only neurons (math_important & ~calib_important)
which belong to this subgroup.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from .io import InterventionMeta


@dataclass
class InterventionConfig:
    intervention_id: str
    family: str
    scale: float
    # how to subselect within the math-only mask
    # one of: 'all', 'layer_set', 'proj_type_set', 'causality_quantile',
    #         'random_math_subset', 'nonmath_random_subset', 'combo'
    selector_kind: str
    selector_args: dict = field(default_factory=dict)
    parameter_group: str = ""

    def meta(self) -> InterventionMeta:
        return InterventionMeta(
            intervention_id=self.intervention_id,
            family=self.family,
            scale=self.scale,
            parameter_group=self.parameter_group or self.selector_kind,
            extra={"selector_kind": self.selector_kind, "selector_args": self.selector_args},
        )


_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def layer_of(name: str) -> int | None:
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def proj_type_of(name: str) -> str | None:
    for part in name.split("."):
        if part.endswith("_proj"):
            return part
    return None


# Builders ------------------------------------------------------------------


def global_sweep(scales: Iterable[float]) -> list[InterventionConfig]:
    return [
        InterventionConfig(
            intervention_id=f"global_s{s:.3f}",
            family="global",
            scale=s,
            selector_kind="all",
            parameter_group="all math neurons",
        )
        for s in scales
    ]


def per_layer(num_layers: int, scales: Iterable[float]) -> list[InterventionConfig]:
    out = []
    for l in range(num_layers):
        for s in scales:
            out.append(InterventionConfig(
                intervention_id=f"layer{l:02d}_s{s:.3f}",
                family="layer",
                scale=s,
                selector_kind="layer_set",
                selector_args={"layers": [l]},
                parameter_group=f"layer {l}",
            ))
    return out


def layer_window(num_layers: int, window: int, stride: int, scales: Iterable[float]) -> list[InterventionConfig]:
    out = []
    for start in range(0, num_layers - window + 1, stride):
        layers = list(range(start, start + window))
        for s in scales:
            out.append(InterventionConfig(
                intervention_id=f"win{start:02d}-{start + window - 1:02d}_s{s:.3f}",
                family="layer_window",
                scale=s,
                selector_kind="layer_set",
                selector_args={"layers": layers},
                parameter_group=f"layers {layers[0]}-{layers[-1]}",
            ))
    return out


def layer_third(num_layers: int, scales: Iterable[float]) -> list[InterventionConfig]:
    third = num_layers // 3
    groups = {
        "early": list(range(0, third)),
        "middle": list(range(third, 2 * third)),
        "late": list(range(2 * third, num_layers)),
    }
    out = []
    for name, layers in groups.items():
        for s in scales:
            out.append(InterventionConfig(
                intervention_id=f"third-{name}_s{s:.3f}",
                family="layer_window",
                scale=s,
                selector_kind="layer_set",
                selector_args={"layers": layers},
                parameter_group=f"{name} third",
            ))
    return out


def by_proj_type(scales: Iterable[float], proj_types: list[str] | None = None) -> list[InterventionConfig]:
    proj_types = proj_types or ["up_proj", "gate_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"]
    out = []
    for pt in proj_types:
        for s in scales:
            out.append(InterventionConfig(
                intervention_id=f"proj-{pt}_s{s:.3f}",
                family="module_type",
                scale=s,
                selector_kind="proj_type_set",
                selector_args={"proj_types": [pt]},
                parameter_group=pt,
            ))
    return out


def causality_buckets(quantiles: Iterable[float], scales: Iterable[float]) -> list[InterventionConfig]:
    """Top-q% of math neurons by importance score (math_important magnitude)."""
    out = []
    for q in quantiles:
        for s in scales:
            out.append(InterventionConfig(
                intervention_id=f"caus-top{int(q * 100):03d}_s{s:.3f}",
                family="causality_bucket",
                scale=s,
                selector_kind="causality_quantile",
                selector_args={"top_fraction": q},
                parameter_group=f"top {q:.0%} by importance",
            ))
    return out


def random_math_controls(n_samples: int, base_fraction: float, scales: Iterable[float], seed_base: int = 0) -> list[InterventionConfig]:
    """Random same-size subsets of math-only neurons (control for cluster interventions)."""
    out = []
    for k in range(n_samples):
        for s in scales:
            out.append(InterventionConfig(
                intervention_id=f"randmath{k:02d}_s{s:.3f}",
                family="random_math",
                scale=s,
                selector_kind="random_math_subset",
                selector_args={"fraction": base_fraction, "seed": seed_base + k},
                parameter_group=f"random math subset {k}",
            ))
    return out


def nonmath_controls(n_samples: int, base_fraction: float, scales: Iterable[float], seed_base: int = 1000) -> list[InterventionConfig]:
    """Random subsets of NON-math neurons (negative control)."""
    out = []
    for k in range(n_samples):
        for s in scales:
            out.append(InterventionConfig(
                intervention_id=f"nonmath{k:02d}_s{s:.3f}",
                family="nonmath",
                scale=s,
                selector_kind="nonmath_random_subset",
                selector_args={"fraction": base_fraction, "seed": seed_base + k},
                parameter_group=f"random non-math subset {k}",
            ))
    return out


def combo(a: InterventionConfig, b: InterventionConfig) -> InterventionConfig:
    """A combined intervention that simultaneously applies two single configs."""
    return InterventionConfig(
        intervention_id=f"combo[{a.intervention_id}+{b.intervention_id}]",
        family="combo",
        scale=float("nan"),  # combos carry per-component scales
        selector_kind="combo",
        selector_args={"a": a.__dict__, "b": b.__dict__},
        parameter_group=f"{a.parameter_group} + {b.parameter_group}",
    )


# Minimal viable pool from section 17 ----------------------------------------


def minimal_viable_pool(
    num_layers: int,
    layer_window_size: int = 4,
    causality_quantiles: tuple[float, ...] = (0.10, 0.20, 0.50, 1.0),
    n_random_controls: int = 20,
) -> list[InterventionConfig]:
    configs: list[InterventionConfig] = []
    configs += global_sweep([1.02, 1.05, 1.08, 1.10])
    configs += per_layer(num_layers, [1.05])
    configs += layer_window(num_layers, window=layer_window_size, stride=layer_window_size, scales=[1.05])
    configs += causality_buckets(causality_quantiles, [1.05])
    configs += random_math_controls(n_random_controls, base_fraction=0.05, scales=[1.05])
    return configs


def full_pool(
    num_layers: int,
    scales: tuple[float, ...] = (1.02, 1.05, 1.08, 1.10),
) -> list[InterventionConfig]:
    configs: list[InterventionConfig] = []
    configs += global_sweep(scales)
    configs += per_layer(num_layers, scales)
    configs += layer_window(num_layers, window=4, stride=2, scales=scales)
    configs += layer_third(num_layers, scales)
    configs += by_proj_type(scales)
    configs += causality_buckets((0.10, 0.20, 0.50, 1.0), scales)
    configs += random_math_controls(10, base_fraction=0.05, scales=scales)
    configs += nonmath_controls(5, base_fraction=0.05, scales=scales)
    return configs
