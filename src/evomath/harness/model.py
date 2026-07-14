"""Model loading and individual patch-in/patch-out (M0).

Design constraint 3 (CLAUDE.md): exactly one base model on the GPU; an
individual is a parameter *delta*, never a model copy.

- LoRA individual: dict  linear_module_name -> (A [r,in], B [out,r]) CPU tensors,
  plus 'alpha'/'r' metadata. Injection is a forward hook adding
  x @ A^T @ B^T * (alpha/r) to the module output. No weight mutation
  -> patch-out (hook removal) restores the base model bit-exactly
  (in-place bf16 add/sub would leave rounding residue and break
  determinism constraint 1).
- Norm individual: dict  norm_param_name -> delta vector (multiplicative,
  g' = g * exp(delta)). Pristine gain clones are kept and copy_()-restored,
  which is bit-exact.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def load_base(cfg: dict):
    os.environ.setdefault("HF_HOME", cfg["hf_home"])
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["model_path"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_path"], torch_dtype=getattr(torch, cfg.get("dtype", "bfloat16"))
    ).to("cuda")
    model.eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    return model, tok


# --------------------------------------------------------------------------- #
# LoRA individuals
# --------------------------------------------------------------------------- #
@dataclass
class LoraIndividual:
    weights: dict[str, tuple[torch.Tensor, torch.Tensor]]  # name -> (A, B) on CPU
    alpha: float = 32.0
    r: int = 16
    meta: dict = field(default_factory=dict)  # e.g. {'lr': 2e-4, 'shard': 3}

    def named_tensors(self):
        for name in sorted(self.weights):
            a, b = self.weights[name]
            yield f"{name}.A", a
            yield f"{name}.B", b


def lora_target_modules(model) -> dict[str, nn.Linear]:
    return {
        name: mod
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and name.endswith(LORA_TARGETS)
    }


def random_lora_individual(model, sigma: float, generator: torch.Generator,
                           r: int = 16, alpha: float = 32.0) -> LoraIndividual:
    """Gaussian A and B (both nonzero so the delta perturbs immediately)."""
    weights = {}
    for name, mod in sorted(lora_target_modules(model).items()):
        out_f, in_f = mod.weight.shape
        a = torch.randn(r, in_f, generator=generator) * sigma
        b = torch.randn(out_f, r, generator=generator) * sigma
        weights[name] = (a, b)
    return LoraIndividual(weights=weights, alpha=alpha, r=r)


def patch_lora(model, ind: LoraIndividual) -> list:
    """Attach forward hooks adding the LoRA delta. Returns handles for unpatching."""
    handles = []
    modules = dict(model.named_modules())
    scale = ind.alpha / ind.r
    for name, (a_cpu, b_cpu) in ind.weights.items():
        mod = modules[name]
        dev, dt = mod.weight.device, mod.weight.dtype
        a = a_cpu.to(device=dev, dtype=dt)
        b = b_cpu.to(device=dev, dtype=dt)

        def hook(module, inputs, output, A=a, B=b, s=scale):
            x = inputs[0]
            return output + (x @ A.T) @ B.T * s

        handles.append(mod.register_forward_hook(hook))
    return handles


def unpatch_lora(handles: list) -> None:
    for h in handles:
        h.remove()


# --------------------------------------------------------------------------- #
# Norm individuals
# --------------------------------------------------------------------------- #
def norm_param_names(model) -> list[str]:
    """All RMSNorm gain parameter names (input/post-attn per block + final)."""
    return sorted(
        name for name, _ in model.named_parameters()
        if "layernorm" in name or name.endswith("model.norm.weight")
    )


def norm_dim(model) -> int:
    params = dict(model.named_parameters())
    return sum(params[n].numel() for n in norm_param_names(model))


def snapshot_norms(model) -> dict[str, torch.Tensor]:
    params = dict(model.named_parameters())
    return {n: params[n].detach().clone() for n in norm_param_names(model)}


def apply_norm_delta(model, delta: torch.Tensor, pristine: dict[str, torch.Tensor]) -> None:
    """g' = g_base * exp(delta); delta is the flat concatenated vector."""
    params = dict(model.named_parameters())
    off = 0
    with torch.no_grad():
        for name in norm_param_names(model):
            p = params[name]
            d = delta[off:off + p.numel()].to(device=p.device, dtype=torch.float32)
            base = pristine[name].to(torch.float32)
            p.copy_((base * torch.exp(d)).to(p.dtype))
            off += p.numel()
    assert off == delta.numel(), f"delta dim {delta.numel()} != norm dim {off}"


def restore_norms(model, pristine: dict[str, torch.Tensor]) -> None:
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, saved in pristine.items():
            params[name].copy_(saved)
