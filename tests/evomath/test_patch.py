"""Patch in/out must be bit-exact (design constraint 1/3). Uses a tiny random
Llama so the test runs on CPU in seconds; the logic is architecture-generic."""

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from evomath.harness import model as hmodel
from evomath.harness.cache import FitnessCache, individual_key


def tiny_llama():
    cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, vocab_size=128)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


def logits_of(model, x):
    with torch.no_grad():
        return model(x).logits.clone()


def test_lora_patch_changes_and_unpatch_restores_bitexact():
    m = tiny_llama()
    x = torch.randint(0, 128, (1, 8))
    base = logits_of(m, x)

    g = torch.Generator().manual_seed(1)
    ind = hmodel.random_lora_individual(m, sigma=0.05, generator=g, r=4)
    assert len(ind.weights) == 2 * 7  # 2 layers x 7 projections

    handles = hmodel.patch_lora(m, ind)
    patched = logits_of(m, x)
    assert not torch.equal(patched, base), "LoRA hook had no effect"

    hmodel.unpatch_lora(handles)
    restored = logits_of(m, x)
    assert torch.equal(restored, base), "unpatch is not bit-exact"


def test_norm_apply_restore_bitexact():
    m = tiny_llama()
    x = torch.randint(0, 128, (1, 8))
    base = logits_of(m, x)
    pristine = hmodel.snapshot_norms(m)
    dim = hmodel.norm_dim(m)
    assert dim == 2 * 2 * 64 + 64  # per-block input+post norms + final norm

    delta = torch.full((dim,), 0.1)
    hmodel.apply_norm_delta(m, delta, pristine)
    assert not torch.equal(logits_of(m, x), base), "norm delta had no effect"

    hmodel.restore_norms(m, pristine)
    assert torch.equal(logits_of(m, x), base), "norm restore is not bit-exact"


def test_norm_zero_delta_is_identity():
    m = tiny_llama()
    pristine = hmodel.snapshot_norms(m)
    before = {n: p.clone() for n, p in m.named_parameters() if "norm" in n.lower()}
    hmodel.apply_norm_delta(m, torch.zeros(hmodel.norm_dim(m)), pristine)
    for n, p in m.named_parameters():
        if n in before:
            assert torch.equal(p, before[n])


def test_cache_key_and_roundtrip(tmp_path):
    g = torch.Generator().manual_seed(2)
    m = tiny_llama()
    ind = hmodel.random_lora_individual(m, sigma=0.05, generator=g, r=4)

    k1 = individual_key(ind.named_tensors(), "loglik", "abc")
    k2 = individual_key(ind.named_tensors(), "loglik", "abc")
    assert k1 == k2, "key not stable"
    assert individual_key(ind.named_tensors(), "acc", "abc") != k1
    assert individual_key(ind.named_tensors(), "loglik", "other") != k1

    cache = FitnessCache(tmp_path / "c.json")
    assert cache.get(k1) is None
    cache.put(k1, 0.5)
    assert cache.get(k1) == 0.5
    # persisted across instances
    assert FitnessCache(tmp_path / "c.json").get(k1) == 0.5
