"""LoRA SFT on rescued trajectories (spec §7).

Three modes (per spec §7.4):
- full LoRA (default)
- layer-mask LoRA: restrict to layers whose MathNeuro density is in top-k
- random-mask LoRA: matched parameter count control
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset


class JsonlSFTDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int = 1024):
        self.records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.records.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        prompt = r["prompt"]
        completion = r["completion"]
        text = prompt + completion
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors="pt")
        prompt_enc = self.tokenizer(prompt, truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = enc["input_ids"][0]
        labels = input_ids.clone()
        # mask prompt tokens
        n_prompt = prompt_enc["input_ids"].shape[1]
        labels[:n_prompt] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": enc["attention_mask"][0],
            "labels": labels,
            "weight": float(r.get("weight", 1.0)),
        }


def _pad_collate(tokenizer, pad_token_id: int):
    def collate(batch):
        max_len = max(b["input_ids"].size(0) for b in batch)
        out_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        out_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        out_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        weights = torch.zeros(len(batch), dtype=torch.float32)
        for i, b in enumerate(batch):
            n = b["input_ids"].size(0)
            out_ids[i, :n] = b["input_ids"]
            out_mask[i, :n] = b["attention_mask"]
            out_labels[i, :n] = b["labels"]
            weights[i] = b["weight"]
        return {"input_ids": out_ids, "attention_mask": out_mask, "labels": out_labels, "weight": weights}
    return collate


def _layer_density(math_important: dict[str, torch.Tensor]) -> dict[int, float]:
    import re
    rng = re.compile(r"\.layers\.(\d+)\.")
    sums: dict[int, list[float]] = {}
    for n, t in math_important.items():
        if "embed" in n:
            continue
        m = rng.search(n)
        if not m:
            continue
        idx = int(m.group(1))
        sums.setdefault(idx, []).append(float(t.float().mean().item()))
    return {k: float(sum(v) / len(v)) for k, v in sums.items()}


def select_target_modules(
    model,
    mode: str,
    top_k_layers: int = 8,
    math_important: dict[str, torch.Tensor] | None = None,
    seed: int = 0,
) -> list[str]:
    import re
    rng = re.compile(r"\.layers\.(\d+)\.")
    # collect all linear module names of interest
    candidates = []
    for n, mod in model.named_modules():
        if any(s in n for s in ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj")):
            if isinstance(mod, torch.nn.Linear):
                candidates.append(n)

    if mode == "full":
        return candidates

    if mode in {"layer_mask", "random_mask"}:
        layers_by_name: dict[int, list[str]] = {}
        for n in candidates:
            m = rng.search(n)
            if m:
                layers_by_name.setdefault(int(m.group(1)), []).append(n)
        if mode == "layer_mask":
            if math_important is None:
                raise ValueError("layer_mask requires math_important to rank layers")
            density = _layer_density(math_important)
            sorted_layers = sorted(density.keys(), key=lambda i: -density[i])[:top_k_layers]
        else:
            import numpy as np
            rng_ = np.random.default_rng(seed)
            sorted_layers = list(rng_.choice(sorted(layers_by_name.keys()), top_k_layers, replace=False))
        return [n for l in sorted_layers for n in layers_by_name.get(int(l), [])]

    raise ValueError(f"unknown target-module mode {mode!r}")


def train(
    model_name: str,
    train_jsonl: Path,
    out_dir: Path,
    target_mode: str = "full",
    masks_pickle: Path | None = None,
    top_k_layers: int = 8,
    epochs: int = 1,
    batch_size: int = 1,
    grad_accum: int = 8,
    lr: float = 2e-4,
    max_length: int = 1024,
    seed: int = 42,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    torch.manual_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda")
    model.gradient_checkpointing_enable()

    math_important = None
    if masks_pickle is not None:
        with open(masks_pickle, "rb") as f:
            math_important = pickle.load(f)["math_important"]

    targets = select_target_modules(model, target_mode, top_k_layers, math_important, seed=seed)
    short_targets = sorted({t.split(".")[-1] for t in targets})
    print(f"[train] target_mode={target_mode}, modules={len(targets)}, types={short_targets}")
    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=short_targets, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds = JsonlSFTDataset(train_jsonl, tokenizer, max_length=max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=_pad_collate(tokenizer, tokenizer.pad_token_id))

    steps_per_epoch = max(1, math.ceil(len(loader) / grad_accum))
    total_steps = steps_per_epoch * epochs
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=int(0.03 * total_steps), num_training_steps=total_steps)

    log_path = out_dir / "train_log.jsonl"
    with open(log_path, "w") as logf:
        global_step = 0
        for ep in range(epochs):
            for i, batch in enumerate(loader):
                batch = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in batch.items()}
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                            labels=batch["labels"])
                w = batch["weight"].to(out.loss.device).mean().clamp(min=1e-6)
                loss = out.loss * w
                (loss / grad_accum).backward()
                if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                    optim.step()
                    sched.step()
                    optim.zero_grad()
                    global_step += 1
                    logf.write(json.dumps({"step": global_step, "epoch": ep,
                                           "loss": float(out.loss.item()), "lr": sched.get_last_lr()[0]}) + "\n")
                    if global_step % 10 == 0:
                        print(f"  [step {global_step}/{total_steps}] loss={out.loss.item():.4f}")
    model.save_pretrained(out_dir / "lora_adapter")
    tokenizer.save_pretrained(out_dir / "lora_adapter")
    print(f"[train] adapter saved → {out_dir / 'lora_adapter'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--train", required=True, help="SFT jsonl produced by datasets.py")
    p.add_argument("--out", required=True)
    p.add_argument("--target-mode", default="full", choices=["full", "layer_mask", "random_mask"])
    p.add_argument("--masks", default=None, help="masks_cache.pkl from overlap experiment (for layer_mask)")
    p.add_argument("--top-k-layers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    train(
        model_name=args.model,
        train_jsonl=Path(args.train),
        out_dir=Path(args.out),
        target_mode=args.target_mode,
        masks_pickle=Path(args.masks) if args.masks else None,
        top_k_layers=args.top_k_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        max_length=args.max_length,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
