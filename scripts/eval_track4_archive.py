"""Evaluate Track 4 archive elites and top-k majority vote on GSM8K.

Default split is full GSM8K test because Track 4 training mutations use GSM8K
train; the evolution proxy used dev, so report the split explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import yaml
from transformers import StoppingCriteria, StoppingCriteriaList

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evomath import REPO_ROOT, load_config  # noqa: E402
from evomath.common.ckpt import load_delta  # noqa: E402
from evomath.common.seeding import seed_everything  # noqa: E402
from evomath.harness import data as hdata  # noqa: E402
from evomath.harness import model as hmodel  # noqa: E402
from evomath.harness.fitness import fitness_acc  # noqa: E402
from evomath.harness.model import LoraIndividual  # noqa: E402
from evomath.harness.parse import parse_gold, parse_pred  # noqa: E402

STOP_STRING = "\nQ:"
PARSE_FAIL_LIMIT = 0.15


def load_lora_from_delta(path: str | Path) -> LoraIndividual:
    obj = load_delta(path)
    return LoraIndividual(
        weights={n: (a.float(), b.float()) for n, (a, b) in obj["weights"].items()},
        alpha=float(obj["alpha"]),
        r=int(obj["r"]),
        meta=dict(obj.get("meta", {})),
    )


def load_docs(cfg: dict, split: str) -> list[dict]:
    if split == "train":
        return hdata.load_train(cfg)
    if split == "dev":
        return hdata.load_dev(cfg)
    if split == "test":
        return hdata.load_test(cfg)
    raise ValueError(f"unknown split {split}")


def load_8shot_samples() -> list[dict]:
    task_path = REPO / "lm_eval/tasks/gsm8k/gsm8k-cot.yaml"
    task = yaml.safe_load(task_path.read_text())
    samples = task["fewshot_config"]["samples"]
    if len(samples) < 8:
        raise RuntimeError(f"expected at least 8 few-shot samples in {task_path}")
    return samples[:8]


def build_8shot_prompt(question: str, shots: list[dict]) -> str:
    prefix = "".join(f"Q: {s['question']}\nA: {s['target']}\n\n" for s in shots)
    return f"{prefix}Q: {question}\nA:"


class StopOnQ(StoppingCriteria):
    def __init__(self, tokenizer, prompt_lens: list[int]):
        self.tokenizer = tokenizer
        self.prompt_lens = prompt_lens
        stop_ids = tokenizer(STOP_STRING, add_special_tokens=False)["input_ids"]
        self.stop_ids = torch.tensor(stop_ids, dtype=torch.long)
        self.done = [False] * len(prompt_lens)

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        for i in range(input_ids.shape[0]):
            if self.done[i]:
                continue
            gen = input_ids[i, self.prompt_lens[i]:].detach().cpu()
            if len(gen) >= len(self.stop_ids) and torch.equal(gen[-len(self.stop_ids):], self.stop_ids):
                self.done[i] = True
                continue
            tail = self.tokenizer.decode(gen[-16:], skip_special_tokens=True)
            if STOP_STRING in tail:
                self.done[i] = True
        return all(self.done)


def truncate_at_stop(text: str) -> str:
    return text.split(STOP_STRING, 1)[0] if STOP_STRING in text else text


@torch.no_grad()
def generate_8shot(model, tokenizer, docs: list[dict], shots: list[dict],
                   batch_size: int, max_new_tokens: int) -> list[str]:
    tokenizer.padding_side = "left"
    outs: list[str] = []
    for start in range(0, len(docs), batch_size):
        prompts = [build_8shot_prompt(d["question"], shots)
                   for d in docs[start:start + batch_size]]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        prompt_len = enc["input_ids"].shape[1]
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=StoppingCriteriaList([StopOnQ(tokenizer, [prompt_len] * len(prompts))]),
        )
        for row in gen:
            outs.append(truncate_at_stop(tokenizer.decode(row[prompt_len:], skip_special_tokens=True)))
    return outs


def accuracy_from_outputs(outs: list[str], docs: list[dict]) -> tuple[float, list[bool]]:
    n_fail = sum(parse_pred(o) is None for o in outs)
    if n_fail / len(outs) > PARSE_FAIL_LIMIT:
        raise RuntimeError(f"parse failure rate {n_fail}/{len(outs)} exceeds {PARSE_FAIL_LIMIT:.0%}")
    correct = []
    for out, doc in zip(outs, docs):
        pred = parse_pred(out)
        correct.append(pred is not None and abs(pred - parse_gold(doc["gold_text"])) <= 1e-4)
    return sum(correct) / len(correct), correct


def eval_accuracy(model, tok, docs: list[dict], cfg: dict, batch_size: int,
                  max_new_tokens: int,
                  prompt_shots: int, shots8: list[dict] | None = None):
    if prompt_shots == 4:
        return fitness_acc(
            model, tok, docs, batch_size=batch_size,
            max_new_tokens=max_new_tokens, return_outputs=True)
    if prompt_shots == 8:
        outs = generate_8shot(
            model, tok, docs, shots8 or load_8shot_samples(),
            batch_size=batch_size, max_new_tokens=max_new_tokens)
        acc, correct = accuracy_from_outputs(outs, docs)
        return acc, outs, correct
    raise ValueError(f"unsupported prompt_shots={prompt_shots}")


def vote_accuracy(outputs_by_name: dict[str, list[str]], docs: list[dict],
                  fallback_name: str) -> dict:
    names = list(outputs_by_name)
    fallback = outputs_by_name[fallback_name]
    correct = []
    parse_fail = 0
    ties = 0
    voted_values = []
    for i, doc in enumerate(docs):
        values = [parse_pred(outputs_by_name[name][i]) for name in names]
        valid = [v for v in values if v is not None]
        if not valid:
            parse_fail += 1
            vote = parse_pred(fallback[i])
        else:
            # Round only for vote-key stability; correctness still uses tolerance.
            counts = Counter(round(v, 4) for v in valid)
            top = counts.most_common()
            if len(top) > 1 and top[0][1] == top[1][1]:
                ties += 1
                vote = parse_pred(fallback[i])
            else:
                vote = float(top[0][0])
        voted_values.append(vote)
        gold = parse_gold(doc["gold_text"])
        correct.append(vote is not None and abs(vote - gold) <= 1e-4)
    return {
        "acc": sum(correct) / len(correct),
        "n": len(correct),
        "parse_fail": parse_fail,
        "ties": ties,
        "voted_values": voted_values,
        "correct": correct,
    }


def disagreement_rate(outputs_by_name: dict[str, list[str]]) -> float:
    names = list(outputs_by_name)
    if len(names) < 2:
        return 0.0
    n = len(next(iter(outputs_by_name.values())))
    disagree = 0
    total = 0
    for i in range(n):
        vals = [parse_pred(outputs_by_name[name][i]) for name in names]
        for a in range(len(vals)):
            for b in range(a + 1, len(vals)):
                if vals[a] is None or vals[b] is None:
                    continue
                total += 1
                disagree += round(vals[a], 4) != round(vals[b], 4)
    return disagree / total if total else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/full_lowmem.yaml")
    p.add_argument("--archive", default="results/evomath/track4_map_elites_10h/s0/archive.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--selection-json", default=None,
                   help="optional list of explicit elite records with delta_path/cell/fitness")
    p.add_argument("--selection-start", type=int, default=0,
                   help="zero-based start within --selection-json (for resumable chunks)")
    p.add_argument("--selection-count", type=int, default=None,
                   help="optional number of selection records to evaluate")
    p.add_argument("--eval-split", choices=["train", "dev", "test"], default="test")
    p.add_argument("--prompt-shots", type=int, choices=[4, 8], default=4)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--include-base", action="store_true")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    seed_everything(args.seed)
    docs = load_docs(cfg, args.eval_split)
    if args.limit is not None:
        docs = docs[:args.limit]
    batch_size = args.batch_size or cfg["gen_batch_size"]
    max_new_tokens = args.max_new_tokens or cfg["max_new_tokens"]

    archive_path = Path(args.archive)
    if not archive_path.is_absolute():
        archive_path = REPO_ROOT / archive_path
    archive = json.loads(archive_path.read_text())
    if args.selection_json:
        sel_path = Path(args.selection_json)
        if not sel_path.is_absolute():
            sel_path = REPO_ROOT / sel_path
        raw_elites = json.loads(sel_path.read_text())
        stop = None if args.selection_count is None else args.selection_start + args.selection_count
        raw_elites = raw_elites[args.selection_start:stop]
        elites = []
        for i, rec in enumerate(raw_elites, 1):
            elites.append({
                "cell": rec.get("cell"),
                "fitness": rec.get("fitness"),
                "meta": {"delta_path": rec["delta_path"]},
                "selection_name": rec.get("name", f"selected_{i}"),
            })
    else:
        elites = sorted(archive["elites"], key=lambda e: e["fitness"], reverse=True)[:args.top_k]
    if not elites:
        raise RuntimeError(f"no elites in {archive_path}")

    out_dir = Path(args.out_dir) if args.out_dir else archive_path.parent / (
        f"eval_{args.eval_split}_{args.prompt_shots}shot_top{args.top_k}")
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "rows.jsonl"
    csv_path = out_dir / "summary.csv"
    outputs_path = out_dir / "outputs.json"
    summary_path = out_dir / "summary.json"

    model, tok = hmodel.load_base(cfg)
    shots8 = load_8shot_samples() if args.prompt_shots == 8 else None
    rows = []
    outputs_by_name: dict[str, list[str]] = {}

    def write_row(row: dict):
        rows.append(row)
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    if args.include_base:
        t0 = time.time()
        acc, outs, correct = eval_accuracy(
            model, tok, docs, cfg, batch_size, max_new_tokens, args.prompt_shots, shots8)
        name = "base"
        outputs_by_name[name] = outs
        write_row({
            "name": name,
            "kind": "base",
            "cell": None,
            "fitness": None,
            "acc": acc,
            "n": len(docs),
            "parse_fail": sum(parse_pred(o) is None for o in outs),
            "seconds": time.time() - t0,
        })

    for rank, elite in enumerate(elites, 1):
        t0 = time.time()
        delta_path = elite["meta"]["delta_path"]
        ind = load_lora_from_delta(delta_path)
        handles = hmodel.patch_lora(model, ind)
        try:
            acc, outs, correct = eval_accuracy(
                model, tok, docs, cfg, batch_size, max_new_tokens, args.prompt_shots, shots8)
        finally:
            hmodel.unpatch_lora(handles)
        if "selection_name" in elite:
            name = f"elite_{rank}_{elite['selection_name']}"
        else:
            name = f"elite_{rank}_cell_{elite['cell'][0]}_{elite['cell'][1]}"
        outputs_by_name[name] = outs
        write_row({
            "name": name,
            "kind": "elite",
            "rank": rank,
            "cell": elite["cell"],
            "fitness": elite["fitness"],
            "delta_path": delta_path,
            "acc": acc,
            "n": len(docs),
            "parse_fail": sum(parse_pred(o) is None for o in outs),
            "seconds": time.time() - t0,
        })

    fallback = next(name for name in outputs_by_name if name.startswith("elite_"))
    vote = vote_accuracy(
        {k: v for k, v in outputs_by_name.items() if k.startswith("elite_")},
        docs,
        fallback_name=fallback,
    )
    vote_row = {
        "name": f"{'selected' if args.selection_json else 'top'}{len(elites) if args.selection_json else args.top_k}_majority_vote",
        "kind": "vote",
        "acc": vote["acc"],
        "n": vote["n"],
        "parse_fail": vote["parse_fail"],
        "ties": vote["ties"],
        "disagreement_rate": disagreement_rate(
            {k: v for k, v in outputs_by_name.items() if k.startswith("elite_")}
        ),
    }
    write_row(vote_row)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = sorted({k for row in rows for k in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    outputs_path.write_text(json.dumps({
        "eval_split": args.eval_split,
        "prompt_shots": args.prompt_shots,
        "max_new_tokens": max_new_tokens,
        "n_docs": len(docs),
        "outputs_by_name": outputs_by_name,
    }))
    summary_path.write_text(json.dumps({
        "archive": str(archive_path.relative_to(REPO_ROOT)),
        "eval_split": args.eval_split,
        "prompt_shots": args.prompt_shots,
        "max_new_tokens": max_new_tokens,
        "n_docs": len(docs),
        "rows": rows,
    }, indent=1))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
