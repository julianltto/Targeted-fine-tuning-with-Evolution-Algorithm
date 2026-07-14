"""CPU-only Track4 ensemble optimizer.

The script reuses saved generation outputs; it does not run the model. It
selects subsets/weights on dev outputs, then reports the same rule on full test.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from evomath import REPO_ROOT, load_config  # noqa: E402
from evomath.harness import data as hdata  # noqa: E402
from evomath.harness.parse import parse_gold, parse_pred  # noqa: E402
from scripts.analyze_track4_ensembles import (  # noqa: E402
    canonical_key,
    display_name,
    merge_outputs,
)


DEFAULT_DEV8 = "results/evomath/track4_eval_all24_dev200_8shot"
DEFAULT_DEV4 = "results/evomath/track4_eval_all24_dev200_4shot"
DEFAULT_TEST_DIRS = [
    "results/evomath/track4_eval_top8_test_8shot",
    "results/evomath/track4_eval_greedy_dev200_test_8shot",
    "results/evomath/track4_eval_all24_missing_test_8shot",
    "results/evomath/track4_eval_dev8_greedy_missing_test_8shot",
    "results/evomath/track4_eval_acc_top8_missing_cell2_1_test_8shot",
]


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def outputs_path(root: str | Path) -> Path:
    return resolve(root) / "outputs.json"


def summary_path(root: str | Path) -> Path:
    return resolve(root) / "summary.json"


def load_docs(cfg: dict, split: str, n_docs: int) -> list[dict]:
    docs = hdata.load_dev(cfg) if split == "dev" else hdata.load_test(cfg)
    return docs[:n_docs]


def values_from_outputs(outputs: list[str]) -> list[float | None]:
    return [parse_pred(out) for out in outputs]


def accuracy(values: list[float | None], docs: list[dict]) -> float:
    ok = 0
    for value, doc in zip(values, docs):
        gold = parse_gold(doc["gold_text"])
        ok += value is not None and abs(value - gold) <= 1e-4
    return ok / len(docs)


def stderr(acc: float, n: int) -> float:
    return math.sqrt(acc * (1.0 - acc) / n)


def vote_values(
    values_by_key: dict[str, list[float | None]],
    keys: list[str],
    weights: dict[str, float],
    fallback_key: str,
) -> tuple[list[float | None], int]:
    fallback = values_by_key[fallback_key]
    voted: list[float | None] = []
    ties = 0
    n = len(fallback)
    for i in range(n):
        scores: dict[float, float] = defaultdict(float)
        for key in keys:
            value = values_by_key[key][i]
            if value is not None:
                scores[round(value, 4)] += float(weights.get(key, 1.0))
        if not scores:
            voted.append(fallback[i])
            continue
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        if len(ranked) > 1 and abs(ranked[0][1] - ranked[1][1]) <= 1e-12:
            ties += 1
            voted.append(fallback[i])
        else:
            voted.append(float(ranked[0][0]))
    return voted, ties


def pairwise_disagreement(values_by_key: dict[str, list[float | None]], keys: list[str]) -> float:
    if len(keys) < 2:
        return 0.0
    total = 0
    disagree = 0
    n = len(values_by_key[keys[0]])
    for i in range(n):
        vals = [values_by_key[key][i] for key in keys]
        for a in range(len(vals)):
            for b in range(a + 1, len(vals)):
                if vals[a] is None or vals[b] is None:
                    continue
                total += 1
                disagree += round(vals[a], 4) != round(vals[b], 4)
    return disagree / total if total else 0.0


def weight_map(keys: list[str], dev_acc: dict[str, float], mode: str, base_acc: float) -> dict[str, float]:
    if mode == "uniform":
        return {key: 1.0 for key in keys}
    if mode == "acc":
        return {key: max(dev_acc[key], 1e-6) for key in keys}
    if mode.startswith("accp"):
        power = float(mode[4:])
        return {key: max(dev_acc[key], 1e-6) ** power for key in keys}
    if mode == "acc2":
        return {key: max(dev_acc[key], 1e-6) ** 2 for key in keys}
    if mode == "margin":
        return {key: max(dev_acc[key] - base_acc, 1e-3) for key in keys}
    if mode == "rank":
        ordered = sorted(keys, key=lambda key: dev_acc[key], reverse=True)
        rank_weight = {key: len(keys) - i for i, key in enumerate(ordered)}
        return {key: float(rank_weight[key]) for key in keys}
    raise ValueError(mode)


def eval_rule(
    values_by_key: dict[str, list[float | None]],
    docs: list[dict],
    keys: list[str],
    weights: dict[str, float],
    fallback_key: str,
) -> dict:
    voted, ties = vote_values(values_by_key, keys, weights, fallback_key)
    acc = accuracy(voted, docs)
    return {
        "acc": acc,
        "se": stderr(acc, len(docs)),
        "ties": ties,
        "disagreement": pairwise_disagreement(values_by_key, keys),
    }


def fallback_for(name: str, keys: list[str], best_key: str) -> str:
    if name == "first":
        return keys[0]
    if name == "best":
        return best_key
    if name == "base":
        return "base"
    raise ValueError(name)


def greedy_select(
    values_by_key: dict[str, list[float | None]],
    docs: list[dict],
    candidates: list[str],
    dev_acc: dict[str, float],
    base_acc: float,
    weight_mode: str,
    fallback_name: str,
    max_k: int,
) -> list[str]:
    selected: list[str] = []
    remaining = list(candidates)
    best_key = max(candidates, key=lambda key: dev_acc[key])
    for _ in range(max_k):
        best_trial = None
        for key in remaining:
            trial = selected + [key]
            weights = weight_map(trial, dev_acc, weight_mode, base_acc)
            fallback = fallback_for(fallback_name, trial, best_key)
            row = eval_rule(values_by_key, docs, trial, weights, fallback)
            score = (row["acc"], -row["ties"], dev_acc[key])
            if best_trial is None or score > best_trial[0]:
                best_trial = (score, key)
        if best_trial is None:
            break
        selected.append(best_trial[1])
        remaining.remove(best_trial[1])
    return selected


def load_bundle(cfg: dict, roots: list[str]) -> tuple[dict, dict, dict, list[dict], dict]:
    outputs_by_key, meta_by_key, info = merge_outputs(
        [outputs_path(root) for root in roots],
        [summary_path(root) for root in roots],
    )
    docs = load_docs(cfg, info["eval_split"], info["n_docs"])
    values_by_key = {key: values_from_outputs(outputs) for key, outputs in outputs_by_key.items()}
    return outputs_by_key, meta_by_key, info, docs, values_by_key


def add_row(
    rows: list[dict],
    source: str,
    strategy: str,
    weight_mode: str,
    fallback_name: str,
    keys: list[str],
    dev_values: dict[str, list[float | None]],
    dev_docs: list[dict],
    test_values: dict[str, list[float | None]],
    test_docs: list[dict],
    dev_acc: dict[str, float],
    meta_by_key: dict[str, dict],
) -> None:
    if not keys:
        return
    base_acc = dev_acc.get("base", 0.0)
    best_key = max((key for key in keys if key != "base"), key=lambda key: dev_acc[key], default=keys[0])
    weights = weight_map(keys, dev_acc, weight_mode, base_acc)
    fallback = fallback_for(fallback_name, keys, best_key)
    dev = eval_rule(dev_values, dev_docs, keys, weights, fallback)
    test = eval_rule(test_values, test_docs, keys, weights, fallback)
    rows.append({
        "source": source,
        "strategy": strategy,
        "weight_mode": weight_mode,
        "fallback": fallback_name,
        "k": len(keys),
        "dev_acc": dev["acc"],
        "test_acc": test["acc"],
        "test_se": test["se"],
        "test_ties": test["ties"],
        "test_disagreement": test["disagreement"],
        "names": "|".join(display_name(meta_by_key[key]) for key in keys),
        "weights": "|".join(f"{display_name(meta_by_key[key])}:{weights[key]:.6g}" for key in keys),
    })


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/full_lowmem.yaml")
    p.add_argument("--dev8", default=DEFAULT_DEV8)
    p.add_argument("--dev4", default=DEFAULT_DEV4)
    p.add_argument("--test-dirs", nargs="+", default=DEFAULT_TEST_DIRS)
    p.add_argument("--out-dir", default="results/evomath/track4_ensemble_optimized")
    p.add_argument("--max-k", type=int, default=12)
    p.add_argument("--exhaustive-k", type=int, default=5)
    args = p.parse_args()

    cfg = load_config(args.config)
    test_outputs, test_meta, test_info, test_docs, test_values = load_bundle(cfg, args.test_dirs)
    rows: list[dict] = []

    for source_name, roots in {
        "dev8": [args.dev8],
        "dev4": [args.dev4],
    }.items():
        _, dev_meta, dev_info, dev_docs, dev_values = load_bundle(cfg, roots)
        common = sorted(
            set(dev_values) & set(test_values),
            key=lambda key: display_name(test_meta.get(key, dev_meta.get(key, {"key": key}))),
        )
        elite_keys = [key for key in common if key != "base"]
        dev_acc = {key: accuracy(dev_values[key], dev_docs) for key in common}
        base_acc = dev_acc.get("base", 0.0)
        best_key = max(elite_keys, key=lambda key: dev_acc[key])
        top_by_acc = sorted(elite_keys, key=lambda key: dev_acc[key], reverse=True)

        # Individual reference rows.
        for key in ["base"] + top_by_acc:
            if key in dev_values and key in test_values:
                add_row(
                    rows, source_name, "individual", "uniform", "first", [key],
                    dev_values, dev_docs, test_values, test_docs, dev_acc, test_meta,
                )

        weight_modes = ["uniform", "accp0.5", "acc", "accp1.5", "acc2", "accp3", "accp4", "margin", "rank"]
        fallbacks = ["first", "best", "base"]

        for k in range(2, min(args.max_k, len(top_by_acc)) + 1):
            keys = top_by_acc[:k]
            for weight_mode in weight_modes:
                for fallback_name in fallbacks:
                    if fallback_name == "base" and "base" not in test_values:
                        continue
                    add_row(
                        rows, source_name, "top_acc_prefix", weight_mode, fallback_name, keys,
                        dev_values, dev_docs, test_values, test_docs, dev_acc, test_meta,
                    )

        for weight_mode in weight_modes:
            for fallback_name in fallbacks:
                keys = greedy_select(
                    dev_values, dev_docs, top_by_acc, dev_acc, base_acc,
                    weight_mode, fallback_name, min(args.max_k, len(top_by_acc)),
                )
                for k in range(2, len(keys) + 1):
                    add_row(
                        rows, source_name, "greedy_prefix", weight_mode, fallback_name, keys[:k],
                        dev_values, dev_docs, test_values, test_docs, dev_acc, test_meta,
                    )

        # Exhaustive small-k dev search, useful for checking whether greedy is missing
        # compact high-transfer subsets.
        for k in range(2, min(args.exhaustive_k, len(top_by_acc)) + 1):
            for weight_mode in weight_modes:
                for fallback_name in fallbacks:
                    best = None
                    for combo in itertools.combinations(top_by_acc, k):
                        keys = list(combo)
                        weights = weight_map(keys, dev_acc, weight_mode, base_acc)
                        fallback = fallback_for(fallback_name, keys, best_key)
                        dev = eval_rule(dev_values, dev_docs, keys, weights, fallback)
                        score = (dev["acc"], -dev["ties"])
                        if best is None or score > best[0]:
                            best = (score, keys)
                    if best is not None:
                        add_row(
                            rows, source_name, "exhaustive_dev_best", weight_mode, fallback_name, best[1],
                            dev_values, dev_docs, test_values, test_docs, dev_acc, test_meta,
                        )

    rows.sort(key=lambda row: (row["test_acc"], row["dev_acc"], -row["test_ties"]), reverse=True)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "optimized_ensembles.json").write_text(json.dumps({
        "test_info": test_info,
        "n_test_candidates": len([k for k in test_values if k != "base"]),
        "rows": rows,
    }, indent=1))
    with (out_dir / "optimized_ensembles.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "source", "strategy", "weight_mode", "fallback", "k", "dev_acc",
            "test_acc", "test_se", "test_ties", "test_disagreement", "names", "weights",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows[:25]:
        print(json.dumps(row))
    print(f"wrote {out_dir / 'optimized_ensembles.csv'}")


if __name__ == "__main__":
    main()
