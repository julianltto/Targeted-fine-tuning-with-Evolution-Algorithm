"""Analyze Track 4 ensemble variants from saved eval outputs.

This is CPU-only. It can merge multiple eval_track4_archive.py output
directories by LoRA delta_path, then recompute majority-vote variants without
rerunning generation.
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
sys.path.insert(0, str(REPO / "src"))

from evomath import REPO_ROOT, load_config  # noqa: E402
from evomath.harness import data as hdata  # noqa: E402
from evomath.harness.parse import parse_gold, parse_pred  # noqa: E402


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_docs(cfg: dict, split: str, n_docs: int) -> list[dict]:
    if split == "dev":
        docs = hdata.load_dev(cfg)
    elif split == "test":
        docs = hdata.load_test(cfg)
    else:
        raise ValueError(split)
    return docs[:n_docs]


def canonical_key(row: dict) -> str:
    if row.get("kind") == "base" or row.get("name") == "base":
        return "base"
    delta_path = row.get("delta_path")
    if delta_path:
        return delta_path
    cell = row.get("cell")
    if cell is not None:
        return f"cell_{cell[0]}_{cell[1]}"
    return row["name"]


def display_name(meta: dict) -> str:
    if meta["key"] == "base":
        return "base"
    cell = meta.get("cell")
    if cell is not None:
        return f"cell_{cell[0]}_{cell[1]}"
    aliases = meta.get("aliases") or [meta["key"]]
    return aliases[0]


def merge_outputs(outputs_paths: list[Path], summary_paths: list[Path]) -> tuple[dict, dict, dict]:
    outputs_by_key: dict[str, list[str]] = {}
    meta_by_key: dict[str, dict] = {}
    n_docs = None
    eval_split = None
    prompt_shots = None

    for outputs_path, summary_path in zip(outputs_paths, summary_paths):
        outputs_obj = json.loads(outputs_path.read_text())
        summary_obj = json.loads(summary_path.read_text())
        if n_docs is None:
            n_docs = outputs_obj["n_docs"]
            eval_split = outputs_obj["eval_split"]
            prompt_shots = outputs_obj["prompt_shots"]
        elif (
            n_docs != outputs_obj["n_docs"]
            or eval_split != outputs_obj["eval_split"]
            or prompt_shots != outputs_obj["prompt_shots"]
        ):
            raise ValueError(
                "all outputs must share eval_split, prompt_shots, and n_docs; "
                f"got {outputs_path}"
            )

        rows_by_name = {row["name"]: row for row in summary_obj["rows"]}
        for name, outs in outputs_obj["outputs_by_name"].items():
            row = rows_by_name.get(name, {"name": name, "kind": "base" if name == "base" else "elite"})
            key = canonical_key(row)
            if key in outputs_by_key and outputs_by_key[key] != outs:
                raise ValueError(f"conflicting outputs for {key}")
            outputs_by_key[key] = outs
            meta = meta_by_key.setdefault(key, {"key": key, "aliases": []})
            meta["aliases"].append(name)
            for field in ("kind", "rank", "cell", "fitness", "delta_path"):
                if field in row and row[field] is not None:
                    meta.setdefault(field, row[field])

    assert n_docs is not None and eval_split is not None and prompt_shots is not None
    return (
        outputs_by_key,
        meta_by_key,
        {"n_docs": n_docs, "eval_split": eval_split, "prompt_shots": prompt_shots},
    )


def output_values(outputs: list[str]) -> list[float | None]:
    return [parse_pred(o) for o in outputs]


def accuracy(values: list[float | None], docs: list[dict]) -> float:
    correct = 0
    for value, doc in zip(values, docs):
        gold = parse_gold(doc["gold_text"])
        correct += value is not None and abs(value - gold) <= 1e-4
    return correct / len(docs)


def std_error(acc: float, n_docs: int) -> float:
    return math.sqrt(acc * (1.0 - acc) / n_docs)


def majority_values(
    values_by_key: dict[str, list[float | None]],
    keys: list[str],
    fallback_key: str,
    weights: dict[str, float] | None = None,
) -> tuple[list[float | None], int]:
    fallback = values_by_key[fallback_key]
    voted = []
    ties = 0
    for i in range(len(fallback)):
        if weights is None:
            vals = [values_by_key[key][i] for key in keys]
            valid = [v for v in vals if v is not None]
            if not valid:
                voted.append(fallback[i])
                continue
            counts = Counter(round(v, 4) for v in valid)
            top = counts.most_common()
            if len(top) > 1 and top[0][1] == top[1][1]:
                ties += 1
                voted.append(fallback[i])
            else:
                voted.append(float(top[0][0]))
        else:
            scores: dict[float, float] = defaultdict(float)
            for key in keys:
                value = values_by_key[key][i]
                if value is not None:
                    scores[round(value, 4)] += weights[key]
            if not scores:
                voted.append(fallback[i])
                continue
            top = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
            if len(top) > 1 and abs(top[0][1] - top[1][1]) <= 1e-12:
                ties += 1
                voted.append(fallback[i])
            else:
                voted.append(float(top[0][0]))
    return voted, ties


def pairwise_disagreement(values_by_key: dict[str, list[float | None]], keys: list[str]) -> float:
    if len(keys) < 2:
        return 0.0
    disagree = 0
    total = 0
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


def load_selection(path: Path, meta_by_key: dict[str, dict]) -> tuple[list[str], list[str]]:
    delta_to_key = {
        meta.get("delta_path"): key
        for key, meta in meta_by_key.items()
        if meta.get("delta_path")
    }
    cell_to_key = {
        tuple(meta.get("cell")): key
        for key, meta in meta_by_key.items()
        if meta.get("cell") is not None
    }
    keys = []
    missing = []
    for rec in json.loads(path.read_text()):
        key = delta_to_key.get(rec.get("delta_path"))
        if key is None and rec.get("cell") is not None:
            key = cell_to_key.get(tuple(rec["cell"]))
        if key is None:
            missing.append(rec.get("name") or rec.get("delta_path") or str(rec.get("cell")))
        else:
            keys.append(key)
    return keys, missing


def load_dev_acc_weights(path: Path, meta_by_key: dict[str, dict]) -> dict[str, float]:
    summary = json.loads(path.read_text())
    delta_to_key = {
        meta.get("delta_path"): key
        for key, meta in meta_by_key.items()
        if meta.get("delta_path")
    }
    cell_to_key = {
        tuple(meta.get("cell")): key
        for key, meta in meta_by_key.items()
        if meta.get("cell") is not None
    }
    weights: dict[str, float] = {}
    for row in summary["rows"]:
        if row.get("kind") == "base" or row.get("name") == "base":
            weights["base"] = float(row["acc"])
            continue
        if row.get("kind") != "elite":
            continue
        key = delta_to_key.get(row.get("delta_path"))
        if key is None and row.get("cell") is not None:
            key = cell_to_key.get(tuple(row["cell"]))
        if key is not None:
            weights[key] = max(float(row["acc"]), 0.0)
    return weights


def add_vote_row(
    rows: list[dict],
    strategy: str,
    keys: list[str],
    values_by_key: dict[str, list[float | None]],
    meta_by_key: dict[str, dict],
    docs: list[dict],
    fallback_key: str | None = None,
    weights: dict[str, float] | None = None,
    missing: list[str] | None = None,
) -> None:
    if not keys:
        rows.append({
            "strategy": strategy,
            "k": 0,
            "acc": "",
            "se": "",
            "ties": "",
            "disagreement": "",
            "names": "",
            "missing": "|".join(missing or []),
        })
        return
    fallback = fallback_key or keys[0]
    voted, ties = majority_values(values_by_key, keys, fallback, weights=weights)
    acc = accuracy(voted, docs)
    rows.append({
        "strategy": strategy,
        "k": len(keys),
        "acc": acc,
        "se": std_error(acc, len(docs)),
        "ties": ties,
        "disagreement": pairwise_disagreement(values_by_key, keys),
        "names": "|".join(display_name(meta_by_key[key]) for key in keys),
        "missing": "|".join(missing or []),
    })


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/full_lowmem.yaml")
    p.add_argument("--outputs", nargs="+", required=True)
    p.add_argument("--summaries", nargs="+", required=True)
    p.add_argument("--selection", action="append", default=[],
                   help="named selection in the form name=path/to/selection.json")
    p.add_argument("--weight-summary", default=None,
                   help="optional summary.json whose individual accuracies define vote weights")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-oracle-k", type=int, default=8)
    args = p.parse_args()

    if len(args.outputs) != len(args.summaries):
        raise ValueError("--outputs and --summaries must have the same length")

    outputs_by_key, meta_by_key, info = merge_outputs(
        [resolve(p) for p in args.outputs],
        [resolve(p) for p in args.summaries],
    )
    cfg = load_config(args.config)
    docs = load_docs(cfg, info["eval_split"], info["n_docs"])
    values_by_key = {key: output_values(outs) for key, outs in outputs_by_key.items()}
    dev_weights = load_dev_acc_weights(resolve(args.weight_summary), meta_by_key) if args.weight_summary else None
    elite_keys = [key for key in outputs_by_key if key != "base"]
    elite_keys = sorted(
        elite_keys,
        key=lambda key: (
            meta_by_key[key].get("rank", 10_000),
            display_name(meta_by_key[key]),
        ),
    )

    rows: list[dict] = []
    for key in ["base"] + elite_keys:
        if key not in values_by_key:
            continue
        acc = accuracy(values_by_key[key], docs)
        rows.append({
            "strategy": "individual",
            "k": 1,
            "acc": acc,
            "se": std_error(acc, len(docs)),
            "ties": 0,
            "disagreement": 0.0,
            "names": display_name(meta_by_key[key]),
            "missing": "",
        })

    if elite_keys:
        add_vote_row(rows, "all_available_majority", elite_keys, values_by_key, meta_by_key, docs)
        if dev_weights is not None and all(key in dev_weights for key in elite_keys):
            add_vote_row(
                rows,
                "all_available_weighted_dev_acc",
                elite_keys,
                values_by_key,
                meta_by_key,
                docs,
                weights=dev_weights,
            )

    selections: dict[str, list[str]] = {}
    for item in args.selection:
        if "=" not in item:
            raise ValueError("--selection must be name=path")
        name, raw_path = item.split("=", 1)
        keys, missing = load_selection(resolve(raw_path), meta_by_key)
        selections[name] = keys
        add_vote_row(rows, name, keys, values_by_key, meta_by_key, docs, missing=missing)
        if dev_weights is not None and not missing and all(key in dev_weights for key in keys):
            add_vote_row(
                rows,
                f"{name}_weighted_dev_acc",
                keys,
                values_by_key,
                meta_by_key,
                docs,
                weights=dev_weights,
            )
        if not missing and len(keys) > 1:
            for i, removed_key in enumerate(keys):
                trial = [key for j, key in enumerate(keys) if j != i]
                add_vote_row(
                    rows,
                    f"{name}_leave_one_out",
                    trial,
                    values_by_key,
                    meta_by_key,
                    docs,
                    missing=[f"removed:{display_name(meta_by_key[removed_key])}"],
                )
            for k in range(1, len(keys) + 1):
                add_vote_row(
                    rows,
                    f"{name}_prefix",
                    keys[:k],
                    values_by_key,
                    meta_by_key,
                    docs,
                )

    if "base" in values_by_key:
        for name, keys in selections.items():
            if keys:
                add_vote_row(
                    rows,
                    f"{name}_plus_base",
                    ["base"] + keys,
                    values_by_key,
                    meta_by_key,
                    docs,
                    fallback_key="base",
                )

    max_oracle_k = min(args.max_oracle_k, len(elite_keys))
    for k in range(1, max_oracle_k + 1):
        best = None
        for combo in itertools.combinations(elite_keys, k):
            voted, ties = majority_values(values_by_key, list(combo), combo[0])
            acc = accuracy(voted, docs)
            row = {
                "strategy": "oracle_subset",
                "k": k,
                "acc": acc,
                "se": std_error(acc, len(docs)),
                "ties": ties,
                "disagreement": pairwise_disagreement(values_by_key, list(combo)),
                "names": "|".join(display_name(meta_by_key[key]) for key in combo),
                "missing": "",
            }
            if best is None or (row["acc"], -row["ties"]) > (best["acc"], -best["ties"]):
                best = row
        if best is not None:
            rows.append(best)

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ensemble_analysis.json").write_text(json.dumps({
        **info,
        "n_candidates": len(elite_keys),
        "rows": rows,
    }, indent=1))
    with (out_dir / "ensemble_analysis.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["strategy", "k", "acc", "se", "ties", "disagreement", "names", "missing"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    best_rows = sorted(
        (row for row in rows if isinstance(row["acc"], float)),
        key=lambda row: (row["acc"], -row["ties"]),
        reverse=True,
    )[:20]
    for row in best_rows:
        print(json.dumps(row))
    print(f"wrote {out_dir / 'ensemble_analysis.json'}")


if __name__ == "__main__":
    main()
