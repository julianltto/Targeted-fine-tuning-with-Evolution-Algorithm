"""Build distillation training datasets from mined trajectories (spec §5–§6).

Reads a JSONL produced by ``mining.mine_trajectories`` and emits a per-problem
training table that respects:

- one positive per problem (configurable: shortest / majority / first)
- length filter (min/max tokens)
- consensus-bucket variants
- rescue-frequency weighting
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


_ANSWER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _approx_token_count(text: str) -> int:
    # whitespace-tokenization proxy — fine for length filtering, not for cost accounting
    return len(text.split())


def _extract_answer(text: str) -> str:
    """Take the last number-like substring as the extracted final answer."""
    if not text:
        return ""
    matches = _ANSWER_RE.findall(text.replace(",", ""))
    return matches[-1] if matches else ""


def load_rescued_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["sol_length"] = df["intervention_solution"].apply(_approx_token_count)
    df["answer_extracted"] = df["intervention_solution"].apply(_extract_answer)
    return df


def per_problem_trajectory_table(
    rescued_df: pd.DataFrame,
    selection: str = "shortest",
    min_tokens: int = 5,
    max_tokens_pct: float = 95.0,
) -> pd.DataFrame:
    """Aggregate rescued trajectories to one row per problem.

    Adds:
        n_positive          r(x): raw rescue count over the *archive* (pre-filter)
        n_positive_kept     trajectories surviving the length filter
        positive_solution   chosen training target (one per problem)
        answer_agreement    fraction of (kept) positive solutions agreeing on the answer
        rescuing_ids        comma-joined intervention ids over the archive (pre-filter)
    """
    if rescued_df.empty:
        return rescued_df

    # raw r(x) and rescuing ids per problem (before length filter)
    raw_counts = rescued_df.groupby("problem_id").size().rename("n_positive")
    raw_ids = (
        rescued_df.groupby("problem_id")["intervention_id"]
        .apply(lambda s: ",".join(sorted(s.unique())))
        .rename("rescuing_ids")
    )

    # length filter (spec §6.2)
    cap = int(np.percentile(rescued_df["sol_length"], max_tokens_pct))
    eligible = rescued_df[
        (rescued_df["sol_length"] >= min_tokens) & (rescued_df["sol_length"] <= cap)
    ].copy()

    out_rows = []
    for pid, grp in eligible.groupby("problem_id"):
        if grp.empty:
            continue
        if selection == "shortest":
            chosen = grp.loc[grp["sol_length"].idxmin()]
        elif selection == "first":
            chosen = grp.iloc[0]
        elif selection == "majority":
            ans_mode = grp["answer_extracted"].mode()
            mode_val = ans_mode.iloc[0] if not ans_mode.empty else ""
            consistent = grp[grp["answer_extracted"] == mode_val]
            chosen = consistent.loc[consistent["sol_length"].idxmin()] if not consistent.empty else grp.iloc[0]
        else:
            raise ValueError(f"unknown selection={selection!r}")

        agree = (
            (grp["answer_extracted"] == chosen["answer_extracted"]).sum() / len(grp)
            if len(grp) else 0.0
        )
        out_rows.append({
            "problem_id": pid,
            "prompt": chosen["prompt"],
            "gold": chosen["gold"],
            "positive_solution": chosen["intervention_solution"],
            "baseline_negative": chosen["baseline_solution"],
            "n_positive": int(raw_counts.get(pid, 0)),
            "n_positive_kept": int(len(grp)),
            "answer_agreement": float(agree),
            "sol_length": int(chosen["sol_length"]),
            "rescuing_ids": raw_ids.get(pid, ""),
        })
    return pd.DataFrame(out_rows)


def filter_consensus(per_problem_df: pd.DataFrame, k: int) -> pd.DataFrame:
    return per_problem_df[per_problem_df["n_positive"] >= k].copy()


def filter_niche(per_problem_df: pd.DataFrame) -> pd.DataFrame:
    return per_problem_df[per_problem_df["n_positive"] == 1].copy()


def filter_global_only(per_problem_df: pd.DataFrame, prefix: str = "global_") -> pd.DataFrame:
    def only_global(ids: str) -> bool:
        return all(part.startswith(prefix) for part in ids.split(","))
    return per_problem_df[per_problem_df["rescuing_ids"].apply(only_global)].copy()


def rescue_frequency_weights(per_problem_df: pd.DataFrame,
                             mode: str = "sqrt",
                             clip_max: float | None = None) -> pd.Series:
    r = per_problem_df["n_positive"].astype(float)
    if mode == "log":
        w = np.log1p(r)
    elif mode == "sqrt":
        w = np.sqrt(r)
    elif mode == "none":
        w = pd.Series(1.0, index=per_problem_df.index)
    else:
        raise ValueError(f"unknown weighting mode {mode!r}")
    if clip_max is not None:
        w = np.minimum(w, clip_max)
    return w


def build_sft_jsonl(per_problem_df: pd.DataFrame, out_path: Path,
                    weight_mode: str = "none") -> Path:
    weights = rescue_frequency_weights(per_problem_df, mode=weight_mode)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for (_, row), w in zip(per_problem_df.iterrows(), weights):
            f.write(json.dumps({
                "problem_id": row["problem_id"],
                "prompt": row["prompt"],
                "completion": row["positive_solution"],
                "weight": float(w),
                "n_positive": int(row["n_positive"]),
            }) + "\n")
    return out_path


def build_dpo_jsonl(per_problem_df: pd.DataFrame, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for _, row in per_problem_df.iterrows():
            f.write(json.dumps({
                "problem_id": row["problem_id"],
                "prompt": row["prompt"],
                "chosen": row["positive_solution"],
                "rejected": row["baseline_negative"],
            }) + "\n")
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rescued", required=True, help="mining JSONL")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--selection", default="shortest", choices=["shortest", "first", "majority"])
    p.add_argument("--consensus-k", type=int, default=3)
    p.add_argument("--weight-mode", default="none", choices=["none", "sqrt", "log"])
    args = p.parse_args()

    df = load_rescued_jsonl(Path(args.rescued))
    pp = per_problem_trajectory_table(df, selection=args.selection)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pp.to_csv(out_dir / "per_problem.csv", index=False)
    build_sft_jsonl(pp, out_dir / "sft_all.jsonl", weight_mode=args.weight_mode)
    build_sft_jsonl(filter_consensus(pp, args.consensus_k), out_dir / "sft_consensus.jsonl", weight_mode=args.weight_mode)
    build_sft_jsonl(filter_niche(pp), out_dir / "sft_niche.jsonl", weight_mode=args.weight_mode)
    build_sft_jsonl(filter_global_only(pp), out_dir / "sft_global_only.jsonl", weight_mode=args.weight_mode)
    build_dpo_jsonl(pp, out_dir / "dpo_all.jsonl")
    print(f"[datasets] {len(pp)} problems → {out_dir}")


if __name__ == "__main__":
    main()
