"""Phase-1 analysis entry point (no GPU).

Reads ``results/overlap_gsm8k/`` (or any overlap experiment root) and produces:

- per_example_rescue_metadata.csv     spec §4.3
- bucket_summary.csv                  spec §1 / §11.2
- family_composition_by_bucket.csv
- family_diversity_by_bucket.csv
- unique_over_global_by_bucket.csv
- hurt_to_rescue_by_bucket.csv
- archive_safe.json / archive_mining.json   spec §3
- figures/rescue_frequency_histogram.png    spec §11.1
- figures/rescue_bucket_panel.png           spec §11.2 (overlap-only metrics)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .archive import mining_archive, safe_archive
from .plots import plot_bucket_panel, plot_rescue_frequency_histogram
from .rescue_stats import (
    bucket_table,
    family_composition_by_bucket,
    family_diversity_by_bucket,
    hurt_to_rescue_ratio_by_bucket,
    jaccard_to_global,
    load_overlap_root,
    rescue_frequency,
    save_rescue_metadata,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="overlap experiment root, e.g. results/overlap_gsm8k")
    p.add_argument("--out", default=None, help="distillation output dir (default: <root>/distillation)")
    p.add_argument("--global-family-prefix", default="global_",
                   help="ID prefix used to identify global-scaling interventions")
    p.add_argument("--consensus-k", type=int, default=5,
                   help="consensus threshold k for spec §1 (also shown on histogram)")
    p.add_argument("--safe-K", type=int, default=5)
    p.add_argument("--safe-lambda", type=float, default=1.0)
    p.add_argument("--mining-K", type=int, default=10)
    p.add_argument("--mining-lambda", type=float, default=0.1)
    args = p.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out) if args.out else root / "distillation"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    data = load_overlap_root(root)
    print(f"[CID] loaded {data.I} interventions × {data.N} samples; "
          f"baseline correct={int(data.baseline.sum())}, wrong={int((~data.baseline).sum())}")

    # archives
    arc_safe = safe_archive(data, K=args.safe_K, lam=args.safe_lambda)
    arc_mine = mining_archive(data, K=args.mining_K, lam=args.mining_lambda)
    (out_dir / "archive_safe.json").write_text(json.dumps({
        "K": args.safe_K, "lambda": args.safe_lambda,
        "selected": arc_safe.selected,
        "rescue_union": arc_safe.rescue_union,
        "hurt_union": arc_safe.hurt_union,
    }, indent=2))
    (out_dir / "archive_mining.json").write_text(json.dumps({
        "K": args.mining_K, "lambda": args.mining_lambda,
        "selected": arc_mine.selected,
        "rescue_union": arc_mine.rescue_union,
        "hurt_union": arc_mine.hurt_union,
    }, indent=2))
    print(f"[CID] safe archive   ({args.safe_K} @ λ={args.safe_lambda}): rescue {arc_safe.rescue_union[-1] if arc_safe.rescue_union else 0}, hurt {arc_safe.hurt_union[-1] if arc_safe.hurt_union else 0}")
    print(f"[CID] mining archive ({args.mining_K} @ λ={args.mining_lambda}): rescue {arc_mine.rescue_union[-1] if arc_mine.rescue_union else 0}, hurt {arc_mine.hurt_union[-1] if arc_mine.hurt_union else 0}")

    # rescue frequency over the full archive (spec §11.1 — uses *all* viable interventions)
    rdf = rescue_frequency(data)
    save_rescue_metadata(rdf, out_dir / "per_example_rescue_metadata.csv")

    # bucket tables (spec §11.2)
    bt = bucket_table(rdf)
    bt.to_csv(out_dir / "bucket_summary.csv")
    fc = family_composition_by_bucket(rdf, data)
    fc.to_csv(out_dir / "family_composition_by_bucket.csv")
    fd = family_diversity_by_bucket(rdf, data)
    fd.to_csv(out_dir / "family_diversity_by_bucket.csv")

    global_ids = [i for i in data.intervention_ids if i.startswith(args.global_family_prefix)]
    gt = jaccard_to_global(rdf, data, global_ids)
    if not gt.empty:
        gt.to_csv(out_dir / "unique_over_global_by_bucket.csv")
    hr = hurt_to_rescue_ratio_by_bucket(rdf, data)
    hr.to_csv(out_dir / "hurt_to_rescue_by_bucket.csv")

    # figures
    plot_rescue_frequency_histogram(rdf, fig_dir, consensus_threshold=args.consensus_k)
    plot_bucket_panel(bt, fc, fd, gt, hr, fig_dir)

    print("[CID] bucket summary:")
    print(bt.to_string())
    print("[CID] figures written to:", fig_dir)


if __name__ == "__main__":
    main()
