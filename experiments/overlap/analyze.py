"""End-to-end analysis from already-collected per-sample correctness files.

Usage:

    python -m experiments.overlap.analyze \
        --root results/overlap \
        --K 20 \
        --null-perms 200 \
        --global-id global_s1.050
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .coverage import (
    coverage_curve,
    permutation_null_coverage,
)
from .family import family_overlap_matrix, family_summary
from .io import load_overlap_data
from .metrics import compute_overlap_matrices
from .plots import (
    plot_coverage_curves,
    plot_effect_dendrogram,
    plot_effect_embedding,
    plot_family_bars,
    plot_overlap_heatmaps,
    plot_synergy_matrix,
)
from .synergy import compute_synergy


def _go_no_go_verdict(
    data,
    family_df: pd.DataFrame,
    M,
    K: int,
    global_id: str | None,
) -> dict:
    """Translate spec section 16 thresholds into a coarse Go / No-Go flag."""
    R = data.rescue_matrix()
    sizes = R.sum(axis=1)
    best_single = int(sizes.max()) if sizes.size else 0
    curve = coverage_curve(data, "greedy", K=K)
    union_at_K = curve.rescue_union[-1] if curve.rescue_union else 0

    if global_id is not None and global_id in data.intervention_ids:
        gi = data.intervention_ids.index(global_id)
        global_rescue = int(sizes[gi])
        unique_over_global = int(np.logical_and(
            np.any(R, axis=0), ~R[gi]
        ).sum())
        unique_ratio = unique_over_global / max(global_rescue, 1)
    else:
        unique_over_global = -1
        unique_ratio = -1.0

    # median Jaccard among the top-rescue half
    top_idx = np.argsort(-sizes)[: max(2, len(sizes) // 2)]
    J = M.rescue_jaccard
    sub = J[np.ix_(top_idx, top_idx)]
    iu = np.triu_indices(sub.shape[0], k=1)
    median_J = float(np.median(sub[iu])) if iu[0].size else float("nan")

    cond_union_2x = union_at_K >= 2 * best_single
    cond_unique = unique_ratio >= 0.3
    cond_jaccard = median_J <= 0.3

    verdict = "GO" if (cond_union_2x and cond_unique and cond_jaccard) else (
        "MAYBE" if (cond_union_2x or cond_unique or cond_jaccard) else "NO-GO"
    )
    return {
        "best_single_rescue": best_single,
        "greedy_union_rescue_at_K": int(union_at_K),
        "K": K,
        "union_geq_2x_best_single": bool(cond_union_2x),
        "unique_over_global": unique_over_global,
        "unique_over_global_ratio": float(unique_ratio),
        "unique_ratio_geq_0.3": bool(cond_unique),
        "median_rescue_jaccard_top_half": median_J,
        "median_jaccard_leq_0.3": bool(cond_jaccard),
        "verdict": verdict,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--lambda-harm", type=float, default=1.0)
    p.add_argument("--null-perms", type=int, default=0)
    p.add_argument("--global-id", default=None,
                   help="intervention_id of the global-scale reference, e.g. global_s1.050")
    p.add_argument("--global-family", default="global")
    p.add_argument("--embedding", choices=["pca", "umap"], default="pca")
    args = p.parse_args()

    root = Path(args.root)
    fig_dir = root / "figures"
    table_dir = root / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    data = load_overlap_data(root)
    print(f"[overlap] loaded {data.I} interventions × {data.N} samples; "
          f"baseline correct={int(data.baseline.sum())}, wrong={int((~data.baseline).sum())}")

    # ------------------------------------------------------------------
    # Pairwise metrics
    # ------------------------------------------------------------------
    M = compute_overlap_matrices(data)
    M.to_long_dataframe().to_csv(table_dir / "pairwise_metrics.csv", index=False)
    np.save(table_dir / "rescue_jaccard.npy", M.rescue_jaccard)
    np.save(table_dir / "rescue_lift.npy", M.rescue_lift)
    plot_overlap_heatmaps(data, M, fig_dir)

    # ------------------------------------------------------------------
    # Per-intervention summary
    # ------------------------------------------------------------------
    R = data.rescue_matrix()
    H = data.hurt_matrix()
    per_iv = pd.DataFrame({
        "intervention_id": data.intervention_ids,
        "family": [data.meta_by_id[i].family for i in data.intervention_ids],
        "scale": [data.meta_by_id[i].scale for i in data.intervention_ids],
        "accuracy": data.correct.mean(axis=1),
        "rescue": R.sum(axis=1),
        "hurt": H.sum(axis=1),
    })
    per_iv["net"] = per_iv["rescue"] - args.lambda_harm * per_iv["hurt"]
    per_iv.sort_values("net", ascending=False).to_csv(table_dir / "per_intervention.csv", index=False)

    # ------------------------------------------------------------------
    # Coverage curves
    # ------------------------------------------------------------------
    curves = [
        coverage_curve(data, m, K=args.K, lambda_harm=args.lambda_harm, seed=s)
        for m, s in [("topk_accuracy", 0), ("topk_rescue", 0),
                     ("greedy", 0), ("random", 0), ("diversity", 0)]
    ]
    null_band = None
    if args.null_perms > 0:
        null_band = permutation_null_coverage(data, K=args.K, n_perms=args.null_perms)
        np.save(table_dir / "permutation_null_coverage.npy", null_band)
    plot_coverage_curves(curves, fig_dir, lambda_harm=args.lambda_harm, null_band=null_band)
    pd.DataFrame([{
        "method": c.method,
        **{f"K{k + 1}_rescue": v for k, v in enumerate(c.rescue_union)},
    } for c in curves]).to_csv(table_dir / "coverage_curves.csv", index=False)

    # ------------------------------------------------------------------
    # Family-level analysis
    # ------------------------------------------------------------------
    fam_df = family_summary(data, K=args.K, lambda_harm=args.lambda_harm,
                            global_family=args.global_family)
    fam_df.to_csv(table_dir / "family_summary.csv")
    family_overlap_matrix(data).to_csv(table_dir / "family_overlap.csv")
    plot_family_bars(fam_df, fig_dir)

    # ------------------------------------------------------------------
    # Embedding + dendrogram
    # ------------------------------------------------------------------
    plot_effect_embedding(data, fig_dir, method=args.embedding)
    plot_effect_dendrogram(data, fig_dir)

    # ------------------------------------------------------------------
    # Synergy (only when combos are present in metadata)
    # ------------------------------------------------------------------
    syn_df = compute_synergy(data, lambda_harm=args.lambda_harm)
    if not syn_df.empty:
        syn_df.to_csv(table_dir / "synergy.csv", index=False)
        plot_synergy_matrix(syn_df, fig_dir)

    # ------------------------------------------------------------------
    # Go / No-Go
    # ------------------------------------------------------------------
    verdict = _go_no_go_verdict(data, fam_df, M, K=args.K, global_id=args.global_id)
    with open(table_dir / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)
    print("[overlap] verdict:", json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
