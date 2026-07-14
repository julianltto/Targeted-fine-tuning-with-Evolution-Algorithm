"""Figures 11.1 and 11.2 from the CID spec.

11.1 — rescue frequency histogram over baseline-wrong problems
11.2 — per-bucket comparison: bucket size, family composition, family diversity,
       uniqueness over global, hurt/rescue ratio of contributing interventions
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_rescue_frequency_histogram(
    rescue_df: pd.DataFrame,
    out_dir: Path,
    consensus_threshold: int | None = None,
    bin_step: int = 1,
) -> Path:
    """§11.1: histogram of r(x) over baseline-wrong examples."""
    r = rescue_df["r"].to_numpy()
    max_r = int(r.max()) if r.size else 0
    n_consensus = int((r >= (consensus_threshold or 5)).sum())
    n_niche = int((r == 1).sum())
    n_never = int((r == 0).sum())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    bins = np.arange(-0.5, max_r + 1.5, bin_step)

    # left: full histogram with r=0 spike (linear scale)
    axes[0].hist(r, bins=bins, color="#4878D0", edgecolor="white")
    axes[0].set_xlabel("rescue count r(x)")
    axes[0].set_ylabel("number of baseline-wrong examples")
    axes[0].set_title(f"All r(x) (linear)  N={len(r)}")
    axes[0].set_xticks(np.arange(0, max_r + 1, max(1, (max_r + 1) // 12)))

    # right: zoom on r>=1, log y to show tail
    if (r >= 1).any():
        axes[1].hist(r[r >= 1], bins=np.arange(0.5, max_r + 1.5, bin_step),
                     color="#EE854A", edgecolor="white")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("rescue count r(x)")
        axes[1].set_ylabel("count (log)")
        axes[1].set_title(f"r(x) ≥ 1 only (log y) — never={n_never}, niche={n_niche}, consensus(≥{consensus_threshold or 5})={n_consensus}")
        if consensus_threshold is not None and consensus_threshold > 0:
            axes[1].axvline(consensus_threshold - 0.5, linestyle="--", color="grey",
                            label=f"consensus k={consensus_threshold}")
            axes[1].legend(loc="upper right", fontsize=9)
        axes[1].set_xticks(np.arange(1, max_r + 1, max(1, max_r // 12)))

    fig.suptitle("§11.1 Rescue frequency histogram", fontsize=13, y=1.0)
    fig.tight_layout()
    return _save(fig, out_dir, "rescue_frequency_histogram.png")


def plot_bucket_panel(
    bucket_table: pd.DataFrame,
    family_comp: pd.DataFrame,
    family_div: pd.DataFrame,
    global_table: pd.DataFrame,
    hurt_ratio: pd.DataFrame,
    out_dir: Path,
) -> Path:
    """§11.2: per-bucket comparison across the metrics derivable from overlap data.

    Layout:
        +-----------------------+-----------------------+
        | bucket sizes          | family composition    |
        +-----------------------+-----------------------+
        | family diversity      | unique over global    |
        +-----------------------+-----------------------+
        | hurt/rescue ratio (full row)                  |
        +-----------------------------------------------+
    """
    fig = plt.figure(figsize=(13, 11))
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.32)

    # (1) bucket sizes
    ax = fig.add_subplot(gs[0, 0])
    bucket_table["n_examples"].plot(kind="bar", ax=ax, color="#4878D0")
    ax.set_title("Bucket sizes (baseline-wrong examples)")
    ax.set_ylabel("n examples")
    ax.set_xlabel("")
    for i, v in enumerate(bucket_table["n_examples"]):
        ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    # (2) family composition (stacked or grouped)
    ax = fig.add_subplot(gs[0, 1])
    if not family_comp.empty:
        family_comp_pct = family_comp.div(family_comp.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
        family_comp_pct.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
        ax.set_ylabel("% of rescued examples in bucket")
        ax.set_title("Which families contribute rescues per bucket")
        ax.legend(fontsize=7, ncol=2, loc="upper right")
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    else:
        ax.text(0.5, 0.5, "no family composition", ha="center", va="center")

    # (3) family diversity per example
    ax = fig.add_subplot(gs[1, 0])
    if not family_div.empty:
        x = np.arange(len(family_div))
        ax.bar(x - 0.2, family_div["mean"], width=0.4, label="mean", color="#4878D0")
        ax.bar(x + 0.2, family_div["median"], width=0.4, label="median", color="#EE854A")
        ax.set_xticks(x)
        ax.set_xticklabels(family_div.index, rotation=15, ha="right")
        ax.set_ylabel("distinct intervention families")
        ax.set_title("Per-example family diversity")
        ax.legend(fontsize=8)

    # (4) unique over global
    ax = fig.add_subplot(gs[1, 1])
    if not global_table.empty:
        x = np.arange(len(global_table))
        ax.bar(x - 0.2, global_table["n_rescued_by_global"], width=0.4,
               label="also by global", color="#4878D0")
        ax.bar(x + 0.2, global_table["unique_over_global"], width=0.4,
               label="unique over global", color="#EE854A")
        ax.set_xticks(x)
        ax.set_xticklabels(global_table.index, rotation=15, ha="right")
        ax.set_ylabel("n examples")
        ax.set_title("Rescue overlap with global archive")
        ax.legend(fontsize=8)

    # (5) hurt/rescue ratio
    ax = fig.add_subplot(gs[2, :])
    if not hurt_ratio.empty and hurt_ratio["mean"].notna().any():
        x = np.arange(len(hurt_ratio))
        ax.bar(x - 0.2, hurt_ratio["mean"], width=0.4, label="mean H/R of contributing interventions", color="#4878D0")
        ax.bar(x + 0.2, hurt_ratio["median"], width=0.4, label="median H/R", color="#EE854A")
        ax.set_xticks(x)
        ax.set_xticklabels(hurt_ratio.index, rotation=15, ha="right")
        ax.set_ylabel("|H_I| / |R_I|  for the interventions that rescued this example")
        ax.set_title("Safety proxy: how hurt-prone are the rescuing interventions?  (lower = safer)")
        ax.axhline(1.0, linestyle="--", color="grey", alpha=0.6, label="break-even (hurt == rescue)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no hurt/rescue data", ha="center", va="center")

    fig.suptitle("Rescue-bucket comparison (CID §11.2)", fontsize=14, y=0.995)
    return _save(fig, out_dir, "rescue_bucket_panel.png")


def plot_quality_panel_from_trajectories(
    bucket_traj_df: pd.DataFrame,
    out_dir: Path,
) -> Path:
    """§11.2 trajectory-derived axes — solution length, answer agreement, etc.

    Requires output from ``distillation.mining`` where intervention solutions
    have been collected as raw text. Each row of ``bucket_traj_df`` should have:
        problem_id, bucket, sol_length, answer_agreement (0..1),
        n_positive, base_logprob (optional), activation_strength (optional)
    """
    metrics = [
        ("sol_length", "solution length (tokens)"),
        ("answer_agreement", "answer agreement across rescuing interventions"),
        ("n_positive", "number of positive trajectories"),
        ("base_logprob", "baseline mean logprob over rescued solution"),
        ("activation_strength", "math-neuron activation strength"),
    ]
    available = [(c, lbl) for c, lbl in metrics if c in bucket_traj_df.columns and bucket_traj_df[c].notna().any()]
    if not available:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "no trajectory-derived metrics available", ha="center", va="center")
        return _save(fig, out_dir, "rescue_quality_panel.png")

    n = len(available)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.5 * rows), squeeze=False)
    axes = axes.flatten()
    for k, (col, label) in enumerate(available):
        ax = axes[k]
        data_by_bucket = [
            bucket_traj_df.loc[bucket_traj_df["bucket"] == b, col].dropna().to_numpy()
            for b in bucket_traj_df["bucket"].unique()
        ]
        ax.boxplot(data_by_bucket, labels=list(bucket_traj_df["bucket"].unique()),
                   showmeans=True, meanline=True)
        ax.set_title(label)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    for k in range(n, len(axes)):
        axes[k].axis("off")

    fig.suptitle("Per-bucket trajectory quality  (CID §11.2)", fontsize=13, y=1.0)
    fig.tight_layout()
    return _save(fig, out_dir, "rescue_quality_panel.png")
