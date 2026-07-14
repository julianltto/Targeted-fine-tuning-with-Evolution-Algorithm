"""Figures specified in section 13 of the spec.

Each function returns ``matplotlib.figure.Figure`` and also saves under ``out_dir``.
PCA is from sklearn; UMAP is optional (used when ``umap-learn`` is installed).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .coverage import CoverageCurve
from .io import OverlapData
from .metrics import OverlapMatrices


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=180, bbox_inches="tight")
    return path


def _heatmap(ax, matrix: np.ndarray, labels: list[str], title: str, vmin=None, vmax=None, cmap="viridis"):
    im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title(title)
    return im


def plot_overlap_heatmaps(
    data: OverlapData,
    M: OverlapMatrices,
    out_dir: Path,
    sort_by: str = "family",
) -> list[Path]:
    ids = M.intervention_ids
    if sort_by == "family":
        order = sorted(range(len(ids)), key=lambda i: (data.meta_by_id[ids[i]].family, ids[i]))
    else:
        order = list(range(len(ids)))
    labels = [ids[i] for i in order]

    paths = []
    for matrix, name, vmin, vmax, cmap in [
        (M.rescue_jaccard, "rescue_jaccard.png", 0.0, 1.0, "viridis"),
        (M.hurt_jaccard, "hurt_jaccard.png", 0.0, 1.0, "magma"),
        (M.rescue_lift, "rescue_lift.png", 0.0, None, "RdBu_r"),
        (M.effect_cosine, "effect_cosine.png", -1.0, 1.0, "RdBu_r"),
        (M.conditional_overlap, "conditional_overlap.png", 0.0, 1.0, "viridis"),
        (M.hypergeom_q, "hypergeom_qvalues.png", 0.0, 1.0, "magma"),
    ]:
        reordered = matrix[np.ix_(order, order)]
        fig, ax = plt.subplots(figsize=(max(8, len(ids) * 0.15), max(7, len(ids) * 0.15)))
        im = _heatmap(ax, reordered, labels, name.replace(".png", ""), vmin=vmin, vmax=vmax, cmap=cmap)
        fig.colorbar(im, ax=ax, fraction=0.04)
        paths.append(_save(fig, out_dir, name))
        plt.close(fig)
    return paths


def plot_coverage_curves(
    curves: list[CoverageCurve],
    out_dir: Path,
    lambda_harm: float = 1.0,
    null_band: np.ndarray | None = None,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for c in curves:
        x = range(1, len(c.rescue_union) + 1)
        axes[0].plot(x, c.rescue_union, marker="o", label=c.method)
        axes[1].plot(x, c.hurt_union, marker="o", label=c.method)
        axes[2].plot(x, c.net(lambda_harm), marker="o", label=c.method)
    if null_band is not None and null_band.size > 0:
        lo = np.percentile(null_band, 2.5, axis=0)
        hi = np.percentile(null_band, 97.5, axis=0)
        med = np.percentile(null_band, 50, axis=0)
        x = range(1, len(med) + 1)
        axes[0].fill_between(x, lo, hi, color="grey", alpha=0.2, label="null 95% CI")
        axes[0].plot(x, med, "--", color="grey", label="null median")
    axes[0].set_title("Union rescued |∪ R_I|")
    axes[1].set_title("Union hurt |∪ H_I|")
    axes[2].set_title(f"Net (λ={lambda_harm})")
    for ax in axes:
        ax.set_xlabel("archive size K")
        ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, out_dir, "coverage_curves.png")


def plot_family_bars(family_summary: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(max(7, len(family_summary) * 0.8), 5))
    x = np.arange(len(family_summary))
    width = 0.27
    ax.bar(x - width, family_summary["union_rescue_total"], width=width, label="union rescued")
    ax.bar(x, family_summary["unique_over_global"], width=width, label="unique over global")
    ax.bar(x + width, family_summary["union_hurt_total"], width=width, label="union hurt")
    ax.set_xticks(x)
    ax.set_xticklabels(family_summary.index, rotation=30, ha="right")
    ax.set_ylabel("examples")
    ax.legend()
    fig.tight_layout()
    return _save(fig, out_dir, "family_bars.png")


def plot_effect_embedding(
    data: OverlapData,
    out_dir: Path,
    method: str = "pca",
) -> Path:
    E = data.effect_matrix().astype(np.float32)
    if method == "pca":
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2).fit_transform(E)
    elif method == "umap":
        import umap
        coords = umap.UMAP(n_neighbors=min(15, max(2, len(E) - 1))).fit_transform(E)
    else:
        raise ValueError(method)

    fams = [data.meta_by_id[i].family for i in data.intervention_ids]
    unique_fams = sorted(set(fams))
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7, 6))
    for k, fam in enumerate(unique_fams):
        idx = [i for i, f in enumerate(fams) if f == fam]
        ax.scatter(coords[idx, 0], coords[idx, 1], color=cmap(k % 10), label=fam, s=30, alpha=0.85)
    ax.legend(fontsize=8)
    ax.set_title(f"Intervention effect embedding ({method.upper()})")
    fig.tight_layout()
    return _save(fig, out_dir, f"effect_embedding_{method}.png")


def plot_effect_dendrogram(data: OverlapData, out_dir: Path) -> Path:
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import pdist
    E = data.effect_matrix().astype(np.float32)
    # Cosine distance is undefined for zero-norm rows; drop those interventions
    # (they neither rescue nor hurt anything, so they carry no signal).
    norms = np.linalg.norm(E, axis=1)
    keep = norms > 0
    ids_kept = [iid for iid, k in zip(data.intervention_ids, keep) if k]
    n_dropped = int((~keep).sum())

    fig, ax = plt.subplots(figsize=(max(8, len(ids_kept) * 0.2), 5))
    if len(ids_kept) < 2:
        ax.text(0.5, 0.5,
                f"not enough interventions with non-zero effect "
                f"(kept={len(ids_kept)}, dropped={n_dropped})",
                ha="center", va="center")
    else:
        dist = pdist(E[keep], metric="cosine")
        Z = linkage(dist, method="average")
        dendrogram(Z, labels=ids_kept, leaf_rotation=90, leaf_font_size=6, ax=ax)
        title = "Effect-vector hierarchical clustering (cosine, average linkage)"
        if n_dropped:
            title += f"  [dropped {n_dropped} zero-effect intervention(s)]"
        ax.set_title(title)
    fig.tight_layout()
    return _save(fig, out_dir, "effect_dendrogram.png")


def plot_synergy_matrix(synergy_df: pd.DataFrame, out_dir: Path) -> Path:
    if synergy_df.empty:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "no combo records", ha="center", va="center")
        return _save(fig, out_dir, "synergy_matrix.png")
    pivot = synergy_df.pivot_table(
        index="i", columns="j", values="syn_rescue", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(max(6, pivot.shape[1] * 0.4), max(5, pivot.shape[0] * 0.4)))
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_title("Synergy: new rescues from combo")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    return _save(fig, out_dir, "synergy_matrix.png")
