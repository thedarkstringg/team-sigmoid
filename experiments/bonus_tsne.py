"""
BONUS (+2 pts): t-SNE visualization.

For each dataset, compute a 2D t-SNE embedding (via sklearn.manifold.TSNE
-- explicitly allowed by the brief: "Implement or use sklearn t-SNE") and
compare it side-by-side against the 2D PCA scatter already computed in
Experiment 7, both colored by true class. Explain the differences.

WHY PCA-REDUCE BEFORE T-SNE: t-SNE's pairwise-affinity computation is
expensive in high-dimensional raw input space (e.g. MNIST's 784 raw
pixels). Standard practice -- and what's used here -- is to first reduce
to a moderate number of PCA dimensions (captures most of the variance,
denoises, and speeds up t-SNE substantially) before running t-SNE on that
reduced representation. This does not defeat the comparison's purpose:
the interesting comparison is "linear 2D projection (PCA) vs non-linear
2D projection (t-SNE)", and t-SNE's non-linearity is what's being
evaluated regardless of whether its *input* was pre-reduced.

KEY DIFFERENCES TO DISCUSS IN THE REPORT:
  - PCA is a LINEAR projection: it finds the directions of maximum global
    variance. Distances in the 2D PCA plot approximate true global
    distances (up to the variance lost by dropping components).
  - t-SNE is a NON-LINEAR projection: it optimizes for preserving LOCAL
    neighborhood structure (points close in high-dim stay close in 2D),
    at the cost of not preserving global distances or density -- cluster
    sizes and inter-cluster distances in a t-SNE plot are not meaningful,
    only which points ended up near each other.
  - Consequence: t-SNE often produces visually tighter, more separated-
    looking clusters than PCA, even when the underlying class separability
    is identical -- this makes t-SNE better for a qualitative "does
    structure exist" check, but PCA more honest for any conclusion that
    depends on distance or density (which is why PCA, not t-SNE, is used
    as the actual input to K-Means/DBSCAN in Experiment 7).
  - t-SNE has no inverse_transform and is stochastic across different
    random seeds/perplexity values; PCA is deterministic and invertible.

Run standalone:  python experiments/bonus_tsne.py
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from experiments.common import get_available_datasets, save_figure, save_results_json, subsample_for_experiments
from src.unsupervised.pca import PCA
from src.utils.preprocessing import fit_scaler, scale_data

MAX_SAMPLES = 8000
PCA_DIMS_BEFORE_TSNE = 50  # standard practice: reduce before t-SNE for speed/denoising


def run_tsne_bonus(datasets: dict, random_state: int = 42) -> dict:
    all_results = {}

    for dataset_name, (X, y) in datasets.items():
        print(f"\n{'='*60}\nBONUS: t-SNE vs PCA -- {dataset_name}\n{'='*60}")

        original_n = X.shape[0]
        X, y = subsample_for_experiments(X, y, max_samples=MAX_SAMPLES, random_state=random_state)
        if X.shape[0] < original_n:
            print(f"NOTE: subsampled to {X.shape[0]} rows (stratified) for compute tractability.")

        scaler = fit_scaler(X)
        X_scaled = scale_data(scaler, X)

        # --- PCA 2D embedding (from-scratch implementation) ---
        n_pca_dims = min(PCA_DIMS_BEFORE_TSNE, X_scaled.shape[1], X_scaled.shape[0] - 1)
        pca = PCA(n_components=n_pca_dims).fit(X_scaled)
        X_pca_reduced = pca.transform(X_scaled)
        X_pca_2d = X_pca_reduced[:, :2]  # first 2 components double as the 2D view

        # --- t-SNE 2D embedding (sklearn, per the brief's explicit allowance) ---
        tsne = TSNE(n_components=2, random_state=random_state, init="pca")
        X_tsne_2d = tsne.fit_transform(X_pca_reduced)  # t-SNE on the PCA-reduced input, see docstring

        # --- side-by-side comparison figure ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y, cmap="tab10", s=10, alpha=0.7)
        ax1.set_title("PCA (linear, 2D)")
        ax1.set_xlabel("PC1")
        ax1.set_ylabel("PC2")

        ax2.scatter(X_tsne_2d[:, 0], X_tsne_2d[:, 1], c=y, cmap="tab10", s=10, alpha=0.7)
        ax2.set_title("t-SNE (non-linear, 2D)")
        ax2.set_xlabel("t-SNE dim 1")
        ax2.set_ylabel("t-SNE dim 2")

        fig.suptitle(f"PCA vs t-SNE: {dataset_name}")
        plt.tight_layout()
        save_figure(fig, f"bonus_tsne_vs_pca_{dataset_name}")
        plt.close(fig)

        print(f"Saved side-by-side PCA vs t-SNE comparison for {dataset_name}.")

        all_results[dataset_name] = {
            "pca_2d": X_pca_2d.tolist(),
            "tsne_2d": X_tsne_2d.tolist(),
            "true_labels": y.tolist(),
            "n_pca_dims_before_tsne": n_pca_dims,
        }

    save_results_json("bonus_tsne_vs_pca", all_results)
    return all_results


if __name__ == "__main__":
    datasets = get_available_datasets()
    run_tsne_bonus(datasets)