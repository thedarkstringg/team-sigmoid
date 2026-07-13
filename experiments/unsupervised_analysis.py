"""
Experiment 7: Unsupervised analysis.

For each dataset:
  - Scree plot; choose the number of PCs capturing >= 90% cumulative variance.
  - Reduce to that many PCs (clustering happens in this reduced space, not
    just 2D -- more informative than clustering on a 2D projection alone).
  - K-Means: sweep k=2..10, plot the elbow (inertia) curve, report ARI vs k,
    and report a "best k" both by the elbow heuristic and by max ARI
    (these can disagree -- worth discussing when they do, since ARI uses
    ground truth the elbow method doesn't have access to).
  - DBSCAN: k-distance plot (k=min_samples) to pick eps via the knee of that
    curve, then a small eps sweep around the knee to also report the ARI-
    maximizing eps as an alternative "best". min_samples is fixed at
    2 * n_reduced_dims, a common rule-of-thumb default.
  - 2D PCA scatter (using PC1/PC2 -- identical to a separate 2-component
    PCA fit, since PCA components are nested/hierarchical) colored by (a)
    true class, (b) K-Means clusters at best k, (c) DBSCAN clusters at
    best eps -- three panels in one figure.
  - Report ARI and noise fraction; discuss alignment with class boundaries.

SCALING NOTE: PCA/K-Means/DBSCAN are distance-based, unlike the tree
ensembles in earlier experiments -- features ARE scaled here (fit on the
full dataset, since this is unsupervised exploratory analysis with no
train/test split).

DATASET SIZE NOTE: after fixing a DBSCAN memory bug (see
src/unsupervised/dbscan.py), 15,000 rows now runs in ~4.5s, so datasets
are capped at 8,000 rows here mainly for K-Means's elbow sweep (9 k values
x 10 k-means++ inits worth of iterations) rather than DBSCAN itself.

Run standalone:  python experiments/unsupervised_analysis.py
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
from sklearn.metrics import adjusted_rand_score

from experiments.common import get_available_datasets, save_figure, save_results_json, subsample_for_experiments
from src.unsupervised.dbscan import DBSCAN, k_distance_plot_values
from src.unsupervised.kmeans import KMeans
from src.unsupervised.pca import PCA
from src.utils.preprocessing import fit_scaler, scale_data

MAX_SAMPLES = 8000
VARIANCE_THRESHOLD = 0.90
K_RANGE = range(2, 11)
DBSCAN_EPS_SWEEP_MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]  # relative to the k-distance knee


def _find_knee(values: np.ndarray) -> int:
    """
    Simple, standard elbow/knee detector: for a curve (values sorted or in
    natural order), find the point of maximum distance from the straight
    line connecting the first and last points. Works for both the K-Means
    inertia curve (decreasing) and the k-distance curve (increasing).
    Returns the INDEX of the knee point.
    """
    n = len(values)
    if n <= 2:
        return 0

    x = np.arange(n)
    y = np.asarray(values, dtype=np.float64)

    # normalize both axes to [0, 1] so the "distance from line" comparison
    # isn't dominated by whichever axis has a larger numeric range
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)

    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])
    line_vec = p2 - p1
    line_vec_norm = line_vec / (np.linalg.norm(line_vec) + 1e-12)

    distances = []
    for i in range(n):
        p = np.array([x_norm[i], y_norm[i]])
        proj_len = np.dot(p - p1, line_vec_norm)
        proj_point = p1 + proj_len * line_vec_norm
        distances.append(np.linalg.norm(p - proj_point))

    return int(np.argmax(distances))


def run_experiment_7(datasets: dict, random_state: int = 42) -> dict:
    all_results = {}

    for dataset_name, (X, y) in datasets.items():
        print(f"\n{'='*60}\nExperiment 7 -- {dataset_name}\n{'='*60}")

        original_n = X.shape[0]
        X, y = subsample_for_experiments(X, y, max_samples=MAX_SAMPLES, random_state=random_state)
        if X.shape[0] < original_n:
            print(f"NOTE: subsampled to {X.shape[0]} rows (stratified) for compute tractability.")

        scaler = fit_scaler(X)
        X_scaled = scale_data(scaler, X)

        # ---------------- Scree plot + >=90% variance PCs ----------------
        n_features = X_scaled.shape[1]
        pca_full = PCA(n_components=min(n_features, X_scaled.shape[0] - 1)).fit(X_scaled)
        scree = pca_full.full_scree_ratios()
        cumulative = np.cumsum(scree)
        n_components_90 = int(np.searchsorted(cumulative, VARIANCE_THRESHOLD) + 1)
        n_components_90 = max(2, min(n_components_90, len(scree)))  # at least 2, for the 2D scatter

        print(f"PCA: {n_components_90}/{n_features} components capture "
              f">= {VARIANCE_THRESHOLD:.0%} variance "
              f"(actual: {cumulative[n_components_90 - 1]:.4f})")

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(np.arange(1, len(cumulative) + 1), cumulative, marker="o", markersize=3)
        ax.axhline(VARIANCE_THRESHOLD, color="red", linestyle="--", alpha=0.6, label=f"{VARIANCE_THRESHOLD:.0%} threshold")
        ax.axvline(n_components_90, color="gray", linestyle="--", alpha=0.6, label=f"{n_components_90} components")
        ax.set_xlabel("Number of principal components")
        ax.set_ylabel("Cumulative explained variance ratio")
        ax.set_title(f"Scree plot: {dataset_name}")
        ax.legend()
        ax.grid(alpha=0.3)
        save_figure(fig, f"experiment_7_scree_{dataset_name}")
        plt.close(fig)

        # reduce to n_components_90 for clustering; PC1/PC2 (first 2 columns)
        # double as the 2D visualization -- identical to a separate 2-component
        # PCA fit, since PCA components are nested
        pca_reduced = PCA(n_components=n_components_90).fit(X_scaled)
        X_reduced = pca_reduced.transform(X_scaled)
        X_2d = X_reduced[:, :2]

        # ---------------- K-Means: elbow + ARI sweep ----------------
        inertias = []
        aris_by_k = []
        labels_by_k = {}
        for k in K_RANGE:
            km = KMeans(n_clusters=k, random_state=random_state).fit(X_reduced)
            inertias.append(km.inertia_)
            ari = adjusted_rand_score(y, km.labels_)
            aris_by_k.append(ari)
            labels_by_k[k] = km.labels_

        elbow_idx = _find_knee(np.array(inertias))
        best_k_elbow = list(K_RANGE)[elbow_idx]
        best_k_ari = list(K_RANGE)[int(np.argmax(aris_by_k))]

        print(f"K-Means: elbow-selected k={best_k_elbow} (ARI={aris_by_k[elbow_idx]:.4f}), "
              f"ARI-maximizing k={best_k_ari} (ARI={max(aris_by_k):.4f})"
              f"{' -- SAME' if best_k_elbow == best_k_ari else ' -- DIFFER (worth discussing)'}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(list(K_RANGE), inertias, marker="o")
        ax1.axvline(best_k_elbow, color="gray", linestyle="--", alpha=0.6, label=f"elbow k={best_k_elbow}")
        ax1.set_xlabel("k")
        ax1.set_ylabel("Inertia")
        ax1.set_title(f"K-Means elbow: {dataset_name}")
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.plot(list(K_RANGE), aris_by_k, marker="s", color="green")
        ax2.axvline(best_k_ari, color="gray", linestyle="--", alpha=0.6, label=f"best ARI k={best_k_ari}")
        ax2.set_xlabel("k")
        ax2.set_ylabel("ARI vs true labels")
        ax2.set_title(f"K-Means ARI vs k: {dataset_name}")
        ax2.legend()
        ax2.grid(alpha=0.3)
        save_figure(fig, f"experiment_7_kmeans_elbow_ari_{dataset_name}")
        plt.close(fig)

        # ---------------- DBSCAN: k-distance plot + eps sweep ----------------
        min_samples = max(3, 2 * n_components_90)
        min_samples = min(min_samples, X_reduced.shape[0] - 1)
        k_dists = k_distance_plot_values(X_reduced, k=min_samples)
        knee_idx = _find_knee(k_dists)
        eps_knee = float(k_dists[knee_idx])

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(np.arange(len(k_dists)), k_dists)
        ax.axhline(eps_knee, color="red", linestyle="--", alpha=0.6, label=f"knee eps={eps_knee:.3f}")
        ax.set_xlabel("Points, sorted by k-distance")
        ax.set_ylabel(f"{min_samples}-NN distance")
        ax.set_title(f"K-distance plot: {dataset_name}")
        ax.legend()
        ax.grid(alpha=0.3)
        save_figure(fig, f"experiment_7_kdistance_{dataset_name}")
        plt.close(fig)

        eps_candidates = [eps_knee * m for m in DBSCAN_EPS_SWEEP_MULTIPLIERS]
        dbscan_results_by_eps = []
        for eps in eps_candidates:
            if eps <= 0:
                continue
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_reduced)
            ari = adjusted_rand_score(y, db.labels_)
            dbscan_results_by_eps.append({
                "eps": eps, "ari": ari, "noise_fraction": db.noise_fraction(),
                "n_clusters": db.n_clusters_, "labels": db.labels_,
            })

        best_dbscan = max(dbscan_results_by_eps, key=lambda r: r["ari"])
        print(f"DBSCAN: knee-selected eps={eps_knee:.4f}, "
              f"ARI-maximizing eps={best_dbscan['eps']:.4f} (ARI={best_dbscan['ari']:.4f}, "
              f"noise={best_dbscan['noise_fraction']:.4f}, n_clusters={best_dbscan['n_clusters']})")

        # ---------------- 2D PCA scatter, 3 panels ----------------
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        sc0 = axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="tab10", s=10, alpha=0.7)
        axes[0].set_title("True class")
        axes[0].set_xlabel("PC1")
        axes[0].set_ylabel("PC2")

        best_km_labels = labels_by_k[best_k_ari]
        axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=best_km_labels, cmap="tab10", s=10, alpha=0.7)
        axes[1].set_title(f"K-Means (k={best_k_ari}, ARI={max(aris_by_k):.3f})")
        axes[1].set_xlabel("PC1")

        axes[2].scatter(X_2d[:, 0], X_2d[:, 1], c=best_dbscan["labels"], cmap="tab10", s=10, alpha=0.7)
        axes[2].set_title(f"DBSCAN (eps={best_dbscan['eps']:.3f}, ARI={best_dbscan['ari']:.3f})")
        axes[2].set_xlabel("PC1")

        fig.suptitle(f"Unsupervised analysis: {dataset_name}")
        plt.tight_layout()
        save_figure(fig, f"experiment_7_pca_scatter_{dataset_name}")
        plt.close(fig)

        # ---------------- discussion note ----------------
        alignment_note = (
            f"K-Means ARI={max(aris_by_k):.4f} and DBSCAN ARI={best_dbscan['ari']:.4f} against true "
            f"labels indicate {'strong' if max(max(aris_by_k), best_dbscan['ari']) > 0.5 else 'weak'} "
            f"alignment between unsupervised cluster structure and the true class boundaries on "
            f"{dataset_name}. Cross-reference against this dataset's supervised classifier accuracy "
            f"(Experiments 1 and 4): datasets where classifiers achieve high accuracy typically also "
            f"show higher ARI here, since both signals reflect the same underlying class separability."
        )
        print(alignment_note)

        all_results[dataset_name] = {
            "n_components_90pct_variance": n_components_90,
            "cumulative_variance_at_90": float(cumulative[n_components_90 - 1]),
            "kmeans": {
                "k_range": list(K_RANGE),
                "inertias": inertias,
                "ari_by_k": aris_by_k,
                "best_k_elbow": best_k_elbow,
                "best_k_ari": best_k_ari,
            },
            "dbscan": {
                "min_samples": min_samples,
                "eps_knee": eps_knee,
                "eps_candidates": eps_candidates,
                "best_eps": best_dbscan["eps"],
                "best_ari": best_dbscan["ari"],
                "best_noise_fraction": best_dbscan["noise_fraction"],
                "best_n_clusters": best_dbscan["n_clusters"],
            },
            "discussion_note": alignment_note,
        }

    save_results_json("experiment_7_unsupervised_analysis", all_results)
    return all_results


if __name__ == "__main__":
    datasets = get_available_datasets()
    run_experiment_7(datasets)
