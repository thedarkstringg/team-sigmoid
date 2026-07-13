"""
DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
implemented from scratch using NumPy only.
"""

from __future__ import annotations

import numpy as np


class DBSCAN:
    def __init__(self, eps: float, min_samples: int) -> None:
        self.eps = eps
        self.min_samples = min_samples

        self.labels_: np.ndarray | None = None

    def _region_query(self, X: np.ndarray, point_idx: int, dist_matrix: np.ndarray) -> np.ndarray:
        """Return indices of all points within eps of the given point
        (including the point itself)."""
        return np.where(dist_matrix[point_idx] <= self.eps)[0]

    def fit(self, X: np.ndarray) -> "DBSCAN":
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]

        # pairwise distance matrix via ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        # -- O(n^2) memory for the final distance matrix, NOT O(n^2*d).
        # The naive broadcasting approach (X[:,None,:] - X[None,:,:]) builds
        # a full [n, n, d] intermediate before summing, which is extremely
        # memory-hungry: at n=3000, d=30 that's 2.16GB and can OOM. This
        # approach only ever materializes [n, n] arrays.
        squared_norms = np.sum(X ** 2, axis=1)
        dist_sq = squared_norms[:, np.newaxis] + squared_norms[np.newaxis, :] - 2 * (X @ X.T)
        dist_sq = np.maximum(dist_sq, 0.0)  # clip tiny negative values from floating-point error
        dist_matrix = np.sqrt(dist_sq)

        UNVISITED = -2
        NOISE = -1

        labels = np.full(n_samples, UNVISITED, dtype=int)
        next_cluster_id = 0

        for point_idx in range(n_samples):
            if labels[point_idx] != UNVISITED:
                continue  # already assigned to a cluster or marked noise

            neighbors = self._region_query(X, point_idx, dist_matrix)

            if len(neighbors) < self.min_samples:
                labels[point_idx] = NOISE
                continue

            # point_idx is a core point -- start a new cluster and expand it
            labels[point_idx] = next_cluster_id

            # FIFO queue for breadth-first expansion, paired with a boolean
            # array for O(1) "already queued" checks. The earlier version
            # used `if nn not in seed_set` (Python list membership, O(n) per
            # check) and `list.remove()` (also O(n)) -- pathologically slow
            # when a cluster's seed_set grows large (e.g. a big eps causing
            # most points to be density-connected), turning expansion into
            # O(n^2) pure-Python work. This version is O(n) total per cluster.
            in_seed_set = np.zeros(n_samples, dtype=bool)
            in_seed_set[neighbors] = True
            seed_set = list(neighbors)

            i = 0
            while i < len(seed_set):
                neighbor_idx = seed_set[i]

                if labels[neighbor_idx] == NOISE:
                    # a border point previously marked noise now joins this cluster
                    labels[neighbor_idx] = next_cluster_id

                if labels[neighbor_idx] != UNVISITED:
                    i += 1
                    continue

                labels[neighbor_idx] = next_cluster_id
                neighbor_neighbors = self._region_query(X, neighbor_idx, dist_matrix)

                if len(neighbor_neighbors) >= self.min_samples:
                    # neighbor is ALSO a core point -- expand the cluster
                    # through its neighborhood too (density-reachability)
                    for nn in neighbor_neighbors:
                        if not in_seed_set[nn]:
                            in_seed_set[nn] = True
                            seed_set.append(nn)

                i += 1

            next_cluster_id += 1

        self.labels_ = labels
        self.n_clusters_ = next_cluster_id
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels_

    def noise_fraction(self) -> float:
        """Fraction of points labeled as noise (-1)."""
        if self.labels_ is None:
            raise RuntimeError("DBSCAN has not been fit yet.")
        return float(np.mean(self.labels_ == -1))

    def __repr__(self) -> str:
        fitted = self.labels_ is not None
        n_clusters = getattr(self, "n_clusters_", None)
        return f"DBSCAN(eps={self.eps}, min_samples={self.min_samples}, fitted={fitted}, n_clusters_={n_clusters})"


def k_distance_plot_values(X: np.ndarray, k: int) -> np.ndarray:
    """
    Compute, for each point, the distance to its k-th nearest neighbor.
    Sort ascending -- used to produce the k-distance plot for choosing eps
    (the "knee" of this sorted curve is the recommended eps).

    k is conventionally set to min_samples, per the project brief.
    """
    X = np.asarray(X, dtype=np.float64)
    n_samples = X.shape[0]

    if k >= n_samples:
        raise ValueError(f"k={k} must be less than n_samples={n_samples}")

    # same fix as DBSCAN.fit(): avoid materializing an [n, n, d] intermediate
    squared_norms = np.sum(X ** 2, axis=1)
    dist_sq = squared_norms[:, np.newaxis] + squared_norms[np.newaxis, :] - 2 * (X @ X.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    dist_matrix = np.sqrt(dist_sq)

    # k-th nearest neighbor distance for each point (excluding itself at distance 0)
    sorted_dists = np.sort(dist_matrix, axis=1)
    kth_distances = sorted_dists[:, k]  # index k, since index 0 is the point itself (distance 0)

    return np.sort(kth_distances)