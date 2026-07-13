"""
Principal Component Analysis (PCA) implemented from scratch using NumPy only:
covariance matrix + eigendecomposition.
"""

from __future__ import annotations

import numpy as np


class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components

        self.components_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self._mean: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PCA":
        X = np.asarray(X, dtype=np.float64)

        if self.n_components > X.shape[1]:
            raise ValueError(
                f"n_components={self.n_components} cannot exceed n_features={X.shape[1]}"
            )

        # center the data (PCA requires zero-mean features)
        self._mean = X.mean(axis=0)
        X_centered = X - self._mean

        # covariance matrix, shape [n_features, n_features]
        # rowvar=False: each column is a variable (feature), each row an observation
        cov_matrix = np.cov(X_centered, rowvar=False)

        # eigendecomposition -- eigh is used (not eig) because the covariance
        # matrix is guaranteed symmetric, which makes eigh both faster and
        # numerically more stable, and guarantees real-valued eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # eigh returns eigenvalues in ASCENDING order -- reverse to descending
        # so the first component captures the most variance
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        # keep only the top n_components
        self.components_ = eigenvectors[:, : self.n_components].T  # shape [n_components, n_features]

        total_variance = eigenvalues.sum()
        if total_variance > 0:
            self.explained_variance_ratio_ = eigenvalues[: self.n_components] / total_variance
        else:
            self.explained_variance_ratio_ = np.zeros(self.n_components)

        # also store ALL eigenvalues' explained variance ratio -- useful for
        # scree plots that show the full cumulative curve, not just the
        # retained n_components
        self._all_explained_variance_ratio = (
            eigenvalues / total_variance if total_variance > 0 else np.zeros_like(eigenvalues)
        )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise RuntimeError("PCA has not been fit yet.")

        X = np.asarray(X, dtype=np.float64)
        X_centered = X - self._mean
        return X_centered @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def full_scree_ratios(self) -> np.ndarray:
        """Explained variance ratio for ALL components (not just the
        retained n_components) -- convenience method for scree plots,
        which typically show the full curve across all features."""
        if self.components_ is None:
            raise RuntimeError("PCA has not been fit yet.")
        return self._all_explained_variance_ratio

    def __repr__(self) -> str:
        fitted = self.components_ is not None
        return f"PCA(n_components={self.n_components}, fitted={fitted})"
