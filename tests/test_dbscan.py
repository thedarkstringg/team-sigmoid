"""
Unit tests for the from-scratch DBSCAN implementation.

Run with:  pytest tests/test_dbscan.py -v
"""

import numpy as np
import pytest
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score

from src.unsupervised.dbscan import DBSCAN, k_distance_plot_values


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def well_separated_blobs():
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=42)
    return X, y


@pytest.fixture
def two_moons():
    return make_moons(n_samples=300, noise=0.05, random_state=42)


# ----------------------------------------------------------------------
# Basic correctness
# ----------------------------------------------------------------------
class TestBasicFit:
    def test_recovers_well_separated_clusters(self, well_separated_blobs):
        X, y_true = well_separated_blobs
        db = DBSCAN(eps=1.0, min_samples=5).fit(X)
        ari = adjusted_rand_score(y_true, db.labels_)
        assert ari > 0.8

    def test_labels_shape(self, well_separated_blobs):
        X, y_true = well_separated_blobs
        db = DBSCAN(eps=1.0, min_samples=5).fit(X)
        assert db.labels_.shape == (X.shape[0],)

    def test_non_convex_clusters_two_moons(self, two_moons):
        """DBSCAN's key advantage over K-Means: density-based clustering
        handles non-convex shapes that centroid-based methods cannot."""
        X, y_true = two_moons
        db = DBSCAN(eps=0.3, min_samples=5).fit(X)
        ari = adjusted_rand_score(y_true, db.labels_)
        assert ari > 0.9


# ----------------------------------------------------------------------
# Agreement with sklearn
# ----------------------------------------------------------------------
class TestSklearnAgreement:
    def test_matches_sklearn_dbscan_labels(self, well_separated_blobs):
        """DBSCAN is deterministic given eps/min_samples (no randomness),
        so results should closely match sklearn's reference implementation."""
        X, y_true = well_separated_blobs
        my_db = DBSCAN(eps=1.0, min_samples=5).fit(X)
        skl_db = SklearnDBSCAN(eps=1.0, min_samples=5).fit(X)

        agreement = adjusted_rand_score(my_db.labels_, skl_db.labels_)
        assert agreement > 0.95

    def test_noise_count_close_to_sklearn(self, well_separated_blobs):
        X, y_true = well_separated_blobs
        my_db = DBSCAN(eps=1.0, min_samples=5).fit(X)
        skl_db = SklearnDBSCAN(eps=1.0, min_samples=5).fit(X)

        my_noise = np.sum(my_db.labels_ == -1)
        skl_noise = np.sum(skl_db.labels_ == -1)
        assert abs(my_noise - skl_noise) <= 2  # allow tiny boundary-point differences


# ----------------------------------------------------------------------
# noise_fraction
# ----------------------------------------------------------------------
class TestNoiseFraction:
    def test_noise_fraction_in_valid_range(self, well_separated_blobs):
        X, y_true = well_separated_blobs
        db = DBSCAN(eps=1.0, min_samples=5).fit(X)
        assert 0.0 <= db.noise_fraction() <= 1.0

    def test_tiny_eps_gives_mostly_noise(self, well_separated_blobs):
        X, y_true = well_separated_blobs
        db = DBSCAN(eps=0.001, min_samples=5).fit(X)
        assert db.noise_fraction() > 0.9

    def test_huge_eps_gives_single_cluster_no_noise(self, well_separated_blobs):
        X, y_true = well_separated_blobs
        db = DBSCAN(eps=1000.0, min_samples=5).fit(X)
        assert db.n_clusters_ == 1
        assert db.noise_fraction() == 0.0

    def test_noise_fraction_raises_before_fit(self):
        db = DBSCAN(eps=1.0, min_samples=5)
        with pytest.raises(RuntimeError):
            db.noise_fraction()


# ----------------------------------------------------------------------
# fit_predict
# ----------------------------------------------------------------------
class TestFitPredict:
    def test_fit_predict_matches_fit_then_labels(self, well_separated_blobs):
        X, y_true = well_separated_blobs
        db = DBSCAN(eps=1.0, min_samples=5)
        labels = db.fit_predict(X)
        assert np.array_equal(labels, db.labels_)


# ----------------------------------------------------------------------
# k_distance_plot_values
# ----------------------------------------------------------------------
class TestKDistancePlot:
    def test_output_length_matches_n_samples(self, well_separated_blobs):
        X, y_true = well_separated_blobs
        k_dists = k_distance_plot_values(X, k=5)
        assert len(k_dists) == X.shape[0]

    def test_output_sorted_ascending(self, well_separated_blobs):
        X, y_true = well_separated_blobs
        k_dists = k_distance_plot_values(X, k=5)
        assert np.all(np.diff(k_dists) >= -1e-9)

    def test_all_distances_non_negative(self, well_separated_blobs):
        X, y_true = well_separated_blobs
        k_dists = k_distance_plot_values(X, k=5)
        assert np.all(k_dists >= 0.0)

    def test_k_too_large_raises(self, well_separated_blobs):
        X, y_true = well_separated_blobs
        with pytest.raises(ValueError):
            k_distance_plot_values(X, k=X.shape[0])  # k must be < n_samples


# ----------------------------------------------------------------------
# Reproducibility (DBSCAN has no randomness, should be perfectly deterministic)
# ----------------------------------------------------------------------
class TestReproducibility:
    def test_identical_results_across_runs(self, well_separated_blobs):
        X, y_true = well_separated_blobs
        db_a = DBSCAN(eps=1.0, min_samples=5).fit(X)
        db_b = DBSCAN(eps=1.0, min_samples=5).fit(X)
        assert np.array_equal(db_a.labels_, db_b.labels_)


# ----------------------------------------------------------------------
# repr
# ----------------------------------------------------------------------
class TestRepr:
    def test_repr_shows_fit_state(self, well_separated_blobs):
        X, y_true = well_separated_blobs
        db_unfit = DBSCAN(eps=1.0, min_samples=5)
        db_fit = DBSCAN(eps=1.0, min_samples=5).fit(X)

        assert "fitted=False" in repr(db_unfit)
        assert "fitted=True" in repr(db_fit)