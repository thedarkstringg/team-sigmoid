"""
Unit tests for the from-scratch PCA implementation.

Run with:  pytest tests/test_pca.py -v
"""

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.decomposition import PCA as SklearnPCA

from src.unsupervised.pca import PCA


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def iris_data():
    return load_iris(return_X_y=True)


@pytest.fixture(scope="module")
def breast_cancer_data():
    return load_breast_cancer(return_X_y=True)


def _sign_aligned_allclose(a: np.ndarray, b: np.ndarray, atol=1e-6) -> bool:
    """Eigenvectors are only defined up to sign; compare allowing a flip."""
    return np.allclose(a, b, atol=atol) or np.allclose(a, -b, atol=atol)


# ----------------------------------------------------------------------
# Basic correctness
# ----------------------------------------------------------------------
class TestBasicFit:
    def test_fit_transform_shape(self, iris_data):
        X, y = iris_data
        pca = PCA(n_components=2).fit(X)
        X_proj = pca.transform(X)
        assert X_proj.shape == (150, 2)

    def test_fit_transform_matches_fit_then_transform(self, iris_data):
        X, y = iris_data
        pca1 = PCA(n_components=2)
        X_ft = pca1.fit_transform(X)

        pca2 = PCA(n_components=2).fit(X)
        X_separate = pca2.transform(X)

        assert np.allclose(X_ft, X_separate)

    def test_components_shape(self, iris_data):
        X, y = iris_data
        pca = PCA(n_components=3).fit(X)
        assert pca.components_.shape == (3, X.shape[1])

    def test_explained_variance_ratio_shape_and_range(self, iris_data):
        X, y = iris_data
        pca = PCA(n_components=2).fit(X)
        assert pca.explained_variance_ratio_.shape == (2,)
        assert np.all(pca.explained_variance_ratio_ >= 0.0)
        assert np.all(pca.explained_variance_ratio_ <= 1.0)


# ----------------------------------------------------------------------
# Agreement with sklearn
# ----------------------------------------------------------------------
class TestSklearnAgreement:
    def test_explained_variance_ratio_matches_sklearn(self, iris_data):
        X, y = iris_data
        my_pca = PCA(n_components=2).fit(X)
        skl_pca = SklearnPCA(n_components=2).fit(X)
        assert np.allclose(my_pca.explained_variance_ratio_, skl_pca.explained_variance_ratio_, atol=1e-6)

    def test_components_match_sklearn_up_to_sign(self, iris_data):
        X, y = iris_data
        my_pca = PCA(n_components=2).fit(X)
        skl_pca = SklearnPCA(n_components=2).fit(X)

        for i in range(2):
            assert _sign_aligned_allclose(my_pca.components_[i], skl_pca.components_[i])

    def test_projection_matches_sklearn_up_to_sign(self, iris_data):
        X, y = iris_data
        my_pca = PCA(n_components=2).fit(X)
        skl_pca = SklearnPCA(n_components=2).fit(X)

        my_proj = my_pca.transform(X)
        skl_proj = skl_pca.transform(X)

        for i in range(2):
            assert _sign_aligned_allclose(my_proj[:, i], skl_proj[:, i])

    def test_matches_sklearn_on_higher_dimensional_data(self, breast_cancer_data):
        X, y = breast_cancer_data
        my_pca = PCA(n_components=5).fit(X)
        skl_pca = SklearnPCA(n_components=5).fit(X)
        assert np.allclose(my_pca.explained_variance_ratio_, skl_pca.explained_variance_ratio_, atol=1e-6)


# ----------------------------------------------------------------------
# Scree plot support
# ----------------------------------------------------------------------
class TestScreeSupport:
    def test_full_scree_ratios_sums_to_one(self, iris_data):
        X, y = iris_data
        pca = PCA(n_components=2).fit(X)
        scree = pca.full_scree_ratios()
        assert len(scree) == X.shape[1]
        assert np.isclose(scree.sum(), 1.0)

    def test_full_scree_ratios_descending(self, iris_data):
        """Eigenvalues (and thus variance ratios) should be sorted descending."""
        X, y = iris_data
        pca = PCA(n_components=2).fit(X)
        scree = pca.full_scree_ratios()
        assert np.all(np.diff(scree) <= 1e-9)  # non-increasing

    def test_full_scree_raises_before_fit(self):
        pca = PCA(n_components=2)
        with pytest.raises(RuntimeError):
            pca.full_scree_ratios()


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------
class TestEdgeCases:
    def test_n_components_exceeds_features_raises(self, iris_data):
        X, y = iris_data
        with pytest.raises(ValueError):
            PCA(n_components=10).fit(X)  # iris only has 4 features

    def test_transform_before_fit_raises(self, iris_data):
        X, y = iris_data
        pca = PCA(n_components=2)
        with pytest.raises(RuntimeError):
            pca.transform(X)

    def test_n_components_equals_n_features(self, iris_data):
        X, y = iris_data
        pca = PCA(n_components=4).fit(X)  # iris has exactly 4 features
        assert np.isclose(pca.explained_variance_ratio_.sum(), 1.0, atol=1e-6)

    def test_n_components_one(self, iris_data):
        X, y = iris_data
        pca = PCA(n_components=1).fit(X)
        X_proj = pca.transform(X)
        assert X_proj.shape == (150, 1)


# ----------------------------------------------------------------------
# repr
# ----------------------------------------------------------------------
class TestRepr:
    def test_repr_shows_fit_state(self, iris_data):
        X, y = iris_data
        pca_unfit = PCA(n_components=2)
        pca_fit = PCA(n_components=2).fit(X)

        assert "fitted=False" in repr(pca_unfit)
        assert "fitted=True" in repr(pca_fit)