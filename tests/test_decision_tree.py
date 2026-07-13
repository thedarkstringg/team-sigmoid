"""
Unit tests for the from-scratch DecisionTree / DecisionStump implementation.

Run with:  pytest tests/test_decision_tree.py -v
"""

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.trees.decision_tree import DecisionStump, DecisionTree


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def breast_cancer_split():
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def xor_data():
    """Tiny 2D dataset with a known, exact split (classic XOR)."""
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    y = np.array([0, 1, 1, 0])
    return X, y


@pytest.fixture
def two_gaussians():
    """Two well-separated Gaussian blobs - should be perfectly separable by a stump."""
    rng = np.random.RandomState(0)
    class0 = rng.normal(loc=-3.0, scale=0.5, size=(30, 2))
    class1 = rng.normal(loc=3.0, scale=0.5, size=(30, 2))
    X = np.vstack([class0, class1])
    y = np.array([0] * 30 + [1] * 30)
    return X, y


# ----------------------------------------------------------------------
# Sanity checks on tiny, known datasets
# ----------------------------------------------------------------------
class TestKnownSplits:
    def test_two_gaussians_perfectly_separable(self, two_gaussians):
        X, y = two_gaussians
        tree = DecisionTree(max_depth=2, random_state=42).fit(X, y)
        preds = tree.predict(X)
        assert accuracy_score(y, preds) == 1.0

    def test_xor_requires_depth_2(self, xor_data):
        """XOR is not linearly separable by a single split; a depth-1 stump
        cannot solve it perfectly, but depth >= 2 can."""
        X, y = xor_data
        stump = DecisionStump(random_state=42).fit(X, y)
        deep_tree = DecisionTree(max_depth=3, random_state=42).fit(X, y)

        stump_acc = accuracy_score(y, stump.predict(X))
        deep_acc = accuracy_score(y, deep_tree.predict(X))

        assert deep_acc == 1.0
        assert deep_acc >= stump_acc


# ----------------------------------------------------------------------
# Agreement with sklearn (must be within 2%, per project brief)
# ----------------------------------------------------------------------
class TestSklearnAgreement:
    @pytest.mark.parametrize("criterion", ["gini", "entropy"])
    def test_accuracy_within_2_percent(self, breast_cancer_split, criterion):
        X_train, X_test, y_train, y_test = breast_cancer_split

        my_tree = DecisionTree(max_depth=5, min_samples_split=2, criterion=criterion, random_state=42)
        my_tree.fit(X_train, y_train)
        my_acc = accuracy_score(y_test, my_tree.predict(X_test))

        skl_tree = DecisionTreeClassifier(max_depth=5, min_samples_split=2, criterion=criterion, random_state=42)
        skl_tree.fit(X_train, y_train)
        skl_acc = accuracy_score(y_test, skl_tree.predict(X_test))

        assert abs(my_acc - skl_acc) <= 0.02, (
            f"Accuracy diff {abs(my_acc - skl_acc):.4f} exceeds 2% "
            f"(mine={my_acc:.4f}, sklearn={skl_acc:.4f})"
        )

    def test_depth_matches_sklearn(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split

        my_tree = DecisionTree(max_depth=5, min_samples_split=2, random_state=42).fit(X_train, y_train)
        skl_tree = DecisionTreeClassifier(max_depth=5, min_samples_split=2, random_state=42).fit(X_train, y_train)

        assert my_tree.depth == skl_tree.get_depth()

    def test_n_leaves_matches_sklearn(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split

        my_tree = DecisionTree(max_depth=5, min_samples_split=2, random_state=42).fit(X_train, y_train)
        skl_tree = DecisionTreeClassifier(max_depth=5, min_samples_split=2, random_state=42).fit(X_train, y_train)

        assert my_tree.n_leaves == skl_tree.get_n_leaves()


# ----------------------------------------------------------------------
# predict_proba correctness
# ----------------------------------------------------------------------
class TestPredictProba:
    def test_proba_shape_and_normalization(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        tree = DecisionTree(max_depth=5, random_state=42).fit(X_train, y_train)

        proba = tree.predict_proba(X_test)
        assert proba.shape == (X_test.shape[0], 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)

    def test_predict_matches_argmax_of_proba(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        tree = DecisionTree(max_depth=5, random_state=42).fit(X_train, y_train)

        proba = tree.predict_proba(X_test)
        preds = tree.predict(X_test)
        expected = tree.classes_[np.argmax(proba, axis=1)]
        assert np.array_equal(preds, expected)


# ----------------------------------------------------------------------
# sample_weight support (required for AdaBoost)
# ----------------------------------------------------------------------
class TestSampleWeight:
    def test_uniform_weights_match_unweighted_fit(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        weights = np.ones(len(y_train))

        tree_unweighted = DecisionTree(max_depth=5, random_state=42).fit(X_train, y_train)
        tree_weighted = DecisionTree(max_depth=5, random_state=42).fit(X_train, y_train, sample_weight=weights)

        assert np.array_equal(tree_unweighted.predict(X_test), tree_weighted.predict(X_test))

    def test_heavily_upweighted_samples_are_classified_correctly(self, breast_cancer_split):
        """If a small subset is massively upweighted, the tree should
        prioritize getting those samples right."""
        X_train, X_test, y_train, y_test = breast_cancer_split
        weights = np.ones(len(y_train))
        weights[:20] = 100.0

        tree = DecisionTree(max_depth=5, random_state=42).fit(X_train, y_train, sample_weight=weights)
        preds_on_upweighted = tree.predict(X_train[:20])
        assert accuracy_score(y_train[:20], preds_on_upweighted) >= 0.9

    def test_decision_stump_accepts_sample_weight(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        weights = np.ones(len(y_train))
        stump = DecisionStump(random_state=42).fit(X_train, y_train, sample_weight=weights)
        assert stump.depth == 1


# ----------------------------------------------------------------------
# feature_importances
# ----------------------------------------------------------------------
class TestFeatureImportances:
    def test_importances_sum_to_one(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        tree = DecisionTree(max_depth=5, random_state=42).fit(X_train, y_train)

        fi = tree.feature_importances()
        assert fi.shape == (X_train.shape[1],)
        assert np.all(fi >= 0.0)
        assert abs(fi.sum() - 1.0) < 1e-9

    def test_importances_raise_before_fit(self):
        tree = DecisionTree()
        with pytest.raises(RuntimeError):
            tree.feature_importances()


# ----------------------------------------------------------------------
# max_features (required for Random Forest)
# ----------------------------------------------------------------------
class TestMaxFeatures:
    @pytest.mark.parametrize("max_features", ["sqrt", "log2", 5, None])
    def test_max_features_runs_without_error(self, breast_cancer_split, max_features):
        X_train, X_test, y_train, y_test = breast_cancer_split
        tree = DecisionTree(max_depth=5, max_features=max_features, random_state=42)
        tree.fit(X_train, y_train)
        preds = tree.predict(X_test)
        assert preds.shape == y_test.shape

    def test_invalid_max_features_raises(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        tree = DecisionTree(max_features="not_a_valid_option")
        with pytest.raises(ValueError):
            tree.fit(X_train, y_train)


# ----------------------------------------------------------------------
# Edge cases (explicitly required by the project brief)
# ----------------------------------------------------------------------
class TestEdgeCases:
    def test_single_feature_dataset(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        X_train_1d = X_train[:, [0]]
        X_test_1d = X_test[:, [0]]

        tree = DecisionTree(max_depth=3, random_state=42).fit(X_train_1d, y_train)
        preds = tree.predict(X_test_1d)
        assert preds.shape == y_test.shape

    def test_all_identical_labels_produces_single_leaf(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        y_same = np.zeros(len(y_train), dtype=int)

        tree = DecisionTree(max_depth=3, random_state=42).fit(X_train, y_same)
        assert tree.root.is_leaf()
        assert tree.depth == 0
        assert np.all(tree.predict(X_test) == 0)

    def test_all_identical_feature_rows(self):
        X = np.ones((10, 3))
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        tree = DecisionTree(max_depth=3, random_state=42).fit(X, y)
        assert tree.root.is_leaf()  # no valid split possible

    def test_min_samples_split_equals_1(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        tree = DecisionTree(min_samples_split=1, max_depth=3, random_state=42)
        tree.fit(X_train, y_train)  # should not raise
        preds = tree.predict(X_test)
        assert preds.shape == y_test.shape

    def test_max_depth_zero_is_single_leaf_stump(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        tree = DecisionTree(max_depth=0, random_state=42).fit(X_train, y_train)

        assert tree.depth == 0
        assert tree.root.is_leaf()
        # prediction should be the majority class for every test point
        majority_class = tree.classes_[np.argmax(tree.root.value)]
        assert np.all(tree.predict(X_test) == majority_class)

    def test_invalid_criterion_raises(self):
        with pytest.raises(ValueError):
            DecisionTree(criterion="not_a_real_criterion")


# ----------------------------------------------------------------------
# repr
# ----------------------------------------------------------------------
class TestRepr:
    def test_repr_shallow_tree_shows_structure(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        tree = DecisionTree(max_depth=2, random_state=42).fit(X_train, y_train)

        text = repr(tree)
        assert "X" in text and "<=" in text
        assert "Leaf" in text

    def test_repr_deep_tree_shows_summary(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        tree = DecisionTree(max_depth=10, random_state=42).fit(X_train, y_train)

        text = repr(tree)
        assert "depth=" in text
        assert "n_leaves=" in text

    def test_repr_unfit_tree(self):
        tree = DecisionTree()
        assert "unfit" in repr(tree)
