"""
Unit tests for the OneVsRestAdaBoost wrapper (multiclass support for the
binary-only AdaBoostClassifier).

Run with:  pytest tests/test_one_vs_rest_adaboost.py -v
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris, make_classification
from sklearn.ensemble import AdaBoostClassifier as SklearnAda
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier

from src.boosting.one_vs_rest_adaboost import OneVsRestAdaBoost


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def iris_split():
    X, y = load_iris(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ----------------------------------------------------------------------
# Basic correctness
# ----------------------------------------------------------------------
class TestBasicFit:
    def test_fit_predict_shapes(self, iris_split):
        X_train, X_test, y_train, y_test = iris_split
        ovr = OneVsRestAdaBoost(n_estimators=20, random_state=42).fit(X_train, y_train)
        preds = ovr.predict(X_test)
        assert preds.shape == y_test.shape

    def test_reasonable_accuracy_on_iris(self, iris_split):
        X_train, X_test, y_train, y_test = iris_split
        ovr = OneVsRestAdaBoost(n_estimators=30, random_state=42).fit(X_train, y_train)
        acc = accuracy_score(y_test, ovr.predict(X_test))
        assert acc > 0.8

    def test_one_binary_classifier_per_class(self, iris_split):
        X_train, X_test, y_train, y_test = iris_split
        ovr = OneVsRestAdaBoost(n_estimators=10, random_state=42).fit(X_train, y_train)
        assert ovr.n_classes_ == 3
        assert len(ovr.estimators_) == 3
        assert set(ovr.estimators_.keys()) == {0, 1, 2}


# ----------------------------------------------------------------------
# Agreement with sklearn
# ----------------------------------------------------------------------
class TestSklearnAgreement:
    def test_accuracy_close_to_sklearn_ovr(self, iris_split):
        X_train, X_test, y_train, y_test = iris_split

        my_ovr = OneVsRestAdaBoost(n_estimators=30, random_state=42).fit(X_train, y_train)
        my_acc = accuracy_score(y_test, my_ovr.predict(X_test))

        skl_ovr = OneVsRestClassifier(
            SklearnAda(
                estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
                n_estimators=30,
                random_state=42,
            )
        ).fit(X_train, y_train)
        skl_acc = accuracy_score(y_test, skl_ovr.predict(X_test))

        assert abs(my_acc - skl_acc) <= 0.05, (
            f"Accuracy diff {abs(my_acc - skl_acc):.4f} exceeds 5% "
            f"(mine={my_acc:.4f}, sklearn={skl_acc:.4f})"
        )


# ----------------------------------------------------------------------
# predict_proba
# ----------------------------------------------------------------------
class TestPredictProba:
    def test_proba_shape_and_normalization(self, iris_split):
        X_train, X_test, y_train, y_test = iris_split
        ovr = OneVsRestAdaBoost(n_estimators=20, random_state=42).fit(X_train, y_train)

        proba = ovr.predict_proba(X_test)
        assert proba.shape == (X_test.shape[0], 3)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)

    def test_predict_matches_argmax_of_proba(self, iris_split):
        X_train, X_test, y_train, y_test = iris_split
        ovr = OneVsRestAdaBoost(n_estimators=20, random_state=42).fit(X_train, y_train)

        proba = ovr.predict_proba(X_test)
        preds = ovr.predict(X_test)
        expected = ovr.classes_[np.argmax(proba, axis=1)]
        assert np.array_equal(preds, expected)


# ----------------------------------------------------------------------
# Scaling to more classes (simulating Covertype-like multiclass)
# ----------------------------------------------------------------------
class TestLargerMulticlass:
    def test_five_class_synthetic_problem(self):
        X, y = make_classification(
            n_samples=1500, n_classes=5, n_informative=10,
            n_clusters_per_class=1, random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        ovr = OneVsRestAdaBoost(n_estimators=20, random_state=42).fit(X_train, y_train)
        acc = accuracy_score(y_test, ovr.predict(X_test))

        # much better than the 1/5 = 0.20 random-guess baseline
        assert acc > 0.5

    def test_seven_class_synthetic_problem(self):
        """Mirrors Covertype's 7 forest-cover classes."""
        X, y = make_classification(
            n_samples=2000, n_classes=7, n_informative=15,
            n_clusters_per_class=1, random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        ovr = OneVsRestAdaBoost(n_estimators=20, random_state=42).fit(X_train, y_train)
        preds = ovr.predict(X_test)

        assert preds.shape == y_test.shape
        assert ovr.n_classes_ == 7
        assert len(ovr.estimators_) == 7


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------
class TestEdgeCases:
    def test_binary_case_still_works(self):
        """K=2 is a degenerate but valid case for a 'one-vs-rest' scheme."""
        X, y = make_classification(n_samples=200, n_classes=2, random_state=42)
        ovr = OneVsRestAdaBoost(n_estimators=10, random_state=42).fit(X, y)
        assert ovr.n_classes_ == 2
        preds = ovr.predict(X)
        assert preds.shape == y.shape

    def test_single_class_raises(self, iris_split):
        X_train, X_test, y_train, y_test = iris_split
        y_single = np.zeros(len(y_train))
        with pytest.raises(ValueError):
            OneVsRestAdaBoost(n_estimators=10).fit(X_train, y_single)


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
class TestReproducibility:
    def test_same_seed_gives_identical_predictions(self, iris_split):
        X_train, X_test, y_train, y_test = iris_split
        ovr1 = OneVsRestAdaBoost(n_estimators=20, random_state=7).fit(X_train, y_train)
        ovr2 = OneVsRestAdaBoost(n_estimators=20, random_state=7).fit(X_train, y_train)
        assert np.array_equal(ovr1.predict(X_test), ovr2.predict(X_test))


# ----------------------------------------------------------------------
# repr
# ----------------------------------------------------------------------
class TestRepr:
    def test_repr_shows_fit_state(self, iris_split):
        X_train, X_test, y_train, y_test = iris_split
        ovr_unfit = OneVsRestAdaBoost(n_estimators=10)
        ovr_fit = OneVsRestAdaBoost(n_estimators=10, random_state=42).fit(X_train, y_train)

        assert "fitted_classifiers=0" in repr(ovr_unfit)
        assert "fitted_classifiers=3" in repr(ovr_fit)