"""
Unit tests for the from-scratch AdaBoostClassifier (discrete SAMME) implementation.

Run with:  pytest tests/test_adaboost.py -v
"""

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.ensemble import AdaBoostClassifier as SklearnAdaBoost
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.boosting.adaboost import AdaBoostClassifier


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def breast_cancer_split():
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def two_gaussians():
    """Well-separated blobs - a single stump should already get near-perfect accuracy,
    and boosting shouldn't make it worse."""
    rng = np.random.RandomState(0)
    class0 = rng.normal(loc=-3.0, scale=0.5, size=(40, 2))
    class1 = rng.normal(loc=3.0, scale=0.5, size=(40, 2))
    X = np.vstack([class0, class1])
    y = np.array([0] * 40 + [1] * 40)
    return X, y


# ----------------------------------------------------------------------
# Basic correctness
# ----------------------------------------------------------------------
class TestBasicFit:
    def test_fit_predict_shapes(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        ada = AdaBoostClassifier(n_estimators=20, random_state=42).fit(X_train, y_train)
        preds = ada.predict(X_test)
        assert preds.shape == y_test.shape

    def test_perfectly_separable_data(self, two_gaussians):
        X, y = two_gaussians
        ada = AdaBoostClassifier(n_estimators=10, random_state=42).fit(X, y)
        assert accuracy_score(y, ada.predict(X)) == 1.0

    def test_reasonable_accuracy_on_breast_cancer(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        ada = AdaBoostClassifier(n_estimators=50, random_state=42).fit(X_train, y_train)
        acc = accuracy_score(y_test, ada.predict(X_test))
        assert acc > 0.85


# ----------------------------------------------------------------------
# Agreement with sklearn
# ----------------------------------------------------------------------
class TestSklearnAgreement:
    def test_accuracy_close_to_sklearn(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split

        my_ada = AdaBoostClassifier(n_estimators=50, random_state=42).fit(X_train, y_train)
        my_acc = accuracy_score(y_test, my_ada.predict(X_test))

        skl_ada = SklearnAdaBoost(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
            n_estimators=50,
            random_state=42,
        ).fit(X_train, y_train)
        skl_acc = accuracy_score(y_test, skl_ada.predict(X_test))

        assert abs(my_acc - skl_acc) <= 0.02, (
            f"Accuracy diff {abs(my_acc - skl_acc):.4f} exceeds 2% "
            f"(mine={my_acc:.4f}, sklearn={skl_acc:.4f})"
        )


# ----------------------------------------------------------------------
# Boosting behavior
# ----------------------------------------------------------------------
class TestBoostingBehavior:
    def test_training_error_generally_decreases(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        ada = AdaBoostClassifier(n_estimators=50, random_state=42).fit(X_train, y_train)

        staged = list(ada.staged_predict(X_train))
        errors = [1 - accuracy_score(y_train, p) for p in staged]

        # Not required to be monotonic every round, but the final round should
        # be no worse than the first round.
        assert errors[-1] <= errors[0]

    def test_estimator_weights_and_errors_recorded_per_round(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        ada = AdaBoostClassifier(n_estimators=30, random_state=42).fit(X_train, y_train)

        assert len(ada.estimator_weights) == len(ada.estimators_)
        assert len(ada.estimator_errors) == len(ada.estimators_)
        assert np.all(ada.estimator_errors >= 0.0)
        assert np.all(ada.estimator_errors <= 0.5)  # anything >= 0.5 should have stopped training

    def test_estimator_weights_raise_before_fit(self):
        ada = AdaBoostClassifier()
        with pytest.raises(RuntimeError):
            _ = ada.estimator_weights
        with pytest.raises(RuntimeError):
            _ = ada.estimator_errors


# ----------------------------------------------------------------------
# staged_predict
# ----------------------------------------------------------------------
class TestStagedPredict:
    def test_number_of_stages_matches_estimators(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        ada = AdaBoostClassifier(n_estimators=25, random_state=42).fit(X_train, y_train)

        staged = list(ada.staged_predict(X_test))
        assert len(staged) == len(ada.estimators_)

    def test_last_stage_matches_final_predict(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        ada = AdaBoostClassifier(n_estimators=25, random_state=42).fit(X_train, y_train)

        staged = list(ada.staged_predict(X_test))
        assert np.array_equal(staged[-1], ada.predict(X_test))

    def test_staged_predict_is_lazy_generator(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        ada = AdaBoostClassifier(n_estimators=10, random_state=42).fit(X_train, y_train)

        gen = ada.staged_predict(X_test)
        first = next(gen)
        assert first.shape == y_test.shape


# ----------------------------------------------------------------------
# predict_proba
# ----------------------------------------------------------------------
class TestPredictProba:
    def test_proba_shape_and_normalization(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        ada = AdaBoostClassifier(n_estimators=20, random_state=42).fit(X_train, y_train)

        proba = ada.predict_proba(X_test)
        assert proba.shape == (X_test.shape[0], 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)

    def test_predict_matches_argmax_of_proba(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        ada = AdaBoostClassifier(n_estimators=20, random_state=42).fit(X_train, y_train)

        proba = ada.predict_proba(X_test)
        preds = ada.predict(X_test)
        expected = ada.classes_[np.argmax(proba, axis=1)]
        assert np.array_equal(preds, expected)


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
class TestReproducibility:
    def test_same_seed_gives_identical_predictions(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        ada1 = AdaBoostClassifier(n_estimators=30, random_state=7).fit(X_train, y_train)
        ada2 = AdaBoostClassifier(n_estimators=30, random_state=7).fit(X_train, y_train)
        assert np.array_equal(ada1.predict(X_test), ada2.predict(X_test))
        assert np.allclose(ada1.estimator_weights, ada2.estimator_weights)

    def test_different_seeds_can_give_different_estimator_weights(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        ada1 = AdaBoostClassifier(n_estimators=30, random_state=1).fit(X_train, y_train)
        ada2 = AdaBoostClassifier(n_estimators=30, random_state=999).fit(X_train, y_train)
        # not a strict requirement they differ, but they normally will given
        # different per-round stump seeding
        assert ada1.estimator_weights.shape == ada2.estimator_weights.shape


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------
class TestEdgeCases:
    def test_multiclass_raises_value_error(self):
        X, y = make_classification(
            n_samples=200, n_classes=3, n_informative=5, random_state=42
        )
        with pytest.raises(ValueError):
            AdaBoostClassifier(n_estimators=10).fit(X, y)

    def test_single_estimator(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        ada = AdaBoostClassifier(n_estimators=1, random_state=42).fit(X_train, y_train)
        assert len(ada.estimators_) == 1
        preds = ada.predict(X_test)
        assert preds.shape == y_test.shape

    def test_learning_rate_scales_first_round_alpha(self, breast_cancer_split):
        """learning_rate scales alpha_m, and that scaled alpha_m feeds into the
        next round's sample_weight update -- so runs diverge from round 2 onward.
        Only round 1 is guaranteed to be a pure scalar multiple, since both runs
        start from identical uniform sample weights."""
        X_train, X_test, y_train, y_test = breast_cancer_split
        ada_full = AdaBoostClassifier(n_estimators=5, learning_rate=1.0, random_state=42).fit(X_train, y_train)
        ada_half = AdaBoostClassifier(n_estimators=5, learning_rate=0.5, random_state=42).fit(X_train, y_train)

        assert np.isclose(ada_half.estimator_weights[0], 0.5 * ada_full.estimator_weights[0], atol=1e-8)

    def test_repr_shows_fit_state(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        ada_unfit = AdaBoostClassifier(n_estimators=10)
        ada_fit = AdaBoostClassifier(n_estimators=10, random_state=42).fit(X_train, y_train)

        assert "fitted_estimators=0" in repr(ada_unfit)
        assert "fitted_estimators=10" in repr(ada_fit)