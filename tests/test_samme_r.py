"""
Unit tests for the SAMME.R (real-valued multiclass AdaBoost) implementation.

Run with:  pytest tests/test_samme_r.py -v

NOTE: includes a regression test (test_training_accuracy_improves_monotonically_ish
on a 7-class problem) that specifically catches a bug found during development:
an earlier version used h_k(x) instead of log(p_k(x)) in the sample-weight
update, which is mathematically wrong (h carries an extra factor of (K-1)
relative to log(p)) and made weight updates catastrophically unstable as K
grew -- harmless at K=2, severely broken by K=7. That bug passed every test
that only used K<=3, which is why a K=7 case is deliberately included here.
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris, make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.boosting.samme_r import SAMMERClassifier


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def iris_split():
    X, y = load_iris(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


@pytest.fixture(scope="module")
def seven_class_split():
    """The scale at which the (K-1)-factor bug was caught -- keep this
    fixture even if other tests seem redundant with it."""
    X, y = make_classification(
        n_samples=2000, n_classes=7, n_informative=15,
        n_clusters_per_class=1, random_state=42,
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ----------------------------------------------------------------------
# Basic correctness
# ----------------------------------------------------------------------
class TestBasicFit:
    def test_fit_predict_shapes(self, iris_split):
        X_train, X_test, y_train, y_test = iris_split
        model = SAMMERClassifier(n_estimators=20, random_state=42).fit(X_train, y_train)
        preds = model.predict(X_test)
        assert preds.shape == y_test.shape

    def test_reasonable_accuracy_on_iris(self, iris_split):
        X_train, X_test, y_train, y_test = iris_split
        model = SAMMERClassifier(n_estimators=30, random_state=42).fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        assert acc > 0.8

    def test_one_shared_ensemble_not_per_class(self, iris_split):
        """Unlike OneVsRestAdaBoost, SAMME.R trains ONE ensemble shared
        across all classes -- len(estimators_) == n_estimators, not
        n_estimators * n_classes."""
        X_train, X_test, y_train, y_test = iris_split
        model = SAMMERClassifier(n_estimators=15, random_state=42).fit(X_train, y_train)
        assert len(model.estimators_) == 15
        assert model.n_classes_ == 3


# ----------------------------------------------------------------------
# Regression test for the (K-1)-factor weight-update bug
# ----------------------------------------------------------------------
class TestSevenClassRegression:
    def test_training_accuracy_improves_substantially(self, seven_class_split):
        """The buggy version got stuck at ~random-chance accuracy (~1/7)
        even on the TRAINING set, because unstable weight updates prevented
        the ensemble from ever learning. A correct implementation should
        comfortably exceed random chance on training data it has seen."""
        X_train, X_test, y_train, y_test = seven_class_split
        model = SAMMERClassifier(n_estimators=30, random_state=42).fit(X_train, y_train)

        train_preds = model.predict(X_train)
        train_acc = accuracy_score(y_train, train_preds)

        random_chance = 1.0 / 7
        assert train_acc > random_chance + 0.15, (
            f"Train accuracy {train_acc:.4f} is too close to random chance "
            f"({random_chance:.4f}) -- possible regression of the (K-1)-factor bug"
        )

    def test_training_error_decreases_across_rounds(self, seven_class_split):
        X_train, X_test, y_train, y_test = seven_class_split
        model = SAMMERClassifier(n_estimators=30, random_state=42).fit(X_train, y_train)

        staged = list(model.staged_predict(X_train))
        early_acc = accuracy_score(y_train, staged[4])   # round 5
        late_acc = accuracy_score(y_train, staged[-1])   # final round

        assert late_acc > early_acc, (
            f"Training accuracy did not improve from round 5 ({early_acc:.4f}) "
            f"to the final round ({late_acc:.4f}) -- the ensemble isn't learning."
        )

    def test_test_accuracy_well_above_random_chance(self, seven_class_split):
        X_train, X_test, y_train, y_test = seven_class_split
        model = SAMMERClassifier(n_estimators=30, random_state=42).fit(X_train, y_train)

        test_acc = accuracy_score(y_test, model.predict(X_test))
        assert test_acc > (1.0 / 7) + 0.15


# ----------------------------------------------------------------------
# predict_proba
# ----------------------------------------------------------------------
class TestPredictProba:
    def test_proba_shape_and_normalization(self, iris_split):
        X_train, X_test, y_train, y_test = iris_split
        model = SAMMERClassifier(n_estimators=20, random_state=42).fit(X_train, y_train)

        proba = model.predict_proba(X_test)
        assert proba.shape == (X_test.shape[0], 3)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)

    def test_predict_matches_argmax_of_proba(self, iris_split):
        X_train, X_test, y_train, y_test = iris_split
        model = SAMMERClassifier(n_estimators=20, random_state=42).fit(X_train, y_train)

        proba = model.predict_proba(X_test)
        preds = model.predict(X_test)
        expected = model.classes_[np.argmax(proba, axis=1)]
        assert np.array_equal(preds, expected)


# ----------------------------------------------------------------------
# staged_predict
# ----------------------------------------------------------------------
class TestStagedPredict:
    def test_number_of_stages_matches_estimators(self, iris_split):
        X_train, X_test, y_train, y_test = iris_split
        model = SAMMERClassifier(n_estimators=25, random_state=42).fit(X_train, y_train)
        staged = list(model.staged_predict(X_test))
        assert len(staged) == len(model.estimators_)

    def test_last_stage_matches_final_predict(self, iris_split):
        X_train, X_test, y_train, y_test = iris_split
        model = SAMMERClassifier(n_estimators=25, random_state=42).fit(X_train, y_train)
        staged = list(model.staged_predict(X_test))
        assert np.array_equal(staged[-1], model.predict(X_test))


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------
class TestEdgeCases:
    def test_binary_case_works(self):
        """K=2 is the case where the (K-1)-factor bug had ZERO effect
        (factor = 1), which is exactly why it went undetected until a
        larger K was tested."""
        X, y = make_classification(n_samples=200, n_classes=2, random_state=42)
        model = SAMMERClassifier(n_estimators=15, random_state=42).fit(X, y)
        assert model.n_classes_ == 2
        acc = accuracy_score(y, model.predict(X))
        assert acc > 0.7

    def test_single_class_raises(self, iris_split):
        X_train, X_test, y_train, y_test = iris_split
        y_single = np.zeros(len(y_train))
        with pytest.raises(ValueError):
            SAMMERClassifier(n_estimators=10).fit(X_train, y_single)

    def test_more_estimators_generally_helps_on_hard_problem(self, seven_class_split):
        X_train, X_test, y_train, y_test = seven_class_split
        acc_few = accuracy_score(
            y_test,
            SAMMERClassifier(n_estimators=10, random_state=42).fit(X_train, y_train).predict(X_test),
        )
        acc_many = accuracy_score(
            y_test,
            SAMMERClassifier(n_estimators=100, random_state=42).fit(X_train, y_train).predict(X_test),
        )
        assert acc_many >= acc_few


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
class TestReproducibility:
    def test_same_seed_gives_identical_predictions(self, iris_split):
        X_train, X_test, y_train, y_test = iris_split
        m1 = SAMMERClassifier(n_estimators=20, random_state=7).fit(X_train, y_train)
        m2 = SAMMERClassifier(n_estimators=20, random_state=7).fit(X_train, y_train)
        assert np.array_equal(m1.predict(X_test), m2.predict(X_test))


# ----------------------------------------------------------------------
# repr
# ----------------------------------------------------------------------
class TestRepr:
    def test_repr_shows_fit_state(self, iris_split):
        X_train, X_test, y_train, y_test = iris_split
        m_unfit = SAMMERClassifier(n_estimators=10)
        m_fit = SAMMERClassifier(n_estimators=10, random_state=42).fit(X_train, y_train)

        assert "fitted_estimators=0" in repr(m_unfit)
        assert "fitted_estimators=10" in repr(m_fit)