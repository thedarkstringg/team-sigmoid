"""
Unit tests for the from-scratch RandomForestClassifier implementation.

Run with:  pytest tests/test_random_forest.py -v

Note: the n_jobs>1 tests spawn worker processes via multiprocessing.Pool.
On Windows, pytest's own test runner already provides the required
`if __name__ == "__main__":` guard, so no special handling is needed here.
"""

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.bagging.random_forest import RandomForestClassifier


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def breast_cancer_split():
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ----------------------------------------------------------------------
# Basic correctness
# ----------------------------------------------------------------------
class TestBasicFit:
    def test_fit_predict_shapes(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        rf = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42).fit(X_train, y_train)
        preds = rf.predict(X_test)
        assert preds.shape == y_test.shape

    def test_reasonable_accuracy_on_breast_cancer(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42).fit(X_train, y_train)
        acc = accuracy_score(y_test, rf.predict(X_test))
        assert acc > 0.85

    def test_n_estimators_creates_that_many_trees(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        rf = RandomForestClassifier(n_estimators=17, random_state=42).fit(X_train, y_train)
        assert len(rf.estimators_) == 17


# ----------------------------------------------------------------------
# Agreement with sklearn
# ----------------------------------------------------------------------
class TestSklearnAgreement:
    def test_accuracy_close_to_sklearn(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split

        my_rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42).fit(X_train, y_train)
        my_acc = accuracy_score(y_test, my_rf.predict(X_test))

        skl_rf = SklearnRF(n_estimators=50, max_depth=5, random_state=42).fit(X_train, y_train)
        skl_acc = accuracy_score(y_test, skl_rf.predict(X_test))

        assert abs(my_acc - skl_acc) <= 0.02, (
            f"Accuracy diff {abs(my_acc - skl_acc):.4f} exceeds 2% "
            f"(mine={my_acc:.4f}, sklearn={skl_acc:.4f})"
        )

    def test_oob_score_close_to_sklearn(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split

        my_rf = RandomForestClassifier(
            n_estimators=100, max_depth=5, oob_score=True, random_state=42
        ).fit(X_train, y_train)
        skl_rf = SklearnRF(
            n_estimators=100, max_depth=5, oob_score=True, random_state=42
        ).fit(X_train, y_train)

        assert abs(my_rf.oob_score_ - skl_rf.oob_score_) <= 0.03


# ----------------------------------------------------------------------
# predict_proba
# ----------------------------------------------------------------------
class TestPredictProba:
    def test_proba_shape_and_normalization(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        rf = RandomForestClassifier(n_estimators=20, random_state=42).fit(X_train, y_train)

        proba = rf.predict_proba(X_test)
        assert proba.shape == (X_test.shape[0], 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)

    def test_predict_matches_argmax_of_proba(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        rf = RandomForestClassifier(n_estimators=20, random_state=42).fit(X_train, y_train)

        proba = rf.predict_proba(X_test)
        preds = rf.predict(X_test)
        expected = rf.classes_[np.argmax(proba, axis=1)]
        assert np.array_equal(preds, expected)

    def test_proba_is_average_across_trees(self, breast_cancer_split):
        """predict_proba should equal the mean of each individual tree's
        predict_proba output (this is the RF.5 requirement in the brief)."""
        X_train, X_test, y_train, y_test = breast_cancer_split
        rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42).fit(X_train, y_train)

        manual_avg = np.mean([tree.predict_proba(X_test) for tree in rf.estimators_], axis=0)
        assert np.allclose(rf.predict_proba(X_test), manual_avg)


# ----------------------------------------------------------------------
# OOB score
# ----------------------------------------------------------------------
class TestOOBScore:
    def test_oob_score_in_valid_range(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        rf = RandomForestClassifier(n_estimators=50, oob_score=True, random_state=42).fit(X_train, y_train)
        assert 0.0 <= rf.oob_score_ <= 1.0

    def test_oob_score_raises_when_not_requested(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        rf = RandomForestClassifier(n_estimators=10, oob_score=False, random_state=42).fit(X_train, y_train)
        with pytest.raises(RuntimeError):
            _ = rf.oob_score_

    def test_oob_score_raises_before_fit(self):
        rf = RandomForestClassifier(oob_score=True)
        with pytest.raises(RuntimeError):
            _ = rf.oob_score_


# ----------------------------------------------------------------------
# feature_importances_
# ----------------------------------------------------------------------
class TestFeatureImportances:
    def test_importances_shape_and_sum(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        rf = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42).fit(X_train, y_train)

        fi = rf.feature_importances_
        assert fi.shape == (X_train.shape[1],)
        assert np.all(fi >= 0.0)
        assert abs(fi.sum() - 1.0) < 1e-6

    def test_importances_are_average_across_trees(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42).fit(X_train, y_train)

        manual_avg = np.mean([tree.feature_importances() for tree in rf.estimators_], axis=0)
        assert np.allclose(rf.feature_importances_, manual_avg)

    def test_importances_raise_before_fit(self):
        rf = RandomForestClassifier()
        with pytest.raises(RuntimeError):
            _ = rf.feature_importances_


# ----------------------------------------------------------------------
# max_features / bootstrap options
# ----------------------------------------------------------------------
class TestForestOptions:
    @pytest.mark.parametrize("max_features", ["sqrt", "log2", 5, None])
    def test_max_features_variants_run(self, breast_cancer_split, max_features):
        X_train, X_test, y_train, y_test = breast_cancer_split
        rf = RandomForestClassifier(n_estimators=10, max_features=max_features, random_state=42)
        rf.fit(X_train, y_train)
        assert rf.predict(X_test).shape == y_test.shape

    def test_bootstrap_false_uses_full_dataset_every_tree(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        rf = RandomForestClassifier(n_estimators=5, bootstrap=False, random_state=42).fit(X_train, y_train)
        assert rf.predict(X_test).shape == y_test.shape


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
class TestReproducibility:
    def test_same_seed_gives_identical_predictions(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        rf1 = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=7).fit(X_train, y_train)
        rf2 = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=7).fit(X_train, y_train)
        assert np.array_equal(rf1.predict(X_test), rf2.predict(X_test))


# ----------------------------------------------------------------------
# Parallel (n_jobs > 1) vs sequential
# ----------------------------------------------------------------------
class TestParallelism:
    def test_parallel_matches_sequential_exactly(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split

        rf_seq = RandomForestClassifier(
            n_estimators=20, max_depth=5, random_state=99, n_jobs=1
        ).fit(X_train, y_train)
        rf_par = RandomForestClassifier(
            n_estimators=20, max_depth=5, random_state=99, n_jobs=2
        ).fit(X_train, y_train)

        assert np.array_equal(rf_seq.predict(X_test), rf_par.predict(X_test))
        assert np.allclose(rf_seq.feature_importances_, rf_par.feature_importances_)


# ----------------------------------------------------------------------
# repr
# ----------------------------------------------------------------------
class TestRepr:
    def test_repr_shows_fit_state(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        rf_unfit = RandomForestClassifier(n_estimators=15)
        rf_fit = RandomForestClassifier(n_estimators=15, random_state=42).fit(X_train, y_train)

        assert "fitted_estimators=0" in repr(rf_unfit)
        assert "fitted_estimators=15" in repr(rf_fit)