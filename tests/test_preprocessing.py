"""
Unit tests for preprocessing utilities: missing values, scaling, imbalance handling.

Run with:  pytest tests/test_preprocessing.py -v
"""

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from src.utils.preprocessing import (
    fit_scaler,
    handle_imbalance,
    handle_missing_values,
    minority_class_fraction,
    scale_data,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def breast_cancer_split():
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def data_with_nans():
    return np.array([
        [1.0, 2.0],
        [np.nan, 4.0],
        [5.0, np.nan],
        [7.0, 8.0],
    ])


@pytest.fixture
def imbalanced_data():
    rng = np.random.RandomState(0)
    X_majority = rng.normal(0, 1, (500, 3))
    X_minority = rng.normal(5, 1, (10, 3))
    X = np.vstack([X_majority, X_minority])
    y = np.array([0] * 500 + [1] * 10)
    return X, y


# ----------------------------------------------------------------------
# Missing values
# ----------------------------------------------------------------------
class TestHandleMissingValues:
    def test_mean_strategy_removes_all_nans(self, data_with_nans):
        result = handle_missing_values(data_with_nans, strategy="mean")
        assert not np.any(np.isnan(result))
        assert result.shape == data_with_nans.shape

    def test_median_strategy_removes_all_nans(self, data_with_nans):
        result = handle_missing_values(data_with_nans, strategy="median")
        assert not np.any(np.isnan(result))

    def test_mean_strategy_correct_value(self, data_with_nans):
        result = handle_missing_values(data_with_nans, strategy="mean")
        # column 0 has values [1, 5, 7] (excluding NaN) -> mean = 13/3
        expected_col0_fill = (1.0 + 5.0 + 7.0) / 3
        assert np.isclose(result[1, 0], expected_col0_fill)

    def test_drop_strategy_removes_rows_with_nans(self, data_with_nans):
        result = handle_missing_values(data_with_nans, strategy="drop")
        assert result.shape[0] == 2  # only rows 0 and 3 have no NaN
        assert not np.any(np.isnan(result))

    def test_no_nans_unchanged(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = handle_missing_values(X, strategy="mean")
        assert np.array_equal(result, X)

    def test_invalid_strategy_raises(self, data_with_nans):
        with pytest.raises(ValueError):
            handle_missing_values(data_with_nans, strategy="not_a_real_strategy")

    def test_custom_missing_value_sentinel(self):
        X = np.array([[1.0, 2.0], [-999.0, 4.0], [5.0, 6.0]])
        result = handle_missing_values(X, strategy="mean", missing_value=-999.0)
        assert not np.any(result == -999.0)
        expected = (1.0 + 5.0) / 2
        assert np.isclose(result[1, 0], expected)


# ----------------------------------------------------------------------
# Scaling
# ----------------------------------------------------------------------
class TestScaling:
    def test_train_scaled_to_zero_mean_unit_variance(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        scaler = fit_scaler(X_train)
        X_train_scaled = scale_data(scaler, X_train)

        assert np.allclose(X_train_scaled.mean(axis=0), 0.0, atol=1e-8)
        assert np.allclose(X_train_scaled.std(axis=0), 1.0, atol=1e-8)

    def test_test_set_uses_train_statistics_not_its_own(self, breast_cancer_split):
        """Fitting on test data would leak test statistics -- verify the test
        set is NOT independently re-centered to exactly mean 0."""
        X_train, X_test, y_train, y_test = breast_cancer_split
        scaler = fit_scaler(X_train)
        X_test_scaled = scale_data(scaler, X_test)

        # test set mean should generally NOT be exactly 0 (it's scaled using
        # train mean/std, and train/test distributions differ slightly)
        assert not np.allclose(X_test_scaled.mean(axis=0), 0.0, atol=1e-8)

    def test_scaler_reused_consistently(self, breast_cancer_split):
        X_train, X_test, y_train, y_test = breast_cancer_split
        scaler = fit_scaler(X_train)
        scaled_once = scale_data(scaler, X_test)
        scaled_twice = scale_data(scaler, X_test)
        assert np.array_equal(scaled_once, scaled_twice)


# ----------------------------------------------------------------------
# Imbalance handling
# ----------------------------------------------------------------------
class TestMinorityClassFraction:
    def test_known_imbalance_ratio(self, imbalanced_data):
        X, y = imbalanced_data
        frac = minority_class_fraction(y)
        assert np.isclose(frac, 10 / 510)

    def test_balanced_data_gives_half(self):
        y = np.array([0, 0, 1, 1])
        assert minority_class_fraction(y) == 0.5


class TestHandleImbalance:
    def test_smote_balances_classes(self, imbalanced_data):
        X, y = imbalanced_data
        X_res, y_res = handle_imbalance(X, y, method="smote", random_state=42)
        assert np.isclose(minority_class_fraction(y_res), 0.5, atol=0.01)
        assert X_res.shape[1] == X.shape[1]

    def test_random_oversample_balances_classes(self, imbalanced_data):
        X, y = imbalanced_data
        X_res, y_res = handle_imbalance(X, y, method="random_oversample", random_state=42)
        assert np.isclose(minority_class_fraction(y_res), 0.5, atol=0.01)

    def test_random_oversample_preserves_all_original_rows(self, imbalanced_data):
        X, y = imbalanced_data
        X_res, y_res = handle_imbalance(X, y, method="random_oversample", random_state=42)
        # resampled dataset must be at least as large as original (only duplicates, never removes)
        assert X_res.shape[0] >= X.shape[0]
        # every original minority sample should still appear in the result
        minority_original = X[y == 1]
        for row in minority_original:
            assert np.any(np.all(np.isclose(X_res, row), axis=1))

    def test_class_weight_returns_sample_weight_not_resampled_data(self, imbalanced_data):
        X, y = imbalanced_data
        X_out, y_out, sample_weight = handle_imbalance(X, y, method="class_weight", random_state=42)
        assert X_out.shape == X.shape  # unchanged, no resampling
        assert y_out.shape == y.shape
        assert sample_weight.shape == y.shape

    def test_class_weight_upweights_minority(self, imbalanced_data):
        X, y = imbalanced_data
        _, _, sample_weight = handle_imbalance(X, y, method="class_weight", random_state=42)
        minority_weight = sample_weight[y == 1][0]
        majority_weight = sample_weight[y == 0][0]
        assert minority_weight > majority_weight

    def test_invalid_method_raises(self, imbalanced_data):
        X, y = imbalanced_data
        with pytest.raises(ValueError):
            handle_imbalance(X, y, method="not_a_real_method")

    def test_already_balanced_data_stays_roughly_balanced(self):
        rng = np.random.RandomState(0)
        X = rng.normal(0, 1, (100, 2))
        y = np.array([0] * 50 + [1] * 50)
        X_res, y_res = handle_imbalance(X, y, method="random_oversample", random_state=42)
        assert X_res.shape[0] == X.shape[0]  # nothing to add, already balanced