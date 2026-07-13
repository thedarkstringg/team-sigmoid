"""
Unit tests for dataset loaders.

Run with:  pytest tests/test_datasets.py -v

Note: load_breast_cancer_data() needs no network and is tested for real.
load_mnist_binary_data() and load_covertype_data() call sklearn's
fetch_openml/fetch_covtype, which download real data over the network --
those are mocked here so the test suite runs fast and offline. This
verifies the loaders' internal logic (subsetting, relabeling, subsampling)
is correct; it does NOT verify the network call itself. You already
confirmed the real network calls work by running them manually.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.utils.datasets import (
    load_breast_cancer_data,
    load_covertype_data,
    load_credit_card_fraud_data,
    load_mnist_binary_data,
)


# ----------------------------------------------------------------------
# Breast Cancer (real, no network needed)
# ----------------------------------------------------------------------
class TestBreastCancerLoader:
    def test_shapes_and_name(self):
        X, y, name = load_breast_cancer_data()
        assert X.shape == (569, 30)
        assert y.shape == (569,)
        assert name == "breast_cancer"

    def test_binary_labels(self):
        X, y, name = load_breast_cancer_data()
        assert set(np.unique(y)) == {0, 1}

    def test_no_missing_values(self):
        X, y, name = load_breast_cancer_data()
        assert not np.any(np.isnan(X))


# ----------------------------------------------------------------------
# Credit Card Fraud (uses a small temp CSV fixture, no network needed)
# ----------------------------------------------------------------------
@pytest.fixture
def fake_credit_card_csv():
    """A tiny fake CSV matching the real dataset's column layout
    (Time, V1..V28, Amount, Class), so we don't need the real 284k-row file
    to test the loader's parsing logic."""
    header = "Time,V1,V2,Amount,Class\n"
    rows = [
        "0,-1.2,0.5,100.0,0\n",
        "1,0.3,-0.8,50.0,0\n",
        "2,2.1,1.4,200.0,1\n",
        "3,-0.5,0.2,75.0,0\n",
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(header)
        f.writelines(rows)
        path = f.name

    yield path
    os.unlink(path)


class TestCreditCardFraudLoader:
    def test_missing_file_raises_with_helpful_message(self):
        with pytest.raises(FileNotFoundError, match="kaggle.com"):
            load_credit_card_fraud_data(csv_path="data/definitely_does_not_exist.csv")

    def test_parses_csv_correctly(self, fake_credit_card_csv):
        X, y, name = load_credit_card_fraud_data(csv_path=fake_credit_card_csv)
        assert X.shape == (4, 4)  # Time, V1, V2, Amount (Class excluded)
        assert y.shape == (4,)
        assert name == "credit_card_fraud"
        assert list(y) == [0, 0, 1, 0]

    def test_class_column_correctly_separated_from_features(self, fake_credit_card_csv):
        X, y, name = load_credit_card_fraud_data(csv_path=fake_credit_card_csv)
        # last feature column should be Amount, not Class
        assert np.allclose(X[:, -1], [100.0, 50.0, 200.0, 75.0])


# ----------------------------------------------------------------------
# MNIST binary subset (mocked network call)
# ----------------------------------------------------------------------
@pytest.fixture
def fake_mnist_bunch():
    rng = np.random.RandomState(0)
    fake_X = rng.rand(2000, 784)
    fake_y = rng.choice([str(d) for d in range(10)], size=2000)
    bunch = MagicMock()
    bunch.data = fake_X
    bunch.target = fake_y
    return bunch


class TestMnistLoader:
    def test_output_is_high_dimensional(self, fake_mnist_bunch):
        with patch("sklearn.datasets.fetch_openml", return_value=fake_mnist_bunch):
            X, y, name = load_mnist_binary_data(digit_a=3, digit_b=8, n_samples=100, random_state=42)
            assert X.shape[1] == 784

    def test_only_requested_digits_included(self, fake_mnist_bunch):
        with patch("sklearn.datasets.fetch_openml", return_value=fake_mnist_bunch):
            X, y, name = load_mnist_binary_data(digit_a=3, digit_b=8, n_samples=1000, random_state=42)
            # relabeled to {0, 1}, so we can't check original digit values directly,
            # but the count should match how many 3s/8s existed in the fake data
            original_mask = (fake_mnist_bunch.target.astype(int) == 3) | (fake_mnist_bunch.target.astype(int) == 8)
            expected_available = original_mask.sum()
            assert X.shape[0] == min(1000, expected_available)

    def test_labels_relabeled_to_binary(self, fake_mnist_bunch):
        with patch("sklearn.datasets.fetch_openml", return_value=fake_mnist_bunch):
            X, y, name = load_mnist_binary_data(digit_a=3, digit_b=8, n_samples=100, random_state=42)
            assert set(np.unique(y)).issubset({0, 1})

    def test_subsampling_respects_n_samples_limit(self, fake_mnist_bunch):
        with patch("sklearn.datasets.fetch_openml", return_value=fake_mnist_bunch):
            X, y, name = load_mnist_binary_data(digit_a=3, digit_b=8, n_samples=50, random_state=42)
            assert X.shape[0] <= 50

    def test_name_reflects_chosen_digits(self, fake_mnist_bunch):
        with patch("sklearn.datasets.fetch_openml", return_value=fake_mnist_bunch):
            X, y, name = load_mnist_binary_data(digit_a=1, digit_b=7, n_samples=50, random_state=42)
            assert name == "mnist_1_vs_7"

    def test_reproducible_with_same_seed(self, fake_mnist_bunch):
        with patch("sklearn.datasets.fetch_openml", return_value=fake_mnist_bunch):
            X1, y1, _ = load_mnist_binary_data(digit_a=3, digit_b=8, n_samples=100, random_state=42)
            X2, y2, _ = load_mnist_binary_data(digit_a=3, digit_b=8, n_samples=100, random_state=42)
            assert np.array_equal(X1, X2)
            assert np.array_equal(y1, y2)


# ----------------------------------------------------------------------
# Covertype (mocked network call)
# ----------------------------------------------------------------------
@pytest.fixture
def fake_covtype_bunch():
    rng = np.random.RandomState(0)
    fake_X = rng.rand(5000, 54)
    fake_y = rng.randint(1, 8, size=5000)  # 7 forest cover classes, labeled 1-7
    bunch = MagicMock()
    bunch.data = fake_X
    bunch.target = fake_y
    return bunch


class TestCovertypeLoader:
    def test_output_shape(self, fake_covtype_bunch):
        with patch("sklearn.datasets.fetch_covtype", return_value=fake_covtype_bunch):
            X, y, name = load_covertype_data(n_samples=1000, random_state=42)
            assert X.shape == (1000, 54)
            assert y.shape == (1000,)
            assert name == "covertype"

    def test_multiclass_labels_preserved(self, fake_covtype_bunch):
        with patch("sklearn.datasets.fetch_covtype", return_value=fake_covtype_bunch):
            X, y, name = load_covertype_data(n_samples=2000, random_state=42)
            assert len(np.unique(y)) > 2  # genuinely multi-class, not binarized

    def test_subsampling_respects_n_samples(self, fake_covtype_bunch):
        with patch("sklearn.datasets.fetch_covtype", return_value=fake_covtype_bunch):
            X, y, name = load_covertype_data(n_samples=500, random_state=42)
            assert X.shape[0] == 500

    def test_reproducible_with_same_seed(self, fake_covtype_bunch):
        with patch("sklearn.datasets.fetch_covtype", return_value=fake_covtype_bunch):
            X1, y1, _ = load_covertype_data(n_samples=500, random_state=42)
            X2, y2, _ = load_covertype_data(n_samples=500, random_state=42)
            assert np.array_equal(X1, X2)
            assert np.array_equal(y1, y2)