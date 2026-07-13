"""
Unit tests for evaluation utilities: metrics and cross-validation.

Run with:  pytest tests/test_evaluation.py -v
"""

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, make_classification

from src.metrics.evaluation import cross_validate_model, evaluate_predictions
from src.trees.decision_tree import DecisionTree


# ----------------------------------------------------------------------
# evaluate_predictions
# ----------------------------------------------------------------------
class TestEvaluatePredictions:
    def test_perfect_predictions_give_perfect_scores(self):
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 1])
        results = evaluate_predictions(y_true, y_pred)
        assert results["accuracy"] == 1.0
        assert results["f1_macro"] == 1.0

    def test_all_wrong_predictions_give_zero_accuracy(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0])
        results = evaluate_predictions(y_true, y_pred)
        assert results["accuracy"] == 0.0

    def test_auc_none_without_proba(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        results = evaluate_predictions(y_true, y_pred)
        assert results["auc_roc"] is None

    def test_binary_auc_computed_with_proba(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        # perfect separation: class 0 gets low prob of class 1, class 1 gets high
        y_proba = np.array([
            [0.9, 0.1],
            [0.1, 0.9],
            [0.8, 0.2],
            [0.2, 0.8],
        ])
        results = evaluate_predictions(y_true, y_pred, y_proba)
        assert results["auc_roc"] == 1.0

    def test_multiclass_auc_computed_with_proba(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        y_proba = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ])
        results = evaluate_predictions(y_true, y_pred, y_proba)
        assert results["auc_roc"] is not None
        assert 0.0 <= results["auc_roc"] <= 1.0

    def test_returns_all_three_keys(self):
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        results = evaluate_predictions(y_true, y_pred)
        assert set(results.keys()) == {"accuracy", "f1_macro", "auc_roc"}


# ----------------------------------------------------------------------
# cross_validate_model
# ----------------------------------------------------------------------
class TestCrossValidateModel:
    def test_correct_number_of_folds(self):
        X, y = load_breast_cancer(return_X_y=True)
        results = cross_validate_model(
            lambda: DecisionTree(max_depth=5, random_state=42),
            X, y, n_splits=5, random_state=42,
        )
        assert len(results["accuracy"]) == 5
        assert len(results["f1_macro"]) == 5

    def test_summary_stats_present(self):
        X, y = load_breast_cancer(return_X_y=True)
        results = cross_validate_model(
            lambda: DecisionTree(max_depth=5, random_state=42),
            X, y, n_splits=3, random_state=42,
        )
        assert "accuracy_mean" in results
        assert "accuracy_std" in results
        assert results["accuracy_mean"] == pytest.approx(np.mean(results["accuracy"]))

    def test_reasonable_accuracy_on_breast_cancer(self):
        X, y = load_breast_cancer(return_X_y=True)
        results = cross_validate_model(
            lambda: DecisionTree(max_depth=5, random_state=42),
            X, y, n_splits=5, random_state=42,
        )
        assert results["accuracy_mean"] > 0.85

    def test_fresh_model_instance_per_fold(self):
        """model_factory must be called fresh each fold -- verify the factory
        is invoked exactly n_splits times, not reused."""
        X, y = load_breast_cancer(return_X_y=True)
        call_count = {"n": 0}

        def counting_factory():
            call_count["n"] += 1
            return DecisionTree(max_depth=5, random_state=42)

        cross_validate_model(counting_factory, X, y, n_splits=4, random_state=42)
        assert call_count["n"] == 4

    def test_reproducible_with_same_seed(self):
        X, y = load_breast_cancer(return_X_y=True)
        results1 = cross_validate_model(
            lambda: DecisionTree(max_depth=5, random_state=42),
            X, y, n_splits=5, random_state=7,
        )
        results2 = cross_validate_model(
            lambda: DecisionTree(max_depth=5, random_state=42),
            X, y, n_splits=5, random_state=7,
        )
        assert results1["accuracy"] == results2["accuracy"]

    def test_works_with_multiclass_data(self):
        X, y = make_classification(
            n_samples=300, n_classes=3, n_informative=6, n_clusters_per_class=1, random_state=42
        )
        results = cross_validate_model(
            lambda: DecisionTree(max_depth=5, random_state=42),
            X, y, n_splits=3, random_state=42,
        )
        assert len(results["accuracy"]) == 3
        assert results["accuracy_mean"] is not None


# ----------------------------------------------------------------------
# preprocess_train_fn hook (used for fold-safe imbalance treatment)
# ----------------------------------------------------------------------
class TestPreprocessTrainFn:
    def test_hook_is_called_once_per_fold(self):
        X, y = load_breast_cancer(return_X_y=True)
        call_count = {"n": 0}

        def counting_preprocess(X_train, y_train):
            call_count["n"] += 1
            return X_train, y_train

        cross_validate_model(
            lambda: DecisionTree(max_depth=5, random_state=42),
            X, y, n_splits=5, random_state=42,
            preprocess_train_fn=counting_preprocess,
        )
        assert call_count["n"] == 5

    def test_hook_can_change_training_set_size(self):
        """Simulates oversampling: hook doubles the training set. Should
        not error and should still produce valid per-fold results."""
        X, y = load_breast_cancer(return_X_y=True)

        def duplicate_data(X_train, y_train):
            return np.vstack([X_train, X_train]), np.concatenate([y_train, y_train])

        results = cross_validate_model(
            lambda: DecisionTree(max_depth=5, random_state=42),
            X, y, n_splits=5, random_state=42,
            preprocess_train_fn=duplicate_data,
        )
        assert len(results["accuracy"]) == 5
        assert all(0.0 <= a <= 1.0 for a in results["accuracy"])

    def test_hook_never_touches_test_fold(self):
        """If the hook corrupted test data, accuracy would collapse to ~0
        for a trivial linearly separable problem. Verify it stays high,
        confirming the hook only sees the training partition."""
        X, y = make_classification(n_samples=200, n_classes=2, random_state=42, class_sep=3.0)

        def poison_if_test_leaked(X_train, y_train):
            # if this ever received test data too, dataset size would be
            # larger than expected for an 80% train fold
            assert X_train.shape[0] < 200
            return X_train, y_train

        results = cross_validate_model(
            lambda: DecisionTree(max_depth=5, random_state=42),
            X, y, n_splits=5, random_state=42,
            preprocess_train_fn=poison_if_test_leaked,
        )
        assert results["accuracy_mean"] > 0.9

    def test_no_hook_is_still_default_behavior(self):
        X, y = load_breast_cancer(return_X_y=True)
        results_no_hook = cross_validate_model(
            lambda: DecisionTree(max_depth=5, random_state=42),
            X, y, n_splits=5, random_state=42,
        )
        results_identity_hook = cross_validate_model(
            lambda: DecisionTree(max_depth=5, random_state=42),
            X, y, n_splits=5, random_state=42,
            preprocess_train_fn=lambda X, y: (X, y),
        )
        assert results_no_hook["accuracy"] == results_identity_hook["accuracy"]