"""
Evaluation utilities: classification metrics and a cross-validation helper
usable across DecisionTree, AdaBoostClassifier, and RandomForestClassifier
(or any object exposing fit/predict/predict_proba).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict:
    """
    Compute accuracy, macro-F1, and AUC-ROC (binary) / macro-averaged AUC
    (multi-class) for a single set of predictions.

    y_proba: predict_proba output, shape [n_samples, n_classes]. Required
    for AUC; if omitted, AUC is reported as None.
    """
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }

    if y_proba is not None:
        n_classes = y_proba.shape[1]
        try:
            if n_classes == 2:
                # sklearn expects the probability of the positive (second) class
                results["auc_roc"] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                results["auc_roc"] = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="macro"
                )
        except ValueError:
            # e.g. a class missing entirely from y_true in a small CV fold
            results["auc_roc"] = None
    else:
        results["auc_roc"] = None

    return results


def cross_validate_model(
    model_factory,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int | None = 42,
    preprocess_train_fn=None,
) -> dict:
    """
    Stratified K-fold cross-validation for any classifier exposing
    fit/predict/predict_proba (DecisionTree, AdaBoostClassifier,
    RandomForestClassifier, or an sklearn estimator all work).

    model_factory: a zero-argument callable that returns a FRESH, UNFIT
    model instance -- e.g. `lambda: DecisionTree(max_depth=5, random_state=42)`.
    A fresh instance per fold is required; reusing one fit model across
    folds would leak information between folds.

    preprocess_train_fn: optional callable (X_train, y_train) -> (X_train, y_train),
    applied ONLY to each fold's TRAINING partition, never the test partition.
    This is how imbalance treatment (SMOTE, random oversampling) should be
    applied inside CV -- e.g.
    `lambda X, y: handle_imbalance(X, y, method="smote", random_state=42)`.
    Applying it before the train/test split instead (to the whole dataset)
    would leak information: SMOTE's synthetic minority points are
    interpolated from real neighbors, and if some of those neighbors end up
    in the test fold, the test fold is no longer truly held out.

    Returns a dict of {metric_name: [scores per fold]}, plus
    {metric_name + "_mean"} and {metric_name + "_std"} summary keys --
    directly usable for the "mean +/- std, box plots or tables" requirement
    in Experiment 4 (head-to-head comparison).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    per_fold = {"accuracy": [], "f1_macro": [], "auc_roc": []}

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if preprocess_train_fn is not None:
            X_train, y_train = preprocess_train_fn(X_train, y_train)

        model = model_factory()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        fold_results = evaluate_predictions(y_test, y_pred, y_proba)
        for key in per_fold:
            per_fold[key].append(fold_results[key])

    summary = dict(per_fold)
    for key in per_fold:
        values = [v for v in per_fold[key] if v is not None]
        summary[f"{key}_mean"] = float(np.mean(values)) if values else None
        summary[f"{key}_std"] = float(np.std(values)) if values else None

    return summary