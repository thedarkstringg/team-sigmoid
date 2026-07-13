"""
Experiment 4: Head-to-head comparison (fixed resources).

n_estimators=100 for both ensembles. 5-fold stratified CV. For each fold,
compute accuracy, macro-F1, and AUC-ROC. Report mean +/- std for:
single tree, AdaBoost, Random Forest, and sklearn's RandomForestClassifier
(as reference).

SCALING NOTE: tree-based models (DecisionTree, AdaBoost's stumps, Random
Forest) split on raw per-feature thresholds and are invariant to monotonic
feature scaling -- StandardScaler wouldn't change any split decision or
resulting prediction. Scaling is therefore skipped here entirely (correct
simplification for this model family, not applicable to Experiment 7's
distance-based PCA/K-Means/DBSCAN, which DO need it).

MULTICLASS NOTE: Covertype uses OneVsRestAdaBoost in place of the
binary-only AdaBoostClassifier -- both expose the same fit/predict/
predict_proba interface, so cross_validate_model works unchanged.

DATASET SIZE NOTE: capped to 5,000 rows (stratified), matching Experiment
3 -- benchmarked at ~140s worst-case per dataset (my RandomForest is the
dominant cost at ~110s of that, consistent with Experiment 3's findings).

Run standalone:  python experiments/head_to_head.py
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier as SklearnRF

from experiments.common import get_available_datasets, is_binary, save_figure, save_results_json, subsample_for_experiments
from src.bagging.random_forest import RandomForestClassifier
from src.boosting.adaboost import AdaBoostClassifier
from src.boosting.one_vs_rest_adaboost import OneVsRestAdaBoost
from src.metrics.evaluation import cross_validate_model
from src.trees.decision_tree import DecisionTree
from src.utils.preprocessing import handle_imbalance, minority_class_fraction

MAX_SAMPLES = 5000
N_ESTIMATORS = 100
N_SPLITS = 5
IMBALANCE_THRESHOLD = 0.05  # minority class fraction below this triggers SMOTE


def run_experiment_4(datasets: dict, random_state: int = 42) -> dict:
    all_results = {}

    for dataset_name, (X, y) in datasets.items():
        print(f"\n{'='*60}\nExperiment 4 -- {dataset_name}\n{'='*60}")

        original_n = X.shape[0]
        X, y = subsample_for_experiments(
            X, y, max_samples=MAX_SAMPLES, random_state=random_state, min_class_samples=50
        )
        if X.shape[0] < original_n:
            print(f"NOTE: subsampled to {X.shape[0]} rows (stratified, min 50/class) for compute tractability.")

        adaboost_cls = AdaBoostClassifier if is_binary(y) else OneVsRestAdaBoost
        if not is_binary(y):
            print("NOTE: multiclass dataset -- using OneVsRestAdaBoost instead of AdaBoostClassifier.")

        # --- imbalance treatment (fold-safe: only applied to each fold's training partition) ---
        minority_frac = minority_class_fraction(y)
        preprocess_train_fn = None
        if minority_frac < IMBALANCE_THRESHOLD:
            print(f"NOTE: severe class imbalance detected (minority class = {minority_frac:.4%} of data) "
                  f"-- applying SMOTE to each fold's TRAINING partition only (never the test partition, "
                  f"to avoid leaking synthetic-neighbor information across the train/test boundary).")
            preprocess_train_fn = lambda X_train, y_train: handle_imbalance(
                X_train, y_train, method="smote", random_state=random_state
            )

        model_factories = {
            "single_tree": lambda: DecisionTree(max_depth=None, random_state=random_state),
            "adaboost": lambda: adaboost_cls(n_estimators=N_ESTIMATORS, random_state=random_state),
            "random_forest": lambda: RandomForestClassifier(
                n_estimators=N_ESTIMATORS, max_depth=None, random_state=random_state
            ),
            "sklearn_rf_reference": lambda: SklearnRF(
                n_estimators=N_ESTIMATORS, max_depth=None, random_state=random_state
            ),
        }

        dataset_results = {}
        accuracy_by_model = {}

        for model_name, factory in model_factories.items():
            cv_results = cross_validate_model(
                factory, X, y, n_splits=N_SPLITS, random_state=random_state,
                preprocess_train_fn=preprocess_train_fn,
            )
            dataset_results[model_name] = cv_results
            accuracy_by_model[model_name] = cv_results["accuracy"]

            print(f"  {model_name:22s}  "
                  f"acc={cv_results['accuracy_mean']:.4f}+/-{cv_results['accuracy_std']:.4f}  "
                  f"f1={cv_results['f1_macro_mean']:.4f}+/-{cv_results['f1_macro_std']:.4f}  "
                  f"auc={cv_results['auc_roc_mean']}")

        # --- box plot of per-fold accuracy across the 4 models ---
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = list(accuracy_by_model.keys())
        data = [accuracy_by_model[m] for m in labels]
        ax.boxplot(data, tick_labels=labels)
        ax.set_ylabel("Accuracy (5-fold CV)")
        ax.set_title(f"Head-to-head comparison: {dataset_name}")
        plt.xticks(rotation=15)
        ax.grid(alpha=0.3, axis="y")
        save_figure(fig, f"experiment_4_head_to_head_{dataset_name}")
        plt.close(fig)

        all_results[dataset_name] = dataset_results

    save_results_json("experiment_4_head_to_head", all_results)
    return all_results


if __name__ == "__main__":
    datasets = get_available_datasets()
    run_experiment_4(datasets)