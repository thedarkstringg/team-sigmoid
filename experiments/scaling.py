"""
Experiment 1: Baseline.

For each dataset:
  - Train a single unpruned DecisionTree (80/20 split).
  - Train a DecisionStump (depth-1).
  - Train sklearn's DecisionTreeClassifier with identical parameters.
  - Report accuracy, macro-F1, and AUC-ROC (binary) / macro AUC (multi-class).
  - Verify the from-scratch tree matches sklearn within 2%.

Run standalone:  python experiments/scaling.py
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree

from experiments.common import get_available_datasets, save_results_json
from src.metrics.evaluation import evaluate_predictions
from src.trees.decision_tree import DecisionStump, DecisionTree
from src.utils.preprocessing import fit_scaler, scale_data


def run_experiment_1(datasets: dict, random_state: int = 42) -> dict:
    """
    datasets: {name: (X, y)}, as returned by experiments.common.get_available_datasets()
    Returns a results dict, one entry per dataset, and also saves it to
    results/experiment_1_baseline.json.
    """
    all_results = {}

    for dataset_name, (X, y) in datasets.items():
        print(f"\n{'='*60}\nExperiment 1 -- {dataset_name}\n{'='*60}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )

        scaler = fit_scaler(X_train)
        X_train_scaled = scale_data(scaler, X_train)
        X_test_scaled = scale_data(scaler, X_test)

        # --- unpruned DecisionTree (my implementation) ---
        my_tree = DecisionTree(max_depth=None, criterion="gini", random_state=random_state)
        my_tree.fit(X_train_scaled, y_train)
        my_tree_preds = my_tree.predict(X_test_scaled)
        my_tree_proba = my_tree.predict_proba(X_test_scaled)
        my_tree_results = evaluate_predictions(y_test, my_tree_preds, my_tree_proba)
        my_tree_results["depth"] = my_tree.depth
        my_tree_results["n_leaves"] = my_tree.n_leaves

        # --- DecisionStump (depth-1) ---
        stump = DecisionStump(criterion="gini", random_state=random_state)
        stump.fit(X_train_scaled, y_train)
        stump_preds = stump.predict(X_test_scaled)
        stump_proba = stump.predict_proba(X_test_scaled)
        stump_results = evaluate_predictions(y_test, stump_preds, stump_proba)

        # --- sklearn DecisionTreeClassifier, identical params ---
        skl_tree = SklearnDecisionTree(max_depth=None, criterion="gini", random_state=random_state)
        skl_tree.fit(X_train_scaled, y_train)
        skl_preds = skl_tree.predict(X_test_scaled)
        skl_proba = skl_tree.predict_proba(X_test_scaled)
        skl_results = evaluate_predictions(y_test, skl_preds, skl_proba)
        skl_results["depth"] = skl_tree.get_depth()
        skl_results["n_leaves"] = skl_tree.get_n_leaves()

        # --- sklearn-agreement check (brief requires <=2% accuracy diff) ---
        acc_diff = abs(my_tree_results["accuracy"] - skl_results["accuracy"])
        within_tolerance = acc_diff <= 0.02

        print(f"My DecisionTree:  acc={my_tree_results['accuracy']:.4f}  "
              f"f1={my_tree_results['f1_macro']:.4f}  "
              f"auc={my_tree_results['auc_roc']}  "
              f"depth={my_tree_results['depth']}  leaves={my_tree_results['n_leaves']}")
        print(f"DecisionStump:    acc={stump_results['accuracy']:.4f}  "
              f"f1={stump_results['f1_macro']:.4f}  auc={stump_results['auc_roc']}")
        print(f"sklearn Tree:     acc={skl_results['accuracy']:.4f}  "
              f"f1={skl_results['f1_macro']:.4f}  "
              f"auc={skl_results['auc_roc']}  "
              f"depth={skl_results['depth']}  leaves={skl_results['n_leaves']}")
        print(f"Accuracy diff vs sklearn: {acc_diff:.4f} "
              f"({'WITHIN' if within_tolerance else 'EXCEEDS'} the 2% tolerance)")

        all_results[dataset_name] = {
            "my_tree": my_tree_results,
            "stump": stump_results,
            "sklearn_tree": skl_results,
            "accuracy_diff_vs_sklearn": acc_diff,
            "within_2_percent_tolerance": within_tolerance,
        }

        if not within_tolerance:
            print(
                f"WARNING: {dataset_name} exceeds the 2% sklearn-agreement "
                f"tolerance required by the brief -- investigate before "
                f"including this in the report."
            )

    save_results_json("experiment_1_baseline", all_results)
    return all_results


if __name__ == "__main__":
    datasets = get_available_datasets()
    run_experiment_1(datasets)