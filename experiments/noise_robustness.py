"""
Experiment 5: Noise robustness.

For each dataset, randomly flip a fraction eta in {0, 0.05, 0.10, 0.20} of
TRAINING labels (eta=0 included as a clean baseline for comparison), train
AdaBoost (100 stumps) and Random Forest (100 trees) on the corrupted
training data, and evaluate both on the CLEAN, untouched test set. Plot
accuracy degradation curves vs eta.

Label flipping: for each selected training sample, its label is replaced
with a uniformly random DIFFERENT class (not just "any" class, which could
trivially reassign the same label and understate the intended noise level).

SCALING / MULTICLASS / DATASET SIZE: same approach as Experiment 4 --
tree-based models are scale-invariant so scaling is skipped; Covertype
uses OneVsRestAdaBoost; datasets are capped to 5,000 rows with a minimum
of 50 samples per class (see subsample_for_experiments), for the same
CV-metric-stability reasons documented in Experiment 4.

Run standalone:  python experiments/noise_robustness.py
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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from experiments.common import get_available_datasets, is_binary, save_figure, save_results_json, subsample_for_experiments
from src.bagging.random_forest import RandomForestClassifier
from src.boosting.adaboost import AdaBoostClassifier
from src.boosting.one_vs_rest_adaboost import OneVsRestAdaBoost
from src.trees.decision_tree import DecisionTree

MAX_SAMPLES = 5000
N_ESTIMATORS = 100
ETA_VALUES = [0.0, 0.05, 0.10, 0.20]


def flip_labels(y: np.ndarray, eta: float, random_state: int = 42) -> np.ndarray:
    """
    Randomly flip a fraction eta of labels in y, each to a uniformly
    random DIFFERENT class (never re-assigning the same label, which
    would silently understate the intended noise level).

    eta=0 returns y unchanged (copy).
    """
    y = np.asarray(y).copy()
    if eta <= 0:
        return y

    rng = np.random.RandomState(random_state)
    classes = np.unique(y)
    n_samples = len(y)
    n_flip = int(round(eta * n_samples))

    flip_indices = rng.choice(n_samples, size=n_flip, replace=False)
    for idx in flip_indices:
        other_classes = classes[classes != y[idx]]
        y[idx] = rng.choice(other_classes)

    return y


def run_experiment_5(datasets: dict, random_state: int = 42) -> dict:
    all_results = {}

    for dataset_name, (X, y) in datasets.items():
        print(f"\n{'='*60}\nExperiment 5 -- {dataset_name}\n{'='*60}")

        original_n = X.shape[0]
        X, y = subsample_for_experiments(
            X, y, max_samples=MAX_SAMPLES, random_state=random_state, min_class_samples=50
        )
        if X.shape[0] < original_n:
            print(f"NOTE: subsampled to {X.shape[0]} rows (stratified, min 50/class) for compute tractability.")

        adaboost_cls = AdaBoostClassifier if is_binary(y) else OneVsRestAdaBoost
        if not is_binary(y):
            print("NOTE: multiclass dataset -- using OneVsRestAdaBoost instead of AdaBoostClassifier.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )

        adaboost_acc = []
        rf_acc = []

        for eta in ETA_VALUES:
            y_train_noisy = flip_labels(y_train, eta, random_state=random_state)
            actual_flip_rate = np.mean(y_train_noisy != y_train)

            ada = adaboost_cls(n_estimators=N_ESTIMATORS, random_state=random_state)
            ada.fit(X_train, y_train_noisy)
            ada_test_acc = accuracy_score(y_test, ada.predict(X_test))
            adaboost_acc.append(ada_test_acc)

            rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=None, random_state=random_state)
            rf.fit(X_train, y_train_noisy)
            rf_test_acc = accuracy_score(y_test, rf.predict(X_test))
            rf_acc.append(rf_test_acc)

            print(f"  eta={eta:.2f} (actual flip rate {actual_flip_rate:.3f}): "
                  f"AdaBoost test_acc={ada_test_acc:.4f}, RF test_acc={rf_test_acc:.4f}")

        ada_degradation = adaboost_acc[0] - adaboost_acc[-1]
        rf_degradation = rf_acc[0] - rf_acc[-1]
        more_sensitive = "AdaBoost" if ada_degradation > rf_degradation else "Random Forest"
        print(f"  Accuracy drop from eta=0 to eta={ETA_VALUES[-1]}: "
              f"AdaBoost={ada_degradation:.4f}, RandomForest={rf_degradation:.4f} "
              f"-- {more_sensitive} degrades more on this dataset.")

        # --- plot ---
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(ETA_VALUES, adaboost_acc, marker="o", label="AdaBoost")
        ax.plot(ETA_VALUES, rf_acc, marker="s", label="Random Forest")
        ax.set_xlabel("Label noise fraction (eta)")
        ax.set_ylabel("Test accuracy (clean test set)")
        ax.set_title(f"Noise robustness: {dataset_name}")
        ax.legend()
        ax.grid(alpha=0.3)
        save_figure(fig, f"experiment_5_noise_robustness_{dataset_name}")
        plt.close(fig)

        all_results[dataset_name] = {
            "eta_values": ETA_VALUES,
            "adaboost_accuracy": adaboost_acc,
            "random_forest_accuracy": rf_acc,
            "adaboost_degradation": ada_degradation,
            "random_forest_degradation": rf_degradation,
            "more_sensitive_model": more_sensitive,
        }

    save_results_json("experiment_5_noise_robustness", all_results)
    return all_results


if __name__ == "__main__":
    datasets = get_available_datasets()
    run_experiment_5(datasets)