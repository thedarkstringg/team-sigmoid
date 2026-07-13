"""
Experiment 2: AdaBoost scaling.

For each BINARY dataset (AdaBoostClassifier in this project is binary-only --
Covertype is excluded here; see the note below):
  - Fit ONE AdaBoostClassifier with n_estimators=200.
  - Use staged_predict to read off train/test accuracy after every round,
    at step intervals of 10 (plus round 1), WITHOUT refitting from scratch
    each time -- staged predictions at round m are identical whether you
    fit with n_estimators=m or n_estimators=200 and read off stage m,
    since AdaBoost's sample-weight evolution up to round m never depends
    on rounds after m.
  - Plot train error and test error vs. number of estimators.
  - Report the round at which test error is minimized (a proxy for "does
    AdaBoost overfit, and when").

NOTE ON SCOPE: AdaBoostClassifier only supports binary classification (per
the brief). OneVsRestAdaBoost handles Covertype's multiclass case for
Experiment 1's baseline, but does not have staged_predict (each of its K
internal binary sub-classifiers would need its own separate curve, which
doesn't reduce to one meaningful "training curve" for the ensemble as a
whole). Covertype is therefore excluded from this specific experiment --
document this as a scope decision in the report rather than a gap.

Run standalone:  python experiments/adaboost_scaling.py
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use("Agg")  # no display needed, just save PNGs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from experiments.common import get_available_datasets, is_binary, save_figure, save_results_json, subsample_for_experiments
from src.boosting.adaboost import AdaBoostClassifier
from src.utils.preprocessing import fit_scaler, scale_data

N_ESTIMATORS_MAX = 200
STEP = 10


def run_experiment_2(datasets: dict, random_state: int = 42) -> dict:
    all_results = {}

    for dataset_name, (X, y) in datasets.items():
        if not is_binary(y):
            print(f"\nExperiment 2 -- {dataset_name}: SKIPPED (multiclass; "
                  f"AdaBoostClassifier is binary-only in this project, see "
                  f"module docstring for the documented scope decision)")
            continue

        print(f"\n{'='*60}\nExperiment 2 -- {dataset_name}\n{'='*60}")

        X, y = subsample_for_experiments(X, y, max_samples=20000, random_state=random_state)
        if X.shape[0] < len(datasets[dataset_name][0]):
            print(f"NOTE: subsampled to {X.shape[0]} rows (stratified) for compute "
                  f"tractability -- see subsample_for_experiments() docstring.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )
        scaler = fit_scaler(X_train)
        X_train_scaled = scale_data(scaler, X_train)
        X_test_scaled = scale_data(scaler, X_test)

        ada = AdaBoostClassifier(n_estimators=N_ESTIMATORS_MAX, random_state=random_state)
        ada.fit(X_train_scaled, y_train)

        actual_n_fit = len(ada.estimators_)
        if actual_n_fit < N_ESTIMATORS_MAX:
            print(f"NOTE: AdaBoost stopped early at round {actual_n_fit} "
                  f"(a weak learner's error reached >= 0.5) -- curves below "
                  f"only go up to that round.")

        # rounds to report: 1, then every STEP-th round, capped at what was actually fit
        rounds_to_report = sorted(set([1] + list(range(STEP, actual_n_fit + 1, STEP)) + [actual_n_fit]))

        train_staged = list(ada.staged_predict(X_train_scaled))
        test_staged = list(ada.staged_predict(X_test_scaled))

        train_errors = []
        test_errors = []
        for r in rounds_to_report:
            train_acc = accuracy_score(y_train, train_staged[r - 1])
            test_acc = accuracy_score(y_test, test_staged[r - 1])
            train_errors.append(1 - train_acc)
            test_errors.append(1 - test_acc)

        best_round = rounds_to_report[int(np.argmin(test_errors))]
        best_test_error = min(test_errors)

        print(f"Fitted {actual_n_fit} estimators. "
              f"Test error minimized at round {best_round} (test_error={best_test_error:.4f}). "
              f"Final round ({rounds_to_report[-1]}): "
              f"train_error={train_errors[-1]:.4f}, test_error={test_errors[-1]:.4f}")

        overfitting = test_errors[-1] > best_test_error + 0.005  # small tolerance
        if overfitting:
            print(f"Test error increased after round {best_round} -- AdaBoost shows signs of overfitting "
                  f"on {dataset_name} beyond that point.")
        else:
            print(f"Test error did not meaningfully increase -- no clear overfitting "
                  f"observed within {N_ESTIMATORS_MAX} rounds on {dataset_name}.")

        # --- plot ---
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(rounds_to_report, train_errors, marker="o", label="Train error")
        ax.plot(rounds_to_report, test_errors, marker="s", label="Test error")
        ax.axvline(best_round, color="gray", linestyle="--", alpha=0.6, label=f"Best round ({best_round})")
        ax.set_xlabel("Number of estimators")
        ax.set_ylabel("Error rate (1 - accuracy)")
        ax.set_title(f"AdaBoost scaling: {dataset_name}")
        ax.legend()
        ax.grid(alpha=0.3)
        save_figure(fig, f"experiment_2_adaboost_scaling_{dataset_name}")
        plt.close(fig)

        all_results[dataset_name] = {
            "rounds": rounds_to_report,
            "train_error": train_errors,
            "test_error": test_errors,
            "best_round": best_round,
            "best_test_error": best_test_error,
            "n_estimators_fit": actual_n_fit,
            "shows_overfitting": overfitting,
        }

    save_results_json("experiment_2_adaboost_scaling", all_results)
    return all_results


if __name__ == "__main__":
    datasets = get_available_datasets()
    run_experiment_2(datasets)