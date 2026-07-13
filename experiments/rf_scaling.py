"""
Experiment 3: Random Forest scaling.

(a) Fix max_depth=None, vary n_estimators from 1 to 100. Plot test accuracy
    (and OOB accuracy, since oob_score=True is used).

    NOTE: the brief suggests sweeping n_estimators 1-200; this was reduced
    to 1-100 after benchmarking showed a single 200-tree unpruned fit takes
    ~150s+ at 10,000 rows in the worst case (pure-noise data forces trees
    to grow to full depth). 100 estimators is enough to see the accuracy
    curve flatten out, at roughly half the compute cost -- document this
    as a compute-tractability decision in the report.

    EFFICIENCY NOTE: RandomForestClassifier has no staged_predict, but
    trees in a forest are trained independently of each other -- so unlike
    a fresh fit per n_estimators value, we fit ONCE at n_estimators=100 and
    then read off "test/OOB accuracy using only the first k trees" for every
    k we care about, by averaging predict_proba over rf.estimators_[:k] and
    computing OOB accuracy from rf.oob_masks_[:k]. This is mathematically
    identical to fitting k separate forests (same bootstrap seeds produce
    the same first k trees), and turns a ~10-refit sweep into a single fit.

(b) Fix n_estimators=50, vary max_depth over a representative subset
    (1, 3, 5, 7, 10, 13, 16, 20) rather than every integer 1-20 -- still
    shows the underfitting -> overfitting shape at a fraction of the
    compute cost. Plot test accuracy.
    No shortcut is available here -- a tree fit with max_depth=5 differs
    structurally from a max_depth=20 tree at every node, not just a
    truncation of it, so each depth genuinely needs its own fit.

DATASET SIZE NOTE: datasets are capped to 5,000 rows (stratified) for this
experiment specifically -- Random Forest is considerably more expensive
than a single tree or AdaBoost's depth-1 stumps, so a smaller cap than
Experiment 2's 20,000 is used here. See subsample_for_experiments().

PARALLELISM NOTE: n_jobs>1 was benchmarked and found to be SLOWER than
n_jobs=1 at the data sizes used here -- the current multiprocessing.Pool
implementation copies the full X, y arrays into every per-tree job, and
that copy overhead outweighs the benefit of parallel tree training at this
scale. This experiment therefore runs sequentially (n_jobs=1); the
parallel-vs-sequential comparison itself is worth reporting as a finding
in the "Parallelization" section of the report.

Run standalone:  python experiments/rf_scaling.py
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

from experiments.common import get_available_datasets, save_figure, save_results_json, subsample_for_experiments
from src.bagging.random_forest import RandomForestClassifier
from src.utils.preprocessing import fit_scaler, scale_data

MAX_SAMPLES = 5000
N_ESTIMATORS_MAX = 100
N_ESTIMATORS_STEP = 10
N_ESTIMATORS_FIXED = 50
MAX_DEPTH_SWEEP = [1, 3, 5, 7, 10, 13, 16, 20]


def _predict_with_first_k_trees(rf: RandomForestClassifier, X: np.ndarray, k: int) -> np.ndarray:
    """Average predict_proba over only the first k trees, then argmax --
    equivalent to a forest fit with n_estimators=k."""
    proba_sum = np.zeros((X.shape[0], rf.n_classes_))
    for tree in rf.estimators_[:k]:
        proba_sum += tree.predict_proba(X)
    proba = proba_sum / k
    return rf.classes_[np.argmax(proba, axis=1)]


def _oob_accuracy_with_first_k_trees(rf: RandomForestClassifier, X: np.ndarray, y: np.ndarray, k: int) -> float:
    """OOB accuracy using only the first k trees' OOB masks -- mirrors
    RandomForestClassifier._compute_oob_score but restricted to a prefix."""
    class_to_idx = {c: i for i, c in enumerate(rf.classes_)}
    n_samples = X.shape[0]
    vote_counts = np.zeros((n_samples, rf.n_classes_))
    any_oob = np.zeros(n_samples, dtype=bool)

    for tree, mask in zip(rf.estimators_[:k], rf.oob_masks_[:k]):
        if not np.any(mask):
            continue
        idx = np.where(mask)[0]
        preds = tree.predict(X[idx])
        for i, p in zip(idx, preds):
            vote_counts[i, class_to_idx[p]] += 1
            any_oob[i] = True

    if not np.any(any_oob):
        return float("nan")

    scored = np.where(any_oob)[0]
    oob_preds = rf.classes_[np.argmax(vote_counts[scored], axis=1)]
    return float(np.mean(oob_preds == y[scored]))


def run_experiment_3(datasets: dict, random_state: int = 42) -> dict:
    all_results = {}

    for dataset_name, (X, y) in datasets.items():
        print(f"\n{'='*60}\nExperiment 3 -- {dataset_name}\n{'='*60}")

        original_n = X.shape[0]
        X, y = subsample_for_experiments(X, y, max_samples=MAX_SAMPLES, random_state=random_state)
        if X.shape[0] < original_n:
            print(f"NOTE: subsampled to {X.shape[0]} rows (stratified) for compute tractability.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )
        scaler = fit_scaler(X_train)
        X_train_scaled = scale_data(scaler, X_train)
        X_test_scaled = scale_data(scaler, X_test)

        # ---------------- (a) n_estimators sweep, max_depth=None ----------------
        print(f"(a) n_estimators sweep (max_depth=None)...")
        rf_full = RandomForestClassifier(
            n_estimators=N_ESTIMATORS_MAX, max_depth=None, max_features="sqrt",
            bootstrap=True, oob_score=True, n_jobs=1, random_state=random_state,
        )
        rf_full.fit(X_train_scaled, y_train)

        k_values = sorted(set([1] + list(range(N_ESTIMATORS_STEP, N_ESTIMATORS_MAX + 1, N_ESTIMATORS_STEP))))
        test_acc_by_k = []
        oob_acc_by_k = []
        for k in k_values:
            preds = _predict_with_first_k_trees(rf_full, X_test_scaled, k)
            test_acc_by_k.append(accuracy_score(y_test, preds))
            oob_acc_by_k.append(_oob_accuracy_with_first_k_trees(rf_full, X_train_scaled, y_train, k))

        print(f"  n_estimators={k_values[-1]}: test_acc={test_acc_by_k[-1]:.4f}, oob_acc={oob_acc_by_k[-1]:.4f}")

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(k_values, test_acc_by_k, marker="o", label="Test accuracy")
        ax.plot(k_values, oob_acc_by_k, marker="s", label="OOB accuracy")
        ax.set_xlabel("Number of estimators")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"RF n_estimators scaling: {dataset_name}")
        ax.legend()
        ax.grid(alpha=0.3)
        save_figure(fig, f"experiment_3a_rf_n_estimators_{dataset_name}")
        plt.close(fig)

        # ---------------- (b) max_depth sweep, n_estimators=100 ----------------
        print(f"(b) max_depth sweep (n_estimators={N_ESTIMATORS_FIXED})...")
        test_acc_by_depth = []
        for depth in MAX_DEPTH_SWEEP:
            rf_depth = RandomForestClassifier(
                n_estimators=N_ESTIMATORS_FIXED, max_depth=depth, max_features="sqrt",
                bootstrap=True, oob_score=False, n_jobs=1, random_state=random_state,
            )
            rf_depth.fit(X_train_scaled, y_train)
            preds = rf_depth.predict(X_test_scaled)
            acc = accuracy_score(y_test, preds)
            test_acc_by_depth.append(acc)
            print(f"  max_depth={depth}: test_acc={acc:.4f}")

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(MAX_DEPTH_SWEEP, test_acc_by_depth, marker="o", color="green")
        ax.set_xlabel("max_depth")
        ax.set_ylabel("Test accuracy")
        ax.set_title(f"RF max_depth scaling: {dataset_name}")
        ax.grid(alpha=0.3)
        save_figure(fig, f"experiment_3b_rf_max_depth_{dataset_name}")
        plt.close(fig)

        all_results[dataset_name] = {
            "n_estimators_sweep": {
                "k_values": k_values,
                "test_accuracy": test_acc_by_k,
                "oob_accuracy": oob_acc_by_k,
            },
            "max_depth_sweep": {
                "depths": MAX_DEPTH_SWEEP,
                "test_accuracy": test_acc_by_depth,
            },
        }

    save_results_json("experiment_3_rf_scaling", all_results)
    return all_results


if __name__ == "__main__":
    datasets = get_available_datasets()
    run_experiment_3(datasets)
