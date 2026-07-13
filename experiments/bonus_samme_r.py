"""
BONUS (+2 pts): SAMME.R vs OneVsRestAdaBoost on Covertype.

Compares three configurations on the (multiclass, 7-class) Covertype
dataset:

  1. SAMME.R, n_estimators=100          -- ONE shared ensemble, 100 total stumps.
  2. OneVsRestAdaBoost, n_estimators=100 -- K=7 SEPARATE ensembles, 700 total
     stumps (the naive "same n_estimators parameter" comparison).
  3. OneVsRestAdaBoost, n_estimators=n_estimators//K -- compute-MATCHED to
     SAMME.R's total stump count (~100), for a fair per-weak-learner comparison.

WHY THIS COMPARISON MATTERS: "n_estimators=100" means very different things
for the two algorithms. OneVsRestAdaBoost trains K independent full AdaBoost
ensembles (one per class, each with n_estimators stumps), so at
n_estimators=100 it actually trains K*100 = 700 total stumps for K=7 classes.
SAMME.R shares ONE ensemble across all classes, so n_estimators=100 means
exactly 100 total stumps. Comparing them at equal n_estimators without
noting this is comparing a 100-stump model against a 700-stump model and
calling it a fair fight -- config #3 above is the actual fair comparison.

CALIBRATION: the SAMME.R paper's core claim is "improved calibration", not
just accuracy -- calibration means predicted probabilities should match
empirical outcome frequencies (e.g. among samples predicted 70% confident,
roughly 70% should actually be correct). This is measured here via log-loss
(lower is better-calibrated; log-loss penalizes confident-but-wrong
predictions much more than accuracy does), computed from each model's
predict_proba output.

Run standalone:  python experiments/bonus_samme_r.py
"""

from __future__ import annotations

import os
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from experiments.common import save_figure, save_results_json, subsample_for_experiments
from src.boosting.one_vs_rest_adaboost import OneVsRestAdaBoost
from src.boosting.samme_r import SAMMERClassifier
from src.utils.datasets import load_covertype_data

MAX_SAMPLES = 5000
N_ESTIMATORS = 100


def run_samme_r_bonus(random_state: int = 42) -> dict:
    print(f"\n{'='*60}\nBONUS: SAMME.R vs OneVsRestAdaBoost -- covertype\n{'='*60}")

    X, y, name = load_covertype_data(n_samples=15000, random_state=random_state)
    original_n = X.shape[0]
    X, y = subsample_for_experiments(X, y, max_samples=MAX_SAMPLES, random_state=random_state)
    if X.shape[0] < original_n:
        print(f"NOTE: subsampled to {X.shape[0]} rows (stratified) for compute tractability.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    K = len(np.unique(y_train))
    n_estimators_matched = max(1, N_ESTIMATORS // K)

    configs = {
        "samme_r": {
            "label": f"SAMME.R (n_estimators={N_ESTIMATORS}, {N_ESTIMATORS} total stumps)",
            "model_fn": lambda: SAMMERClassifier(n_estimators=N_ESTIMATORS, random_state=random_state),
        },
        "ovr_naive": {
            "label": f"OneVsRestAdaBoost (n_estimators={N_ESTIMATORS}, {N_ESTIMATORS * K} total stumps)",
            "model_fn": lambda: OneVsRestAdaBoost(n_estimators=N_ESTIMATORS, random_state=random_state),
        },
        "ovr_compute_matched": {
            "label": f"OneVsRestAdaBoost (n_estimators={n_estimators_matched}, "
                     f"~{n_estimators_matched * K} total stumps, compute-matched)",
            "model_fn": lambda: OneVsRestAdaBoost(n_estimators=n_estimators_matched, random_state=random_state),
        },
    }

    results = {}
    for key, cfg in configs.items():
        t0 = time.time()
        model = cfg["model_fn"]()
        model.fit(X_train, y_train)
        fit_time = time.time() - t0

        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)

        acc = accuracy_score(y_test, preds)
        ll = log_loss(y_test, proba, labels=model.classes_)

        results[key] = {
            "label": cfg["label"],
            "accuracy": acc,
            "log_loss": ll,
            "fit_time_seconds": fit_time,
        }
        print(f"  {cfg['label']}")
        print(f"    accuracy={acc:.4f}  log_loss={ll:.4f}  fit_time={fit_time:.2f}s")

    print(f"\nInterpretation:")
    print(f"  Naive comparison (same n_estimators param): "
          f"SAMME.R uses {N_ESTIMATORS} total stumps vs OvR's {N_ESTIMATORS * K} -- "
          f"any OvR advantage there comes partly from {K}x more compute, not algorithm quality.")
    print(f"  Compute-matched comparison (~equal total stumps): "
          f"SAMME.R acc={results['samme_r']['accuracy']:.4f} vs "
          f"OvR acc={results['ovr_compute_matched']['accuracy']:.4f}")
    print(f"  Calibration (log-loss, lower=better): "
          f"SAMME.R={results['samme_r']['log_loss']:.4f} vs "
          f"OvR(matched)={results['ovr_compute_matched']['log_loss']:.4f}")
    print(f"  NOTE: on this specific run, report whichever pattern actually printed above --"
          f" don't assume SAMME.R automatically wins on calibration. In development testing"
          f" (Iris, signal-bearing synthetic multiclass data), SAMME.R consistently showed"
          f" HIGHER accuracy than compute-matched OvR, but WORSE (higher) log-loss --"
          f" i.e. more accurate but less calibrated. This is a legitimate, reportable finding:"
          f" AdaBoost-style ensembles (both SAMME and SAMME.R) accumulate votes across rounds"
          f" without decay, which pushes predicted probabilities toward overconfidence as"
          f" training progresses -- an inherent property of the algorithm family, not a bug."
          f" State the actual observed pattern in the report rather than assuming the"
          f" 'improved calibration' framing from the original paper holds against every"
          f" possible baseline; it was proposed relative to discrete SAMME specifically,"
          f" not relative to OneVsRestAdaBoost.")

    # --- bar chart: accuracy and log-loss side by side ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    keys = list(configs.keys())
    short_labels = ["SAMME.R\n(100 stumps)", "OvR naive\n(700 stumps)", "OvR matched\n(~98 stumps)"]

    ax1.bar(short_labels, [results[k]["accuracy"] for k in keys], color=["C0", "C1", "C2"])
    ax1.set_ylabel("Test accuracy")
    ax1.set_title("Accuracy")
    ax1.grid(alpha=0.3, axis="y")

    ax2.bar(short_labels, [results[k]["log_loss"] for k in keys], color=["C0", "C1", "C2"])
    ax2.set_ylabel("Log-loss (lower = better calibrated)")
    ax2.set_title("Calibration")
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("SAMME.R vs OneVsRestAdaBoost: covertype")
    plt.tight_layout()
    save_figure(fig, "bonus_samme_r_vs_ovr_covertype")
    plt.close(fig)

    save_results_json("bonus_samme_r_vs_ovr", results)
    return results


if __name__ == "__main__":
    run_samme_r_bonus()