"""
Dataset loaders for the four datasets used in this project:

  1. Breast Cancer Wisconsin  -- clean baseline (sklearn built-in)
  2. Credit Card Fraud        -- severe imbalance, 0.17% minority class (Kaggle CSV)
  3. MNIST 2-class subset     -- high-dimensional, 784 features (sklearn fetch_openml)
  4. Covertype subset         -- multi-class (sklearn built-in fetch_covtype)

Every loader returns (X, y, name) as plain NumPy arrays so they drop
straight into preprocessing.py / evaluation.py / the model classes.
All loaders fix random_state=42 for any subsampling, per the
reproducibility requirement in the brief.
"""

from __future__ import annotations

import os

import numpy as np


def load_breast_cancer_data():
    """Binary, 569 samples, 30 features, ~63/37 class balance. No download needed."""
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    return data.data, data.target, "breast_cancer"


def load_credit_card_fraud_data(csv_path: str = "data/creditcard.csv"):
    """
    Binary, ~284,807 samples, 30 features (V1-V28 PCA'd + Time + Amount),
    ~0.17% fraud rate -- this is the required severe-imbalance dataset.

    Must be downloaded manually first (requires a Kaggle account):
        1. https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
        2. Download creditcard.csv, place it at data/creditcard.csv
    This cannot be scripted into download_data.sh without Kaggle API
    credentials, so it's a documented manual step -- note this in your README.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Credit Card Fraud dataset not found at {csv_path!r}.\n"
            f"Download it manually from "
            f"https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            f"and place creditcard.csv at that path."
        )

    import csv as csv_module

    with open(csv_path, newline="") as f:
        reader = csv_module.reader(f)
        header = next(reader)
        rows = [row for row in reader]

    data = np.array(rows, dtype=np.float64)
    X = data[:, :-1]  # all columns except the last (Class)
    y = data[:, -1].astype(int)  # Class: 0 = normal, 1 = fraud

    return X, y, "credit_card_fraud"


def load_mnist_binary_data(
    digit_a: int = 3,
    digit_b: int = 8,
    n_samples: int = 6000,
    random_state: int = 42,
):
    """
    Binary subset of MNIST (two digit classes), high-dimensional: 784 features
    -- this is the required high-dimensional dataset.

    First call downloads and caches MNIST via sklearn (can take a minute);
    subsequent calls reuse the local cache.
    """
    from sklearn.datasets import fetch_openml

    mnist = fetch_openml("mnist_784", version=1, return_X_y=False, as_frame=False)
    X_all, y_all = mnist.data, mnist.target.astype(int)

    mask = (y_all == digit_a) | (y_all == digit_b)
    X_subset = X_all[mask]
    y_subset = y_all[mask]

    rng = np.random.RandomState(random_state)
    if n_samples < len(y_subset):
        idx = rng.choice(len(y_subset), size=n_samples, replace=False)
        X_subset = X_subset[idx]
        y_subset = y_subset[idx]

    # relabel to {0, 1} for compatibility with the binary-only AdaBoost implementation
    y_binary = (y_subset == digit_b).astype(int)

    return X_subset.astype(np.float64), y_binary, f"mnist_{digit_a}_vs_{digit_b}"


def load_covertype_data(n_samples: int = 15000, random_state: int = 42):
    """
    Multi-class (7 forest cover types), 54 features. Used for Random Forest /
    Decision Tree multi-class experiments, and as the target dataset for the
    SAMME.R multiclass AdaBoost bonus.

    NOTE: the from-scratch AdaBoostClassifier in this project only supports
    binary classification (per the brief). Running the core AdaBoost
    experiments on this dataset requires either a one-vs-rest wrapper or
    the SAMME.R bonus extension -- don't feed this directly into
    AdaBoostClassifier.fit() as-is, it will raise ValueError.

    First call downloads via sklearn (~75MB, can take a minute); subsequent
    calls reuse the local cache.
    """
    from sklearn.datasets import fetch_covtype

    data = fetch_covtype()
    X_all, y_all = data.data, data.target

    rng = np.random.RandomState(random_state)
    if n_samples < len(y_all):
        idx = rng.choice(len(y_all), size=n_samples, replace=False)
        X_all = X_all[idx]
        y_all = y_all[idx]

    return X_all.astype(np.float64), y_all, "covertype"


# ----------------------------------------------------------------------
# Convenience registry
# ----------------------------------------------------------------------
def load_all_datasets(credit_card_csv_path: str = "data/creditcard.csv") -> dict:
    """
    Load all four datasets into a single dict keyed by name:
        {"breast_cancer": (X, y), "credit_card_fraud": (X, y), ...}

    Used by run_all.py to iterate "for each dataset, run experiments 1-7".
    Raises FileNotFoundError immediately if the Credit Card CSV is missing,
    with instructions on how to get it.
    """
    datasets = {}

    X, y, name = load_breast_cancer_data()
    datasets[name] = (X, y)

    X, y, name = load_credit_card_fraud_data(credit_card_csv_path)
    datasets[name] = (X, y)

    X, y, name = load_mnist_binary_data()
    datasets[name] = (X, y)

    X, y, name = load_covertype_data()
    datasets[name] = (X, y)

    return datasets