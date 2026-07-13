"""
Preprocessing utilities: feature scaling, missing-value handling, and
class-imbalance treatment (SMOTE / random oversampling / class weights).

Per the project brief, StandardScaler may be sklearn's own or a custom
implementation -- this uses sklearn's, since only DecisionTree, AdaBoost,
RandomForest, PCA, KMeans, and DBSCAN are required to be built from scratch.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


# ----------------------------------------------------------------------
# Missing values
# ----------------------------------------------------------------------
def handle_missing_values(
    X: np.ndarray, strategy: str = "mean", missing_value=np.nan
) -> np.ndarray:
    """
    Handle missing values in a feature matrix.

    strategy:
        "mean"   -- impute with per-column mean (computed ignoring NaNs)
        "median" -- impute with per-column median
        "drop"   -- drop rows containing any missing value (returns fewer rows)

    Why "mean"/"median" over "drop" by default: dropping rows discards
    potentially informative samples and can bias imbalanced datasets even
    further if missingness correlates with the minority class. Imputation
    is generally preferred unless missingness is extremely rare, in which
    case dropping barely changes the dataset. Document your actual choice
    per-dataset in the report.
    """
    X = np.array(X, dtype=np.float64, copy=True)

    if strategy == "drop":
        if np.isnan(missing_value):
            mask = ~np.any(np.isnan(X), axis=1)
        else:
            mask = ~np.any(X == missing_value, axis=1)
        return X[mask]

    if strategy not in ("mean", "median"):
        raise ValueError(f"Unknown strategy: {strategy!r}")

    if np.isnan(missing_value):
        col_mask = np.isnan(X)
    else:
        col_mask = X == missing_value
        X = np.where(col_mask, np.nan, X)  # normalize sentinel to NaN for nanmean/nanmedian

    fill_fn = np.nanmean if strategy == "mean" else np.nanmedian
    fill_values = fill_fn(X, axis=0)

    inds = np.where(np.isnan(X))
    X[inds] = np.take(fill_values, inds[1])

    return X


# ----------------------------------------------------------------------
# Scaling
# ----------------------------------------------------------------------
def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit a StandardScaler on TRAINING data only (never on test data --
    fitting on test data leaks test-set statistics into preprocessing)."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def scale_data(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    """Apply an already-fit scaler to any split (train, val, or test)."""
    return scaler.transform(X)


# ----------------------------------------------------------------------
# Imbalance treatment
# ----------------------------------------------------------------------
def minority_class_fraction(y: np.ndarray) -> float:
    """Fraction of samples belonging to the smallest class."""
    _, counts = np.unique(y, return_counts=True)
    return counts.min() / counts.sum()


def handle_imbalance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str = "smote",
    random_state: int | None = 42,
):
    """
    Apply an imbalance treatment strategy to the TRAINING split only
    (never touch the test split -- test data must reflect real-world
    class proportions to give an honest performance estimate).

    method:
        "smote"             -- Synthetic Minority Oversampling (imbalanced-learn).
                                Generates synthetic minority samples by interpolating
                                between real minority neighbors, rather than exact
                                duplicates -- reduces overfitting vs plain duplication.
        "random_oversample" -- Duplicate minority-class samples (with replacement)
                                until classes are balanced. Simple, no extra
                                dependency, but can encourage overfitting to
                                the exact duplicated points.
        "class_weight"      -- Don't resample; instead return a sample_weight
                                array (inverse class frequency) to pass into
                                DecisionTree.fit(..., sample_weight=...) or
                                RandomForest's underlying trees. Does NOT apply
                                to AdaBoost, which manages its own internal
                                sample weights during boosting.

    Returns:
        For "smote" / "random_oversample": (X_resampled, y_resampled)
        For "class_weight": (X_train, y_train, sample_weight)
    """
    if method == "smote":
        from imblearn.over_sampling import SMOTE

        smote = SMOTE(random_state=random_state)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        return X_res, y_res

    elif method == "random_oversample":
        rng = np.random.RandomState(random_state)
        classes, counts = np.unique(y_train, return_counts=True)
        max_count = counts.max()

        X_parts, y_parts = [X_train], [y_train]
        for c, count in zip(classes, counts):
            n_needed = max_count - count
            if n_needed <= 0:
                continue
            class_indices = np.where(y_train == c)[0]
            resampled_indices = rng.choice(class_indices, size=n_needed, replace=True)
            X_parts.append(X_train[resampled_indices])
            y_parts.append(y_train[resampled_indices])

        X_res = np.vstack(X_parts)
        y_res = np.concatenate(y_parts)

        # shuffle so the resampled rows aren't all clustered at the end
        perm = rng.permutation(len(y_res))
        return X_res[perm], y_res[perm]

    elif method == "class_weight":
        classes, counts = np.unique(y_train, return_counts=True)
        class_weight_map = {c: len(y_train) / (len(classes) * count) for c, count in zip(classes, counts)}
        sample_weight = np.array([class_weight_map[label] for label in y_train])
        return X_train, y_train, sample_weight

    else:
        raise ValueError(f"Unknown imbalance method: {method!r}")
