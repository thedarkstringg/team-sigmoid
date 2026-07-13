"""
Random Forest Classifier implemented from scratch using NumPy only,
built on top of DecisionTree with bootstrap aggregation (bagging) and
feature sub-sampling. Supports multiprocessing.Pool parallelism.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np

from src.trees.decision_tree import DecisionTree


def _train_one_tree(args):
    """Module-level function (required so it can be pickled for multiprocessing.Pool).

    args: (tree_index, X, y, bootstrap_indices, oob_mask, tree_params)
    Returns: (tree, oob_mask)
    """
    tree_index, X, y, bootstrap_indices, oob_mask, tree_params = args

    X_boot = X[bootstrap_indices]
    y_boot = y[bootstrap_indices]

    tree_seed = tree_params["random_state"]
    if tree_seed is not None:
        tree_seed = tree_seed + tree_index

    tree = DecisionTree(
        max_depth=tree_params["max_depth"],
        min_samples_split=tree_params["min_samples_split"],
        criterion=tree_params["criterion"],
        max_features=tree_params["max_features"],
        random_state=tree_seed,
    )
    tree.fit(X_boot, y_boot)
    return tree, oob_mask


class RandomForestClassifier:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        max_features: int | str | None = "sqrt",
        min_samples_split: int = 2,
        criterion: str = "gini",
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: int = 1,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.estimators_: list[DecisionTree] = []
        self.oob_masks_: list[np.ndarray] = []
        self.classes_: np.ndarray | None = None
        self.n_classes_: int = 0
        self.n_features_: int = 0
        self._oob_score: float | None = None
        self._feature_importances: np.ndarray | None = None

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        n_samples = X.shape[0]

        rng = np.random.RandomState(self.random_state)

        # --- build bootstrap samples (or use full dataset if bootstrap=False) ---
        train_jobs = []
        for t in range(self.n_estimators):
            if self.bootstrap:
                bootstrap_indices = rng.randint(0, n_samples, size=n_samples)
                in_bag = np.zeros(n_samples, dtype=bool)
                in_bag[bootstrap_indices] = True
                oob_mask = ~in_bag
            else:
                bootstrap_indices = np.arange(n_samples)
                oob_mask = np.zeros(n_samples, dtype=bool)  # nothing is OOB

            tree_params = {
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "criterion": self.criterion,
                "max_features": self.max_features,
                "random_state": self.random_state,
            }
            train_jobs.append((t, X, y, bootstrap_indices, oob_mask, tree_params))

        # --- train trees, sequentially or in parallel ---
        if self.n_jobs is not None and self.n_jobs > 1:
            with mp.Pool(processes=self.n_jobs) as pool:
                results = pool.map(_train_one_tree, train_jobs)
        else:
            results = [_train_one_tree(job) for job in train_jobs]

        self.estimators_ = [tree for tree, _ in results]
        self.oob_masks_ = [mask for _, mask in results]  # exposed publicly for experiments

        # --- feature importances: average across all trees ---
        importances = np.array([tree.feature_importances() for tree in self.estimators_])
        self._feature_importances = importances.mean(axis=0)

        # --- OOB score ---
        if self.oob_score:
            self._oob_score = self._compute_oob_score(X, y, self.oob_masks_)

        return self

    def _compute_oob_score(self, X: np.ndarray, y: np.ndarray, oob_masks: list[np.ndarray]) -> float:
        n_samples = X.shape[0]
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        vote_counts = np.zeros((n_samples, self.n_classes_))
        any_oob_votes = np.zeros(n_samples, dtype=bool)

        for tree, mask in zip(self.estimators_, oob_masks):
            if not np.any(mask):
                continue
            oob_indices = np.where(mask)[0]
            preds = tree.predict(X[oob_indices])
            for idx, pred in zip(oob_indices, preds):
                vote_counts[idx, class_to_idx[pred]] += 1
                any_oob_votes[idx] = True

        if not np.any(any_oob_votes):
            # degenerate case: e.g. very few trees, every sample was in-bag somewhere always
            return float("nan")

        scored_indices = np.where(any_oob_votes)[0]
        oob_preds = self.classes_[np.argmax(vote_counts[scored_indices], axis=1)]
        return float(np.mean(oob_preds == y[scored_indices]))

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average of per-tree probability vectors."""
        X = np.asarray(X, dtype=np.float64)
        proba_sum = np.zeros((X.shape[0], self.n_classes_))
        for tree in self.estimators_:
            proba_sum += tree.predict_proba(X)
        return proba_sum / len(self.estimators_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Majority vote across trees (equivalent to argmax of averaged probabilities)."""
        proba = self.predict_proba(X)
        class_idx = np.argmax(proba, axis=1)
        return self.classes_[class_idx]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def oob_score_(self) -> float:
        if not self.oob_score:
            raise RuntimeError("oob_score_ is only available when oob_score=True was set.")
        if self._oob_score is None:
            raise RuntimeError("RandomForestClassifier has not been fit yet.")
        return self._oob_score

    @property
    def feature_importances_(self) -> np.ndarray:
        if self._feature_importances is None:
            raise RuntimeError("RandomForestClassifier has not been fit yet.")
        return self._feature_importances

    def __repr__(self) -> str:
        n_fit = len(self.estimators_)
        return (
            f"RandomForestClassifier(n_estimators={self.n_estimators}, fitted_estimators={n_fit}, "
            f"max_depth={self.max_depth}, max_features={self.max_features!r}, "
            f"bootstrap={self.bootstrap}, n_jobs={self.n_jobs})"
        )
