"""
Decision Tree Classifier (CART) implemented from scratch using NumPy only.

Supports continuous features, Gini/entropy impurity, weighted samples
(needed by AdaBoost), max_features subsampling (needed by Random Forest),
and feature importance / repr introspection.
"""

from __future__ import annotations

import numpy as np


class Node:
    """A single node in the decision tree."""

    __slots__ = (
        "feature_index", "threshold", "left", "right",
        "value", "samples", "impurity",
    )

    def __init__(
        self,
        feature_index: int | None = None,
        threshold: float | None = None,
        left: "Node | None" = None,
        right: "Node | None" = None,
        value: np.ndarray | None = None,
        samples: int = 0,
        impurity: float = 0.0,
    ) -> None:
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value          # class distribution (counts or weighted counts), shape [n_classes]
        self.samples = samples      # number of samples reaching this node
        self.impurity = impurity    # impurity value at this node

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTree:
    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        criterion: str = "gini",
        max_features: int | str | None = None,
        random_state: int | None = None,
    ) -> None:
        if criterion not in ("gini", "entropy"):
            raise ValueError(f"criterion must be 'gini' or 'entropy', got {criterion!r}")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state

        self.root: Node | None = None
        self.n_classes_: int = 0
        self.classes_: np.ndarray | None = None
        self.n_features_: int = 0
        self._feature_importances: np.ndarray | None = None
        self._rng: np.random.RandomState = np.random.RandomState(random_state)

    # ------------------------------------------------------------------
    # Impurity functions
    # ------------------------------------------------------------------
    def _impurity(self, weighted_counts: np.ndarray) -> float:
        """Weighted Gini or entropy impurity from class weight totals."""
        total = weighted_counts.sum()
        if total <= 0:
            return 0.0
        p = weighted_counts / total
        if self.criterion == "gini":
            return 1.0 - np.sum(p ** 2)
        else:  # entropy
            eps = 1e-12
            return -np.sum(p * np.log2(p + eps))

    def _impurity_vectorized(self, weighted_counts: np.ndarray, totals: np.ndarray) -> np.ndarray:
        """Same as _impurity, but computes impurity for MANY candidate
        splits at once. weighted_counts: shape [n_candidates, n_classes].
        totals: shape [n_candidates] (row sums, passed in since the caller
        already has them, avoiding a redundant sum here).
        Returns: shape [n_candidates]."""
        safe_totals = np.where(totals > 0, totals, 1.0)  # avoid divide-by-zero; result is masked out by caller anyway
        p = weighted_counts / safe_totals[:, np.newaxis]
        if self.criterion == "gini":
            result = 1.0 - np.sum(p ** 2, axis=1)
        else:  # entropy
            eps = 1e-12
            result = -np.sum(p * np.log2(p + eps), axis=1)
        result[totals <= 0] = 0.0
        return result

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "DecisionTree":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        self._feature_importances = np.zeros(self.n_features_)
        self._rng = np.random.RandomState(self.random_state)

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0], dtype=np.float64)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)

        # map class labels -> contiguous indices [0, n_classes) for fast bincount-style counting
        self._class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([self._class_to_idx[label] for label in y])

        self.root = self._build_tree(X, y_idx, sample_weight, depth=0)

        total_importance = self._feature_importances.sum()
        if total_importance > 0:
            self._feature_importances /= total_importance

        return self

    def _weighted_class_counts(self, y_idx: np.ndarray, sample_weight: np.ndarray) -> np.ndarray:
        counts = np.zeros(self.n_classes_)
        for c in range(self.n_classes_):
            counts[c] = sample_weight[y_idx == c].sum()
        return counts

    def _build_tree(self, X: np.ndarray, y_idx: np.ndarray, sample_weight: np.ndarray, depth: int) -> Node:
        n_samples = X.shape[0]
        class_counts = self._weighted_class_counts(y_idx, sample_weight)
        node_impurity = self._impurity(class_counts)

        node = Node(value=class_counts.copy(), samples=n_samples, impurity=node_impurity)

        # --- stopping criteria ---
        if (
            n_samples < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
            or np.all(y_idx == y_idx[0])           # pure node
            or self._all_identical_rows(X)          # no valid split possible
        ):
            return node

        best = self._best_split(X, y_idx, sample_weight, node_impurity)
        if best is None:
            return node  # no split reduces impurity

        feature_index, threshold, gain, left_mask, right_mask = best

        self._feature_importances[feature_index] += gain * n_samples

        node.feature_index = feature_index
        node.threshold = threshold
        node.left = self._build_tree(X[left_mask], y_idx[left_mask], sample_weight[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y_idx[right_mask], sample_weight[right_mask], depth + 1)
        return node

    @staticmethod
    def _all_identical_rows(X: np.ndarray) -> bool:
        if X.shape[0] <= 1:
            return True
        return np.all(X == X[0], axis=None)

    def _select_features(self) -> np.ndarray:
        """Return the indices of features to consider at this node (max_features support)."""
        if self.max_features is None:
            return np.arange(self.n_features_)

        if self.max_features == "sqrt":
            k = max(1, int(np.floor(np.sqrt(self.n_features_))))
        elif self.max_features == "log2":
            k = max(1, int(np.floor(np.log2(self.n_features_))))
        elif isinstance(self.max_features, int):
            k = max(1, min(self.max_features, self.n_features_))
        else:
            raise ValueError(f"Unsupported max_features: {self.max_features!r}")

        return self._rng.choice(self.n_features_, size=k, replace=False)

    def _best_split(self, X: np.ndarray, y_idx: np.ndarray, sample_weight: np.ndarray, parent_impurity: float):
        n_samples = X.shape[0]
        total_weight = sample_weight.sum()
        if total_weight <= 0:
            return None

        best_gain = -1.0  # sentinel below zero: gain is always >= 0 mathematically for
                          # Gini/entropy, so this lets ties at gain == 0 still be selected
                          # (matches sklearn's default min_impurity_decrease=0.0 behavior,
                          # which is required to solve cases like XOR at depth 2).
        best = None  # (feature_index, threshold, gain, left_mask, right_mask)

        total_counts = self._weighted_class_counts(y_idx, sample_weight)

        # one-hot weighted class matrix: shape [n_samples, n_classes].
        # row i has sample_weight[i] in column y_idx[i], zero elsewhere.
        # This lets us get cumulative per-class weight via a single cumsum,
        # instead of a Python for-loop over every sample.
        weighted_one_hot = np.zeros((n_samples, self.n_classes_))
        weighted_one_hot[np.arange(n_samples), y_idx] = sample_weight

        feature_indices = self._select_features()

        for j in feature_indices:
            col = X[:, j]
            order = np.argsort(col, kind="mergesort")
            sorted_col = col[order]
            sorted_one_hot = weighted_one_hot[order]

            # cumulative weighted class counts for "everything up to and
            # including index i" -- vectorized via cumsum instead of a loop
            cum_left_counts = np.cumsum(sorted_one_hot, axis=0)  # shape [n_samples, n_classes]
            cum_left_weight = cum_left_counts.sum(axis=1)  # shape [n_samples]

            # only indices 0..n_samples-2 are valid split points (need a
            # right side); drop the last row
            left_counts_at_i = cum_left_counts[:-1]  # shape [n_samples-1, n_classes]
            left_weight_at_i = cum_left_weight[:-1]  # shape [n_samples-1]
            right_weight_at_i = total_weight - left_weight_at_i
            right_counts_at_i = total_counts[np.newaxis, :] - left_counts_at_i

            # mask out invalid split points: adjacent identical values
            # (can't split between equal values), or a zero-weight side
            valid = (sorted_col[:-1] != sorted_col[1:]) & (left_weight_at_i > 0) & (right_weight_at_i > 0)

            if not np.any(valid):
                continue

            # vectorized impurity for every candidate split point at once
            left_impurity = self._impurity_vectorized(left_counts_at_i, left_weight_at_i)
            right_impurity = self._impurity_vectorized(right_counts_at_i, right_weight_at_i)

            weighted_child_impurity = (
                (left_weight_at_i / total_weight) * left_impurity
                + (right_weight_at_i / total_weight) * right_impurity
            )
            gains = parent_impurity - weighted_child_impurity
            gains[~valid] = -1.0  # exclude invalid points from argmax

            local_best_i = np.argmax(gains)
            local_best_gain = gains[local_best_i]

            if local_best_gain > best_gain:
                threshold = (sorted_col[local_best_i] + sorted_col[local_best_i + 1]) / 2.0
                left_mask = X[:, j] <= threshold
                right_mask = ~left_mask
                best_gain = local_best_gain
                best = (j, threshold, local_best_gain, left_mask, right_mask)

        return best

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _traverse(self, x: np.ndarray, node: Node) -> Node:
        while not node.is_leaf():
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        probs = np.zeros((X.shape[0], self.n_classes_))
        for i, x in enumerate(X):
            leaf = self._traverse(x, self.root)
            total = leaf.value.sum()
            probs[i] = leaf.value / total if total > 0 else np.ones(self.n_classes_) / self.n_classes_
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        class_idx = np.argmax(proba, axis=1)
        return self.classes_[class_idx]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def depth(self) -> int:
        def _depth(node: Node) -> int:
            if node is None or node.is_leaf():
                return 0
            return 1 + max(_depth(node.left), _depth(node.right))
        return _depth(self.root) if self.root is not None else 0

    @property
    def n_leaves(self) -> int:
        def _count(node: Node) -> int:
            if node is None:
                return 0
            if node.is_leaf():
                return 1
            return _count(node.left) + _count(node.right)
        return _count(self.root)

    def feature_importances(self) -> np.ndarray:
        if self._feature_importances is None:
            raise RuntimeError("Tree has not been fit yet.")
        return self._feature_importances

    def __repr__(self) -> str:
        if self.root is None:
            return "DecisionTree(unfit)"
        if self.depth > 4:
            return f"DecisionTree(depth={self.depth}, n_leaves={self.n_leaves}, criterion={self.criterion!r})"

        lines = []

        def _print(node: Node, indent: str = "") -> None:
            if node.is_leaf():
                counts = node.value
                lines.append(
                    f"{indent}Leaf: {self.criterion}={node.impurity:.3f}, "
                    f"samples={node.samples}, class_dist={np.round(counts, 2).tolist()}"
                )
            else:
                lines.append(
                    f"{indent}[X{node.feature_index} <= {node.threshold:.3f}] "
                    f"{self.criterion}={node.impurity:.3f}, samples={node.samples}"
                )
                _print(node.left, indent + "  ")
                _print(node.right, indent + "  ")

        _print(self.root)
        return "\n".join(lines)


class DecisionStump(DecisionTree):
    """A depth-1 DecisionTree, used as the weak learner in AdaBoost."""

    def __init__(self, criterion: str = "gini", random_state: int | None = None) -> None:
        super().__init__(max_depth=1, min_samples_split=2, criterion=criterion, random_state=random_state)