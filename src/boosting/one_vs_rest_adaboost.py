"""
One-vs-Rest wrapper for AdaBoostClassifier, needed because the from-scratch
AdaBoostClassifier (discrete SAMME) only supports binary classification.

Per the project brief: "For multi-class datasets, you may binarize or keep
as multi-class. AdaBoost must support binary classification; implement
one-vs-rest or SAMME.R as needed." This module implements the one-vs-rest
approach; the SAMME.R bonus (+2 pts) is a separate, more involved extension
that natively handles K>2 classes in a single AdaBoost ensemble.

Strategy: train K independent binary AdaBoostClassifier instances, one per
class (class k vs. all other classes combined). At prediction time, take
each binary classifier's estimated probability that a sample belongs to
its "positive" class, and predict the class whose classifier is most
confident.
"""

from __future__ import annotations

import numpy as np

from src.boosting.adaboost import AdaBoostClassifier


class OneVsRestAdaBoost:
    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        criterion: str = "gini",
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.random_state = random_state

        self.classes_: np.ndarray | None = None
        self.n_classes_: int = 0
        self.estimators_: dict[object, AdaBoostClassifier] = {}

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "OneVsRestAdaBoost":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        if self.n_classes_ < 2:
            raise ValueError(
                f"OneVsRestAdaBoost requires at least 2 classes; got {self.n_classes_}."
            )

        self.estimators_ = {}

        for i, class_label in enumerate(self.classes_):
            # binary labels: 1 if this sample belongs to class_label, else 0
            y_binary = (y == class_label).astype(int)

            # separate seed per binary sub-problem, same pattern as AdaBoost's
            # own per-round seeding, so results are still fully reproducible
            sub_seed = None if self.random_state is None else self.random_state + i

            clf = AdaBoostClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                criterion=self.criterion,
                random_state=sub_seed,
            )

            # Edge case: a class-vs-rest split can itself be very imbalanced
            # (e.g. 1-of-7 Covertype classes vs the other 6 combined). If the
            # very first stump can't beat chance on this split, AdaBoost raises
            # ValueError -- let it propagate with added context about which
            # class's binary sub-problem failed, rather than a confusing
            # internal AdaBoost error.
            try:
                clf.fit(X, y_binary)
            except ValueError as e:
                raise ValueError(
                    f"Failed to fit binary sub-classifier for class {class_label!r} "
                    f"vs. rest: {e}"
                ) from e

            self.estimators_[class_label] = clf

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _class_scores(self, X: np.ndarray) -> np.ndarray:
        """Return array of shape [n_samples, n_classes]: each column is that
        class's binary classifier's estimated P(this sample IS that class)."""
        X = np.asarray(X, dtype=np.float64)
        scores = np.zeros((X.shape[0], self.n_classes_))

        for i, class_label in enumerate(self.classes_):
            clf = self.estimators_[class_label]
            proba = clf.predict_proba(X)  # shape [n_samples, 2]: [P(not class), P(class)]
            scores[:, i] = proba[:, 1]

        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self._class_scores(X)
        class_idx = np.argmax(scores, axis=1)
        return self.classes_[class_idx]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Normalize the per-class binary scores to sum to 1 across classes.
        This is the standard OvR approach: raw scores from independently
        trained binary classifiers aren't a true joint probability
        distribution, but normalizing gives a usable approximation
        (documented choice, same spirit as AdaBoost's own softmax proba)."""
        scores = self._class_scores(X)
        totals = scores.sum(axis=1, keepdims=True)

        # guard against all-zero rows (all binary classifiers assign ~0
        # probability to their positive class for this sample)
        totals = np.where(totals == 0, 1.0, totals)
        return scores / totals

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        n_fit = len(self.estimators_)
        return (
            f"OneVsRestAdaBoost(n_estimators={self.n_estimators}, "
            f"fitted_classifiers={n_fit}, learning_rate={self.learning_rate}, "
            f"criterion={self.criterion!r})"
        )