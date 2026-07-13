"""
AdaBoost (discrete SAMME variant) implemented from scratch using NumPy only,
built on top of DecisionStump (depth-1 DecisionTree) weak learners.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from src.trees.decision_tree import DecisionStump


class AdaBoostClassifier:
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

        self.estimators_: list[DecisionStump] = []
        self._estimator_weights: np.ndarray | None = None
        self._estimator_errors: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self.n_classes_: int = 0

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaBoostClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        n_samples = X.shape[0]

        if self.n_classes_ != 2:
            raise ValueError(
                f"This SAMME implementation targets binary classification; got "
                f"{self.n_classes_} classes. (alpha_m formula uses ln(K-1); for K=2 "
                f"this term is ln(1)=0, i.e. the classic binary AdaBoost update.)"
            )

        self.estimators_ = []
        alphas = []
        errors = []

        # w^(1)_i = 1/N
        sample_weight = np.full(n_samples, 1.0 / n_samples)

        for m in range(self.n_estimators):
            # separate generator per round: random_state + m, as required by the spec
            round_seed = None if self.random_state is None else self.random_state + m
            stump = DecisionStump(criterion=self.criterion, random_state=round_seed)
            stump.fit(X, y, sample_weight=sample_weight)

            pred = stump.predict(X)
            incorrect = (pred != y)

            # epsilon_m = weighted error rate
            epsilon_m = np.sum(sample_weight * incorrect) / np.sum(sample_weight)

            # clip to avoid division by zero / log(0)
            epsilon_m = np.clip(epsilon_m, 1e-10, None)

            if epsilon_m >= 0.5:
                # Weak learner is no better than random guessing (or worse).
                # Per the brief: stop early rather than let alpha_m go negative
                # and corrupt the ensemble. Document: we choose early termination.
                if m == 0:
                    # If even the very first stump can't beat chance, there is
                    # no usable ensemble at all -- surface this loudly.
                    raise ValueError(
                        f"First weak learner has weighted error {epsilon_m:.4f} >= 0.5; "
                        f"cannot build an AdaBoost ensemble from a learner no better than chance."
                    )
                break

            # alpha_m = ln((1-eps)/eps) + ln(K-1); for K=2, ln(K-1)=ln(1)=0
            alpha_m = self.learning_rate * (
                np.log((1 - epsilon_m) / epsilon_m) + np.log(self.n_classes_ - 1)
            )

            self.estimators_.append(stump)
            alphas.append(alpha_m)
            errors.append(epsilon_m)

            # update + renormalize sample weights
            sample_weight = sample_weight * np.exp(alpha_m * incorrect)
            sample_weight = sample_weight / sample_weight.sum()

        self._estimator_weights = np.array(alphas)
        self._estimator_errors = np.array(errors)
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _stump_vote_contribution(self, stump: DecisionStump, alpha: float, X: np.ndarray) -> np.ndarray:
        """Return shape [n_samples, n_classes]: this single stump's
        alpha-weighted vote, one-hot per sample. Vectorized via
        np.searchsorted instead of a Python per-sample loop (self.classes_
        is sorted ascending, since it comes from np.unique)."""
        pred = stump.predict(X)
        pred_idx = np.searchsorted(self.classes_, pred)

        n_samples = X.shape[0]
        contribution = np.zeros((n_samples, self.n_classes_))
        contribution[np.arange(n_samples), pred_idx] = alpha
        return contribution

    def _weighted_class_votes(self, X: np.ndarray, up_to: int | None = None) -> np.ndarray:
        """Return array of shape [n_samples, n_classes] with alpha-weighted vote sums."""
        estimators = self.estimators_ if up_to is None else self.estimators_[:up_to]
        alphas = self._estimator_weights if up_to is None else self._estimator_weights[:up_to]

        n_samples = X.shape[0]
        votes = np.zeros((n_samples, self.n_classes_))

        for stump, alpha in zip(estimators, alphas):
            votes += self._stump_vote_contribution(stump, alpha, X)

        return votes

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        votes = self._weighted_class_votes(X)
        class_idx = np.argmax(votes, axis=1)
        return self.classes_[class_idx]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Softmax over the alpha-weighted vote sums (documented choice --
        turns raw vote totals into a normalized probability distribution)."""
        X = np.asarray(X, dtype=np.float64)
        votes = self._weighted_class_votes(X)

        # softmax, numerically stable
        shifted = votes - votes.max(axis=1, keepdims=True)
        exp_votes = np.exp(shifted)
        proba = exp_votes / exp_votes.sum(axis=1, keepdims=True)
        return proba

    def staged_predict(self, X: np.ndarray) -> Iterator[np.ndarray]:
        """Yield predictions after each boosting round (1..M).

        Accumulates votes incrementally (each stump's prediction is
        computed exactly ONCE, not recomputed at every later stage) --
        O(M*N) total instead of the O(M^2*N) cost of recomputing all
        prior estimators' predictions from scratch at every stage, which
        matters a lot once M and N are both in the hundreds/hundreds-of-
        thousands range."""
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        running_votes = np.zeros((n_samples, self.n_classes_))

        for stump, alpha in zip(self.estimators_, self._estimator_weights):
            running_votes += self._stump_vote_contribution(stump, alpha, X)
            class_idx = np.argmax(running_votes, axis=1)
            yield self.classes_[class_idx]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def estimator_weights(self) -> np.ndarray:
        if self._estimator_weights is None:
            raise RuntimeError("AdaBoostClassifier has not been fit yet.")
        return self._estimator_weights

    @property
    def estimator_errors(self) -> np.ndarray:
        if self._estimator_errors is None:
            raise RuntimeError("AdaBoostClassifier has not been fit yet.")
        return self._estimator_errors

    def __repr__(self) -> str:
        n_fit = len(self.estimators_)
        return (
            f"AdaBoostClassifier(n_estimators={self.n_estimators}, fitted_estimators={n_fit}, "
            f"learning_rate={self.learning_rate}, criterion={self.criterion!r})"
        )