"""
BONUS (+2 pts): SAMME.R -- real-valued multiclass AdaBoost.

Implements the SAMME.R algorithm from Zhu, Zou, Rosset, Hastie (2009),
"Multi-class AdaBoost" -- the real-valued variant of SAMME. Unlike
discrete SAMME (which uses each weak learner's hard class prediction),
SAMME.R uses each weak learner's estimated CLASS PROBABILITIES directly,
which the original paper shows gives faster convergence and better-
calibrated probability estimates, especially as the number of classes
grows. Natively handles K >= 2 classes in ONE ensemble -- no one-vs-rest
wrapper needed (unlike this project's discrete AdaBoostClassifier, which
is binary-only and relies on OneVsRestAdaBoost for K>2).

ALGORITHM (per the paper):
  Let K = number of classes. Initialize sample weights w_i = 1/n.
  For m = 1..M:
    1. Fit a weak learner (DecisionStump) on the weighted data, get
       per-class probability estimates p_k(x) for k = 1..K.
    2. Compute the real-valued class score:
         h_k(x) = (K-1) * ( log(p_k(x)) - (1/K) * sum_{k'} log(p_{k'}(x)) )
       (a symmetric, zero-sum-across-classes log-odds-like score).
    3. Update sample weights:
         w_i <- w_i * exp( -((K-1)/K) * y_i . log(p(x_i)) )
       where y_i is the K-dim symmetric code: +1 for the true class,
       -1/(K-1) for every other class, and log(p(x_i)) is THIS round's
       log class-probabilities (the update uses log(p) directly, NOT
       h -- h carries an extra factor of (K-1) relative to log(p), which
       would make weight updates increasingly unstable as K grows if
       used here by mistake). Renormalize weights to sum to 1.
  Final prediction: argmax_k sum_m h_k^(m)(x) (sum the real-valued scores
  across all M rounds, per class, then take the class with the highest
  total score).

NOTE ON VALIDATION: recent sklearn versions REMOVED the algorithm="SAMME.R"
option from AdaBoostClassifier entirely (only discrete SAMME remains), so
this cannot be diffed directly against a live sklearn reference the way
the rest of this project's models were. It's validated instead against
the algorithm as specified in the original paper, and cross-checked for
sane behavior (probabilities well-calibrated, weights stay positive and
finite, accuracy competitive with OneVsRestAdaBoost) via the test suite.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from src.trees.decision_tree import DecisionStump


class SAMMERClassifier:
    def __init__(
        self,
        n_estimators: int = 50,
        criterion: str = "gini",
        random_state: int | None = None,
        eps: float = 1e-10,
    ) -> None:
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.random_state = random_state
        self.eps = eps  # clips probabilities away from 0 to avoid log(0)

        self.estimators_: list[DecisionStump] = []
        self.classes_: np.ndarray | None = None
        self.n_classes_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SAMMERClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        K = self.n_classes_
        n_samples = X.shape[0]

        if K < 2:
            raise ValueError(f"SAMMERClassifier requires at least 2 classes; got {K}.")

        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[label] for label in y])

        # symmetric K-dim code: +1 for the true class, -1/(K-1) for every other class
        Y_code = np.full((n_samples, K), -1.0 / (K - 1))
        Y_code[np.arange(n_samples), y_idx] = 1.0

        sample_weight = np.full(n_samples, 1.0 / n_samples)
        self.estimators_ = []

        for m in range(self.n_estimators):
            round_seed = None if self.random_state is None else self.random_state + m
            stump = DecisionStump(criterion=self.criterion, random_state=round_seed)
            stump.fit(X, y, sample_weight=sample_weight)

            proba = stump.predict_proba(X)  # shape [n, K], in the SAME class order as self.classes_
            proba = np.clip(proba, self.eps, 1.0)  # avoid log(0)
            log_proba = np.log(proba)

            # h_k(x) = (K-1) * (log p_k(x) - mean_k' log p_k'(x)) -- used for
            # the FINAL prediction score (see _cumulative_h), not the weight update.
            h = (K - 1) * (log_proba - log_proba.mean(axis=1, keepdims=True))  # shape [n, K]

            # weight update: w_i *= exp(-((K-1)/K) * y_i . log(p(x_i)))
            # IMPORTANT: this uses log_proba directly, NOT h. h already has an
            # extra factor of (K-1) baked in (from the formula above), so using
            # h here would make weight updates (K-1) times too aggressive --
            # harmless at K=2 (factor=1) but increasingly destabilizing as K
            # grows (e.g. factor=6 at K=7), causing weight collapse and the
            # ensemble failing to learn at all for larger K.
            exponent = -((K - 1) / K) * np.sum(Y_code * log_proba, axis=1)
            sample_weight = sample_weight * np.exp(exponent)
            sample_weight = sample_weight / sample_weight.sum()

            self.estimators_.append(stump)

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _cumulative_h(self, X: np.ndarray, up_to: int | None = None) -> np.ndarray:
        """Sum of h_k^(m)(x) across the first `up_to` rounds (all rounds if None)."""
        K = self.n_classes_
        estimators = self.estimators_ if up_to is None else self.estimators_[:up_to]

        n_samples = X.shape[0]
        total_h = np.zeros((n_samples, K))

        for stump in estimators:
            proba = np.clip(stump.predict_proba(X), self.eps, 1.0)
            log_proba = np.log(proba)
            h = (K - 1) * (log_proba - log_proba.mean(axis=1, keepdims=True))
            total_h += h

        return total_h

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        total_h = self._cumulative_h(X)
        class_idx = np.argmax(total_h, axis=1)
        return self.classes_[class_idx]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Softmax over the cumulative real-valued scores -- turns the
        raw per-class score totals into a normalized probability
        distribution (same documented approach as AdaBoostClassifier's
        predict_proba)."""
        X = np.asarray(X, dtype=np.float64)
        total_h = self._cumulative_h(X)

        shifted = total_h - total_h.max(axis=1, keepdims=True)
        exp_h = np.exp(shifted)
        return exp_h / exp_h.sum(axis=1, keepdims=True)

    def staged_predict(self, X: np.ndarray) -> Iterator[np.ndarray]:
        """Yield predictions after each boosting round (1..M), accumulating
        incrementally (each stump's h(x) computed exactly once)."""
        X = np.asarray(X, dtype=np.float64)
        K = self.n_classes_
        n_samples = X.shape[0]
        running_h = np.zeros((n_samples, K))

        for stump in self.estimators_:
            proba = np.clip(stump.predict_proba(X), self.eps, 1.0)
            log_proba = np.log(proba)
            h = (K - 1) * (log_proba - log_proba.mean(axis=1, keepdims=True))
            running_h += h

            class_idx = np.argmax(running_h, axis=1)
            yield self.classes_[class_idx]

    def __repr__(self) -> str:
        n_fit = len(self.estimators_)
        return (
            f"SAMMERClassifier(n_estimators={self.n_estimators}, fitted_estimators={n_fit}, "
            f"n_classes={self.n_classes_}, criterion={self.criterion!r})"
        )