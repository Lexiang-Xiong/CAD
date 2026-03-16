"""
Cognitive Anomaly Detection (CAD) framework using GMM-based geometric detection.

This module implements the Phase 1 (Calibration) and Phase 2 (Diagnosis) of the CAD framework:
- Phase 1: Fit a Gaussian Mixture Model on the nominal cognitive state manifold
- Phase 2: Calculate hallucination scores as geometric anomalies (surprisal)

The detector is trained with weak supervision - only requiring ground-truth correct answers.
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from typing import Optional


class CognitiveAnomalyDetector:
    """
    Cognitive Anomaly Detection (CAD) framework.

    This detector projects a VLM's generative process onto an interpretable,
    low-dimensional cognitive state space and identifies hallucinations as
    geometric anomalies with high information-theoretic surprisal.

    Attributes:
        n_components: Number of Gaussian components in the GMM
        random_state: Random seed for reproducibility
    """

    def __init__(self, n_components: int = 5, random_state: int = 42):
        """
        Initialize the Cognitive Anomaly Detector.

        Args:
            n_components: Number of GMM components (K)
            random_state: Random seed for reproducibility
        """
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            random_state=random_state,
            n_init=10,
            reg_covar=1e-4
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.n_components = n_components

    def fit(self, X_nominal: np.ndarray):
        """
        Phase 1: Calibrate the nominal cognitive state manifold.

        Fit the GMM on samples where the model answered correctly (nominal states).
        This requires weak supervision - only ground-truth correct/incorrect labels.

        Args:
            X_nominal: Array of shape (n_samples, 3) containing [H_Evi, S_Conf, H_Ans]
                      for samples where the model answered correctly
        """
        if X_nominal.shape[0] < self.n_components:
            raise ValueError(
                f"Need at least {self.n_components} samples for fitting, "
                f"got {X_nominal.shape[0]}"
            )

        X_scaled = self.scaler.fit_transform(X_nominal)
        self.gmm.fit(X_scaled)
        self.is_fitted = True

    def predict_surprisal(self, X_test: np.ndarray) -> np.ndarray:
        """
        Phase 2: Calculate hallucination scores (surprisal).

        Compute the surprisal S_hall = -log p(v | M_GMM) for each test sample.
        Higher scores indicate higher likelihood of hallucination.

        Args:
            X_test: Array of shape (n_samples, 3) containing [H_Evi, S_Conf, H_Ans]

        Returns:
            Array of hallucination scores (surprisal values)
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction.")

        X_scaled = self.scaler.transform(X_test)
        # score_samples returns log probability, negated gives surprisal
        return -self.gmm.score_samples(X_scaled)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Calculate per-sample log-likelihood under the nominal model.

        Args:
            X_test: Array of shape (n_samples, 3)

        Returns:
            Log-likelihood scores
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction.")

        X_scaled = self.scaler.transform(X_test)
        return self.gmm.score_samples(X_scaled)

    def fit_predict(self, X_nominal: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        """
        Convenience method to fit on nominal data and predict on test data.

        Args:
            X_nominal: Nominal training data
            X_test: Test data for prediction

        Returns:
            Array of hallucination scores
        """
        self.fit(X_nominal)
        return self.predict_surprisal(X_test)


def select_optimal_k(X_nominal: np.ndarray, k_range: range) -> dict:
    """
    Select optimal number of GMM components using BIC criterion.

    Args:
        X_nominal: Training data for GMM fitting
        k_range: Range of K values to try

    Returns:
        Dictionary with 'k' (optimal K) and 'bic_scores' (BIC for each K)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_nominal)

    bic_scores = []
    for k in k_range:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type='full',
            random_state=42,
            n_init=10,
            reg_covar=1e-4
        )
        gmm.fit(X_scaled)
        bic_scores.append(gmm.bic(X_scaled))

    optimal_k = list(k_range)[np.argmin(bic_scores)]

    return {
        'k': optimal_k,
        'bic_scores': bic_scores,
        'k_range': list(k_range)
    }
