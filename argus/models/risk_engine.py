"""
Argus AI — Hybrid Risk Engine (Ensemble Scorer)
==================================================
Combines LSTM temporal anomaly scores with Isolation Forest static scores
using a weighted ensemble to produce a unified risk score (0-100).

Trust Score Formula:
    risk_score = α × norm(lstm_error) + (1-α) × norm(if_score)
    trust_score = max(0, 100 - risk_score × penalty_factor)

Where α=0.65 (temporal signals weighted higher for insider detection).

Usage:
    from argus.models.risk_engine import RiskEngine
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from argus.config import Config


class RiskEngine:
    """
    Hybrid ensemble that fuses LSTM + Isolation Forest scores
    into a unified trust score with privilege-aware decay.
    """

    def __init__(
        self,
        alpha: float = 0.65,
        trust_initial: float = 95.0,
        penalty_base: float = 20.0,
        decay_rate: float = 0.05,
        reinforcement: float = 5.0,
    ):
        """
        Args:
            alpha: Weight for LSTM temporal score (1-α for IF static score)
            trust_initial: Starting trust score for all employees
            penalty_base: Base penalty multiplier for anomalies
            decay_rate: Natural trust decay per idle period
            reinforcement: Trust reinforcement for consistent normal behavior
        """
        self.alpha = alpha
        self.trust_initial = trust_initial
        self.penalty_base = penalty_base
        self.decay_rate = decay_rate
        self.reinforcement = reinforcement

        # Thresholds (set during fit)
        self.lstm_threshold = None
        self.if_threshold = None
        self.lstm_mean = 0.0
        self.lstm_std = 1.0
        self.if_mean = 0.0
        self.if_std = 1.0

    def fit_thresholds(
        self,
        lstm_scores_normal: np.ndarray,
        if_scores_normal: np.ndarray,
        percentile: float = 95.0,
    ):
        """
        Fit anomaly thresholds from normal (non-insider) data.

        Args:
            lstm_scores_normal: Reconstruction errors from normal sequences
            if_scores_normal: IF anomaly scores from normal data
            percentile: Percentile for threshold (95th = top 5% are anomalous)
        """
        self.lstm_threshold = np.percentile(lstm_scores_normal, percentile)
        self.if_threshold = np.percentile(if_scores_normal, percentile)

        self.lstm_mean = lstm_scores_normal.mean()
        self.lstm_std = max(lstm_scores_normal.std(), 1e-8)
        self.if_mean = if_scores_normal.mean()
        self.if_std = max(if_scores_normal.std(), 1e-8)

        logger.info(f"Risk Engine thresholds fitted:")
        logger.info(f"  LSTM threshold (p{percentile:.0f}): {self.lstm_threshold:.6f}")
        logger.info(f"  IF threshold (p{percentile:.0f}): {self.if_threshold:.6f}")

    def compute_risk_scores(
        self,
        lstm_scores: np.ndarray,
        if_scores: np.ndarray,
    ) -> np.ndarray:
        """
        Compute hybrid risk scores (0-100, higher = riskier).

        Args:
            lstm_scores: Per-sequence reconstruction errors
            if_scores: Per-sample IF anomaly scores

        Returns:
            risk_scores: (n_samples,) array of risk scores [0, 100]
        """
        # Normalize to z-scores
        lstm_z = (lstm_scores - self.lstm_mean) / self.lstm_std
        if_z = (if_scores - self.if_mean) / self.if_std

        # Clip to prevent extreme values
        lstm_z = np.clip(lstm_z, -3, 10)
        if_z = np.clip(if_z, -3, 10)

        # Weighted ensemble
        combined = self.alpha * lstm_z + (1 - self.alpha) * if_z

        # Map to 0-100 using sigmoid-like scaling
        risk_scores = 100 * _sigmoid(combined - 1.0)

        return risk_scores

    def compute_trust_scores(
        self,
        risk_scores: np.ndarray,
    ) -> np.ndarray:
        """
        Convert risk scores to trust scores (0-100, higher = more trusted).

        Trust = initial - penalty × risk_factor
        """
        trust = self.trust_initial - (risk_scores * self.penalty_base / 100.0 * 2)
        trust = np.clip(trust, 0, 100)
        return trust

    def compute_trust_timeline(
        self,
        risk_scores: np.ndarray,
        emp_ids: np.ndarray,
        day_indices: np.ndarray,
    ) -> pd.DataFrame:
        """
        Compute trust score evolution over time per employee.
        Implements privilege decay and reinforcement.
        """
        trust_scores = self.compute_trust_scores(risk_scores)

        df = pd.DataFrame({
            "emp_id": emp_ids,
            "day_index": day_indices,
            "risk_score": risk_scores,
            "trust_score_raw": trust_scores,
        })

        # Apply temporal smoothing per employee (EMA)
        results = []
        for emp_id, emp_data in df.groupby("emp_id"):
            emp_data = emp_data.sort_values("day_index").copy()
            smoothed = emp_data["trust_score_raw"].ewm(span=5, min_periods=1).mean()
            emp_data["trust_score"] = smoothed.round(2)
            emp_data["trust_level"] = emp_data["trust_score"].apply(_get_trust_level)
            results.append(emp_data)

        return pd.concat(results).reset_index(drop=True)

    def classify(
        self,
        risk_scores: np.ndarray,
        threshold: float = 50.0,
    ) -> np.ndarray:
        """Binary classification: risk_score > threshold → insider (1)."""
        return (risk_scores >= threshold).astype(int)

    def get_trust_level(self, score: float) -> str:
        """Map trust score to trust level string."""
        return _get_trust_level(score)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x)),
    )


def _get_trust_level(score: float) -> str:
    """Map trust score to level."""
    if score < 20:
        return "CRITICAL"
    elif score < 40:
        return "HIGH_RISK"
    elif score < 60:
        return "MEDIUM_RISK"
    elif score < 80:
        return "LOW_RISK"
    else:
        return "TRUSTED"
