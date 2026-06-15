"""
Argus AI — Isolation Forest for Static Feature Anomaly Detection
==================================================================
Detects point-in-time anomalies from the 47-dimensional feature vectors.
Complements the LSTM's temporal anomaly detection.

Research Basis:
    - Liu et al. (2008): "Isolation Forest"
    - Le & Zincir-Heywood (2021): "Ensemble insider detection"

Usage:
    from argus.models.isolation_forest import train_isolation_forest
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from argus.config import Config


def train_isolation_forest(
    X_train: np.ndarray,
    feature_names: list[str] | None = None,
    n_estimators: int = 500,
    contamination: float = 0.05,
    max_features: float = 1.0,
    seed: int = 42,
    model_save_path: Path | None = None,
) -> tuple[IsolationForest, StandardScaler, dict]:
    """
    Train Isolation Forest on feature vectors.

    Args:
        X_train: (n_samples, 47) — feature matrix
        feature_names: Optional feature names for importance analysis
        n_estimators: Number of isolation trees
        contamination: Expected anomaly fraction
        max_features: Max features per tree
        seed: Random seed
        model_save_path: Path to save model

    Returns:
        (model, scaler, info)
    """
    logger.info(f"Training Isolation Forest")
    logger.info(f"  Samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
    logger.info(f"  Trees: {n_estimators}, Contamination: {contamination}")

    # ─── Standardize ───
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # ─── Train ───
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_features=max_features,
        random_state=seed,
        n_jobs=-1,
        verbose=0,
    )
    model.fit(X_scaled)

    # ─── Anomaly scores ───
    train_scores = -model.decision_function(X_scaled)  # Higher = more anomalous
    train_preds = model.predict(X_scaled)
    n_anomalies = (train_preds == -1).sum()

    # ─── Feature importance (path-length based) ───
    feature_importance = {}
    if feature_names and hasattr(model, 'estimators_features_'):
        importances = np.zeros(X_train.shape[1])
        for tree, features_idx in zip(model.estimators_, model.estimators_features_):
            tree_importances = tree.feature_importances_
            for i, feat_idx in enumerate(features_idx):
                if i < len(tree_importances):
                    importances[feat_idx] += tree_importances[i]
        importances /= len(model.estimators_)
        feature_importance = dict(zip(feature_names, importances))
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    info = {
        "n_estimators": n_estimators,
        "contamination": contamination,
        "n_samples": X_train.shape[0],
        "n_anomalies_train": int(n_anomalies),
        "anomaly_rate_train": float(n_anomalies / len(train_preds)),
        "score_mean": float(train_scores.mean()),
        "score_std": float(train_scores.std()),
        "score_p95": float(np.percentile(train_scores, 95)),
        "score_p99": float(np.percentile(train_scores, 99)),
        "feature_importance": feature_importance,
    }

    # ─── Save ───
    if model_save_path:
        model_save_path = Path(model_save_path)
        joblib.dump({
            "model": model,
            "scaler": scaler,
            "info": info,
        }, model_save_path)
        logger.info(f"  Model saved to: {model_save_path}")

    logger.success(f"✅ Isolation Forest training complete!")
    logger.info(f"   Anomalies detected (train): {n_anomalies} ({n_anomalies/len(train_preds)*100:.2f}%)")
    logger.info(f"   Score range: [{train_scores.min():.4f}, {train_scores.max():.4f}]")
    logger.info(f"   95th percentile: {info['score_p95']:.4f}")

    return model, scaler, info


def compute_if_scores(
    model: IsolationForest,
    scaler: StandardScaler,
    X: np.ndarray,
) -> np.ndarray:
    """Compute anomaly scores for new data. Higher = more anomalous."""
    X_scaled = scaler.transform(X)
    scores = -model.decision_function(X_scaled)
    return scores
