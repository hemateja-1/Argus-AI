"""
Argus AI — End-to-End Training Pipeline
=========================================
Trains the full hybrid model stack:
    1. Splits data into train/val/test
    2. Trains LSTM Autoencoder on normal sequences
    3. Trains Isolation Forest on normal feature vectors
    4. Calibrates the Risk Engine thresholds
    5. Evaluates on the full test set
    6. Saves all models and reports metrics

Usage:
    python -m argus.train
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report, precision_recall_curve,
)
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from argus.config import Config
from argus.models.lstm_autoencoder import (
    LSTMAutoencoder, train_autoencoder,
    compute_anomaly_scores, extract_twin_embeddings,
)
from argus.models.isolation_forest import (
    train_isolation_forest, compute_if_scores,
)
from argus.models.risk_engine import RiskEngine


def main():
    parser = argparse.ArgumentParser(description="Argus AI — Train Full Pipeline")
    parser.add_argument("--epochs", type=int, default=50, help="LSTM training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--alpha", type=float, default=0.65, help="LSTM weight in ensemble")
    args = parser.parse_args()

    Config.setup()
    np.random.seed(args.seed)

    processed_dir = Config.paths.PROCESSED_DATA
    models_dir = Config.paths.MODELS
    results_dir = Config.paths.RESULTS
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════
    #  STEP 1: Load data
    # ═══════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 1: Loading processed data...")
    logger.info("=" * 60)

    X_sequences = np.load(processed_dir / "X_sequences.npy")
    y_labels = np.load(processed_dir / "y_labels.npy")
    features_df = pd.read_csv(processed_dir / "features_47d.csv")
    seq_meta = pd.read_csv(processed_dir / "sequence_meta.csv")

    logger.info(f"  Sequences: {X_sequences.shape}")
    logger.info(f"  Labels: {y_labels.sum()} positive / {(1-y_labels).sum()} negative")
    logger.info(f"  Feature vectors: {len(features_df)}")

    # ═══════════════════════════════════════════════════════════
    #  STEP 2: Train/Val/Test split
    # ═══════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 2: Splitting data...")
    logger.info("=" * 60)

    # Split sequences: 70% train, 15% val, 15% test
    # Stratify by label to maintain class balance
    X_trainval, X_test, y_trainval, y_test, meta_trainval, meta_test = train_test_split(
        X_sequences, y_labels, seq_meta.to_dict("records"),
        test_size=0.15, random_state=args.seed, stratify=y_labels,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=0.176,  # 0.176 of 0.85 ≈ 0.15 of total
        random_state=args.seed, stratify=y_trainval,
    )

    # Normal-only training data (for autoencoder)
    X_train_normal = X_train[y_train == 0]

    logger.info(f"  Train: {X_train.shape[0]} ({y_train.sum()} pos)")
    logger.info(f"  Train (normal only): {X_train_normal.shape[0]}")
    logger.info(f"  Val:   {X_val.shape[0]} ({y_val.sum()} pos)")
    logger.info(f"  Test:  {X_test.shape[0]} ({y_test.sum()} pos)")

    # ═══════════════════════════════════════════════════════════
    #  STEP 3: Train LSTM Autoencoder
    # ═══════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 3: Training LSTM Autoencoder...")
    logger.info("=" * 60)

    X_val_normal = X_val[y_val == 0]

    lstm_model, lstm_history = train_autoencoder(
        X_train=X_train_normal,
        X_val=X_val_normal,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=Config.model.LSTM_PATIENCE,
        model_save_path=models_dir / "lstm_autoencoder.pt",
        device=Config.model.DEVICE,
    )

    # Get normalization params from saved model
    checkpoint = __import__("torch").load(models_dir / "lstm_autoencoder.pt", weights_only=False)
    lstm_mean = checkpoint["mean"]
    lstm_std = checkpoint["std"]

    # Compute LSTM anomaly scores on all splits
    lstm_scores_train = compute_anomaly_scores(lstm_model, X_train, lstm_mean, lstm_std, Config.model.DEVICE)
    lstm_scores_val = compute_anomaly_scores(lstm_model, X_val, lstm_mean, lstm_std, Config.model.DEVICE)
    lstm_scores_test = compute_anomaly_scores(lstm_model, X_test, lstm_mean, lstm_std, Config.model.DEVICE)

    logger.info(f"  LSTM scores — Normal mean: {lstm_scores_train[y_train==0].mean():.6f}, "
                f"Insider mean: {lstm_scores_train[y_train==1].mean():.6f}")

    # ═══════════════════════════════════════════════════════════
    #  STEP 4: Train Isolation Forest
    # ═══════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 4: Training Isolation Forest...")
    logger.info("=" * 60)

    # Use the last day of each sequence as the feature vector for IF
    X_train_static = X_train[:, -1, :]  # Last timestep
    X_val_static = X_val[:, -1, :]
    X_test_static = X_test[:, -1, :]

    feature_cols = _get_feature_columns()

    if_model, if_scaler, if_info = train_isolation_forest(
        X_train=X_train_static[y_train == 0],  # Train on normal only
        feature_names=feature_cols,
        n_estimators=Config.model.IF_N_ESTIMATORS,
        contamination=Config.model.IF_CONTAMINATION,
        seed=args.seed,
        model_save_path=models_dir / "isolation_forest.joblib",
    )

    # Compute IF scores on all splits
    if_scores_train = compute_if_scores(if_model, if_scaler, X_train_static)
    if_scores_val = compute_if_scores(if_model, if_scaler, X_val_static)
    if_scores_test = compute_if_scores(if_model, if_scaler, X_test_static)

    # ═══════════════════════════════════════════════════════════
    #  STEP 5: Calibrate Risk Engine
    # ═══════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 5: Calibrating Risk Engine...")
    logger.info("=" * 60)

    engine = RiskEngine(alpha=args.alpha)
    engine.fit_thresholds(
        lstm_scores_normal=lstm_scores_train[y_train == 0],
        if_scores_normal=if_scores_train[y_train == 0],
        percentile=Config.model.ANOMALY_PERCENTILE,
    )

    # Compute risk scores on test set
    risk_scores_test = engine.compute_risk_scores(lstm_scores_test, if_scores_test)
    trust_scores_test = engine.compute_trust_scores(risk_scores_test)

    # ═══════════════════════════════════════════════════════════
    #  STEP 6: Find optimal threshold & evaluate
    # ═══════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STEP 6: Evaluating on test set...")
    logger.info("=" * 60)

    # Use validation set to find optimal risk threshold
    risk_scores_val = engine.compute_risk_scores(lstm_scores_val, if_scores_val)

    best_f1 = 0
    best_threshold = 50.0
    for thresh in np.arange(20, 80, 1):
        preds = (risk_scores_val >= thresh).astype(int)
        if preds.sum() == 0:
            continue
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    logger.info(f"  Optimal threshold (from val): {best_threshold:.1f} (F1={best_f1:.4f})")

    # ─── Test set metrics ───
    y_pred = (risk_scores_test >= best_threshold).astype(int)

    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    test_precision = precision_score(y_test, y_pred, zero_division=0)
    test_recall = recall_score(y_test, y_pred, zero_division=0)
    test_auc = roc_auc_score(y_test, risk_scores_test) if y_test.sum() > 0 else 0.0

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    fpr = fp / max(1, fp + tn)

    logger.success("=" * 60)
    logger.success("RESULTS — Test Set Performance")
    logger.success("=" * 60)
    logger.info(f"  F1 Score:     {test_f1:.4f}")
    logger.info(f"  Precision:    {test_precision:.4f}")
    logger.info(f"  Recall:       {test_recall:.4f}")
    logger.info(f"  AUC-ROC:      {test_auc:.4f}")
    logger.info(f"  FPR:          {fpr:.4f} ({fpr*100:.2f}%)")
    logger.info(f"  Confusion Matrix:")
    logger.info(f"    TP={tp}  FP={fp}")
    logger.info(f"    FN={fn}  TN={tn}")

    # ─── Also evaluate individual models ───
    lstm_only_preds = (lstm_scores_test > np.percentile(lstm_scores_train[y_train == 0], 95)).astype(int)
    if_only_preds = (if_scores_test > np.percentile(if_scores_train[y_train == 0], 95)).astype(int)

    lstm_f1 = f1_score(y_test, lstm_only_preds, zero_division=0)
    if_f1 = f1_score(y_test, if_only_preds, zero_division=0)

    logger.info(f"\n  Ablation:")
    logger.info(f"    LSTM only F1:    {lstm_f1:.4f}")
    logger.info(f"    IF only F1:      {if_f1:.4f}")
    logger.info(f"    Hybrid F1:       {test_f1:.4f}  ← {'✅ BEST' if test_f1 >= max(lstm_f1, if_f1) else '⚠️'}")

    # ═══════════════════════════════════════════════════════════
    #  STEP 7: Save results
    # ═══════════════════════════════════════════════════════════

    # Save risk engine
    joblib.dump(engine, models_dir / "risk_engine.joblib")

    # Save metrics
    metrics = {
        "model": "Hybrid (LSTM + IF)",
        "alpha": args.alpha,
        "threshold": float(best_threshold),
        "test": {
            "f1": float(test_f1),
            "precision": float(test_precision),
            "recall": float(test_recall),
            "auc_roc": float(test_auc),
            "fpr": float(fpr),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        },
        "ablation": {
            "lstm_only_f1": float(lstm_f1),
            "if_only_f1": float(if_f1),
            "hybrid_f1": float(test_f1),
        },
        "lstm_training": {
            "epochs_run": len(lstm_history["train_loss"]),
            "final_train_loss": float(lstm_history["train_loss"][-1]),
            "final_val_loss": float(lstm_history["val_loss"][-1]) if lstm_history["val_loss"] else None,
        },
    }

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save test predictions for analysis
    test_results = pd.DataFrame({
        "lstm_score": lstm_scores_test,
        "if_score": if_scores_test,
        "risk_score": risk_scores_test,
        "trust_score": trust_scores_test,
        "y_true": y_test,
        "y_pred": y_pred,
    })
    test_results.to_csv(results_dir / "test_predictions.csv", index=False)

    # Extract and save twin embeddings for visualization
    logger.info("Extracting Digital Twin embeddings...")
    embeddings = extract_twin_embeddings(lstm_model, X_test, lstm_mean, lstm_std, Config.model.DEVICE)
    np.save(results_dir / "twin_embeddings.npy", embeddings)

    logger.success(f"\n✅ Full pipeline complete! All artifacts saved to:")
    logger.info(f"   Models:  {models_dir}")
    logger.info(f"   Results: {results_dir}")

    return metrics


def _get_feature_columns() -> list[str]:
    """Feature column names (duplicated here to avoid circular import)."""
    return [
        "login_hour", "logout_hour", "session_duration_hrs", "is_weekend",
        "is_after_hours", "time_since_last_session", "login_regularity_score",
        "temporal_entropy",
        "files_accessed", "emails_sent", "emails_received", "urls_visited",
        "usb_events", "data_volume_mb", "unique_systems_accessed",
        "is_new_device", "device_count", "unique_pcs", "geo_anomaly_flag",
        "vpn_usage",
        "external_email_ratio", "avg_attachment_size", "unique_recipients",
        "cc_bcc_ratio", "email_content_sentiment", "unusual_recipient_flag",
        "file_copy_count", "usb_file_transfers", "large_download_flag",
        "sensitive_file_access", "data_egress_volume", "print_count",
        "cloud_upload_count",
        "access_to_role_ratio", "peer_deviation_score", "weekday_vs_weekend_ratio",
        "morning_vs_evening_ratio", "productive_vs_idle_ratio",
        "command_diversity_index",
        "action_sequence_entropy", "longest_unusual_chain",
        "role_boundary_crossings", "privilege_escalation_count",
        "session_action_diversity", "repeat_pattern_score",
        "novelty_score", "behavioral_velocity",
    ]


if __name__ == "__main__":
    main()
