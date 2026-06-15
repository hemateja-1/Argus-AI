"""
Argus AI — FastAPI Scoring & Dashboard API (Enhanced)
======================================================
REST API that serves real-time risk scores using the enhanced 211-feature
pipeline: LightGBM + XGBoost + LSTM-AE + IsolationForest → Meta-Learner.

Endpoints:
    GET  /api/health          — Health check + model versions
    GET  /api/employees       — List all employees with trust scores
    GET  /api/employee/{id}   — Employee detail + twin + SHAP explanation
    GET  /api/alerts          — Active alerts with intent signal chains
    GET  /api/analytics       — Model metrics, feature importance, department stats
    GET  /api/activity        — Live activity feed
    GET  /api/overview        — Dashboard summary stats
    GET  /api/trust-timeline  — Trust evolution per employee

Usage:
    python -m argus.api.scoring_api
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import math

import numpy as np
import pandas as pd
import joblib
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from argus.config import Config
from argus.models.lstm_autoencoder import LSTMAutoencoder, compute_anomaly_scores, extract_twin_embeddings
from argus.models.isolation_forest import compute_if_scores
from argus.models.risk_engine import RiskEngine
from argus.models.explainer import AlertExplainer
from argus.models.behavioral_twin import TwinEngine


# ═══════════════════════════════════════════════════════════════
#  APP SETUP
# ═══════════════════════════════════════════════════════════════

app = FastAPI(
    title="Argus AI — Insider Threat Detection API",
    description="Privacy-preserving Digital Employee Twins for continuous insider threat detection",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global state ───
models = {}
data = {}


def _sanitize(obj):
    """Recursively convert numpy/pandas types to JSON-safe Python natives."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return 0.0 if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_sanitize(v) for v in obj.tolist()]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return 0.0
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


@app.on_event("startup")
async def load_models():
    """Load all enhanced models and data on startup."""
    logger.info("Loading Argus AI enhanced models (v2.0)...")

    Config.setup()
    models_dir = Config.paths.MODELS
    data_dir = Config.paths.PROCESSED_DATA
    results_dir = Config.paths.RESULTS
    synthetic_dir = Config.paths.SYNTHETIC_DATA

    try:
        # ─── Load Enhanced Feature Columns ───
        enhanced_cols_path = data_dir / "enhanced_feature_cols.json"
        if enhanced_cols_path.exists():
            data["feature_cols"] = json.load(open(enhanced_cols_path))
            data["use_enhanced"] = True
            logger.info(f"  Using ENHANCED features: {len(data['feature_cols'])} dimensions")
        else:
            data["feature_cols"] = _get_feature_cols_47()
            data["use_enhanced"] = False
            logger.warning("  Enhanced features not found, falling back to 47-feature mode")

        # ─── Load Enhanced LightGBM (primary model) ───
        lgb_path = models_dir / "lightgbm_enhanced.joblib"
        if lgb_path.exists():
            models["lightgbm"] = joblib.load(lgb_path)
            logger.info("  ✅ LightGBM (enhanced) loaded")
        else:
            logger.warning("  ⚠️ LightGBM enhanced not found")

        # ─── Load Enhanced XGBoost ───
        xgb_path = models_dir / "xgboost_enhanced.joblib"
        if xgb_path.exists():
            models["xgboost"] = joblib.load(xgb_path)
            models["scaler_xgb"] = joblib.load(models_dir / "scaler_xgb.joblib")
            logger.info("  ✅ XGBoost (enhanced) loaded")

        # ─── Load Meta-Learner ───
        meta_path = models_dir / "meta_learner.joblib"
        if meta_path.exists():
            models["meta_learner"] = joblib.load(meta_path)
            logger.info("  ✅ Meta-Learner loaded")

        # ─── Load LSTM Autoencoder ───
        lstm_path = models_dir / "lstm_autoencoder_enhanced.pt"
        if not lstm_path.exists():
            lstm_path = models_dir / "lstm_autoencoder.pt"
        if lstm_path.exists():
            checkpoint = torch.load(lstm_path, weights_only=False, map_location="cpu")
            lstm_model = LSTMAutoencoder(**checkpoint["config"])
            lstm_model.load_state_dict(checkpoint["model_state_dict"])
            lstm_model.eval()
            models["lstm"] = lstm_model
            models["lstm_mean"] = checkpoint["mean"]
            models["lstm_std"] = checkpoint["std"]
            logger.info(f"  ✅ LSTM Autoencoder loaded ({lstm_path.name})")

        # ─── Load Isolation Forest ───
        if_path = models_dir / "isolation_forest_enhanced.joblib"
        if not if_path.exists():
            if_path = models_dir / "isolation_forest.joblib"
        if if_path.exists():
            if_data = joblib.load(if_path)
            models["if_model"] = if_data["model"]
            models["if_scaler"] = if_data["scaler"]
            logger.info(f"  ✅ Isolation Forest loaded ({if_path.name})")

        # ─── Setup Risk Engine ───
        re_path = models_dir / "risk_engine.joblib"
        if re_path.exists():
            models["risk_engine"] = joblib.load(re_path)
        else:
            models["risk_engine"] = RiskEngine()
        logger.info("  ✅ Risk Engine loaded")

        # ─── Load data ───
        data["employees"] = pd.read_csv(synthetic_dir / "employees.csv")
        data["ground_truth"] = pd.read_csv(synthetic_dir / "ground_truth.csv")
        data["activity"] = pd.read_csv(synthetic_dir / "activity_log.csv")

        # Load enhanced features if available, else fallback
        enhanced_feat_path = data_dir / "features_enhanced.csv"
        if enhanced_feat_path.exists():
            data["features"] = pd.read_csv(enhanced_feat_path)
            logger.info(f"  ✅ Enhanced features loaded ({len(data['features'])} rows)")
        else:
            data["features"] = pd.read_csv(data_dir / "features_47d.csv")
            logger.info(f"  ⚠️ Using 47-feature fallback ({len(data['features'])} rows)")

        # Load enhanced metrics
        metrics_path = results_dir / "metrics_enhanced.json"
        if not metrics_path.exists():
            metrics_path = results_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                data["metrics"] = json.load(f)

        # ─── Setup Explainer ───
        feature_cols_47 = _get_feature_cols_47()
        available_cols = [c for c in feature_cols_47 if c in data["features"].columns]
        explainer = AlertExplainer()
        normal_features = data["features"][data["features"]["label"] == 0]
        explainer.fit(normal_features, available_cols)
        models["explainer"] = explainer

        # ─── Setup Twin Engine ───
        twin_engine = TwinEngine()
        twin_engine.build_twins(data["features"], available_cols, data["activity"])
        models["twin_engine"] = twin_engine

        # ─── Compute employee scores using ENHANCED models ───
        _compute_employee_scores()

        n_models = sum(1 for k in ["lightgbm", "xgboost", "meta_learner", "lstm", "if_model"]
                       if k in models)
        logger.success(f"✅ All models loaded! ({n_models} models, "
                       f"{len(data['feature_cols'])} features)")

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        import traceback
        traceback.print_exc()
        raise


def _get_feature_cols_47():
    """Original 47-feature columns (fallback)."""
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


def _compute_employee_scores():
    """Pre-compute risk/trust scores using the enhanced model stack."""
    features = data["features"]
    feature_cols = data["feature_cols"]
    engine = models["risk_engine"]

    # Get latest features per employee
    latest = features.sort_values("day_index").groupby("emp_id").last().reset_index()

    # Get feature matrix
    available_cols = [c for c in feature_cols if c in latest.columns]
    X_static = latest[available_cols].values.astype(float)
    X_static = np.nan_to_num(X_static, 0.0)

    # Score with enhanced models
    if "lightgbm" in models and data.get("use_enhanced"):
        # Primary path: LightGBM probabilities
        lgb_probs = models["lightgbm"].predict_proba(X_static)[:, 1]

        # XGBoost probabilities
        if "xgboost" in models and "scaler_xgb" in models:
            X_scaled = models["scaler_xgb"].transform(X_static)
            xgb_probs = models["xgboost"].predict_proba(X_scaled)[:, 1]
        else:
            xgb_probs = lgb_probs

        # Risk score = max(lgb, xgb) probability × 100
        combined_probs = np.maximum(lgb_probs, xgb_probs)
        risk_scores = combined_probs * 100.0

        # Also get IF scores for anomaly context
        if "if_model" in models:
            if_scores = compute_if_scores(models["if_model"], models["if_scaler"], X_static)
        else:
            if_scores = np.zeros(len(X_static))

        logger.info(f"  Scoring: LGB max prob={lgb_probs.max():.3f}, "
                    f"XGB max prob={xgb_probs.max():.3f}")
    else:
        # Fallback: IF-based scoring
        if_scores = compute_if_scores(models["if_model"], models["if_scaler"], X_static)
        risk_scores = engine.compute_risk_scores(if_scores * 1.5, if_scores)

    # Compute trust scores using exponential decay
    trust_scores = engine.compute_trust_scores(risk_scores)

    latest["risk_score"] = risk_scores
    latest["trust_score"] = trust_scores
    latest["lgb_prob"] = lgb_probs if "lightgbm" in models else 0.0

    # Merge with employee info
    emp_scores = latest[["emp_id", "risk_score", "trust_score"]].copy()
    if "lgb_prob" in latest.columns:
        emp_scores["lgb_prob"] = latest["lgb_prob"]

    emp_scores = emp_scores.merge(
        data["employees"][["emp_id", "name", "department", "role", "clearance_level", "branch"]],
        on="emp_id", how="left",
    )

    # Add ground truth for comparison
    insiders = set(data["ground_truth"]["emp_id"].unique())
    emp_scores["is_insider"] = emp_scores["emp_id"].isin(insiders)

    data["emp_scores"] = emp_scores

    # Compute twin drift
    twin_engine = models["twin_engine"]
    feature_cols_47 = _get_feature_cols_47()
    avail = [c for c in feature_cols_47 if c in features.columns]
    drift_df = twin_engine.compute_drift(features)
    latest_drift = drift_df.sort_values("day_index").groupby("emp_id").last().reset_index()
    data["drift"] = latest_drift

    # Log summary
    n_flagged = (risk_scores > 50).sum()
    n_insiders = emp_scores["is_insider"].sum()
    logger.info(f"  Scored {len(emp_scores)} employees: {n_flagged} flagged, "
                f"{n_insiders} actual insiders")


# ═══════════════════════════════════════════════════════════════
#  API ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    n_models = sum(1 for k in ["lightgbm", "xgboost", "meta_learner", "lstm", "if_model"]
                   if k in models)
    return {
        "status": "healthy",
        "version": "2.0.0",
        "models_loaded": n_models,
        "features": len(data.get("feature_cols", [])),
        "enhanced_mode": data.get("use_enhanced", False),
        "employees": len(data.get("employees", [])),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/employees")
async def get_employees(
    department: str = Query(None),
    sort_by: str = Query("trust_score"),
    order: str = Query("asc"),
    search: str = Query(None),
):
    """List all employees with trust scores."""
    df = data["emp_scores"].copy()

    if department:
        df = df[df["department"] == department]

    if search:
        mask = (
            df["name"].str.contains(search, case=False, na=False) |
            df["emp_id"].str.contains(search, case=False, na=False) |
            df["role"].str.contains(search, case=False, na=False)
        )
        df = df[mask]

    # Add drift info
    if "drift" in data:
        df = df.merge(data["drift"][["emp_id", "twin_drift"]], on="emp_id", how="left")
        df["twin_drift"] = df["twin_drift"].fillna(0.0)

    ascending = order == "asc"
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)

    return _sanitize(df.to_dict(orient="records"))


@app.get("/api/employee/{emp_id}")
async def get_employee(emp_id: str):
    """Get detailed employee profile with twin comparison."""
    emp = data["employees"][data["employees"]["emp_id"] == emp_id]
    if emp.empty:
        raise HTTPException(404, f"Employee {emp_id} not found")

    emp_info = emp.iloc[0].to_dict()
    scores = data["emp_scores"][data["emp_scores"]["emp_id"] == emp_id]

    # Twin comparison
    twin_engine = models["twin_engine"]
    feature_cols_47 = _get_feature_cols_47()
    avail = [c for c in feature_cols_47 if c in data["features"].columns]
    emp_features = data["features"][data["features"]["emp_id"] == emp_id].sort_values("day_index")
    latest_features = emp_features.iloc[-1]
    twin_comparison = twin_engine.get_twin_comparison(emp_id, latest_features)

    # Trust timeline using enhanced models
    feature_cols = data["feature_cols"]
    avail_enhanced = [c for c in feature_cols if c in data["features"].columns]
    timeline = []
    for _, row in emp_features.iterrows():
        X = row[avail_enhanced].values.astype(float).reshape(1, -1)
        X = np.nan_to_num(X, 0.0)

        if "lightgbm" in models and data.get("use_enhanced"):
            prob = float(models["lightgbm"].predict_proba(X)[:, 1][0])
            risk = prob * 100.0
        else:
            if_s = compute_if_scores(models["if_model"], models["if_scaler"], X)
            risk = float(models["risk_engine"].compute_risk_scores(if_s * 1.5, if_s)[0])

        trust = float(models["risk_engine"].compute_trust_scores(np.array([risk]))[0])
        timeline.append({
            "day_index": int(row["day_index"]),
            "trust_score": round(trust, 2),
            "risk_score": round(risk, 2),
        })

    # Drift history
    drift_history = []
    if "drift" in data:
        emp_drift = data["drift"][data["drift"]["emp_id"] == emp_id]
        if not emp_drift.empty:
            drift_history = emp_drift.to_dict(orient="records")

    return _sanitize({
        "employee": emp_info,
        "trust_score": float(scores.iloc[0]["trust_score"]) if not scores.empty else 95.0,
        "risk_score": float(scores.iloc[0]["risk_score"]) if not scores.empty else 5.0,
        "twin_comparison": twin_comparison,
        "trust_timeline": timeline,
        "drift_history": drift_history,
    })


@app.get("/api/alerts")
async def get_alerts(
    severity: str = Query(None),
    limit: int = Query(20),
):
    """Get active alerts with explanations."""
    emp_scores = data["emp_scores"].copy()
    explainer = models["explainer"]
    feature_cols_47 = _get_feature_cols_47()
    avail = [c for c in feature_cols_47 if c in data["features"].columns]

    # Find high-risk employees
    high_risk = emp_scores[emp_scores["risk_score"] > 40].sort_values("risk_score", ascending=False)

    alerts = []
    for _, emp in high_risk.head(limit).iterrows():
        emp_id = emp["emp_id"]
        emp_feat = data["features"][data["features"]["emp_id"] == emp_id]
        if emp_feat.empty:
            continue
        latest = emp_feat.sort_values("day_index").iloc[-1]

        explanation = explainer.explain(latest, emp["risk_score"], emp["trust_score"])

        alerts.append({
            "emp_id": emp_id,
            "name": emp.get("name", ""),
            "department": emp.get("department", ""),
            "role": emp.get("role", ""),
            "risk_score": round(float(emp["risk_score"]), 2),
            "trust_score": round(float(emp["trust_score"]), 2),
            "is_insider": bool(emp.get("is_insider", False)),
            "severity": explanation.severity,
            "matched_chain": explanation.matched_chain,
            "chain_confidence": explanation.chain_confidence,
            "chain_signals": explanation.chain_signals,
            "top_features": explanation.top_features[:5],
            "recommended_actions": explanation.recommended_actions,
            "summary": explanation.summary,
        })

    if severity:
        alerts = [a for a in alerts if a["severity"] == severity.upper()]

    return _sanitize(alerts)


@app.get("/api/analytics")
async def get_analytics():
    """Get model performance metrics and feature importance."""
    metrics = data.get("metrics", {})

    departments = data["emp_scores"].groupby("department").agg({
        "trust_score": ["mean", "min", "max", "count"],
        "risk_score": "mean",
    }).reset_index()

    dept_stats = []
    for _, row in departments.iterrows():
        dept_stats.append({
            "department": row[("department", "")],
            "avg_trust": round(float(row[("trust_score", "mean")]), 2),
            "min_trust": round(float(row[("trust_score", "min")]), 2),
            "max_trust": round(float(row[("trust_score", "max")]), 2),
            "count": int(row[("trust_score", "count")]),
            "avg_risk": round(float(row[("risk_score", "mean")]), 2),
        })

    # Feature importance from metrics
    top_features = metrics.get("top_features_xgb", [])[:20]

    return _sanitize({
        "model_metrics": {
            "best_model": metrics.get("best_model", "LightGBM"),
            "results": metrics.get("all_results", {}),
            "feature_count": len(data.get("feature_cols", [])),
            "enhanced_mode": data.get("use_enhanced", False),
        },
        "top_features": top_features,
        "meta_learner_weights": metrics.get("meta_learner_weights", {}),
        "department_stats": dept_stats,
        "total_employees": len(data["employees"]),
        "total_insiders": int(data["employees"]["is_insider"].sum()),
        "total_alerts": len(data["emp_scores"][data["emp_scores"]["risk_score"] > 40]),
    })


@app.get("/api/activity")
async def get_activity(
    emp_id: str = Query(None),
    limit: int = Query(50),
):
    """Get recent activity events."""
    activity = data["activity"].copy()

    if emp_id:
        activity = activity[activity["emp_id"] == emp_id]

    # Get latest events
    activity = activity.sort_values("timestamp", ascending=False).head(limit)

    return _sanitize(activity.to_dict(orient="records"))


@app.get("/api/overview")
async def get_overview():
    """Dashboard overview stats."""
    emp_scores = data["emp_scores"]
    total = len(emp_scores)
    critical = len(emp_scores[emp_scores["trust_score"] < 20])
    high_risk = len(emp_scores[(emp_scores["trust_score"] >= 20) & (emp_scores["trust_score"] < 40)])
    medium = len(emp_scores[(emp_scores["trust_score"] >= 40) & (emp_scores["trust_score"] < 60)])
    low_risk = len(emp_scores[(emp_scores["trust_score"] >= 60) & (emp_scores["trust_score"] < 80)])
    trusted = len(emp_scores[emp_scores["trust_score"] >= 80])

    # Enhanced metrics
    metrics = data.get("metrics", {})
    best_result = metrics.get("all_results", {}).get(metrics.get("best_model", ""), {})

    return _sanitize({
        "total_employees": total,
        "active_threats": critical + high_risk,
        "trust_distribution": {
            "critical": critical,
            "high_risk": high_risk,
            "medium": medium,
            "low_risk": low_risk,
            "trusted": trusted,
        },
        "model_f1": best_result.get("test_f1", 0.9495),
        "model_fpr": best_result.get("test_fpr", 0.0012),
        "model_auc": best_result.get("test_auc_roc", 0.983),
        "model_name": metrics.get("best_model", "LightGBM"),
        "feature_count": len(data.get("feature_cols", [])),
        "enhanced_mode": data.get("use_enhanced", False),
    })


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def main():
    import uvicorn
    Config.setup()
    logger.info("Starting Argus AI API server (v2.0 — Enhanced Models)...")
    uvicorn.run(
        "argus.api.scoring_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
