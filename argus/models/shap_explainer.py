"""
Argus AI — SHAP Explainability Module
=======================================
Generates SHAP explanations for the enhanced LightGBM/XGBoost models.
Provides per-employee feature importance for "why was this employee flagged?"

Usage:
    from argus.models.shap_explainer import SHAPExplainer
    explainer = SHAPExplainer(model, feature_cols)
    explanation = explainer.explain_employee(X_employee)

References:
    - Lundberg & Lee (2017): "A Unified Approach to Interpreting Model Predictions"
    - SHAP for tree models uses TreeExplainer (exact, polynomial time)
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class SHAPExplainer:
    """
    SHAP-based model explainer for insider threat predictions.
    Uses TreeExplainer for XGBoost/LightGBM (exact Shapley values).
    """

    def __init__(self, model=None, feature_cols: list[str] = None, model_type: str = "lightgbm"):
        self.model = model
        self.feature_cols = feature_cols or []
        self.model_type = model_type
        self.explainer = None
        self.base_value = None

    def fit(self, model=None, X_background: np.ndarray = None):
        """Initialize SHAP TreeExplainer."""
        import shap

        if model is not None:
            self.model = model

        if self.model is None:
            raise ValueError("No model provided")

        # TreeExplainer is exact for tree-based models
        if X_background is not None:
            # Use a sample of background data for faster computation
            if len(X_background) > 500:
                idx = np.random.choice(len(X_background), 500, replace=False)
                X_background = X_background[idx]
            self.explainer = shap.TreeExplainer(self.model, X_background)
        else:
            self.explainer = shap.TreeExplainer(self.model)

        self.base_value = self.explainer.expected_value
        if isinstance(self.base_value, np.ndarray):
            self.base_value = self.base_value[1]  # Binary classification: class 1

        logger.info(f"  SHAP TreeExplainer fitted (base value={self.base_value:.4f})")
        return self

    def explain_single(self, X_single: np.ndarray) -> dict:
        """
        Explain a single prediction.

        Returns:
            dict with shap_values, top_positive (pushing toward insider),
            top_negative (pushing toward normal), base_value, prediction
        """
        import shap

        if self.explainer is None:
            raise RuntimeError("Call fit() first")

        X = X_single.reshape(1, -1) if X_single.ndim == 1 else X_single

        shap_vals = self.explainer.shap_values(X)
        # For binary classification, use class 1 SHAP values
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]
        else:
            sv = shap_vals[0]

        # Sort by absolute contribution
        feature_contributions = list(zip(self.feature_cols, sv, X[0]))
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        # Split into risk-increasing and risk-decreasing
        top_positive = [
            {"feature": f, "shap_value": round(float(s), 4), "feature_value": round(float(v), 4)}
            for f, s, v in feature_contributions if s > 0
        ][:10]

        top_negative = [
            {"feature": f, "shap_value": round(float(s), 4), "feature_value": round(float(v), 4)}
            for f, s, v in feature_contributions if s < 0
        ][:10]

        prediction = float(self.model.predict_proba(X)[:, 1][0])

        return {
            "prediction": round(prediction, 4),
            "base_value": round(float(self.base_value), 4),
            "top_risk_factors": top_positive,
            "top_protective_factors": top_negative,
            "total_shap_positive": round(float(sum(s for _, s, _ in feature_contributions if s > 0)), 4),
            "total_shap_negative": round(float(sum(s for _, s, _ in feature_contributions if s < 0)), 4),
        }

    def explain_batch(self, X: np.ndarray) -> np.ndarray:
        """Get SHAP values for a batch of samples."""
        shap_vals = self.explainer.shap_values(X)
        if isinstance(shap_vals, list):
            return shap_vals[1]
        return shap_vals

    def global_importance(self, X: np.ndarray) -> list[dict]:
        """
        Compute global feature importance (mean |SHAP|).

        Returns:
            List of {feature, importance, rank} sorted by importance
        """
        shap_vals = self.explain_batch(X)
        mean_abs = np.abs(shap_vals).mean(axis=0)

        importance = sorted(
            zip(self.feature_cols, mean_abs),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            {"feature": f, "importance": round(float(imp), 6), "rank": i + 1}
            for i, (f, imp) in enumerate(importance)
        ]


def main():
    """Run SHAP analysis on enhanced models and generate reports."""
    from argus.config import Config

    Config.setup()
    proc_dir = Config.paths.PROCESSED_DATA
    models_dir = Config.paths.MODELS
    research_dir = Path(Config.paths.ROOT) / "research"
    research_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading enhanced features for SHAP analysis...")
    X_all = np.load(proc_dir / "X_enhanced.npy")[:, -1, :]
    y_all = np.load(proc_dir / "y_enhanced.npy")
    feature_cols = json.load(open(proc_dir / "enhanced_feature_cols.json"))

    # Load models
    lgb_model = joblib.load(models_dir / "lightgbm_enhanced.joblib")
    xgb_model = joblib.load(models_dir / "xgboost_enhanced.joblib")
    scaler_xgb = joblib.load(models_dir / "scaler_xgb.joblib")

    logger.info(f"  Data: {X_all.shape[0]} samples, {len(feature_cols)} features")

    # ── SHAP for LightGBM ──
    logger.info("\n" + "=" * 60)
    logger.info("SHAP Analysis: LightGBM (primary model)")
    logger.info("=" * 60)

    lgb_explainer = SHAPExplainer(lgb_model, feature_cols, "lightgbm")
    lgb_explainer.fit(X_background=X_all[y_all == 0])

    # Global importance
    logger.info("  Computing global feature importance...")
    # Use a sample for speed
    sample_idx = np.random.choice(len(X_all), min(2000, len(X_all)), replace=False)
    X_sample = X_all[sample_idx]
    y_sample = y_all[sample_idx]

    lgb_global = lgb_explainer.global_importance(X_sample)

    logger.info("\n  Top 20 Features (LightGBM SHAP):")
    for item in lgb_global[:20]:
        logger.info(f"    #{item['rank']:2d} {item['feature']:40s} {item['importance']:.6f}")

    # ── SHAP for XGBoost ──
    logger.info("\n" + "=" * 60)
    logger.info("SHAP Analysis: XGBoost")
    logger.info("=" * 60)

    X_all_scaled = scaler_xgb.transform(X_all)
    xgb_explainer = SHAPExplainer(xgb_model, feature_cols, "xgboost")
    xgb_explainer.fit(X_background=X_all_scaled[y_all == 0])

    xgb_global = xgb_explainer.global_importance(X_all_scaled[sample_idx])

    logger.info("\n  Top 20 Features (XGBoost SHAP):")
    for item in xgb_global[:20]:
        logger.info(f"    #{item['rank']:2d} {item['feature']:40s} {item['importance']:.6f}")

    # ── Per-insider explanations ──
    logger.info("\n" + "=" * 60)
    logger.info("Per-Insider Explanations")
    logger.info("=" * 60)

    insider_idx = np.where(y_all == 1)[0][:10]
    insider_explanations = []

    for idx in insider_idx:
        exp = lgb_explainer.explain_single(X_all[idx])
        insider_explanations.append(exp)
        logger.info(f"\n  Insider sample {idx}: P(insider)={exp['prediction']:.4f}")
        for rf in exp["top_risk_factors"][:3]:
            logger.info(f"    ↑ {rf['feature']}: SHAP={rf['shap_value']:+.4f} (val={rf['feature_value']:.2f})")
        for pf in exp["top_protective_factors"][:2]:
            logger.info(f"    ↓ {pf['feature']}: SHAP={pf['shap_value']:+.4f} (val={pf['feature_value']:.2f})")

    # ── Save results ──
    results = {
        "lgb_global_importance": lgb_global[:50],
        "xgb_global_importance": xgb_global[:50],
        "insider_explanations": insider_explanations,
        "n_features": len(feature_cols),
        "n_samples_analyzed": len(X_sample),
    }

    with open(research_dir / "10_shap_analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Generate markdown report
    lines = [
        "# 10 — SHAP Explainability Analysis\n",
        f"**Date**: 2026-06-16",
        f"**Features**: {len(feature_cols)} enhanced dimensions",
        f"**Method**: TreeExplainer (exact Shapley values for GBDT)\n",
        "---\n",
        "## Global Feature Importance (LightGBM)\n",
        "| Rank | Feature | Mean |SHAP| |",
        "|------|---------|------------|",
    ]
    for item in lgb_global[:30]:
        lines.append(f"| {item['rank']} | {item['feature']} | {item['importance']:.6f} |")

    lines.append("\n## Global Feature Importance (XGBoost)\n")
    lines.append("| Rank | Feature | Mean |SHAP| |")
    lines.append("|------|---------|------------|")
    for item in xgb_global[:30]:
        lines.append(f"| {item['rank']} | {item['feature']} | {item['importance']:.6f} |")

    lines.append("\n## Per-Insider Explanations (Top 10)\n")
    for i, exp in enumerate(insider_explanations):
        lines.append(f"\n### Insider {i+1} — P(insider) = {exp['prediction']:.4f}")
        lines.append("\n**Risk-increasing factors:**")
        for rf in exp["top_risk_factors"][:5]:
            lines.append(f"- ↑ `{rf['feature']}`: SHAP={rf['shap_value']:+.4f} (value={rf['feature_value']:.2f})")
        lines.append("\n**Risk-decreasing factors:**")
        for pf in exp["top_protective_factors"][:3]:
            lines.append(f"- ↓ `{pf['feature']}`: SHAP={pf['shap_value']:+.4f} (value={pf['feature_value']:.2f})")

    with open(research_dir / "10_shap_analysis.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Save explainers
    joblib.dump(lgb_explainer, models_dir / "shap_lgb_explainer.joblib")

    logger.success(f"\n✅ SHAP Analysis complete!")
    logger.info(f"  Report: {research_dir / '10_shap_analysis.md'}")
    logger.info(f"  JSON: {research_dir / '10_shap_analysis.json'}")


if __name__ == "__main__":
    main()
