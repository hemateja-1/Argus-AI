# 🛠️ Argus AI — Technology Stack

## Overview

Argus AI is built as a **two-tier application**: a Python ML backend (data pipeline, model training, scoring API) and a Next.js frontend dashboard. This document details every technology used, version requirements, and justifications.

---

## Backend: Python ML Pipeline

### Core ML & Deep Learning

| Package | Version | Purpose | Justification |
|---------|---------|---------|---------------|
| **PyTorch** | ≥ 2.0 | LSTM Autoencoder model | Industry-standard for research-grade deep learning. Better debugging than TF. GPU-accelerated. |
| **scikit-learn** | ≥ 1.5 | Isolation Forest, preprocessing, metrics, StandardScaler | Gold standard for classical ML. Isolation Forest implementation is well-tested. |
| **XGBoost** | ≥ 2.0 | Baseline comparison models | Proven on tabular data. Useful for ablation study baselines. |
| **LightGBM** | ≥ 4.0 | Alternative gradient boosting baseline | Faster training than XGBoost. Good for quick experiments. |

### Data Processing & Feature Engineering

| Package | Version | Purpose | Justification |
|---------|---------|---------|---------------|
| **pandas** | ≥ 2.0 | DataFrame operations, feature engineering | Standard data manipulation. Handles CSV ingestion of CERT data. |
| **numpy** | ≥ 1.26 | Numerical computation, array operations | Foundation for all numerical work. |
| **scipy** | ≥ 1.12 | Statistical tests, distributions, FFT for circadian profiles | KS tests, GMM for synthetic data, Fourier analysis. |
| **polars** | ≥ 0.20 | Fast data loading (optional, for large CERT files) | 10-50× faster than pandas for large CSV reads. |

### Synthetic Data Generation

| Package | Version | Purpose | Justification |
|---------|---------|---------|---------------|
| **sdv** (Synthetic Data Vault) | ≥ 1.0 | CTGAN-based synthetic tabular data generation | Research-grade implementation of Xu et al. (2019) CTGAN paper. Preserves column correlations. |
| **faker** | ≥ 28.0 | Generate realistic employee names, departments, IDs | For realistic demo data that looks authentic. |

### Explainability

| Package | Version | Purpose | Justification |
|---------|---------|---------|---------------|
| **SHAP** | ≥ 0.45 | Feature-level explanations for Isolation Forest & XGBoost | Industry standard. Produces both global and local explanations. |

### API & Serving

| Package | Version | Purpose | Justification |
|---------|---------|---------|---------------|
| **FastAPI** | ≥ 0.110 | REST API for real-time scoring | Fastest Python web framework. Auto-generates OpenAPI docs. Async support. |
| **uvicorn** | ≥ 0.29 | ASGI server | Production-grade async server for FastAPI. |
| **pydantic** | ≥ 2.0 | Data validation, request/response schemas | Type-safe API contracts. Built into FastAPI. |
| **websockets** | ≥ 12.0 | Real-time trust score streaming | For dashboard live updates via WebSocket. |

### Experiment Tracking & Model Management

| Package | Version | Purpose | Justification |
|---------|---------|---------|---------------|
| **mlflow** | ≥ 2.12 | Experiment tracking, model registry | Track hyperparameters, metrics across experiments. Model versioning. |
| **optuna** | ≥ 3.6 | Hyperparameter optimization | Bayesian optimization. Used in MuleShield with great results. |
| **joblib** | ≥ 1.4 | Model serialization | Fast pickle for scikit-learn models. |

### Privacy & Federated Learning

| Package | Version | Purpose | Justification |
|---------|---------|---------|---------------|
| **flower** | ≥ 1.8 | Federated Learning framework | Clean API. Supports custom strategies (FedAvg, FedProx). |
| **opacus** | ≥ 1.4 | Differential Privacy for PyTorch | Facebook's DP library. Integrates with PyTorch optimizers. |

### Visualization & Reporting

| Package | Version | Purpose | Justification |
|---------|---------|---------|---------------|
| **matplotlib** | ≥ 3.9 | Static plots, research figures | Standard for academic-quality plots. |
| **seaborn** | ≥ 0.13 | Statistical visualizations | Better defaults than matplotlib for distributions, heatmaps. |
| **plotly** | ≥ 5.22 | Interactive plots (optional, for notebooks) | Interactive hover, zoom for EDA notebooks. |

### Utilities

| Package | Version | Purpose | Justification |
|---------|---------|---------|---------------|
| **python-dotenv** | ≥ 1.0 | Environment variable management | Keep API keys, paths out of code. |
| **loguru** | ≥ 0.7 | Structured logging | Better than stdlib logging. Auto-formatting, colors. |
| **tqdm** | ≥ 4.66 | Progress bars | Essential for long training runs. |
| **rich** | ≥ 13.0 | Pretty terminal output | Rich tables, panels for experiment results. |

---

## Frontend: Next.js Dashboard

### Core Framework

| Package | Version | Purpose | Justification |
|---------|---------|---------|---------------|
| **Next.js** | 16.2 (Turbopack) | React-based SSR/SSG framework | Latest version with Turbopack for fast builds. Server-side API routes for Gemini proxy. |
| **React** | 19.2 | UI component library | Latest concurrent features. Industry standard. |
| **TypeScript** | 5.x | Type-safe JavaScript | Catch bugs early. Better IDE support. |

### Data Visualization

| Package | Version | Purpose | Justification |
|---------|---------|---------|---------------|
| **Chart.js** | 4.x | Core charting (line, bar, radar, doughnut) | Lightweight, performant, extensive customization. |
| **react-chartjs-2** | 5.x | React wrapper for Chart.js | Declarative Chart.js in React components. |
| **recharts** | 2.x | Heatmaps, treemaps, area charts | Built for React. Good for the trust heatmap grid. |
| **d3** | 7.x | Custom visualizations (timeline, network) | Maximum flexibility for unique visualizations. |

### Styling

| Tool | Purpose | Justification |
|------|---------|---------------|
| **Vanilla CSS** | Custom design system | Maximum control. No framework lock-in. Premium dark theme. |
| **CSS Custom Properties** | Design tokens (colors, spacing, typography) | Centralized theming. Easy dark/light mode. |
| **Google Fonts (Inter)** | Modern typography | Clean, professional, excellent readability. |

### AI Integration

| Package | Version | Purpose | Justification |
|---------|---------|---------|---------------|
| **@google/genai** | Latest | Gemini 2.0 Flash Lite API client | AI-powered threat reports, recommendations, and analyst chat. Server-side only (key never reaches browser). |

### API Communication

| Package | Version | Purpose | Justification |
|---------|---------|---------|---------------|
| **Native fetch** | Built-in | HTTP client for FastAPI backend | React 19 + Next.js 16 built-in fetch with AbortSignal timeout. No external dependency needed. |

### Utilities

| Package | Version | Purpose |
|---------|---------|---------|
| **date-fns** | 3.x | Date formatting and manipulation |
| **lucide-react** | Latest | Icon library |
| **framer-motion** | 11.x | Animations and transitions |

---

## Infrastructure & DevOps

### Development Environment

| Tool | Version | Purpose |
|------|---------|---------|
| **Python** | 3.11+ | Runtime (venv-managed) |
| **Node.js** | 18+ LTS | Frontend runtime |
| **npm** | 9+ | Package management |
| **Git** | Latest | Version control |
| **VS Code / Cursor** | Latest | IDE |

### GPU Support

| Component | Requirement |
|-----------|-------------|
| **CUDA** | 12.x (for local GPU training) |
| **cuDNN** | 9.x |
| **GPU RAM** | ≥ 4GB recommended |
| **Fallback** | Google Colab Pro (if local GPU insufficient) |

### Project Management

| Tool | Purpose |
|------|---------|
| **GitHub** | Repository hosting, issues, PRs |
| **GitHub Actions** | CI/CD (optional) |
| **Jupyter Lab** | EDA notebooks |

---

## Python `requirements.txt`

```
# Core ML
torch>=2.0.0
scikit-learn>=1.5.0
xgboost>=2.0.0
lightgbm>=4.0.0

# Data Processing
pandas>=2.0.0
numpy>=1.26.0
scipy>=1.12.0

# Synthetic Data Generation
sdv>=1.0.0
faker>=28.0.0

# Explainability
shap>=0.45.0

# API
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
pydantic>=2.0.0
websockets>=12.0

# Experiment Tracking
mlflow>=2.12.0
optuna>=3.6.0
joblib>=1.4.0

# Privacy (optional — for demo)
# flower>=1.8.0
# opacus>=1.4.0

# Visualization
matplotlib>=3.9.0
seaborn>=0.13.0

# Utilities
python-dotenv>=1.0.0
loguru>=0.7.0
tqdm>=4.66.0
rich>=13.0.0

# Jupyter (for notebooks)
jupyterlab>=4.0.0
ipywidgets>=8.0.0
```

---

## Version Compatibility Matrix

| Component | Min Version | Tested With | Notes |
|-----------|------------|-------------|-------|
| Python | 3.11 | 3.11.9 | 3.12 also works, but some packages may lag |
| PyTorch | 2.0 | 2.3.0 | CUDA 12.1 recommended for GPU |
| scikit-learn | 1.5 | 1.5.1 | Isolation Forest API stable |
| LightGBM | 4.0 | 4.5.0 | Primary model (F1=0.992) |
| FastAPI | 0.110 | 0.115 | Pydantic v2 required |
| Next.js | 16 | 16.2.9 | App router + Turbopack |
| Node.js | 18 | 20.x LTS | LTS recommended |
| @google/genai | Latest | 1.x | Gemini 2.0 Flash Lite |

---

## Architecture Decision Records (ADRs)

### ADR-1: PyTorch over TensorFlow
**Decision**: Use PyTorch for the LSTM Autoencoder.
**Rationale**: Better debugging experience (eager execution default), stronger research community adoption, easier integration with Opacus for differential privacy.

### ADR-2: Isolation Forest over One-Class SVM
**Decision**: Use Isolation Forest as the static anomaly detector.
**Rationale**: Better performance on high-dimensional data (47 features), faster training, handles mixed feature types, well-understood contamination parameter.

### ADR-3: FastAPI over Flask/Django
**Decision**: Use FastAPI for the scoring API.
**Rationale**: Native async support (important for real-time scoring), auto-generated OpenAPI documentation, built-in Pydantic validation, significantly faster than Flask.

### ADR-4: Vanilla CSS over TailwindCSS
**Decision**: Use vanilla CSS with custom properties for the dashboard.
**Rationale**: Maximum design control for a premium aesthetic. No framework learning curve. CSS custom properties provide the same design token benefits as Tailwind. Smaller bundle size.

### ADR-5: CTGAN over basic random generation
**Decision**: Use CTGAN (via SDV) for synthetic data generation.
**Rationale**: Preserves column correlations and statistical distributions from reference data. Published research (Xu et al., 2019) validates the approach. Produces more realistic data than naive random sampling.

### ADR-6: Next.js over Streamlit
**Decision**: Use Next.js for the dashboard (with Streamlit as fallback).
**Rationale**: Hackathon requires "demo prototype link" — Next.js produces a more polished, deployable demo. Streamlit is faster to build but looks less professional. Frontend team member can own this.

### ADR-7: LightGBM as Primary Supervised Model
**Decision**: Use LightGBM (not just unsupervised IF + LSTM-AE) as the primary scorer.
**Rationale**: With labeled insider data available (synthetic), supervised GBDT outperforms unsupervised anomaly detection. LightGBM achieved F1=0.992 with 211 enhanced features vs IF's F1=0.873 baseline. Training is fast (<10s), and tree-based models handle the 211 features with mixed types naturally.

### ADR-8: SHAP TreeExplainer over LIME
**Decision**: Use SHAP with TreeExplainer for model explanations.
**Rationale**: TreeExplainer computes exact Shapley values in polynomial time for tree ensembles (vs LIME's approximate perturbation-based approach). Exact values are critical for compliance teams who need deterministic explanations. Top feature: `clearance_normalized` (SHAP=0.610).

### ADR-9: Federated Stacking over Gradient Averaging
**Decision**: Use one-shot federated stacking instead of FedAvg for privacy-compliant deployment.
**Rationale**: Traditional FedAvg requires iterative gradient exchange (privacy risk) and struggles with heterogeneous department data. Federated stacking shares only scalar predictions (P(insider) per employee), not gradients or features. AUC=0.974 with zero raw data leaving departments.

### ADR-10: Meta-Learner Stacking over Simple Averaging
**Decision**: Use a logistic regression meta-learner to combine base model predictions.
**Rationale**: Simple averaging treats all models equally. Meta-learner learns optimal weights: LightGBM gets highest weight (best individual F1), LSTM-AE provides complementary temporal signal. Cross-validated to avoid overfitting.

### ADR-11: Gemini AI for Analyst-Facing Explanations
**Decision**: Integrate Google Gemini 2.0 Flash Lite for natural language threat analysis.
**Rationale**: SHAP provides quantitative feature-level explanations, but analysts need qualitative context — "why does this pattern matter for a recruiter in HR?" Gemini converts SHAP data + employee context into structured threat assessments, response recommendations, and interactive Q&A. The API key stays server-side via Next.js API routes, so it never reaches the browser. Flash Lite is chosen for speed (<2s generation) and cost efficiency.
