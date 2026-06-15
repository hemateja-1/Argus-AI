# 🔨 Argus AI — Build Guide

## Step-by-Step Implementation Guide

This guide walks through building Argus AI from scratch — all as Python scripts, no notebooks.

---

## Phase 0: Environment Setup

### Step 0.1: Create Project Structure

```bash
# Navigate to project root
cd "C:\Github\BOB Hackathon"

# Create Python virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 0.2: Verify GPU

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

### Step 0.3: Download CERT Dataset

Option A — From Kaggle:
```bash
# If you have kaggle CLI configured
kaggle datasets download -d <cert-dataset-slug> -p data/cert_r4.2/
```

Option B — Manual Download:
1. Go to https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247
2. Download the r4.2 archive
3. Extract to `data/cert_r4.2/`

Expected files after extraction:
```
data/cert_r4.2/
├── logon.csv
├── email.csv
├── file.csv
├── device.csv
├── http.csv
├── psychometric.csv
└── answers/
    └── insiders.csv
```

---

## Phase 1: Data Foundation (Day 1-2)

### Step 1.1: Project Scaffolding

Create the `argus/` Python package with proper `__init__.py` files.

**Files to create**:
- `argus/__init__.py`
- `argus/config.py` — Global config (paths, hyperparams, feature lists)
- `argus/data/__init__.py`
- `argus/data/schemas.py` — Pydantic data models
- `argus/models/__init__.py`
- `argus/privacy/__init__.py`
- `argus/api/__init__.py`
- `argus/evaluation/__init__.py`

### Step 1.2: Build Synthetic Banking Data Generator

**File**: `argus/data/synthetic_generator.py`

This is the **first script to build** because it gives us data to work with immediately, independent of CERT download.

**What it does**:
1. Generate 200 employee profiles across 5 departments
2. For each employee, simulate 90 days of normal activity using Markov chains + GMM
3. Select 15-20 employees as insiders
4. Inject 6 banking-specific threat scenarios with realistic ramp-up
5. Add noise (sick days, holidays, legitimate overtime)
6. Output: `employees.csv`, `activity_log.csv`, `ground_truth.csv`

**Run**:
```bash
python -m argus.data.synthetic_generator
# Output: data/synthetic/employees.csv, activity_log.csv, ground_truth.csv
```

**Verification**:
```bash
python -m argus.data.synthetic_generator --validate
# Checks: schema compliance, class balance, temporal consistency
```

### Step 1.3: Build CERT Data Loader

**File**: `argus/data/cert_loader.py`

**What it does**:
1. Load and parse all 5 CERT CSV files
2. Merge on (user, date) pairs
3. Handle timestamps across time zones
4. Join with answer key for ground truth labels
5. Output: unified DataFrame with user-day granularity

**Run**:
```bash
python -m argus.data.cert_loader --input data/cert_r4.2/ --output data/processed/cert_unified.parquet
```

### Step 1.4: Build Feature Engineering Pipeline

**File**: `argus/data/feature_engineer.py`

**What it does**:
1. Takes unified activity data (from CERT or synthetic)
2. Aggregates to employee-day granularity
3. Computes all 47 features across 8 categories
4. Outputs feature matrix + labels

**Run**:
```bash
# For synthetic data
python -m argus.data.feature_engineer --source synthetic --output data/processed/synthetic_features.parquet

# For CERT data
python -m argus.data.feature_engineer --source cert --output data/processed/cert_features.parquet
```

### Step 1.5: Data Exploration Script

**File**: `argus/data/explore.py`

**What it does**:
1. Load processed features
2. Print summary statistics
3. Generate distribution plots for all 47 features
4. Compare insider vs. normal distributions
5. Compute KS statistics for each feature
6. Save analysis report + plots to `results/eda/`

**Run**:
```bash
python -m argus.data.explore --source synthetic
# Output: results/eda/feature_distributions.png, results/eda/eda_report.md
```

---

## Phase 2: Core ML Models (Day 2-3)

### Step 2.1: Train Isolation Forest

**File**: `experiments/train_if.py`

**What it does**:
1. Load processed features
2. Split by timeline (train: normal only, val/test: mixed)
3. Train Isolation Forest with contamination=0.05
4. Compute anomaly scores for all employee-days
5. Evaluate: F1, AUC-ROC, PR-AUC at multiple thresholds
6. Save model + results

**Run**:
```bash
python experiments/train_if.py --data data/processed/synthetic_features.parquet --output models/isolation_forest.joblib
# Output: models/isolation_forest.joblib, results/if_evaluation.md
```

### Step 2.2: Build & Train LSTM Autoencoder

**File**: `argus/models/lstm_autoencoder.py` (model definition)
**File**: `experiments/train_lstm.py` (training script)

**What it does**:
1. Build sequence dataset (7-day windows × 47 features)
2. Train LSTM Autoencoder on normal behavior only
3. Compute reconstruction error for all sequences
4. Set anomaly threshold at 95th percentile of normal errors
5. Evaluate temporal anomaly detection
6. Save model + threshold

**Run**:
```bash
python experiments/train_lstm.py --data data/processed/synthetic_features.parquet --epochs 50 --device cuda
# Output: models/lstm_autoencoder.pt, results/lstm_evaluation.md
```

### Step 2.3: Build Hybrid Ensemble

**File**: `argus/models/risk_engine.py` (ensemble logic)
**File**: `experiments/train_hybrid.py` (training + alpha optimization)

**What it does**:
1. Load both trained models (IF + LSTM)
2. Generate scores from both models for validation set
3. Optimize ensemble weight α via grid search on validation F1
4. Evaluate hybrid ensemble on test set
5. Compare: IF-only vs. LSTM-only vs. Hybrid

**Run**:
```bash
python experiments/train_hybrid.py --if-model models/isolation_forest.joblib --lstm-model models/lstm_autoencoder.pt
# Output: models/hybrid_config.json, results/hybrid_evaluation.md, results/model_comparison.png
```

---

## Phase 3: Intelligence Layer (Day 3-4)

### Step 3.1: Build Privilege Context Engine

**File**: `argus/models/privilege_engine.py`

**What it does**:
1. Define role-resource risk matrix
2. Implement privilege decay function
3. Implement dynamic trust score computation
4. Unit tests for trust score scenarios

**Verify**:
```bash
python -m argus.models.privilege_engine --test
# Runs built-in test scenarios, prints trust score timelines
```

### Step 3.2: Build Explainable Alert Engine

**File**: `argus/models/explainer.py`

**What it does**:
1. Intent signal chain matcher
2. Natural language alert template generator
3. Peer constellation analyzer (z-scores vs. peer group)
4. Risk factor summarizer

**Verify**:
```bash
python -m argus.models.explainer --demo
# Generates sample alerts for each threat scenario, prints to console
```

### Step 3.3: Build Digital Employee Twin

**File**: `argus/models/behavioral_twin.py`

**What it does**:
1. Circadian profile construction (FFT of login times)
2. Access embedding (resource frequency vector)
3. Rolling behavioral baseline (EMA of 47 features)
4. Drift velocity computation
5. Deviation scoring: current vs. twin

**Verify**:
```bash
python -m argus.models.behavioral_twin --employee EMP_001 --data data/processed/synthetic_features.parquet
# Prints twin profile + deviation analysis for one employee
```

---

## Phase 4: Full Pipeline Integration (Day 4-5)

### Step 4.1: End-to-End Scoring Pipeline

**File**: `argus/pipeline.py`

**What it does**:
1. Orchestrates the full scoring flow:
   - Load activity data → Feature engineer → Twin comparison → Risk scoring → Privilege context → Trust score → Alert generation
2. Processes all employees for a given day
3. Outputs ranked alert list

**Run**:
```bash
python -m argus.pipeline --data data/synthetic/ --date 2025-08-15
# Output: results/alerts_2025-08-15.json
```

### Step 4.2: FastAPI Scoring API

**File**: `argus/api/scoring_api.py`

**What it does**:
1. REST API wrapping the pipeline
2. Endpoints: `/score`, `/employees`, `/alerts`, `/twin/{emp_id}`, `/peers/{emp_id}`
3. WebSocket endpoint for real-time streaming
4. Serves data to the frontend dashboard

**Run**:
```bash
python -m argus.api.scoring_api
# Starts at http://localhost:8000
# Docs at http://localhost:8000/docs
```

---

## Phase 5: Dashboard & Demo (Day 5-6)

### Step 5.1: Build Next.js Dashboard

**Setup**:
```bash
cd dashboard
npx -y create-next-app@latest ./ --typescript --app --no-tailwind --no-eslint
npm install chart.js react-chartjs-2 recharts axios framer-motion lucide-react date-fns
```

**Key Pages**:
- `/` — Trust score heatmap (all employees)
- `/alerts` — Active alert queue with intent chains
- `/employee/[id]` — Employee detail: twin comparison, timeline, privilege decay
- `/model` — Model performance metrics

### Step 5.2: Connect Dashboard to API

Configure API proxy in `next.config.js`:
```javascript
module.exports = {
    async rewrites() {
        return [{ source: '/api/:path*', destination: 'http://localhost:8000/api/:path*' }]
    }
}
```

---

## Phase 6: Polish & Documentation (Day 6-7)

### Step 6.1: Ablation Study

**File**: `experiments/ablation_study.py`

Test the contribution of each module:
- LSTM only vs. IF only vs. Hybrid
- With vs. without Privilege Context
- With vs. without Peer Constellation
- With vs. without Intent Chains

### Step 6.2: Generate Evaluation Report

**File**: `argus/evaluation/evaluate.py`

Produce:
- ROC/PR curves
- Confusion matrices at multiple thresholds
- Feature importance (SHAP)
- Model comparison table
- Per-scenario detection rates

**Run**:
```bash
python -m argus.evaluation.evaluate --all
# Output: results/final_evaluation/
```

### Step 6.3: Demo Walkthrough Script

**File**: `demo/run_demo.py`

Automated demo that:
1. Shows 3 employees being monitored in real-time
2. Injects a threat scenario live
3. Shows trust score dropping
4. Generates and displays the alert
5. Shows peer comparison
6. Prints recommended action

**Run**:
```bash
python demo/run_demo.py
```

---

## Quick Reference: Key Commands

```bash
# Setup
python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt

# ONE-CLICK DEMO (recommended for first run)
python demo.py                      # Generate data -> Train -> API -> Dashboard

# Step-by-step alternative
python -m argus.data.synthetic_generator
python -m argus.data.feature_engineer --source synthetic
python -m argus.data.enhanced_feature_engineer  # 47 -> 211 features
python -m argus.models.train_enhanced           # LightGBM + XGBoost + Meta
python -m argus.api.scoring_api                 # Start API at :8000
cd dashboard && npm install && npm run dev      # Dashboard at :3000

# Experiments
python -m argus.experiments.cross_validation    # 5-fold stratified CV
python -m argus.experiments.ablation_study      # Feature/model ablation

# Dashboard (separate terminal)
cd dashboard
npm install
npm run dev
# Open http://localhost:3000
```

---

## Build Checklist

- [x] Phase 0: Environment setup + CERT download
- [x] Phase 1.1: Project scaffolding
- [x] Phase 1.2: Synthetic data generator
- [x] Phase 1.3: CERT data loader
- [x] Phase 1.4: Feature engineering pipeline (47 features)
- [x] Phase 1.5: Data exploration script
- [x] Phase 2.1: Isolation Forest training
- [x] Phase 2.2: LSTM Autoencoder training
- [x] Phase 2.3: Hybrid ensemble optimization
- [x] Phase 3.1: Privilege context engine
- [x] Phase 3.2: Explainable alert engine
- [x] Phase 3.3: Digital employee twin
- [x] Phase 4.1: End-to-end pipeline
- [x] Phase 4.2: FastAPI scoring API (v2.0 — enhanced models)
- [x] Phase 5.1: Next.js dashboard
- [x] Phase 5.2: Dashboard-API integration (live data + mock fallback)
- [x] Phase 6.1: Ablation study (feature + model ablation)
- [x] Phase 6.2: Evaluation report (SHAP analysis, CV results)
- [x] Phase 6.3: Demo walkthrough (`demo.py` — one-click)

### Enhanced Pipeline (v2.0) — Added Post-Initial Build

- [x] Deep feature investigation (dead features, correlations)
- [x] Enhanced feature engineering (47 → 211 features)
- [x] LightGBM/XGBoost supervised training (F1=0.992)
- [x] Meta-learner stacking ensemble
- [x] SHAP TreeExplainer integration
- [x] Federated stacking (privacy-compliant alternative)
- [x] 5-fold stratified cross-validation (F1=0.893 ± 0.055)
- [x] Feature/model ablation study (rolling windows most critical: -8.3% F1 drop)
- [x] Adversarial robustness testing (7 evasion strategies)
- [x] Latency benchmarks (API <200ms, batch >150K/sec)
- [x] SHAP waterfall dashboard component
- [x] `/api/explain/{emp_id}` endpoint
- [x] Gemini AI integration (threat reports, recommendations, chat)
- [x] Mock data alignment with ground_truth.csv (14 insiders, 6 scenarios)
- [x] Data pipeline documentation (research/15_data_pipeline.md)
- [x] Updated README and documentation

### Dashboard Setup (Gemini AI)

```bash
cd dashboard
npm install
cp .env.example .env.local
# Edit .env.local → add GEMINI_API_KEY from Google AI Studio
npm run dev
```

> `.env.local` is gitignored. The API key stays server-side via `/api/gemini` route.
