# 07 — Federated Learning Strategy Research & Selection

**Date**: 2026-06-16  
**Decision**: Federated Stacking (over FedAvg MLP, FedXGB Bagging, SecureBoost)

---

## 1. Problem Statement

We need privacy-preserving model training across 5 bank departments (retail_banking, treasury,
it_admin, hr, compliance) without sharing raw employee behavioral data. Each department must
retain full control of its data while contributing to a global insider threat model.

**Constraints:**
- Horizontal partitioning: same 211 features, different employees per department
- Extreme class imbalance: 2.89% positive rate overall, varies by department
- Must work with tree-based models (XGBoost/LightGBM) — our best performers (F1=0.95)
- Aligned with India's Digital Personal Data Protection Act (DPDPA)

---

## 2. Approaches Evaluated

### 2.1 FedAvg with MLP (Attempted First)

**Method:** McMahan et al. (2017) — "Communication-Efficient Learning of Deep Networks
from Decentralized Data." AISTATS. Each department trains a copy of a shared MLP, gradients
are averaged on the server, and the global model is updated.

**Our implementation:** 211→64→32→1 MLP, 10 rounds, 3 local epochs, ε=2.0 DP.

**Our results:**
- Test F1: **0.526** (vs centralized LightGBM F1=0.950)
- Test AUC-ROC: 0.942
- Test Precision: 0.769, Recall: 0.400
- TP=20, FP=6, FN=30, TN=1671

**Why it underperformed:**
1. **Architecture mismatch:** MLP is fundamentally weaker than gradient boosting for tabular
   data. Grinsztajn et al. (2022) "Why do tree-based models still outperform deep learning
   on typical tabular data?" showed trees dominate for <10K samples.
2. **DP noise degradation:** σ=0.6656 noise on 13K parameters destroys signal in small batches.
3. **Insufficient rounds:** 10 rounds too few for convergence with DP noise.
4. **Heterogeneous department data:** Non-IID distribution across departments (treasury has
   0 insiders in some splits, compliance has 150) causes gradient conflict.

**Verdict:** ❌ Rejected. The privacy-utility tradeoff is too severe for this architecture.

---

### 2.2 Federated XGBoost Bagging (Tree Concatenation)

**Method:** Flower framework (2023) — `FedXgbBagging` strategy. Each department trains local
XGBoost trees on its data. Trees are sent to the server and concatenated into a single global
ensemble. The model is passed back and departments train additional trees.

**Key references:**
- Flower AI: "Federated XGBoost Quickstart" (flower.ai/docs)
- Li et al. (2020): "Federated Optimization in Heterogeneous Networks" (FedProx)

**Expected performance:** Near centralized (~2-3% drop based on literature).

**Pros:**
- Uses XGBoost natively (no architecture compromise)
- Bagging is inherently robust to non-IID data
- Iteratively grows the ensemble over rounds
- Flower has production-ready code

**Caveats:**
1. ⚠️ **Tree structures leak information.** Split points (e.g., "data_volume_mb > 15.3")
   reveal data distribution boundaries. An adversary with access to tree structures can
   infer feature ranges in each department's data.
2. ⚠️ **Requires Flower framework dependency** — adds installation complexity, potential
   version conflicts with our existing PyTorch/sklearn stack.
3. ⚠️ **Multiple communication rounds** — each round requires sending full tree structures
   (which grow with each round). For 500 trees × 5 depts, this is significant bandwidth.
4. ⚠️ **XGBoost only** — cannot natively combine XGBoost + LightGBM + LSTM in one federated
   framework. We'd need separate federation for each model type.

**Verdict:** ⚠️ Strong option but tree leakage is a concern for banking compliance.

---

### 2.3 SecureBoost (Vertical Federated Learning)

**Method:** Cheng et al. (2019) — "SecureBoost: A Lossless Federated Learning Framework."
IEEE Intelligent Systems. Uses homomorphic encryption for parties holding different features
for the same set of users.

**Key property:** LOSSLESS — achieves identical accuracy to centralized model.

**Why NOT for us:**
1. ❌ **Wrong partitioning paradigm.** Our data is horizontally partitioned (same 211 features,
   different employees). SecureBoost requires vertical partitioning (different features,
   same employees).
2. ❌ **Heavy cryptographic overhead.** Requires Paillier homomorphic encryption, which is
   100-1000x slower than plaintext training.
3. ❌ **Requires FATE framework** — a separate, heavy framework (WeBank/FedAI) not compatible
   with our sklearn/PyTorch stack.

**Verdict:** ❌ Rejected. Designed for vertical FL, not our horizontal partitioning.

---

### 2.4 Federated Stacking (Selected ✅)

**Method:** Train local XGBoost + LightGBM at each department independently. Share only
prediction probabilities (soft labels) with the server. A Meta-Learner on the server combines
all department predictions into a final risk score.

**Key references:**
- **FENS** — "Revisiting Ensembling in One-Shot Federated Learning" (NeurIPS 2024):
  Constructs a secondary stacking aggregator on local model predictions.
- **Co-Boosting** — "Enhancing One-Shot FL Through Data and Ensemble Co-Boosting" (ICLR 2024):
  Mutual improvement between ensemble weights and synthetic data quality.
- **FedLPA** — "One-shot FL with Layer-Wise Posterior Aggregation" (NeurIPS 2024):
  Laplace approximation for better aggregation under non-IID conditions.
- Federated ensemble in healthcare — NIH studies show 2-5% accuracy drop vs centralized,
  with prediction-only sharing providing strongest privacy guarantees.

**Expected performance:** F1 ≈ 0.90-0.93 (2-5% below centralized F1=0.95).

**Pros:**
1. ✅ **Uses our best models** — XGBoost + LightGBM natively (not a compromised MLP)
2. ✅ **Maximum privacy** — only P(insider) probabilities leave each department. An adversary
   seeing P=0.72 cannot reconstruct the original 211 behavioral features.
3. ✅ **One-shot** — single communication round, no iterative training needed
4. ✅ **Heterogeneous models** — each department can use different algorithms, hyperparameters,
   or even augmentation strategies tuned for its own distribution
5. ✅ **No extra dependencies** — uses existing sklearn, XGBoost, LightGBM stack
6. ✅ **DPDPA compliant** — data minimization (predictions only), purpose limitation,
   storage limitation all satisfied by construction

**Caveats:**
1. ⚠️ Departments with very few insiders (treasury: 0-38 in some splits) may produce
   poorly calibrated local models. Mitigation: weight departments by validation performance.
2. ⚠️ Requires a small shared validation set for the Meta-Learner to train on. However,
   only predictions (not raw data) from this set are shared.
3. ⚠️ Each department model sees only its own data distribution, potentially missing
   cross-department attack patterns. Mitigation: include centralized model in ensemble.

---

## 3. Decision Matrix

| Criterion                | FedAvg MLP | FedXGB Bagging | SecureBoost | **Fed Stacking** |
|--------------------------|-----------|----------------|-------------|------------------|
| Expected F1              | 0.526     | ~0.92-0.94     | ~0.95       | **~0.90-0.93**   |
| Uses our best models     | ❌ MLP    | ⚠️ XGB only    | ⚠️ XGB only | **✅ XGB+LGB**   |
| Privacy level            | ✅ DP     | ⚠️ Trees leak  | ✅ Encrypted | **✅✅ Preds only** |
| Implementation effort    | Done      | Medium (Flower) | Heavy (FATE)| **Low**          |
| Communication rounds     | 10+       | 5-10           | 10+         | **1 (one-shot)** |
| Data partitioning fit    | ✅ Horiz  | ✅ Horiz       | ❌ Vertical | **✅ Horiz**     |
| Extra dependencies       | PyTorch   | Flower         | FATE        | **None**         |
| Banking compliance       | ✅        | ⚠️ Tree leak   | ✅          | **✅**           |
| Academic backing (2024)  | 2017      | 2023           | 2019        | **ICLR+NeurIPS** |

## 4. Implementation Architecture

```
Phase 1: Local Training (per department, in parallel)
  Each department independently:
  1. Takes its local employee-day feature matrix (X_dept, y_dept)
  2. Applies SMOTE+Tomek for local augmentation
  3. Trains XGBoost (500 trees) + LightGBM (500 trees)
  4. Evaluates on department-local validation set

Phase 2: Prediction Sharing (one-shot communication)
  Each department:
  1. Generates P(insider) from both models on shared validation indices
  2. Sends ONLY the probability vectors to the server
  3. No raw data, no model parameters, no gradients leave the department

Phase 3: Server Aggregation
  Server:
  1. Receives 10 probability vectors (5 depts × 2 models)
  2. Optionally receives department-level LSTM/IF anomaly scores
  3. Trains a Meta-Learner (Logistic Regression) on the stacked predictions
  4. Learns optimal department weights (e.g., compliance may be upweighted
     because it has more insider samples)

Phase 4: Inference
  For a new employee-day:
  1. Department's local models produce P(insider)
  2. Server Meta-Learner combines all department predictions
  3. Final risk score = Meta-Learner output
```

## 5. Privacy Guarantee

What's shared at each phase:
- Phase 1: ❌ Nothing (all local)
- Phase 2: ✅ Only P(insider) ∈ [0,1] per employee-day — a single float
- Phase 3: ❌ Nothing (server-side only)
- Phase 4: ✅ Only P(insider) for the queried employee

**Information leakage analysis:**
- Given P(insider) = 0.72 for employee EMP_047 on day 45
- An adversary CANNOT reconstruct: login_hour, data_volume_mb, unique_systems, etc.
- The 211-dimensional feature vector is compressed to a single scalar
- This satisfies the data minimization principle of DPDPA
