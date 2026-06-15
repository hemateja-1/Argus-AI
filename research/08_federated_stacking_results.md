# 08 — Federated Stacking Experiment Results

**Strategy**: One-shot Federated Stacking (FENS / Co-Boosting)

**Models per dept**: XGBoost (500 trees) + LightGBM (500 trees)

**Data shared**: Prediction probabilities ONLY

---

## Test Set Performance

| Metric | Federated Stacking | Centralized LGB | FedAvg MLP |
|--------|-------------------|-----------------|------------|
| **F1** | **0.5060** | 0.9495 | 0.5263 |
| Precision | 0.6364 | 0.9592 | 0.7692 |
| Recall | 0.4200 | 0.9400 | 0.4000 |
| AUC-ROC | 0.9744 | 0.9827 | 0.9420 |
| TP | 21 | 47 | 20 |
| FP | 12 | 2 | 6 |
| FN | 29 | 3 | 30 |

## Department-Level Performance

| Department | Samples | Positives | XGB F1 | LGB F1 |
|------------|---------|-----------|--------|--------|
| it_admin | 1426 | 20 | 1.0000 | 1.0000 |
| retail_banking | 2410 | 60 | 1.0000 | 1.0000 |
| compliance | 2010 | 96 | 0.9730 | 0.9730 |
| hr | 1210 | 57 | 0.9565 | 0.9565 |

## Meta-Learner Weights

| Department Model | Weight |
|------------------|--------|
| compliance_xgb | +3.0418 |
| compliance_lgb | -0.2916 |
| hr_xgb | -0.8625 |
| hr_lgb | +2.1750 |
| it_admin_xgb | +1.0251 |
| it_admin_lgb | +2.9056 |
| retail_banking_xgb | +1.5326 |
| retail_banking_lgb | +1.5072 |

## Privacy Report

- Raw data shared: **0**
- Model parameters shared: **0**
- Prediction values shared: **13776**
- DPDPA compliance: ✅