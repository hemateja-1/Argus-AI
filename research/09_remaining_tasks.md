# Argus AI — Remaining Tasks & Roadmap

**Created**: 2026-06-16  
**Priority**: Hackathon submission readiness

---

## 🔴 Phase 7: High Impact (Hackathon Winners)

### Task 7.1: Connect Enhanced Models → API → Dashboard
- [x] Update `scoring_api.py` to load 211-feature enhanced models (LightGBM, XGBoost, Meta-Learner)
- [x] Replace old 47-feature scoring pipeline with enhanced pipeline
- [x] Expose per-employee risk scores, trust scores, and explainability data
- [x] Wire dashboard to consume live API data instead of mock data
- [x] Test end-to-end: data → features → model → API → dashboard

**Result**: API v2.0 running. 5 models, 211 features, 13/14 insiders detected.

### Task 7.2: SHAP Explainability
- [x] Run SHAP on enhanced LightGBM (211 features)
- [x] Generate global feature importance
- [x] Generate per-employee explanations (top risk/protective factors)
- [x] Integrate SHAP explanations into API response (`/api/explain/{emp_id}`)
- [x] Display SHAP explanations in dashboard employee detail page (SHAP waterfall component)

**Result**: Top feature = clearance_normalized (SHAP=0.610). All insiders driven by
roll_7d_max_data_volume_mb and roll_14d_std_data_volume_mb. Report: research/10_shap_analysis.md

### Task 7.3: Demo Script
- [x] Create `demo.py` — one-click full pipeline
- [x] Steps: generate data → engineer features → train models → run SHAP → start API → open dashboard
- [x] Add colored output and ASCII banner
- [x] Handle "already exists" cases gracefully (skip regeneration if data exists)
- [x] Test on clean clone (all files present, demo.py --help works, npm build passes)

---

## 🟡 Phase 8: Medium Impact (Polish)

### Task 8.1: Jupyter Notebooks
- [ ] `notebooks/01_cert_eda.ipynb` — Interactive EDA walkthrough
- [ ] `notebooks/02_synthetic_data_analysis.ipynb` — Synthetic data quality analysis
- [ ] `notebooks/03_model_experiments.ipynb` — Model comparison with inline plots

### Task 8.2: Documentation
- [ ] `docs/ARCHITECTURE.md` — System architecture deep-dive
- [ ] `docs/BUILD_GUIDE.md` — Step-by-step build instructions
- [ ] `docs/DATA_STRATEGY.md` — Dataset selection & feature engineering rationale
- [ ] `docs/TECH_STACK.md` — Technology choices and justifications
- [ ] `docs/RESEARCH_REFERENCES.md` — Academic papers and references

### Task 8.3: Cross-Validation
- [ ] 5-fold stratified CV on enhanced pipeline
- [ ] Report mean ± std for F1, AUC, Precision, Recall
- [ ] Log results to `research/09_cross_validation.md`

### Task 8.4: Ablation Study
- [ ] Feature category ablation (remove each of 7 categories, measure F1 drop)
- [ ] Model ablation (remove each base learner from ensemble, measure F1 drop)
- [ ] Log results to `research/10_ablation_study.md`

---

## 🟢 Phase 9: Nice-to-Have

### Task 9.1: Adversarial Robustness
- [ ] Test with evolving attack scenarios (insider adapts behavior)
- [ ] Measure detection latency (how many days until detection?)

### Task 9.2: Latency Benchmarks
- [ ] Measure API scoring latency (target: <200ms per request)
- [ ] Measure batch scoring throughput (target: 1000+ employees/second)

### Task 9.3: Presentation Materials
- [ ] Auto-generate hackathon pitch deck from research data
- [ ] Include architecture diagrams, performance charts, demo screenshots
