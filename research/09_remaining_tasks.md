# Argus AI — Remaining Tasks & Roadmap

**Created**: 2026-06-16  
**Priority**: Hackathon submission readiness

---

## 🔴 Phase 7: High Impact (Hackathon Winners)

### Task 7.1: Connect Enhanced Models → API → Dashboard
- [ ] Update `scoring_api.py` to load 211-feature enhanced models (LightGBM, XGBoost, Meta-Learner)
- [ ] Replace old 47-feature scoring pipeline with enhanced pipeline
- [ ] Expose per-employee risk scores, trust scores, and explainability data
- [ ] Wire dashboard to consume live API data instead of mock data
- [ ] Test end-to-end: data → features → model → API → dashboard

### Task 7.2: SHAP Explainability
- [ ] Run SHAP on enhanced LightGBM (211 features)
- [ ] Generate global feature importance (beeswarm plot)
- [ ] Generate per-employee waterfall plots (top 10 features driving each prediction)
- [ ] Integrate SHAP explanations into API response (`/api/v1/explain/{emp_id}`)
- [ ] Display SHAP explanations in dashboard employee detail page

### Task 7.3: Demo Script
- [ ] Create `demo/run_demo.py` — one-click full pipeline
- [ ] Steps: generate data → engineer features → train models → start API → open dashboard
- [ ] Add progress bars and colored output
- [ ] Handle "already exists" cases gracefully (skip regeneration if data exists)
- [ ] Test on clean clone

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
