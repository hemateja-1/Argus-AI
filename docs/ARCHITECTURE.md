# 🏗️ Argus AI — System Architecture

## Overview

Argus AI follows a **modular, layered architecture** with clear separation between data ingestion, behavioral profiling, risk scoring, contextual intelligence, and presentation layers. Each module is independently testable and can be developed in parallel.

---

## High-Level Architecture Diagram

```
                        ┌─────────────────────────────┐
                        │      DATA SOURCES            │
                        │                             │
                        │  ┌─────┐ ┌──────┐ ┌──────┐ │
                        │  │CERT │ │ LANL │ │Synth │ │
                        │  │r4.2 │ │      │ │Bank  │ │
                        │  └──┬──┘ └──┬───┘ └──┬───┘ │
                        └─────┼───────┼────────┼─────┘
                              │       │        │
                              ▼       ▼        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 1: DATA PIPELINE                       │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐   │
│  │ Data Loader   │   │ Preprocessor │   │ Feature Engineer │   │
│  │              │   │              │   │                  │   │
│  │ • cert_loader │──▶│ • Clean      │──▶│ • 47 base     │   │
│  │ • synth_gen   │   │ • Validate   │   │ • → 211       │   │
│  │ • schemas     │   │ • Normalize  │   │   enhanced    │   │
│  └──────────────┘   └──────────────┘   │ • Temporal     │   │
│                                         │ • Volume       │   │
│                                         │ • Device       │   │
│                                         │ • Deltas       │   │
│                                         │ • Rolling      │   │
│                                         │ • Z-scores     │   │
│                                         │ • Interactions │   │
│                                         └────────┬─────────┘   │
└──────────────────────────────────────────────────┼─────────────┘
                                                   │
                                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                LAYER 2: BEHAVIORAL PROFILING                    │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              Digital Employee Twin Builder              │    │
│  │                                                        │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  │    │
│  │  │  Circadian   │  │   Access    │  │  Behavioral  │  │    │
│  │  │  Profile     │  │  Embedding  │  │  Baseline    │  │    │
│  │  │             │  │             │  │              │  │    │
│  │  │ FFT of login │  │ Resource    │  │ Rolling μ/σ  │  │    │
│  │  │ hourly dist  │  │ graph embed │  │ of 47 feats  │  │    │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬───────┘  │    │
│  │         │                │                │           │    │
│  │         └────────────────┼────────────────┘           │    │
│  │                          │                             │    │
│  │                          ▼                             │    │
│  │              ┌──────────────────────┐                 │    │
│  │              │  BEHAVIORAL GENOME   │                 │    │
│  │              │  (compressed vector) │                 │    │
│  │              └──────────┬───────────┘                 │    │
│  └─────────────────────────┼──────────────────────────────┘    │
└────────────────────────────┼───────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                LAYER 3: RISK SCORING ENGINE                     │
│                                                                 │
│  ┌──────────────────────┐    ┌──────────────────────┐         │
│  │  PATH A: Temporal     │    │  PATH B: Static      │         │
│  │                      │    │                      │         │
│  │  LSTM Autoencoder    │    │  Isolation Forest    │         │
│  │                      │    │                      │         │
│  │  Input: 7×47 window  │    │  Input: 1×53 vector  │         │
│  │  Encoder: 47→32→16   │    │  Trees: 500          │         │
│  │  Decoder: 16→32→47   │    │  Contamination: 0.05 │         │
│  │                      │    │                      │         │
│  │  Score: Recon Error   │    │  Score: Isolation    │         │
│  │  → Temporal Anomaly  │    │  → Static Anomaly    │         │
│  │    Score (0-100)      │    │    Score (0-100)     │         │
│  └──────────┬───────────┘    └──────────┬───────────┘         │
│             │                           │                      │
│             └─────────┬─────────────────┘                      │
│                       ▼                                        │
│           ┌──────────────────────┐                             │
│           │  ENSEMBLE FUSION     │                             │
│           │                      │                             │
│           │  Score = α×LSTM +    │                             │
│           │        (1-α)×IF      │                             │
│           │  α ≈ 0.65            │                             │
│           │                      │                             │
│           │  → Raw Risk Score    │                             │
│           └──────────┬───────────┘                             │
└──────────────────────┼─────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              LAYER 4: PRIVILEGE CONTEXT ENGINE                  │
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐                   │
│  │ Role-Resource     │  │ Privilege Decay   │                   │
│  │ Risk Matrix       │  │ Function          │                   │
│  │                  │  │                  │                   │
│  │ Maps: (role,     │  │ T(t) = T(t-1) ×  │                   │
│  │ resource) →      │  │ e^(-λΔt) +       │                   │
│  │ risk_multiplier  │  │ reinforcement(t)  │                   │
│  └────────┬─────────┘  └────────┬─────────┘                   │
│           │                     │                              │
│           └──────────┬──────────┘                              │
│                      ▼                                         │
│          ┌───────────────────────┐                              │
│          │  DYNAMIC TRUST SCORE  │                              │
│          │      (0 — 100)        │                              │
│          │                       │                              │
│          │  0-20: CRITICAL       │                              │
│          │  20-40: HIGH RISK     │                              │
│          │  40-60: MEDIUM RISK   │                              │
│          │  60-80: LOW RISK      │                              │
│          │  80-100: TRUSTED      │                              │
│          └───────────┬───────────┘                              │
└──────────────────────┼─────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              LAYER 5: EXPLAINABLE ALERT ENGINE                  │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ Intent Signal │  │ Peer         │  │ Natural Language     │ │
│  │ Chain         │  │ Constellation│  │ Alert Generator      │ │
│  │ Detector      │  │ Analyzer     │  │                      │ │
│  │              │  │              │  │ Produces human-      │ │
│  │ Matches      │  │ Compares vs  │  │ readable explanations│ │
│  │ sequences to │  │ same-role    │  │ with feature         │ │
│  │ known attack │  │ peers via    │  │ attributions and     │ │
│  │ patterns     │  │ z-scores     │  │ recommended actions  │ │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘ │
│         │                 │                      │              │
│         └─────────────────┼──────────────────────┘              │
│                           ▼                                     │
│                ┌────────────────────┐                           │
│                │   ALERT OBJECT     │                           │
│                │                    │                           │
│                │ • Trust score      │                           │
│                │ • Risk factors     │                           │
│                │ • Intent chain     │                           │
│                │ • Peer comparison  │                           │
│                │ • Recommended      │                           │
│                │   action           │                           │
│                └────────┬───────────┘                           │
└─────────────────────────┼──────────────────────────────────────┘
                          │
              ┌───────────┼───────────┐
              ▼                       ▼
┌──────────────────────┐  ┌──────────────────────┐
│   FastAPI REST API    │  │  Next.js 16 Dashboard │
│                      │  │                      │
│ GET /api/overview    │  │ • Command Center     │
│ GET /api/employees   │  │ • Alert Queue        │
│ GET /api/employee/id │  │ • Employee Detail    │
│ GET /api/alerts      │  │ • Twin Comparison    │
│ GET /api/analytics   │  │ • SHAP Waterfall     │
│ GET /api/explain/id  │  │ • Privilege Decay    │
│ GET /api/activity    │  │ • Model Analytics    │
│                      │  │                      │
└──────────────────────┘  └───────────┬──────────┘
                                      │
                                      ▼
                          ┌──────────────────────┐
                          │  Gemini AI Layer      │
                          │                      │
                          │ • Threat Reports     │
                          │ • Recommendations    │
                          │ • Interactive Chat   │
                          │                      │
                          │ POST /api/gemini     │
                          │ (server-side proxy)  │
                          └──────────────────────┘
```

---

## Module Details

### Layer 1: Data Pipeline

#### Input Sources

| Source | Files | Records | Use |
|--------|-------|---------|-----|
| Synthetic Banking | employees.csv, activity_log.csv, ground_truth.csv | 505K events, 200 employees | Primary training, evaluation, and demo |
| CERT r4.2 | logon.csv, email.csv, file.csv, device.csv, http.csv | ~32M events | Methodology reference (scenario design) |

#### Feature Engineering Pipeline (47 Features)

The feature engineer transforms raw event logs into **employee-day** feature vectors:

**Aggregation Granularity**: One feature vector per employee per day.

```python
# Pseudocode for feature extraction
for each (employee, date) pair:
    events = filter_events(employee, date)
    
    features = {
        # Temporal (8 features)
        "login_hour": first_login_hour(events),
        "logout_hour": last_logout_hour(events),
        "session_duration_hrs": total_session_time(events),
        "is_weekend": date.weekday() >= 5,
        "is_after_hours": any_event_outside(events, 9, 18),
        "time_since_last_session": hours_since(last_session),
        "login_regularity_score": regularity(login_times, historical),
        "temporal_entropy": entropy(hourly_event_distribution),
        
        # Access Volume (7 features)
        "files_accessed": count(events, type="file"),
        "emails_sent": count(events, type="email_sent"),
        "emails_received": count(events, type="email_received"),
        "urls_visited": count(events, type="http"),
        "usb_events": count(events, type="device"),
        "data_volume_mb": sum(events.data_size),
        "unique_systems_accessed": nunique(events.system),
        
        # Device & Location (5 features)
        "is_new_device": device not in historical_devices,
        "device_count": nunique(events.device),
        "unique_pcs": nunique(events.pc),
        "geo_anomaly_flag": is_unusual_location(events),
        "vpn_usage": any_vpn(events),
        
        # Communication (6 features)
        "external_email_ratio": external_emails / total_emails,
        "avg_attachment_size": mean(email_attachments.size),
        "unique_recipients": nunique(email_recipients),
        "cc_bcc_ratio": cc_bcc_count / total_emails,
        "email_content_sentiment": sentiment_score(email_content),
        "unusual_recipient_flag": any_new_external_recipient,
        
        # Data Movement (7 features)
        "file_copy_count": count(events, type="file_copy"),
        "usb_file_transfers": count(usb_file_events),
        "large_download_flag": any(download > threshold),
        "sensitive_file_access": count(sensitive_file_events),
        "data_egress_volume": sum(outbound_data),
        "print_count": count(print_events),
        "cloud_upload_count": count(cloud_upload_events),
        
        # Behavioral Ratios (6 features)
        "access_to_role_ratio": actual_access / expected_for_role,
        "peer_deviation_score": z_score_vs_peers,
        "weekday_vs_weekend_ratio": weekday_activity / weekend_activity,
        "morning_vs_evening_ratio": am_events / pm_events,
        "productive_vs_idle_ratio": active_time / idle_time,
        "command_diversity_index": shannon_entropy(command_types),
        
        # Sequence Features (8 features)
        "action_sequence_entropy": entropy(action_sequence),
        "longest_unusual_chain": max_len(unusual_subsequences),
        "role_boundary_crossings": count(cross_role_access),
        "privilege_escalation_count": count(priv_esc_events),
        "session_action_diversity": nunique(action_types) / total_actions,
        "repeat_pattern_score": autocorrelation(action_sequence),
        "novelty_score": fraction_never_seen_before(actions),
        "behavioral_velocity": drift_rate(features, historical),
    }
```

---

### Layer 2: Behavioral Profiling — Digital Employee Twin

#### Behavioral Genome Components

```
┌────────────────────────────────────────────────────┐
│              BEHAVIORAL GENOME                      │
│                                                    │
│  ┌──────────────────┐                              │
│  │ Circadian Profile │  Fourier coefficients of    │
│  │ (8-dim vector)    │  hourly login distribution  │
│  └──────────────────┘                              │
│                                                    │
│  ┌──────────────────┐                              │
│  │ Access Embedding  │  Dense vector from resource  │
│  │ (16-dim vector)   │  access graph (node2vec)    │
│  └──────────────────┘                              │
│                                                    │
│  ┌──────────────────┐                              │
│  │ Behavioral Stats  │  Rolling mean/std of 47     │
│  │ (94-dim: 47×2)    │  features over 30-day window│
│  └──────────────────┘                              │
│                                                    │
│  ┌──────────────────┐                              │
│  │ Drift Velocity    │  Rate of behavioral change  │
│  │ (1-dim scalar)    │  over last 14 days          │
│  └──────────────────┘                              │
│                                                    │
│  Total Genome Size: 119 dimensions                 │
└────────────────────────────────────────────────────┘
```

#### Twin Update Strategy

The twin evolves using an **Exponential Moving Average** (EMA):

```
twin_baseline(t) = α × current_features + (1-α) × twin_baseline(t-1)
where α = 0.05 (slow update — 20-day effective window)
```

This ensures:
- Normal behavioral evolution is absorbed (promotions, new projects)
- Sudden anomalies are NOT absorbed (they trigger alerts instead)
- Slow-burn threats are caught (twin updates too slowly to mask gradual deviation)

---

### Layer 3: Risk Scoring Engine

#### LSTM Autoencoder Architecture

```
Input: (batch_size, seq_len=7, features=47)
                │
                ▼
┌─────────────────────────────────┐
│  LSTM Encoder Layer 1           │
│  Input: 47 → Hidden: 32        │
│  Dropout: 0.2                   │
├─────────────────────────────────┤
│  LSTM Encoder Layer 2           │
│  Input: 32 → Hidden: 32        │
│  Dropout: 0.2                   │
├─────────────────────────────────┤
│  Dense: 32 → 16                 │
│  → LATENT SPACE (16-dim)        │  ← Behavioral embedding
├─────────────────────────────────┤
│  Dense: 16 → 32                 │
├─────────────────────────────────┤
│  LSTM Decoder Layer 1           │
│  Input: 32 → Hidden: 32        │
│  Dropout: 0.2                   │
├─────────────────────────────────┤
│  LSTM Decoder Layer 2           │
│  Input: 32 → Output: 47        │
└─────────────────────────────────┘
                │
                ▼
Output: (batch_size, seq_len=7, features=47)

Loss: MSE(input, output)
Anomaly Score: per-sample reconstruction error
```

**Training Protocol**:
- Train on **normal employees only** (no insiders in training set)
- 50 epochs, batch size 32, learning rate 1e-3
- Early stopping on validation loss (patience=10)
- Anomaly threshold: 95th percentile of normal reconstruction error

#### Isolation Forest Configuration

```python
IsolationForest(
    n_estimators=500,
    contamination=0.05,      # Expected 5% anomaly rate
    max_samples='auto',
    max_features=1.0,
    random_state=42,
    n_jobs=-1
)
```

#### Ensemble Scoring

```python
def hybrid_score(lstm_recon_error, if_anomaly_score, alpha=0.65):
    """
    Combine temporal and static anomaly scores.
    
    alpha is optimized on validation set to maximize F1.
    Higher alpha → more weight on temporal patterns.
    """
    # Normalize both to [0, 100]
    lstm_normalized = normalize_to_100(lstm_recon_error)
    if_normalized = normalize_to_100(if_anomaly_score)
    
    return alpha * lstm_normalized + (1 - alpha) * if_normalized
```

---

### Layer 4: Privilege Context Engine

#### Role-Resource Risk Matrix

```python
# Risk multiplier matrix
# ROLE_RESOURCE_RISK[role][resource] → multiplier
# 1.0 = expected for this role
# Higher = increasingly suspicious

ROLE_RESOURCE_RISK = {
    "retail_banking": {
        "customer_records": 1.0,
        "crm_system": 1.0,
        "treasury_data": 8.0,
        "hr_records": 5.0,
        "admin_console": 10.0,
        "audit_logs": 6.0,
        "production_db": 9.0,
    },
    "treasury": {
        "treasury_data": 1.0,
        "customer_records": 2.0,
        "hr_records": 6.0,
        "admin_console": 9.0,
        "audit_logs": 3.0,
    },
    "it_admin": {
        "admin_console": 1.0,
        "staging_db": 1.5,
        "production_db": 3.0,
        "customer_records": 7.0,
        "treasury_data": 8.0,
        "audit_logs": 2.0,
    },
    "hr": {
        "employee_records": 1.0,
        "payroll_system": 1.5,
        "customer_records": 6.0,
        "treasury_data": 8.0,
        "admin_console": 9.0,
    },
    "compliance": {
        "audit_logs": 1.0,
        "customer_records": 2.0,
        "treasury_data": 2.5,
        "admin_console": 7.0,
    },
}
```

#### Privilege Decay Function

```python
def compute_trust_score(
    previous_trust: float,
    raw_risk_score: float,
    role_multiplier: float,
    time_since_last_normal_hours: float,
    decay_rate: float = 0.05,
    reinforcement_normal: float = 5.0,
    penalty_anomaly: float = 20.0,
) -> float:
    """
    T(t) = T(t-1) × e^(-λ × Δt) + reinforcement(t) - penalty(t)
    
    Trust is a PERISHABLE resource that must be continuously earned.
    """
    # Natural decay
    decay = math.exp(-decay_rate * time_since_last_normal_hours)
    decayed_trust = previous_trust * decay
    
    # Context-adjusted risk
    adjusted_risk = raw_risk_score * role_multiplier
    
    # Reinforcement or penalty
    if adjusted_risk < 20:  # Normal behavior
        delta = reinforcement_normal
    elif adjusted_risk < 50:  # Mild concern
        delta = 0
    else:  # Anomalous
        delta = -penalty_anomaly * (adjusted_risk / 100)
    
    # Final trust score, clamped to [0, 100]
    new_trust = max(0, min(100, decayed_trust + delta))
    return new_trust
```

---

### Layer 5: Explainable Alert Engine

#### Intent Signal Chains

Pre-defined attack patterns that the system matches against:

```python
INTENT_CHAINS = {
    "data_exfiltration": {
        "sequence": [
            "login_after_hours",
            "access_sensitive_data",
            "bulk_download",
            "usb_connect",
            "file_copy_to_usb",
        ],
        "min_match": 3,  # At least 3 of 5 steps must match
        "severity": "CRITICAL",
    },
    "pre_resignation_theft": {
        "sequence": [
            "job_search_browsing",
            "increased_file_access",
            "bulk_email_to_personal",
            "cloud_upload_spike",
        ],
        "min_match": 3,
        "severity": "HIGH",
    },
    "privilege_abuse": {
        "sequence": [
            "login_from_new_device",
            "privilege_escalation",
            "cross_role_data_access",
            "audit_log_access",
        ],
        "min_match": 2,
        "severity": "CRITICAL",
    },
    "credential_compromise": {
        "sequence": [
            "login_from_unusual_location",
            "rapid_system_switching",
            "access_pattern_mismatch",
            "data_access_outside_role",
        ],
        "min_match": 2,
        "severity": "HIGH",
    },
    "slow_burn_recon": {
        "sequence": [
            "gradual_scope_expansion",
            "new_system_access_weekly",
            "increasing_data_volume",
            "off_role_queries",
        ],
        "min_match": 3,
        "severity": "MEDIUM",
        "window_days": 30,  # Look back 30 days for this pattern
    },
    "collusion": {
        "sequence": [
            "coordinated_access_timing",
            "complementary_data_access",
            "shared_external_recipient",
            "overlapping_usb_usage",
        ],
        "min_match": 2,
        "severity": "CRITICAL",
    },
}
```

---

## API Design

### REST Endpoints

| Method | Endpoint | Description | Response |
|--------|---------|-------------|----------|
| `POST` | `/api/score` | Score a single employee's current activity | Trust score + risk factors |
| `POST` | `/api/score/batch` | Score multiple employees | Array of trust scores |
| `GET` | `/api/employees` | List all employees with current trust scores | Paginated employee list |
| `GET` | `/api/employees/{id}` | Get employee details + twin profile | Employee object |
| `GET` | `/api/employees/{id}/twin` | Get Digital Twin behavioral genome | Twin object |
| `GET` | `/api/employees/{id}/timeline` | Get activity timeline with anomaly markers | Timeline events |
| `GET` | `/api/alerts` | Get active alerts, sorted by severity | Alert list |
| `GET` | `/api/alerts/{id}` | Get detailed alert with explanation | Alert detail |
| `GET` | `/api/peers/{id}` | Get peer constellation comparison | Peer comparison |
| `GET` | `/api/stats` | Get overall system statistics | Dashboard stats |
| `GET` | `/api/model/performance` | Get model performance metrics | Metrics object |
| `WS` | `/ws/stream` | WebSocket for real-time trust score updates | Streaming updates |

### Example API Response: `/api/score`

```json
{
    "employee_id": "EMP_047",
    "name": "Raj Sharma",
    "department": "retail_banking",
    "role": "relationship_manager",
    "trust_score": 23,
    "trust_level": "HIGH_RISK",
    "previous_trust_score": 94,
    "timestamp": "2025-06-15T22:47:00+05:30",
    "risk_factors": [
        {
            "factor": "New workstation detected",
            "detail": "WS_DEL_019 — never used before",
            "impact": -15,
            "icon": "📍"
        },
        {
            "factor": "After-hours access",
            "detail": "22:47 IST — outside normal hours (09:00-18:00)",
            "impact": -12,
            "icon": "🕐"
        },
        {
            "factor": "Excessive data access",
            "detail": "847 customer records — 15× daily average (56)",
            "impact": -25,
            "icon": "📊"
        },
        {
            "factor": "USB device connected",
            "detail": "No USB usage in 36-month history",
            "impact": -10,
            "icon": "💾"
        },
        {
            "factor": "Cross-role resource access",
            "detail": "Treasury database — not in Relationship Manager scope",
            "impact": -20,
            "icon": "🏦"
        }
    ],
    "intent_chain_match": {
        "pattern": "data_exfiltration",
        "confidence": 0.89,
        "matched_steps": ["access_sensitive_data", "bulk_download", "usb_connect"]
    },
    "peer_comparison": {
        "peer_group_size": 47,
        "peer_avg_records_accessed": 52,
        "peer_treasury_access_pct": 0.0,
        "deviation_z_score": 14.2
    },
    "recommended_action": "SUSPEND_SESSION",
    "recommended_detail": "Suspend session immediately. Notify CISO. Initiate investigation."
}
```

---

## Dashboard Views

### 1. Trust Score Heatmap
- Grid of all employees, color-coded by trust score
- Green (80-100) → Yellow (60-80) → Orange (40-60) → Red (20-40) → Dark Red (0-20)
- Click any cell to drill into employee detail

### 2. Real-Time Alert Queue
- Sorted by severity (Critical → High → Medium → Low)
- Each alert shows: employee, trust score change, top risk factor, time
- One-click to open full investigation view

### 3. Digital Twin Comparison
- Side-by-side: "Expected Behavior" (from twin) vs. "Actual Behavior"
- Radar chart with 8 behavioral dimensions
- Highlighted dimensions where actual ≠ expected

### 4. Privilege Decay Curve
- Real-time line chart showing trust score over the current day
- Annotations at each event (login, access, USB, etc.)
- Threshold lines showing alert trigger points

### 5. Timeline View
- Chronological event stream for selected employee
- Events color-coded by risk contribution
- Intent chain overlay showing matched attack patterns

### 6. Model Performance
- ROC/PR curves, confusion matrix, F1 over time
- Active model version and performance trends

### 7. SHAP Explainability (NEW)
- Per-employee SHAP waterfall chart showing top risk/protective factors
- Feature importance bars with SHAP values and feature values
- Base value → prediction flow visualization

---

## Enhanced Pipeline v2.0

> The following section documents the **enhanced pipeline** that replaces the original Layer 3 
> risk scoring with a supervised ensemble achieving F1=0.949 and AUC=0.983.

### Feature Engineering: 47 → 211 Dimensions

The enhanced feature engineer (`argus/data/enhanced_feature_engineer.py`) expands the original 
47 base features into **211 dimensions** across 7 new categories:

| Category | Count | Description |
|----------|-------|-------------|
| Base features | 47 | Original temporal, access, device, communication, data movement, behavioral, sequence features |
| Clearance | 1 | `clearance_normalized` — security clearance level (0-1) |
| Rolling 7-day | ~35 | `roll_7d_{mean,std,max,sum}_{feature}` — 7-day windowed statistics |
| Rolling 14-day | ~55 | `roll_14d_{mean,std,max,sum}_{feature}` — 14-day windowed statistics |
| Expanding | ~10 | `expanding_{max,mean}_{feature}` — all-time expanding statistics |
| Deltas | ~20 | `delta_{feature}`, `abs_delta_{feature}` — day-over-day changes |
| Z-scores | ~20 | `zscore_{dept,role}_{feature}` — peer-relative deviation scores |

### Enhanced Risk Scoring Engine

```
                      211 Features
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
    ┌──────────────┐ ┌──────────┐ ┌──────────────┐
    │  LightGBM    │ │ XGBoost  │ │ LSTM-AE +    │
    │              │ │          │ │ Isolation     │
    │  500 trees   │ │ 500 trees│ │ Forest        │
    │  depth=6     │ │ depth=6  │ │ (anomaly)     │
    │  F1=0.949    │ │ F1=0.935 │ │ F1=0.873     │
    └──────┬───────┘ └────┬─────┘ └──────┬───────┘
           │              │              │
           └──────────────┼──────────────┘
                          ▼
              ┌──────────────────────┐
              │  META-LEARNER        │
              │  (Logistic Regression│
              │   on probabilities)  │
              │                      │
              │  Combines P(insider) │
              │  from all 3 paths    │
              └──────────┬───────────┘
                         ▼
                   FINAL SCORE
                   (0.0 — 1.0)
```

**Key metrics (5-Fold CV)**:
- F1: **0.935 ± 0.022**
- AUC-ROC: **0.991 ± 0.002**
- Precision: **0.990 ± 0.008**
- FPR: **0.03%** (well below 2% target)

### SHAP Explainability Layer

Uses `TreeExplainer` (exact Shapley values for GBDT):

**Top global features** (by mean |SHAP|):
1. `clearance_normalized` — 0.610
2. `roll_7d_max_data_volume_mb` — 0.532
3. `expanding_max_systems` — 0.446
4. `login_hour` — 0.375
5. `temporal_entropy` — 0.372

**Per-employee explanations** via `/api/explain/{emp_id}`:
- Top 5 risk-increasing factors (feature, SHAP value, raw value)
- Top 3 protective factors
- Base value → prediction probability flow

### Federated Stacking (Privacy-Compliant Alternative)

For regulatory environments requiring data locality:

```
  Dept A        Dept B        Dept C
    │             │             │
    ▼             ▼             ▼
  Local LGB    Local LGB    Local LGB
    │             │             │
    ▼             ▼             ▼
  P(insider)   P(insider)   P(insider)
    │             │             │
    └─────────────┼─────────────┘
                  ▼
        ┌──────────────────┐
        │  Global Meta-    │
        │  Learner         │
        │  (on predictions │
        │   only — no raw  │
        │   data shared)   │
        └──────────────────┘
```

- AUC: 0.974 (vs centralized 0.983)
- Only scalar probabilities leave each department
- Zero raw feature sharing — minimal privacy risk

### Updated API Endpoints (v2.0)

| Method | Endpoint | Description |
|--------|---------|-------------|
| `GET` | `/api/health` | Health check — models loaded, feature count, enhanced mode |
| `GET` | `/api/overview` | Dashboard overview — threats, metrics, distribution |
| `GET` | `/api/employees` | All employees with trust/risk scores (sortable, filterable) |
| `GET` | `/api/employee/{emp_id}` | Employee detail + twin comparison + trust timeline |
| `GET` | `/api/alerts?limit=N` | Top N flagged employees with intent chains |
| `GET` | `/api/analytics` | Model metrics, top features, department stats |
| `GET` | `/api/explain/{emp_id}` | SHAP explanation for individual employee |
| `GET` | `/api/activity` | Live activity feed |

### File Structure

```
argus/
├── api/
│   └── scoring_api.py          # FastAPI server (v2.0)
├── data/
│   ├── synthetic_generator.py  # Banking scenario generator
│   ├── enhanced_feature_engineer.py  # 47 → 211 features
│   └── feature_engineer.py     # Original 47 features
├── models/
│   ├── lstm_autoencoder.py     # Temporal anomaly detector
│   ├── isolation_forest.py     # Static anomaly detector
│   ├── shap_explainer.py       # TreeExplainer module
│   └── digital_twin.py        # Behavioral genome builder
├── privacy/
│   └── federated_stacking.py   # One-shot federated learning
├── experiments/
│   ├── cross_validation.py     # 5-fold stratified CV
│   └── ablation_study.py       # Feature/model ablation
└── scoring/
    ├── trust_engine.py         # Privilege decay function
    ├── intent_chains.py        # Attack pattern matcher
    └── alert_engine.py         # Alert generation
```
