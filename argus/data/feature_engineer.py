"""
Argus AI — Feature Engineering Pipeline
=========================================
Transforms raw activity logs into 47-dimensional feature vectors per employee-day.
Produces sequences ready for LSTM autoencoder and tabular data for Isolation Forest.

Feature Categories (47 total):
    - Temporal (8):       Login time, session duration, after-hours, regularity
    - Access Volume (7):  Files, emails, URLs, USB, data volume, unique systems
    - Device/Location (5): New device, device count, unique PCs, geo anomaly, VPN
    - Communication (6):  External email ratio, attachment size, recipients, sentiment
    - Data Movement (7):  File copy, USB transfers, downloads, sensitive access, egress
    - Behavioral Ratio (6): Access-to-role, peer deviation, weekday/weekend patterns
    - Sequence (8):       Action entropy, unusual chains, role boundary, privilege esc

Usage:
    python -m argus.data.feature_engineer
    python -m argus.data.feature_engineer --sequence-length 7
"""

import sys
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from argus.config import Config

# Systems that are outside typical scope for most roles
SENSITIVE_SYSTEMS = {
    "Production_CBS", "Customer_Records_DB", "Treasury_DB",
    "Audit_Logs", "Admin_Console", "Servers",
}

# Role → allowed systems mapping
ROLE_SYSTEM_SCOPE = {
    "relationship_manager": {"CRM", "CBS", "Email", "Reports"},
    "teller": {"CBS", "Teller_Terminal", "Email"},
    "branch_manager": {"CRM", "CBS", "Email", "Reports"},
    "trader": {"Treasury_Platform", "Bloomberg", "Email"},
    "treasury_analyst": {"Treasury_Platform", "Reports", "Email"},
    "system_admin": {"Admin_Console", "Servers", "Email", "JIRA"},
    "dba_admin": {"DB_Console", "Staging_DB", "Email", "JIRA"},
    "help_desk": {"Ticketing", "AD_Console", "Email"},
    "hr_generalist": {"HRMS", "Email", "Documents"},
    "recruiter": {"HRMS", "Email", "ATS"},
    "payroll": {"Payroll_System", "HRMS", "Email"},
    "aml_analyst": {"AML_Platform", "CBS", "Email", "Reports"},
    "auditor": {"Audit_System", "CBS", "Email"},
    "risk_officer": {"Risk_Platform", "CBS", "Email", "Reports"},
}


def engineer_features(
    activity_path: str | Path | None = None,
    employees_path: str | Path | None = None,
    ground_truth_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    sequence_length: int = 7,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Transform raw activity logs into 47-feature vectors.

    Returns:
        (features_df, X_sequences, y_labels)
        - features_df: DataFrame with emp_id, day_index, 47 features, label
        - X_sequences: shape (n_samples, seq_len, 47) for LSTM
        - y_labels: shape (n_samples,) binary labels
    """
    data_dir = Config.paths.SYNTHETIC_DATA
    out_dir = Path(output_dir) if output_dir else Config.paths.PROCESSED_DATA
    out_dir.mkdir(parents=True, exist_ok=True)

    activity_path = activity_path or data_dir / "activity_log.csv"
    employees_path = employees_path or data_dir / "employees.csv"
    ground_truth_path = ground_truth_path or data_dir / "ground_truth.csv"

    logger.info("Loading data files...")
    activity = pd.read_csv(activity_path)
    employees = pd.read_csv(employees_path)
    ground_truth = pd.read_csv(ground_truth_path)

    activity["timestamp"] = pd.to_datetime(activity["timestamp"])

    # Build insider lookup: emp_id → (start_day, end_day)
    insider_map = {}
    for _, row in ground_truth.iterrows():
        insider_map[row["emp_id"]] = (row["attack_start_day"], row["attack_end_day"])

    # Build employee role lookup
    emp_roles = dict(zip(employees["emp_id"], employees["role"]))

    # ─── Step 1: Compute per-employee-day features ───
    logger.info("Step 1/3: Computing 47-dimensional feature vectors...")
    all_features = []

    grouped = activity.groupby(["emp_id", "day_index"])
    total_groups = len(grouped)

    for (emp_id, day_idx), day_events in tqdm(grouped, total=total_groups, desc="Features"):
        features = _compute_day_features(emp_id, day_idx, day_events, emp_roles.get(emp_id, ""))

        # Label: 1 if insider AND within attack window
        label = 0
        if emp_id in insider_map:
            start, end = insider_map[emp_id]
            if start <= day_idx <= end:
                label = 1

        features["label"] = label
        all_features.append(features)

    features_df = pd.DataFrame(all_features)
    features_df = features_df.sort_values(["emp_id", "day_index"]).reset_index(drop=True)

    # ─── Step 2: Compute peer-relative features ───
    logger.info("Step 2/3: Computing peer-relative features...")
    features_df = _add_peer_features(features_df, employees)

    # ─── Step 3: Build sequences for LSTM ───
    logger.info(f"Step 3/3: Building {sequence_length}-day sequences for LSTM...")
    feature_cols = _get_feature_columns()
    X_sequences, y_labels, seq_meta = _build_sequences(
        features_df, feature_cols, sequence_length
    )

    # ─── Save outputs ───
    features_df.to_csv(out_dir / "features_47d.csv", index=False)
    np.save(out_dir / "X_sequences.npy", X_sequences)
    np.save(out_dir / "y_labels.npy", y_labels)
    pd.DataFrame(seq_meta).to_csv(out_dir / "sequence_meta.csv", index=False)

    # ─── Summary stats ───
    n_positive = int(y_labels.sum())
    n_total = len(y_labels)
    logger.success(f"✅ Feature engineering complete!")
    logger.info(f"   Feature vectors: {len(features_df):,} (emp-day pairs)")
    logger.info(f"   Feature dimensions: {len(feature_cols)}")
    logger.info(f"   LSTM sequences: {X_sequences.shape} (samples, seq_len, features)")
    logger.info(f"   Labels: {n_positive} positive / {n_total - n_positive} negative ({n_positive/max(1,n_total)*100:.2f}% positive)")
    logger.info(f"   Saved to: {out_dir}")

    return features_df, X_sequences, y_labels


def _compute_day_features(
    emp_id: str,
    day_idx: int,
    events: pd.DataFrame,
    role: str,
) -> dict:
    """Compute all 47 features for a single employee-day."""
    n_events = len(events)

    # ─── Temporal (8) ───
    login_event = events[events["action_type"] == "login"]
    logout_event = events[events["action_type"] == "logout"]

    if len(login_event) > 0:
        login_ts = login_event.iloc[0]["timestamp"]
        login_hour = login_ts.hour + login_ts.minute / 60
    else:
        login_hour = events.iloc[0]["timestamp"].hour + events.iloc[0]["timestamp"].minute / 60

    if len(logout_event) > 0:
        logout_ts = logout_event.iloc[0]["timestamp"]
        logout_hour = logout_ts.hour + logout_ts.minute / 60
    else:
        logout_hour = events.iloc[-1]["timestamp"].hour + events.iloc[-1]["timestamp"].minute / 60

    session_duration = max(0, logout_hour - login_hour)
    is_weekend = int(events.iloc[0]["timestamp"].weekday() >= 5) if "timestamp" in events else 0
    is_after_hours = int(events.get("is_after_hours", pd.Series([False])).any())

    # Temporal entropy: how spread out are actions across the day
    hours = events["timestamp"].dt.hour.values
    hour_counts = Counter(hours)
    probs = np.array(list(hour_counts.values()), dtype=float)
    probs = probs / probs.sum()
    temporal_entropy = float(scipy_stats.entropy(probs)) if len(probs) > 1 else 0.0

    # Login regularity score (deviation from expected — will be normalized later)
    login_regularity = abs(login_hour - 9.0)

    # ─── Access Volume (7) ───
    action_types = events["action_type"].values
    files_accessed = sum(1 for a in action_types if "file" in str(a).lower() or "doc" in str(a).lower() or "view" in str(a).lower())
    emails_sent = sum(1 for a in action_types if "send_email" in str(a))
    emails_received = sum(1 for a in action_types if "read_email" in str(a))
    urls_visited = sum(1 for a in action_types if "browse" in str(a).lower() or "web" in str(a).lower())
    usb_events = sum(1 for a in action_types if "usb" in str(a).lower())
    data_volume = events["data_volume_mb"].sum()
    unique_systems = events["system"].nunique()

    # ─── Device & Location (5) ───
    is_new_device = int(events.get("is_new_device", pd.Series([False])).any())
    device_count = events["device_id"].nunique() if "device_id" in events else 1
    unique_pcs = events["device_id"].apply(lambda x: x if not str(x).startswith("USB") else "").nunique() if "device_id" in events else 1
    geo_anomaly = int(events["geo_location"].nunique() > 1) if "geo_location" in events else 0
    vpn_usage = int(any("vpn" in str(ip).lower() or str(ip).startswith("103.") for ip in events.get("ip_address", [])))

    # ─── Communication (6) ───
    total_emails = emails_sent + emails_received
    external_email = sum(1 for a in action_types if "external" in str(a))
    external_email_ratio = external_email / max(1, total_emails)
    avg_attachment_size = data_volume / max(1, emails_sent) if emails_sent > 0 else 0.0
    unique_recipients = max(1, emails_sent)  # Simplified: proxy
    cc_bcc_ratio = 0.0  # Placeholder
    email_sentiment = 0.0  # Placeholder
    unusual_recipient = int(external_email > 3)

    # ─── Data Movement (7) ───
    file_copy = sum(1 for a in action_types if "copy" in str(a).lower() or "export" in str(a).lower())
    usb_transfers = sum(1 for a in action_types if "usb" in str(a).lower() and "connect" not in str(a).lower())
    large_download = int(data_volume > 10.0)
    sensitive_access = sum(1 for s in events["system"] if s in SENSITIVE_SYSTEMS)
    data_egress = data_volume * (0.3 if file_copy > 0 or usb_events > 0 else 0.05)
    print_count = sum(1 for a in action_types if "print" in str(a).lower())
    cloud_upload = sum(1 for a in action_types if "cloud" in str(a).lower() or "upload" in str(a).lower())

    # ─── Behavioral Ratios (6) ───
    allowed_systems = ROLE_SYSTEM_SCOPE.get(role, set())
    out_of_scope = sum(1 for s in events["system"] if s not in allowed_systems and s != "Auth" and s != "Device_Manager" and s != "Web_Browser")
    access_to_role_ratio = out_of_scope / max(1, n_events)
    peer_deviation = 0.0  # Will be computed in _add_peer_features
    weekday_weekend_ratio = 0.0  # Will be context-dependent
    morning_vs_evening = (sum(1 for h in hours if h < 12) / max(1, len(hours)))
    productive_idle = n_events / max(1, session_duration * 4)  # Actions per quarter-hour
    cmd_diversity = len(set(action_types)) / max(1, n_events)

    # ─── Sequence (8) ───
    action_list = [str(a) for a in action_types]
    action_counts = Counter(action_list)
    action_probs = np.array(list(action_counts.values()), dtype=float)
    action_probs = action_probs / action_probs.sum()
    action_entropy = float(scipy_stats.entropy(action_probs)) if len(action_probs) > 1 else 0.0

    # Longest unusual chain: consecutive out-of-scope actions
    longest_unusual = 0
    current_chain = 0
    for s in events["system"]:
        if s not in allowed_systems and s != "Auth" and s != "Device_Manager":
            current_chain += 1
            longest_unusual = max(longest_unusual, current_chain)
        else:
            current_chain = 0

    role_boundary_crossings = out_of_scope
    priv_escalation_count = sum(1 for a in action_types if "escalat" in str(a).lower() or "superadmin" in str(a).lower())
    session_action_diversity = len(set(action_types))
    repeat_pattern = 1.0 - cmd_diversity  # Inverse of diversity
    novelty_score = is_new_device + geo_anomaly + int(usb_events > 0) + int(sensitive_access > 0)
    behavioral_velocity = n_events / max(1, session_duration)

    return {
        "emp_id": emp_id,
        "day_index": day_idx,
        "date": str(events.iloc[0]["timestamp"].date()),
        # Temporal (8)
        "login_hour": round(login_hour, 2),
        "logout_hour": round(logout_hour, 2),
        "session_duration_hrs": round(session_duration, 2),
        "is_weekend": is_weekend,
        "is_after_hours": is_after_hours,
        "time_since_last_session": 0.0,  # Computed in sequence pass
        "login_regularity_score": round(login_regularity, 2),
        "temporal_entropy": round(temporal_entropy, 4),
        # Access Volume (7)
        "files_accessed": files_accessed,
        "emails_sent": emails_sent,
        "emails_received": emails_received,
        "urls_visited": urls_visited,
        "usb_events": usb_events,
        "data_volume_mb": round(data_volume, 3),
        "unique_systems_accessed": unique_systems,
        # Device & Location (5)
        "is_new_device": is_new_device,
        "device_count": device_count,
        "unique_pcs": unique_pcs,
        "geo_anomaly_flag": geo_anomaly,
        "vpn_usage": vpn_usage,
        # Communication (6)
        "external_email_ratio": round(external_email_ratio, 4),
        "avg_attachment_size": round(avg_attachment_size, 3),
        "unique_recipients": unique_recipients,
        "cc_bcc_ratio": cc_bcc_ratio,
        "email_content_sentiment": email_sentiment,
        "unusual_recipient_flag": unusual_recipient,
        # Data Movement (7)
        "file_copy_count": file_copy,
        "usb_file_transfers": usb_transfers,
        "large_download_flag": large_download,
        "sensitive_file_access": sensitive_access,
        "data_egress_volume": round(data_egress, 3),
        "print_count": print_count,
        "cloud_upload_count": cloud_upload,
        # Behavioral Ratios (6)
        "access_to_role_ratio": round(access_to_role_ratio, 4),
        "peer_deviation_score": peer_deviation,
        "weekday_vs_weekend_ratio": weekday_weekend_ratio,
        "morning_vs_evening_ratio": round(morning_vs_evening, 4),
        "productive_vs_idle_ratio": round(productive_idle, 4),
        "command_diversity_index": round(cmd_diversity, 4),
        # Sequence (8)
        "action_sequence_entropy": round(action_entropy, 4),
        "longest_unusual_chain": longest_unusual,
        "role_boundary_crossings": role_boundary_crossings,
        "privilege_escalation_count": priv_escalation_count,
        "session_action_diversity": session_action_diversity,
        "repeat_pattern_score": round(repeat_pattern, 4),
        "novelty_score": novelty_score,
        "behavioral_velocity": round(behavioral_velocity, 4),
    }


def _add_peer_features(features_df: pd.DataFrame, employees_df: pd.DataFrame) -> pd.DataFrame:
    """Add peer-relative features (deviation from department average)."""
    # Merge department info
    features_df = features_df.merge(
        employees_df[["emp_id", "department", "role"]],
        on="emp_id", how="left",
    )

    # Compute department-day averages for key features
    peer_cols = ["data_volume_mb", "files_accessed", "unique_systems_accessed",
                 "login_hour", "session_duration_hrs"]

    dept_day_avg = features_df.groupby(["department", "day_index"])[peer_cols].transform("mean")
    dept_day_std = features_df.groupby(["department", "day_index"])[peer_cols].transform("std").fillna(1.0).replace(0, 1.0)

    # Peer deviation: average z-score across peer features
    z_scores = (features_df[peer_cols] - dept_day_avg) / dept_day_std
    features_df["peer_deviation_score"] = z_scores.mean(axis=1).round(4)

    # Time since last session
    features_df = features_df.sort_values(["emp_id", "day_index"])
    features_df["time_since_last_session"] = features_df.groupby("emp_id")["day_index"].diff().fillna(1.0)

    # Weekday vs weekend ratio (per employee rolling)
    features_df["weekday_vs_weekend_ratio"] = features_df.groupby("emp_id")["is_weekend"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    ).round(4)

    # Drop helper columns
    features_df = features_df.drop(columns=["department", "role"], errors="ignore")

    return features_df


def _get_feature_columns() -> list[str]:
    """Return the 47 feature column names in order."""
    return [
        # Temporal (8)
        "login_hour", "logout_hour", "session_duration_hrs", "is_weekend",
        "is_after_hours", "time_since_last_session", "login_regularity_score",
        "temporal_entropy",
        # Access Volume (7)
        "files_accessed", "emails_sent", "emails_received", "urls_visited",
        "usb_events", "data_volume_mb", "unique_systems_accessed",
        # Device & Location (5)
        "is_new_device", "device_count", "unique_pcs", "geo_anomaly_flag",
        "vpn_usage",
        # Communication (6)
        "external_email_ratio", "avg_attachment_size", "unique_recipients",
        "cc_bcc_ratio", "email_content_sentiment", "unusual_recipient_flag",
        # Data Movement (7)
        "file_copy_count", "usb_file_transfers", "large_download_flag",
        "sensitive_file_access", "data_egress_volume", "print_count",
        "cloud_upload_count",
        # Behavioral Ratios (6)
        "access_to_role_ratio", "peer_deviation_score", "weekday_vs_weekend_ratio",
        "morning_vs_evening_ratio", "productive_vs_idle_ratio",
        "command_diversity_index",
        # Sequence (8)
        "action_sequence_entropy", "longest_unusual_chain",
        "role_boundary_crossings", "privilege_escalation_count",
        "session_action_diversity", "repeat_pattern_score",
        "novelty_score", "behavioral_velocity",
    ]


def _build_sequences(
    features_df: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Build sliding-window sequences for LSTM input."""
    sequences = []
    labels = []
    meta = []

    for emp_id, emp_data in features_df.groupby("emp_id"):
        emp_data = emp_data.sort_values("day_index")
        values = emp_data[feature_cols].values.astype(np.float32)
        day_labels = emp_data["label"].values

        # Replace NaN/inf with 0
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        if len(values) < seq_len:
            continue

        for i in range(len(values) - seq_len + 1):
            seq = values[i:i + seq_len]
            # Label: 1 if ANY day in the window is insider-active
            lbl = int(day_labels[i:i + seq_len].max())
            sequences.append(seq)
            labels.append(lbl)
            meta.append({
                "emp_id": emp_id,
                "start_day": int(emp_data.iloc[i]["day_index"]),
                "end_day": int(emp_data.iloc[i + seq_len - 1]["day_index"]),
                "label": lbl,
            })

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    return X, y, meta


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Argus AI — Feature Engineering Pipeline")
    parser.add_argument("--sequence-length", type=int, default=7, help="LSTM sequence length")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    Config.paths.ensure_dirs()
    engineer_features(
        sequence_length=args.sequence_length,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
