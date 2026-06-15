"""
Argus AI — Pydantic Data Schemas
=================================
Defines strict data models for employees, activities, features, and ground truth.
Used for validation throughout the pipeline.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class EmployeeProfile(BaseModel):
    """Employee profile schema."""
    emp_id: str = Field(..., pattern=r"^EMP_\d{3}$")
    name: str
    department: str
    role: str
    clearance_level: int = Field(..., ge=1, le=5)
    branch: str
    tenure_months: int = Field(..., ge=0)
    hire_date: str
    typical_login_hour: float = Field(..., ge=0, le=24)
    typical_logout_hour: float = Field(..., ge=0, le=24)
    typical_actions_per_day: int = Field(..., ge=0)
    typical_records_per_day: int = Field(..., ge=0)
    is_insider: bool = False
    insider_scenario: Optional[str] = None
    attack_start_day: Optional[int] = None
    attack_end_day: Optional[int] = None


class ActivityEvent(BaseModel):
    """Single activity event schema."""
    event_id: str
    timestamp: datetime
    emp_id: str
    action_type: str
    system: str
    resource: Optional[str] = None
    records_accessed: int = 0
    data_volume_mb: float = 0.0
    device_id: str = ""
    ip_address: str = ""
    is_after_hours: bool = False
    is_new_device: bool = False
    geo_location: str = ""
    day_index: int = 0


class GroundTruth(BaseModel):
    """Ground truth label for an insider."""
    emp_id: str
    is_insider: bool
    scenario: str
    attack_start_day: int
    attack_end_day: int
    description: str


class FeatureVector(BaseModel):
    """47-feature employee-day vector."""
    emp_id: str
    day_index: int
    date: str
    label: int = 0  # 0=normal, 1=insider-active

    # Temporal (8)
    login_hour: float = 0.0
    logout_hour: float = 0.0
    session_duration_hrs: float = 0.0
    is_weekend: int = 0
    is_after_hours: int = 0
    time_since_last_session: float = 0.0
    login_regularity_score: float = 0.0
    temporal_entropy: float = 0.0

    # Access Volume (7)
    files_accessed: int = 0
    emails_sent: int = 0
    emails_received: int = 0
    urls_visited: int = 0
    usb_events: int = 0
    data_volume_mb: float = 0.0
    unique_systems_accessed: int = 0

    # Device & Location (5)
    is_new_device: int = 0
    device_count: int = 0
    unique_pcs: int = 0
    geo_anomaly_flag: int = 0
    vpn_usage: int = 0

    # Communication (6)
    external_email_ratio: float = 0.0
    avg_attachment_size: float = 0.0
    unique_recipients: int = 0
    cc_bcc_ratio: float = 0.0
    email_content_sentiment: float = 0.0
    unusual_recipient_flag: int = 0

    # Data Movement (7)
    file_copy_count: int = 0
    usb_file_transfers: int = 0
    large_download_flag: int = 0
    sensitive_file_access: int = 0
    data_egress_volume: float = 0.0
    print_count: int = 0
    cloud_upload_count: int = 0

    # Behavioral Ratios (6)
    access_to_role_ratio: float = 0.0
    peer_deviation_score: float = 0.0
    weekday_vs_weekend_ratio: float = 0.0
    morning_vs_evening_ratio: float = 0.0
    productive_vs_idle_ratio: float = 0.0
    command_diversity_index: float = 0.0

    # Sequence (8)
    action_sequence_entropy: float = 0.0
    longest_unusual_chain: int = 0
    role_boundary_crossings: int = 0
    privilege_escalation_count: int = 0
    session_action_diversity: float = 0.0
    repeat_pattern_score: float = 0.0
    novelty_score: float = 0.0
    behavioral_velocity: float = 0.0
