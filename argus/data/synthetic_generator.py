"""
Argus AI — Synthetic Banking Data Generator
=============================================
Generates realistic banking employee activity data with insider threat scenarios.

Research Basis:
    - Markov Chain activity models (Le & Zincir-Heywood, 2021)
    - Gaussian Mixture temporal distributions (Yuan & Wu, 2021)
    - Scenario injection (Glasser & Lindauer, 2013 / CERT methodology)
    - Perturbation-based threat generation

Usage:
    python -m argus.data.synthetic_generator
    python -m argus.data.synthetic_generator --validate
    python -m argus.data.synthetic_generator --employees 200 --days 90
"""

import os
import sys
import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from argus.config import Config

# ═══════════════════════════════════════════════════════════════
#  CONSTANTS & ROLE DEFINITIONS
# ═══════════════════════════════════════════════════════════════

INDIAN_FIRST_NAMES = [
    "Aarav", "Priya", "Vikram", "Ananya", "Rohan", "Kavya", "Amit", "Sneha",
    "Raj", "Deepa", "Karthik", "Meera", "Sanjay", "Pooja", "Varun", "Neha",
    "Arjun", "Divya", "Rahul", "Nisha", "Suresh", "Lakshmi", "Aditya", "Riya",
    "Vivek", "Shruti", "Manish", "Swati", "Nikhil", "Anjali", "Gaurav", "Tanvi",
    "Harsh", "Pallavi", "Saurabh", "Komal", "Akash", "Megha", "Vishal", "Simran",
    "Prakash", "Aarti", "Devesh", "Bhavna", "Tarun", "Jyoti", "Kunal", "Vandana",
    "Mohit", "Geeta", "Ashish", "Preeti", "Rajesh", "Sarita", "Nilesh", "Madhuri",
    "Dinesh", "Rekha", "Sunil", "Anita", "Pankaj", "Sunita", "Vijay", "Shalini",
    "Ajay", "Rashmi", "Manoj", "Smita", "Shyam", "Namita", "Ramesh", "Archana",
    "Sachin", "Sapna", "Tushar", "Chhaya", "Yogesh", "Renuka", "Hemant", "Varsha",
]

INDIAN_LAST_NAMES = [
    "Sharma", "Patel", "Singh", "Mehta", "Gupta", "Reddy", "Nair", "Joshi",
    "Malhotra", "Krishnan", "Iyer", "Das", "Verma", "Bose", "Agarwal", "Tiwari",
    "Menon", "Saxena", "Nanda", "Kapoor", "Chauhan", "Rao", "Mishra", "Pandey",
    "Kulkarni", "Desai", "Shah", "Pillai", "Banerjee", "Mukherjee", "Sen", "Roy",
    "Ghosh", "Dutta", "Sinha", "Kumar", "Yadav", "Jain", "Thakur", "Trivedi",
]

BRANCHES = [
    "Mumbai Main", "Delhi NCR", "Bangalore Tech", "Chennai South",
    "Pune Central", "Kolkata Main", "Hyderabad", "Ahmedabad",
]

# Role definitions with behavioral baselines
ROLE_PROFILES = {
    "retail_banking": {
        "relationship_manager": {
            "clearance": 3, "login_mean": 9.25, "login_std": 0.3,
            "logout_mean": 18.5, "logout_std": 0.4,
            "actions_mean": 45, "actions_std": 12,
            "records_mean": 55, "records_std": 15,
            "emails_mean": 8, "emails_std": 3,
            "systems": ["CRM", "CBS", "Email"],
            "data_volume_mean": 2.3, "data_volume_std": 1.1,
        },
        "teller": {
            "clearance": 1, "login_mean": 9.0, "login_std": 0.15,
            "logout_mean": 17.0, "logout_std": 0.2,
            "actions_mean": 60, "actions_std": 15,
            "records_mean": 80, "records_std": 20,
            "emails_mean": 3, "emails_std": 2,
            "systems": ["CBS", "Teller_Terminal"],
            "data_volume_mean": 1.5, "data_volume_std": 0.5,
        },
        "branch_manager": {
            "clearance": 4, "login_mean": 8.75, "login_std": 0.25,
            "logout_mean": 19.0, "logout_std": 0.5,
            "actions_mean": 35, "actions_std": 10,
            "records_mean": 30, "records_std": 10,
            "emails_mean": 15, "emails_std": 5,
            "systems": ["CRM", "CBS", "Email", "Reports"],
            "data_volume_mean": 3.0, "data_volume_std": 1.5,
        },
    },
    "treasury": {
        "trader": {
            "clearance": 4, "login_mean": 8.5, "login_std": 0.2,
            "logout_mean": 18.0, "logout_std": 0.6,
            "actions_mean": 50, "actions_std": 15,
            "records_mean": 25, "records_std": 8,
            "emails_mean": 10, "emails_std": 4,
            "systems": ["Treasury_Platform", "Bloomberg", "Email"],
            "data_volume_mean": 5.0, "data_volume_std": 2.5,
        },
        "treasury_analyst": {
            "clearance": 4, "login_mean": 9.0, "login_std": 0.25,
            "logout_mean": 18.5, "logout_std": 0.4,
            "actions_mean": 40, "actions_std": 10,
            "records_mean": 20, "records_std": 6,
            "emails_mean": 7, "emails_std": 3,
            "systems": ["Treasury_Platform", "Reports", "Email"],
            "data_volume_mean": 4.0, "data_volume_std": 2.0,
        },
    },
    "it_admin": {
        "system_admin": {
            "clearance": 5, "login_mean": 9.5, "login_std": 0.5,
            "logout_mean": 18.5, "logout_std": 0.8,
            "actions_mean": 30, "actions_std": 12,
            "records_mean": 10, "records_std": 5,
            "emails_mean": 6, "emails_std": 3,
            "systems": ["Admin_Console", "Servers", "Email", "JIRA"],
            "data_volume_mean": 1.0, "data_volume_std": 0.8,
        },
        "dba_admin": {
            "clearance": 5, "login_mean": 9.0, "login_std": 0.3,
            "logout_mean": 18.0, "logout_std": 0.5,
            "actions_mean": 25, "actions_std": 8,
            "records_mean": 15, "records_std": 8,
            "emails_mean": 5, "emails_std": 2,
            "systems": ["DB_Console", "Staging_DB", "Email", "JIRA"],
            "data_volume_mean": 2.0, "data_volume_std": 1.5,
        },
        "help_desk": {
            "clearance": 2, "login_mean": 9.0, "login_std": 0.2,
            "logout_mean": 17.5, "logout_std": 0.3,
            "actions_mean": 40, "actions_std": 10,
            "records_mean": 20, "records_std": 8,
            "emails_mean": 12, "emails_std": 4,
            "systems": ["Ticketing", "AD_Console", "Email"],
            "data_volume_mean": 0.5, "data_volume_std": 0.3,
        },
    },
    "hr": {
        "hr_generalist": {
            "clearance": 2, "login_mean": 9.25, "login_std": 0.2,
            "logout_mean": 18.0, "logout_std": 0.3,
            "actions_mean": 35, "actions_std": 10,
            "records_mean": 25, "records_std": 8,
            "emails_mean": 10, "emails_std": 4,
            "systems": ["HRMS", "Email", "Documents"],
            "data_volume_mean": 1.0, "data_volume_std": 0.5,
        },
        "recruiter": {
            "clearance": 2, "login_mean": 9.5, "login_std": 0.3,
            "logout_mean": 18.0, "logout_std": 0.4,
            "actions_mean": 30, "actions_std": 8,
            "records_mean": 15, "records_std": 5,
            "emails_mean": 15, "emails_std": 5,
            "systems": ["HRMS", "Email", "ATS"],
            "data_volume_mean": 0.8, "data_volume_std": 0.4,
        },
        "payroll": {
            "clearance": 3, "login_mean": 9.0, "login_std": 0.15,
            "logout_mean": 17.5, "logout_std": 0.2,
            "actions_mean": 25, "actions_std": 6,
            "records_mean": 30, "records_std": 10,
            "emails_mean": 5, "emails_std": 2,
            "systems": ["Payroll_System", "HRMS", "Email"],
            "data_volume_mean": 1.5, "data_volume_std": 0.8,
        },
    },
    "compliance": {
        "aml_analyst": {
            "clearance": 4, "login_mean": 9.0, "login_std": 0.25,
            "logout_mean": 18.5, "logout_std": 0.5,
            "actions_mean": 40, "actions_std": 12,
            "records_mean": 35, "records_std": 12,
            "emails_mean": 8, "emails_std": 3,
            "systems": ["AML_Platform", "CBS", "Email", "Reports"],
            "data_volume_mean": 3.0, "data_volume_std": 1.5,
        },
        "auditor": {
            "clearance": 3, "login_mean": 9.5, "login_std": 0.3,
            "logout_mean": 18.0, "logout_std": 0.4,
            "actions_mean": 30, "actions_std": 8,
            "records_mean": 20, "records_std": 8,
            "emails_mean": 6, "emails_std": 3,
            "systems": ["Audit_System", "CBS", "Email"],
            "data_volume_mean": 2.0, "data_volume_std": 1.0,
        },
        "risk_officer": {
            "clearance": 4, "login_mean": 9.0, "login_std": 0.2,
            "logout_mean": 18.5, "logout_std": 0.5,
            "actions_mean": 35, "actions_std": 10,
            "records_mean": 25, "records_std": 8,
            "emails_mean": 10, "emails_std": 4,
            "systems": ["Risk_Platform", "CBS", "Email", "Reports"],
            "data_volume_mean": 2.5, "data_volume_std": 1.2,
        },
    },
}

# Action types per system (Markov chain states)
SYSTEM_ACTIONS = {
    "CRM": ["view_record", "update_record", "search_customer", "generate_report"],
    "CBS": ["account_lookup", "transaction_view", "balance_check", "statement_gen"],
    "Email": ["read_email", "send_email", "send_email_external", "download_attachment"],
    "Treasury_Platform": ["portfolio_view", "trade_execute", "risk_calc", "report_gen"],
    "Bloomberg": ["market_data", "news_check", "analysis_run"],
    "Admin_Console": ["config_change", "user_management", "log_review", "system_check"],
    "Servers": ["ssh_login", "file_access", "service_restart", "log_check"],
    "DB_Console": ["query_run", "schema_check", "backup_verify", "perf_monitor"],
    "Staging_DB": ["query_run", "test_data", "schema_change"],
    "HRMS": ["employee_lookup", "record_update", "leave_approval", "report_gen"],
    "AML_Platform": ["alert_review", "case_create", "sar_filing", "rule_config"],
    "Audit_System": ["audit_trail", "compliance_check", "report_gen"],
    "Risk_Platform": ["risk_assessment", "dashboard_view", "report_gen", "model_review"],
    "Ticketing": ["ticket_create", "ticket_resolve", "ticket_update", "knowledge_base"],
    "AD_Console": ["password_reset", "account_unlock", "group_modify"],
    "Reports": ["view_report", "generate_report", "export_report"],
    "Documents": ["upload_doc", "download_doc", "review_doc"],
    "Teller_Terminal": ["cash_deposit", "cash_withdrawal", "check_processing", "balance_inquiry"],
    "Payroll_System": ["payroll_run", "salary_view", "deduction_update", "tax_calc"],
    "ATS": ["resume_review", "interview_schedule", "offer_create"],
    "JIRA": ["ticket_create", "ticket_update", "sprint_review"],
    # Threat-related systems (normally not accessed by most roles)
    "Production_CBS": ["direct_query", "data_export", "schema_modify"],
    "Customer_Records_DB": ["bulk_read", "export_csv", "search_pii"],
    "Treasury_DB": ["portfolio_data", "trade_history", "position_report"],
    "Audit_Logs": ["log_read", "log_export", "log_modify"],
}

# ═══════════════════════════════════════════════════════════════
#  INSIDER THREAT SCENARIOS
# ═══════════════════════════════════════════════════════════════

THREAT_SCENARIOS = {
    "data_exfiltration": {
        "description": "Bulk customer data download via USB after hours",
        "eligible_roles": ["relationship_manager", "teller", "aml_analyst"],
        "ramp_up_days": 14,
        "attack_days": 5,
        "perturbations": {
            "after_hours_prob": 0.8,
            "records_multiplier_ramp": [1.5, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0],
            "data_volume_multiplier": 12.0,
            "usb_connect": True,
            "new_systems": ["Customer_Records_DB"],
            "file_copy_prob": 0.9,
        },
    },
    "privilege_escalation": {
        "description": "IT admin using superadmin credentials on production systems",
        "eligible_roles": ["help_desk", "system_admin", "dba_admin"],
        "ramp_up_days": 7,
        "attack_days": 3,
        "perturbations": {
            "after_hours_prob": 0.6,
            "new_systems": ["Production_CBS", "Audit_Logs"],
            "priv_escalation": True,
            "admin_account_creation": True,
            "records_multiplier_ramp": [1.0, 1.5, 2.0, 5.0],
        },
    },
    "pre_resignation_theft": {
        "description": "Employee downloading data before leaving the organization",
        "eligible_roles": ["relationship_manager", "treasury_analyst", "aml_analyst", "trader"],
        "ramp_up_days": 21,
        "attack_days": 7,
        "perturbations": {
            "job_search_prob": 0.4,
            "records_multiplier_ramp": [1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
            "email_external_multiplier": 3.0,
            "cloud_upload_prob": 0.5,
            "large_attachment_prob": 0.6,
        },
    },
    "unauthorized_snooping": {
        "description": "HR employee accessing customer financial records without authorization",
        "eligible_roles": ["hr_generalist", "recruiter", "payroll"],
        "ramp_up_days": 21,
        "attack_days": 14,
        "perturbations": {
            "new_systems": ["CBS", "Customer_Records_DB"],
            "records_multiplier_ramp": [1.0, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5],
            "cross_role_access": True,
        },
    },
    "credential_compromise": {
        "description": "Account used from two locations simultaneously (impossible travel)",
        "eligible_roles": ["relationship_manager", "branch_manager", "system_admin"],
        "ramp_up_days": 0,
        "attack_days": 5,
        "perturbations": {
            "geo_anomaly": True,
            "rapid_system_switching": True,
            "new_device_prob": 0.9,
            "records_multiplier_ramp": [3.0, 5.0, 4.0, 3.0, 2.0],
            "login_time_anomaly": True,
        },
    },
    "slow_burn_recon": {
        "description": "Gradually expanding access scope over weeks",
        "eligible_roles": ["system_admin", "dba_admin", "aml_analyst", "risk_officer"],
        "ramp_up_days": 30,
        "attack_days": 30,
        "perturbations": {
            "new_system_per_week": 1,
            "records_multiplier_ramp": [1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5],
            "data_volume_growth": 0.05,
        },
    },
}


# ═══════════════════════════════════════════════════════════════
#  EMPLOYEE GENERATOR
# ═══════════════════════════════════════════════════════════════

def generate_employees(
    num_employees: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate employee profiles across 5 departments."""
    rng = np.random.default_rng(seed)
    random.seed(seed)

    employees = []
    emp_idx = 0

    # Shuffle names
    first_names = INDIAN_FIRST_NAMES.copy()
    last_names = INDIAN_LAST_NAMES.copy()

    dept_config = Config.data.DEPARTMENTS

    for dept_name, dept_info in dept_config.items():
        dept_count = dept_info["count"]
        roles = dept_info["roles"]
        role_profiles = ROLE_PROFILES[dept_name]

        # Distribute employees across roles within department
        role_counts = _distribute_counts(dept_count, len(roles), rng)

        for role, count in zip(roles, role_counts):
            profile = role_profiles[role]
            for _ in range(count):
                emp_idx += 1
                emp_id = f"EMP_{emp_idx:03d}"

                # Name
                fname = random.choice(first_names)
                lname = random.choice(last_names)
                name = f"{fname} {lname}"

                # Tenure (months) — Poisson-ish distribution
                tenure = max(3, int(rng.normal(36, 18)))

                # Hire date
                hire_date = datetime(2025, 6, 15) - timedelta(days=tenure * 30)

                employees.append({
                    "emp_id": emp_id,
                    "name": name,
                    "department": dept_name,
                    "role": role,
                    "clearance_level": profile["clearance"],
                    "branch": random.choice(BRANCHES),
                    "tenure_months": tenure,
                    "hire_date": hire_date.strftime("%Y-%m-%d"),
                    "typical_login_hour": profile["login_mean"],
                    "typical_logout_hour": profile["logout_mean"],
                    "typical_actions_per_day": profile["actions_mean"],
                    "typical_records_per_day": profile["records_mean"],
                    "is_insider": False,
                    "insider_scenario": None,
                    "attack_start_day": None,
                    "attack_end_day": None,
                })

    df = pd.DataFrame(employees)
    logger.info(f"Generated {len(df)} employee profiles across {len(dept_config)} departments")
    return df


def _distribute_counts(total: int, n: int, rng) -> list:
    """Distribute `total` into `n` roughly-equal buckets."""
    base = total // n
    remainder = total % n
    counts = [base] * n
    for i in range(remainder):
        counts[i] += 1
    rng.shuffle(counts)
    return counts


# ═══════════════════════════════════════════════════════════════
#  INSIDER SELECTION
# ═══════════════════════════════════════════════════════════════

def select_insiders(
    employees_df: pd.DataFrame,
    insider_ratio: float = 0.08,
    num_days: int = 90,
    seed: int = 42,
) -> pd.DataFrame:
    """Select insider employees and assign threat scenarios."""
    rng = np.random.default_rng(seed)
    random.seed(seed)

    num_insiders = max(1, int(len(employees_df) * insider_ratio))
    scenarios = list(THREAT_SCENARIOS.keys())
    df = employees_df.copy()

    assigned = 0
    scenario_idx = 0

    # Ensure we cover all 6 scenarios
    for scenario_name in scenarios:
        scenario = THREAT_SCENARIOS[scenario_name]
        eligible = df[
            (df["role"].isin(scenario["eligible_roles"])) &
            (~df["is_insider"])
        ]

        if len(eligible) == 0:
            logger.warning(f"No eligible employees for scenario: {scenario_name}")
            continue

        # Pick 2-3 insiders per scenario
        n_pick = min(len(eligible), rng.integers(2, 4))
        if assigned + n_pick > num_insiders:
            n_pick = max(1, num_insiders - assigned)

        chosen_ids = eligible.sample(n=n_pick, random_state=seed + scenario_idx)["emp_id"].tolist()

        for eid in chosen_ids:
            ramp = scenario["ramp_up_days"]
            attack = scenario["attack_days"]
            total_threat = ramp + attack

            # Start the attack in the second half of the observation period
            earliest_start = max(30, num_days - total_threat - 10)
            latest_start = max(earliest_start, num_days - total_threat)
            start_day = rng.integers(earliest_start, latest_start + 1)
            end_day = min(num_days - 1, start_day + total_threat)

            mask = df["emp_id"] == eid
            df.loc[mask, "is_insider"] = True
            df.loc[mask, "insider_scenario"] = scenario_name
            df.loc[mask, "attack_start_day"] = start_day
            df.loc[mask, "attack_end_day"] = end_day

            assigned += 1
            logger.debug(f"  Insider: {eid} → {scenario_name} (days {start_day}-{end_day})")

        scenario_idx += 1

    total_insiders = df["is_insider"].sum()
    logger.info(f"Assigned {total_insiders} insiders across {len(scenarios)} scenarios ({total_insiders/len(df)*100:.1f}%)")
    return df


# ═══════════════════════════════════════════════════════════════
#  ACTIVITY GENERATION (Markov Chain + GMM)
# ═══════════════════════════════════════════════════════════════

def generate_activity_for_employee(
    emp: dict,
    day_index: int,
    base_date: datetime,
    rng: np.random.Generator,
) -> list[dict]:
    """Generate one day of activity for a single employee using Markov chain model."""
    dept = emp["department"]
    role = emp["role"]
    profile = ROLE_PROFILES[dept][role]

    current_date = base_date + timedelta(days=day_index)
    is_weekend = current_date.weekday() >= 5

    # Weekend: 5% chance of working (unless insider on attack day)
    if is_weekend and rng.random() > 0.05:
        return []

    # Sick day / holiday: 3% chance
    if rng.random() < 0.03:
        return []

    # ─── Temporal: Login/Logout times (GMM) ───
    login_hour = max(6.0, rng.normal(profile["login_mean"], profile["login_std"]))
    logout_hour = min(23.0, rng.normal(profile["logout_mean"], profile["logout_std"]))
    if logout_hour <= login_hour:
        logout_hour = login_hour + 1.0

    # ─── Action count (Poisson-ish) ───
    n_actions = max(5, int(rng.normal(profile["actions_mean"], profile["actions_std"])))

    # ─── Generate action sequence (Markov chain) ───
    systems = profile["systems"]
    events = []
    emp_id = emp["emp_id"]
    device_id = f"WS_{emp['branch'][:3].upper()}_{rng.integers(1, 100):03d}"
    ip_base = f"10.{rng.integers(1, 10)}.{rng.integers(1, 50)}"

    for action_idx in range(n_actions):
        # Time within session (uniformly spread)
        progress = action_idx / max(1, n_actions - 1)
        hour = login_hour + progress * (logout_hour - login_hour)
        minute = rng.integers(0, 60)
        second = rng.integers(0, 60)

        h = int(hour)
        m = int((hour - h) * 60)
        ts = current_date.replace(hour=min(23, h), minute=m, second=second)

        # Pick system (weighted toward primary systems)
        weights = [2.0 if i == 0 else 1.0 for i in range(len(systems))]
        weights = np.array(weights) / sum(weights)
        system = rng.choice(systems, p=weights)

        # Pick action within system
        actions_pool = SYSTEM_ACTIONS.get(system, ["generic_action"])
        action = rng.choice(actions_pool)

        # Records accessed (Poisson)
        records = 0
        if action in ["view_record", "account_lookup", "search_customer", "employee_lookup",
                       "alert_review", "bulk_read", "balance_check", "transaction_view"]:
            records = max(0, int(rng.poisson(profile["records_mean"] / max(1, n_actions) * 2)))

        # Data volume
        data_vol = max(0.0, rng.exponential(profile["data_volume_mean"] / max(1, n_actions)))

        events.append({
            "event_id": f"EVT_{emp_id}_{day_index:03d}_{action_idx:04d}",
            "timestamp": ts.isoformat(),
            "emp_id": emp_id,
            "day_index": day_index,
            "action_type": action,
            "system": system,
            "resource": _get_resource(system, action),
            "records_accessed": records,
            "data_volume_mb": round(data_vol, 3),
            "device_id": device_id,
            "ip_address": f"{ip_base}.{rng.integers(1, 255)}",
            "is_after_hours": hour < 9.0 or hour > 18.0,
            "is_new_device": False,
            "is_weekend": is_weekend,
            "geo_location": emp["branch"],
        })

    # Add login/logout events
    login_ts = current_date.replace(hour=int(login_hour), minute=int((login_hour % 1) * 60))
    logout_ts = current_date.replace(hour=min(23, int(logout_hour)), minute=int((logout_hour % 1) * 60))

    events.insert(0, {
        "event_id": f"EVT_{emp_id}_{day_index:03d}_LOGIN",
        "timestamp": login_ts.isoformat(),
        "emp_id": emp_id, "day_index": day_index,
        "action_type": "login", "system": "Auth",
        "resource": None, "records_accessed": 0, "data_volume_mb": 0.0,
        "device_id": device_id, "ip_address": f"{ip_base}.{rng.integers(1, 255)}",
        "is_after_hours": login_hour < 9.0 or login_hour > 18.0,
        "is_new_device": False, "is_weekend": is_weekend,
        "geo_location": emp["branch"],
    })
    events.append({
        "event_id": f"EVT_{emp_id}_{day_index:03d}_LOGOUT",
        "timestamp": logout_ts.isoformat(),
        "emp_id": emp_id, "day_index": day_index,
        "action_type": "logout", "system": "Auth",
        "resource": None, "records_accessed": 0, "data_volume_mb": 0.0,
        "device_id": device_id, "ip_address": f"{ip_base}.{rng.integers(1, 255)}",
        "is_after_hours": logout_hour < 9.0 or logout_hour > 18.0,
        "is_new_device": False, "is_weekend": is_weekend,
        "geo_location": emp["branch"],
    })

    return events


def _get_resource(system: str, action: str) -> str:
    """Map system+action to a resource name."""
    resource_map = {
        "CBS": "customer_records",
        "CRM": "crm_data",
        "Treasury_Platform": "treasury_data",
        "HRMS": "employee_records",
        "AML_Platform": "aml_cases",
        "Admin_Console": "system_config",
        "Production_CBS": "production_data",
        "Customer_Records_DB": "customer_pii",
        "Treasury_DB": "treasury_portfolio",
        "Audit_Logs": "audit_trail",
        "Payroll_System": "payroll_data",
    }
    return resource_map.get(system, f"{system.lower()}_data")


# ═══════════════════════════════════════════════════════════════
#  THREAT INJECTION (Perturbation)
# ═══════════════════════════════════════════════════════════════

def inject_threat(
    events: list[dict],
    emp: dict,
    day_index: int,
    base_date: datetime,
    rng: np.random.Generator,
) -> list[dict]:
    """Inject insider threat perturbations into an employee's daily activity."""
    scenario_name = emp.get("insider_scenario")
    if not scenario_name or not emp.get("is_insider"):
        return events

    start_day = emp.get("attack_start_day", 999)
    end_day = emp.get("attack_end_day", 999)

    if day_index < start_day or day_index > end_day:
        return events

    scenario = THREAT_SCENARIOS[scenario_name]
    perturb = scenario["perturbations"]
    ramp_days = scenario["ramp_up_days"]
    days_into_attack = day_index - start_day

    # Ramp factor: how far into the attack we are (0.0 → 1.0)
    ramp_factor = min(1.0, days_into_attack / max(1, ramp_days))

    # Records multiplier from ramp schedule
    ramp_schedule = perturb.get("records_multiplier_ramp", [1.0])
    schedule_idx = min(days_into_attack, len(ramp_schedule) - 1)
    records_mult = ramp_schedule[schedule_idx]

    dept = emp["department"]
    role = emp["role"]
    profile = ROLE_PROFILES[dept][role]
    current_date = base_date + timedelta(days=day_index)

    modified_events = list(events)

    # ── After-hours access ──
    if perturb.get("after_hours_prob", 0) > 0 and rng.random() < perturb["after_hours_prob"] * ramp_factor:
        late_hour = rng.uniform(20.0, 23.0)
        for evt in modified_events:
            if evt["action_type"] not in ("login", "logout"):
                ts = datetime.fromisoformat(evt["timestamp"])
                new_ts = ts.replace(hour=int(late_hour), minute=rng.integers(0, 60))
                evt["timestamp"] = new_ts.isoformat()
                evt["is_after_hours"] = True

    # ── Inflate records accessed ──
    for evt in modified_events:
        if evt["records_accessed"] > 0:
            evt["records_accessed"] = int(evt["records_accessed"] * records_mult)
        evt["data_volume_mb"] = round(
            evt["data_volume_mb"] * perturb.get("data_volume_multiplier", records_mult),
            3
        )

    # ── New systems (cross-role access) ──
    new_systems = perturb.get("new_systems", [])
    if new_systems and ramp_factor > 0.3:
        for sys_name in new_systems:
            actions_pool = SYSTEM_ACTIONS.get(sys_name, ["access_data"])
            n_extra = max(1, int(3 * ramp_factor))
            for _ in range(n_extra):
                hour = rng.uniform(20.0, 23.5) if perturb.get("after_hours_prob", 0) > 0.5 else rng.uniform(14.0, 18.0)
                ts = current_date.replace(hour=int(hour), minute=rng.integers(0, 60), second=rng.integers(0, 60))
                modified_events.append({
                    "event_id": f"EVT_{emp['emp_id']}_{day_index:03d}_THREAT_{rng.integers(1000, 9999)}",
                    "timestamp": ts.isoformat(),
                    "emp_id": emp["emp_id"],
                    "day_index": day_index,
                    "action_type": rng.choice(actions_pool),
                    "system": sys_name,
                    "resource": _get_resource(sys_name, ""),
                    "records_accessed": int(rng.poisson(50) * records_mult),
                    "data_volume_mb": round(rng.exponential(5.0) * ramp_factor, 3),
                    "device_id": f"WS_{emp['branch'][:3].upper()}_NEW",
                    "ip_address": f"10.99.99.{rng.integers(1, 255)}",
                    "is_after_hours": hour > 18.0 or hour < 9.0,
                    "is_new_device": True,
                    "is_weekend": current_date.weekday() >= 5,
                    "geo_location": emp["branch"],
                })

    # ── USB connect ──
    if perturb.get("usb_connect") and ramp_factor > 0.7:
        ts = current_date.replace(hour=rng.integers(20, 23), minute=rng.integers(0, 60))
        modified_events.append({
            "event_id": f"EVT_{emp['emp_id']}_{day_index:03d}_USB",
            "timestamp": ts.isoformat(),
            "emp_id": emp["emp_id"], "day_index": day_index,
            "action_type": "usb_connect",
            "system": "Device_Manager",
            "resource": "removable_media",
            "records_accessed": 0,
            "data_volume_mb": round(rng.exponential(10.0), 3),
            "device_id": f"USB_{rng.integers(100, 999)}",
            "ip_address": "", "is_after_hours": True,
            "is_new_device": True, "is_weekend": current_date.weekday() >= 5,
            "geo_location": emp["branch"],
        })

    # ── Privilege escalation ──
    if perturb.get("priv_escalation") and ramp_factor > 0.5:
        ts = current_date.replace(hour=rng.integers(21, 23), minute=rng.integers(0, 60))
        modified_events.append({
            "event_id": f"EVT_{emp['emp_id']}_{day_index:03d}_PRIVESC",
            "timestamp": ts.isoformat(),
            "emp_id": emp["emp_id"], "day_index": day_index,
            "action_type": "privilege_escalation",
            "system": "Admin_Console",
            "resource": "superadmin_credentials",
            "records_accessed": 0, "data_volume_mb": 0.0,
            "device_id": f"WS_{emp['branch'][:3].upper()}_NEW",
            "ip_address": f"10.99.99.{rng.integers(1, 255)}",
            "is_after_hours": True, "is_new_device": True,
            "is_weekend": current_date.weekday() >= 5,
            "geo_location": emp["branch"],
        })

    # ── Geo anomaly (credential compromise) ──
    if perturb.get("geo_anomaly") and rng.random() < 0.7:
        anomaly_locations = ["Delhi NCR", "Goa", "Jaipur", "Lucknow"]
        anomaly_loc = rng.choice([l for l in anomaly_locations if l != emp["branch"]])
        for evt in modified_events[:3]:
            evt["geo_location"] = anomaly_loc
            evt["is_new_device"] = True
            evt["ip_address"] = f"103.{rng.integers(1, 255)}.{rng.integers(1, 255)}.{rng.integers(1, 255)}"

    # ── Job search browsing ──
    if perturb.get("job_search_prob", 0) > 0 and rng.random() < perturb["job_search_prob"] * ramp_factor:
        ts = current_date.replace(hour=rng.integers(12, 14), minute=rng.integers(0, 60))
        modified_events.append({
            "event_id": f"EVT_{emp['emp_id']}_{day_index:03d}_JOBSEARCH",
            "timestamp": ts.isoformat(),
            "emp_id": emp["emp_id"], "day_index": day_index,
            "action_type": "web_browse_job_search",
            "system": "Web_Browser",
            "resource": "linkedin_indeed",
            "records_accessed": 0, "data_volume_mb": round(rng.exponential(0.5), 3),
            "device_id": f"WS_{emp['branch'][:3].upper()}",
            "ip_address": f"10.{rng.integers(1,10)}.{rng.integers(1,50)}.{rng.integers(1,255)}",
            "is_after_hours": False, "is_new_device": False,
            "is_weekend": current_date.weekday() >= 5,
            "geo_location": emp["branch"],
        })

    return modified_events


# ═══════════════════════════════════════════════════════════════
#  MAIN GENERATION PIPELINE
# ═══════════════════════════════════════════════════════════════

def generate_dataset(
    num_employees: int = 200,
    num_days: int = 90,
    insider_ratio: float = 0.08,
    seed: int = 42,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate the full synthetic banking dataset.

    Returns:
        (employees_df, activity_df, ground_truth_df)
    """
    if output_dir is None:
        output_dir = Config.paths.SYNTHETIC_DATA

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    base_date = datetime(2025, 3, 15)  # Start date for simulation

    # Step 1: Generate employees
    logger.info("Step 1/4: Generating employee profiles...")
    employees_df = generate_employees(num_employees, seed)

    # Step 2: Select insiders
    logger.info("Step 2/4: Selecting insider threat actors...")
    employees_df = select_insiders(employees_df, insider_ratio, num_days, seed)

    # Step 3: Generate activity logs
    logger.info("Step 3/4: Generating daily activity logs...")
    all_events = []

    for _, emp in tqdm(employees_df.iterrows(), total=len(employees_df), desc="Employees"):
        emp_dict = emp.to_dict()
        emp_rng = np.random.default_rng(seed + hash(emp["emp_id"]) % 10000)

        for day in range(num_days):
            # Generate normal activity
            events = generate_activity_for_employee(emp_dict, day, base_date, emp_rng)

            # Inject threat perturbations if insider
            if emp_dict.get("is_insider"):
                events = inject_threat(events, emp_dict, day, base_date, emp_rng)

            all_events.extend(events)

    activity_df = pd.DataFrame(all_events)
    activity_df = activity_df.sort_values("timestamp").reset_index(drop=True)

    # Step 4: Build ground truth
    logger.info("Step 4/4: Building ground truth labels...")
    insiders = employees_df[employees_df["is_insider"]]
    ground_truth = []
    for _, emp in insiders.iterrows():
        ground_truth.append({
            "emp_id": emp["emp_id"],
            "name": emp["name"],
            "department": emp["department"],
            "role": emp["role"],
            "is_insider": True,
            "scenario": emp["insider_scenario"],
            "attack_start_day": emp["attack_start_day"],
            "attack_end_day": emp["attack_end_day"],
            "description": THREAT_SCENARIOS[emp["insider_scenario"]]["description"],
        })
    ground_truth_df = pd.DataFrame(ground_truth)

    # ─── Save to disk ───
    employees_path = output_dir / "employees.csv"
    activity_path = output_dir / "activity_log.csv"
    truth_path = output_dir / "ground_truth.csv"

    employees_df.to_csv(employees_path, index=False)
    activity_df.to_csv(activity_path, index=False)
    ground_truth_df.to_csv(truth_path, index=False)

    # ─── Summary ───
    logger.success(f"✅ Dataset generated successfully!")
    logger.info(f"   Employees:  {len(employees_df)} ({insiders.shape[0]} insiders)")
    logger.info(f"   Activities: {len(activity_df):,} events")
    logger.info(f"   Duration:   {num_days} days")
    logger.info(f"   Scenarios:  {ground_truth_df['scenario'].nunique()} types")
    logger.info(f"   Saved to:   {output_dir}")

    # Per-scenario breakdown
    if len(ground_truth_df) > 0:
        logger.info("   Scenario breakdown:")
        for scenario, count in ground_truth_df["scenario"].value_counts().items():
            logger.info(f"     • {scenario}: {count} insider(s)")

    return employees_df, activity_df, ground_truth_df


# ═══════════════════════════════════════════════════════════════
#  VALIDATION
# ═══════════════════════════════════════════════════════════════

def validate_dataset(output_dir: Path | None = None):
    """Validate generated dataset for correctness."""
    if output_dir is None:
        output_dir = Config.paths.SYNTHETIC_DATA

    output_dir = Path(output_dir)
    logger.info("🔍 Validating dataset...")

    employees = pd.read_csv(output_dir / "employees.csv")
    activity = pd.read_csv(output_dir / "activity_log.csv")
    ground_truth = pd.read_csv(output_dir / "ground_truth.csv")

    checks = []

    # Check 1: Employee count
    checks.append(("Employee count > 0", len(employees) > 0))

    # Check 2: Activity count
    checks.append(("Activity events > 0", len(activity) > 0))

    # Check 3: All employees have activities
    emp_with_activity = activity["emp_id"].nunique()
    checks.append(("All employees active", emp_with_activity >= len(employees) * 0.95))

    # Check 4: Insider ratio
    insider_ratio = employees["is_insider"].mean()
    checks.append((f"Insider ratio ~8% (actual: {insider_ratio:.1%})", 0.04 < insider_ratio < 0.15))

    # Check 5: All 6 scenarios present
    n_scenarios = ground_truth["scenario"].nunique()
    checks.append((f"All 6 scenarios present ({n_scenarios})", n_scenarios == 6))

    # Check 6: No NaN in critical columns
    critical_cols = ["emp_id", "timestamp", "action_type", "system"]
    no_nulls = activity[critical_cols].notna().all().all()
    checks.append(("No NaN in critical columns", no_nulls))

    # Check 7: Timestamps are parseable
    try:
        pd.to_datetime(activity["timestamp"].iloc[:100])
        checks.append(("Timestamps parseable", True))
    except Exception:
        checks.append(("Timestamps parseable", False))

    # Check 8: Ground truth matches employees
    truth_ids = set(ground_truth["emp_id"])
    insider_ids = set(employees[employees["is_insider"]]["emp_id"])
    checks.append(("Ground truth matches insider employees", truth_ids == insider_ids))

    # Report
    logger.info("─" * 50)
    all_pass = True
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        logger.info(f"  {status} {check_name}")
        if not passed:
            all_pass = False

    logger.info("─" * 50)
    if all_pass:
        logger.success("✅ All validation checks passed!")
    else:
        logger.error("❌ Some validation checks failed!")

    # Stats
    logger.info(f"\n📊 Dataset Statistics:")
    logger.info(f"   Employees: {len(employees)}")
    logger.info(f"   Total events: {len(activity):,}")
    logger.info(f"   Events/employee/day (avg): {len(activity) / (len(employees) * activity['day_index'].nunique()):.1f}")
    logger.info(f"   Insiders: {len(ground_truth)}")
    logger.info(f"   Date range: {activity['timestamp'].min()} → {activity['timestamp'].max()}")

    return all_pass


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Argus AI — Synthetic Banking Data Generator")
    parser.add_argument("--employees", type=int, default=200, help="Number of employees")
    parser.add_argument("--days", type=int, default=90, help="Number of days")
    parser.add_argument("--insider-ratio", type=float, default=0.08, help="Insider ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--validate", action="store_true", help="Validate existing dataset")
    args = parser.parse_args()

    Config.paths.ensure_dirs()

    if args.validate:
        output_dir = Path(args.output) if args.output else None
        validate_dataset(output_dir)
    else:
        output_dir = Path(args.output) if args.output else None
        generate_dataset(
            num_employees=args.employees,
            num_days=args.days,
            insider_ratio=args.insider_ratio,
            seed=args.seed,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
