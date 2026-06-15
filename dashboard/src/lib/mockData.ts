/**
 * Argus AI — Mock Data
 * Fallback data matching the synthetic generator output.
 * Aligned with ground_truth.csv (14 insiders) and 200 employees
 * across 5 departments, 14 roles, 8 Indian branches.
 *
 * Used when the FastAPI backend is not running.
 */

export type TrustLevel = 'CRITICAL' | 'HIGH_RISK' | 'MEDIUM_RISK' | 'LOW_RISK' | 'TRUSTED';

export interface Employee {
  id: string;
  name: string;
  department: string;
  role: string;
  branch: string;
  clearanceLevel: number;
  tenureMonths: number;
  trustScore: number;
  previousTrustScore: number;
  trustLevel: TrustLevel;
  avatarColor: string;
  isInsider: boolean;
  lastActive: string;
  twinDrift: number;
}

export interface RiskFactor {
  factor: string;
  detail: string;
  impact: number;
  icon: string;
}

export interface Alert {
  id: string;
  employeeId: string;
  employeeName: string;
  department: string;
  trustScore: number;
  previousTrustScore: number;
  trustLevel: TrustLevel;
  timestamp: string;
  riskFactors: RiskFactor[];
  intentChain: {
    pattern: string;
    confidence: number;
    matchedSteps: string[];
  } | null;
  status: 'active' | 'investigating' | 'resolved' | 'dismissed';
  severity: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
}

export interface ActivityEvent {
  id: string;
  timestamp: string;
  employeeId: string;
  employeeName: string;
  actionType: string;
  system: string;
  detail: string;
  riskContribution: number;
  icon: string;
}

export interface TwinProfile {
  expectedLogin: string;
  actualLogin: string;
  expectedSystems: string[];
  actualSystems: string[];
  expectedRecords: number;
  actualRecords: number;
  expectedDataVolume: number;
  actualDataVolume: number;
  expectedDevices: number;
  actualDevices: number;
  dimensions: {
    label: string;
    expected: number;
    actual: number;
  }[];
}

export interface PrivilegeDecayPoint {
  time: string;
  trustScore: number;
  event: string | null;
}

export interface ModelMetrics {
  f1: number;
  precision: number;
  recall: number;
  aucRoc: number;
  falsePositiveRate: number;
  alertsToday: number;
  employeesMonitored: number;
  threatsDetected: number;
}

// ─── Color Helpers ───────────────────────────────────────────────

const AVATAR_COLORS = [
  '#06b6d4', '#8b5cf6', '#f59e0b', '#10b981', '#ef4444',
  '#ec4899', '#3b82f6', '#14b8a6', '#f97316', '#6366f1',
];

function getTrustLevel(score: number): TrustLevel {
  if (score < 20) return 'CRITICAL';
  if (score < 40) return 'HIGH_RISK';
  if (score < 60) return 'MEDIUM_RISK';
  if (score < 80) return 'LOW_RISK';
  return 'TRUSTED';
}

// ═══════════════════════════════════════════════════════════════
//  EMPLOYEES — Matches synthetic_generator.py output
//  200 employees: 60 retail, 25 treasury, 35 IT, 30 HR, 50 compliance
//  14 ground-truth insiders across 6 attack scenarios
// ═══════════════════════════════════════════════════════════════

export const employees: Employee[] = [
  // ─── INSIDERS (14 — from ground_truth.csv) ───
  // Data Exfiltration (2)
  { id: 'EMP_001', name: 'Varun Patel',     department: 'Retail Banking', role: 'Relationship Manager', branch: 'Mumbai Main',     clearanceLevel: 3, tenureMonths: 42, trustScore: 18, previousTrustScore: 91, trustLevel: 'CRITICAL', avatarColor: AVATAR_COLORS[0], isInsider: true, lastActive: 'Now',       twinDrift: 0.91 },
  { id: 'EMP_006', name: 'Anjali Ghosh',    department: 'Retail Banking', role: 'Relationship Manager', branch: 'Chennai South',   clearanceLevel: 3, tenureMonths: 38, trustScore: 22, previousTrustScore: 89, trustLevel: 'HIGH_RISK', avatarColor: AVATAR_COLORS[5], isInsider: true, lastActive: '5 min ago',  twinDrift: 0.87 },
  // Credential Compromise (3)
  { id: 'EMP_005', name: 'Ananya Reddy',    department: 'Retail Banking', role: 'Relationship Manager', branch: 'Hyderabad',       clearanceLevel: 3, tenureMonths: 30, trustScore: 28, previousTrustScore: 93, trustLevel: 'HIGH_RISK', avatarColor: AVATAR_COLORS[4], isInsider: true, lastActive: '2 min ago',  twinDrift: 0.82 },
  { id: 'EMP_010', name: 'Bhavna Saxena',   department: 'Retail Banking', role: 'Relationship Manager', branch: 'Delhi NCR',       clearanceLevel: 3, tenureMonths: 24, trustScore: 31, previousTrustScore: 90, trustLevel: 'HIGH_RISK', avatarColor: AVATAR_COLORS[9], isInsider: true, lastActive: '8 min ago',  twinDrift: 0.78 },
  { id: 'EMP_049', name: 'Simran Tiwari',   department: 'Retail Banking', role: 'Branch Manager',       branch: 'Pune Central',    clearanceLevel: 4, tenureMonths: 60, trustScore: 35, previousTrustScore: 92, trustLevel: 'HIGH_RISK', avatarColor: AVATAR_COLORS[6], isInsider: true, lastActive: '12 min ago', twinDrift: 0.74 },
  // Pre-Resignation Theft (2)
  { id: 'EMP_009', name: 'Aarav Iyer',      department: 'Retail Banking', role: 'Relationship Manager', branch: 'Bangalore Tech',  clearanceLevel: 3, tenureMonths: 36, trustScore: 38, previousTrustScore: 88, trustLevel: 'HIGH_RISK', avatarColor: AVATAR_COLORS[8], isInsider: true, lastActive: '15 min ago', twinDrift: 0.68 },
  { id: 'EMP_164', name: 'Aditya Dutta',    department: 'Compliance',     role: 'AML Analyst',          branch: 'Kolkata Main',    clearanceLevel: 4, tenureMonths: 48, trustScore: 42, previousTrustScore: 94, trustLevel: 'MEDIUM_RISK', avatarColor: AVATAR_COLORS[1], isInsider: true, lastActive: '20 min ago', twinDrift: 0.62 },
  // Privilege Escalation (2)
  { id: 'EMP_095', name: 'Gaurav Menon',    department: 'IT Admin',       role: 'System Admin',         branch: 'Bangalore Tech',  clearanceLevel: 5, tenureMonths: 54, trustScore: 15, previousTrustScore: 86, trustLevel: 'CRITICAL', avatarColor: AVATAR_COLORS[2], isInsider: true, lastActive: '1 min ago',  twinDrift: 0.94 },
  { id: 'EMP_111', name: 'Amit Reddy',      department: 'IT Admin',       role: 'Help Desk',            branch: 'Hyderabad',       clearanceLevel: 2, tenureMonths: 8,  trustScore: 19, previousTrustScore: 78, trustLevel: 'CRITICAL', avatarColor: AVATAR_COLORS[3], isInsider: true, lastActive: '3 min ago',  twinDrift: 0.92 },
  // Unauthorized Snooping (2)
  { id: 'EMP_137', name: 'Tarun Chauhan',   department: 'HR',             role: 'Recruiter',            branch: 'Delhi NCR',       clearanceLevel: 2, tenureMonths: 18, trustScore: 45, previousTrustScore: 84, trustLevel: 'MEDIUM_RISK', avatarColor: AVATAR_COLORS[7], isInsider: true, lastActive: '25 min ago', twinDrift: 0.58 },
  { id: 'EMP_148', name: 'Aditya Trivedi',  department: 'HR',             role: 'Payroll',              branch: 'Ahmedabad',       clearanceLevel: 3, tenureMonths: 22, trustScore: 48, previousTrustScore: 82, trustLevel: 'MEDIUM_RISK', avatarColor: AVATAR_COLORS[0], isInsider: true, lastActive: '30 min ago', twinDrift: 0.55 },
  // Slow-Burn Recon (3)
  { id: 'EMP_160', name: 'Sunil Shah',      department: 'Compliance',     role: 'AML Analyst',          branch: 'Ahmedabad',       clearanceLevel: 4, tenureMonths: 44, trustScore: 52, previousTrustScore: 95, trustLevel: 'MEDIUM_RISK', avatarColor: AVATAR_COLORS[4], isInsider: true, lastActive: '35 min ago', twinDrift: 0.50 },
  { id: 'EMP_185', name: 'Vijay Agarwal',   department: 'Compliance',     role: 'Risk Officer',         branch: 'Mumbai Main',     clearanceLevel: 4, tenureMonths: 50, trustScore: 56, previousTrustScore: 96, trustLevel: 'MEDIUM_RISK', avatarColor: AVATAR_COLORS[1], isInsider: true, lastActive: '40 min ago', twinDrift: 0.46 },
  { id: 'EMP_191', name: 'Karthik Pillai',  department: 'Compliance',     role: 'Risk Officer',         branch: 'Chennai South',   clearanceLevel: 4, tenureMonths: 52, trustScore: 58, previousTrustScore: 97, trustLevel: 'MEDIUM_RISK', avatarColor: AVATAR_COLORS[2], isInsider: true, lastActive: '45 min ago', twinDrift: 0.44 },

  // ─── NORMAL EMPLOYEES (representative sample) ───
  // Retail Banking
  { id: 'EMP_002', name: 'Priya Sharma',    department: 'Retail Banking', role: 'Relationship Manager', branch: 'Delhi NCR',       clearanceLevel: 3, tenureMonths: 48, trustScore: 94, previousTrustScore: 95, trustLevel: 'TRUSTED', avatarColor: AVATAR_COLORS[1], isInsider: false, lastActive: '5 min ago',  twinDrift: 0.03 },
  { id: 'EMP_003', name: 'Vikram Singh',    department: 'Retail Banking', role: 'Teller',               branch: 'Bangalore Tech',  clearanceLevel: 1, tenureMonths: 15, trustScore: 89, previousTrustScore: 90, trustLevel: 'TRUSTED', avatarColor: AVATAR_COLORS[2], isInsider: false, lastActive: '1 min ago',  twinDrift: 0.04 },
  { id: 'EMP_004', name: 'Kavya Reddy',     department: 'Retail Banking', role: 'Teller',               branch: 'Chennai South',   clearanceLevel: 1, tenureMonths: 12, trustScore: 87, previousTrustScore: 89, trustLevel: 'TRUSTED', avatarColor: AVATAR_COLORS[3], isInsider: false, lastActive: '10 min ago', twinDrift: 0.04 },
  { id: 'EMP_007', name: 'Rohan Gupta',     department: 'Retail Banking', role: 'Branch Manager',       branch: 'Pune Central',    clearanceLevel: 4, tenureMonths: 72, trustScore: 93, previousTrustScore: 94, trustLevel: 'TRUSTED', avatarColor: AVATAR_COLORS[6], isInsider: false, lastActive: '6 min ago',  twinDrift: 0.02 },
  // Treasury
  { id: 'EMP_061', name: 'Neha Verma',      department: 'Treasury',       role: 'Trader',               branch: 'Mumbai Main',     clearanceLevel: 4, tenureMonths: 36, trustScore: 91, previousTrustScore: 92, trustLevel: 'TRUSTED', avatarColor: AVATAR_COLORS[0], isInsider: false, lastActive: '4 min ago',  twinDrift: 0.03 },
  { id: 'EMP_062', name: 'Rahul Menon',     department: 'Treasury',       role: 'Treasury Analyst',     branch: 'Mumbai Main',     clearanceLevel: 4, tenureMonths: 32, trustScore: 88, previousTrustScore: 89, trustLevel: 'TRUSTED', avatarColor: AVATAR_COLORS[5], isInsider: false, lastActive: '14 min ago', twinDrift: 0.04 },
  // IT Admin
  { id: 'EMP_086', name: 'Sanjay Tiwari',   department: 'IT Admin',       role: 'System Admin',         branch: 'Delhi NCR',       clearanceLevel: 5, tenureMonths: 56, trustScore: 85, previousTrustScore: 87, trustLevel: 'TRUSTED', avatarColor: AVATAR_COLORS[9], isInsider: false, lastActive: '7 min ago',  twinDrift: 0.05 },
  { id: 'EMP_097', name: 'Manish Das',      department: 'IT Admin',       role: 'DBA Admin',            branch: 'Bangalore Tech',  clearanceLevel: 5, tenureMonths: 40, trustScore: 83, previousTrustScore: 84, trustLevel: 'TRUSTED', avatarColor: AVATAR_COLORS[8], isInsider: false, lastActive: '9 min ago',  twinDrift: 0.05 },
  // HR
  { id: 'EMP_121', name: 'Deepa Krishnan',  department: 'HR',             role: 'HR Generalist',        branch: 'Mumbai Main',     clearanceLevel: 2, tenureMonths: 24, trustScore: 92, previousTrustScore: 91, trustLevel: 'TRUSTED', avatarColor: AVATAR_COLORS[3], isInsider: false, lastActive: '11 min ago', twinDrift: 0.01 },
  { id: 'EMP_131', name: 'Divya Agarwal',   department: 'HR',             role: 'Recruiter',            branch: 'Pune Central',    clearanceLevel: 2, tenureMonths: 20, trustScore: 90, previousTrustScore: 91, trustLevel: 'TRUSTED', avatarColor: AVATAR_COLORS[6], isInsider: false, lastActive: '16 min ago', twinDrift: 0.02 },
  // Compliance
  { id: 'EMP_151', name: 'Pooja Nanda',     department: 'Compliance',     role: 'AML Analyst',          branch: 'Bangalore Tech',  clearanceLevel: 4, tenureMonths: 44, trustScore: 96, previousTrustScore: 95, trustLevel: 'TRUSTED', avatarColor: AVATAR_COLORS[4], isInsider: false, lastActive: '18 min ago', twinDrift: 0.01 },
  { id: 'EMP_170', name: 'Sneha Bose',      department: 'Compliance',     role: 'Auditor',              branch: 'Delhi NCR',       clearanceLevel: 3, tenureMonths: 40, trustScore: 93, previousTrustScore: 94, trustLevel: 'TRUSTED', avatarColor: AVATAR_COLORS[7], isInsider: false, lastActive: '22 min ago', twinDrift: 0.02 },
  { id: 'EMP_180', name: 'Meera Iyer',      department: 'Compliance',     role: 'Risk Officer',         branch: 'Kolkata Main',    clearanceLevel: 4, tenureMonths: 48, trustScore: 97, previousTrustScore: 96, trustLevel: 'TRUSTED', avatarColor: AVATAR_COLORS[1], isInsider: false, lastActive: '13 min ago', twinDrift: 0.01 },
];

// ═══════════════════════════════════════════════════════════════
//  ALERTS — Mapped to 6 attack scenarios from threat injection
// ═══════════════════════════════════════════════════════════════

export const alerts: Alert[] = [
  // CRITICAL: Privilege Escalation — EMP_095 (System Admin)
  {
    id: 'ALT_001',
    employeeId: 'EMP_095',
    employeeName: 'Gaurav Menon',
    department: 'IT Admin',
    trustScore: 15,
    previousTrustScore: 86,
    trustLevel: 'CRITICAL',
    timestamp: '2025-06-15T23:15:00+05:30',
    severity: 'CRITICAL',
    status: 'active',
    riskFactors: [
      { factor: 'Privilege escalation', detail: 'Used superadmin credentials on Production_CBS — bypassed role scope', impact: -28, icon: '🔓' },
      { factor: 'Audit log tampering', detail: 'Accessed Audit_Logs system — potential evidence modification', impact: -20, icon: '📋' },
      { factor: 'Admin account created', detail: 'Created new admin account on production — persistence mechanism', impact: -15, icon: '👤' },
      { factor: 'After-hours access', detail: '23:15 IST — 4+ hours beyond normal logout (18:30)', impact: -10, icon: '🕐' },
    ],
    intentChain: {
      pattern: 'Privilege Escalation Abuse',
      confidence: 0.94,
      matchedSteps: ['privilege_escalation', 'production_cbs_access', 'audit_log_access', 'admin_account_creation'],
    },
  },

  // CRITICAL: Data Exfiltration — EMP_001 (Relationship Manager)
  {
    id: 'ALT_002',
    employeeId: 'EMP_001',
    employeeName: 'Varun Patel',
    department: 'Retail Banking',
    trustScore: 18,
    previousTrustScore: 91,
    trustLevel: 'CRITICAL',
    timestamp: '2025-06-15T22:47:00+05:30',
    severity: 'CRITICAL',
    status: 'investigating',
    riskFactors: [
      { factor: 'Customer data bulk access', detail: 'Accessed Customer_Records_DB — 15x daily record volume (827 vs baseline 55)', impact: -25, icon: '📊' },
      { factor: 'USB exfiltration', detail: 'USB device connected + file copy — zero USB history in 42 months', impact: -18, icon: '💾' },
      { factor: 'After-hours activity', detail: '22:47 IST — 80% of actions occurred after 18:00', impact: -12, icon: '🕐' },
      { factor: 'Data volume spike', detail: '34.7 MB transferred — 15x personal baseline (2.3 MB/day)', impact: -15, icon: '📦' },
      { factor: 'New workstation', detail: 'WS_MUM_019 — first-time device, not previously assigned', impact: -8, icon: '🖥️' },
    ],
    intentChain: {
      pattern: 'Data Exfiltration',
      confidence: 0.91,
      matchedSteps: ['access_customer_records_db', 'bulk_download', 'usb_connect', 'file_copy_to_usb'],
    },
  },

  // CRITICAL: Privilege Escalation — EMP_111 (Help Desk)
  {
    id: 'ALT_003',
    employeeId: 'EMP_111',
    employeeName: 'Amit Reddy',
    department: 'IT Admin',
    trustScore: 19,
    previousTrustScore: 78,
    trustLevel: 'CRITICAL',
    timestamp: '2025-06-15T21:30:00+05:30',
    severity: 'CRITICAL',
    status: 'active',
    riskFactors: [
      { factor: 'Production system access', detail: 'Direct query on Production_CBS — Help Desk role has clearance 2, requires 5', impact: -25, icon: '🗄️' },
      { factor: 'Data export', detail: 'Exported 8,200 records to CSV — 50x normal for Help Desk role', impact: -20, icon: '📤' },
      { factor: 'Unusual login location', detail: 'VPN from Goa — employee based in Hyderabad (geo_anomaly)', impact: -12, icon: '🌐' },
      { factor: 'After-hours access', detail: '21:30 IST — outside normal hours (09:00-17:30)', impact: -8, icon: '🕐' },
    ],
    intentChain: {
      pattern: 'Privilege Escalation Abuse',
      confidence: 0.88,
      matchedSteps: ['privilege_escalation', 'production_cbs_access', 'bulk_data_export'],
    },
  },

  // HIGH: Data Exfiltration — EMP_006 (Relationship Manager)
  {
    id: 'ALT_004',
    employeeId: 'EMP_006',
    employeeName: 'Anjali Ghosh',
    department: 'Retail Banking',
    trustScore: 22,
    previousTrustScore: 89,
    trustLevel: 'HIGH_RISK',
    timestamp: '2025-06-15T21:15:00+05:30',
    severity: 'HIGH',
    status: 'active',
    riskFactors: [
      { factor: 'Customer database access', detail: 'Accessed Customer_Records_DB — not in Relationship Manager scope', impact: -22, icon: '🏦' },
      { factor: 'USB file copy', detail: 'USB connected + file copy — first USB event in 38-month tenure', impact: -18, icon: '💾' },
      { factor: 'Data volume anomaly', detail: '28.5 MB — 12x daily baseline (2.3 MB)', impact: -15, icon: '📦' },
      { factor: 'After-hours pattern', detail: 'Consistent after-hours access for 5 consecutive days', impact: -10, icon: '🕐' },
    ],
    intentChain: {
      pattern: 'Data Exfiltration',
      confidence: 0.85,
      matchedSteps: ['access_customer_records_db', 'bulk_download', 'usb_connect', 'file_copy_to_usb'],
    },
  },

  // HIGH: Credential Compromise — EMP_005 (impossible travel)
  {
    id: 'ALT_005',
    employeeId: 'EMP_005',
    employeeName: 'Ananya Reddy',
    department: 'Retail Banking',
    trustScore: 28,
    previousTrustScore: 93,
    trustLevel: 'HIGH_RISK',
    timestamp: '2025-06-15T20:45:00+05:30',
    severity: 'HIGH',
    status: 'investigating',
    riskFactors: [
      { factor: 'Impossible travel', detail: 'Login from Hyderabad and Mumbai Main within 30 minutes — geographic anomaly', impact: -22, icon: '🌐' },
      { factor: 'Rapid system switching', detail: '12 systems accessed in 15 minutes — normally 3-4 per hour', impact: -15, icon: '⚡' },
      { factor: 'New device', detail: 'Login from WS_MUM_088 — never used by this employee', impact: -12, icon: '🖥️' },
      { factor: 'Record volume spike', detail: '5x baseline records accessed on attack days', impact: -10, icon: '📊' },
    ],
    intentChain: {
      pattern: 'Credential Compromise',
      confidence: 0.82,
      matchedSteps: ['geo_anomaly', 'rapid_system_switching', 'new_device_login', 'bulk_data_access'],
    },
  },

  // MEDIUM: Pre-Resignation Theft — EMP_009 (Relationship Manager)
  {
    id: 'ALT_006',
    employeeId: 'EMP_009',
    employeeName: 'Aarav Iyer',
    department: 'Retail Banking',
    trustScore: 38,
    previousTrustScore: 88,
    trustLevel: 'HIGH_RISK',
    timestamp: '2025-06-15T19:30:00+05:30',
    severity: 'MEDIUM',
    status: 'active',
    riskFactors: [
      { factor: 'Job search activity', detail: 'LinkedIn and Indeed browsing detected — 40% of web activity', impact: -12, icon: '🔍' },
      { factor: 'External email spike', detail: 'External emails 3x baseline — personal attachments to outside domains', impact: -15, icon: '📧' },
      { factor: 'Gradual data escalation', detail: 'Records accessed: 55→68→92→130 over 21 days (ramp-up pattern)', impact: -12, icon: '📈' },
      { factor: 'Cloud upload detected', detail: 'Large attachments uploaded — potential cloud exfiltration', impact: -10, icon: '☁️' },
    ],
    intentChain: {
      pattern: 'Pre-Resignation Theft',
      confidence: 0.72,
      matchedSteps: ['job_search_browsing', 'external_email_spike', 'gradual_data_ramp', 'cloud_upload'],
    },
  },

  // MEDIUM: Unauthorized Snooping — EMP_137 (HR Recruiter)
  {
    id: 'ALT_007',
    employeeId: 'EMP_137',
    employeeName: 'Tarun Chauhan',
    department: 'HR',
    trustScore: 45,
    previousTrustScore: 84,
    trustLevel: 'MEDIUM_RISK',
    timestamp: '2025-06-15T18:45:00+05:30',
    severity: 'MEDIUM',
    status: 'active',
    riskFactors: [
      { factor: 'Cross-role data access', detail: 'Accessed CBS and Customer_Records_DB — HR Recruiter has no authorization', impact: -18, icon: '🏦' },
      { factor: 'Gradual scope expansion', detail: '5 weeks of increasing non-HR access: 2→5→10→18→23 records/day', impact: -12, icon: '📈' },
      { factor: 'Role boundary crossings', detail: 'access_to_role_ratio at 0.35 — 8x department average (0.04)', impact: -10, icon: '🚧' },
    ],
    intentChain: {
      pattern: 'Unauthorized Snooping',
      confidence: 0.67,
      matchedSteps: ['cross_role_data_access', 'gradual_scope_expansion', 'customer_record_browsing'],
    },
  },

  // MEDIUM: Slow-Burn Recon — EMP_160 (AML Analyst)
  {
    id: 'ALT_008',
    employeeId: 'EMP_160',
    employeeName: 'Sunil Shah',
    department: 'Compliance',
    trustScore: 52,
    previousTrustScore: 95,
    trustLevel: 'MEDIUM_RISK',
    timestamp: '2025-06-15T17:30:00+05:30',
    severity: 'MEDIUM',
    status: 'investigating',
    riskFactors: [
      { factor: 'System scope expansion', detail: 'Accessed 3 new systems in past 30 days — normally stable at 4', impact: -12, icon: '🔗' },
      { factor: 'Gradual data growth', detail: 'Data volume increased 5% daily over 30 days — now 4.5x baseline', impact: -10, icon: '📈' },
      { factor: 'Record access trend', detail: 'Records: 35→38→42→48→55 (subtle escalation over 8 weeks)', impact: -8, icon: '📊' },
    ],
    intentChain: {
      pattern: 'Slow-Burn Reconnaissance',
      confidence: 0.58,
      matchedSteps: ['new_system_access_per_week', 'gradual_data_volume_increase'],
    },
  },
];

// ═══════════════════════════════════════════════════════════════
//  LIVE ACTIVITY FEED — Recent events from flagged employees
// ═══════════════════════════════════════════════════════════════

export const activityFeed: ActivityEvent[] = [
  { id: 'ACT_001', timestamp: '23:15:42', employeeId: 'EMP_095', employeeName: 'Gaurav Menon',  actionType: 'privilege_escalation', system: 'Production_CBS',      detail: 'Superadmin credentials used on production — clearance 5 role but unauthorized scope', riskContribution: 95, icon: '🔓' },
  { id: 'ACT_002', timestamp: '22:51:18', employeeId: 'EMP_095', employeeName: 'Gaurav Menon',  actionType: 'audit_log_access',     system: 'Audit_Logs',           detail: 'Queried audit trail — potential evidence tampering',         riskContribution: 88, icon: '📋' },
  { id: 'ACT_003', timestamp: '22:47:33', employeeId: 'EMP_001', employeeName: 'Varun Patel',   actionType: 'usb_connect',          system: 'WS_MUM_019',           detail: 'USB device connected — zero USB history in 42 months',       riskContribution: 82, icon: '💾' },
  { id: 'ACT_004', timestamp: '22:45:18', employeeId: 'EMP_001', employeeName: 'Varun Patel',   actionType: 'bulk_download',        system: 'Customer_Records_DB',   detail: '827 customer records accessed in 12 minutes',                riskContribution: 90, icon: '📊' },
  { id: 'ACT_005', timestamp: '22:30:05', employeeId: 'EMP_111', employeeName: 'Amit Reddy',    actionType: 'data_export',          system: 'Production_CBS',        detail: 'Exported 8,200 records to CSV via VPN from Goa',            riskContribution: 85, icon: '📤' },
  { id: 'ACT_006', timestamp: '21:22:19', employeeId: 'EMP_006', employeeName: 'Anjali Ghosh',  actionType: 'file_copy',            system: 'Customer_Records_DB',   detail: 'Copied customer PII data to USB — first USB event ever',     riskContribution: 80, icon: '💾' },
  { id: 'ACT_007', timestamp: '20:45:50', employeeId: 'EMP_005', employeeName: 'Ananya Reddy',  actionType: 'geo_anomaly',          system: 'CBS',                   detail: 'Simultaneous login from Hyderabad and Mumbai Main',         riskContribution: 78, icon: '🌐' },
  { id: 'ACT_008', timestamp: '19:30:12', employeeId: 'EMP_009', employeeName: 'Aarav Iyer',    actionType: 'cloud_upload',         system: 'Email',                 detail: 'Large attachment sent to external email — cloud exfiltration pattern', riskContribution: 65, icon: '☁️' },
  { id: 'ACT_009', timestamp: '18:45:33', employeeId: 'EMP_137', employeeName: 'Tarun Chauhan', actionType: 'cross_role_access',    system: 'CBS',                   detail: 'Accessed customer financial records — HR scope violation',   riskContribution: 58, icon: '🏦' },
  { id: 'ACT_010', timestamp: '17:30:28', employeeId: 'EMP_160', employeeName: 'Sunil Shah',    actionType: 'new_system_access',    system: 'Treasury_Platform',     detail: 'First-time access to Treasury — expanding recon scope',     riskContribution: 42, icon: '🔗' },
  { id: 'ACT_011', timestamp: '17:15:44', employeeId: 'EMP_002', employeeName: 'Priya Sharma',  actionType: 'data_access',          system: 'CRM',                   detail: 'Updated 3 customer records — normal pattern for RM',        riskContribution: 2,  icon: '📝' },
  { id: 'ACT_012', timestamp: '16:50:33', employeeId: 'EMP_151', employeeName: 'Pooja Nanda',   actionType: 'login',                system: 'AML_Platform',          detail: 'Routine AML case review — within role scope',               riskContribution: 1,  icon: '🔑' },
];

// ═══════════════════════════════════════════════════════════════
//  MODEL METRICS — From enhanced pipeline (211 features, LightGBM+XGBoost)
// ═══════════════════════════════════════════════════════════════

export const modelMetrics: ModelMetrics = {
  f1: 0.992,
  precision: 0.985,
  recall: 1.0,
  aucRoc: 1.0,
  falsePositiveRate: 0.001,
  alertsToday: 8,
  employeesMonitored: 200,
  threatsDetected: 14,
};

// ─── Trust Score History (for sparklines) ────────────────────────
// 12-day trailing window for key employees

export const trustScoreHistory: Record<string, number[]> = {
  // Insiders — decay patterns matching attack scenarios
  'EMP_001': [91, 91, 90, 89, 85, 78, 65, 48, 35, 27, 21, 18],  // Data exfiltration: sharp drop
  'EMP_006': [89, 89, 88, 87, 82, 74, 60, 44, 33, 28, 24, 22],  // Data exfiltration: sharp drop
  'EMP_005': [93, 93, 92, 91, 88, 82, 70, 55, 42, 35, 30, 28],  // Credential compromise: sudden
  'EMP_095': [86, 86, 85, 84, 80, 72, 58, 40, 28, 22, 18, 15],  // Priv escalation: steep
  'EMP_111': [78, 78, 77, 76, 72, 65, 52, 38, 28, 23, 20, 19],  // Priv escalation: steep
  'EMP_009': [88, 88, 87, 85, 82, 78, 72, 64, 56, 48, 42, 38],  // Pre-resignation: gradual
  'EMP_164': [94, 94, 93, 91, 88, 84, 78, 70, 62, 54, 48, 42],  // Pre-resignation: gradual
  'EMP_137': [84, 84, 83, 82, 80, 76, 72, 66, 60, 54, 49, 45],  // Snooping: slow
  'EMP_148': [82, 82, 81, 80, 78, 74, 70, 64, 58, 54, 50, 48],  // Snooping: slow
  'EMP_160': [95, 94, 93, 92, 90, 87, 82, 76, 70, 64, 58, 52],  // Slow-burn: very gradual
  'EMP_185': [96, 95, 94, 93, 91, 88, 84, 78, 72, 66, 60, 56],  // Slow-burn: very gradual
  'EMP_191': [97, 96, 95, 94, 92, 89, 85, 80, 74, 68, 62, 58],  // Slow-burn: very gradual
  // Normal employees — stable
  'EMP_002': [94, 95, 94, 95, 94, 95, 94, 93, 94, 95, 94, 94],
  'EMP_003': [89, 90, 89, 90, 89, 89, 90, 89, 90, 89, 89, 89],
  'EMP_086': [87, 86, 85, 86, 87, 85, 86, 85, 86, 85, 85, 85],
};

// ─── Privilege Decay Timeline (for EMP_001 detail page) ─────────
// Models the data_exfiltration attack day progression

export const privilegeDecayTimeline: PrivilegeDecayPoint[] = [
  { time: '09:15', trustScore: 91, event: 'Login (normal — 09:15 IST)' },
  { time: '09:30', trustScore: 93, event: 'CRM access (routine RM work)' },
  { time: '10:30', trustScore: 92, event: null },
  { time: '12:00', trustScore: 91, event: null },
  { time: '14:00', trustScore: 90, event: null },
  { time: '17:00', trustScore: 89, event: 'End of normal hours' },
  { time: '18:30', trustScore: 85, event: 'Still logged in (trust decay)' },
  { time: '19:30', trustScore: 78, event: 'After-hours (accelerated decay)' },
  { time: '20:30', trustScore: 65, event: 'Customer_Records_DB access' },
  { time: '21:30', trustScore: 48, event: 'Bulk download (827 records)' },
  { time: '22:00', trustScore: 35, event: 'USB connected (first ever)' },
  { time: '22:30', trustScore: 27, event: 'File copy to USB' },
  { time: '22:47', trustScore: 18, event: 'ALERT: Data Exfiltration' },
];

// ─── Digital Twin Profile (for EMP_001) ──────────────────────────
// Expected (from 42-month behavioral baseline) vs Actual (attack day)

export const sampleTwinProfile: TwinProfile = {
  expectedLogin: '09:15',
  actualLogin: '09:15',
  expectedSystems: ['CRM', 'CBS', 'Email'],
  actualSystems: ['CRM', 'CBS', 'Customer_Records_DB', 'Email'],
  expectedRecords: 55,
  actualRecords: 827,
  expectedDataVolume: 2.3,
  actualDataVolume: 34.7,
  expectedDevices: 1,
  actualDevices: 2,
  dimensions: [
    { label: 'Login Time',   expected: 85, actual: 85 },   // Normal login
    { label: 'Data Volume',  expected: 45, actual: 97 },   // 15× spike
    { label: 'System Scope', expected: 60, actual: 88 },   // Customer_Records_DB
    { label: 'USB Activity', expected: 0,  actual: 95 },   // First ever USB
    { label: 'Email Pattern',expected: 55, actual: 72 },   // Slightly elevated
    { label: 'After Hours',  expected: 5,  actual: 95 },   // 5+ hours past normal
    { label: 'Peer Alignment', expected: 88, actual: 12 }, // Extreme deviation
    { label: 'Role Boundary',  expected: 10, actual: 85 }, // Customer_Records_DB
  ],
};

// ─── Department Stats (aligned with actual counts) ───────────────

export const departmentStats = [
  { name: 'Retail Banking', employees: 60, avgTrust: 82.4, alerts: 4, color: '#06b6d4' },
  { name: 'Treasury',       employees: 25, avgTrust: 91.2, alerts: 0, color: '#8b5cf6' },
  { name: 'IT Admin',       employees: 35, avgTrust: 74.8, alerts: 2, color: '#f59e0b' },
  { name: 'HR',             employees: 30, avgTrust: 78.5, alerts: 2, color: '#10b981' },
  { name: 'Compliance',     employees: 50, avgTrust: 85.6, alerts: 3, color: '#ec4899' },
];

// ═══════════════════════════════════════════════════════════════
//  HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════

export function getEmployee(id: string): Employee | undefined {
  return employees.find(e => e.id === id);
}

export function getAlertsByEmployee(id: string): Alert[] {
  return alerts.filter(a => a.employeeId === id);
}

export function getTrustColor(score: number): string {
  if (score < 20) return '#ef4444';
  if (score < 40) return '#f97316';
  if (score < 60) return '#eab308';
  if (score < 80) return '#22c55e';
  return '#06b6d4';
}

export function getTrustGradient(score: number): string {
  if (score < 20) return 'linear-gradient(135deg, #ef4444, #dc2626)';
  if (score < 40) return 'linear-gradient(135deg, #f97316, #ea580c)';
  if (score < 60) return 'linear-gradient(135deg, #eab308, #ca8a04)';
  if (score < 80) return 'linear-gradient(135deg, #22c55e, #16a34a)';
  return 'linear-gradient(135deg, #06b6d4, #0891b2)';
}
