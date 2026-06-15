'use client';

import { use } from 'react';
import Sidebar from '@/components/Sidebar';
import GeminiReport from '@/components/GeminiReport';
import {
  getEmployee, getAlertsByEmployee, getTrustColor,
  privilegeDecayTimeline, sampleTwinProfile, trustScoreHistory, activityFeed,
} from '@/lib/mockData';
import { useEmployee, useShapExplanation, useApiStatus } from '@/lib/hooks';
import {
  ArrowLeft, Shield, AlertTriangle, Clock, Fingerprint,
  Cpu, TrendingDown, Eye, Zap, Brain,
} from 'lucide-react';
import Link from 'next/link';
import {
  Chart as ChartJS,
  CategoryScale, LinearScale, PointElement, LineElement,
  RadialLinearScale, BarElement, Filler, Tooltip, Legend,
} from 'chart.js';
import { Line, Radar, Bar } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, RadialLinearScale, Filler, Tooltip, Legend);

// ─── SHAP Waterfall Component ───
function ShapWaterfall({ shapData }: { shapData: any }) {
  if (!shapData) return null;

  const riskFactors = shapData.top_risk_factors?.slice(0, 8) || [];
  const protFactors = shapData.top_protective_factors?.slice(0, 4) || [];
  const allFactors = [
    ...riskFactors.map((f: any) => ({ ...f, type: 'risk' })),
    ...protFactors.map((f: any) => ({ ...f, type: 'protect' })),
  ];

  if (allFactors.length === 0) return null;

  const maxVal = Math.max(...allFactors.map((f: any) => Math.abs(f.shap_value)));

  return (
    <div className="flex flex-col gap-8">
      <div className="flex items-center justify-between mb-8">
        <div className="text-xs text-muted">
          Base: {(shapData.base_value * 100).toFixed(1)}% → Prediction: {(shapData.prediction * 100).toFixed(1)}%
        </div>
        <div className="text-xs text-mono" style={{ color: shapData.prediction > 0.5 ? '#ef4444' : '#22c55e' }}>
          P(insider) = {(shapData.prediction * 100).toFixed(1)}%
        </div>
      </div>

      {allFactors.map((f: any, i: number) => {
        const barWidth = Math.min(100, (Math.abs(f.shap_value) / maxVal) * 100);
        const isRisk = f.type === 'risk';
        const color = isRisk ? '#ef4444' : '#22c55e';
        const bgColor = isRisk ? 'rgba(239,68,68,0.08)' : 'rgba(34,197,94,0.08)';
        const featureName = f.feature.replace(/_/g, ' ').replace(/roll /g, '').replace(/zscore /g, 'z:');
        return (
          <div key={i} style={{
            display: 'grid', gridTemplateColumns: '180px 1fr 60px',
            alignItems: 'center', gap: 12, padding: '6px 10px',
            borderRadius: 'var(--radius-md)', background: bgColor,
            border: `1px solid ${color}15`,
          }}>
            <div className="text-xs text-mono" style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {isRisk ? '↑' : '↓'} {featureName}
            </div>
            <div style={{ position: 'relative', height: 14, background: 'rgba(148,163,184,0.06)', borderRadius: 4 }}>
              <div style={{
                position: 'absolute',
                [isRisk ? 'left' : 'right']: 0,
                top: 0, height: '100%',
                width: `${barWidth}%`,
                background: `linear-gradient(${isRisk ? '90deg' : '270deg'}, ${color}80, ${color}20)`,
                borderRadius: 4,
                transition: 'width 0.8s ease',
              }} />
            </div>
            <div className="text-xs text-mono" style={{ color, textAlign: 'right', fontWeight: 700 }}>
              {f.shap_value > 0 ? '+' : ''}{f.shap_value.toFixed(3)}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function EmployeeDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);

  // Live data
  const { live: apiLive } = useApiStatus();
  const { data: liveDetail } = useEmployee(id);
  const { data: shapData, loading: shapLoading } = useShapExplanation(id);

  // Fallback to mock data
  const mockEmp = getEmployee(id);
  const empAlerts = getAlertsByEmployee(id);

  // Use live data if available
  const empName = liveDetail?.employee?.name as string || mockEmp?.name || id;
  const empRole = liveDetail?.employee?.role as string || mockEmp?.role || '';
  const empDept = liveDetail?.employee?.department as string || mockEmp?.department || '';
  const empBranch = liveDetail?.employee?.branch as string || mockEmp?.branch || '';
  const empClearance = (liveDetail?.employee?.clearance_level as number) || mockEmp?.clearanceLevel || 1;
  const trustScore = Math.round(liveDetail?.trust_score ?? mockEmp?.trustScore ?? 95);
  const riskScore = Math.round(liveDetail?.risk_score ?? 5);
  const twinDrift = mockEmp?.twinDrift ?? 0;
  const isHighRisk = trustScore < 40;

  // Trust timeline from API
  const liveTimeline = liveDetail?.trust_timeline || [];

  // Default to sample data for demo
  const twin = sampleTwinProfile;
  const decay = liveTimeline.length > 5
    ? liveTimeline.map(t => ({
        time: `Day ${t.day_index}`,
        trustScore: Math.round(t.trust_score),
        event: t.risk_score > 70 ? `Risk spike: ${Math.round(t.risk_score)}` : null,
      }))
    : privilegeDecayTimeline;

  function getTrustLevel(score: number) {
    if (score < 20) return 'CRITICAL';
    if (score < 40) return 'HIGH_RISK';
    if (score < 60) return 'MEDIUM_RISK';
    if (score < 80) return 'LOW_RISK';
    return 'TRUSTED';
  }

  if (!mockEmp && !liveDetail) {
    return (
      <div className="app-layout">
        <Sidebar />
        <main className="main-content">
          <div className="page-content" style={{ padding: 60, textAlign: 'center' }}>
            <h2>Employee not found</h2>
            <Link href="/employees" style={{ marginTop: 16, display: 'inline-block' }}>← Back to employees</Link>
          </div>
        </main>
      </div>
    );
  }

  const color = getTrustColor(trustScore);
  const trustLevel = getTrustLevel(trustScore);

  // ─── Radar Chart (Twin Comparison) ───
  const radarData = {
    labels: twin.dimensions.map(d => d.label),
    datasets: [
      {
        label: 'Expected (Twin)',
        data: twin.dimensions.map(d => d.expected),
        borderColor: '#06b6d4',
        backgroundColor: 'rgba(6, 182, 212, 0.1)',
        borderWidth: 2,
        pointBackgroundColor: '#06b6d4',
        pointRadius: 3,
      },
      {
        label: 'Actual (Current)',
        data: twin.dimensions.map(d => d.actual),
        borderColor: '#ef4444',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        borderWidth: 2,
        pointBackgroundColor: '#ef4444',
        pointRadius: 3,
      },
    ],
  };

  const radarOptions = {
    responsive: true,
    maintainAspectRatio: true,
    scales: {
      r: {
        beginAtZero: true,
        max: 100,
        ticks: {
          stepSize: 25,
          color: 'rgba(148, 163, 184, 0.3)',
          backdropColor: 'transparent',
          font: { size: 9, family: 'Inter' },
        },
        grid: { color: 'rgba(148, 163, 184, 0.06)' },
        angleLines: { color: 'rgba(148, 163, 184, 0.06)' },
        pointLabels: {
          color: 'rgba(148, 163, 184, 0.7)',
          font: { size: 11, family: 'Inter', weight: 'normal' as const },
        },
      },
    },
    plugins: {
      legend: {
        position: 'bottom' as const,
        labels: { color: '#94a3b8', font: { size: 11, family: 'Inter' }, padding: 16, usePointStyle: true, pointStyleWidth: 8 },
      },
      tooltip: {
        backgroundColor: 'rgba(15, 23, 42, 0.9)',
        titleFont: { family: 'Inter', size: 12 },
        bodyFont: { family: 'Inter', size: 11 },
        borderColor: 'rgba(148, 163, 184, 0.1)',
        borderWidth: 1, padding: 10, cornerRadius: 8,
      },
    },
  };

  // ─── Privilege Decay Chart ───
  const decayData = {
    labels: decay.map(d => d.time),
    datasets: [{
      label: 'Trust Score',
      data: decay.map(d => d.trustScore),
      borderColor: '#06b6d4',
      backgroundColor: (ctx: any) => {
        if (!ctx.chart?.ctx) return 'rgba(6, 182, 212, 0.1)';
        const gradient = ctx.chart.ctx.createLinearGradient(0, 0, 0, ctx.chart.height || 300);
        gradient.addColorStop(0, 'rgba(6, 182, 212, 0.2)');
        gradient.addColorStop(0.5, 'rgba(6, 182, 212, 0.05)');
        gradient.addColorStop(1, 'rgba(239, 68, 68, 0.05)');
        return gradient;
      },
      fill: true,
      tension: 0.35,
      borderWidth: 2.5,
      pointRadius: decay.map(d => d.event ? 5 : 0),
      pointBackgroundColor: decay.map(d => d.trustScore < 30 ? '#ef4444' : d.trustScore < 60 ? '#eab308' : '#06b6d4'),
      pointBorderColor: 'rgba(0,0,0,0.4)',
      pointBorderWidth: 2,
    }],
  };

  const decayOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: { min: 0, max: 100, grid: { color: 'rgba(148, 163, 184, 0.05)' }, ticks: { color: '#64748b', font: { size: 11, family: 'JetBrains Mono' } } },
      x: { grid: { color: 'rgba(148, 163, 184, 0.03)' }, ticks: { color: '#64748b', font: { size: 10, family: 'JetBrains Mono' } } },
    },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: 'rgba(15, 23, 42, 0.95)',
        titleFont: { family: 'Inter', size: 12 },
        bodyFont: { family: 'Inter', size: 11 },
        borderColor: 'rgba(148, 163, 184, 0.15)',
        borderWidth: 1, padding: 12, cornerRadius: 8,
        callbacks: {
          afterBody: (items: any) => {
            const idx = items[0]?.dataIndex;
            const point = decay[idx];
            return point?.event ? `\n📌 ${point.event}` : '';
          },
        },
      },
    },
  };

  return (
    <div className="app-layout">
      <Sidebar />
      <main className="main-content">
        <div className="page-header">
          <Link href="/employees" className="flex items-center gap-4 text-sm text-muted mb-16" style={{ textDecoration: 'none' }}>
            <ArrowLeft size={14} /> Back to Employees
          </Link>
          {apiLive && (
            <span className="text-xs text-mono" style={{ color: '#22c55e', float: 'right', marginTop: -24 }}>
              ● LIVE API
            </span>
          )}
        </div>

        <div className="page-content" style={{ paddingTop: 0 }}>
          {/* Employee Header Card */}
          <div className={`card ${isHighRisk ? 'card-glow-red' : 'card-glow-cyan'}`} style={{ marginBottom: 24 }}>
            <div style={{ padding: '24px 28px', display: 'flex', alignItems: 'center', gap: 24 }}>
              <div className="avatar" style={{ width: 56, height: 56, fontSize: 20, background: mockEmp?.avatarColor || '#06b6d4' }}>
                {empName.split(' ').map((n: string) => n[0]).join('')}
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 20, fontWeight: 700 }}>{empName}</div>
                <div className="text-sm text-muted mt-4">
                  {empRole} • {empDept} • {empBranch} • Clearance {empClearance}/5
                </div>
                <div className="text-xs text-muted mt-4 text-mono">{id} • Risk Score: {riskScore}</div>
              </div>
              <div style={{ textAlign: 'center', minWidth: 100 }}>
                <div style={{ fontSize: 42, fontWeight: 900, color, lineHeight: 1, letterSpacing: '-0.04em' }}>
                  {trustScore}
                </div>
                <div className="text-xs text-muted mt-4">Trust Score</div>
                <div style={{
                  marginTop: 8, padding: '3px 10px', borderRadius: 'var(--radius-full)',
                  background: `${color}15`, color, fontSize: 10, fontWeight: 700,
                  border: `1px solid ${color}30`, textTransform: 'uppercase', letterSpacing: '0.06em',
                }}>
                  {trustLevel.replace('_', ' ')}
                </div>
              </div>
            </div>
          </div>

          {/* Two-Column Layout */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
            {/* Twin Comparison Radar */}
            <div className="card">
              <div className="card-header">
                <div className="card-title">
                  <Fingerprint size={16} style={{ color: 'var(--cyan-500)' }} />
                  Digital Twin Comparison
                </div>
                <span className="text-xs text-muted">Expected vs Actual</span>
              </div>
              <div className="card-body" style={{ display: 'flex', justifyContent: 'center' }}>
                <div style={{ width: '100%', maxWidth: 380 }}>
                  <Radar data={radarData} options={radarOptions} />
                </div>
              </div>
            </div>

            {/* SHAP Explainability — NEW */}
            <div className="card" style={{ border: shapData ? '1px solid rgba(139,92,246,0.2)' : undefined }}>
              <div className="card-header">
                <div className="card-title">
                  <Brain size={16} style={{ color: '#8b5cf6' }} />
                  SHAP Explainability
                </div>
                <span className="text-xs text-mono" style={{ color: '#8b5cf6' }}>
                  {shapData ? 'TreeExplainer' : shapLoading ? 'Loading...' : 'Unavailable'}
                </span>
              </div>
              <div className="card-body">
                {shapData ? (
                  <ShapWaterfall shapData={shapData} />
                ) : shapLoading ? (
                  <div style={{ padding: 40, textAlign: 'center' }}>
                    <div className="text-sm text-muted">Calculating SHAP values...</div>
                  </div>
                ) : (
                  <div style={{ padding: 40, textAlign: 'center' }}>
                    <div className="text-sm text-muted">Start the API server to see SHAP explanations</div>
                    <div className="text-xs text-muted mt-8 text-mono">python -m argus.api.scoring_api</div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Gemini AI Analysis */}
          <GeminiReport
            employeeData={{
              emp_id: id,
              name: empName,
              department: empDept,
              role: empRole,
              branch: empBranch,
              clearance_level: empClearance,
              trust_score: trustScore,
              risk_score: riskScore,
            }}
            shapData={shapData}
            alertData={empAlerts?.[0] ? {
              severity: empAlerts[0].severity,
              matched_chain: empAlerts[0].intentChain?.pattern || '',
              chain_confidence: empAlerts[0].intentChain?.confidence || 0,
              chain_signals: empAlerts[0].intentChain?.matchedSteps || [],
              top_features: empAlerts[0].riskFactors?.map((rf: { factor: string; detail: string; impact: number }) => ({
                feature: rf.factor,
                zscore: rf.impact,
                value: rf.impact,
              })) || [],
              summary: empAlerts[0].riskFactors?.map((rf: { detail: string }) => rf.detail).join('; ') || '',
            } : null}
          />

          {/* Behavioral Genome Deviation */}
          <div className="card mt-24">
            <div className="card-header">
              <div className="card-title">
                <Cpu size={16} style={{ color: 'var(--cyan-500)' }} />
                Behavioral Genome Deviation
              </div>
              <span className="text-xs text-mono" style={{ color: twinDrift > 0.5 ? '#ef4444' : '#22c55e' }}>
                Drift: {twinDrift.toFixed(2)}
              </span>
            </div>
            <div className="card-body">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 12 }}>
                {[
                  { label: 'Login Time', expected: twin.expectedLogin, actual: twin.actualLogin, icon: '🕐' },
                  { label: 'Records Accessed', expected: `${twin.expectedRecords}/day`, actual: `${twin.actualRecords}/day`, icon: '📊' },
                  { label: 'Data Volume', expected: `${twin.expectedDataVolume} MB`, actual: `${twin.actualDataVolume} MB`, icon: '💾' },
                  { label: 'Devices Used', expected: `${twin.expectedDevices}`, actual: `${twin.actualDevices}`, icon: '🖥️' },
                  { label: 'Systems', expected: twin.expectedSystems.join(', '), actual: twin.actualSystems.join(', '), icon: '🔗' },
                ].map((item, i) => (
                  <div key={i} style={{
                    display: 'grid', gridTemplateColumns: '110px 1fr 1fr',
                    gap: 12, padding: '10px 12px', borderRadius: 'var(--radius-md)',
                    background: 'rgba(15,23,42,0.3)', border: '1px solid var(--border-subtle)',
                    alignItems: 'center',
                  }}>
                    <div className="flex items-center gap-8">
                      <span>{item.icon}</span>
                      <span className="text-xs font-semibold">{item.label}</span>
                    </div>
                    <div>
                      <div className="text-xs text-muted">Expected</div>
                      <div className="text-sm text-mono" style={{ color: 'var(--cyan-400)' }}>{item.expected}</div>
                    </div>
                    <div>
                      <div className="text-xs text-muted">Actual</div>
                      <div className="text-sm text-mono" style={{ color: item.expected !== item.actual ? '#ef4444' : 'var(--trust-low)' }}>
                        {item.actual}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Privilege Decay Chart — Full Width */}
          <div className="card mt-24">
            <div className="card-header">
              <div className="card-title">
                <TrendingDown size={16} style={{ color: 'var(--cyan-500)' }} />
                Privilege Decay Timeline
              </div>
              {liveTimeline.length > 0 && (
                <span className="text-xs text-mono" style={{ color: '#22c55e' }}>● Live data ({liveTimeline.length} days)</span>
              )}
            </div>
            <div className="card-body" style={{ height: 280 }}>
              <Line data={decayData} options={decayOptions} />
            </div>
          </div>

          {/* Activity Timeline */}
          <div className="card mt-24">
            <div className="card-header">
              <div className="card-title">
                <Clock size={16} style={{ color: 'var(--cyan-500)' }} />
                Activity Timeline
              </div>
            </div>
            <div className="card-body" style={{ padding: '8px 20px 20px' }}>
              {activityFeed.filter(a => a.employeeId === id || (isHighRisk && a.riskContribution > 60)).slice(0, 8).map((event) => {
                const riskLevel = event.riskContribution > 70 ? 'high' : event.riskContribution > 30 ? 'medium' : 'low';
                return (
                  <div key={event.id} className="feed-item">
                    <span className="feed-time">{event.timestamp}</span>
                    <span className="feed-icon">{event.icon}</span>
                    <div className="feed-content">
                      <span className="feed-employee">{event.employeeName}</span>{' '}
                      <span className="feed-detail">{event.detail}</span>
                    </div>
                    <span className={`feed-risk ${riskLevel}`}>{event.riskContribution}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
