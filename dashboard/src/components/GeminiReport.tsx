'use client';

import { useState, useCallback } from 'react';
import { Brain, Sparkles, FileText, Shield, MessageSquare, Loader2, RefreshCw, Copy, Check, ChevronDown, ChevronUp } from 'lucide-react';

// ─── Types ───

interface GeminiReportProps {
  employeeData: Record<string, unknown>;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  shapData: any;
  alertData: Record<string, unknown> | null;
}

// ─── Markdown Renderer (lightweight) ───

function RenderMarkdown({ text }: { text: string }) {
  const lines = text.split('\n');
  const elements: React.JSX.Element[] = [];

  lines.forEach((line, i) => {
    const trimmed = line.trim();

    if (trimmed.startsWith('### ')) {
      elements.push(<h4 key={i} style={{ fontSize: 13, fontWeight: 700, color: '#e2e8f0', marginTop: 16, marginBottom: 6 }}>{trimmed.slice(4)}</h4>);
    } else if (trimmed.startsWith('## ')) {
      elements.push(<h3 key={i} style={{ fontSize: 14, fontWeight: 700, color: '#06b6d4', marginTop: 20, marginBottom: 8 }}>{trimmed.slice(3)}</h3>);
    } else if (trimmed.startsWith('# ')) {
      elements.push(<h2 key={i} style={{ fontSize: 16, fontWeight: 800, color: '#f1f5f9', marginTop: 12, marginBottom: 10 }}>{trimmed.slice(2)}</h2>);
    } else if (trimmed.startsWith('- **') || trimmed.startsWith('* **')) {
      const content = trimmed.slice(2);
      elements.push(
        <div key={i} style={{ display: 'flex', gap: 8, marginBottom: 4, paddingLeft: 8 }}>
          <span style={{ color: '#06b6d4', flexShrink: 0 }}>•</span>
          <span style={{ fontSize: 12.5, lineHeight: 1.6, color: '#cbd5e1' }}
            dangerouslySetInnerHTML={{ __html: content.replace(/\*\*(.*?)\*\*/g, '<strong style="color:#e2e8f0">$1</strong>') }} />
        </div>
      );
    } else if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
      elements.push(
        <div key={i} style={{ display: 'flex', gap: 8, marginBottom: 3, paddingLeft: 8 }}>
          <span style={{ color: '#64748b', flexShrink: 0 }}>•</span>
          <span style={{ fontSize: 12.5, lineHeight: 1.6, color: '#94a3b8' }}
            dangerouslySetInnerHTML={{ __html: trimmed.slice(2).replace(/\*\*(.*?)\*\*/g, '<strong style="color:#e2e8f0">$1</strong>') }} />
        </div>
      );
    } else if (/^\d+\.\s/.test(trimmed)) {
      const num = trimmed.match(/^(\d+)\.\s/)?.[1];
      const content = trimmed.replace(/^\d+\.\s/, '');
      elements.push(
        <div key={i} style={{ display: 'flex', gap: 8, marginBottom: 4, paddingLeft: 4 }}>
          <span style={{ color: '#8b5cf6', fontWeight: 700, fontSize: 12, minWidth: 18 }}>{num}.</span>
          <span style={{ fontSize: 12.5, lineHeight: 1.6, color: '#cbd5e1' }}
            dangerouslySetInnerHTML={{ __html: content.replace(/\*\*(.*?)\*\*/g, '<strong style="color:#e2e8f0">$1</strong>') }} />
        </div>
      );
    } else if (trimmed === '') {
      elements.push(<div key={i} style={{ height: 6 }} />);
    } else {
      elements.push(
        <p key={i} style={{ fontSize: 12.5, lineHeight: 1.7, color: '#94a3b8', marginBottom: 4 }}
          dangerouslySetInnerHTML={{ __html: trimmed.replace(/\*\*(.*?)\*\*/g, '<strong style="color:#e2e8f0">$1</strong>') }} />
      );
    }
  });

  return <>{elements}</>;
}

// ─── Tab Button ───

function TabButton({ active, onClick, icon: Icon, label }: {
  active: boolean; onClick: () => void; icon: React.ComponentType<{ size: number }>; label: string
}) {
  return (
    <button
      onClick={onClick}
      style={{
        display: 'flex', alignItems: 'center', gap: 6,
        padding: '6px 12px', borderRadius: 'var(--radius-md)',
        border: `1px solid ${active ? 'rgba(139,92,246,0.4)' : 'var(--border-subtle)'}`,
        background: active ? 'rgba(139,92,246,0.1)' : 'transparent',
        color: active ? '#a78bfa' : '#64748b',
        fontSize: 11, fontWeight: 600, cursor: 'pointer',
        transition: 'all 0.2s',
        fontFamily: 'Inter, sans-serif',
      }}
    >
      <Icon size={13} />
      {label}
    </button>
  );
}

// ─── Main Component ───

export default function GeminiReport({ employeeData, shapData, alertData }: GeminiReportProps) {
  const [activeTab, setActiveTab] = useState<'report' | 'recommendations' | 'chat'>('report');
  const [report, setReport] = useState<string>('');
  const [recommendations, setRecommendations] = useState<string>('');
  const [chatMessages, setChatMessages] = useState<Array<{ role: 'user' | 'ai'; text: string }>>([]);
  const [chatInput, setChatInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [expanded, setExpanded] = useState(true);
  const [error, setError] = useState<string>('');

  const callGemini = useCallback(async (type: string, extra?: Record<string, unknown>) => {
    setLoading(true);
    setError('');
    try {
      const res = await fetch('/api/gemini', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type,
          employeeData,
          shapData,
          alertData,
          ...extra,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'API error');
      return data.result as string;
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to generate';
      setError(msg);
      return '';
    } finally {
      setLoading(false);
    }
  }, [employeeData, shapData, alertData]);

  const generateReport = useCallback(async () => {
    const result = await callGemini('threat_report');
    if (result) setReport(result);
  }, [callGemini]);

  const generateRecommendations = useCallback(async () => {
    const result = await callGemini('recommendation');
    if (result) setRecommendations(result);
  }, [callGemini]);

  const sendChat = useCallback(async () => {
    if (!chatInput.trim()) return;
    const userMsg = chatInput.trim();
    setChatInput('');
    setChatMessages(prev => [...prev, { role: 'user', text: userMsg }]);

    const context = `You are Argus AI, an insider threat detection assistant for Bank of Baroda.
You're analyzing employee ${employeeData.name} (${employeeData.role} in ${employeeData.department}).
Trust score: ${employeeData.trust_score}/100, Risk score: ${employeeData.risk_score}/100.
${shapData ? `Model prediction: ${((shapData.prediction as number) * 100).toFixed(1)}% insider probability.` : ''}
${alertData ? `Alert severity: ${alertData.severity}, Kill chain: ${alertData.matched_chain}` : ''}

Answer the analyst's question concisely and professionally. If they ask about features, explain in plain English what each SHAP feature means for banking context.

Analyst question: ${userMsg}`;

    const result = await callGemini('chat', { message: context });
    if (result) {
      setChatMessages(prev => [...prev, { role: 'ai', text: result }]);
    }
  }, [chatInput, callGemini, employeeData, shapData, alertData]);

  const copyToClipboard = useCallback(() => {
    const text = activeTab === 'report' ? report : recommendations;
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [activeTab, report, recommendations]);

  const empName = (employeeData.name as string) || 'Employee';

  return (
    <div className="card mt-24" style={{ border: '1px solid rgba(139,92,246,0.15)' }}>
      <div className="card-header" style={{ cursor: 'pointer' }} onClick={() => setExpanded(!expanded)}>
        <div className="card-title" style={{ gap: 8 }}>
          <div style={{
            width: 28, height: 28, borderRadius: 'var(--radius-md)',
            background: 'linear-gradient(135deg, #8b5cf6 0%, #06b6d4 100%)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <Sparkles size={14} color="white" />
          </div>
          <span>Gemini AI Analysis</span>
          <span style={{
            fontSize: 9, fontWeight: 700, padding: '2px 6px',
            borderRadius: 'var(--radius-sm)',
            background: 'rgba(139,92,246,0.15)', color: '#a78bfa',
            letterSpacing: '0.05em',
          }}>
            FLASH LITE
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {error && <span className="text-xs" style={{ color: '#ef4444' }}>Error</span>}
          {expanded ? <ChevronUp size={16} color="#64748b" /> : <ChevronDown size={16} color="#64748b" />}
        </div>
      </div>

      {expanded && (
        <div className="card-body">
          {/* Tabs */}
          <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
            <TabButton active={activeTab === 'report'} onClick={() => setActiveTab('report')} icon={FileText} label="Threat Report" />
            <TabButton active={activeTab === 'recommendations'} onClick={() => setActiveTab('recommendations')} icon={Shield} label="Recommendations" />
            <TabButton active={activeTab === 'chat'} onClick={() => setActiveTab('chat')} icon={MessageSquare} label="Ask AI" />
          </div>

          {/* Report Tab */}
          {activeTab === 'report' && (
            <div>
              {!report ? (
                <div style={{ textAlign: 'center', padding: '28px 20px' }}>
                  <Brain size={32} style={{ color: '#8b5cf6', opacity: 0.5, margin: '0 auto 12px' }} />
                  <div className="text-sm" style={{ color: '#94a3b8', marginBottom: 14 }}>
                    Generate an AI-powered threat assessment for <strong style={{ color: '#e2e8f0' }}>{empName}</strong>
                  </div>
                  <button
                    onClick={generateReport}
                    disabled={loading}
                    style={{
                      display: 'inline-flex', alignItems: 'center', gap: 8,
                      padding: '10px 20px', borderRadius: 'var(--radius-md)',
                      background: 'linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)',
                      color: 'white', border: 'none', fontSize: 13, fontWeight: 600,
                      cursor: loading ? 'wait' : 'pointer', opacity: loading ? 0.7 : 1,
                      fontFamily: 'Inter, sans-serif',
                    }}
                  >
                    {loading ? <Loader2 size={14} className="spin" /> : <Sparkles size={14} />}
                    {loading ? 'Analyzing with Gemini...' : 'Generate Threat Report'}
                  </button>
                </div>
              ) : (
                <div>
                  <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginBottom: 12 }}>
                    <button onClick={generateReport} disabled={loading} style={{
                      display: 'flex', alignItems: 'center', gap: 4,
                      padding: '4px 10px', borderRadius: 'var(--radius-sm)',
                      border: '1px solid var(--border-subtle)', background: 'transparent',
                      color: '#64748b', fontSize: 11, cursor: 'pointer', fontFamily: 'Inter, sans-serif',
                    }}>
                      <RefreshCw size={11} /> Regenerate
                    </button>
                    <button onClick={copyToClipboard} style={{
                      display: 'flex', alignItems: 'center', gap: 4,
                      padding: '4px 10px', borderRadius: 'var(--radius-sm)',
                      border: '1px solid var(--border-subtle)', background: 'transparent',
                      color: '#64748b', fontSize: 11, cursor: 'pointer', fontFamily: 'Inter, sans-serif',
                    }}>
                      {copied ? <Check size={11} color="#22c55e" /> : <Copy size={11} />}
                      {copied ? 'Copied!' : 'Copy'}
                    </button>
                  </div>
                  <div style={{
                    padding: '16px 20px', borderRadius: 'var(--radius-md)',
                    background: 'rgba(15,23,42,0.4)', border: '1px solid var(--border-subtle)',
                    maxHeight: 500, overflowY: 'auto',
                  }}>
                    <RenderMarkdown text={report} />
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Recommendations Tab */}
          {activeTab === 'recommendations' && (
            <div>
              {!recommendations ? (
                <div style={{ textAlign: 'center', padding: '28px 20px' }}>
                  <Shield size={32} style={{ color: '#06b6d4', opacity: 0.5, margin: '0 auto 12px' }} />
                  <div className="text-sm" style={{ color: '#94a3b8', marginBottom: 14 }}>
                    Get AI-powered response recommendations for <strong style={{ color: '#e2e8f0' }}>{empName}</strong>
                  </div>
                  <button
                    onClick={generateRecommendations}
                    disabled={loading}
                    style={{
                      display: 'inline-flex', alignItems: 'center', gap: 8,
                      padding: '10px 20px', borderRadius: 'var(--radius-md)',
                      background: 'linear-gradient(135deg, #06b6d4 0%, #0891b2 100%)',
                      color: 'white', border: 'none', fontSize: 13, fontWeight: 600,
                      cursor: loading ? 'wait' : 'pointer', opacity: loading ? 0.7 : 1,
                      fontFamily: 'Inter, sans-serif',
                    }}
                  >
                    {loading ? <Loader2 size={14} className="spin" /> : <Shield size={14} />}
                    {loading ? 'Generating...' : 'Generate Recommendations'}
                  </button>
                </div>
              ) : (
                <div>
                  <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginBottom: 12 }}>
                    <button onClick={generateRecommendations} disabled={loading} style={{
                      display: 'flex', alignItems: 'center', gap: 4,
                      padding: '4px 10px', borderRadius: 'var(--radius-sm)',
                      border: '1px solid var(--border-subtle)', background: 'transparent',
                      color: '#64748b', fontSize: 11, cursor: 'pointer', fontFamily: 'Inter, sans-serif',
                    }}>
                      <RefreshCw size={11} /> Regenerate
                    </button>
                    <button onClick={copyToClipboard} style={{
                      display: 'flex', alignItems: 'center', gap: 4,
                      padding: '4px 10px', borderRadius: 'var(--radius-sm)',
                      border: '1px solid var(--border-subtle)', background: 'transparent',
                      color: '#64748b', fontSize: 11, cursor: 'pointer', fontFamily: 'Inter, sans-serif',
                    }}>
                      {copied ? <Check size={11} color="#22c55e" /> : <Copy size={11} />}
                      {copied ? 'Copied!' : 'Copy'}
                    </button>
                  </div>
                  <div style={{
                    padding: '16px 20px', borderRadius: 'var(--radius-md)',
                    background: 'rgba(15,23,42,0.4)', border: '1px solid var(--border-subtle)',
                    maxHeight: 500, overflowY: 'auto',
                  }}>
                    <RenderMarkdown text={recommendations} />
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Chat Tab */}
          {activeTab === 'chat' && (
            <div>
              {/* Chat Messages */}
              <div style={{
                minHeight: 200, maxHeight: 400, overflowY: 'auto',
                padding: '12px 16px', borderRadius: 'var(--radius-md)',
                background: 'rgba(15,23,42,0.4)', border: '1px solid var(--border-subtle)',
                marginBottom: 12,
              }}>
                {chatMessages.length === 0 ? (
                  <div style={{ padding: '40px 20px', textAlign: 'center' }}>
                    <MessageSquare size={28} style={{ color: '#8b5cf6', opacity: 0.4, margin: '0 auto 12px' }} />
                    <div className="text-sm" style={{ color: '#64748b', marginBottom: 8 }}>
                      Ask Argus AI about this employee
                    </div>
                    <div className="text-xs" style={{ color: '#475569' }}>
                      Try: &quot;Why was this person flagged?&quot; or &quot;What does clearance_normalized mean?&quot;
                    </div>
                  </div>
                ) : (
                  chatMessages.map((msg, i) => (
                    <div key={i} style={{
                      display: 'flex', gap: 10, marginBottom: 14,
                      flexDirection: msg.role === 'user' ? 'row-reverse' : 'row',
                    }}>
                      <div style={{
                        width: 26, height: 26, borderRadius: 'var(--radius-full)', flexShrink: 0,
                        background: msg.role === 'user' ? '#1e293b' : 'linear-gradient(135deg, #8b5cf6, #06b6d4)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: 11, color: 'white', fontWeight: 700,
                      }}>
                        {msg.role === 'user' ? 'You' : 'AI'}
                      </div>
                      <div style={{
                        padding: '10px 14px', borderRadius: 'var(--radius-md)',
                        background: msg.role === 'user' ? 'rgba(30,41,59,0.8)' : 'rgba(139,92,246,0.06)',
                        border: `1px solid ${msg.role === 'user' ? 'var(--border-subtle)' : 'rgba(139,92,246,0.15)'}`,
                        maxWidth: '85%',
                      }}>
                        {msg.role === 'ai' ? (
                          <RenderMarkdown text={msg.text} />
                        ) : (
                          <span style={{ fontSize: 12.5, color: '#cbd5e1' }}>{msg.text}</span>
                        )}
                      </div>
                    </div>
                  ))
                )}
                {loading && (
                  <div style={{ display: 'flex', gap: 10, marginBottom: 14 }}>
                    <div style={{
                      width: 26, height: 26, borderRadius: 'var(--radius-full)',
                      background: 'linear-gradient(135deg, #8b5cf6, #06b6d4)',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                    }}>
                      <Loader2 size={12} color="white" className="spin" />
                    </div>
                    <div style={{ padding: '10px 14px', color: '#64748b', fontSize: 12 }}>
                      Thinking...
                    </div>
                  </div>
                )}
              </div>

              {/* Chat Input */}
              <div style={{ display: 'flex', gap: 8 }}>
                <input
                  type="text"
                  value={chatInput}
                  onChange={e => setChatInput(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && !loading && sendChat()}
                  placeholder="Ask about this employee's behavior..."
                  disabled={loading}
                  style={{
                    flex: 1, padding: '10px 14px', borderRadius: 'var(--radius-md)',
                    border: '1px solid var(--border-subtle)',
                    background: 'rgba(15,23,42,0.6)', color: '#e2e8f0',
                    fontSize: 13, outline: 'none', fontFamily: 'Inter, sans-serif',
                  }}
                />
                <button
                  onClick={sendChat}
                  disabled={loading || !chatInput.trim()}
                  style={{
                    padding: '10px 16px', borderRadius: 'var(--radius-md)',
                    background: chatInput.trim() ? 'linear-gradient(135deg, #8b5cf6, #7c3aed)' : 'rgba(30,41,59,0.5)',
                    color: 'white', border: 'none', fontSize: 12, fontWeight: 600,
                    cursor: loading || !chatInput.trim() ? 'not-allowed' : 'pointer',
                    fontFamily: 'Inter, sans-serif',
                  }}
                >
                  Send
                </button>
              </div>
            </div>
          )}

          {/* Error */}
          {error && (
            <div style={{
              marginTop: 12, padding: '8px 14px', borderRadius: 'var(--radius-sm)',
              background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)',
              fontSize: 12, color: '#ef4444',
            }}>
              {error}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
