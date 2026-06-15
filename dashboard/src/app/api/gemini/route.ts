import { GoogleGenAI } from '@google/genai';
import { NextRequest, NextResponse } from 'next/server';

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || '' });

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { type, employeeData, shapData, alertData } = body;

    if (!process.env.GEMINI_API_KEY) {
      return NextResponse.json({ error: 'GEMINI_API_KEY not configured' }, { status: 500 });
    }

    let prompt = '';

    if (type === 'threat_report') {
      prompt = buildThreatReportPrompt(employeeData, shapData, alertData);
    } else if (type === 'risk_summary') {
      prompt = buildRiskSummaryPrompt(alertData);
    } else if (type === 'recommendation') {
      prompt = buildRecommendationPrompt(employeeData, shapData, alertData);
    } else if (type === 'chat') {
      prompt = body.message;
    } else {
      return NextResponse.json({ error: 'Invalid type' }, { status: 400 });
    }

    const response = await ai.models.generateContent({
      model: 'gemini-2.0-flash-lite',
      contents: prompt,
      config: {
        temperature: 0.3,
        maxOutputTokens: 2048,
      },
    });

    const text = response.text || '';

    return NextResponse.json({ result: text });
  } catch (error: unknown) {
    const msg = error instanceof Error ? error.message : 'Unknown error';
    console.error('[Gemini API Error]', msg);
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}

// ─── Prompt Builders ───

function buildThreatReportPrompt(
  employee: Record<string, unknown>,
  shap: Record<string, unknown> | null,
  alert: Record<string, unknown> | null,
): string {
  const shapSection = shap
    ? `
SHAP Explainability:
- Prediction (P(insider)): ${((shap.prediction as number) * 100).toFixed(1)}%
- Base value: ${((shap.base_value as number) * 100).toFixed(1)}%
- Top risk factors: ${JSON.stringify((shap.top_risk_factors as Array<Record<string, unknown>>)?.slice(0, 6))}
- Top protective factors: ${JSON.stringify((shap.top_protective_factors as Array<Record<string, unknown>>)?.slice(0, 4))}
`
    : 'SHAP data not available.';

  const alertSection = alert
    ? `
Alert Details:
- Severity: ${alert.severity}
- Kill chain match: ${alert.matched_chain}
- Chain confidence: ${alert.chain_confidence}%
- Key signals: ${JSON.stringify(alert.chain_signals)}
- Anomalous features: ${JSON.stringify((alert.top_features as Array<Record<string, unknown>>)?.slice(0, 5))}
- Summary: ${alert.summary}
`
    : '';

  return `You are an insider threat analyst AI for an Indian bank (Bank of Baroda). Generate a structured threat assessment report for a security analyst.

EMPLOYEE PROFILE:
- ID: ${employee.emp_id}
- Name: ${employee.name}
- Department: ${employee.department}
- Role: ${employee.role}
- Branch: ${employee.branch}
- Clearance Level: ${employee.clearance_level}/5
- Trust Score: ${employee.trust_score}/100
- Risk Score: ${employee.risk_score}/100

${shapSection}
${alertSection}

Write a clear, professional threat assessment report with these sections:
1. **Executive Summary** (2-3 sentences: who, what risk level, urgency)
2. **Behavioral Analysis** (what specific behaviors were detected, why they're anomalous)
3. **Key Risk Indicators** (bullet points of the top SHAP factors in plain English)
4. **Possible Threat Scenarios** (what attack patterns this could indicate: data exfiltration, privilege abuse, etc.)
5. **Recommended Actions** (specific steps for the SOC analyst: investigate, monitor, escalate)
6. **Confidence Assessment** (how confident is the model, what gaps exist)

Use plain English. Avoid jargon. Be specific about what the employee did differently from peers.
If trust score is above 70, note this is likely a false positive or low-risk anomaly.
Format with markdown headers and bullet points.`;
}

function buildRiskSummaryPrompt(alerts: Record<string, unknown>[]): string {
  const alertSummaries = alerts.slice(0, 10).map((a, i) =>
    `${i + 1}. ${a.name} (${a.department}/${a.role}) — Risk: ${a.risk_score}, Trust: ${a.trust_score}, Chain: ${a.matched_chain}`
  ).join('\n');

  return `You are an insider threat analyst AI for Bank of Baroda. Generate a daily threat landscape briefing.

CURRENT ALERTS (Top ${alerts.length}):
${alertSummaries}

Total flagged employees: ${alerts.length}

Write a concise daily briefing with:
1. **Threat Overview** (1-2 sentences summarizing today's landscape)
2. **Priority Cases** (which employees need immediate attention and why)
3. **Pattern Analysis** (any common themes across flagged employees)
4. **Risk Trend** (is the overall risk increasing, stable, or decreasing)
5. **Recommended Focus** (where the SOC team should focus today)

Keep it under 300 words. Be actionable and specific.`;
}

function buildRecommendationPrompt(
  employee: Record<string, unknown>,
  shap: Record<string, unknown> | null,
  alert: Record<string, unknown> | null,
): string {
  return `You are an insider threat response advisor for Bank of Baroda. Based on the following data, provide specific actionable recommendations.

Employee: ${employee.name} (${employee.role} in ${employee.department})
Trust Score: ${employee.trust_score}/100
Risk Score: ${employee.risk_score}/100
${alert ? `Alert: ${alert.severity} severity, matched ${alert.matched_chain}` : ''}
${shap ? `Model confidence: ${((shap.prediction as number) * 100).toFixed(1)}%` : ''}
${alert ? `Key signals: ${JSON.stringify((alert.chain_signals as string[])?.slice(0, 5))}` : ''}

Provide exactly 5 specific, actionable recommendations in order of priority. For each:
- **Action** (what to do)
- **Rationale** (why this is important)
- **Timeline** (immediate, within 24h, within 1 week)
- **Owner** (SOC analyst, manager, HR, CISO)

Be specific to Indian banking regulations (RBI guidelines) where relevant.
Format as a numbered list with sub-bullets.`;
}
