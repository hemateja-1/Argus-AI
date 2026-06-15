# Argus AI — Security Dashboard

Next.js 16 dashboard for the Argus insider threat detection platform.

## Setup

### 1. Install dependencies

```bash
cd dashboard
npm install
```

### 2. Configure environment

Create a `.env.local` file in the `dashboard/` directory:

```env
# Gemini AI Integration (Google AI Studio)
GEMINI_API_KEY=your_google_ai_studio_api_key_here
```

> **Note**: `.env.local` is gitignored and will NOT be committed. Each developer needs their own key from [Google AI Studio](https://aistudio.google.com/).

### 3. Start the API server (optional, for live data)

```bash
# From project root
python -m argus.api.scoring_api
```

### 4. Run the dashboard

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Features

- **Command Center** — Real-time trust score heatmap for all 200 employees
- **Alerts** — Kill chain analysis with severity classification
- **Employees** — Sortable, searchable employee directory with live risk scores
- **Employee Detail** — Digital twin comparison, SHAP explainability, privilege decay timeline
- **Gemini AI Analysis** — AI-powered threat reports, recommendations, and interactive Q&A
- **Analytics** — Model performance metrics, feature importance, department stats
- **Digital Twin** — Behavioral genome deviation analysis

## Architecture

```
dashboard/
  src/
    app/
      api/gemini/route.ts    — Server-side Gemini API proxy (key stays secret)
      employee/[id]/page.tsx — Employee detail with AI analysis
      alerts/page.tsx        — Alert center
      analytics/page.tsx     — Model analytics
    components/
      GeminiReport.tsx       — AI report/chat component
      Sidebar.tsx            — Navigation
    lib/
      api.ts                 — FastAPI client
      hooks.ts               — React hooks for live data
      mockData.ts            — Fallback mock data
  .env.local                 — API keys (gitignored)
```

## AI Features (Gemini Integration)

The dashboard integrates **Gemini 2.0 Flash Lite** for:

1. **Threat Reports** — AI-generated threat assessment for any employee, explaining SHAP factors in plain English
2. **Response Recommendations** — Prioritized action items with timelines and RBI compliance notes
3. **Interactive Q&A** — Ask questions about any employee's behavior in natural language

All Gemini calls go through a Next.js API route (`/api/gemini`) so the API key stays server-side.
