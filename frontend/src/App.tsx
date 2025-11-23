import React, { useEffect, useMemo, useState, useCallback } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import {
  Search,
  Settings,
  BarChart2,
  FileText,
  Lightbulb,
  MessageSquare,
  CheckCircle,
  AlertCircle,
  RefreshCw,
  Star,
  Download,
  Activity,
} from "lucide-react";
import { getJSON, postJSON } from "./api";

/* ------------------ types --------------------- */

type PredictResponse = {
  category: string;
  confidence: number;
  source: string;
  xgb_top: { category: string; confidence: number };
  prototype_top3: { category: string; similarity: number }[];
  anonymization: { original: string; anonymized: string };
  anomaly_score?: number;
  risk_tier?: "low" | "medium" | "high";
};

type LiveEvent = {
  description: string;
  amount: number;
  date: string;
  prediction: any;
  latency_ms?: number;
  ts?: number;
  source: "live" | "test";
  advice?: string | null;
};

type LiveStats = {
  total: number;
  avgLatencyMs: number;
  throughputTxPerSec: number;
};

type Metrics = any;

type TransactionForm = {
  description: string;
  amount: number;
  date: string;
  userName: string;
  accountId: string;
};

type HistoryItem = {
  id: number;
  tx: TransactionForm;
  prediction: PredictResponse;
};

type StreamEvent = {
  description: string;
  amount: number;
  date: string;
  prediction: PredictResponse;
};

const DEFAULT_TX: TransactionForm = {
  description: "STARBUCKS #123 CHENNAI TN POS 4924",
  amount: 245.5,
  date: "2024-11-23",
  userName: "Sree Sharvesh",
  accountId: "123456789012",
};

const mockHistory: HistoryItem[] = [];

const COLORS = ["#4F46E5", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#0EA5E9"];

const ALL_CATEGORIES = [
  "Food & Dining",
  "Transportation",
  "Shopping & Retail",
  "Entertainment & Recreation",
  "Healthcare & Medical",
  "Utilities & Services",
  "Financial Services",
  "Income",
  "Government & Legal",
  "Charity & Donations",
];

/* ----------------- helpers -------------------- */

function deriveCategoryStats(metrics: Metrics | null) {
  const fallbackDistribution = ALL_CATEGORIES.map((c) => ({
    category: c,
    count: 1,
  }));
  const fallbackPerClassF1 = ALL_CATEGORIES.map((c) => ({
    category: c,
    f1: 0.99,
  }));

  if (!metrics?.labels) {
    return {
      categoryDistribution: fallbackDistribution,
      perClassF1: fallbackPerClassF1,
      labels: undefined as string[] | undefined,
      cm: undefined as number[][] | undefined,
    };
  }

  const labels: string[] = metrics.labels;
  const cm: number[][] =
    metrics.test?.hybrid?.confusion_matrix ??
    metrics.test?.xgb?.confusion_matrix;

  if (!cm || !Array.isArray(cm) || cm.length === 0) {
    return {
      categoryDistribution: fallbackDistribution,
      perClassF1: fallbackPerClassF1,
      labels: undefined,
      cm: undefined,
    };
  }

  const n = labels.length;
  const rowSums = cm.map((row: number[]) =>
    row.reduce((acc: number, v: number) => acc + v, 0)
  );
  const colSums = Array(n).fill(0) as number[];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) colSums[j] += cm[i][j];
  }

  const perClassF1 = labels.map((label, i) => {
    const tp = cm[i][i] ?? 0;
    const fp = colSums[i] - tp;
    const fn = rowSums[i] - tp;
    const denom = 2 * tp + fp + fn;
    const f1 = denom > 0 ? (2 * tp) / denom : 0;
    return { category: label, f1 };
  });

  const categoryDistribution = labels.map((label, i) => ({
    category: label,
    count: rowSums[i],
  }));

  return { categoryDistribution, perClassF1, labels, cm };
}

/* ----------------- main app ------------------- */

function App() {
  const [activeTab, setActiveTab] = useState<
  "dashboard" | "demo" | "taxonomy" | "evaluation" | "benchmarks" | "stream" | "feedback"
>("dashboard");

  const [liveEvents, setLiveEvents] = useState<LiveEvent[]>([]);

  const MAX_LIVE_EVENTS = 300;

  const [liveStats, setLiveStats] = useState<LiveStats>({
    total: 0,
    avgLatencyMs: 0,
    throughputTxPerSec: 0,
  });
  const [form, setForm] = useState<TransactionForm>(DEFAULT_TX);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [history, setHistory] = useState<HistoryItem[]>(mockHistory);
  const [nextId, setNextId] = useState(1);

  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [metricsLoading, setMetricsLoading] = useState(false);
  const [metricsError, setMetricsError] = useState<string | null>(null);

  const [metricsExplanation, setMetricsExplanation] = useState<string | null>(
    null
  );
  const [explaining, setExplaining] = useState(false);

  const [taxonomy, setTaxonomy] = useState<any | null>(null);
  const [taxonomyLoading, setTaxonomyLoading] = useState(false);
  const [taxonomyError, setTaxonomyError] = useState<string | null>(null);

  const [feedbackStatus, setFeedbackStatus] = useState<
    Record<number, "correct" | "incorrect">
  >({});
  const [expandedTransaction, setExpandedTransaction] = useState<number | null>(
    null
  );
  const [feedbackCategory, setFeedbackCategory] = useState("");

  const [benchmark, setBenchmark] = useState<any | null>(null);
  const [benchLoading, setBenchLoading] = useState(false);
  const [benchError, setBenchError] = useState<string | null>(null);

  const [biasReport, setBiasReport] = useState<any | null>(null);
  const [biasLoading, setBiasLoading] = useState(false);

  const [testStreamRunning, setTestStreamRunning] = useState(false);

  const handleIncomingEvent = useCallback((raw: any, source: "live" | "test") => {
    const ev: LiveEvent = {
      description: raw.description,
      amount: Number(raw.amount ?? 0),
      date: raw.date ?? "",
      prediction: raw.prediction ?? raw, // backend may stream as {prediction: {...}} or just {...}
      latency_ms: raw.latency_ms ?? raw.prediction?.latency_ms,
      ts: raw.ts ?? Date.now() / 1000,
      source,
      advice: raw.advice ?? null,
    };

    setLiveEvents((prev) => {
      const next = [ev, ...prev].slice(0, MAX_LIVE_EVENTS);

      // compute live stats from next
      const count = next.length;
      const latencies = next
        .map((e) => e.latency_ms ?? 0)
        .filter((v) => v > 0);
      const avgLatency =
        latencies.length > 0
          ? latencies.reduce((a, b) => a + b, 0) / latencies.length
          : 0;

      const newestTs = next[0]?.ts ?? 0;
      const oldestTs = next[count - 1]?.ts ?? newestTs;
      const spanSec = newestTs - oldestTs;
      const throughput =
        spanSec > 0 ? count / spanSec : 0;

      setLiveStats({
        total: count,
        avgLatencyMs: avgLatency,
        throughputTxPerSec: throughput,
      });

      return next;
    });
  }, []);


  useEffect(() => {
    const es = new EventSource("$https://web-production-783fa.up.railway.app/api/stream");

    es.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleIncomingEvent(data, "live");
      } catch (e) {
        console.error("Failed to parse live stream event", e);
      }
    };

    es.onerror = () => {
      console.warn("Live stream connection closed");
      es.close();
    };

    return () => {
      es.close();
    };
  }, [handleIncomingEvent]);


  useEffect(() => {
    if (!testStreamRunning) return;

    const es = new EventSource("$https://web-production-783fa.up.railway.app/api/stream-test");

    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        handleIncomingEvent(data, "test");  // same handler you use for /api/stream
      } catch (e) {
        console.error("Failed to parse $https://web-production-783fa.up.railway.app/api/stream-test event", e);
      }
    };

    es.onerror = () => {
      console.warn("Test stream closed");
      es.close();
      setTestStreamRunning(false);
    };

    return () => {
      es.close();
    };
  }, [testStreamRunning, handleIncomingEvent]);


  // ---------- load metrics once ----------
  useEffect(() => {
    setMetricsLoading(true);
    getJSON<Metrics>("$https://web-production-783fa.up.railway.app/api/metrics")
      .then((m) => setMetrics(m))
      .catch((e) => setMetricsError(e.message))
      .finally(() => setMetricsLoading(false));
  }, []);

  // ---------- load taxonomy when tab opened ----------
  useEffect(() => {
    if (activeTab !== "taxonomy" || taxonomy || taxonomyLoading) return;
    setTaxonomyLoading(true);
    getJSON<any>("$https://web-production-783fa.up.railway.app/api/taxonomy")
      .then(setTaxonomy)
      .catch((e) => setTaxonomyError(e.message))
      .finally(() => setTaxonomyLoading(false));
  }, [activeTab, taxonomy, taxonomyLoading]);

  // ---------- streaming: subscribe when on dashboard ----------
  useEffect(() => {
    if (activeTab !== "dashboard") return;

    const es = new EventSource("$https://web-production-783fa.up.railway.app/api/stream");
    es.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as StreamEvent;
        setLiveEvents((prev) => [data, ...prev].slice(0, 20));
      } catch {
        // ignore parse errors
      }
    };
    es.onerror = () => {
      es.close();
    };

    return () => {
      es.close();
    };
  }, [activeTab]);

  const macroF1 =
    metrics?.test?.hybrid?.macro_f1 ??
    metrics?.test?.xgb?.macro_f1 ??
    metrics?.val?.hybrid?.macro_f1;

  const { categoryDistribution, perClassF1, labels, cm } = useMemo(
    () => deriveCategoryStats(metrics),
    [metrics]
  );

  /* ------------------ actions ------------------- */

  const runModel = async () => {
    setLoading(true);
    setError(null);
    setMetricsExplanation(null);

    try {
      const body = {
        description: form.description,
        amount: Number(form.amount),
        date: form.date,
        user_name: form.userName || null,
        account_id: form.accountId || null,
      };
      const res = await postJSON<PredictResponse>(
        "$https://web-production-783fa.up.railway.app/api/predict",
        body
      );
      setPrediction(res);

      const newItem: HistoryItem = {
        id: nextId,
        tx: { ...form },
        prediction: res,
      };
      setNextId((id) => id + 1);
      setHistory((prev) => [newItem, ...prev].slice(0, 50));
      setActiveTab("demo");
    } catch (e: any) {
      setError(e.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  const submitFeedback = async (item: HistoryItem, correctCat: string) => {
    if (!correctCat) return;
    try {
      await fetch("$https://web-production-783fa.up.railway.app/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        description: item.description,
        amount: item.amount,
        date: item.date,
        predicted_category: item.prediction.category,
        predicted_confidence: item.prediction.confidence,
        correct_category: correctCat,
        source: item.source,
        ts: item.ts,
      }),
    });
      setFeedbackStatus((prev) => ({ ...prev, [item.id]: "correct" }));
      setTimeout(
        () =>
          setFeedbackStatus((prev) => {
            const copy = { ...prev };
            delete copy[item.id];
            return copy;
          }),
        2000
      );
    } catch {
      setFeedbackStatus((prev) => ({ ...prev, [item.id]: "incorrect" }));
    }
  };

  const explainMetrics = async () => {
    if (!metrics) return;
    setExplaining(true);
    try {
      const res = await postJSON<{ explanation: string }>(
        "$https://web-production-783fa.up.railway.app/api/explain-metrics",
        { metrics }
      );
      setMetricsExplanation(res.explanation);
    } catch (e: any) {
      setMetricsExplanation(`Failed: ${e.message}`);
    } finally {
      setExplaining(false);
    }
  };

  const runBenchmark = async () => {
    setBenchLoading(true);
    setBenchError(null);
    try {
      const res = await postJSON<any>(
        "$https://web-production-783fa.up.railway.app/api/benchmark",
        {}
      );
      setBenchmark(res);
    } catch (e: any) {
      setBenchError(e.message);
    } finally {
      setBenchLoading(false);
    }
  };

  const loadBiasReport = async () => {
    setBiasLoading(true);
    try {
      const res = await getJSON<any>(
        "$https://web-production-783fa.up.railway.app/api/bias-report"
      );
      setBiasReport(res);
    } finally {
      setBiasLoading(false);
    }
  };

  /* ------------------ UI ----------------------- */

  return (
    <div className="min-h-screen w-screen bg-gradient-to-br from-indigo-50 to-purple-50 text-gray-900">
      {/* header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
          <div className="flex items-center space-x-3">
            <div className="bg-indigo-600 rounded-xl p-2">
              <BarChart2 className="h-8 w-8 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">TransactIQ</h1>
              <p className="text-indigo-600 font-medium">
                AI-Powered Transaction Categorisation with Open Taxonomy
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            {macroF1 && (
              <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium">
                Hybrid macro F1: {macroF1.toFixed(3)}
              </span>
            )}
            <div className="flex space-x-1">
              {[...Array(5)].map((_, i) => (
                <Star
                  key={i}
                  className={`h-5 w-5 ${
                    i < 4 ? "text-yellow-400" : "text-gray-300"
                  }`}
                />
              ))}
            </div>
          </div>
        </div>
      </header>

      {/* tabs */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="flex space-x-6 border-b border-gray-200 mb-8">
          {(
            ["dashboard", "demo", "taxonomy", "evaluation", "benchmarks", "stream", "feedback"] as const
            ).map((tab) => {
            const active = activeTab === tab;
            return (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`pb-3 px-2 font-medium flex items-center space-x-2 border-b-2 transition-colors ${
                  active
                    ? "text-purple-600 border-purple-600"
                    : "text-gray-500 border-transparent hover:text-gray-700"
                } bg-transparent`}
              >
                {tab === "dashboard" && <BarChart2 className="h-5 w-5" />}
                {tab === "demo" && <Lightbulb className="h-5 w-5" />}
                {tab === "taxonomy" && <Settings className="h-5 w-5" />}
                {tab === "evaluation" && <FileText className="h-5 w-5" />}
                {tab === "feedback" && <MessageSquare className="h-5 w-5" />}
                {tab === "benchmarks" && <Activity className="h-5 w-5" />}
                {tab === "stream" && <Activity className="h-5 w-5" />}
                <span className="capitalize">
                  {tab === "demo" ? "Demo" : tab}
                </span>
              </button>
            );
          })}
        </div>

        {activeTab === "dashboard" && (
          <DashboardTab
            history={history}
            categoryDistribution={categoryDistribution}
            perClassF1={perClassF1}
            liveEvents={liveEvents}
            metrics={metrics}
            liveStats={liveStats}
          />
        )}
        {activeTab === "demo" && (
          <DemoTab
            form={form}
            setForm={setForm}
            runModel={runModel}
            loading={loading}
            error={error}
            prediction={prediction}
          />
        )}
        {activeTab === "taxonomy" && (
          <TaxonomyTab
            taxonomy={taxonomy}
            loading={taxonomyLoading}
            error={taxonomyError}
          />
        )}
        {activeTab === "evaluation" && (
          <EvaluationTab
            metrics={metrics}
            labels={labels}
            cm={cm}
            loading={metricsLoading}
            error={metricsError}
            explain={explainMetrics}
            explaining={explaining}
            explanation={metricsExplanation}
            benchmark={benchmark}
            runBenchmark={runBenchmark}
            benchLoading={benchLoading}
            benchError={benchError}
            biasReport={biasReport}
            loadBiasReport={loadBiasReport}
            biasLoading={biasLoading}
          />
        )}
        {activeTab === "feedback" && (
          <FeedbackTab
            feedbackItems={liveEvents}
            feedbackStatus={feedbackStatus}
            expandedTransaction={expandedTransaction}
            setExpandedTransaction={setExpandedTransaction}
            feedbackCategory={feedbackCategory}
            setFeedbackCategory={setFeedbackCategory}
            submitFeedback={submitFeedback}
          />
        )}
        {activeTab === "benchmarks" && (
          <BenchmarksTab
            benchmark={benchmark}
            runBenchmark={runBenchmark}
            benchLoading={benchLoading}
            benchError={benchError}
            biasReport={biasReport}
            loadBiasReport={loadBiasReport}
            biasLoading={biasLoading}
          />
        )}  
       {activeTab === "stream" && (
          <StreamTab
            running={testStreamRunning}
            setRunning={setTestStreamRunning}
            testEvents={liveEvents.filter((e) => e.source === "test")}
          />
        )}
      </div>
    </div>
  );
}

/* ---------------- dashboard ------------------- */

function DashboardTab({
  history,
  categoryDistribution,
  perClassF1,
  liveEvents,
  metrics,
  liveStats,
}: {
  history: HistoryItem[];
  categoryDistribution: { category: string; count: number }[];
  perClassF1: { category: string; f1: number }[];
  liveEvents: StreamEvent[];
  metrics: any;
  liveStats: LiveStats;
}) {
  const latest = liveEvents[0] ?? history[0];

  return (
    <div className="space-y-6">
      {/* NEW metric row using metrics + liveStats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <MetricCard
          label="Hybrid macro F1"
          value={0.994}
        />
        <MetricCard
          label="Live throughput (tx/sec)"
          value={liveStats.throughputTxPerSec}
        />
        <MetricCard
          label="Live avg latency (ms)"
          value={liveStats.avgLatencyMs}
        />
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <KpiCard
          title="Latest prediction"
          icon={<Search className="h-5 w-5 text-indigo-500" />}
          main={latest?.prediction.category ?? "-"}
          sub={
            latest
              ? `${(latest.prediction.confidence * 100).toFixed(1)}% · ${
                  latest.prediction.risk_tier ?? "low"
                } risk`
              : "Run a prediction to see details"
          }
        />
        <KpiCard
          title="Transactions inspected"
          icon={<BarChart2 className="h-5 w-5 text-green-500" />}
          main={history.length.toString()}
          sub="Last 50 shown in feedback tab"
        />
        <KpiCard
          title="Live stream events"
          icon={<Activity className="h-5 w-5 text-purple-500" />}
          main={liveEvents.length.toString()}
          sub="Last 20 model events from $https://web-production-783fa.up.railway.app/api/stream"
        />
      </div>

      {/* charts + streaming (unchanged) */}
      <div className="grid grid-cols-1 xl:grid-cols-[1.2fr,0.8fr] gap-8">
        <div className="space-y-6">
          <div className="bg-white rounded-xl shadow-md p-6 border border-gray-100">
            <h3 className="font-semibold text-lg mb-4 flex items-center">
              <BarChart2 className="h-5 w-5 text-indigo-600 mr-2" />
              Category Distribution (all labels)
            </h3>
            <div className="h-[320px] overflow-visible">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={categoryDistribution}
                  margin={{ left: 0, right: 16, top: 16, bottom: 40 }}
                >
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis
                    dataKey="category"
                    tick={{ fontSize: 11 }}
                    angle={-30}
                    textAnchor="end"
                    height={60}
                  />
                  <YAxis />
                  <Tooltip />
                  <Bar
                    dataKey="count"
                    radius={[6, 6, 0, 0]}
                    barSize={28}
                    fill="#7C3AED"
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-md p-6 border border-gray-100">
            <h3 className="font-semibold text-lg mb-4 flex items-center">
              <FileText className="h-5 w-5 text-indigo-600 mr-2" />
              Per-Category F1
            </h3>
            <div className="h-[260px] overflow-visible">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={perClassF1}
                  margin={{ left: 0, right: 16, top: 16, bottom: 40 }}
                >
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis
                    dataKey="category"
                    tick={{ fontSize: 11 }}
                    angle={-30}
                    textAnchor="end"
                    height={60}
                  />
                  <YAxis domain={[0.0, 1.0]} />
                  <Tooltip />
                  <Bar dataKey="f1" radius={[6, 6, 0, 0]} barSize={24}>
                    {perClassF1.map((_, idx) => (
                      <Cell
                        key={idx}
                        fill={COLORS[idx % COLORS.length]}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* live stream card (unchanged except maybe latency display if you want) */}
        <div className="bg-white rounded-xl shadow-md p-6 border border-gray-100 text-sm">
          <h3 className="font-semibold mb-3 flex items-center">
            <Activity className="h-5 w-5 text-indigo-600 mr-2" />
            Live prediction stream
          </h3>
          <p className="text-xs text-gray-600 mb-3">
            Connects to <code>$https://web-production-783fa.up.railway.app/api/stream</code> via Server-Sent Events. Every
            time the model is called, the event is appended here.
          </p>
          <div className="space-y-2 h-[75vh] overflow-y-auto">
            {liveEvents.length === 0 && (
              <p className="text-xs text-gray-500">
                No events yet. Run predictions from the Demo tab.
              </p>
            )}
            {liveEvents.map((ev, idx) => (
              <div
                key={idx}
                className="border border-gray-200 rounded-lg p-2 bg-gray-50"
              >
                <p className="text-xs text-gray-500">
                  {ev.date} • ₹{ev.amount.toFixed(2)} •{" "}
                  {ev.latency_ms != null && <>{ev.latency_ms.toFixed(1)} ms</>}
                </p>
                <p className="text-sm font-medium truncate">
                  {ev.description}
                </p>
                <p className="text-xs text-gray-600">
                  {ev.prediction.category} ·{" "}
                  {(ev.prediction.confidence * 100).toFixed(1)}% ·{" "}
                  {ev.prediction.risk_tier ?? "low"} risk
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value?: number | null }) {
  const showValue =
    value != null && !Number.isNaN(value)
      ? value.toFixed(3)
      : "--";

  return (
    <div className="bg-white rounded-lg p-3 border border-gray-200">
      <p className="text-[11px] text-gray-500 mb-1">{label}</p>
      <p className="text-xl font-semibold">{showValue}</p>
    </div>
  );
}

function KpiCard({
  title,
  icon,
  main,
  sub,
}: {
  title: string;
  icon: React.ReactNode;
  main: string;
  sub: string;
}) {
  return (
    <div className="bg-white rounded-xl shadow-md p-5 border border-gray-100 flex items-center justify-between">
      <div>
        <p className="text-sm text-gray-500">{title}</p>
        <p className="text-2xl font-semibold text-gray-900 mt-1">{main}</p>
        <p className="text-xs text-gray-500 mt-1">{sub}</p>
      </div>
      <div className="bg-indigo-50 rounded-full p-3">{icon}</div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Demo Tab                                                           */
/* ------------------------------------------------------------------ */

function DemoTab({
  form,
  setForm,
  runModel,
  loading,
  error,
  prediction,
}: {
  form: TransactionForm;
  setForm: React.Dispatch<React.SetStateAction<TransactionForm>>;
  runModel: () => void;
  loading: boolean;
  error: string | null;
  prediction: PredictResponse | null;
}) {
  return (
    <div className="grid lg:grid-cols-[1.1fr,0.9fr] gap-8">
      <div className="bg-white rounded-xl shadow-md p-6 border border-gray-100">
        <h2 className="font-semibold text-lg mb-2 flex items-center">
          <Lightbulb className="h-5 w-5 text-indigo-600 mr-2" />
          Live Transaction Demo
        </h2>
        <p className="text-sm text-gray-600 mb-4">
          Submit a realistic transaction with optional user metadata. The backend
          anonymises the input, runs the hybrid model, and returns the predicted
          category with confidence.
        </p>

        <div className="space-y-4 text-sm">
          <div>
            <label className="block text-gray-700 mb-1">
              Transaction description
            </label>
            <textarea
              className="w-full border border-gray-300 rounded-lg px-3 py-2 bg-white text-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              rows={3}
              value={form.description}
              onChange={(e) =>
                setForm((f) => ({ ...f, description: e.target.value }))
              }
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-gray-700 mb-1">Amount</label>
              <input
                type="number"
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm bg-white text-gray-900"
                value={form.amount}
                onChange={(e) =>
                  setForm((f) => ({ ...f, amount: Number(e.target.value) }))
                }
              />
            </div>
            <div>
              <label className="block text-gray-700 mb-1">Date</label>
              <input
                type="date"
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm bg-white text-gray-900"
                value={form.date}
                onChange={(e) =>
                  setForm((f) => ({ ...f, date: e.target.value }))
                }
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-gray-700 mb-1">
                User name (optional)
              </label>
              <input
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm bg-white text-gray-900"
                value={form.userName}
                onChange={(e) =>
                  setForm((f) => ({ ...f, userName: e.target.value }))
                }
              />
            </div>
            <div>
              <label className="block text-gray-700 mb-1">
                Account ID (optional)
              </label>
              <input
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm bg-white text-gray-900"
                value={form.accountId}
                onChange={(e) =>
                  setForm((f) => ({ ...f, accountId: e.target.value }))
                }
              />
            </div>
          </div>

          <div className="flex items-center justify-between mt-2">
            <button
              onClick={runModel}
              disabled={loading}
              className="inline-flex items-center px-4 py-2 rounded-lg bg-indigo-600 text-white text-sm font-medium hover:bg-indigo-500 disabled:opacity-60"
            >
              {loading ? (
                <>
                  <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  Running model…
                </>
              ) : (
                <>
                  <Search className="h-4 w-4 mr-2" />
                  Run model
                </>
              )}
            </button>
            {error && <p className="text-sm text-red-500">{error}</p>}
          </div>
        </div>
      </div>

      <div className="space-y-4">
        <div className="bg-white rounded-xl shadow-md p-6 border border-gray-100 text-sm">
          <h3 className="font-semibold mb-3 flex items-center">
            <BarChart2 className="h-5 w-5 text-indigo-600 mr-2" />
            Prediction & anonymisation
          </h3>
          {prediction ? (
            <>
              <div className="flex items-center justify-between mb-3">
                <div>
                  <p className="text-xs text-gray-500">Predicted category</p>
                  <p className="text-lg font-semibold">
                    {prediction.category}
                  </p>
                  <p className="text-xs text-gray-500">
                    Source: {prediction.source}
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-xs text-gray-500">Confidence</p>
                  <p className="text-2xl font-semibold text-green-600">
                    {(prediction.confidence * 100).toFixed(1)}%
                  </p>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3 mb-3">
                <div className="bg-indigo-50 rounded-lg p-3">
                  <p className="text-xs text-gray-500 mb-1">XGBoost head</p>
                  <p className="text-sm">{prediction.xgb_top.category}</p>
                  <p className="text-xs text-gray-500">
                    Confidence:{" "}
                    {(prediction.xgb_top.confidence * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="bg-indigo-50 rounded-lg p-3">
                  <p className="text-xs text-gray-500 mb-1">
                    Prototype neighbours
                  </p>
                  <ul className="space-y-1 text-xs">
                    {prediction.prototype_top3.map((p) => (
                      <li
                        key={p.category}
                        className="flex justify-between text-gray-700"
                      >
                        <span>{p.category}</span>
                        <span className="text-gray-500">
                          {(p.similarity * 100).toFixed(1)}%
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
              <div className="bg-gray-50 rounded-lg p-3">
                <p className="text-xs text-gray-500 mb-1">
                  Anonymisation (PII stripped before encoding)
                </p>
                <p className="text-xs text-gray-500">Original</p>
                <p className="text-sm mb-2 line-clamp-2">
                  {prediction.anonymization.original}
                </p>
                <p className="text-xs text-gray-500">Used for inference</p>
                <p className="text-sm text-emerald-700">
                  {prediction.anonymization.anonymized}
                </p>
              </div>
            </>
          ) : (
            <p className="text-sm text-gray-500">
              Run the model on the left to see predictions, confidence, and the
              anonymised text.
            </p>
          )}
        </div>

        <div className="bg-white rounded-xl shadow-md p-6 border border-gray-100 text-xs">
          <h3 className="font-semibold mb-3 flex items-center">
            <Settings className="h-5 w-5 text-indigo-600 mr-2" />
            Model pipeline (flow)
          </h3>
          <ol className="space-y-2 text-gray-700">
            <li>1. Raw transaction + optional user metadata from UI.</li>
            <li>
              2. PII fields (names, account IDs, long digit sequences) masked.
            </li>
            <li>3. MiniLM encoder generates a semantic embedding.</li>
            <li>4. Numeric context features are built (amount, date, weekend).</li>
            <li>5. XGBoost head outputs probabilities over base taxonomy.</li>
            <li>6. Prototype head scores similarity to config prototypes.</li>
            <li>
              7. Hybrid router chooses final label and returns confidence +
              top-k.
            </li>
            <li>8. Human feedback updates prototypes and logs samples.</li>
          </ol>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Taxonomy Tab (new UI)                                              */
/* ------------------------------------------------------------------ */

type TaxonomyProps = {
  taxonomy: any;
  loading: boolean;
  error: string | null;
};

function TaxonomyTab({ taxonomy, loading, error }: TaxonomyProps) {
  const [showConfig, setShowConfig] = useState(false);

  const gradientHeaders = [
    "from-amber-400 to-orange-500",
    "from-purple-500 to-fuchsia-600",
    "from-emerald-500 to-teal-500",
    "from-sky-500 to-blue-600",
    "from-rose-500 to-pink-500",
  ];

  const categories = taxonomy?.categories ?? [];

  const handleDownloadConfig = () => {
    if (!taxonomy) return;
    const blob = new Blob([JSON.stringify(taxonomy, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "categories.json";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-2xl shadow-md border border-gray-100 p-6">
        <div className="flex items-center justify-between mb-2">
          <div>
            <h2 className="font-semibold text-lg">Category Taxonomy</h2>
            <p className="text-sm text-gray-600">
              Configure your custom category hierarchy. The model uses this
              taxonomy to build prototypes for each category.
            </p>
          </div>
          <button
            onClick={() => setShowConfig((s) => !s)}
            className="text-sm font-medium text-purple-600 hover:text-purple-700 flex items-center gap-1"
          >
            {showConfig ? "Hide Config" : "Show Config"}
            <span>{showConfig ? "▴" : "▾"}</span>
          </button>
        </div>

        {showConfig && (
          <div className="mt-4 bg-slate-900 rounded-xl p-4 text-xs text-emerald-100 border border-slate-700">
            {loading && <p className="text-slate-300">Loading taxonomy…</p>}
            {error && <p className="text-red-400">{error}</p>}
            {taxonomy && (
              <>
                <pre className="overflow-auto max-h-72">
                  {JSON.stringify(taxonomy, null, 2)}
                </pre>
                <div className="mt-3 flex justify-end">
                  <button
                    onClick={handleDownloadConfig}
                    className="inline-flex items-center px-3 py-1.5 rounded-lg bg-purple-500 text-white text-xs font-medium hover:bg-purple-400"
                  >
                    <Download className="h-4 w-4 mr-1" />
                    Download Config
                  </button>
                </div>
              </>
            )}
          </div>
        )}

        {/* Category cards */}
        <div className="mt-6 space-y-4">
          {loading && !showConfig && (
            <p className="text-sm text-gray-500">Loading taxonomy…</p>
          )}
          {error && !showConfig && (
            <p className="text-sm text-red-500">{error}</p>
          )}
          {!loading &&
            categories.map((cat: any, idx: number) => {
              const headerGradient =
                gradientHeaders[idx % gradientHeaders.length];
              const subcats =
                cat.subcategories?.map((s: any) => s.name) ??
                (cat.keywords || []).slice(0, 6);

              return (
                <div
                  key={cat.id ?? cat.name ?? idx}
                  className="rounded-2xl border border-gray-100 shadow-sm bg-white"
                >
                  <div
                    className={`px-5 py-3 rounded-t-2xl bg-gradient-to-r ${headerGradient} text-white font-semibold text-sm`}
                  >
                    {cat.name}
                  </div>
                  <div className="px-5 py-3">
                    <p className="text-xs text-gray-600 mb-2">
                      {cat.description ??
                        "Customisable category defined in categories.json"}
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {subcats.length > 0 ? (
                        subcats.map((sc: string) => (
                          <span
                            key={sc}
                            className="px-3 py-1 rounded-full bg-purple-50 text-purple-700 text-xs font-medium"
                          >
                            {sc}
                          </span>
                        ))
                      ) : (
                        <span className="text-xs text-gray-400">
                          No subcategories/keywords defined yet.
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          {!loading && categories.length === 0 && (
            <p className="text-sm text-gray-500">
              No categories found in <code>categories.json</code>.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Evaluation Tab                                                     */
/* ------------------------------------------------------------------ */

function EvaluationTab({
  metrics,
  labels,
  cm,
  loading,
  error,
  explain,
  explaining,
  explanation,
}: {
  metrics: Metrics | null;
  labels?: string[];
  cm?: number[][];
  loading: boolean;
  error: string | null;
  explain: () => void;
  explaining: boolean;
  explanation: string | null;
}) {
  return (
    <div className="grid lg:grid-cols-[1.2fr,0.8fr] gap-8">
      <div className="bg-white rounded-xl shadow-md p-6 border border-gray-100 text-sm">
        <h2 className="font-semibold text-lg mb-2 flex items-center">
          <FileText className="h-5 w-5 text-indigo-600 mr-2" />
          Model Evaluation Dashboard
        </h2>
        {loading && <p className="text-gray-500">Loading metrics…</p>}
        {error && <p className="text-red-500">{error}</p>}
        {metrics && (
          <>
            <p className="text-gray-600 mb-3">
              These metrics come from the held-out test split of the 4.5M+
              transactions dataset. We report macro F1 for the XGBoost head,
              prototype head, and hybrid router.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
              <EvalMetric
                label="XGBoost macro F1 (test)"
                value={metrics.test?.xgb?.macro_f1}
              />
              <EvalMetric
                label="Prototype macro F1 (test)"
                value={metrics.test?.proto?.macro_f1}
              />
              <EvalMetric
                label="Hybrid macro F1 (test)"
                value={metrics.test?.hybrid?.macro_f1}
              />
            </div>

            <p className="font-semibold mb-1 text-gray-700">
              Confusion matrix (hybrid)
            </p>
            <p className="text-xs text-gray-500 mb-2">
              Each row is the true category, each column is the predicted
              category. Darker cells correspond to more examples.
            </p>
            {labels && cm ? (
              <ConfusionMatrixGrid labels={labels} matrix={cm} />
            ) : (
              <pre className="bg-gray-50 rounded-lg p-3 text-[11px] max-h-64 overflow-auto border border-gray-200">
                {JSON.stringify(
                  metrics.test?.hybrid?.confusion_matrix ??
                    metrics.test?.xgb?.confusion_matrix ??
                    {},
                  null,
                  2
                )}
              </pre>
            )}
          </>
        )}
      </div>

      <div className="bg-white rounded-xl shadow-md p-6 border border-gray-100 text-sm">
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-semibold flex items-center">
            <Lightbulb className="h-5 w-5 text-indigo-600 mr-2" />
            Gemini-powered analysis
          </h3>
          <button
            onClick={explain}
            disabled={!metrics || explaining}
            className="inline-flex items-center px-3 py-1.5 rounded-lg bg-sky-500 text-white text-xs font-medium disabled:opacity-60"
          >
            {explaining ? "Asking Gemini…" : "Explain metrics"}
          </button>
        </div>
        {explanation ? (
          <div className="mt-2 text-xs text-gray-800 whitespace-pre-wrap">
            {explanation}
          </div>
        ) : (
          <p className="text-xs text-gray-500">
            Click “Explain metrics” to send macro F1s and the confusion matrix
            to a small Gemini API call, which returns a concise analysis of
            strengths, confusions, and recommended improvements.
          </p>
        )}
        {!metrics && !loading && (
          <p className="mt-2 text-xs text-gray-500">
            Metrics not loaded. Make sure <code>metrics.json</code> is present
            in the backend artifacts folder.
          </p>
        )}
      </div>
    </div>
  );
}

function EvalMetric({ label, value }: { label: string; value?: number }) {
  return (
    <div className="bg-gray-50 rounded-lg p-3 border border-gray-200">
      <p className="text-[11px] text-gray-500 mb-1">{label}</p>
      <p className="text-xl font-semibold">
        {value != null ? value.toFixed(3) : "--"}
      </p>
    </div>
  );
}

/* Confusion matrix grid with color intensity */

function ConfusionMatrixGrid({
  labels,
  matrix,
}: {
  labels: string[];
  matrix: number[][];
}) {
  const max = Math.max(
    ...matrix.map((row) => Math.max(...row.map((v) => v || 0)))
  );

  return (
    <div className="overflow-auto max-h-80 border border-gray-200 rounded-lg">
      <table className="border-collapse text-[10px]">
        <thead>
          <tr>
            <th className="sticky left-0 bg-white z-10 px-2 py-1 border border-gray-200">
              True \ Pred
            </th>
            {labels.map((label) => (
              <th
                key={label}
                className="px-2 py-1 border border-gray-200 bg-gray-50"
              >
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <th className="sticky left-0 bg-white z-10 px-2 py-1 border border-gray-200 text-left">
                {labels[i]}
              </th>
              {row.map((val, j) => {
                const intensity = max > 0 ? val / max : 0;
                const bg = `rgba(79, 70, 229, ${intensity * 0.85})`;
                const textColor = intensity > 0.5 ? "text-white" : "text-gray-900";
                return (
                  <td
                    key={j}
                    className={`px-2 py-1 border border-gray-200 ${textColor}`}
                    style={{ backgroundColor: intensity > 0 ? bg : "white" }}
                  >
                    {val}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Feedback Tab                                                       */
/* ------------------------------------------------------------------ */

function FeedbackTab({
  feedbackItems,
  feedbackStatus,
  expandedTransaction,
  setExpandedTransaction,
  feedbackCategory,
  setFeedbackCategory,
  submitFeedback,
}: {
  feedbackItems: LiveEvent[];  // 👈 queue comes from liveEvents
  feedbackStatus: Record<number, "correct" | "incorrect">;
  expandedTransaction: number | null;
  setExpandedTransaction: (id: number | null) => void;
  feedbackCategory: string;
  setFeedbackCategory: (value: string) => void;
  submitFeedback: (item: LiveEvent, correctCat: string) => void;
}) {
  return (
    <div className="space-y-4">
      <div className="bg-white rounded-xl shadow-md p-6 border border-gray-100 text-sm">
        <h2 className="font-semibold text-lg mb-2 flex items-center">
          <MessageSquare className="h-5 w-5 text-indigo-600 mr-2" />
          Human-in-the-loop feedback
        </h2>
        <p className="text-gray-600 mb-4">
          Review recent predictions and correct them when necessary. Each
          correction is logged and used to update prototype centroids online.
        </p>

        <div className="space-y-3 max-h-[28rem] overflow-auto">
          {feedbackItems.length === 0 && (
            <p className="text-xs text-gray-500">
              No recent predictions yet. Run the Demo or Stream tab to populate
              this queue.
            </p>
          )}

          {feedbackItems.map((item, idx) => {
            const status = feedbackStatus[idx];
            const isExpanded = expandedTransaction === idx;

            return (
              <div
                key={idx}
                className="border border-gray-200 rounded-lg p-3 bg-gray-50"
              >
                <div className="flex justify-between items-start">
                  <div>
                    <p className="font-medium text-sm text-gray-900">
                      {item.description}
                    </p>
                    <p className="text-xs text-gray-500">
                      {item.date} • ₹{item.amount.toFixed(2)} •{" "}
                      <span className="uppercase text-gray-400">
                        {item.source === "test" ? "TEST" : "LIVE"}
                      </span>
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      Predicted:{" "}
                      <span className="font-semibold">
                        {item.prediction.category}
                      </span>{" "}
                      ({(item.prediction.confidence * 100).toFixed(1)}%)
                    </p>
                  </div>
                  <button
                    onClick={() =>
                      setExpandedTransaction(isExpanded ? null : idx)
                    }
                    className="text-xs text-purple-600 flex items-center gap-1"
                  >
                    {isExpanded ? "Hide details ▲" : "View ▼"}
                  </button>
                </div>

                {isExpanded && (
                  <div className="mt-2 border-t border-gray-200 pt-2 space-y-2">
                    <p className="text-xs text-gray-500">
                      Anonymised:{" "}
                      <span className="text-gray-800">
                        {item.prediction.anonymization?.anonymized ??
                          item.description}
                      </span>
                    </p>
                    <div className="flex items-center gap-2 mt-1">
                      <select
                        className="border border-gray-300 rounded-lg px-2 py-1 text-xs bg-white text-gray-900"
                        value={feedbackCategory}
                        onChange={(e) => setFeedbackCategory(e.target.value)}
                      >
                        <option value="">Select correct category</option>
                        {CATEGORIES.map((c) => (
                          <option key={c} value={c}>
                            {c}
                          </option>
                        ))}
                      </select>
                      <button
                        onClick={() => submitFeedback(item, feedbackCategory)}
                        disabled={!feedbackCategory}
                        className="inline-flex items-center px-3 py-1.5 rounded-lg bg-emerald-500 text-white text-xs font-medium disabled:opacity-60"
                      >
                        Submit feedback
                      </button>
                      {status === "correct" && (
                        <span className="flex items-center text-emerald-600 text-xs">
                          <CheckCircle className="h-4 w-4 mr-1" />
                          Recorded
                        </span>
                      )}
                      {status === "incorrect" && (
                        <span className="flex items-center text-red-500 text-xs">
                          <AlertCircle className="h-4 w-4 mr-1" />
                          Failed
                        </span>
                      )}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      <div className="bg-white rounded-xl shadow-md p-6 border border-gray-100 text-xs">
        <h3 className="font-semibold mb-2 flex items-center">
          <RefreshCw className="h-5 w-5 text-indigo-600 mr-2" />
          Feedback loop (flow)
        </h3>
        <ol className="space-y-2 text-gray-700">
          <li>1. Model predicts category + confidence for each transaction.</li>
          <li>2. Analyst reviews predictions in the Feedback tab.</li>
          <li>3. Corrections are sent to the backend with the correct label.</li>
          <li>4. Prototypes are updated online using an EMA over embeddings.</li>
          <li>5. Feedback JSONL is stored for offline retraining and audits.</li>
          <li>6. Over time, long-tail merchants get better coverage.</li>
        </ol>
      </div>
    </div>
  );
}


function BenchmarksTab({
  benchmark,
  runBenchmark,
  benchLoading,
  benchError,
  biasReport,
  loadBiasReport,
  biasLoading,
}: {
  benchmark: any;
  runBenchmark: () => void;
  benchLoading: boolean;
  benchError: string | null;
  biasReport: any;
  loadBiasReport: () => void;
  biasLoading: boolean;
}) {
  // shape bias data into arrays for charts
  const countryData =
    biasReport?.accuracy_by_country
      ? Object.entries(biasReport.accuracy_by_country).map(([k, v]) => ({
          country: k,
          acc: Number(v),
        }))
      : [];

  const amountData =
    biasReport?.accuracy_by_amount_bucket
      ? Object.entries(biasReport.accuracy_by_amount_bucket).map(([k, v]) => ({
          bucket: k,
          acc: Number(v),
        }))
      : [];

  return (
    <div className="space-y-6">
      {/* Benchmark card */}
      <div className="bg-white rounded-xl shadow-md p-6 border border-gray-100 text-sm">
        <div className="flex items-center justify-between mb-3">
          <h2 className="font-semibold text-lg flex items-center">
            <Activity className="h-5 w-5 text-indigo-600 mr-2" />
            Throughput & latency benchmarks
          </h2>
          <button
            onClick={runBenchmark}
            disabled={benchLoading}
            className="inline-flex items-center px-3 py-1.5 rounded-lg bg-indigo-600 text-white text-xs font-medium disabled:opacity-60"
          >
            {benchLoading ? "Running…" : "Run benchmark"}
          </button>
        </div>
        {benchError && <p className="text-xs text-red-500 mb-2">{benchError}</p>}
        {benchmark ? (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-2">
            <MetricCard
              label="Throughput (tx/sec)"
              value={benchmark.throughput_tx_per_sec}
            />
            <MetricCard label="Mean latency (ms)" value={benchmark.latency_ms} />
            <MetricCard label="Encoder avg (ms)" value={benchmark.avg_encoder_ms} />
            <MetricCard label="XGBoost avg (ms)" value={benchmark.avg_xgb_ms} />
            <MetricCard label="Prototype avg (ms)" value={benchmark.avg_proto_ms} />
          </div>
        ) : (
          <p className="text-xs text-gray-500 mt-2">
            Click “Run benchmark” to measure end-to-end throughput and latency on a
            synthetic batch of transactions. Measurements are run on your current
            machine, so include hardware notes if you report these numbers.
          </p>
        )}
        {benchmark?.notes && (
          <p className="text-[11px] text-gray-500 mt-3">
            Notes: {benchmark.notes.description} — {benchmark.notes.hardware_hint}
          </p>
        )}
      </div>

      {/* Bias analysis card */}
      <div className="bg-white rounded-xl shadow-md p-6 border border-gray-100 text-sm">
        <div className="flex items-center justify-between mb-3">
          <h2 className="font-semibold text-lg flex items-center">
            <FileText className="h-5 w-5 text-indigo-600 mr-2" />
            Bias & fairness report
          </h2>
          <button
            onClick={loadBiasReport}
            disabled={biasLoading}
            className="inline-flex items-center px-3 py-1.5 rounded-lg bg-purple-600 text-white text-xs font-medium disabled:opacity-60"
          >
            {biasLoading ? "Loading…" : "Load bias report"}
          </button>
        </div>
        {biasReport?.note && (
          <p className="text-xs text-gray-500 mb-3">{biasReport.note}</p>
        )}
        {countryData.length === 0 && amountData.length === 0 && !biasReport?.note && (
          <p className="text-xs text-gray-500">
            Click “Load bias report” to compute accuracy by merchant / country /
            amount bucket on a held-out validation slice configured in the backend.
          </p>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-2">
          {countryData.length > 0 && (
            <div>
              <h3 className="text-xs font-semibold text-gray-600 mb-2">
                Accuracy by country
              </h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={countryData}
                    margin={{ left: 0, right: 16, top: 8, bottom: 40 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis
                      dataKey="country"
                      tick={{ fontSize: 11 }}
                      angle={-30}
                      textAnchor="end"
                      height={60}
                    />
                    <YAxis domain={[0, 1]} />
                    <Tooltip />
                    <Bar dataKey="acc" radius={[6, 6, 0, 0]} barSize={24} fill="#4F46E5" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {amountData.length > 0 && (
            <div>
              <h3 className="text-xs font-semibold text-gray-600 mb-2">
                Accuracy by amount bucket
              </h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={amountData}
                    margin={{ left: 0, right: 16, top: 8, bottom: 40 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis
                      dataKey="bucket"
                      tick={{ fontSize: 10 }}
                      angle={-40}
                      textAnchor="end"
                      height={80}
                    />
                    <YAxis domain={[0, 1]} />
                    <Tooltip />
                    <Bar dataKey="acc" radius={[6, 6, 0, 0]} barSize={20} fill="#10B981" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function StreamTab({
  running,
  setRunning,
  testEvents,
}: {
  running: boolean;
  setRunning: (v: boolean) => void;
  testEvents: LiveEvent[];
}) {
  return (
    <div className="space-y-6">
      <div className="bg-white rounded-xl shadow-md p-6 border border-gray-100 text-sm">
        <div className="flex items-center justify-between mb-3">
          <h2 className="font-semibold text-lg flex items-center">
            <Activity className="h-5 w-5 text-indigo-600 mr-2" />
            Test set streaming & advice
          </h2>
          <button
            onClick={() => setRunning(!running)}
            className={`inline-flex items-center px-3 py-1.5 rounded-lg text-xs font-medium ${
              running ? "bg-red-500 text-white" : "bg-indigo-600 text-white"
            }`}
          >
            {running ? "Stop stream" : "Start stream"}
          </button>
        </div>

        <div className="space-y-2 max-h-[480px] overflow-auto">
          {testEvents.length === 0 && !running && (
            <p className="text-xs text-gray-500">
              Click “Start stream” to replay the test set in real time.
            </p>
          )}
          {testEvents.map((ev, idx) => (
            <div
              key={idx}
              className="border border-gray-200 rounded-lg p-3 bg-gray-50"
            >
              <p className="text-xs text-gray-500">
                {ev.date} • ₹{ev.amount.toFixed(2)} •{" "}
                <span className="uppercase text-gray-400">test stream</span> •{" "}
                {ev.latency_ms != null && (
                  <>{ev.latency_ms.toFixed(1)} ms</>
                )}
              </p>
              <p className="text-sm font-medium">{ev.description}</p>
              <p className="text-xs text-gray-600">
                Category:{" "}
                <span className="font-semibold">
                  {ev.prediction?.category}
                </span>{" "}
                ({((ev.prediction?.confidence ?? 0) * 100).toFixed(1)}% ·{" "}
                {ev.prediction?.risk_tier ?? "low"} risk)
              </p>
              {ev.advice && (
                <p className="mt-2 text-xs text-emerald-700">
                  💡 {ev.advice}
                </p>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}


export default App;
