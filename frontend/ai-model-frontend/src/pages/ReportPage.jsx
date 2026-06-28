import React, { useState, useEffect, useCallback } from "react";
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from "recharts";
import axios from "axios";


const COLORS = {
  "Overdose":           "#ff4757",
  "Underdose":          "#ffa502",
  "Drug Interaction":   "#ff6b81",
  "Contraindication":   "#c44569",
  "Frequency Conflict": "#f8b739",
  "Clean":              "#2ed573",
  "HIGH":               "#ff4757",
  "MEDIUM":             "#ffa502",
  "LOW":                "#0090ff",
  "NONE":               "#2ed573",
};

const CHART_DEFAULTS = {
  style: { fontFamily: "'IBM Plex Mono', monospace", fontSize: 11 },
};

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "#0f1318", border: "1px solid #2a3540",
      borderRadius: 6, padding: "10px 14px"
    }}>
      <div style={{ color: "#8899aa", fontSize: 11, marginBottom: 6 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color, fontSize: 12, fontFamily: "IBM Plex Mono" }}>
          {p.name}: <b>{p.value}</b>
        </div>
      ))}
    </div>
  );
};

function StatCard({ label, value, sub, color, icon }) {
  return (
    <div className="stat-card" style={{ borderColor: color ? `${color}30` : undefined }}>
      <div className="stat-icon" style={{ background: color ? `${color}15` : undefined }}>
        {icon}
      </div>
      <div>
        <div className="stat-value" style={{ color: color || "var(--text)" }}>{value}</div>
        <div className="stat-label">{label}</div>
        {sub && <div className="stat-sub">{sub}</div>}
      </div>
    </div>
  );
}

export default function ReportPage() {
  const [stats, setStats]     = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [tab, setTab]         = useState("overview");

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [sRes, hRes] = await Promise.all([
        axios.get("/api/stats"),
        axios.get("/api/history?limit=100")
      ]);
      setStats(sRes.data);
      setHistory(hRes.data.items || []);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const downloadReport = () => {
    if (!stats) return;
    const report = {
      generated: new Date().toISOString(),
      summary: stats,
      analyses: history.slice(0, 20)
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href = url; a.download = `medai-report-${Date.now()}.json`;
    a.click(); URL.revokeObjectURL(url);
  };

  if (loading) return (
    <div className="report-loading">
      <div className="spinner" style={{ width: 40, height: 40, borderWidth: 3 }} />
      <div className="loading-text">Loading analytics…</div>
    </div>
  );

  if (!stats || stats.total === 0) return (
    <div className="report-empty">
      <div style={{ fontSize: 56, opacity: 0.3 }}>📊</div>
      <div style={{ fontSize: 18, fontWeight: 600, color: "var(--text2)" }}>No data yet</div>
      <div style={{ fontSize: 13, color: "var(--text3)" }}>
        Analyze some prescriptions first to see your report.
      </div>
    </div>
  );

  // Prepare chart data
  const byTypeData = Object.entries(stats.by_type || {}).map(([name, value]) => ({
    name, value, fill: COLORS[name] || "#8899aa"
  })).sort((a, b) => b.value - a.value);

  const bySeverityData = ["HIGH","MEDIUM","LOW","NONE"].map(s => ({
    name: s, value: stats.by_severity?.[s] || 0, fill: COLORS[s]
  })).filter(d => d.value > 0);

  const trendData = (stats.by_day || []).map(d => ({
    date: d.date.slice(5),
    Contradictions: d.contradictions,
    Clean: d.clean,
    Total: d.total
  }));

  const pieData = [
    { name: "Contradictions", value: stats.contradictions, fill: "#ff4757" },
    { name: "Clean",          value: stats.clean,          fill: "#2ed573" },
  ].filter(d => d.value > 0);

  const textVsImage = [
    { name: "Text Notes", value: stats.text,  fill: "#0090ff" },
    { name: "Images",     value: stats.image, fill: "#9b59b6" },
  ].filter(d => d.value > 0);

  return (
    <div className="report-page">
      <div className="report-header">
        <div>
          <h1 className="page-title">Analytics Report</h1>
          <p className="page-sub">
            Aggregate insights across {stats.total} prescription analyses
          </p>
        </div>
        <div className="report-actions">
          <button className="btn btn-outline" onClick={load}>↻ Refresh</button>
          <button className="btn btn-primary" onClick={downloadReport}>⬇ Export JSON</button>
        </div>
      </div>

      {/* Stat cards */}
      <div className="stat-grid">
        <StatCard icon="📋" label="Total Analyzed"    value={stats.total}           color="#0090ff" />
        <StatCard icon="⚠️" label="Contradictions"    value={stats.contradictions}  color="#ff4757"
          sub={`${stats.total > 0 ? ((stats.contradictions/stats.total)*100).toFixed(0) : 0}% of total`} />
        <StatCard icon="✅" label="Clean"             value={stats.clean}           color="#2ed573" />
        <StatCard icon="📝" label="Text Notes"        value={stats.text}            color="#0090ff" />
        <StatCard icon="🖼️" label="Image Scans"       value={stats.image}           color="#9b59b6" />
        <StatCard icon="🔴" label="HIGH Severity"     value={stats.by_severity?.HIGH || 0}   color="#ff4757" />
      </div>

      {/* Tab switcher */}
      <div className="report-tabs">
        {["overview","trends","types","history"].map(t => (
          <button key={t} className={`tab-btn ${tab===t?"active":""}`} onClick={() => setTab(t)}>
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {/* OVERVIEW TAB */}
      {tab === "overview" && (
        <div className="charts-grid-2">
          <div className="chart-card">
            <div className="card-title">Verdict Distribution</div>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={pieData} dataKey="value" nameKey="name"
                     cx="50%" cy="50%" outerRadius={80} label={({ name, percent }) =>
                       `${name} ${(percent*100).toFixed(0)}%`}
                     labelLine={{ stroke: "#2a3540" }}>
                  {pieData.map((d,i) => <Cell key={i} fill={d.fill} />)}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-card">
            <div className="card-title">Analysis Type</div>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={textVsImage} dataKey="value" nameKey="name"
                     cx="50%" cy="50%" outerRadius={80}
                     label={({ name, percent }) => `${name} ${(percent*100).toFixed(0)}%`}
                     labelLine={{ stroke: "#2a3540" }}>
                  {textVsImage.map((d,i) => <Cell key={i} fill={d.fill} />)}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-card">
            <div className="card-title">Severity Breakdown</div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={bySeverityData} {...CHART_DEFAULTS}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e2730" />
                <XAxis dataKey="name" tick={{ fill: "#8899aa", fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: "#8899aa", fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="value" radius={[4,4,0,0]}>
                  {bySeverityData.map((d,i) => <Cell key={i} fill={d.fill} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-card">
            <div className="card-title">Issues by Type</div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={byTypeData} layout="vertical" {...CHART_DEFAULTS}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e2730" />
                <XAxis type="number" tick={{ fill: "#8899aa", fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis dataKey="name" type="category" width={130}
                       tick={{ fill: "#8899aa", fontSize: 10 }} axisLine={false} tickLine={false} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="value" radius={[0,4,4,0]}>
                  {byTypeData.map((d,i) => <Cell key={i} fill={d.fill} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* TRENDS TAB */}
      {tab === "trends" && (
        <div className="charts-single">
          <div className="chart-card full-width">
            <div className="card-title">Daily Activity (last 30 days)</div>
            {trendData.length === 0 ? (
              <div className="chart-empty">Not enough data yet. Analyze more prescriptions to see trends.</div>
            ) : (
              <ResponsiveContainer width="100%" height={320}>
                <LineChart data={trendData} {...CHART_DEFAULTS}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e2730" />
                  <XAxis dataKey="date" tick={{ fill: "#8899aa", fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: "#8899aa", fontSize: 11 }} axisLine={false} tickLine={false} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ paddingTop: 16, fontSize: 12, fontFamily: "'IBM Plex Mono'" }} />
                  <Line type="monotone" dataKey="Total"          stroke="#0090ff" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="Contradictions" stroke="#ff4757" strokeWidth={2} dot={{ fill: "#ff4757", r: 3 }} />
                  <Line type="monotone" dataKey="Clean"          stroke="#2ed573" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>

          <div className="chart-card full-width">
            <div className="card-title">Daily Contradiction Rate</div>
            {trendData.length === 0 ? (
              <div className="chart-empty">No data yet.</div>
            ) : (
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={trendData} {...CHART_DEFAULTS}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e2730" />
                  <XAxis dataKey="date" tick={{ fill: "#8899aa", fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: "#8899aa", fontSize: 11 }} axisLine={false} tickLine={false} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="Contradictions" fill="#ff4757" radius={[4,4,0,0]} />
                  <Bar dataKey="Clean"          fill="#2ed573" radius={[4,4,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      )}

      {/* TYPES TAB */}
      {tab === "types" && (
        <div className="charts-single">
          <div className="chart-card full-width">
            <div className="card-title">Contradiction Type Distribution</div>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={byTypeData} {...CHART_DEFAULTS}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e2730" />
                <XAxis dataKey="name" tick={{ fill: "#8899aa", fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: "#8899aa", fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="value" radius={[4,4,0,0]}>
                  {byTypeData.map((d,i) => <Cell key={i} fill={d.fill} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="type-table">
            <div className="section-label">Type Breakdown</div>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Contradiction Type</th>
                  <th>Count</th>
                  <th>% of Contradictions</th>
                  <th>Severity</th>
                </tr>
              </thead>
              <tbody>
                {byTypeData.map((d, i) => (
                  <tr key={i}>
                    <td><span style={{ color: d.fill }}>●</span> {d.name}</td>
                    <td style={{ fontFamily: "var(--mono)", color: "var(--accent)" }}>{d.value}</td>
                    <td style={{ fontFamily: "var(--mono)", color: "var(--text2)" }}>
                      {stats.contradictions > 0 ? ((d.value/stats.contradictions)*100).toFixed(1) : 0}%
                    </td>
                    <td>
                      <span className={`badge badge-${
                        ["Overdose","Drug Interaction","Contraindication"].includes(d.name) ? "high" : "medium"
                      }`}>
                        {["Overdose","Drug Interaction","Contraindication"].includes(d.name) ? "HIGH" : "MEDIUM"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* HISTORY TAB */}
      {tab === "history" && (
        <div className="history-section">
          <div className="section-label">Recent Analyses ({history.length})</div>
          {history.length === 0 ? (
            <div className="chart-empty">No history yet.</div>
          ) : (
            <div className="history-list">
              {history.map((item) => (
                <div key={item.id} className={`history-item ${item.has_contradiction ? "has-issue" : "clean"}`}>
                  <div className="history-verdict">
                    {item.has_contradiction ? "⚠️" : "✅"}
                  </div>
                  <div className="history-body">
                    <div className="history-text">{item.text?.slice(0,100)}{item.text?.length > 100 ? "…" : ""}</div>
                    <div className="history-meta">
                      <span className="history-type-tag">{item.analysis_type}</span>
                      {item.has_contradiction && (
                        <span className={`badge badge-${(item.severity||"medium").toLowerCase()}`}>
                          {item.severity}
                        </span>
                      )}
                      {(item.contradiction_types||[]).map((t,i) => (
                        <span key={i} className="badge badge-neutral">{t}</span>
                      ))}
                      <span className="history-time">
                        {new Date(item.timestamp).toLocaleString()}
                      </span>
                    </div>
                  </div>
                  <div className="history-conf">
                    <span style={{ fontFamily:"var(--mono)", fontSize:12, color: item.has_contradiction ? "var(--danger)" : "var(--success)" }}>
                      {(item.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}