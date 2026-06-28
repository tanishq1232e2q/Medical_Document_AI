import React, { useState, useEffect, useCallback } from "react";
import axios from "axios";
// import "./HistoryPage.css";

function IssuesBadges({ result }) {
  const issues = result.rule_result?.issues || [];
  return (
    <div className="badges-row">
      {issues.slice(0, 3).map((iss, i) => (
        <span key={i} className={`badge badge-${(iss.severity||"medium").toLowerCase()}`}>
          {iss.type}
        </span>
      ))}
      {issues.length > 3 && (
        <span className="badge badge-neutral">+{issues.length - 3}</span>
      )}
    </div>
  );
}

function EntitiesRow({ entities }) {
  const meds = entities?.MEDICATION || entities?.medications || [];
  if (!meds.length) return null;
  return (
    <div className="entity-row">
      {meds.slice(0, 5).map((m, i) => (
        <span key={i} className="chip chip-medication" style={{ fontSize: 10 }}>{m}</span>
      ))}
      {meds.length > 5 && <span className="chip chip-medication" style={{ fontSize: 10 }}>+{meds.length - 5}</span>}
    </div>
  );
}

function HistoryDetailPanel({ item, onClose, onDelete }) {
  if (!item) return null;

  const issues = item.rule_result?.issues || [];
  const allIssues = [...issues];
  if (item.llm_result?.has_contradiction && !item.rule_result?.has_contradiction) {
    allIssues.push({
      type: item.llm_result.contradiction_type || "Clinical Issue",
      warning: item.llm_result.llm_response,
      correction: item.llm_result.correction,
      severity: "MEDIUM"
    });
  }

  return (
    <div className="detail-panel">
      <div className="detail-header">
        <div>
          <div className="detail-source">{item.source}</div>
          <div className="detail-time">{new Date(item.timestamp).toLocaleString()}</div>
        </div>
        <div className="detail-actions">
          <button className="btn btn-danger" onClick={() => onDelete(item.id)}>Delete</button>
          <button className="btn btn-outline" onClick={onClose}>✕</button>
        </div>
      </div>

      <div className="detail-text-box">{item.text}</div>

      <div className={`verdict-banner ${item.has_contradiction ? "CONTRADICTION" : "CLEAN"}`}>
        <div className="verdict-icon">{item.has_contradiction ? "⚠️" : "✅"}</div>
        <div>
          <div className={`verdict-label ${item.has_contradiction ? "CONTRADICTION" : "CLEAN"}`}>
            {item.has_contradiction ? "Contradiction Detected" : "Clean"}
          </div>
          <div className="verdict-meta">
            {item.severity} severity · {(item.confidence * 100).toFixed(0)}% confidence ·{" "}
            {item.analysis_type} analysis
          </div>
        </div>
      </div>

      {allIssues.length > 0 && (
        <>
          <div className="section-label">Issues</div>
          {allIssues.map((iss, i) => (
            <div key={i} className={`issue-item ${iss.severity || "MEDIUM"}`}>
              <div className={`issue-icon ${iss.severity || "MEDIUM"}`}>⚠️</div>
              <div>
                <div className="issue-type">{iss.type}</div>
                <div className="issue-warning">{iss.warning}</div>
                {iss.correction && <div className="issue-correction">{iss.correction}</div>}
              </div>
            </div>
          ))}
        </>
      )}

      {item.corrections?.length > 0 && (
        <>
          <div className="section-label" style={{ marginTop: 16 }}>Corrections</div>
          {[...new Set(item.corrections)].map((c, i) => (
            <div key={i} className="correction-item">
              <span className="correction-arrow">→</span>
              <span>{c}</span>
            </div>
          ))}
        </>
      )}

      {item.entities && Object.keys(item.entities).length > 0 && (
        <>
          <div className="section-label" style={{ marginTop: 16 }}>Entities</div>
          {Object.entries(item.entities).map(([cat, items]) => {
            if (!items.length) return null;
            const chipClass = {
              MEDICATION: "chip-medication", medications: "chip-medication",
              SYMPTOM: "chip-symptom",    symptoms: "chip-symptom",
              TEST: "chip-test",          tests: "chip-test",
              DIAGNOSIS: "chip-diagnosis",diagnoses: "chip-diagnosis",
              DRUG_DOSES: "chip-dose",    drug_dose_pairs: "chip-dose",
            }[cat] || "chip-medication";
            return (
              <div key={cat} style={{ marginBottom: 10 }}>
                <div className="entity-label" style={{ fontSize: 10, marginBottom: 6 }}>
                  {cat.toUpperCase()}
                </div>
                <div className="entity-chips">
                  {items.slice(0, 8).map((it, i) => (
                    <span key={i} className={`chip ${chipClass}`}>{it}</span>
                  ))}
                </div>
              </div>
            );
          })}
        </>
      )}

      {item.ocr && (
        <>
          <div className="section-label" style={{ marginTop: 16 }}>OCR Info</div>
          <div className="ocr-meta">
            <span className="ocr-stat"><b>{item.ocr.word_count}</b> words</span>
            <span className="ocr-stat"><b>{(item.ocr.confidence * 100).toFixed(0)}%</b> conf</span>
            <span className="ocr-stat">Strategy: <b>{item.ocr.strategy}</b></span>
          </div>
        </>
      )}
    </div>
  );
}

export default function HistoryPage() {
  const [items, setItems]   = useState([]);
  const [total, setTotal]   = useState(0);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("all");
  const [selected, setSelected] = useState(null);
  const [search, setSearch] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const typeParam = filter !== "all" ? `&type=${filter}` : "";
      const res = await axios.get(`/api/history?limit=100${typeParam}`);
      setItems(res.data.items || []);
      setTotal(res.data.total || 0);
    } catch(e) { console.error(e); }
    finally { setLoading(false); }
  }, [filter]);

  useEffect(() => { load(); }, [load]);

  const handleDelete = async (id) => {
    try {
      await axios.delete(`/api/history/${id}`);
      setItems(prev => prev.filter(i => i.id !== id));
      if (selected?.id === id) setSelected(null);
    } catch(e) { alert("Delete failed"); }
  };

  const filtered = items.filter(item => {
    if (!search) return true;
    const s = search.toLowerCase();
    return item.text?.toLowerCase().includes(s) ||
           item.source?.toLowerCase().includes(s);
  });

  return (
    <div className="history-page">
      <div className="history-left">
        <div className="history-toolbar">
          <h1 className="page-title">History</h1>
          <p className="page-sub">{total} total analyses</p>

          <div className="filter-row">
            <input
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search prescriptions…"
              style={{ maxWidth: 240 }}
            />
            <div className="filter-tabs">
              {["all","text","image"].map(f => (
                <button key={f} style={{padding:"0.4rem", borderRadius:"0.4rem 0 0 0.4rem",background:"none", color:"white",margin:"0rem 0.3rem"}} className={`tab-btn ${filter===f?"active":""}`}
                        onClick={() => setFilter(f)}>
                  {f.charAt(0).toUpperCase()+f.slice(1)}
                </button>
              ))}
            </div>
          </div>
        </div>

        {loading ? (
          <div style={{ padding: 40, textAlign: "center" }}>
            <div className="spinner" />
          </div>
        ) : filtered.length === 0 ? (
          <div className="empty-history">
            <div style={{ fontSize: 40, opacity: 0.3 }}>📭</div>
            <div>No analyses found</div>
          </div>
        ) : (
          <div className="history-items">
            {filtered.map(item => (
              <div
                key={item.id}
                className={`history-card ${item.has_contradiction ? "has-issue" : "clean"} ${selected?.id===item.id ? "selected" : ""}`}
                onClick={() => setSelected(item)}
              >
                <div className="hcard-top">
                  <span className="hcard-verdict-icon">
                    {item.has_contradiction ? "⚠️" : "✅"}
                  </span>
                  <span className={`badge badge-${item.has_contradiction ? (item.severity||"medium").toLowerCase() : "clean"}`}>
                    {item.has_contradiction ? item.severity : "CLEAN"}
                  </span>
                  <span className="hcard-type">{item.analysis_type}</span>
                  <span className="hcard-time">
                    {new Date(item.timestamp).toLocaleDateString()}
                  </span>
                </div>
                <div className="hcard-text">{item.text?.slice(0, 120)}{item.text?.length > 120 ? "…" : ""}</div>
                <IssuesBadges result={item} />
                <EntitiesRow entities={item.entities || {}} />
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="history-right">
        {selected ? (
          <HistoryDetailPanel item={selected} onClose={() => setSelected(null)} onDelete={handleDelete} />
        ) : (
          <div className="detail-empty">
            <div style={{ fontSize: 48, opacity: 0.2 }}>📋</div>
            <div>Select an item to view details</div>
          </div>
        )}
      </div>
    </div>
  );
}