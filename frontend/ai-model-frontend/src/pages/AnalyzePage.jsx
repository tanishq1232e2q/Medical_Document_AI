import React, { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import axios from "axios";
// import "./AnalyzePage.css";

// ── Entity section ────────────────────────────────────────────────────────
const ENTITY_CONFIG = {
  MEDICATION: { label: "Medications", chipClass: "chip-medication", icon: "💊" },
  SYMPTOM:    { label: "Symptoms",    chipClass: "chip-symptom",    icon: "🩺" },
  TEST:       { label: "Tests / Labs", chipClass: "chip-test",      icon: "🔬" },
  DIAGNOSIS:  { label: "Diagnoses",   chipClass: "chip-diagnosis",  icon: "📋" },
  DRUG_DOSES: { label: "Drug + Dose", chipClass: "chip-dose",       icon: "⚖️" },
};

const ISSUE_ICON = {
  "Overdose":           { icon: "⬆️", color: "HIGH" },
  "Underdose":          { icon: "⬇️", color: "MEDIUM" },
  "Drug Interaction":   { icon: "⚡", color: "HIGH" },
  "Contraindication":   { icon: "🚫", color: "HIGH" },
  "Frequency Conflict": { icon: "🔄", color: "MEDIUM" },
};

// NEW: visual badge so a finding's origin is obvious at a glance instead of
// rule-engine and LLM output looking like an unexplained duplicate.
const SOURCE_BADGE = {
  rule: { label: "Rule Engine", bg: "#1f6feb22", fg: "#58a6ff" },
  llm:  { label: "LLM",         bg: "#9b59b622", fg: "#bf8fe0" },
  both: { label: "Rule + LLM",  bg: "#2ecc7122", fg: "#3fb950" },
};

function SourceTag({ source }) {
  const cfg = SOURCE_BADGE[source] || SOURCE_BADGE.rule;
  return (
    <span style={{
      fontSize: 11, fontWeight: 600, padding: "2px 8px", borderRadius: 999,
      background: cfg.bg, color: cfg.fg, marginLeft: 8, whiteSpace: "nowrap",
    }}>
      {cfg.label}
    </span>
  );
}

function EntitiesPanel({ entities }) {
  if (!entities || Object.keys(entities).length === 0)
    return <p style={{ color: "var(--text3)", fontSize: 13 }}>No entities extracted.</p>;

  return (
    <div className="entities-grid">
      {Object.entries(ENTITY_CONFIG).map(([key, cfg]) => {
        const items = entities[key] || [];
        if (!items.length) return null;
        return (
          <div key={key} className="entity-group">
            <div className="entity-label">
              <span>{cfg.icon}</span>
              <span>{cfg.label}</span>
              <span className="entity-count">{items.length}</span>
            </div>
            <div className="entity-chips">
              {items.map((item, i) => (
                <span key={i} className={`chip ${cfg.chipClass}`}>{item}</span>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function IssueCard({ issue }) {
  const cfg = ISSUE_ICON[issue.type] || { icon: "⚠️", color: "MEDIUM" };
  return (
    <div className={`issue-item ${issue.severity || cfg.color}`}>
      <div className={`issue-icon ${issue.severity || cfg.color}`}>{cfg.icon}</div>
      <div style={{ flex: 1 }}>
        <div className="issue-type" style={{ display: "flex", alignItems: "center" }}>
          {issue.type}
          <SourceTag source={issue.detected_by} />
        </div>
        <div className="issue-warning">{issue.warning}</div>
        {issue.correction && (
          <div className="issue-correction">{issue.correction}</div>
        )}
      </div>
    </div>
  );
}

function ResultPanel({ result }) {
  const verdict = result.has_contradiction ? "CONTRADICTION" : "CLEAN";

  // Prefer the backend's merged, source-tagged issue list (result.issues —
  // see _tag_and_merge_issues in llm_fix_patch.py). It's authoritative
  // because it already accounts for the suppression gate server-side.
  // Fall back to client-side derivation only for older API responses that
  // don't send `issues` yet.
  let allIssues;
  if (Array.isArray(result.issues)) {
    allIssues = result.issues;
  } else {
    const ruleIssues = (result.rule_result?.issues || []).map(i => ({
      ...i,
      detected_by: i.detected_by || "rule",
    }));
    allIssues = [...ruleIssues];
    if (result.llm_result?.has_contradiction) {
      const matchIdx = allIssues.findIndex(i => i.type === result.llm_result.contradiction_type);
      if (matchIdx >= 0) {
        allIssues[matchIdx] = { ...allIssues[matchIdx], detected_by: "both" };
      } else {
        allIssues.push({
          type: result.llm_result.contradiction_type || "LLM Finding",
          warning: result.llm_result.llm_response,
          correction: result.llm_result.correction,
          severity: "MEDIUM",
          detected_by: "llm",
        });
      }
    }
  }

  return (
    <div className="result-panel">
      {/* Verdict Banner */}
      <div className={`verdict-banner ${verdict}`}>
        <div className="verdict-icon">{verdict === "CONTRADICTION" ? "⚠️" : "✅"}</div>
        <div>
          <div className={`verdict-label ${verdict}`}>
            {verdict === "CONTRADICTION" ? "Contradiction Detected" : "Prescription Safe"}
          </div>
          <div className="verdict-meta">
            {verdict === "CONTRADICTION"
              ? `${allIssues.length} issue${allIssues.length !== 1 ? "s" : ""} found · ${result.severity} severity · ${(result.confidence * 100).toFixed(0)}% confidence`
              : `No clinical contradictions detected · Confidence: ${(result.confidence * 100).toFixed(0)}%`
            }
          </div>
        </div>
        {verdict === "CONTRADICTION" && (
          <div className="badge-row" style={{ marginLeft: "auto", display: "flex", gap: 6, flexWrap: "wrap" }}>
            {(result.contradiction_types || []).map((t, i) => (
              <span key={i} className={`badge badge-${(result.severity || "medium").toLowerCase()}`}>{t}</span>
            ))}
          </div>
        )}
      </div>

      {/* Issues */}
      {allIssues.length > 0 && (
        <>
          <div className="section-label">Issues Found</div>
          {allIssues.map((issue, i) => <IssueCard key={i} issue={issue} />)}
        </>
      )}

      {/* Corrections Summary */}
      {result.corrections?.length > 0 && (
        <>
          <div className="section-label" style={{ marginTop: 20 }}>Corrections</div>
          <div className="corrections-list">
            {[...new Set(result.corrections)].map((c, i) => (
              <div key={i} className="correction-item">
                <span className="correction-arrow">→</span>
                <span>{c}</span>
              </div>
            ))}
          </div>
        </>
      )}

      {/* Entities */}
      <div className="section-label" style={{ marginTop: 20 }}>Extracted Entities</div>
      <EntitiesPanel entities={result.entities} />

      {/* LLM Response */}
      {result.llm_result?.llm_response && (
        <>
          <div className="section-label" style={{ marginTop: 20 }}>LLM Analysis</div>
          <div className="llm-box" style={{ padding: "16px", backgroundColor: "var(--bg2)", borderRadius: 6 }}>
            {/* Main LLM Analysis */}
            <div style={{ display: "flex", gap: 12, marginBottom: result.llm_result.correction ? 16 : 0 }}>
              <span style={{ fontSize: 20 }}>🤖</span>
              <div style={{ flex: 1 }}>
                <span style={{ fontWeight: 600, color: "var(--text1)" }}>LLM:</span>
                <div style={{ marginTop: 6, color: "var(--text2)", lineHeight: 1.5 }}>
                  {result.llm_result.llm_response
                    .split("Correction:")[0]
                    .trim()}
                </div>
              </div>
            </div>

            {/* Corrections Section */}
            {result.llm_result.correction && (
              <div style={{
                paddingTop: 12,
                borderTop: "1px solid var(--border)",
                display: "flex",
                gap: 12
              }}>
                <span style={{ fontSize: 20 }}>✏️</span>
                <div style={{ flex: 1 }}>
                  <span style={{ fontWeight: 600, color: "var(--text1)" }}>Corrections:</span>
                  <div style={{ marginTop: 8, color: "var(--text2)" }}>
                    <div style={{ display: "flex", gap: 8, alignItems: "flex-start" }}>
                      <span style={{ color: "var(--accent)", marginTop: 2 }}>→</span>
                      <span>{result.llm_result.correction}</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </>
      )}

      {/* OCR Info (if image) */}
      {result.ocr && (
        <>
          <div className="section-label" style={{ marginTop: 20 }}>OCR Results</div>
          <div className="ocr-meta">
            <span className="ocr-stat"><b>{result.ocr.word_count}</b> words</span>
            <span className="ocr-stat"><b>{(result.ocr.confidence * 100).toFixed(0)}%</b> confidence</span>
            <span className="ocr-stat">Drug: <b>{result.ocr.has_drug ? "✓" : "✗"}</b></span>
            <span className="ocr-stat">Dose: <b>{result.ocr.has_dose ? "✓" : "✗"}</b></span>
            <span className="ocr-stat">Strategy: <b>{result.ocr.strategy}</b></span>
          </div>
          {result.ocr.text && (
            <div className="ocr-text">{result.ocr.text.slice(0, 400)}{result.ocr.text.length > 400 ? "…" : ""}</div>
          )}
        </>
      )}
    </div>
  );
}

// ── Image dropzone ────────────────────────────────────────────────────────
function ImageDropzone({ onFile, preview, onClear }) {
  const onDrop = useCallback(files => { if (files[0]) onFile(files[0]); }, [onFile]);
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: { "image/*": [".jpg",".jpeg",".png",".bmp",".tiff"] }, multiple: false
  });

  if (preview) return (
    <div className="image-preview-wrap">
      <img src={preview} alt="prescription" className="image-preview" />
      <button className="btn btn-outline image-clear-btn" onClick={onClear}>Remove</button>
    </div>
  );

  return (
    <div {...getRootProps()} className={`dropzone ${isDragActive ? "active" : ""}`}>
      <input {...getInputProps()} />
      <div className="dropzone-icon">📷</div>
      <div className="dropzone-text">
        {isDragActive ? "Drop image here" : "Drag & drop or click to upload"}
      </div>
      <div className="dropzone-sub">JPG, PNG, BMP, TIFF · Max 10MB</div>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────
export default function AnalyzePage() {
  const [tab, setTab]         = useState("text"); // "text" | "image"
  const [text, setText]       = useState("");
  const [file, setFile]       = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult]   = useState(null);
  const [error, setError]     = useState(null);

  const handleFileSelect = (f) => {
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  };

  const clearFile = () => {
    setFile(null);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(null);
    setResult(null);
  };

  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      let res;
      if (tab === "text") {
        res = await axios.post("/api/analyze/text", { text });
      } else {
        const formData = new FormData();
        formData.append("image", file);
        res = await axios.post("/api/analyze/image", formData, {
          headers: { "Content-Type": "multipart/form-data" }
        });
      }
      setResult(res.data);
    } catch (e) {
      setError(e.response?.data?.error || e.message || "Analysis failed");
    } finally {
      setLoading(false);
    }
  };

  const canAnalyze = !loading && (tab === "text" ? text.trim().length > 10 : file !== null);

  const EXAMPLES = [
    "Patient: 32yr, Bronchial Asthma. Prescribed Atenolol 50mg once daily.",
    "Patient: 68yr, AF. Warfarin 5mg OD and Ibuprofen 400mg TDS for knee pain.",
    "Patient: 62yr, Parkinson disease on Levodopa. Prescribed Metoclopramide 10mg TDS.",
    "Patient with documented penicillin allergy. Amoxicillin 500mg TDS for throat infection.",
    "Patient: 55yr, Hypertension. Prescribed Amlodipine 50mg once daily.",
    "Patient: 28yr, Allergic rhinitis. Cetirizine 10mg OD at night.",
  ];

  return (
    <div className="analyze-page">
      {/* ── Left Panel ── */}
      <div className="analyze-left">
        <div className="input-header">
          <h1 style={{color:"white"}} className="">Prescription Analysis</h1>
          <p className="page-sub">Enter a prescription note or upload an image for AI safety screening.</p>
        </div>

        {/* Tab switcher */}
        <div style={{marginTop:"1rem"}} className="tab-switcher">
          <button style={{padding:"0.4rem", borderRadius:"0.4rem 0 0 0.4rem",background:"none"}} className={`tab-btn nav-link active ${tab === "text" ? "active" : ""}`} onClick={() => { setTab("text"); setResult(null); }}>
            📝 Text Note
          </button>
          <button style={{padding:"0.4rem", borderRadius:"0 0.4rem 0.4rem 0",background:"none"}} className={`nav-link active tab-btn ${tab === "image" ? "active" : ""}`} onClick={() => { setTab("image"); setResult(null); }}>
            🖼️ Image / Scan
          </button>
        </div>

        {tab === "text" ? (
          <>
            <textarea
              value={text}
              onChange={e => setText(e.target.value)}
              placeholder="Enter prescription or clinical note here…&#10;&#10;Example: Patient: 45yr, COPD. Prescribed Metoprolol 25mg BD."
              rows={10}
              className="text-input"
            />

          </>
        ) : (
          <ImageDropzone onFile={handleFileSelect} preview={preview} onClear={clearFile} />
        )}

        <div className="action-row">
          <button
            className="btn btn-primary analyze-btn"
            onClick={handleAnalyze}
            disabled={!canAnalyze}
          >
            {loading ? <><span className="spinner" /> Analyzing…</> : "⚕ Analyze Prescription"}
          </button>
          {result && (
            <button className="btn btn-outline" onClick={() => { setResult(null); if (tab==="text") setText(""); else clearFile(); }}>
              Clear
            </button>
          )}
        </div>

        {error && (
          <div className="error-box">
            <span>⚠️</span> {error}
          </div>
        )}

        {/* Pipeline info */}
        <div className="pipeline-info card">
          <div className="section-label">Analysis Pipeline</div>
          <div className="pipeline-steps">
            {[
              ["1", "EasyOCR", "Extracts text from prescription images"],
              ["2", "BioClinicalBERT", "Extracts medications, symptoms, tests"],
              ["3", "Rule Engine", "Dose verification + brand→generic resolution"],
              ["4", "FLAN-T5 / Llama", "Clinical contradiction detection (LLM-primary)"],
            ].map(([n, name, desc]) => (
              <div key={n} className="pipeline-step">
                <span className="step-num">{n}</span>
                <div>
                  <div className="step-name">{name}</div>
                  <div className="step-desc">{desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Right Panel ── */}
      <div className="analyze-right">
        {loading && (
          <div className="loading-panel">
            <div className="spinner" style={{ width: 40, height: 40, borderWidth: 3 }} />
            <div className="loading-text">Running clinical analysis…</div>
            <div className="loading-steps">
              {tab === "image" && <div className="loading-step">📷 OCR extraction</div>}
              <div className="loading-step">🧬 NER entity extraction</div>
              <div className="loading-step">📏 Dose verification</div>
              <div className="loading-step">🤖 LLM contradiction check</div>
            </div>
          </div>
        )}

        {!loading && !result && (
          <div className="empty-right">
            <div className="empty-icon">⚕</div>
            <div className="empty-title">Analysis results appear here</div>
            <div className="empty-sub">
              Enter a prescription note or upload a prescription image to start.
            </div>
          </div>
        )}

        {!loading && result && <ResultPanel result={result} />}
      </div>
    </div>
  );
}