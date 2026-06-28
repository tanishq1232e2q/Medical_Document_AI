import React, { useState, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route, NavLink } from "react-router-dom";
import AnalyzePage from "./pages/AnalyzePage";
import ReportPage from "./pages/ReportPage";
import HistoryPage from "./pages/HistoryPage";
import "./App.css";

function App() {
  const [backendStatus, setBackendStatus] = useState(null);

  useEffect(() => {
    fetch("/api/health")
      .then(r => r.json())
      .then(d => setBackendStatus(d))
      .catch(() => setBackendStatus({ status: "offline" }));
  }, []);

  return (
    <Router>
      <div className="app">
        <nav className="navbar">
          <div className="nav-brand">
            <span className="nav-logo">⚕</span>
            <span className="nav-title">MedAI</span>
            <span className="nav-sub">Prescription Safety</span>
          </div>
          <div className="nav-links">
            <NavLink to="/" end className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>
              Analyze
            </NavLink>
            <NavLink to="/history" className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>
              History
            </NavLink>
            <NavLink to="/report" className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>
              Report
            </NavLink>
          </div>
          <div className="nav-status">
            <span className={`status-dot ${backendStatus?.status === "ok" ? "online" : "offline"}`} />
            <span className="status-text">
              {backendStatus?.status === "ok"
                ? (backendStatus.mode === "full_ml" ? "ML Active" : "Dev Mode")
                : "Backend Offline"}
            </span>
          </div>
        </nav>

        <main className="main-content">
          <Routes>
            <Route path="/" element={<AnalyzePage />} />
            <Route path="/history" element={<HistoryPage />} />
            <Route path="/report" element={<ReportPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;