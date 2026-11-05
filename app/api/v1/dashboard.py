"""GET /monitor/dashboard — a single, dependency-free HTML page for the
Health Control Center (issue #38).

Deliberately simple per the issue's own instruction ("Keep the UI
simple. Do not spend too much time on design. Backend correctness is
the priority.") — one static HTML file with vanilla JS calling the
JSON endpoints in monitor.py, no build step, no framework, nothing to
install. Everything it shows comes from those endpoints; this file has
no business logic of its own beyond fetch + render.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["monitoring"])

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Eliara — Health Control Center</title>
<style>
  :root { color-scheme: light; }
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: #f4f6fc; color: #1a1a2e; }
  header { background: #1e2761; color: #fff; padding: 16px 24px; display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 18px; margin: 0; font-weight: 600; }
  select, button { font-size: 14px; padding: 6px 10px; border-radius: 6px; border: 1px solid #ccc; }
  main { padding: 24px; max-width: 1100px; margin: 0 auto; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }
  .card { background: #fff; border-radius: 10px; padding: 16px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
  .card h3 { margin: 0 0 8px; font-size: 13px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.04em; }
  .status { font-size: 22px; font-weight: 700; }
  .HEALTHY { color: #1f9d55; } .WARNING { color: #b7791f; } .CRITICAL { color: #c0392b; }
  .OPEN { color: #c0392b; } .DIAGNOSED { color: #b7791f; } .ACKNOWLEDGED { color: #2563eb; }
  .ESCALATED { color: #7c2d12; font-weight: 700; } .CLOSED { color: #6b7280; }
  section { background: #fff; border-radius: 10px; padding: 16px 18px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
  section h2 { font-size: 14px; margin: 0 0 12px; color: #1e2761; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #eee; }
  th { color: #6b7280; font-weight: 600; font-size: 11px; text-transform: uppercase; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }
  .badge.HEALTHY { background: #dff5e5; } .badge.WARNING { background: #fdf1d6; } .badge.CRITICAL { background: #fbe2e0; }
  .badge.OPEN { background: #fbe2e0; } .badge.DIAGNOSED { background: #fdf1d6; }
  .badge.ACKNOWLEDGED { background: #dbeafe; } .badge.ESCALATED { background: #fee2c8; } .badge.CLOSED { background: #eee; }
  .muted { color: #6b7280; font-size: 12px; }
  .note { background: #fef9e7; border: 1px solid #f5e6a8; border-radius: 8px; padding: 10px 14px; font-size: 12px; color: #7a5c00; margin-bottom: 16px; }
  .empty { color: #6b7280; font-size: 13px; padding: 12px 0; }
  .refresh-info { font-size: 11px; color: #9aa5c8; margin-left: auto; }
</style>
</head>
<body>
<header>
  <h1>Eliara — Health Control Center</h1>
  <select id="companySelect"></select>
  <button onclick="loadAll()">Refresh</button>
  <span class="refresh-info" id="lastUpdated"></span>
</header>
<main>
  <div class="note">
    Auto-heal / automated remediation is <strong>not implemented</strong> in this system (see issues #36/#37).
    Every incident below requires human review — there is no "automatically recoverable" category.
  </div>

  <div class="grid" id="overviewGrid"></div>

  <section>
    <h2>Latency &amp; Error Rate</h2>
    <div id="metricsBody" class="empty">Loading…</div>
  </section>

  <section>
    <h2>System / Data / AI Health — Components</h2>
    <div id="componentsBody" class="empty">Loading…</div>
  </section>

  <section>
    <h2>Active Incidents</h2>
    <div id="incidentsBody" class="empty">Loading…</div>
  </section>
</main>

<script>
const API = "/api/v1/monitor";

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} -> HTTP ${res.status}`);
  return res.json();
}

async function loadCompanies() {
  try {
    const health = await getJSON(`${API}/health`);
    const select = document.getElementById("companySelect");
    select.innerHTML = "";
    Object.keys(health).forEach(cid => {
      const opt = document.createElement("option");
      opt.value = cid; opt.textContent = cid;
      select.appendChild(opt);
    });
    select.onchange = loadAll;
    return Object.keys(health);
  } catch (e) {
    document.getElementById("overviewGrid").innerHTML = `<div class="card"><h3>Error</h3><div class="status CRITICAL">Could not reach /monitor/health</div></div>`;
    return [];
  }
}

function statusCard(title, status) {
  return `<div class="card"><h3>${title}</h3><div class="status ${status}">${status}</div></div>`;
}

function badge(value) {
  return `<span class="badge ${value}">${value}</span>`;
}

async function loadAll() {
  const cid = document.getElementById("companySelect").value;
  if (!cid) return;

  const health = await getJSON(`${API}/health?company_id=${encodeURIComponent(cid)}`);
  const h = health[cid];

  document.getElementById("overviewGrid").innerHTML =
    statusCard("Overall", h.overall_status) +
    statusCard("System Health", h.system.overall_status) +
    statusCard("Data Health", h.data.status) +
    statusCard("AI Quality", h.ai.status);

  const metrics = await getJSON(`${API}/metrics?company_id=${encodeURIComponent(cid)}`);
  document.getElementById("metricsBody").innerHTML = `
    <table>
      <tr><th>P50 latency</th><th>P95 latency</th><th>Error rate</th><th>Sample size</th><th>Window</th></tr>
      <tr>
        <td>${metrics.latency.p50_ms ?? "—"} ms</td>
        <td>${metrics.latency.p95_ms ?? "—"} ms</td>
        <td>${metrics.error_rate.error_rate ?? "—"}</td>
        <td>${metrics.latency.sample_size}</td>
        <td>${metrics.window_minutes} min</td>
      </tr>
    </table>`;

  const rows = [];
  h.system.components.forEach(c => rows.push(["system", c.name, c.status, c.detail || JSON.stringify(c.metrics)]));
  (h.data.anomalies || []).forEach(a => rows.push(["data", a.metric, a.severity, a.category]));
  (h.ai.anomalies || []).forEach(a => rows.push(["ai", a.metric, a.severity, a.category]));
  document.getElementById("componentsBody").innerHTML = rows.length ? `
    <table>
      <tr><th>Area</th><th>Component / Metric</th><th>Status</th><th>Detail</th></tr>
      ${rows.map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${badge(r[2])}</td><td class="muted">${r[3]}</td></tr>`).join("")}
    </table>` : `<div class="empty">No components reported.</div>`;

  const incidents = await getJSON(`${API}/incidents?company_id=${encodeURIComponent(cid)}`);
  const active = incidents.filter(i => i.status !== "CLOSED");
  document.getElementById("incidentsBody").innerHTML = active.length ? `
    <table>
      <tr><th>Severity</th><th>Category</th><th>Title</th><th>Status</th><th>Occurrences</th><th>First seen</th><th>Last seen</th></tr>
      ${active.map(i => `<tr>
        <td>${badge(i.severity)}</td>
        <td>${i.category}</td>
        <td>${i.title}</td>
        <td>${badge(i.status)} <span class="muted">(human review required)</span></td>
        <td>${i.occurrence_count}</td>
        <td class="muted">${new Date(i.first_seen).toLocaleString()}</td>
        <td class="muted">${new Date(i.last_seen).toLocaleString()}</td>
      </tr>`).join("")}
    </table>` : `<div class="empty">No active incidents for this company.</div>`;

  document.getElementById("lastUpdated").textContent = "Updated " + new Date().toLocaleTimeString();
}

(async () => {
  await loadCompanies();
  await loadAll();
})();
</script>
</body>
</html>
"""


@router.get("/monitor/dashboard", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    return HTMLResponse(_DASHBOARD_HTML)
