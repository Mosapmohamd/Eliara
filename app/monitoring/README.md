# Eliara Monitoring — Phased Plan

Monitoring + diagnosis only, never self-healing — see the non-goals list
in the tracking issue. This directory is built in phases, each its own
issue/PR, so review stays manageable and each layer is proven before the
next is built on top of it.

## Phase 1 — Observability (this delivery)

- `models.py` — `RequestTrace`: one row of end-to-end telemetry per
  request, following the spec's Observability field list.
- `config.py` — `MonitoringConfig`: every threshold used anywhere in the
  monitoring layer, defined once, env-driven (`ELIARA_MONITOR_*`).
- `telemetry.py` — `TraceRecorder`: writes traces, one JSONL file per
  day per company under `storage_dir/<company_id>/` — the exact same
  proven pattern as `app/core/audit.py`'s `AuditTrail` (best-effort,
  never raises, per-company isolation by construction).
- Wired into `Orchestrator._handle()` (writes alongside the existing
  audit record), `CompanyContextManager`, and `app/main.py`'s startup.
- Concurrency proven, not assumed: `tests/unit/test_telemetry.py`
  includes a multi-threaded test recording 120 traces across two
  companies simultaneously and verifying zero cross-contamination —
  the spec's explicit requirement ("Add tests proving two simultaneous
  requests cannot mix").

### What Phase 1 deliberately does NOT capture yet

Some Observability fields from the spec aren't populated yet because the
orchestrator doesn't currently expose them as discrete, trackable values
(routing_confidence, entity_confidence, entity_type, sql_retries,
retrieval_ms, empty_result's distinction from a genuinely-zero business
result). Populating these truthfully requires small, targeted changes
inside the orchestrator's routing/entity-resolution/SQL-retry paths —
scoped as follow-up work, not invented as placeholder values now. See the
tracking issue for the specific fields deferred and why.

`endpoint` is also not populated at the orchestrator level (it doesn't
know the literal HTTP path) — a follow-up can thread it in from the API
layer if useful, or it can be joined from the same `request_id` already
in every `http_request` log line.

## Phase 2+ (not yet built)

- System health checks (`health.py`) — DB/LLM/retrieval availability,
  latency percentiles, computed from `RequestTrace` data over
  `MonitoringConfig.health_window_minutes`.
- Data health (`data_health.py`) — freshness, suspicious zero/row-count
  changes, company-aware (via a `CompanyConfig` extension once there's a
  concrete per-company override to add — not speculative fields today).
- AI quality monitoring (`ai_health.py`) — routing/entity/SQL-retry
  failure rates, computed from the same trace data.
- Anomaly engine, incident engine, root-cause diagnosis (advisory only —
  never executes tools or corrective actions), monitoring API endpoints,
  and the simple control-center view.

Each phase gets a real design pass against the actual data Phase 1
starts collecting, rather than being speculatively built now.
