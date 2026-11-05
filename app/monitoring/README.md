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

## Phase 2 — System Health (issue #31)

- `health.py` — deterministic, no-LLM component checks: database
  connectivity (`SELECT 1` via the existing `run_metadata_sql`), LLM key
  presence (not a live API call — see the module docstring for why),
  retrieval index availability, cache round-trip, and latency/error-rate
  computed from Phase 1's `RequestTrace` data over
  `MonitoringConfig.health_window_minutes`.
- `HealthChecker.check_company()` runs every check independently guarded
  (one check's own bug degrades that component to CRITICAL with a detail
  message, never raises out of the whole snapshot) and returns a
  `HealthSnapshot` with per-component `HEALTHY`/`WARNING`/`CRITICAL` plus
  an aggregate `overall_status`.
- **Not wired into the live chat request path** — this is on-demand,
  pull-based computation, meant to be called from a future `/monitor/*`
  endpoint (Phase 7 / issue #38), never from anything a user's question
  travels through. That's a stronger version of "must never crash the
  request path" than a try/except around a call site would give: it's
  structurally impossible for a health-check bug to affect a chat
  response, because health.py is never invoked while handling one.
- 23 new tests, including a direct test per acceptance criterion:
  deliberately breaking each component (DB connection error, empty
  cache, empty index, missing API key) and confirming the health state
  actually changes — not just that a healthy state reports correctly.

### Known Phase 2 gap, reported honestly rather than fabricated

`check_fallback_llm()` always reports `WARNING` — there is no fallback
LLM configuration anywhere in `Settings` today (single
`orchestrator_model`/`sqlgen_model`, no fallback). Adding a real fallback
check requires that capability to exist first; this is a placeholder
that tells the truth about the gap rather than claiming `HEALTHY` for
something that isn't there.

## Phase 3+ (not yet built)

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
