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

## Phase 3 — Data Health & Freshness (issue #32)

- `data_health.py` — company-aware, config-driven business-data checks:
  staleness (reusing `ReadOnlyExecutor.data_boundaries()` from before
  monitoring existed), critical-view smoke checks, and a generic
  configured-rule engine (`nonzero`, `row_count_min`, `row_count_max`,
  `null_rate_max`) that covers suspicious-zero, row-count-range, and
  NULL-rate checks as instances of one mechanism rather than three
  near-duplicate ones.
- `DataAnomaly` — a structured finding (not a free-form log string, per
  the issue's explicit requirement), field-shape-compatible with the
  master spec's future Anomaly Engine so it slots in without reshaping
  when #33/#34 arrive.
- New `CompanyConfig` fields: `critical_views: list[str]` and
  `data_health_checks: list[dict]` — both empty for Beta and Tire Guru
  in the committed `companies.yaml` today. The mechanism is fully built
  and tested; the actual business thresholds ("which views are
  critical," "which metric should never be zero") need someone who
  knows each company's real data to configure them — fabricating those
  numbers would be exactly the kind of invented certainty this project
  has consistently avoided (same principle as `check_fallback_llm`'s
  honest gap in Phase 2).
- `EXPECTED_EMPTY` vs `SUSPICIOUS_ZERO`, handled explicitly: an empty
  view result is never flagged unless a rule sets `must_have_rows` —
  the issue's explicit requirement to distinguish the two.
- 24 new tests in `tests/unit/test_data_health.py`, one per acceptance
  criterion (stale data → anomaly, broken view → anomaly, configurable
  thresholds, empty-vs-suspicious-zero distinction, evidence present on
  every anomaly) plus 3 new tests for the `CompanyConfig` fields
  themselves.

## Phase 4 — AI-Quality Anomaly Monitoring (issue #33)

- `ai_health.py` — deterministic checks over `RequestTrace` data:
  entity-resolution failures (per-occurrence, using `entity_status` —
  see below), suspicious-zero results (a *resolved* entity with an
  empty result — deliberately distinct from an unresolved entity's
  empty result, which is already covered by the entity-resolution
  check, so the same request never gets double-flagged), SQL fallback
  rate, LLM/request failure rate (reuses Phase 1's `db_error_rate_*`
  thresholds), and clarification rate (measured and reported, but never
  escalates severity — no confirmed threshold exists for it yet, same
  honest-gap pattern as Phase 2/3).
- **Wired real data into `RequestTrace` that Phase 1 had deferred**:
  `entity_status` (from `Resolution.status` — `_resolve_filters()` and
  the playbook-entity path now record every resolution outcome onto
  `self._last_entity_resolutions`) and `sql_retries` (from
  `_run_sql_path`'s actual corrective-attempt count). `entity_confidence`
  stays deliberately unpopulated — `Resolution` has no numeric
  confidence, only the categorical status, and mapping status onto an
  invented float would be exactly the kind of fabricated certainty this
  project has avoided throughout; `entity_status` carries the real
  signal instead.
- Reuses `DataAnomaly` (Phase 3) as the output shape rather than
  inventing a fourth anomaly type — `new_anomaly()` was made a public,
  shared constructor for this reason.
- 23 new tests in `tests/unit/test_ai_health.py`, one per acceptance
  criterion, plus 3 new integration tests in `tests/unit/test_orchestrator.py`
  proving `entity_status`/`sql_retries` land correctly in a real
  recorded trace (not just that the derivation logic works in
  isolation).

## Phase 5 — Unified Incident Engine (issue #34)

- `incidents.py` — converts anomalies from Phases 2/3/4 into
  deduplicated, auditable `Incident` records with a validated lifecycle.
- **Lifecycle states are deliberately `OPEN -> DIAGNOSED -> ACKNOWLEDGED
  -> CLOSED` (plus `ESCALATED`), NOT the issue's literally-worded
  `OPEN->DIAGNOSED->HEALING->VERIFIED->CLOSED`** — confirmed explicitly
  with the project owner before implementing. `HEALING`/`VERIFIED`
  implies an automated remediation step, contradicting both the master
  monitoring spec's non-goals ("Do NOT add HEALING or AUTO-RECOVERY
  states yet") and the explicit decision to exclude issues #36/#37
  (self-healing) from this work entirely. See `incidents.py`'s module
  docstring for the full reasoning; `test_no_healing_or_verified_states_exist`
  pins this structurally, not just by convention.
- Deduplication key: `(company_id, category, metric)` — the same
  underlying failure recurring collapses into one incident with an
  incrementing `occurrence_count`, never a new incident per occurrence.
  A `CLOSED` incident whose failure recurs reopens the same incident
  rather than spamming a new one.
- Every mutation goes through validated transitions (`InvalidTransition`
  raised on an illegal jump) and is recorded in the incident's own
  `history` list — auditable, not just mutable.
- `set_diagnosis()` exists as a thin, validated setter for Phase 6
  (#35) to use — `diagnosis` stays `None` until #35 is built; this
  phase never fabricates one.
- Concurrency proven: many threads reporting the same underlying
  failure simultaneously dedupe into exactly one incident (matches
  Phase 1's concurrency-safety bar).
- 26 new tests in `tests/unit/test_incidents.py`, one per acceptance
  criterion plus the dedup/reopen/concurrency edge cases.

## Phase 6 — Root-Cause Diagnosis (issue #35) — not yet built

Deferred out of strict numeric order: #38 (Phase 7, below) was built
first, ahead of #35, at the project owner's explicit request to have a
working dashboard ready for a specific demo deadline. `Incident.
diagnosis` and `IncidentEngine.set_diagnosis()` (Phase 5) already exist
specifically so #35 can be added later without reshaping anything —
advisory only, never executing tools or corrective actions, per the
master spec.

## Phase 7 — Health Control Center (issue #38)

- `app/api/v1/monitor.py` — read-only, company-scoped JSON API:
  `GET /api/v1/monitor/health` (combined system+data+AI, all companies
  or one via `?company_id=`), `/health/system`, `/health/data`,
  `/health/ai`, `/incidents`, `/incidents/{id}`, `/metrics`,
  `/traces/{request_id}`. Every check runs on-demand (pull-based) when
  an endpoint is hit — hitting `/health/data` or `/health/ai` also
  feeds any anomalies found into the shared `IncidentEngine` via
  `app.state.incident_engine`, so browsing the dashboard is what keeps
  incidents current until a real scheduler exists.
- `app/api/v1/dashboard.py` — `GET /monitor/dashboard`, one
  dependency-free HTML page (vanilla JS, no build step) that calls the
  API above and renders overall status, component breakdowns, recent
  latency/error-rate, and active incidents. Deliberately simple per the
  issue's own instruction ("Keep the UI simple... Backend correctness
  is the priority").
- **No auto-heal data anywhere** — issue #38 asks for "auto-heal
  attempts and success rate" and separating "human-required" from
  "automatically recoverable" incidents. That capability doesn't exist
  (issues #36/#37 explicitly excluded from this work). The dashboard
  says so plainly, in the UI itself, not just in a code comment —
  every incident is shown as requiring human review because that's
  true of all of them right now.
- No LLM anywhere in this phase — proven structurally in tests
  (`test_health_endpoints_never_require_an_llm`,
  `test_dashboard_does_not_import_llm_client`), not just asserted.
- 18 new **live integration tests** in
  `tests/integration/test_monitor_endpoints.py` — these spin up the
  real FastAPI app and make real HTTP requests (not just calling the
  underlying functions directly), including a request that sends a real
  `/chat` call and confirms the resulting trace is visible through
  `/monitor/metrics` — proof the whole pipeline (Phase 1 telemetry →
  Phase 7 dashboard) is actually wired together end to end.

## Phase 8+ (not yet built)

- Root-cause diagnosis (#35) — advisory only, never executes tools or
  corrective actions; consumes incidents via `set_diagnosis()`.
- Monitoring API endpoints and the simple control-center view (#38).
- Issues #36/#37 (self-healing engine, post-heal verification) are
  explicitly excluded from this work — see incidents.py's module
  docstring.

Each phase gets a real design pass against the actual data earlier
phases collect, rather than being speculatively built now.
