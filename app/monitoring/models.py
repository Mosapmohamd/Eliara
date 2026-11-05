"""Data models for the Eliara monitoring layer.

Phase 1 (this file): request-level telemetry only. Health/anomaly/incident
models arrive in later phases — see monitoring/README.md for the phased
plan. Kept in their own file per phase rather than one giant models.py, so
each phase's schema changes are reviewable independently.

RequestTrace is a plain frozen dataclass, not pydantic — traces are built
incrementally across a request's lifecycle from data the orchestrator
already computed and trusts (not parsed from untrusted input), so
pydantic's validation cost buys nothing here. Frozen + `dataclasses.replace`
gives the same "no shared mutable state" safety pydantic would, at lower
construction cost on the hot path.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class RequestTrace:
    """One row of end-to-end telemetry for a single Eliara request.

    Field selection follows the monitoring spec's Observability section
    directly. Every field has a safe default so a partially-populated
    trace (e.g. a request that failed before routing completed) can still
    be recorded — a monitoring gap must never become a request-blocking
    error.

    Built incrementally via `with_fields()`, which returns a NEW instance
    rather than mutating in place — a reference held earlier in a
    request's lifecycle can't be surprised by a later stage's update. One
    instance lives per request (held in a local variable, never a
    module-level singleton — see monitoring/telemetry.py for how the
    request lifecycle threads it through).
    """

    request_id: str
    timestamp: str  # ISO 8601 UTC, set at creation
    company_id: str | None = None
    session_id: str | None = None
    endpoint: str | None = None
    question: str | None = None

    # Routing
    route: str | None = None  # ChatOutcome.decision: use_view / needs_sql / ...
    routing_confidence: float | None = None
    selected_view: str | None = None

    # Entity resolution
    entity_type: str | None = None
    entity_confidence: float | None = None
    entity_resolved: bool | None = None

    # SQL
    sql_generated: bool = False
    sql_retries: int = 0

    # Timing (ms) — names match ChatOutcome's existing fields so mapping
    # an outcome onto a trace is a direct field-for-field copy.
    routing_ms: int = 0
    execution_ms: int = 0
    answer_ms: int = 0
    latency_ms: int = 0
    retrieval_ms: int | None = None

    # LLM
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0

    # Result shape
    row_count: int | None = None
    empty_result: bool | None = None
    clarification: bool = False

    # Outcome
    status: str = "success"  # "success" | "error"
    error_type: str | None = None
    data_as_of: str | None = None

    def with_fields(self, **updates) -> RequestTrace:
        return dataclasses.replace(self, **updates)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
