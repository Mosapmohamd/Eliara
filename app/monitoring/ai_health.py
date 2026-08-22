"""AI-quality anomaly monitoring — Phase 4 of the monitoring system
(issue #33).

Detects failures where infrastructure is healthy but Eliara's routing,
entity resolution, SQL fallback, or answer pipeline behaves abnormally.
Deterministic rules over RequestTrace data (Phase 1), configurable
thresholds (Phase 1's MonitoringConfig), reusing DataAnomaly (Phase 3)
as the output shape rather than inventing a fourth anomaly type.

Every metric here is computed from fields the orchestrator now actually
populates (Phase 4 wired entity_status and sql_retries into RequestTrace
for exactly this — see app/orchestrator/orchestrator.py's telemetry
recording and app/monitoring/README.md's Phase 4 section for what
changed and why).
"""

from __future__ import annotations

from app.monitoring.config import MonitoringConfig
from app.monitoring.data_health import DataAnomaly, new_anomaly
from app.monitoring.health import CRITICAL, HEALTHY, WARNING

ROUTING_FAILURE = "ROUTING_FAILURE"
ENTITY_RESOLUTION_FAILURE = "ENTITY_RESOLUTION_FAILURE"
SUSPICIOUS_ZERO_RESULT = "SUSPICIOUS_ZERO_RESULT"
SQL_FALLBACK_RATE = "SQL_FALLBACK_RATE"
CLARIFICATION_RATE = "CLARIFICATION_RATE"
LLM_FAILURE_RATE = "LLM_FAILURE_RATE"


def _rate(count: int, total: int) -> float | None:
    return round(count / total, 4) if total else None


def check_entity_resolution_failures(company_id: str, traces: list[dict]) -> tuple[str, list[DataAnomaly]]:
    """Issue #33 acceptance criterion: a known entity-resolution failure
    is detected and classified. A single "unknown" or "ambiguous"
    entity_status is flagged per-occurrence (not just a rate), since one
    genuinely-failed lookup is itself the finding an operator wants to
    see — the rate check (below, folded into the general segmentation)
    is a separate, coarser signal."""
    anomalies = []
    for t in traces:
        status = t.get("entity_status")
        if status in ("unknown", "ambiguous"):
            anomalies.append(new_anomaly(
                company_id, ENTITY_RESOLUTION_FAILURE, WARNING, "entity_status",
                observed=status, expected="exact | resolved | no_index",
                evidence={
                    "request_id": t.get("request_id"), "entity_type": t.get("entity_type"),
                    "question": t.get("question"),
                },
            ))
    severity = WARNING if anomalies else HEALTHY
    return severity, anomalies


def check_suspicious_zero_results(company_id: str, traces: list[dict]) -> tuple[str, list[DataAnomaly]]:
    """Issue #33 acceptance criterion: entity exists + unexpected zero
    rows can trigger a suspicious-zero anomaly — distinct from
    check_entity_resolution_failures because the entity resolved FINE
    (status exact/resolved), but the query for that resolved entity
    still came back with zero rows. That combination is what's
    suspicious; a zero-row result for an unresolved/ambiguous entity is
    an entity-resolution problem, not a data-suspicious one, and is
    already covered by the check above — this function intentionally
    does not double-flag the same request.
    """
    anomalies = []
    for t in traces:
        if (
            t.get("entity_status") in ("exact", "resolved")
            and t.get("empty_result") is True
        ):
            anomalies.append(new_anomaly(
                company_id, SUSPICIOUS_ZERO_RESULT, WARNING, "empty_result",
                observed=True, expected="rows for a resolved entity",
                evidence={
                    "request_id": t.get("request_id"), "entity_type": t.get("entity_type"),
                    "route": t.get("route"), "selected_view": t.get("selected_view"),
                },
            ))
    severity = WARNING if anomalies else HEALTHY
    return severity, anomalies


def check_sql_fallback_rate(company_id: str, traces: list[dict], config: MonitoringConfig) -> tuple[str, list[DataAnomaly]]:
    """Issue #33 acceptance criterion: SQL fallback/retry rates are
    measurable over a time window. Reuses Phase 1's
    generated_sql_rate_warning threshold (no new number invented)."""
    if not traces:
        return HEALTHY, []
    fallback_count = sum(1 for t in traces if t.get("sql_generated"))
    rate = _rate(fallback_count, len(traces))
    if rate is None or rate < config.generated_sql_rate_warning:
        return HEALTHY, []
    anomaly = new_anomaly(
        company_id, SQL_FALLBACK_RATE, WARNING, "sql_fallback_rate",
        observed=rate, expected=f"< {config.generated_sql_rate_warning}",
        evidence={"sample_size": len(traces), "fallback_count": fallback_count},
    )
    return WARNING, [anomaly]


def check_clarification_rate(company_id: str, traces: list[dict]) -> tuple[str, list[DataAnomaly]]:
    """No configured threshold exists yet for clarification rate — this
    reports the measured rate as a metric (issue #33 asks for it to be
    "measurable"), but does not fabricate a threshold to flag it against.
    Always HEALTHY; the rate itself is in the evidence for whoever wants
    to look. A real threshold is a config addition once someone
    confirms what an abnormal clarification rate looks like for a given
    company — same honesty pattern as Phase 2/3's other reported gaps."""
    if not traces:
        return HEALTHY, []
    clarify_count = sum(1 for t in traces if t.get("clarification"))
    rate = _rate(clarify_count, len(traces))
    anomaly = new_anomaly(
        company_id, CLARIFICATION_RATE, HEALTHY, "clarification_rate",
        observed=rate, expected="no threshold configured yet",
        evidence={"sample_size": len(traces), "clarify_count": clarify_count},
    )
    return HEALTHY, [anomaly]


def check_llm_failure_rate(company_id: str, traces: list[dict], config: MonitoringConfig) -> tuple[str, list[DataAnomaly]]:
    """Reuses Phase 1's db_error_rate thresholds as the general
    request-failure-rate threshold — RequestTrace's status field covers
    any failure reaching that point, not narrowly DB errors; no new
    threshold invented for this specific check."""
    if not traces:
        return HEALTHY, []
    error_count = sum(1 for t in traces if t.get("status") == "error")
    rate = _rate(error_count, len(traces))
    if rate >= config.db_error_rate_critical:
        severity = CRITICAL
    elif rate >= config.db_error_rate_warning:
        severity = WARNING
    else:
        return HEALTHY, []
    anomaly = new_anomaly(
        company_id, LLM_FAILURE_RATE, severity, "request_failure_rate",
        observed=rate, expected=f"< {config.db_error_rate_warning}",
        evidence={"sample_size": len(traces), "error_count": error_count},
    )
    return severity, [anomaly]


def check_company_ai_quality(company_id: str, traces: list[dict], config: MonitoringConfig | None = None) -> tuple[str, list[DataAnomaly]]:
    """Runs every AI-quality check for one company's recent traces, each
    independently guarded — matching health.py/data_health.py's
    established per-check-guarding principle."""
    config = config or MonitoringConfig()
    all_anomalies: list[DataAnomaly] = []
    severities: list[str] = []

    def _guarded(fn):
        try:
            sev, anomalies = fn()
            severities.append(sev)
            all_anomalies.extend(anomalies)
        except Exception:  # noqa: BLE001 - one check's bug must degrade, not crash the whole pass
            severities.append(CRITICAL)

    _guarded(lambda: check_entity_resolution_failures(company_id, traces))
    _guarded(lambda: check_suspicious_zero_results(company_id, traces))
    _guarded(lambda: check_sql_fallback_rate(company_id, traces, config))
    _guarded(lambda: check_clarification_rate(company_id, traces))
    _guarded(lambda: check_llm_failure_rate(company_id, traces, config))

    overall = CRITICAL if CRITICAL in severities else (WARNING if WARNING in severities else HEALTHY)
    return overall, all_anomalies
