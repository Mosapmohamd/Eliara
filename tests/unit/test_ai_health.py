"""Tests for app.monitoring.ai_health — Phase 4 (issue #33).

Every acceptance criterion gets a direct test:
- known entity-resolution failure -> detected and classified
- entity exists + unexpected zero rows -> suspicious-zero anomaly
- SQL fallback/retry rates measurable over a time window
- metrics segmentable by company/route/view (via the trace fields
  already present, exercised directly)
- normal empty results do not automatically become incidents
"""

from __future__ import annotations

from app.monitoring.ai_health import (
    CLARIFICATION_RATE,
    ENTITY_RESOLUTION_FAILURE,
    LLM_FAILURE_RATE,
    SQL_FALLBACK_RATE,
    SUSPICIOUS_ZERO_RESULT,
    check_clarification_rate,
    check_company_ai_quality,
    check_entity_resolution_failures,
    check_llm_failure_rate,
    check_sql_fallback_rate,
    check_suspicious_zero_results,
)
from app.monitoring.config import MonitoringConfig
from app.monitoring.health import CRITICAL, HEALTHY, WARNING


def _trace(**overrides) -> dict:
    defaults = {
        "request_id": "r1", "company_id": "e01", "route": "use_view",
        "selected_view": "vw_q002", "entity_status": None, "entity_type": None,
        "empty_result": False, "sql_generated": False, "clarification": False,
        "status": "success",
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------- entity resolution failures


def test_no_traces_is_healthy():
    severity, anomalies = check_entity_resolution_failures("e01", [])
    assert severity == HEALTHY
    assert anomalies == []


def test_unknown_entity_status_is_flagged():
    """Issue #33 acceptance criterion, tested directly: a known
    entity-resolution failure is detected and classified."""
    traces = [_trace(entity_status="unknown", entity_type="customer_name")]
    severity, anomalies = check_entity_resolution_failures("e01", traces)
    assert severity == WARNING
    assert len(anomalies) == 1
    assert anomalies[0].category == ENTITY_RESOLUTION_FAILURE
    assert anomalies[0].observed_value == "unknown"


def test_ambiguous_entity_status_is_flagged():
    traces = [_trace(entity_status="ambiguous")]
    severity, anomalies = check_entity_resolution_failures("e01", traces)
    assert severity == WARNING
    assert anomalies[0].observed_value == "ambiguous"


def test_resolved_entity_status_is_not_flagged():
    traces = [_trace(entity_status="resolved"), _trace(entity_status="exact")]
    severity, anomalies = check_entity_resolution_failures("e01", traces)
    assert severity == HEALTHY
    assert anomalies == []


def test_no_index_status_is_not_flagged_as_a_failure():
    """no_index means "nothing to resolve against" — a degraded-but-
    usable state (Resolution.usable includes it), not a failure."""
    traces = [_trace(entity_status="no_index")]
    severity, _anomalies = check_entity_resolution_failures("e01", traces)
    assert severity == HEALTHY


def test_entity_failure_evidence_includes_request_id():
    traces = [_trace(request_id="specific-req-42", entity_status="unknown")]
    _, anomalies = check_entity_resolution_failures("e01", traces)
    assert anomalies[0].evidence["request_id"] == "specific-req-42"


# ---------------------------------------------------------- suspicious zero


def test_resolved_entity_with_zero_rows_is_suspicious():
    """Issue #33 acceptance criterion, tested directly: entity exists +
    unexpected zero rows triggers a suspicious-zero anomaly."""
    traces = [_trace(entity_status="resolved", empty_result=True)]
    severity, anomalies = check_suspicious_zero_results("e01", traces)
    assert severity == WARNING
    assert anomalies[0].category == SUSPICIOUS_ZERO_RESULT


def test_exact_entity_with_zero_rows_is_suspicious():
    traces = [_trace(entity_status="exact", empty_result=True)]
    severity, _anomalies = check_suspicious_zero_results("e01", traces)
    assert severity == WARNING


def test_unresolved_entity_with_zero_rows_is_not_double_flagged_here():
    """An ambiguous/unknown entity with zero rows is an entity-
    resolution problem, already covered by
    check_entity_resolution_failures — this function must not also
    flag it, or the same request produces two overlapping anomalies."""
    traces = [_trace(entity_status="ambiguous", empty_result=True)]
    severity, anomalies = check_suspicious_zero_results("e01", traces)
    assert severity == HEALTHY
    assert anomalies == []


def test_normal_empty_result_without_any_entity_is_not_flagged():
    """Issue #33 acceptance criterion, tested directly: normal empty
    results do not automatically become incidents. A general question
    with no entity involved (entity_status None) that legitimately
    returns zero rows must not be flagged."""
    traces = [_trace(entity_status=None, empty_result=True)]
    severity, anomalies = check_suspicious_zero_results("e01", traces)
    assert severity == HEALTHY
    assert anomalies == []


def test_resolved_entity_with_actual_rows_is_not_flagged():
    traces = [_trace(entity_status="resolved", empty_result=False)]
    severity, anomalies = check_suspicious_zero_results("e01", traces)
    assert severity == HEALTHY
    assert anomalies == []


# ------------------------------------------------------------- SQL fallback rate


def test_sql_fallback_rate_measurable_over_a_window():
    """Issue #33 acceptance criterion: SQL fallback/retry rates are
    measurable over a time window."""
    traces = [_trace(sql_generated=True) for _ in range(3)] + [_trace(sql_generated=False) for _ in range(7)]
    config = MonitoringConfig(generated_sql_rate_warning=0.30)
    severity, anomalies = check_sql_fallback_rate("e01", traces, config)
    assert severity == WARNING
    assert anomalies[0].observed_value == 0.3


def test_sql_fallback_rate_below_threshold_is_healthy():
    traces = [_trace(sql_generated=True) for _ in range(1)] + [_trace(sql_generated=False) for _ in range(9)]
    config = MonitoringConfig(generated_sql_rate_warning=0.30)
    severity, anomalies = check_sql_fallback_rate("e01", traces, config)
    assert severity == HEALTHY
    assert anomalies == []


def test_sql_fallback_rate_threshold_is_configurable():
    traces = [_trace(sql_generated=True) for _ in range(3)] + [_trace(sql_generated=False) for _ in range(7)]
    lenient = MonitoringConfig(generated_sql_rate_warning=0.90)
    strict = MonitoringConfig(generated_sql_rate_warning=0.10)
    assert check_sql_fallback_rate("e01", traces, lenient)[0] == HEALTHY
    assert check_sql_fallback_rate("e01", traces, strict)[0] == WARNING


def test_sql_fallback_rate_category_is_correct():
    traces = [_trace(sql_generated=True) for _ in range(5)]
    config = MonitoringConfig(generated_sql_rate_warning=0.30)
    _, anomalies = check_sql_fallback_rate("e01", traces, config)
    assert anomalies[0].category == SQL_FALLBACK_RATE


# ------------------------------------------------------------- clarification rate


def test_clarification_rate_is_measured_but_never_flags_without_a_threshold():
    """No configured threshold exists for this yet — reports the rate as
    a metric, never escalates severity (honest gap, not fabricated)."""
    traces = [_trace(clarification=True) for _ in range(5)] + [_trace(clarification=False) for _ in range(5)]
    severity, anomalies = check_clarification_rate("e01", traces)
    assert severity == HEALTHY
    assert anomalies[0].category == CLARIFICATION_RATE
    assert anomalies[0].observed_value == 0.5
    assert anomalies[0].severity == HEALTHY


def test_clarification_rate_empty_traces():
    severity, anomalies = check_clarification_rate("e01", [])
    assert severity == HEALTHY
    assert anomalies == []


# ------------------------------------------------------------- LLM/request failure rate


def test_llm_failure_rate_measurable_and_thresholded():
    traces = [_trace(status="error") for _ in range(2)] + [_trace(status="success") for _ in range(8)]
    config = MonitoringConfig(db_error_rate_warning=0.02, db_error_rate_critical=0.10)
    severity, anomalies = check_llm_failure_rate("e01", traces, config)
    assert severity == CRITICAL
    assert anomalies[0].category == LLM_FAILURE_RATE


def test_llm_failure_rate_healthy_with_no_errors():
    traces = [_trace(status="success") for _ in range(10)]
    config = MonitoringConfig()
    severity, anomalies = check_llm_failure_rate("e01", traces, config)
    assert severity == HEALTHY
    assert anomalies == []


# ------------------------------------------------------------------ segmentation


def test_metrics_are_segmentable_by_company_route_view():
    """Issue #33 acceptance criterion: metrics can be segmented by
    company/route/view where relevant — exercised directly by filtering
    the trace list before passing it in, since every trace already
    carries company_id/route/selected_view (Phase 1 fields)."""
    all_traces = [
        _trace(company_id="e01", route="use_view", selected_view="vw_a", sql_generated=True),
        _trace(company_id="e01", route="needs_sql", selected_view=None, sql_generated=True),
        _trace(company_id="e02", route="use_view", selected_view="vw_b", sql_generated=False),
    ]
    e01_only = [t for t in all_traces if t["company_id"] == "e01"]
    e02_only = [t for t in all_traces if t["company_id"] == "e02"]
    config = MonitoringConfig(generated_sql_rate_warning=0.5)
    e01_severity, _ = check_sql_fallback_rate("e01", e01_only, config)
    e02_severity, _ = check_sql_fallback_rate("e02", e02_only, config)
    assert e01_severity == WARNING  # 2/2 generated
    assert e02_severity == HEALTHY  # 0/1 generated


# ------------------------------------------------------------------ combined


def test_check_company_ai_quality_aggregates_worst_status():
    traces = [_trace(status="error") for _ in range(5)] + [_trace(status="success") for _ in range(5)]
    config = MonitoringConfig(db_error_rate_warning=0.02, db_error_rate_critical=0.10)
    severity, anomalies = check_company_ai_quality("e01", traces, config)
    assert severity == CRITICAL
    categories = {a.category for a in anomalies}
    assert LLM_FAILURE_RATE in categories


def test_check_company_ai_quality_all_clean_is_healthy():
    traces = [_trace(entity_status="resolved", empty_result=False, sql_generated=False, status="success") for _ in range(10)]
    config = MonitoringConfig()
    severity, anomalies = check_company_ai_quality("e01", traces, config)
    # clarification_rate always emits an informational anomaly (HEALTHY
    # severity) — overall must still be HEALTHY since nothing else fired.
    assert severity == HEALTHY
    assert all(a.severity == HEALTHY for a in anomalies)


def test_check_company_ai_quality_with_no_traces_is_healthy():
    severity, _anomalies = check_company_ai_quality("e01", [], MonitoringConfig())
    assert severity == HEALTHY
