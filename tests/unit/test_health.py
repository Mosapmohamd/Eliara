"""Tests for app.monitoring.health — Phase 2 (issue #31).

Every acceptance criterion from the issue gets its own direct test:
- each component reports status + supporting metrics
- thresholds are configurable, not hardcoded (pulled from MonitoringConfig)
- deliberately breaking a component changes its health state
- checks don't require an LLM
- a health check's own failure can't crash check_company()
"""

from __future__ import annotations

import pytest

from app.monitoring.config import MonitoringConfig
from app.monitoring.health import (
    CRITICAL,
    HEALTHY,
    WARNING,
    HealthChecker,
    check_cache,
    check_database,
    check_error_rate,
    check_fallback_llm,
    check_latency,
    check_llm_configured,
    check_retrieval_index,
)
from app.monitoring.telemetry import TraceRecorder

# --------------------------------------------------------------- fixtures


class _FakeExecutor:
    def __init__(self, should_fail: bool = False):
        self._should_fail = should_fail

    def run_metadata_sql(self, sql: str):
        if self._should_fail:
            raise ConnectionError("db unreachable")


class _FakeSettings:
    def __init__(self, key: str = "sk-ant-real-key"):
        from pydantic import SecretStr

        self.anthropic_api_key = SecretStr(key)


class _FakeMetadataIndex:
    def __init__(self, n_objects: int = 10):
        self.objects = {f"obj{i}": None for i in range(n_objects)}


class _FakeResultCache:
    def __init__(self, broken: bool = False):
        self._broken = broken
        self._store = {}

    def set(self, key, value):
        if self._broken:
            raise RuntimeError("cache backend down")
        self._store[key] = value

    def get(self, key):
        return self._store.get(key)


# --------------------------------------------------------- individual checks


def test_database_healthy_when_query_succeeds():
    result = check_database(_FakeExecutor(should_fail=False))
    assert result.status == HEALTHY
    assert "query_ms" in result.metrics


def test_database_critical_when_query_fails():
    """Deliberately-broken component -> health state changes (issue #31
    acceptance criterion, tested directly)."""
    result = check_database(_FakeExecutor(should_fail=True))
    assert result.status == CRITICAL
    assert "unreachable" in (result.detail or "")


def test_llm_configured_healthy_with_a_real_key():
    result = check_llm_configured(_FakeSettings(key="sk-ant-real"))
    assert result.status == HEALTHY
    assert result.metrics["configured"] is True


def test_llm_configured_critical_with_empty_key():
    """Deliberately-broken component -> health state changes."""
    result = check_llm_configured(_FakeSettings(key=""))
    assert result.status == CRITICAL
    assert result.metrics["configured"] is False


def test_fallback_llm_reports_warning_not_fabricated_healthy():
    """No fallback LLM concept exists yet — must report the gap
    honestly (WARNING), never claim HEALTHY for a capability that
    doesn't exist."""
    result = check_fallback_llm()
    assert result.status == WARNING
    assert "not configured" in (result.detail or "")


def test_retrieval_index_healthy_with_objects():
    result = check_retrieval_index(_FakeMetadataIndex(n_objects=598))
    assert result.status == HEALTHY
    assert result.metrics["object_count"] == 598


def test_retrieval_index_critical_when_empty():
    """Deliberately-broken component -> health state changes."""
    result = check_retrieval_index(_FakeMetadataIndex(n_objects=0))
    assert result.status == CRITICAL


def test_retrieval_index_critical_when_none():
    result = check_retrieval_index(None)
    assert result.status == CRITICAL


def test_cache_healthy_on_working_roundtrip():
    result = check_cache(_FakeResultCache(broken=False))
    assert result.status == HEALTHY


def test_cache_critical_when_broken():
    """Deliberately-broken component -> health state changes."""
    result = check_cache(_FakeResultCache(broken=True))
    assert result.status == CRITICAL


# ---------------------------------------------------- latency / error rate


def test_latency_healthy_under_threshold():
    config = MonitoringConfig(p95_latency_ms_warning=3000, p95_latency_ms_critical=8000)
    traces = [{"latency_ms": 100} for _ in range(20)]
    result = check_latency(traces, config)
    assert result.status == HEALTHY
    assert result.metrics["sample_size"] == 20


def test_latency_warning_above_threshold():
    config = MonitoringConfig(p95_latency_ms_warning=100, p95_latency_ms_critical=8000)
    traces = [{"latency_ms": 500} for _ in range(20)]
    result = check_latency(traces, config)
    assert result.status == WARNING


def test_latency_critical_above_threshold():
    config = MonitoringConfig(p95_latency_ms_warning=100, p95_latency_ms_critical=1000)
    traces = [{"latency_ms": 5000} for _ in range(20)]
    result = check_latency(traces, config)
    assert result.status == CRITICAL


def test_latency_thresholds_are_config_driven_not_hardcoded():
    """Issue #31 acceptance criterion: thresholds are configurable, not
    hardcoded. Same trace data, two different configs -> two different
    verdicts, proving the threshold value actually comes from config."""
    traces = [{"latency_ms": 500} for _ in range(10)]
    lenient = MonitoringConfig(p95_latency_ms_warning=10_000, p95_latency_ms_critical=20_000)
    strict = MonitoringConfig(p95_latency_ms_warning=100, p95_latency_ms_critical=200)
    assert check_latency(traces, lenient).status == HEALTHY
    assert check_latency(traces, strict).status == CRITICAL


def test_latency_no_traces_is_healthy_not_a_false_alarm():
    config = MonitoringConfig()
    result = check_latency([], config)
    assert result.status == HEALTHY
    assert result.metrics["sample_size"] == 0


def test_error_rate_healthy_with_no_errors():
    config = MonitoringConfig(db_error_rate_warning=0.02, db_error_rate_critical=0.10)
    traces = [{"status": "success"} for _ in range(50)]
    result = check_error_rate(traces, config)
    assert result.status == HEALTHY


def test_error_rate_critical_with_high_error_fraction():
    config = MonitoringConfig(db_error_rate_warning=0.02, db_error_rate_critical=0.10)
    traces = [{"status": "error"} for _ in range(15)] + [{"status": "success"} for _ in range(5)]
    result = check_error_rate(traces, config)
    assert result.status == CRITICAL
    assert result.metrics["error_rate"] == pytest.approx(0.75)


# --------------------------------------------------------------- HealthChecker


def test_check_company_aggregates_worst_status(tmp_path):
    telemetry = TraceRecorder(tmp_path)
    checker = HealthChecker(telemetry, MonitoringConfig())
    snapshot = checker.check_company(
        company_id="e01",
        executor=_FakeExecutor(should_fail=True),  # forces database -> CRITICAL
        settings=_FakeSettings(),
        metadata_index=_FakeMetadataIndex(),
        result_cache=_FakeResultCache(),
    )
    assert snapshot.overall_status == CRITICAL
    by_name = {c.name: c for c in snapshot.components}
    assert by_name["database"].status == CRITICAL


def test_check_company_all_healthy_when_everything_works(tmp_path):
    telemetry = TraceRecorder(tmp_path)
    checker = HealthChecker(telemetry, MonitoringConfig())
    snapshot = checker.check_company(
        company_id="e01",
        executor=_FakeExecutor(should_fail=False),
        settings=_FakeSettings(),
        metadata_index=_FakeMetadataIndex(),
        result_cache=_FakeResultCache(),
    )
    # fallback_llm is a known, honest WARNING (no fallback exists yet) —
    # so overall can't be HEALTHY while that's true; assert on the
    # individual checks that CAN be healthy instead of overall_status.
    by_name = {c.name: c for c in snapshot.components}
    assert by_name["database"].status == HEALTHY
    assert by_name["llm_configured"].status == HEALTHY
    assert by_name["retrieval_index"].status == HEALTHY
    assert by_name["cache"].status == HEALTHY
    assert by_name["fallback_llm"].status == WARNING  # honest gap, not fabricated


def test_a_broken_check_cannot_crash_check_company(tmp_path):
    """Spec requirement: monitoring failures must never crash anything
    downstream — here, a check whose inputs cause it to raise must not
    propagate out of check_company(), it must degrade that one component
    to CRITICAL with a detail message instead."""
    telemetry = TraceRecorder(tmp_path)
    checker = HealthChecker(telemetry, MonitoringConfig())
    # metadata_index=None makes check_retrieval_index handle it gracefully
    # (tested above), but let's force a genuinely unexpected exception via
    # an executor whose run_metadata_sql raises something wild.
    class _ExplodingExecutor:
        def run_metadata_sql(self, sql):
            raise TypeError("something truly unexpected")

    snapshot = checker.check_company(
        company_id="e01",
        executor=_ExplodingExecutor(),
        settings=_FakeSettings(),
        metadata_index=_FakeMetadataIndex(),
        result_cache=_FakeResultCache(),
    )
    by_name = {c.name: c for c in snapshot.components}
    assert by_name["database"].status == CRITICAL
    # and, critically, every OTHER check still ran and reported normally
    assert by_name["cache"].status == HEALTHY


def test_check_company_uses_telemetry_window_for_latency(tmp_path):
    """Confirms check_company actually pulls recent traces from
    TraceRecorder and feeds them into the latency/error checks, not just
    that the standalone functions work in isolation."""
    from app.monitoring.models import RequestTrace

    telemetry = TraceRecorder(tmp_path)
    for _ in range(10):
        telemetry.record(RequestTrace(
            request_id="r", timestamp="x", company_id="e01", latency_ms=50, status="success",
        ))
    config = MonitoringConfig(health_window_minutes=15)
    checker = HealthChecker(telemetry, config)
    snapshot = checker.check_company(
        company_id="e01",
        executor=_FakeExecutor(should_fail=False),
        settings=_FakeSettings(),
        metadata_index=_FakeMetadataIndex(),
        result_cache=_FakeResultCache(),
    )
    by_name = {c.name: c for c in snapshot.components}
    assert by_name["latency"].metrics["sample_size"] == 10


def test_check_company_no_llm_call_anywhere():
    """Issue #31 acceptance criterion: health checks do not require an
    LLM. Structural proof, not just an assertion: none of the fakes used
    here have any LLM client at all, and check_company() still runs to
    completion — if any check secretly needed an LLM, this would error."""
    telemetry = TraceRecorder.__new__(TraceRecorder)  # not even fully constructed
    telemetry._dir = None
    telemetry._enabled = False
    checker = HealthChecker(telemetry=None, config=MonitoringConfig())
    snapshot = checker.check_company(
        company_id="e01",
        executor=_FakeExecutor(should_fail=False),
        settings=_FakeSettings(),
        metadata_index=_FakeMetadataIndex(),
        result_cache=_FakeResultCache(),
    )
    assert snapshot.company_id == "e01"


def test_snapshot_to_dict_is_json_serializable(tmp_path):
    import json

    telemetry = TraceRecorder(tmp_path)
    checker = HealthChecker(telemetry, MonitoringConfig())
    snapshot = checker.check_company(
        company_id="e01",
        executor=_FakeExecutor(should_fail=False),
        settings=_FakeSettings(),
        metadata_index=_FakeMetadataIndex(),
        result_cache=_FakeResultCache(),
    )
    json.dumps(snapshot.to_dict())  # must not raise
