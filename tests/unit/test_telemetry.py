"""Tests for app.monitoring — Phase 1 (observability/telemetry only).

Covers TraceRecorder's own correctness (mirroring test_audit.py's proven
pattern, since TraceRecorder deliberately mirrors AuditTrail), RequestTrace's
immutability, and — the spec's explicit requirement — that two concurrent
requests' traces, entities, results, and timings can never mix.
"""

import json
import threading
from datetime import UTC, datetime

from app.monitoring.config import MonitoringConfig, get_monitoring_config
from app.monitoring.models import RequestTrace
from app.monitoring.telemetry import TraceRecorder


def _trace(**overrides) -> RequestTrace:
    defaults = {
        "request_id": "r1",
        "timestamp": datetime.now(UTC).isoformat(),
        "company_id": "e01",
        "session_id": "s1",
        "question": "top customers?",
        "route": "use_view",
        "selected_view": "vw_q002",
        "latency_ms": 120,
        "input_tokens": 10,
        "output_tokens": 5,
    }
    defaults.update(overrides)
    return RequestTrace(**defaults)


# ------------------------------------------------------------- RequestTrace


def test_request_trace_is_frozen():
    """No shared mutable state — a trace can't be edited in place once
    built, only replaced via with_fields() into a new instance."""
    trace = _trace()
    try:
        trace.latency_ms = 999  # type: ignore[misc]
        raised = False
    except Exception:  # noqa: BLE001 - proving *any* mutation attempt is rejected, not one specific exception type
        raised = True
    assert raised


def test_with_fields_returns_a_new_instance_not_a_mutation():
    original = _trace(latency_ms=100)
    updated = original.with_fields(latency_ms=200)
    assert original.latency_ms == 100  # untouched
    assert updated.latency_ms == 200
    assert original is not updated


def test_partial_trace_has_safe_defaults():
    """A request that failed before routing completed must still be
    recordable — no required field beyond request_id/timestamp."""
    trace = RequestTrace(request_id="r1", timestamp="2026-01-01T00:00:00Z")
    assert trace.route is None
    assert trace.status == "success"
    assert trace.sql_retries == 0


# ------------------------------------------------------------- TraceRecorder


def test_writes_daily_jsonl_per_company(tmp_path):
    recorder = TraceRecorder(tmp_path)
    recorder.record(_trace(request_id="r1"))
    recorder.record(_trace(request_id="r2", route="needs_sql"))

    files = list(tmp_path.rglob("traces-*.jsonl"))
    assert len(files) == 1
    assert files[0].parent.name == "e01"
    lines = [json.loads(line) for line in files[0].read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0]["request_id"] == "r1"
    assert lines[1]["route"] == "needs_sql"


def test_company_isolated_into_separate_subdirectories(tmp_path):
    recorder = TraceRecorder(tmp_path)
    recorder.record(_trace(company_id="e01"))
    recorder.record(_trace(company_id="e02"))

    e01_files = list((tmp_path / "e01").glob("traces-*.jsonl"))
    e02_files = list((tmp_path / "e02").glob("traces-*.jsonl"))
    assert len(e01_files) == 1
    assert len(e02_files) == 1


def test_disabled_writes_nothing(tmp_path):
    recorder = TraceRecorder(tmp_path / "off", enabled=False)
    recorder.record(_trace())
    assert not (tmp_path / "off").exists()


def test_write_failure_never_raises(tmp_path):
    """Spec requirement: a monitoring gap must never become a
    request-blocking error."""
    recorder = TraceRecorder(tmp_path)
    recorder._dir = tmp_path / "does" / "not" / "exist"  # force open() failure
    recorder.record(_trace())  # must not raise


def test_no_company_id_falls_back_to_unknown_bucket(tmp_path):
    recorder = TraceRecorder(tmp_path)
    recorder.record(_trace(company_id=None))
    assert (tmp_path / "unknown").exists()


def test_read_trace_finds_a_recorded_request(tmp_path):
    recorder = TraceRecorder(tmp_path)
    recorder.record(_trace(request_id="findme", company_id="e01", route="use_view"))
    found = recorder.read_trace("e01", "findme")
    assert found is not None
    assert found["route"] == "use_view"


def test_read_trace_returns_none_for_unknown_request(tmp_path):
    recorder = TraceRecorder(tmp_path)
    recorder.record(_trace(request_id="r1", company_id="e01"))
    assert recorder.read_trace("e01", "does-not-exist") is None


def test_read_trace_returns_none_for_unknown_company(tmp_path):
    recorder = TraceRecorder(tmp_path)
    assert recorder.read_trace("no-such-company", "r1") is None


# --------------------------------------------------------- concurrency (spec)


def test_concurrent_traces_never_interleave_or_drop_fields(tmp_path):
    """The spec's explicit test requirement: two simultaneous requests must
    not mix traces. Fires many threads recording distinctly-shaped traces
    for two different companies at once, then verifies every single trace
    landed intact, in the right company's file, with no field bleeding
    from one trace into another and no line dropped or corrupted by a
    torn concurrent write."""
    recorder = TraceRecorder(tmp_path)
    n_per_company = 60

    def record_many(company_id: str, route: str):
        for i in range(n_per_company):
            recorder.record(
                _trace(
                    request_id=f"{company_id}-{i}",
                    company_id=company_id,
                    route=route,
                    selected_view=f"vw_{company_id}_{i}",
                    latency_ms=i,
                )
            )

    threads = [
        threading.Thread(target=record_many, args=("e01", "use_view")),
        threading.Thread(target=record_many, args=("e02", "needs_sql")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for company_id, route in (("e01", "use_view"), ("e02", "needs_sql")):
        files = list((tmp_path / company_id).glob("traces-*.jsonl"))
        assert len(files) == 1
        lines = [json.loads(line) for line in files[0].read_text().splitlines()]
        assert len(lines) == n_per_company  # no dropped/torn writes
        seen_ids = set()
        for entry in lines:
            assert entry["company_id"] == company_id  # no cross-company bleed
            assert entry["route"] == route  # no field corruption
            assert entry["selected_view"].startswith(f"vw_{company_id}_")
            seen_ids.add(entry["request_id"])
        assert len(seen_ids) == n_per_company  # every request distinct, none lost


def test_concurrent_writes_to_two_companies_do_not_block_each_other_incorrectly(tmp_path):
    """A coarser sanity check alongside the fine-grained test above: run
    interleaved single-record calls across companies from multiple threads
    and confirm every record is retrievable afterwards by exact id."""
    recorder = TraceRecorder(tmp_path)
    results = {}
    lock = threading.Lock()

    def worker(idx: int):
        company_id = "e01" if idx % 2 == 0 else "e02"
        request_id = f"req-{idx}"
        recorder.record(_trace(request_id=request_id, company_id=company_id))
        with lock:
            results[request_id] = company_id

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for request_id, company_id in results.items():
        found = recorder.read_trace(company_id, request_id)
        assert found is not None, f"{request_id} missing from {company_id}'s traces"
        assert found["request_id"] == request_id


# ------------------------------------------------------------------ config


def test_monitoring_config_loads_defaults():
    get_monitoring_config.cache_clear()
    config = get_monitoring_config()
    assert config.enabled is True
    assert config.p95_latency_ms_warning > 0
    assert config.p95_latency_ms_critical > config.p95_latency_ms_warning


def test_monitoring_config_respects_env_override(monkeypatch):
    get_monitoring_config.cache_clear()
    monkeypatch.setenv("ELIARA_MONITOR_ENABLED", "false")
    config = get_monitoring_config()
    assert config.enabled is False
    get_monitoring_config.cache_clear()


def test_monitoring_config_is_a_single_source_of_truth_for_thresholds():
    """Every threshold field must have a real, sane default — this test
    exists so an accidentally-added threshold field without a default
    fails loudly here rather than at first production use."""
    config = MonitoringConfig()
    for field_name in (
        "p95_latency_ms_warning", "p95_latency_ms_critical",
        "llm_timeout_rate_warning", "db_error_rate_warning",
        "stale_data_hours_warning", "generated_sql_rate_warning",
        "entity_failure_rate_warning",
    ):
        assert getattr(config, field_name) is not None
