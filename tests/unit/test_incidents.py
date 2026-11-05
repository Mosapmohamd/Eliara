"""Tests for app.monitoring.incidents — Phase 5 (issue #34).

Every acceptance criterion gets a direct test:
- health/anomaly events can create incidents
- repeated identical failure does not create alert spam
- incident transitions are validated and auditable
- evidence links back to originating request traces
- incident history can be queried by status/category/company/time

Also covers the explicit lifecycle-state decision: OPEN -> DIAGNOSED ->
ACKNOWLEDGED -> CLOSED -> ESCALATED, deliberately NOT the issue's
literally-worded HEALING/VERIFIED states (see incidents.py's module
docstring for why).
"""

from __future__ import annotations

import dataclasses

import pytest

from app.monitoring.incidents import (
    ACKNOWLEDGED,
    CLOSED,
    DIAGNOSED,
    ESCALATED,
    OPEN,
    WARNING,
    IncidentEngine,
    InvalidTransition,
)


@dataclasses.dataclass
class _FakeAnomaly:
    company_id: str
    category: str
    severity: str
    metric: str
    evidence: dict = dataclasses.field(default_factory=dict)


def _anomaly(**overrides):
    defaults = {
        "company_id": "e01", "category": "STALE_DATA", "severity": WARNING,
        "metric": "data_age_hours", "evidence": {"request_id": "r1"},
    }
    defaults.update(overrides)
    return _FakeAnomaly(**defaults)


# ------------------------------------------------------------------ creation


def test_anomaly_creates_an_incident():
    """Issue #34 acceptance criterion: health/anomaly events can create
    incidents."""
    engine = IncidentEngine()
    incident = engine.record_anomaly(_anomaly())
    assert incident.status == OPEN
    assert incident.category == "STALE_DATA"
    assert incident.company_id == "e01"
    assert incident.occurrence_count == 1


def test_healthy_severity_anomaly_does_not_create_an_incident():
    """Normal/informational anomalies (severity HEALTHY, e.g. ai_health's
    clarification-rate report) must not become incidents — mirrors
    "normal empty results do not automatically become incidents" one
    layer up."""
    engine = IncidentEngine()
    incidents = engine.record_anomalies([_anomaly(severity="HEALTHY")])
    assert incidents == []


def test_repeated_identical_failure_does_not_spam():
    """Issue #34 acceptance criterion, tested directly: repeated
    identical failure does not create alert spam — same
    company/category/metric collapses into ONE incident with an
    incrementing occurrence_count."""
    engine = IncidentEngine()
    first = engine.record_anomaly(_anomaly(evidence={"request_id": "r1"}))
    second = engine.record_anomaly(_anomaly(evidence={"request_id": "r2"}))
    third = engine.record_anomaly(_anomaly(evidence={"request_id": "r3"}))
    assert first.incident_id == second.incident_id == third.incident_id
    assert third.occurrence_count == 3
    assert len(engine.query()) == 1  # not three separate incidents


def test_different_category_creates_a_separate_incident():
    engine = IncidentEngine()
    a = engine.record_anomaly(_anomaly(category="STALE_DATA"))
    b = engine.record_anomaly(_anomaly(category="BROKEN_VIEW"))
    assert a.incident_id != b.incident_id
    assert len(engine.query()) == 2


def test_different_company_creates_a_separate_incident():
    engine = IncidentEngine()
    a = engine.record_anomaly(_anomaly(company_id="e01"))
    b = engine.record_anomaly(_anomaly(company_id="e02"))
    assert a.incident_id != b.incident_id


def test_occurrence_count_and_first_last_seen_maintained():
    engine = IncidentEngine()
    first = engine.record_anomaly(_anomaly())
    import time
    time.sleep(0.01)
    third = engine.record_anomaly(_anomaly())
    assert third.occurrence_count == 2
    assert third.first_seen == first.first_seen  # unchanged
    assert third.last_seen != first.first_seen  # updated


# --------------------------------------------------------------- transitions


def test_valid_transition_succeeds():
    """Issue #34 acceptance criterion: incident transitions are
    validated."""
    engine = IncidentEngine()
    incident = engine.record_anomaly(_anomaly())
    updated = engine.transition(incident.incident_id, DIAGNOSED)
    assert updated.status == DIAGNOSED


def test_invalid_transition_is_rejected():
    """Issue #34 acceptance criterion, tested directly: incident
    transitions are validated — an invalid jump must raise, not
    silently succeed."""
    engine = IncidentEngine()
    incident = engine.record_anomaly(_anomaly())
    engine.transition(incident.incident_id, CLOSED)
    with pytest.raises(InvalidTransition):
        engine.transition(incident.incident_id, DIAGNOSED)  # CLOSED can only go to OPEN


def test_transitions_are_auditable_via_history():
    """Issue #34 acceptance criterion: transitions are auditable."""
    engine = IncidentEngine()
    incident = engine.record_anomaly(_anomaly())
    engine.transition(incident.incident_id, DIAGNOSED, reason="root cause found")
    engine.transition(incident.incident_id, ACKNOWLEDGED, reason="owner assigned")
    assert len(incident.history) == 2
    assert incident.history[0]["from_status"] == OPEN
    assert incident.history[0]["to_status"] == DIAGNOSED
    assert incident.history[0]["reason"] == "root cause found"
    assert incident.history[1]["to_status"] == ACKNOWLEDGED


def test_escalated_is_a_valid_state():
    engine = IncidentEngine()
    incident = engine.record_anomaly(_anomaly())
    updated = engine.transition(incident.incident_id, ESCALATED)
    assert updated.status == ESCALATED


def test_closed_incident_reopens_on_recurrence_instead_of_spamming():
    """A CLOSED incident whose underlying failure recurs must reopen the
    SAME incident (bumping occurrence_count), not create a new one —
    the anti-spam requirement applied across the CLOSED boundary too."""
    engine = IncidentEngine()
    incident = engine.record_anomaly(_anomaly())
    engine.transition(incident.incident_id, CLOSED)
    recurrence = engine.record_anomaly(_anomaly())
    assert recurrence.incident_id == incident.incident_id
    assert recurrence.status == OPEN
    assert recurrence.occurrence_count == 2
    assert len(engine.query()) == 1


def test_no_healing_or_verified_states_exist():
    """Explicit design decision: the issue's literal
    OPEN->DIAGNOSED->HEALING->VERIFIED->CLOSED lifecycle is NOT
    implemented — HEALING/VERIFIED would imply automated remediation,
    contradicting the master spec's non-goals and the explicit decision
    to exclude #36/#37. Structural proof: these names aren't even
    importable constants."""
    import app.monitoring.incidents as incidents_module

    assert not hasattr(incidents_module, "HEALING")
    assert not hasattr(incidents_module, "VERIFIED")


def test_unknown_incident_id_raises_keyerror():
    engine = IncidentEngine()
    with pytest.raises(KeyError):
        engine.transition("does-not-exist", CLOSED)


# ------------------------------------------------------------------ evidence


def test_evidence_links_back_to_request_ids():
    """Issue #34 acceptance criterion: evidence links back to
    originating request traces."""
    engine = IncidentEngine()
    engine.record_anomaly(_anomaly(evidence={"request_id": "req-abc"}))
    incident = engine.record_anomaly(_anomaly(evidence={"request_id": "req-xyz"}))
    assert "req-abc" in incident.request_ids
    assert "req-xyz" in incident.request_ids
    assert len(incident.evidence) == 2


def test_evidence_is_capped_per_incident():
    """A hot-looping failure shouldn't grow one incident's evidence list
    unboundedly."""
    engine = IncidentEngine()
    for i in range(50):
        engine.record_anomaly(_anomaly(evidence={"request_id": f"r{i}"}))
    incident = engine.query()[0]
    assert len(incident.evidence) <= 20


# ------------------------------------------------------------------- diagnosis


def test_set_diagnosis_updates_incident_and_transitions_to_diagnosed():
    engine = IncidentEngine()
    incident = engine.record_anomaly(_anomaly())
    updated = engine.set_diagnosis(incident.incident_id, {
        "category": "DATA_ERROR", "probable_cause": "upstream refresh job failed",
        "confidence": 0.7, "requires_human_review": True,
    })
    assert updated.diagnosis["probable_cause"] == "upstream refresh job failed"
    assert updated.status == DIAGNOSED


def test_diagnosis_defaults_to_none():
    """Confirms Phase 5 never fabricates a diagnosis — that's Phase 6's
    (#35) job entirely; this incident engine only stores what it's
    given."""
    engine = IncidentEngine()
    incident = engine.record_anomaly(_anomaly())
    assert incident.diagnosis is None


# ---------------------------------------------------------------------- query


def test_query_by_company():
    engine = IncidentEngine()
    engine.record_anomaly(_anomaly(company_id="e01"))
    engine.record_anomaly(_anomaly(company_id="e02"))
    assert len(engine.query(company_id="e01")) == 1


def test_query_by_category():
    engine = IncidentEngine()
    engine.record_anomaly(_anomaly(category="STALE_DATA"))
    engine.record_anomaly(_anomaly(category="BROKEN_VIEW"))
    assert len(engine.query(category="BROKEN_VIEW")) == 1


def test_query_by_status():
    engine = IncidentEngine()
    a = engine.record_anomaly(_anomaly())
    engine.record_anomaly(_anomaly(category="BROKEN_VIEW"))
    engine.transition(a.incident_id, CLOSED)
    assert len(engine.query(status=CLOSED)) == 1
    assert len(engine.query(status=OPEN)) == 1


def test_query_by_time():
    import time

    engine = IncidentEngine()
    engine.record_anomaly(_anomaly())
    cutoff = __import__("datetime").datetime.now(__import__("datetime").UTC).isoformat()
    time.sleep(0.01)
    engine.record_anomaly(_anomaly(category="BROKEN_VIEW"))
    recent = engine.query(since=cutoff)
    assert len(recent) == 1


def test_query_with_no_filters_returns_everything_sorted_by_recency():
    engine = IncidentEngine()
    engine.record_anomaly(_anomaly(category="A"))
    engine.record_anomaly(_anomaly(category="B"))
    results = engine.query()
    assert len(results) == 2
    assert results[0].last_seen >= results[1].last_seen


# --------------------------------------------------------------- persistence


def test_incidents_persist_to_disk_per_company(tmp_path):
    engine = IncidentEngine(storage_dir=tmp_path)
    incident = engine.record_anomaly(_anomaly(company_id="e01"))
    files = list((tmp_path / "e01").glob("*.json"))
    assert len(files) == 1
    assert files[0].stem == incident.incident_id


def test_disabled_engine_writes_nothing(tmp_path):
    engine = IncidentEngine(storage_dir=tmp_path / "off", enabled=False)
    engine.record_anomaly(_anomaly())
    assert not (tmp_path / "off").exists()


def test_persist_failure_never_raises(tmp_path):
    engine = IncidentEngine(storage_dir=tmp_path)
    engine._dir = tmp_path / "does" / "not" / "exist" / "cant" / "mkdir" / "\x00bad"
    engine.record_anomaly(_anomaly())  # must not raise


# --------------------------------------------------------------- concurrency


def test_concurrent_anomalies_for_same_key_do_not_create_duplicate_incidents():
    """The spec's general concurrency-safety principle, applied here:
    many threads reporting the same underlying failure simultaneously
    must dedupe into exactly one incident, not race into several."""
    import threading

    engine = IncidentEngine()

    def worker(i):
        engine.record_anomaly(_anomaly(evidence={"request_id": f"r{i}"}))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    incidents = engine.query()
    assert len(incidents) == 1
    assert incidents[0].occurrence_count == 50
