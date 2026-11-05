"""Tests for app.monitoring.data_health — Phase 3 (issue #32).

Every acceptance criterion gets a direct test:
- stale test data -> a data-health anomaly
- a broken/unqueryable critical view -> an anomaly
- thresholds/check definitions are configurable per company
- legitimate empty results are distinguished from configured suspicious-zero
- anomaly output includes evidence
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from app.execution.executor import ReadOnlyExecutor
from app.monitoring.config import MonitoringConfig
from app.monitoring.data_health import (
    BROKEN_VIEW,
    NULL_RATE_ANOMALY,
    ROW_COUNT_ANOMALY,
    STALE_DATA,
    SUSPICIOUS_ZERO,
    check_company_data_health,
    check_critical_views,
    check_staleness,
    check_value_rules,
)
from app.monitoring.health import CRITICAL, HEALTHY, WARNING


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "data_health.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE facts(id INTEGER PRIMARY KEY, posting_date TEXT, revenue REAL, cost REAL)"
    )
    conn.execute("CREATE VIEW vw_healthy AS SELECT * FROM facts")
    conn.execute("CREATE VIEW vw_broken AS SELECT nonexistent_column FROM facts")
    conn.execute("CREATE VIEW vw_empty AS SELECT * FROM facts WHERE 1=0")
    conn.execute(
        "CREATE VIEW vw_all_zero_revenue AS "
        "SELECT id, revenue, cost FROM facts WHERE revenue = 0"
    )
    conn.commit()
    conn.close()
    ex = ReadOnlyExecutor(path)
    yield ex
    ex.close()


def _insert_row(db_path, posting_date, revenue=100.0, cost=None):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO facts(posting_date, revenue, cost) VALUES (?, ?, ?)",
        (posting_date, revenue, cost),
    )
    conn.commit()
    conn.close()


# --------------------------------------------------------------- staleness


def test_fresh_data_is_healthy(db, tmp_path):
    _insert_row(tmp_path / "data_health.db", datetime.now(UTC).isoformat())
    config = MonitoringConfig(stale_data_hours_warning=48, stale_data_hours_critical=168)
    severity, anomalies = check_staleness("e01", db, "facts", "posting_date", config)
    assert severity == HEALTHY
    assert anomalies == []


def test_stale_data_produces_an_anomaly(db, tmp_path):
    """Issue #32 acceptance criterion, tested directly: stale test data
    produces a data-health anomaly."""
    old_date = (datetime.now(UTC) - timedelta(hours=200)).isoformat()
    _insert_row(tmp_path / "data_health.db", old_date)
    config = MonitoringConfig(stale_data_hours_warning=48, stale_data_hours_critical=168)
    severity, anomalies = check_staleness("e01", db, "facts", "posting_date", config)
    assert severity == CRITICAL
    assert len(anomalies) == 1
    assert anomalies[0].category == STALE_DATA
    assert anomalies[0].company_id == "e01"


def test_staleness_thresholds_are_configurable(db, tmp_path):
    """Acceptance criterion: thresholds are configurable per company —
    same data, two configs, two verdicts."""
    old_date = (datetime.now(UTC) - timedelta(hours=100)).isoformat()
    _insert_row(tmp_path / "data_health.db", old_date)
    lenient = MonitoringConfig(stale_data_hours_warning=1000, stale_data_hours_critical=2000)
    strict = MonitoringConfig(stale_data_hours_warning=10, stale_data_hours_critical=50)
    assert check_staleness("e01", db, "facts", "posting_date", lenient)[0] == HEALTHY
    assert check_staleness("e01", db, "facts", "posting_date", strict)[0] == CRITICAL


def test_staleness_skips_gracefully_when_not_configured(db):
    """No boundaries_table configured -> no check runs, not a fabricated
    CRITICAL — matches data_boundaries()'s existing graceful-skip design."""
    config = MonitoringConfig()
    severity, anomalies = check_staleness("e01", db, None, None, config)
    assert severity == HEALTHY
    assert anomalies == []


def test_staleness_anomaly_includes_evidence(db, tmp_path):
    """Acceptance criterion: output includes evidence for later
    root-cause diagnosis."""
    old_date = (datetime.now(UTC) - timedelta(hours=300)).isoformat()
    _insert_row(tmp_path / "data_health.db", old_date)
    config = MonitoringConfig(stale_data_hours_warning=48, stale_data_hours_critical=168)
    _, anomalies = check_staleness("e01", db, "facts", "posting_date", config)
    assert "last_date" in anomalies[0].evidence
    assert "boundaries_table" in anomalies[0].evidence


# ---------------------------------------------------------- critical views


def test_healthy_critical_view_produces_no_anomaly(db):
    severity, anomalies = check_critical_views("e01", db, ["vw_healthy"])
    assert severity == HEALTHY
    assert anomalies == []


def test_broken_critical_view_produces_an_anomaly(db):
    """Issue #32 acceptance criterion, tested directly: a broken/
    unqueryable critical view produces an anomaly."""
    severity, anomalies = check_critical_views("e01", db, ["vw_broken"])
    assert severity == CRITICAL
    assert len(anomalies) == 1
    assert anomalies[0].category == BROKEN_VIEW
    assert anomalies[0].evidence["view"] == "vw_broken"


def test_empty_critical_view_is_not_broken(db):
    """A view with zero rows is not the same finding as a broken view —
    check_critical_views only detects execution failure, not emptiness
    (emptiness is check_value_rules's job, with its
    EXPECTED_EMPTY/must_have_rows distinction)."""
    severity, anomalies = check_critical_views("e01", db, ["vw_empty"])
    assert severity == HEALTHY
    assert anomalies == []


def test_multiple_critical_views_each_evaluated_independently(db):
    severity, anomalies = check_critical_views("e01", db, ["vw_healthy", "vw_broken", "vw_empty"])
    assert severity == CRITICAL  # one broken view is enough to fail overall
    assert len(anomalies) == 1  # only the actually-broken one flagged
    assert anomalies[0].evidence["view"] == "vw_broken"


def test_no_critical_views_configured_is_healthy(db):
    severity, anomalies = check_critical_views("e01", db, [])
    assert severity == HEALTHY
    assert anomalies == []


# ------------------------------------------------------------- value rules


def test_legitimately_empty_result_is_not_flagged_by_default(db):
    """Acceptance criterion: legitimate zero results are distinguished
    from configured suspicious-zero conditions — an empty view with no
    must_have_rows configured must NOT be flagged."""
    rules = [{"view": "vw_empty", "check": "nonzero", "column": "revenue"}]
    severity, anomalies = check_value_rules("e01", db, rules)
    assert severity == HEALTHY
    assert anomalies == []


def test_empty_result_is_flagged_when_must_have_rows_is_set(db):
    rules = [{"view": "vw_empty", "check": "nonzero", "column": "revenue", "must_have_rows": True}]
    severity, anomalies = check_value_rules("e01", db, rules)
    assert severity == WARNING
    assert len(anomalies) == 1
    assert anomalies[0].category == ROW_COUNT_ANOMALY


def test_all_zero_column_is_flagged_as_suspicious(db, tmp_path):
    _insert_row(tmp_path / "data_health.db", "2026-01-01", revenue=0.0)
    _insert_row(tmp_path / "data_health.db", "2026-01-02", revenue=0.0)
    rules = [{"view": "vw_all_zero_revenue", "check": "nonzero", "column": "revenue"}]
    severity, anomalies = check_value_rules("e01", db, rules)
    assert severity == WARNING
    assert anomalies[0].category == SUSPICIOUS_ZERO
    assert anomalies[0].metric == "revenue"


def test_row_count_min_violation_is_flagged(db, tmp_path):
    _insert_row(tmp_path / "data_health.db", "2026-01-01")
    rules = [{"view": "vw_healthy", "check": "row_count_min", "value": 5}]
    severity, anomalies = check_value_rules("e01", db, rules)
    assert severity == WARNING
    assert anomalies[0].category == ROW_COUNT_ANOMALY
    assert anomalies[0].observed_value == 1


def test_row_count_within_range_is_not_flagged(db, tmp_path):
    for i in range(3):
        _insert_row(tmp_path / "data_health.db", f"2026-01-0{i+1}")
    rules = [{"view": "vw_healthy", "check": "row_count_min", "value": 1}]
    severity, anomalies = check_value_rules("e01", db, rules)
    assert severity == HEALTHY
    assert anomalies == []


def test_null_rate_violation_is_flagged(db, tmp_path):
    for _ in range(8):
        _insert_row(tmp_path / "data_health.db", "2026-01-01", cost=None)
    for _ in range(2):
        _insert_row(tmp_path / "data_health.db", "2026-01-01", cost=5.0)
    rules = [{"view": "vw_healthy", "check": "null_rate_max", "column": "cost", "value": 0.5}]
    severity, anomalies = check_value_rules("e01", db, rules)
    assert severity == WARNING
    assert anomalies[0].category == NULL_RATE_ANOMALY
    assert anomalies[0].observed_value == pytest.approx(0.8)


def test_null_rate_within_threshold_is_not_flagged(db, tmp_path):
    for _ in range(9):
        _insert_row(tmp_path / "data_health.db", "2026-01-01", cost=5.0)
    _insert_row(tmp_path / "data_health.db", "2026-01-01", cost=None)
    rules = [{"view": "vw_healthy", "check": "null_rate_max", "column": "cost", "value": 0.5}]
    severity, _anomalies = check_value_rules("e01", db, rules)
    assert severity == HEALTHY


def test_rule_referencing_broken_view_is_flagged_not_crashed(db):
    rules = [{"view": "vw_broken", "check": "nonzero", "column": "revenue"}]
    severity, anomalies = check_value_rules("e01", db, rules)
    assert severity == CRITICAL
    assert anomalies[0].category == BROKEN_VIEW


def test_no_rules_configured_is_healthy(db):
    severity, anomalies = check_value_rules("e01", db, [])
    assert severity == HEALTHY
    assert anomalies == []


def test_value_rule_anomaly_includes_evidence(db, tmp_path):
    """Acceptance criterion: output includes evidence for later
    root-cause diagnosis."""
    _insert_row(tmp_path / "data_health.db", "2026-01-01", revenue=0.0)
    rules = [{"view": "vw_all_zero_revenue", "check": "nonzero", "column": "revenue", "description": "revenue should never be zero"}]
    _, anomalies = check_value_rules("e01", db, rules)
    assert anomalies[0].evidence["view"] == "vw_all_zero_revenue"
    assert anomalies[0].evidence["rule"]["description"] == "revenue should never be zero"


# ------------------------------------------------------------------ combined


def test_check_company_data_health_aggregates_everything(db, tmp_path):
    old_date = (datetime.now(UTC) - timedelta(hours=300)).isoformat()
    _insert_row(tmp_path / "data_health.db", old_date, revenue=0.0)
    config = MonitoringConfig(stale_data_hours_warning=48, stale_data_hours_critical=168)
    severity, anomalies = check_company_data_health(
        company_id="e01", executor=db,
        boundaries_table="facts", boundaries_date_column="posting_date",
        critical_views=["vw_broken"],
        data_health_checks=[{"view": "vw_all_zero_revenue", "check": "nonzero", "column": "revenue"}],
        config=config,
    )
    assert severity == CRITICAL  # broken view alone forces this
    categories = {a.category for a in anomalies}
    assert STALE_DATA in categories
    assert BROKEN_VIEW in categories
    assert SUSPICIOUS_ZERO in categories


def test_check_company_data_health_all_clean_is_healthy(db, tmp_path):
    _insert_row(tmp_path / "data_health.db", datetime.now(UTC).isoformat(), revenue=500.0)
    config = MonitoringConfig(stale_data_hours_warning=48, stale_data_hours_critical=168)
    severity, anomalies = check_company_data_health(
        company_id="e01", executor=db,
        boundaries_table="facts", boundaries_date_column="posting_date",
        critical_views=["vw_healthy"],
        data_health_checks=[],
        config=config,
    )
    assert severity == HEALTHY
    assert anomalies == []


def test_a_check_that_explodes_does_not_crash_the_whole_pass(db):
    """A check's own bug must degrade to a CRITICAL 'UNKNOWN' anomaly for
    just that check, never raise out of check_company_data_health —
    matches health.py's per-check guarding principle exactly."""
    config = MonitoringConfig()
    # None as critical_views would normally be fine (empty-list-like
    # iteration would fail) — force an actual TypeError to prove guarding.
    severity, anomalies = check_company_data_health(
        company_id="e01", executor=db,
        boundaries_table=None, boundaries_date_column=None,
        critical_views=None,  # type: ignore[arg-type]
        data_health_checks=[],
        config=config,
    )
    assert severity == CRITICAL
    assert any(a.metric == "critical_views" for a in anomalies)


def test_anomaly_to_dict_is_json_serializable(db, tmp_path):
    import json

    old_date = (datetime.now(UTC) - timedelta(hours=300)).isoformat()
    _insert_row(tmp_path / "data_health.db", old_date)
    config = MonitoringConfig(stale_data_hours_warning=48, stale_data_hours_critical=168)
    _, anomalies = check_staleness("e01", db, "facts", "posting_date", config)
    json.dumps(anomalies[0].to_dict())  # must not raise
