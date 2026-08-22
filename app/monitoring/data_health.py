"""Business data health and freshness monitoring — Phase 3 of the
monitoring system (issue #32).

Detects stale, missing, structurally abnormal, or invalid business data
before Eliara presents it as trustworthy insight. Every check here is
config-driven, never inferred from a fabricated baseline — this project's
consistent rule (see health.py's fallback_llm gap, telemetry's deferred
fields): state exactly what's configured, report a gap honestly when
nothing is configured, never invent a threshold nobody confirmed.

Company-aware by construction: every check takes the specific company's
CompanyConfig.critical_views / data_health_checks, not a global list —
Beta and Tire Guru can have (and do have) completely different critical
views and check definitions.

Anomalies are structured (DataAnomaly), not free-form log strings, per
the issue's explicit requirement — this is also the shape #33/#34's
anomaly/incident engine will consume, so it's designed to be a stable
foundation, not a throwaway internal detail.
"""

from __future__ import annotations

import dataclasses
import uuid
from datetime import UTC, datetime

from app.core.logging import get_logger
from app.monitoring.config import MonitoringConfig
from app.monitoring.health import CRITICAL, HEALTHY, WARNING

log = get_logger("monitoring.data_health")

# Categories, matching the master spec's Anomaly Engine category list
# (kept as strings, not an enum, so a later phase's anomaly engine can
# extend the set without a code change here).
STALE_DATA = "STALE_DATA"
BROKEN_VIEW = "BROKEN_VIEW"
ROW_COUNT_ANOMALY = "ROW_COUNT_ANOMALY"
NULL_RATE_ANOMALY = "NULL_RATE_ANOMALY"
SUSPICIOUS_ZERO = "SUSPICIOUS_ZERO"


@dataclasses.dataclass(frozen=True)
class DataAnomaly:
    """One structured data-health finding. Deliberately mirrors the
    field shape from the monitoring spec's Anomaly Engine section
    (anomaly_id, category, severity, metric, observed_value,
    expected_value, evidence) even though the full anomaly engine
    (issue #33/#34) doesn't exist yet — so this doesn't need reshaping
    when that phase arrives, it slots straight in."""

    anomaly_id: str
    timestamp: str
    company_id: str
    category: str  # STALE_DATA | BROKEN_VIEW | ROW_COUNT_ANOMALY | NULL_RATE_ANOMALY | SUSPICIOUS_ZERO
    severity: str  # WARNING | CRITICAL — matches health.py's states (issue #32 explicitly asks for this)
    metric: str
    observed_value: object
    expected_value: object
    evidence: dict
    status: str = "OPEN"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _new_anomaly(company_id: str, category: str, severity: str, metric: str, observed, expected, evidence: dict) -> DataAnomaly:
    return DataAnomaly(
        anomaly_id=uuid.uuid4().hex[:12],
        timestamp=datetime.now(UTC).isoformat(),
        company_id=company_id,
        category=category,
        severity=severity,
        metric=metric,
        observed_value=observed,
        expected_value=expected,
        evidence=evidence,
    )


# ------------------------------------------------------------- freshness


def check_staleness(company_id: str, executor, boundaries_table: str | None, boundaries_date_column: str | None, config: MonitoringConfig) -> tuple[str, list[DataAnomaly]]:
    """HEALTHY/WARNING/CRITICAL + zero or one STALE_DATA anomaly.

    Uses ReadOnlyExecutor.data_boundaries(), which already handles "not
    configured" gracefully (returns None) — this function inherits that:
    no boundaries_table configured means no staleness check runs, not a
    fabricated CRITICAL.
    """
    boundaries = executor.data_boundaries(table=boundaries_table, date_column=boundaries_date_column)
    if boundaries is None:
        return HEALTHY, []
    try:
        last_date = datetime.fromisoformat(boundaries.last_date).replace(tzinfo=UTC)
    except (ValueError, TypeError):
        return HEALTHY, []  # can't parse — not this check's job to guess a format
    age_hours = (datetime.now(UTC) - last_date).total_seconds() / 3600
    if age_hours >= config.stale_data_hours_critical:
        severity = CRITICAL
    elif age_hours >= config.stale_data_hours_warning:
        severity = WARNING
    else:
        return HEALTHY, []
    anomaly = _new_anomaly(
        company_id, STALE_DATA, severity, "data_age_hours",
        observed=round(age_hours, 1), expected=f"< {config.stale_data_hours_warning}h",
        evidence={"last_date": boundaries.last_date, "boundaries_table": boundaries_table},
    )
    return severity, [anomaly]


# ------------------------------------------------------------- critical views


def check_critical_views(company_id: str, executor, critical_views: list[str]) -> tuple[str, list[DataAnomaly]]:
    """Smoke-tests each configured critical view — does it execute at
    all. A view that fails outright (bad SQL, missing table, the SQRT-
    style "no such function" class of bug) produces a BROKEN_VIEW
    anomaly; a view that executes and returns zero rows is NOT an
    anomaly here (see the EXPECTED_EMPTY vs SUSPICIOUS_ZERO distinction
    in check_value_rules — a broken view and a legitimately-empty view
    are different findings, this function only detects the former)."""
    anomalies: list[DataAnomaly] = []
    for view in critical_views:
        try:
            executor.run_view(view, limit=1)
        except Exception as exc:  # noqa: BLE001 - any failure here is exactly what this check exists to catch
            anomalies.append(_new_anomaly(
                company_id, BROKEN_VIEW, CRITICAL, "view_queryable",
                observed=f"{type(exc).__name__}: {exc}", expected="query succeeds",
                evidence={"view": view},
            ))
    severity = CRITICAL if anomalies else HEALTHY
    return severity, anomalies


# ------------------------------------------------------------- configured value rules


def check_value_rules(company_id: str, executor, checks: list[dict]) -> tuple[str, list[DataAnomaly]]:
    """Runs each configured data_health_checks entry. Every rule type
    distinguishes a legitimately-empty result (EXPECTED_EMPTY — not
    flagged, unless the rule explicitly says must_have_rows) from a
    genuinely suspicious value (flagged) — the issue's explicit
    "distinguish EXPECTED_EMPTY_RESULT from SUSPICIOUS_ZERO_RESULT"
    requirement.

    Rule shape: {"view": str, "column": str,
    "check": "nonzero" | "row_count_min" | "row_count_max" | "null_rate_max",
    "value": float, "description": str, "must_have_rows": bool}
    """
    anomalies: list[DataAnomaly] = []
    for rule in checks:
        view = rule.get("view")
        check_type = rule.get("check")
        try:
            result = executor.run_view(view)
        except Exception as exc:  # noqa: BLE001 - a rule referencing a broken view is itself a finding, not a crash
            anomalies.append(_new_anomaly(
                company_id, BROKEN_VIEW, CRITICAL, "rule_view_queryable",
                observed=f"{type(exc).__name__}: {exc}", expected="query succeeds",
                evidence={"view": view, "rule": rule},
            ))
            continue

        row_count = result.row_count
        if row_count == 0:
            if rule.get("must_have_rows"):
                anomalies.append(_new_anomaly(
                    company_id, ROW_COUNT_ANOMALY, WARNING, "row_count",
                    observed=0, expected=">= 1 row (must_have_rows)",
                    evidence={"view": view, "rule": rule},
                ))
            # else: EXPECTED_EMPTY — a legitimately empty result, not flagged
            continue

        if check_type == "row_count_min" and row_count < rule.get("value", 0):
            anomalies.append(_new_anomaly(
                company_id, ROW_COUNT_ANOMALY, WARNING, "row_count",
                observed=row_count, expected=f">= {rule.get('value')}",
                evidence={"view": view, "rule": rule},
            ))
        elif check_type == "row_count_max" and row_count > rule.get("value", float("inf")):
            anomalies.append(_new_anomaly(
                company_id, ROW_COUNT_ANOMALY, WARNING, "row_count",
                observed=row_count, expected=f"<= {rule.get('value')}",
                evidence={"view": view, "rule": rule},
            ))
        elif check_type == "nonzero":
            column = rule.get("column")
            col_idx = result.columns.index(column) if column in result.columns else None
            if col_idx is not None:
                values = [row[col_idx] for row in result.rows]
                if values and all((v == 0 or v is None) for v in values):
                    anomalies.append(_new_anomaly(
                        company_id, SUSPICIOUS_ZERO, WARNING, column,
                        observed=0, expected="nonzero",
                        evidence={"view": view, "column": column, "rule": rule},
                    ))
        elif check_type == "null_rate_max":
            column = rule.get("column")
            col_idx = result.columns.index(column) if column in result.columns else None
            if col_idx is not None and result.rows:
                null_count = sum(1 for row in result.rows if row[col_idx] is None)
                null_rate = null_count / len(result.rows)
                if null_rate > rule.get("value", 1.0):
                    anomalies.append(_new_anomaly(
                        company_id, NULL_RATE_ANOMALY, WARNING, f"{column}_null_rate",
                        observed=round(null_rate, 4), expected=f"<= {rule.get('value')}",
                        evidence={"view": view, "column": column, "rule": rule},
                    ))

    severity = HEALTHY
    if any(a.severity == CRITICAL for a in anomalies):
        severity = CRITICAL
    elif anomalies:
        severity = WARNING
    return severity, anomalies


# ------------------------------------------------------------------- combined


def check_company_data_health(
    *, company_id: str, executor, boundaries_table: str | None, boundaries_date_column: str | None,
    critical_views: list[str], data_health_checks: list[dict], config: MonitoringConfig,
) -> tuple[str, list[DataAnomaly]]:
    """Runs every data-health check for one company, each independently
    guarded (matching health.py's principle) so one check's own bug
    degrades gracefully rather than crashing the whole pass."""
    all_anomalies: list[DataAnomaly] = []
    severities: list[str] = []

    def _guarded(name: str, fn):
        try:
            sev, anomalies = fn()
            severities.append(sev)
            all_anomalies.extend(anomalies)
        except Exception as exc:
            log.warning("data_health_check_failed", check=name, company_id=company_id, exc_info=True)
            severities.append(CRITICAL)
            all_anomalies.append(_new_anomaly(
                company_id, "UNKNOWN", CRITICAL, name,
                observed=f"{type(exc).__name__}: {exc}", expected="check runs without error",
                evidence={"check": name},
            ))

    _guarded("staleness", lambda: check_staleness(company_id, executor, boundaries_table, boundaries_date_column, config))
    _guarded("critical_views", lambda: check_critical_views(company_id, executor, critical_views))
    _guarded("value_rules", lambda: check_value_rules(company_id, executor, data_health_checks))

    overall = CRITICAL if CRITICAL in severities else (WARNING if WARNING in severities else HEALTHY)
    return overall, all_anomalies
