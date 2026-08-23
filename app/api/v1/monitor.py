"""Monitoring API — Phase 7 (issue #38), the Health Control Center's backend.

Every endpoint here is read-only and company-scoped via the same
get_company_context() every other route uses (issue #38: "Apply existing
authentication/security conventions"). No endpoint requires an LLM to
compute anything — every check in health.py/data_health.py/ai_health.py
is deterministic (issue #38's explicit acceptance criterion, inherited
from #31's).

Checks run on-demand (pull-based) when an endpoint is hit — there is no
background scheduler yet, same deliberate scope boundary as /detect's
"manually triggered, not scheduled" note. Calling /monitor/health/*
also feeds any anomalies found into the shared IncidentEngine, so
hitting the dashboard is also what keeps incidents current for now.

No auto-heal data anywhere in these responses (see incidents.py's
module docstring) — issue #38 asks for "auto-heal attempts and success
rate" and "automatically recoverable incidents" to be visible, but that
capability doesn't exist in this codebase (issues #36/#37 explicitly
excluded). Every incident is reported as requiring human review because
that's true of all of them right now, not a placeholder.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request

from app.api.deps import get_company_context
from app.core.config import get_settings
from app.core.logging import get_logger
from app.monitoring.ai_health import check_company_ai_quality
from app.monitoring.config import get_monitoring_config
from app.monitoring.data_health import check_company_data_health
from app.monitoring.health import HealthChecker, check_error_rate, check_latency
from app.monitoring.incidents import IncidentEngine

log = get_logger("monitoring.api")

router = APIRouter(prefix="/monitor", tags=["monitoring"])


def _telemetry(request: Request):
    return getattr(request.app.state, "telemetry", None)


def _incident_engine(request: Request) -> IncidentEngine:
    engine = getattr(request.app.state, "incident_engine", None)
    if engine is None:
        # Should always be set by main.py's lifespan — this fallback
        # exists so a missing wire-up degrades to "incidents aren't
        # persisted across requests" rather than 500ing the dashboard.
        engine = IncidentEngine()
        request.app.state.incident_engine = engine
    return engine


def _recent_traces(request: Request, company_id: str, window_minutes: int) -> list[dict]:
    telemetry = _telemetry(request)
    if telemetry is None:
        return []
    since = time.time() - window_minutes * 60
    return telemetry.read_recent(company_id, since)


@router.get("/health/system")
async def health_system(request: Request, company_id: str):
    ctx = get_company_context(request, company_id)
    config = get_monitoring_config()
    checker = HealthChecker(_telemetry(request), config)
    snapshot = checker.check_company(
        company_id=company_id, executor=ctx.executor, settings=get_settings(),
        metadata_index=ctx.metadata_index, result_cache=ctx.result_cache,
    )
    return snapshot.to_dict()


@router.get("/health/data")
async def health_data(request: Request, company_id: str):
    ctx = get_company_context(request, company_id)
    config = get_monitoring_config()
    severity, anomalies = check_company_data_health(
        company_id=company_id, executor=ctx.executor,
        boundaries_table=ctx.config.boundaries_table,
        boundaries_date_column=ctx.config.boundaries_date_column,
        critical_views=ctx.config.critical_views,
        data_health_checks=ctx.config.data_health_checks,
        config=config,
    )
    _incident_engine(request).record_anomalies(anomalies)
    return {"company_id": company_id, "status": severity, "anomalies": [a.to_dict() for a in anomalies]}


@router.get("/health/ai")
async def health_ai(request: Request, company_id: str):
    get_company_context(request, company_id)  # validates company exists/healthy
    config = get_monitoring_config()
    traces = _recent_traces(request, company_id, config.health_window_minutes)
    severity, anomalies = check_company_ai_quality(company_id, traces, config)
    _incident_engine(request).record_anomalies(anomalies)
    return {
        "company_id": company_id, "status": severity,
        "anomalies": [a.to_dict() for a in anomalies], "sample_size": len(traces),
    }


@router.get("/health")
async def health_overall(request: Request, company_id: str | None = None):
    """One combined snapshot (system + data + AI) per company — the
    "team can identify current unhealthy component(s) from one screen"
    acceptance criterion's data source."""
    manager = getattr(request.app.state, "company_manager", None)
    if manager is None:
        raise HTTPException(status_code=503, detail="company_manager not initialized")
    company_ids = [company_id] if company_id else manager.all_ids()

    results = {}
    for cid in company_ids:
        try:
            system = await health_system(request, cid)
            data = await health_data(request, cid)
            ai = await health_ai(request, cid)
            statuses = [system["overall_status"], data["status"], ai["status"]]
            overall = "CRITICAL" if "CRITICAL" in statuses else ("WARNING" if "WARNING" in statuses else "HEALTHY")
            results[cid] = {"overall_status": overall, "system": system, "data": data, "ai": ai}
        except HTTPException:
            raise
        except Exception:
            log.warning("health_overall_failed", company_id=cid, exc_info=True)
            results[cid] = {"overall_status": "CRITICAL", "error": "health computation failed for this company"}
    return results


@router.get("/incidents")
async def list_incidents(
    request: Request, company_id: str | None = None, status: str | None = None, category: str | None = None,
):
    engine = _incident_engine(request)
    incidents = engine.query(company_id=company_id, status=status, category=category)
    return [i.to_dict() for i in incidents]


@router.get("/incidents/{incident_id}")
async def get_incident(request: Request, incident_id: str):
    incident = _incident_engine(request).get(incident_id)
    if incident is None:
        raise HTTPException(status_code=404, detail="incident not found")
    return incident.to_dict()


@router.get("/metrics")
async def metrics(request: Request, company_id: str):
    get_company_context(request, company_id)
    config = get_monitoring_config()
    traces = _recent_traces(request, company_id, config.health_window_minutes)
    return {
        "company_id": company_id,
        "window_minutes": config.health_window_minutes,
        "latency": check_latency(traces, config).metrics,
        "error_rate": check_error_rate(traces, config).metrics,
    }


@router.get("/traces/{request_id}")
async def get_trace(request: Request, request_id: str, company_id: str):
    telemetry = _telemetry(request)
    trace = telemetry.read_trace(company_id, request_id) if telemetry else None
    if trace is None:
        raise HTTPException(status_code=404, detail="trace not found")
    return trace
