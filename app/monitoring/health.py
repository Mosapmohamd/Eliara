"""System health checks — Phase 2 of the monitoring system (issue #31).

Deterministic only — no LLM is used to decide a health state (spec
requirement: "Health checks do not require an LLM to decide basic
availability"). Every threshold comes from MonitoringConfig (Phase 1),
nothing here hardcodes a number.

This module is a pure, on-demand computation: call check_company() and
get a snapshot back. It is NOT wired into the live chat request path —
health checks belong behind their own endpoint (Phase 7 / issue #38's
Control Center), never on the path a user's question travels, so a
health-check bug structurally cannot affect a chat response. Every
individual check is still wrapped defensively inside check_company() as
a second layer, matching the spec's "must never crash the Eliara request
path" principle even though this module isn't on that path today.
"""

from __future__ import annotations

import dataclasses
import time
from datetime import UTC, datetime

from app.core.logging import get_logger
from app.monitoring.config import MonitoringConfig

log = get_logger("monitoring.health")

HEALTHY = "HEALTHY"
WARNING = "WARNING"
CRITICAL = "CRITICAL"


@dataclasses.dataclass(frozen=True)
class ComponentHealth:
    name: str
    status: str  # HEALTHY | WARNING | CRITICAL
    metrics: dict = dataclasses.field(default_factory=dict)
    detail: str | None = None


@dataclasses.dataclass(frozen=True)
class HealthSnapshot:
    company_id: str
    timestamp: str
    overall_status: str
    components: tuple[ComponentHealth, ...]

    def to_dict(self) -> dict:
        return {
            "company_id": self.company_id,
            "timestamp": self.timestamp,
            "overall_status": self.overall_status,
            "components": [dataclasses.asdict(c) for c in self.components],
        }


def _worst(statuses: list[str]) -> str:
    if CRITICAL in statuses:
        return CRITICAL
    if WARNING in statuses:
        return WARNING
    return HEALTHY


# ------------------------------------------------------------- individual checks


def check_database(executor) -> ComponentHealth:
    """Deterministic connectivity/readability check — a trivial SELECT 1
    through the existing read-only metadata path (ReadOnlyExecutor.
    run_metadata_sql, already used for schema introspection). Makes no
    assumption about any specific table, so it works identically for any
    company regardless of schema."""
    try:
        start = time.perf_counter()
        executor.run_metadata_sql("SELECT 1")
        elapsed_ms = (time.perf_counter() - start) * 1000
        return ComponentHealth("database", HEALTHY, {"query_ms": round(elapsed_ms, 1)})
    except Exception as exc:  # noqa: BLE001 - any DB failure must degrade this component, not crash the check
        return ComponentHealth("database", CRITICAL, detail=f"{type(exc).__name__}: {exc}")


def check_llm_configured(settings) -> ComponentHealth:
    """Deliberately checks key presence only, not a live API call — the
    spec requires basic-availability checks to not need an LLM to decide
    anything, and a real API call on every health poll would spend money
    and latency to learn something (key presence) already knowable
    without one. Actual observed LLM error/timeout behavior comes from
    real traffic — see check_llm_error_rate."""
    configured = bool(getattr(settings, "anthropic_api_key", None) and settings.anthropic_api_key.get_secret_value())
    return ComponentHealth(
        "llm_configured", HEALTHY if configured else CRITICAL, {"configured": configured}
    )


def check_fallback_llm() -> ComponentHealth:
    """No fallback-LLM concept exists in Settings yet (single
    orchestrator_model/sqlgen_model, no configured fallback) — reported
    honestly as a gap rather than fabricating a check against a
    capability that doesn't exist. See app/monitoring/README.md's Phase 2
    gaps list. Fixed WARNING (not CRITICAL) since the system has no
    fallback today by design, not by failure — this reports the gap's
    existence, not a live outage."""
    return ComponentHealth(
        "fallback_llm", WARNING, detail="not configured — no fallback model exists in Settings yet"
    )


def check_retrieval_index(metadata_index) -> ComponentHealth:
    count = len(metadata_index.objects) if metadata_index is not None else 0
    return ComponentHealth("retrieval_index", HEALTHY if count > 0 else CRITICAL, {"object_count": count})


def check_cache(result_cache) -> ComponentHealth:
    """A real set()+get() round-trip with a throwaway key — deterministic,
    catches an actually-broken cache rather than just checking the object
    exists."""
    try:
        probe_key = ("__monitoring_health_probe__",)
        result_cache.set(probe_key, "ok")
        value = result_cache.get(probe_key)
        return ComponentHealth("cache", HEALTHY if value == "ok" else WARNING)
    except Exception as exc:  # noqa: BLE001 - any cache failure must degrade this component, not crash the check
        return ComponentHealth("cache", CRITICAL, detail=f"{type(exc).__name__}: {exc}")


def check_latency(traces: list[dict], config: MonitoringConfig) -> ComponentHealth:
    if not traces:
        return ComponentHealth("latency", HEALTHY, {"p50_ms": None, "p95_ms": None, "sample_size": 0})
    latencies = sorted(t.get("latency_ms", 0) or 0 for t in traces)
    p50 = latencies[len(latencies) // 2]
    p95_idx = min(int(len(latencies) * 0.95), len(latencies) - 1)
    p95 = latencies[p95_idx]
    if p95 >= config.p95_latency_ms_critical:
        status = CRITICAL
    elif p95 >= config.p95_latency_ms_warning:
        status = WARNING
    else:
        status = HEALTHY
    return ComponentHealth("latency", status, {"p50_ms": p50, "p95_ms": p95, "sample_size": len(traces)})


def check_error_rate(traces: list[dict], config: MonitoringConfig) -> ComponentHealth:
    if not traces:
        return ComponentHealth("error_rate", HEALTHY, {"error_rate": None, "sample_size": 0})
    error_count = sum(1 for t in traces if t.get("status") == "error")
    rate = error_count / len(traces)
    if rate >= config.db_error_rate_critical:
        status = CRITICAL
    elif rate >= config.db_error_rate_warning:
        status = WARNING
    else:
        status = HEALTHY
    return ComponentHealth("error_rate", status, {"error_rate": round(rate, 4), "sample_size": len(traces)})


# ------------------------------------------------------------------- checker


class HealthChecker:
    def __init__(self, telemetry, config: MonitoringConfig | None = None) -> None:
        self._telemetry = telemetry
        self._config = config or MonitoringConfig()

    def check_company(
        self, *, company_id: str, executor, settings, metadata_index, result_cache
    ) -> HealthSnapshot:
        """Runs every component check, each independently guarded so one
        check's own bug degrades that one component to CRITICAL with a
        clear detail message rather than raising out of the whole
        snapshot — mirrors the request-path principle even though this
        function isn't on the request path itself."""
        components: list[ComponentHealth] = []

        def _guarded(name: str, fn):
            try:
                components.append(fn())
            except Exception as exc:
                log.warning("health_check_failed", check=name, company_id=company_id, exc_info=True)
                components.append(ComponentHealth(name, CRITICAL, detail=f"health check itself failed: {exc}"))

        _guarded("database", lambda: check_database(executor))
        _guarded("llm_configured", lambda: check_llm_configured(settings))
        _guarded("fallback_llm", check_fallback_llm)
        _guarded("retrieval_index", lambda: check_retrieval_index(metadata_index))
        _guarded("cache", lambda: check_cache(result_cache))

        since = time.time() - self._config.health_window_minutes * 60
        traces = self._telemetry.read_recent(company_id, since) if self._telemetry is not None else []
        _guarded("latency", lambda: check_latency(traces, self._config))
        _guarded("error_rate", lambda: check_error_rate(traces, self._config))

        overall = _worst([c.status for c in components])
        return HealthSnapshot(
            company_id=company_id,
            timestamp=datetime.now(UTC).isoformat(),
            overall_status=overall,
            components=tuple(components),
        )
