"""Central monitoring configuration.

Follows the same pattern as app.core.config.Settings — pydantic-settings,
ELIARA_ prefix, .env-driven. Every threshold used anywhere in the
monitoring layer is defined here, once. No module outside this file
should hardcode a threshold value — see the spec's Configuration section
("Do not scatter threshold values across source files").

Per-company overrides (spec: "tenant_overrides") are intentionally NOT a
flat env var per company — that doesn't scale past a couple of companies
and duplicates the pattern CompanyConfig already exists for. Instead,
company-specific threshold overrides belong in CompanyConfig itself
(app/company/registry.py) as an optional field, resolved at read time by
whoever consumes a threshold. This file defines only the global defaults;
wiring company-specific overrides through is Phase 2+ (data health), once
there's a concrete per-company threshold to override — no field is added
speculatively ahead of an actual use.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MonitoringConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ELIARA_MONITOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Master switch. When False, telemetry recording is skipped entirely
    # (not just the API/UI) — zero overhead on the request path, not just
    # zero visibility, since a monitoring system that's "off" but still
    # doing work isn't really off.
    enabled: bool = True

    # Where request_traces (and, in later phases, health_snapshots/
    # anomalies/incidents) are stored. Same directory convention as
    # ELIARA_AUDIT_DIR — one JSONL-per-day file per company, chosen for
    # the same reason audit.py already uses it: simplest thing that's
    # actually safe to write from concurrent requests without a DB
    # connection pool of its own.
    storage_dir: str = "monitoring_data"

    # --- Thresholds (Phase 2+ consumers; defined here now so every
    # threshold lives in exactly one place from the start, per the spec) ---
    p95_latency_ms_warning: int = Field(default=3000, description="P95 latency WARNING threshold")
    p95_latency_ms_critical: int = Field(default=8000, description="P95 latency CRITICAL threshold")
    llm_timeout_rate_warning: float = Field(default=0.05, description="Fraction of LLM calls timing out")
    llm_timeout_rate_critical: float = Field(default=0.15, description="")
    db_error_rate_warning: float = Field(default=0.02, description="Fraction of DB calls failing")
    db_error_rate_critical: float = Field(default=0.10, description="")
    stale_data_hours_warning: int = Field(default=48, description="Hours since last data refresh")
    stale_data_hours_critical: int = Field(default=168, description="")
    generated_sql_rate_warning: float = Field(
        default=0.30, description="Fraction of requests falling back to generated SQL"
    )
    entity_failure_rate_warning: float = Field(default=0.10, description="Fraction of entity resolutions failing")

    # Health-window: how much recent history a health check considers
    # (e.g. "P95 over the last N minutes"), not an all-time average.
    health_window_minutes: int = 15


@lru_cache
def get_monitoring_config() -> MonitoringConfig:
    """Process-wide cached singleton — same @lru_cache pattern as
    app.core.config.get_settings(). The config itself is immutable once
    loaded, so sharing the instance across requests is safe (this is a
    config object, not per-request state; contrast with RequestTrace,
    which is never shared)."""
    return MonitoringConfig()
