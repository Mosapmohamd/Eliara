"""Company registry: maps a `company_id` to its configuration.

Loaded once at startup from a YAML file (path set by
``Settings.companies_config_path`` / ``ELIARA_COMPANIES_CONFIG``). This is
the single place that knows which companies exist and where each one's
resources (database, playbooks, prompts) live on disk — no other module
should hardcode a company's db path, view list, or directory.
"""

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from app.core.errors import EliaraError
from app.core.logging import get_logger

log = get_logger("company_registry")

_COMPANY_ID_RE = re.compile(r"^[a-z0-9_]+$")


class UnknownCompany(EliaraError):
    status_code = 404
    public_message = "Unknown company."


class CompanyRegistryError(EliaraError):
    public_message = "Internal configuration error."


class CompanyConfig(BaseModel):
    """Everything company-specific that used to live in global Settings."""

    company_id: str
    display_name: str
    db_path: Path

    # Curated views for the /scan proactive-attention prefix. Empty means
    # /scan degrades to "no findings configured for this company" rather
    # than erroring — a company can operate fully without this.
    scan_views: list[str] = Field(default_factory=list)

    # Optional per-company overrides. None = fall back to shared/defaults.
    prompts_dir: Path | None = None
    playbooks_dir: Path | None = None
    embedding_cache_dir: Path | None = None

    # Table/column used by ReadOnlyExecutor.data_boundaries() for the
    # "data through <date>" deep-health line. None = skip that check
    # gracefully rather than assuming a Beta-specific fact table exists.
    boundaries_table: str | None = None
    boundaries_date_column: str | None = None

    # Table this company's curated-question registry lives in. Defaults to
    # the "canonical" name; a company whose registry has a different name
    # (and possibly a different column set — see MetadataLoader._load_registry's
    # column-adaptation logic) points this at that table instead.
    registry_table: str = "chatbot_question_view_registry"

    # Views considered critical for this company's business-data health
    # monitoring (app/monitoring/data_health.py, issue #32) — smoke-tested
    # for basic queryability. Empty means no critical-view checks run for
    # this company yet, not an error; a company operates fully without
    # this configured.
    critical_views: list[str] = Field(default_factory=list)

    # Configured data-health value checks (issue #32) — each entry:
    # {"view": str, "column": str, "check": "nonzero"|"row_count_min"|
    # "row_count_max"|"null_rate_max", "value": float, "description": str,
    # "must_have_rows": bool}. Deliberately config-driven, not inferred
    # from historical baselines — this project's baseline is "state
    # exactly what's configured, never fabricate a threshold nobody
    # confirmed." Empty list means no value checks configured yet, not a
    # gap in the mechanism itself — see app/monitoring/data_health.py.
    data_health_checks: list[dict] = Field(default_factory=list)

    # "full" (default) or "partial" — a company that resolves and boots
    # successfully (its database is real and readable) but is missing
    # curated playbooks/scan views/business glossary should be marked
    # "partial" so this is visible in /health/deep and not just in a YAML
    # comment. A partial company still answers questions normally — this
    # is a completeness signal, not a health signal.
    status: str = "full"

    @field_validator("status")
    @classmethod
    def _valid_status(cls, v: str) -> str:
        if v not in ("full", "partial"):
            raise ValueError(f'status must be "full" or "partial", got {v!r}')
        return v

    @field_validator("company_id")
    @classmethod
    def _valid_slug(cls, v: str) -> str:
        if not _COMPANY_ID_RE.match(v):
            raise ValueError(
                f"company_id must match ^[a-z0-9_]+$, got {v!r}"
            )
        return v


class CompanyRegistry:
    def __init__(self, companies: dict[str, CompanyConfig]) -> None:
        self._companies = companies

    @classmethod
    def from_file(cls, path: Path | str) -> "CompanyRegistry":
        path = Path(path)
        if not path.exists():
            raise CompanyRegistryError(
                internal_detail=f"companies config not found: {path}"
            )
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        entries = raw.get("companies") or {}
        if not entries:
            raise CompanyRegistryError(
                internal_detail=f"companies config has no 'companies' entries: {path}"
            )
        companies: dict[str, CompanyConfig] = {}
        for company_id, fields in entries.items():
            fields = dict(fields or {})
            fields.setdefault("company_id", company_id)
            try:
                cfg = CompanyConfig(**fields)
            except Exception as exc:
                raise CompanyRegistryError(
                    internal_detail=f"invalid config for company {company_id!r}: {exc}"
                ) from exc
            if cfg.company_id != company_id:
                raise CompanyRegistryError(
                    internal_detail=(
                        f"company_id mismatch: key {company_id!r} vs "
                        f"declared {cfg.company_id!r}"
                    )
                )
            companies[company_id] = cfg
        log.info("companies_registry_loaded", companies=sorted(companies))
        return cls(companies)

    def get(self, company_id: str) -> CompanyConfig:
        try:
            return self._companies[company_id]
        except KeyError:
            raise UnknownCompany(
                internal_detail=f"unknown company_id: {company_id!r}"
            ) from None

    def all_ids(self) -> list[str]:
        return sorted(self._companies)

    def __len__(self) -> int:
        return len(self._companies)
