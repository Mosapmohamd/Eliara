"""FastAPI application factory.

The lifespan hook is where M1/M2 will attach the read-only DB pool and the
metadata index build; in M0 it only configures logging.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.legacy import router as legacy_router
from app.api.middleware import RequestContextMiddleware
from app.api.v1.catalogue import router as catalogue_router
from app.api.v1.chat import router as chat_router
from app.api.v1.compat import router as compat_router
from app.api.v1.health import router as health_router
from app.api.v1.sessions import router as sessions_router
from app.company.context import CompanyContextManager
from app.company.registry import CompanyRegistry, CompanyRegistryError
from app.core.audit import AuditTrail
from app.core.cache import RateLimiter
from app.core.config import get_settings
from app.core.errors import EliaraError
from app.core.logging import get_logger, setup_logging
from app.llm.anthropic_client import AnthropicClient
from app.monitoring.config import get_monitoring_config
from app.monitoring.telemetry import TraceRecorder

log = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    log.info("startup", environment=settings.environment)
    app.state.metrics = {"requests_total": 0, "cache_hits": 0}
    # Rate limiting stays process-global (platform-level, not company
    # data) but keys are namespaced per company at the call site so one
    # company's traffic can't exhaust another's quota — see api/v1/chat.py.
    app.state.rate_limiter = RateLimiter(settings.chat_rate_limit_per_min)
    # One shared audit trail; every record carries company_id and is filed
    # under audit_dir/<company_id>/ (see app/core/audit.py).
    audit = AuditTrail(settings.audit_dir, enabled=settings.audit_enabled)
    # Monitoring telemetry — Phase 1 (observability only, see
    # app/monitoring/README.md). Same directory-per-company pattern as
    # audit, deliberately: it's the same proven concurrency-safe shape,
    # not a coincidence.
    monitor_config = get_monitoring_config()
    telemetry = TraceRecorder(monitor_config.storage_dir, enabled=monitor_config.enabled)

    try:
        registry = CompanyRegistry.from_file(settings.companies_config_path)
    except CompanyRegistryError:
        log.exception("companies_registry_load_failed", path=str(settings.companies_config_path))
        registry = None

    if registry is not None:
        manager = CompanyContextManager(registry, settings, AnthropicClient(settings), audit, telemetry)
        # Eager, per-company try/except build: a broken/missing database for
        # one company must not prevent the platform from starting, and must
        # not prevent OTHER companies from serving traffic.
        manager.build_all_eagerly()
        app.state.company_manager = manager
    else:
        app.state.company_manager = None

    yield

    if getattr(app.state, "company_manager", None):
        app.state.company_manager.shutdown()
    log.info("shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Eliara Analytics Platform",
        version="0.1.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    settings = get_settings()
    origins = settings.cors_origins.strip()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if origins == "*"
        else [o.strip() for o in origins.split(",") if o.strip()],
        allow_origin_regex=None if origins == "*" else r"https://.*\.vercel\.app",
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestContextMiddleware)
    app.include_router(health_router, prefix="/api/v1")
    app.include_router(chat_router, prefix="/api/v1")
    app.include_router(sessions_router, prefix="/api/v1")
    app.include_router(catalogue_router, prefix="/api/v1")
    app.include_router(compat_router, prefix="/api/v1")
    if settings.legacy_api_enabled:
        app.include_router(legacy_router)  # root level, no prefix — on purpose

    @app.exception_handler(EliaraError)
    async def eliara_error_handler(_: Request, exc: EliaraError) -> JSONResponse:
        log.warning("domain_error", error=type(exc).__name__, detail=exc.internal_detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.public_message},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
        # log.exception() (not .error(..., exc_info=True)) — this IS an
        # active exception-handling context (a registered FastAPI exception
        # handler receiving the caught exception), even though ruff's
        # static analysis for LOG014 only recognizes a literal `except:`
        # block as one. The traceback is essential here: this is the
        # catch-all for genuinely unexpected errors, the one place you most
        # need it.
        log.exception("unhandled_error", error=type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={"error": "An internal error occurred. Please try again."},
        )

    return app


app = create_app()
