"""Per-company runtime: one isolated set of resources per `company_id`.

This is the direct generalization of what ``app/main.py``'s old
``lifespan()`` did for a single database — narrowed from "the one app" to
"one company," and repeated once per registered company. It deliberately
does not reimplement any existing component: ``ReadOnlyExecutor``,
``DatabaseWatcher``, ``build_discovery``, ``build_entity_index``,
``PromptManager``, ``PlaybookLibrary``, ``ResultCache``,
``InMemoryConversationStore`` and ``Orchestrator`` are all reused exactly
as they exist elsewhere in the codebase.

Isolation guarantee: each ``CompanyContext`` owns its own
``ReadOnlyExecutor`` (its own connection pool), its own discovery
index/retriever/entity index, its own result cache, its own conversation
store, and its own database watcher. There is no shared mutable state
between two companies' contexts — a request resolved to company A's
context can only ever reach company A's objects.
"""

import threading
from dataclasses import dataclass

from app.company.registry import CompanyConfig, CompanyRegistry
from app.core.audit import AuditTrail
from app.core.cache import ResultCache
from app.core.config import Settings
from app.core.logging import get_logger
from app.discovery.entity_resolver import EntityIndex, build_entity_index
from app.discovery.index import MetadataIndex
from app.discovery.search import HybridRetriever
from app.discovery.service import build_discovery
from app.execution.db_watcher import DatabaseWatcher
from app.execution.executor import ReadOnlyExecutor
from app.llm.anthropic_client import AnthropicClient
from app.orchestrator.conversation import InMemoryConversationStore
from app.orchestrator.orchestrator import Orchestrator
from app.prompts.loader import PromptManager

log = get_logger("company_context")


@dataclass
class CompanyContext:
    company_id: str
    config: CompanyConfig
    executor: ReadOnlyExecutor
    metadata_index: MetadataIndex
    retriever: HybridRetriever
    entities: EntityIndex
    prompts: PromptManager
    conversations: InMemoryConversationStore
    result_cache: ResultCache
    orchestrator: Orchestrator
    watcher: DatabaseWatcher
    # Set False if this company's context failed to build at startup
    # (e.g. its database file was missing) — one broken company must not
    # take the process down, and /health/deep needs to report this per
    # company rather than crash.
    healthy: bool = True
    startup_error: str | None = None


class CompanyContextManager:
    def __init__(
        self,
        registry: CompanyRegistry,
        settings: Settings,
        llm: AnthropicClient,
        audit: AuditTrail,
        telemetry=None,
    ) -> None:
        self._registry = registry
        self._settings = settings
        self._llm = llm
        self._audit = audit
        self._telemetry = telemetry
        self._contexts: dict[str, CompanyContext] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------- lookup

    def get(self, company_id: str) -> CompanyContext:
        """Return the cached context for a company, building it on first
        use if it wasn't built eagerly at startup (or if an eager build
        failed and this is a retry)."""
        ctx = self._contexts.get(company_id)
        if ctx is not None and ctx.healthy:
            return ctx
        with self._lock:
            ctx = self._contexts.get(company_id)
            if ctx is not None and ctx.healthy:
                return ctx
            ctx = self._build(company_id)
            self._contexts[company_id] = ctx
        return ctx

    def all_ids(self) -> list[str]:
        return self._registry.all_ids()

    def all_contexts(self) -> dict[str, CompanyContext]:
        return dict(self._contexts)

    # ------------------------------------------------------------- lifecycle

    def build_all_eagerly(self) -> None:
        """Build every registered company's context at startup, each in its
        own try/except — one company's broken database must not prevent
        the other companies (or the process) from starting."""
        for company_id in self._registry.all_ids():
            try:
                with self._lock:
                    self._contexts[company_id] = self._build(company_id)
                log.info("company_context_ready", company_id=company_id)
            except Exception as exc:
                log.exception(
                    "company_context_build_failed",
                    company_id=company_id,
                    error=str(exc),
                )
                cfg = self._registry.get(company_id)
                self._contexts[company_id] = _failed_context(company_id, cfg, str(exc))

    def shutdown(self) -> None:
        for company_id, ctx in self._contexts.items():
            try:
                if ctx.watcher is not None:
                    ctx.watcher.stop()
                if ctx.executor is not None:
                    ctx.executor.close()
            except Exception:
                log.warning("company_context_shutdown_failed", company_id=company_id, exc_info=True)

    # ------------------------------------------------------------- building

    def _build(self, company_id: str) -> CompanyContext:
        cfg = self._registry.get(company_id)  # raises UnknownCompany if not registered

        executor = ReadOnlyExecutor(
            cfg.db_path,
            query_timeout_s=self._settings.query_timeout_s,
            max_rows=self._settings.max_rows,
        )
        index, retriever = build_discovery(executor, self._settings, cfg.registry_table)
        entities = build_entity_index(
            executor,
            index.objects,
            include_facts=self._settings.entity_index_include_facts,
            overrides=self._settings.entity_index_sources,
        )
        prompts = PromptManager.for_company(cfg)
        result_cache = ResultCache(ttl_s=self._settings.result_cache_ttl_s)
        conversations = InMemoryConversationStore(
            history_size=self._settings.history_size,
            ttl_min=self._settings.session_ttl_min,
        )
        orchestrator = Orchestrator(
            retriever=retriever,
            index=index,
            executor=executor,
            prompts=prompts,
            conversations=conversations,
            llm=self._llm,
            settings=self._settings,
            result_cache=result_cache,
            audit=self._audit,
            telemetry=self._telemetry,
            company_id=company_id,
            scan_views=cfg.scan_views,
            playbooks_dir=cfg.playbooks_dir,
            boundaries_table=cfg.boundaries_table,
            boundaries_date_column=cfg.boundaries_date_column,
        )
        orchestrator._entities = entities

        ctx = CompanyContext(
            company_id=company_id,
            config=cfg,
            executor=executor,
            metadata_index=index,
            retriever=retriever,
            entities=entities,
            prompts=prompts,
            conversations=conversations,
            result_cache=result_cache,
            orchestrator=orchestrator,
            watcher=DatabaseWatcher(cfg.db_path, interval_s=self._settings.db_watch_interval_s),
        )

        def _on_refresh(company_id: str = company_id) -> None:
            self._refresh(company_id)

        ctx.watcher.on_change(_on_refresh)
        ctx.watcher.start()
        return ctx

    def _refresh(self, company_id: str) -> None:
        """Atomic swap scoped to ONE company. A refresh for Beta must never
        rebuild or interrupt Tire Guru's executor/index/retriever/entities/
        cache — this method only ever touches the single context passed in."""
        ctx = self._contexts.get(company_id)
        if ctx is None or not ctx.healthy:
            return
        ctx.executor.reopen()
        ctx.result_cache.clear()
        new_index, new_retriever = build_discovery(ctx.executor, self._settings, ctx.config.registry_table)
        new_entities = build_entity_index(
            ctx.executor,
            new_index.objects,
            include_facts=self._settings.entity_index_include_facts,
            overrides=self._settings.entity_index_sources,
        )
        ctx.metadata_index = new_index
        ctx.retriever = new_retriever
        ctx.entities = new_entities
        ctx.orchestrator._index = new_index
        ctx.orchestrator._retriever = new_retriever
        ctx.orchestrator._entities = new_entities
        log.info("company_context_refreshed", company_id=company_id)


def _failed_context(company_id: str, cfg: CompanyConfig, error: str) -> CompanyContext:
    """A placeholder for a company whose startup build failed — carries no
    live resources, but keeps the manager's dict shape uniform so
    /health/deep and repeated .get() retries have something to inspect."""
    return CompanyContext(
        company_id=company_id,
        config=cfg,
        executor=None,  # type: ignore[arg-type]
        metadata_index=None,  # type: ignore[arg-type]
        retriever=None,  # type: ignore[arg-type]
        entities=None,  # type: ignore[arg-type]
        prompts=None,  # type: ignore[arg-type]
        conversations=None,  # type: ignore[arg-type]
        result_cache=None,  # type: ignore[arg-type]
        orchestrator=None,  # type: ignore[arg-type]
        watcher=None,  # type: ignore[arg-type]
        healthy=False,
        startup_error=error,
    )
