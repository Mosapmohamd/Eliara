"""The orchestration pipeline (Phase 2 design, M4 scope: view path).

    retriever -> Sonnet #1 (RoutingDecision) -> executor -> Sonnet #2 (answer)

Deterministic code sits between the two model calls; every model output is
verified server-side before it touches the database.
"""

import asyncio
import re
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from pydantic import BaseModel

from app.core.config import Settings
from app.core.errors import EliaraError, RoutingError, SQLExecutionError, SQLValidationError
from app.core.logging import get_logger, get_request_id
from app.core.models import QueryResult
from app.detection.attention_queue import AttentionQueue, build_attention_queue
from app.discovery.entity_resolver import EntityIndex, Resolution
from app.discovery.index import MetadataIndex
from app.discovery.search import HybridRetriever
from app.execution.aggregate import summarise
from app.execution.executor import ReadOnlyExecutor
from app.execution.formatter import to_llm_payload
from app.execution.refine import RefineError, apply_refinement
from app.execution.visual import build_visual
from app.llm.anthropic_client import AnthropicClient
from app.monitoring.models import RequestTrace
from app.orchestrator.agent import ReasoningAgent
from app.orchestrator.agent_models import AgentTrace
from app.orchestrator.conversation import InMemoryConversationStore, Message, ResultEntry
from app.orchestrator.decision_models import RoutingDecision
from app.orchestrator.playbooks import PlaybookLibrary, run_playbook
from app.orchestrator.verification import VerificationReport, verify
from app.prompts.loader import PromptManager
from app.sqlgen.generator import SQLGenerator
from app.sqlgen.schema_context import build_slice
from app.sqlgen.validator import validate_sql

log = get_logger("orchestrator")

_OUT_OF_SCOPE_ANSWER = (
    "I can help with questions about the company's business data — sales, "
    "customers, inventory, procurement, suppliers, and financial performance. "
    "This request falls outside that scope."
)
_SQL_CAUTION = "CUSTOM_QUERY_NOT_BUSINESS_VALIDATED"

# User-facing loading text for the "stage" SSE event. Internal decision names
# (use_view, needs_sql, ...) never reach the client — only this copy does.
_STAGE_TEXT = {
    "use_view": "Querying your data...",
    "refine": "Refining previous results...",
    "run_playbook": "Running the analysis...",
    "needs_sql": "Building a custom query...",
    "investigate": "Investigating...",
    "greeting": "One moment...",
    "clarify": "One moment...",
    "out_of_scope": "One moment...",
}


def _stage_message(decision: str) -> str:
    return _STAGE_TEXT.get(decision, "Working on it...")


# Sentinel: distinguishes "field absent" from "field present and None".
_MISSING = object()


class ChatOutcome(BaseModel):
    answer: str
    view_used: str | None = None
    # Provenance — what produced this number, and how settled is it.
    formula_version: str | None = None
    assumption_status: str | None = None
    verification: VerificationReport | None = None
    # Populated only on the investigate path. A composed answer cannot name one
    # approved view in view_used, so the ordered trace takes that role.
    trace: AgentTrace | None = None
    sql_generated: bool = False
    decision: str
    cache_hit: bool = False
    latency_ms: int = 0
    routing_ms: int = 0
    execution_ms: int = 0
    answer_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class Orchestrator:
    def __init__(
        self,
        retriever: HybridRetriever,
        index: MetadataIndex,
        executor: ReadOnlyExecutor,
        prompts: PromptManager,
        conversations: InMemoryConversationStore,
        llm: AnthropicClient,
        settings: Settings,
        result_cache=None,
        audit=None,
        telemetry=None,
        company_id: str | None = None,
        scan_views: list[str] | None = None,
        playbooks_dir=None,
        boundaries_table: str | None = None,
        boundaries_date_column: str | None = None,
    ) -> None:
        self._retriever = retriever
        self._index = index
        self._executor = executor
        self._prompts = prompts
        self._conversations = conversations
        self._llm = llm
        self._settings = settings
        self._sqlgen = SQLGenerator(llm, prompts, settings)
        self._cache = result_cache
        self._audit = audit
        # Monitoring telemetry (app.monitoring.telemetry.TraceRecorder) —
        # optional for the same reason audit is: single-company/test
        # callers keep working unmodified without it.
        self._telemetry = telemetry
        # company_id is optional so single-company/test callers keep working
        # unmodified; the company-aware call sites (CompanyContextManager)
        # always pass it, and it flows straight into every audit record.
        self._company_id = company_id
        # Falls back to the old global settings.scan_views when not given,
        # so any caller built before company-scoped config still works.
        self._scan_views = scan_views if scan_views is not None else settings.scan_views
        self._boundaries_table = boundaries_table
        self._boundaries_date_column = boundaries_date_column
        self._entities: EntityIndex | None = None
        self._playbooks = (
            PlaybookLibrary.load(playbooks_dir) if settings.playbooks_enabled else None
        )
        self._last_result = None
        self._last_provenance: tuple[str | None, str | None] = (None, None)
        self._last_verification: VerificationReport | None = None
        self._last_trace: AgentTrace | None = None
        self._last_outcome: ChatOutcome | None = None
        self._warned_no_provenance = False
        # Built lazily: the agent is off by default and costs nothing until a
        # routing decision actually asks for it.
        self._agent: ReasoningAgent | None = None

    async def handle(
        self, session_id: str, message: str,
        emit: Callable[[dict], Awaitable[None]] | None = None,
    ) -> ChatOutcome:
        """Answer a message within a hard wall-clock budget.

        Without this ceiling a slow pair of LLM calls (2 calls x timeout x
        retries) can outlive Cloudflare's 100s edge timeout. The tunnel then
        returns 524, `utils/api.ts` throws on !res.ok, and the user sees
        "Error While Responding" with no idea why — the intermittent failures
        seen in production. A graceful in-budget message beats a 524.
        """
        try:
            return await asyncio.wait_for(
                self._handle(session_id, message, emit=emit),
                timeout=self._settings.request_deadline_s,
            )
        except TimeoutError:
            log.warning("request_deadline_exceeded", session=session_id,
                        budget_s=self._settings.request_deadline_s)
            timeout_outcome = ChatOutcome(
                answer=(
                    "That question took longer than expected to work through. "
                    "Please try again — narrowing it to a specific period, "
                    "customer, or item usually helps."
                ),
                decision="timeout",
                view_used=None,
                sql_generated=False,
                cache_hit=False,
                routing_ms=0,
                execution_ms=0,
                answer_ms=0,
                input_tokens=0,
                output_tokens=0,
            )
            self._last_outcome = timeout_outcome
            if emit is not None:
                await emit({"type": "token", "value": timeout_outcome.answer})
            return timeout_outcome

    async def _handle(
        self, session_id: str, message: str,
        emit: Callable[[dict], Awaitable[None]] | None = None,
    ) -> ChatOutcome:
        token_sent = False

        async def _stage_emit(event: dict) -> None:
            # Used for stage events raised directly in _handle. Always safe
            # to call — a no-op in batch mode.
            if emit is not None:
                await emit(event)

        async def _tracking_emit(event: dict) -> None:
            # Passed to path methods and _answer_call. Tracks whether a real
            # token was streamed, so the fallback below doesn't duplicate
            # text that already went out chunk-by-chunk.
            nonlocal token_sent
            if event.get("type") == "token":
                token_sent = True
            await emit(event)

        # MUST be None (not a wrapper) when emit is None — _answer_call and
        # every path method use "is this None" to choose batch vs streaming.
        # A previous version of this passed a never-None wrapper here, which
        # silently forced every answer call into streaming mode even in
        # batch — the two live bugs this fixed are documented on
        # _answer_call and ReasoningAgent._land.
        path_emit = _tracking_emit if emit is not None else None

        start = time.perf_counter()
        tokens_in = tokens_out = 0
        self._last_cache_hit = False
        self._last_generated_sql = None
        self._last_result = None
        self._last_provenance = (None, None)
        self._last_verification = None
        self._last_trace = None
        self._t_execution = 0
        self._t_answer = 0

        self._conversations.purge_expired()
        history = self._conversations.get_history(session_id)
        boundaries = self._executor.data_boundaries(
            table=self._boundaries_table, date_column=self._boundaries_date_column
        )
        data_as_of = boundaries.last_date if boundaries else "unknown"

        async def _bypass_outcome(decision: str, result: tuple[str, int, int]) -> ChatOutcome:
            answer, t_in, t_out = result
            nonlocal tokens_in, tokens_out
            tokens_in += t_in
            tokens_out += t_out
            if emit is not None and not token_sent and answer:
                await _stage_emit({"type": "token", "value": answer})
            self._conversations.append(session_id, Message(role="user", content=message))
            self._conversations.append(session_id, Message(role="assistant", content=answer))
            outcome = ChatOutcome(
                answer=answer,
                decision=decision,
                view_used=None,
                sql_generated=False,
                cache_hit=False,
                latency_ms=int((time.perf_counter() - start) * 1000),
                routing_ms=0,
                execution_ms=0,
                answer_ms=self._t_answer,
                input_tokens=tokens_in,
                output_tokens=tokens_out,
            )
            log.info(
                "chat_request_complete",
                decision=decision, view=None, sql_generated=False,
                cache_hit=False, latency_ms=outcome.latency_ms,
                input_tokens=tokens_in, output_tokens=tokens_out,
            )
            self._last_outcome = outcome
            return outcome

        # Two bypass paths, checked BEFORE retrieval and BEFORE the routing
        # call: an explicit prefix is the user stating intent, so there is
        # nothing to infer and nothing to get wrong. This is the same fix
        # for the same underlying problem in both cases — "why did margin
        # drop" failed to reach investigate THREE separate times through the
        # router despite the routing prompt naming it as the canonical
        # trigger example; a prefix sidesteps that class of failure entirely
        # rather than trying to out-word-smith it a fourth time.
        if self._is_scan_command(message):
            return await _bypass_outcome("scan", await self._run_scan_path())

        external = self._external_question(message)
        if external is not None:
            return await _bypass_outcome(
                "external",
                await self._run_external_path(external, data_as_of=data_as_of, emit=path_emit),
            )

        investigate_question = self._investigate_prefix_question(message)
        if investigate_question is not None:
            if not self._settings.agent_enabled:
                result = (
                    ("The investigation feature isn't enabled on this server yet. "
                     "Ask a specific question instead and I'll answer from a single "
                     "view or query."),
                    0, 0,
                )
            else:
                result = await self._run_investigate_path(
                    investigate_question, history, data_as_of, session_id, emit=path_emit
                )
            return await _bypass_outcome("investigate", result)

        candidates = self._retriever.search(message, k=self._settings.top_k_views)

        # Token budget: full column lists only for the top 3 candidates —
        # that's where endpoint filters get extracted; the rest are identified
        # by name + canonical question alone.
        prompt_candidates = []
        for position, candidate in enumerate(candidates):
            data = candidate.model_dump()
            data["columns"] = candidate.columns[:15] if position < 3 else []
            prompt_candidates.append(data)

        playbooks = []
        if self._playbooks is not None:
            known = set(self._index.objects)
            playbooks = [
                {
                    "name": p.name,
                    "title": p.title,
                    "description": p.description,
                    "requires_entity": p.requires_entity,
                    "entity_kind": p.entity_kind,
                }
                for p in self._playbooks.available(known)
            ]

        working_set = self._conversations.working_set(session_id)
        working_set_prompt = [
            {
                "label": w.label,
                "source": w.source,
                "row_count": w.result.row_count,
                "columns": w.result.columns,
            }
            for w in working_set
        ]

        intent_prompt = self._prompts.render(
            "orchestrator_intent",
            data_as_of=data_as_of,
            history=[m.model_dump() for m in history],
            message=message,
            candidates=prompt_candidates,
            playbooks=playbooks,
            working_set=working_set_prompt,
            agent_available=self._settings.agent_enabled,
        )
        await _stage_emit({"type": "stage", "value": "Understanding your question..."})
        decision, response = await self._llm.structured_call(
            intent_prompt, RoutingDecision, model=self._settings.orchestrator_model
        )
        routing_ms = response.latency_ms
        tokens_in += response.input_tokens
        tokens_out += response.output_tokens
        log.info("routing_decision", decision=decision.decision, view=decision.view_name)
        await _stage_emit({"type": "stage", "value": _stage_message(decision.decision)})

        candidate_by_name = {c.view_name: c for c in candidates}

        try:
            if decision.decision == "greeting":
                # instant, zero-token reply from the externalized template
                answer = self._prompts.render("greeting_message").user
            elif decision.decision == "out_of_scope":
                answer = _OUT_OF_SCOPE_ANSWER
            elif decision.decision == "clarify":
                answer = decision.clarification or "Could you make the question more specific?"
            elif decision.decision == "run_playbook":
                answer, t_in, t_out = await self._run_playbook_path(
                    decision, message, data_as_of, emit=path_emit
                )
                tokens_in += t_in
                tokens_out += t_out
            elif decision.decision == "investigate":
                answer, t_in, t_out = await self._run_investigate_path(
                    message, history, data_as_of, session_id, emit=path_emit
                )
                tokens_in += t_in
                tokens_out += t_out
            elif decision.decision == "refine":
                answer, t_in, t_out = await self._run_refine_path(
                    decision, working_set, message, data_as_of, emit=path_emit
                )
                tokens_in += t_in
                tokens_out += t_out
            elif decision.decision == "needs_sql":
                answer, t_in, t_out = await self._run_sql_path(
                    decision, candidates, message, data_as_of, emit=path_emit
                )
                tokens_in += t_in
                tokens_out += t_out
            else:
                answer, t_in, t_out = await self._run_view_path(
                    decision, candidate_by_name, message, data_as_of, emit=path_emit
                )
                tokens_in += t_in
                tokens_out += t_out
        except EliaraError as exc:
            # Every raise in the block above is a KNOWN, anticipated failure —
            # "the model named a view/playbook it wasn't shown", "a decision
            # arrived without the field it requires" — never a surprise, and
            # every one already carries a public_message. Before this fix,
            # each one propagated uncaught: on the batch endpoint that meant
            # an HTTP 500 instead of an answer; on the streaming endpoint it
            # meant a bare {"type":"error"} with no readable text at all — a
            # dead turn. This mirrors the same fix already applied to
            # SQLValidationError/SQLExecutionError inside _run_sql_path, but
            # scoped once here instead of wrapping each raise site
            # individually, which is exactly how six of them stayed
            # unguarded through that first pass.
            log.warning(
                "routing_path_failed", decision=decision.decision, reason=exc.internal_detail
            )
            answer = exc.public_message

        # Every path delivers its text through "token" events — for the
        # narrated paths that already happened chunk-by-chunk in
        # _answer_call; instant paths (greeting, clarify, out_of_scope, a
        # refine with nothing to refine yet) never called it, so send the
        # whole answer as one token here. A streaming client's contract is
        # "the text is in token events" full stop, not "except sometimes".
        if emit is not None and not token_sent and answer:
            await _stage_emit({"type": "token", "value": answer})

        # Remember the result centrally, right before history append — covers
        # every path that sets self._last_result to a single-shape table
        # (view, sql, refine, playbook's primary_result). Investigate is
        # excluded: its trace mixes several differently-shaped results, so
        # there's no single coherent table to hand a later refinement.
        if self._last_result is not None:
            self._conversations.remember_result(session_id, self._last_result, message)
        self._conversations.append(session_id, Message(role="user", content=message))
        self._conversations.append(session_id, Message(role="assistant", content=answer))

        outcome = ChatOutcome(
            answer=answer,
            view_used=decision.view_name if decision.decision == "use_view" else None,
            sql_generated=decision.decision == "needs_sql",
            decision=decision.decision,
            formula_version=self._last_provenance[0],
            assumption_status=self._last_provenance[1],
            verification=self._last_verification,
            trace=self._last_trace,
            cache_hit=self._last_cache_hit,
            latency_ms=int((time.perf_counter() - start) * 1000),
            routing_ms=routing_ms,
            execution_ms=self._t_execution,
            answer_ms=self._t_answer,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
        )
        if self._audit is not None:
            self._audit.record(
                company_id=self._company_id or "unknown",
                session_id=session_id,
                question=message,
                decision=outcome.decision,
                view_used=outcome.view_used,
                generated_sql=self._last_generated_sql,
                cache_hit=outcome.cache_hit,
                latency_ms=outcome.latency_ms,
                input_tokens=outcome.input_tokens,
                output_tokens=outcome.output_tokens,
                answer=outcome.answer,
            )
        if self._telemetry is not None:
            self._telemetry.record(
                RequestTrace(
                    request_id=get_request_id() or "unknown",
                    timestamp=datetime.now(UTC).isoformat(),
                    company_id=self._company_id,
                    session_id=session_id,
                    question=message,
                    route=outcome.decision,
                    selected_view=outcome.view_used,
                    sql_generated=outcome.sql_generated,
                    routing_ms=outcome.routing_ms,
                    execution_ms=outcome.execution_ms,
                    answer_ms=outcome.answer_ms,
                    latency_ms=outcome.latency_ms,
                    input_tokens=outcome.input_tokens,
                    output_tokens=outcome.output_tokens,
                    row_count=self._last_result.row_count if self._last_result else None,
                    empty_result=(self._last_result.row_count == 0) if self._last_result else None,
                    clarification=outcome.decision == "clarify",
                    status="success",
                    data_as_of=data_as_of,
                )
            )
        log.info(
            "chat_request_complete",
            decision=outcome.decision,
            view=outcome.view_used,
            sql_generated=outcome.sql_generated,
            cache_hit=outcome.cache_hit,
            latency_ms=outcome.latency_ms,
            input_tokens=outcome.input_tokens,
            output_tokens=outcome.output_tokens,
        )
        self._last_outcome = outcome
        return outcome

    async def _answer_call(
        self, prompt, *, model: str, max_tokens: int,
        emit: Callable[[dict], Awaitable[None]] | None = None,
        web_search: bool = False,
    ):
        """The one call site every answer-producing path routes its final
        narration through. Batch mode (emit=None) is byte-identical to the
        old direct `self._llm.call(...)` — nothing about the non-streaming
        path changes. Streaming mode emits a "token" event per chunk as the
        model generates it, so the same prompt produces the same final
        LLMResponse either way; only the delivery differs.
        """
        # web_search is passed ONLY when actually on. Every analytics path
        # calls with it off, and those calls must stay byte-identical to what
        # they were before this capability existed — the external path is
        # additive, not a change to the paths that carry governed data.
        extra = {"web_search": True} if web_search else {}
        if emit is None:
            return await self._llm.call(
                prompt, model=model, max_tokens=max_tokens, **extra
            )
        parts: list[str] = []
        final = None
        if web_search:
            extra["max_searches"] = self._settings.external_max_searches
        async for chunk in self._llm.stream(
            prompt, model=model, max_tokens=max_tokens, **extra
        ):
            if chunk.text:
                parts.append(chunk.text)
                await emit({"type": "token", "value": chunk.text})
            if chunk.done:
                final = chunk.response
        final.text = "".join(parts)
        return final

    async def stream(self, session_id: str, message: str):
        """Async generator of SSE-ready event dicts for one turn, matching
        the frontend's exact contract:

          {"type": "stage", "value": "<human-readable loading text>"}
          {"type": "token", "value": "<answer text, one chunk>"}
          {"type": "visual", "value": {...}}   (only when a chart applies)
          {"type": "done"}                     (always last, no payload)
          {"type": "error", "detail": str}     (in place of done, on failure)

        Every path delivers its answer through "token" events — narrated
        paths stream chunk-by-chunk from `_answer_call`; instant paths
        (greeting, clarify, a refine with nothing to refine yet) send the
        whole answer as one token, from the fallback in `_handle`. A client
        reading this stream never needs to fall back to `done` for text —
        `done` carries nothing but the fact that the stream is over.

        `visual` reuses the exact same chart-shape detection as the legacy
        `/ask` endpoint (`app.execution.visual.build_visual`) — same trend/
        ranking/table logic, same reasoning for why distribution is absent.

        Bridges the callback-style `emit` used internally into a generator
        the API layer can iterate: `_handle` runs as a background task and
        pushes events onto a queue as it goes; this pulls them off in real
        time. `handle()` is untouched by this — it is still the direct,
        simplest path for any caller that doesn't need progressive delivery.

        The wall-clock deadline in handle() still applies — a turn that would
        time out in the batch path times out here too: the timeout message
        arrives as a token event, then done, same as any other turn.
        """
        queue: asyncio.Queue = asyncio.Queue()
        _DONE = object()

        async def _emit(event: dict) -> None:
            await queue.put(event)

        async def _run() -> None:
            try:
                outcome = await self.handle(session_id, message, emit=_emit)
                visual = build_visual(self._last_result, outcome.view_used)
                if visual is not None:
                    await queue.put({"type": "visual", "value": visual})
                await queue.put({"type": "done"})
            except Exception as exc:  # noqa: BLE001 - surface to the client, don't crash the stream
                log.warning("stream_failed", error=type(exc).__name__)
                await queue.put({"type": "error", "detail": type(exc).__name__})
            finally:
                await queue.put(_DONE)

        task = asyncio.create_task(_run())
        try:
            while True:
                item = await queue.get()
                if item is _DONE:
                    break
                yield item
        finally:
            if not task.done():
                task.cancel()

    def _cached_run(self, key: tuple, execute):
        if self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None:
                self._last_cache_hit = True
                log.info("result_cache_hit", cache_scope=str(key[:2]))
                return cached
        result = execute()
        if self._cache is not None:
            self._cache.set(key, result)
        return result


    # ------------------------------------------------------------- entities
    def _resolve_filters(
        self, filters: dict[str, str]
    ) -> tuple[dict[str, str], str | None]:
        """Map user-typed names onto stored values before querying.

        Returns (filters, clarification). When a clarification is returned the
        query is NOT run — an ambiguous or unknown entity is answered directly
        and deterministically, which is both faster and more honest than
        running a filter we know matches nothing.
        """
        resolved: dict[str, str] = {}
        for column, value in filters.items():
            outcome: Resolution = self._entities.resolve(column, value)
            if outcome.status == "resolved":
                log.info(
                    "entity_resolved",
                    column=column,
                    requested=outcome.requested,
                    matched=outcome.value,
                )
            elif outcome.status == "ambiguous":
                options = "\n".join(f"- {c}" for c in outcome.candidates)
                return filters, (
                    f'Several records match "{outcome.requested}". '
                    f"Which one do you mean?\n\n{options}\n\n"
                    "You can reply with the full name or the code."
                )
            elif outcome.status == "unknown":
                log.info("entity_unknown", column=column, requested=outcome.requested)
                return filters, (
                    f'I could not find "{outcome.requested}" in the '
                    f"{column.rsplit('_', 1)[0].replace('_', ' ')} records. "
                    "Please check the spelling, or give me the code instead."
                )
            resolved[column] = outcome.value or value
        return resolved, None


    # ---------------------------------------------------- glossary + checking
    def _glossary_for(self, message: str) -> str | None:
        """Definitions for glossary terms the question actually mentions.

        The business glossary is loaded at startup and shown to the answer
        model so a term like "dead stock" or "churn risk" is interpreted
        from the company's own definition, not the model's general
        knowledge. Two-tier match: an exact substring match (handles
        single-word terms and exact phrases, same as before) plus a
        distinctive-word match so a multi-word term like "customer churn"
        still matches a question that only says "churn" — a plain substring
        check only ever matched in the other direction (term inside the
        full question), so a question shorter than the term never matched.
        """
        glossary = getattr(self._index, "glossary", None)
        if not glossary:
            return None
        lowered = message.lower()
        message_words = set(re.findall(r"[a-z0-9]+", lowered))
        hits = []
        for term, definition in glossary.items():
            if not term:
                continue
            term_lower = term.lower()
            term_words = set(re.findall(r"[a-z0-9]+", term_lower))
            matched = term_lower in lowered
            if not matched:
                distinctive = {w for w in term_words if len(w) >= 4}
                matched = bool(distinctive & message_words)
            if matched:
                hits.append(f"- {term}: {definition}")
        return "\n".join(hits[:5]) if hits else None

    def _verified(self, answer: str, result, message: str) -> str:
        """Ground the narrative's figures in the result set after generation."""
        report = verify(
            answer,
            result,
            question=message,
            strict=self._settings.verification_strict,
        )
        self._last_verification = report
        if report.status == "warn":
            log.warning(
                "answer_verification_warn",
                checked=report.checked,
                grounded=report.grounded,
                ungrounded=report.ungrounded,
            )
            if self._settings.verification_strict:
                return answer + (report.caveat() or "")
        return answer


    def _investigate_prefix_question(self, message: str) -> str | None:
        """Same contract as _external_question: standalone leading token,
        case-insensitive, or no match at all — never a near-miss."""
        prefix = self._settings.investigate_prefix.lower()
        stripped = message.strip()
        lowered = stripped.lower()
        if not lowered.startswith(prefix):
            return None
        rest = stripped[len(prefix):]
        if rest and not rest[0].isspace():
            return None
        return rest.strip() or "give me a general assessment based on the data available"

    def _is_scan_command(self, message: str) -> bool:
        """/scan takes no argument — it's a command, not a question. Still a
        standalone-token match like the other prefixes, so "/scanning" or
        similar can't accidentally trigger it."""
        if not self._settings.scan_enabled:
            return False
        prefix = self._settings.scan_prefix.lower()
        stripped = message.strip().lower()
        return stripped == prefix or stripped.startswith(prefix + " ")

    async def _run_scan_path(self) -> tuple[str, int, int]:
        """Rank every pre-registered view and narrate the HIGH-tier findings
        in plain text — built entirely in Python, ZERO LLM calls. Not a
        design shortcut: every number here already comes straight from a
        governed view, so there is nothing for a model to add except risk.
        A wrong reorder recommendation is a much bigger liability than a
        wrong prose answer, so this path can't produce one.

        Best-effort per view — a view that doesn't exist on this deployment,
        or errors, is skipped and logged, not a failure for the whole scan.
        The real production names for these views were never confirmed
        beyond app/orchestrator/definitions/stock_action_plan.yaml itself;
        an operator changes settings.scan_views once those are verified.
        """
        queues: list[AttentionQueue] = []
        skipped: list[str] = []
        for view_name in self._scan_views:
            try:
                result = self._executor.run_view(view_name, limit=200)
            except EliaraError as exc:
                log.warning("scan_view_unavailable", view=view_name, reason=exc.internal_detail)
                skipped.append(view_name)
                continue
            queue = build_attention_queue(
                result, view_name, max_items=self._settings.scan_max_items_per_view
            )
            if queue is not None:
                queues.append(queue)

        if not queues:
            return (
                ("No configured views returned rankable data — nothing to report. "
                "This usually means the view names in scan_views need updating "
                "for this deployment."),
                0, 0,
            )

        high_items = [(q, item) for q in queues for item in q.items if item.tier == "HIGH"]
        lines = [
            (f"Scanned {len(queues)} view{'s' if len(queues) != 1 else ''} — "
            f"{len(high_items)} item{'s' if len(high_items) != 1 else ''} at the top tier.")
        ]
        if high_items:
            lines.append("")
            lines.append("Needs attention:")
            for queue, item in high_items[:15]:
                lines.append(f"- {item.label} — {item.value:,.0f} ({queue.value_column}, {queue.view_name})")
        if skipped:
            lines.append("")
            lines.append(f"({len(skipped)} configured view(s) unavailable on this deployment)")
        lines.append("")
        lines.append(
            "Ask about any item by name for more detail, or /investigate a "
            "question for a deeper look."
        )
        return "\n".join(lines), 0, 0

    # ------------------------------------------------------------- external
    def _external_question(self, message: str) -> str | None:
        """Return the question with the prefix stripped, or None.

        Case-insensitive, and the prefix must be a standalone leading token —
        "/searching for a customer" is NOT a match, so a real analytics
        question can't be swallowed by a near-miss.
        """
        if not self._settings.external_enabled:
            return None
        prefix = self._settings.external_prefix.lower()
        stripped = message.strip()
        lowered = stripped.lower()
        if not lowered.startswith(prefix):
            return None
        rest = stripped[len(prefix):]
        if rest and not rest[0].isspace():
            return None
        return rest.strip()

    async def _run_external_path(
        self, question: str, data_as_of: str,
        emit: Callable[[dict], Awaitable[None]] | None = None,
    ) -> tuple[str, int, int]:
        """Answer from general knowledge / the web — never from the database.

        Deliberately does NOT: write to the working set (there's no result to
        refine), produce a visual (no rows), or run verification (verify()
        grounds figures against a QueryResult, and there isn't one — running
        it here would flag every legitimate external number as fabricated).
        """
        if not question:
            return (
                (f"Add a question after {self._settings.external_prefix} — for example "
                f"\"{self._settings.external_prefix} who are the main OEM headlamp "
                "manufacturers in China?\""),
                0,
                0,
            )

        await self._emit_stage(emit, "Looking that up...")
        prompt = self._prompts.render(
            "external_answer",
            message=question,
            web_search=self._settings.external_web_search,
            char_budget=self._settings.answer_char_budget,
        )
        response = await self._answer_call(
            prompt,
            model=self._settings.orchestrator_model,
            max_tokens=1500,
            emit=emit,
            web_search=self._settings.external_web_search,
        )
        self._t_answer = response.latency_ms
        log.info(
            "external_answer_complete",
            web_search=self._settings.external_web_search,
            output_tokens=response.output_tokens,
        )
        return response.text, response.input_tokens, response.output_tokens

    @staticmethod
    async def _emit_stage(emit, text: str) -> None:
        if emit is not None:
            await emit({"type": "stage", "value": text})

    # --------------------------------------------------------------- refine
    async def _run_refine_path(
        self, decision: RoutingDecision, working_set: list[ResultEntry],
        message: str, data_as_of: str,
        emit: Callable[[dict], Awaitable[None]] | None = None,
    ) -> tuple[str, int, int]:
        """Sort / filter / limit a result already in memory. No query, no SQL
        generation — see app/execution/refine.py for why this is safe and why
        it's deliberately narrow."""
        if decision.refine is None:
            raise RoutingError(internal_detail="refine decision without a refine spec")
        if not (0 <= decision.refine_target < len(working_set)):
            return (
                ("I don't have an earlier result to refine — could you ask the "
                 "full question?"),
                0,
                0,
            )

        entry = working_set[decision.refine_target]
        try:
            result = apply_refinement(entry.result, decision.refine)
        except RefineError as exc:
            log.info("refine_rejected", column=exc.column, available=exc.available)
            return (
                (f'"{exc.column}" isn\'t in that result — it has '
                 f"{', '.join(exc.available)}. Could you rephrase?"),
                0,
                0,
            )

        self._last_result = result
        self._last_provenance = (None, None)
        self._t_execution = 0
        payload, shown = to_llm_payload(result, max_chars=self._settings.payload_max_chars)
        stats = summarise(result, shown)

        answer_prompt = self._prompts.render(
            "orchestrator_answer",
            data_as_of=data_as_of,
            caution=None,
            message=message,
            source=f"{entry.source} (previous result, refined — no new query)",
            truncated=result.truncated or shown < result.row_count,
            row_count=shown,
            data=payload,
            stats=stats,
            glossary=self._glossary_for(message),
            char_budget=self._settings.answer_char_budget,
            max_bullets=self._settings.answer_max_bullets,
        )
        response = await self._answer_call(
            answer_prompt, model=self._settings.orchestrator_model, max_tokens=1200,
            emit=emit,
        )
        self._t_answer = response.latency_ms
        answer = self._verified(response.text, result, message)
        return answer, response.input_tokens, response.output_tokens

    # ----------------------------------------------------------- investigate
    def _build_agent(self) -> ReasoningAgent:
        """Wire the loop to the SAME collaborators the pipeline uses.

        Nothing new is granted here — the agent reaches the catalogue through
        the same retriever, the database through the same read-only executor,
        and generated SQL through the same Haiku + AST gate.
        """

        return ReasoningAgent(
            retriever=self._retriever,
            index=self._index,
            executor=self._executor,
            prompts=self._prompts,
            llm=self._llm,
            settings=self._settings,
            entities=self._entities,
            sqlgen=self._sqlgen,
            result_cache=self._cache,
        )

    async def _run_investigate_path(
        self, message: str, history, data_as_of: str, _session_id: str | None = None,
        emit: Callable[[dict], Awaitable[None]] | None = None,
    ) -> tuple[str, int, int]:
        """Hand the turn to the bounded reasoning loop.

        Reached only when the model returns decision="investigate" AND the
        agent is enabled. With the flag off the intent prompt never offers the
        verdict, so this is unreachable in the default configuration.
        """
        if not self._settings.agent_enabled:
            raise RoutingError(
                internal_detail="model chose investigate while the agent is disabled"
            )
        if self._agent is None:
            self._agent = self._build_agent()

        import time as _time

        started = _time.perf_counter()
        working_set = self._conversations.working_set(_session_id) if _session_id else []
        answer, trace = await self._agent.run(
            message, history=history, data_as_of=data_as_of, working_set=working_set,
            emit=emit,
        )
        self._t_answer = int((_time.perf_counter() - started) * 1000)
        self._last_trace = trace
        self._last_generated_sql = next(
            (c.generated_sql for c in trace.calls if c.generated_sql), None
        )
        self._last_provenance = (None, None)
        log.info(
            "investigate_complete",
            steps=trace.steps_used,
            stopped=trace.stopped_because,
            views=trace.views_used,
            unvalidated=trace.unvalidated,
        )
        return answer, trace.input_tokens, trace.output_tokens

    # ------------------------------------------------------------- playbooks
    async def _run_playbook_path(
        self, decision: RoutingDecision, message: str, data_as_of: str,
        emit: Callable[[dict], Awaitable[None]] | None = None,
    ) -> tuple[str, int, int]:
        """Run a multi-step workflow and synthesise it in ONE call.

        Every step is a curated view chosen by a human, so the model never
        picks queries here — it only picked the playbook. Steps execute in
        Python and are aggregated in Python; a single synthesis call writes
        the briefing.
        """
        playbook = self._playbooks.get(decision.playbook or "") if self._playbooks else None
        if playbook is None:
            raise RoutingError(
                internal_detail=f"model chose unknown playbook: {decision.playbook!r}"
            )

        entity = decision.playbook_entity
        if playbook.requires_entity:
            if not entity:
                return (
                    (f"Which {playbook.entity_kind or 'record'} would you like me to "
                    "look at? Give me a name or a code."),
                    0,
                    0,
                )
            if self._entities is not None:
                column = f"{playbook.entity_kind or 'customer'}_name"
                outcome = self._entities.resolve(column, entity)
                if outcome.status == "ambiguous":
                    options = "\n".join(f"- {c}" for c in outcome.candidates)
                    return (
                        f'Several records match "{entity}". Which one?\n\n{options}',
                        0,
                        0,
                    )
                if outcome.status == "unknown":
                    return (
                        (f'I could not find "{entity}" in the records. '
                        "Please check the spelling, or give me the code instead."),
                        0,
                        0,
                    )
                entity = outcome.value or entity

        import time as _time

        started = _time.perf_counter()
        run = run_playbook(
            playbook, self._executor, set(self._index.objects), entity
        )
        self._t_execution = int((_time.perf_counter() - started) * 1000)
        self._last_result = run.primary_result
        self._last_provenance = (None, None)

        if not run.any_data:
            return (
                (f"I ran the {playbook.title.lower()} but found no data for it in "
                "the current dataset."),
                0,
                0,
            )

        skipped = [s for s in run.steps if s.status == "missing"]
        coverage = None
        if skipped:
            coverage = (
                f"{len(skipped)} of {len(run.steps)} sections are unavailable in "
                "this dataset and were skipped."
            )

        answer_prompt = self._prompts.render(
            "orchestrator_answer",
            data_as_of=data_as_of,
            caution=coverage,
            message=message,
            source=f"{playbook.title.lower()} ({len(run.steps) - len(skipped)} sections)",
            truncated=False,
            row_count=0,
            data=run.payload,
            stats=None,
            glossary=self._glossary_for(message),
            char_budget=self._settings.answer_char_budget + 600,
            max_bullets=self._settings.answer_max_bullets + 2,
            playbook_synthesis=playbook.synthesis,
        )
        response = await self._answer_call(
            answer_prompt, model=self._settings.orchestrator_model, max_tokens=1600,
            emit=emit,
        )
        self._t_answer = response.latency_ms
        answer = self._verified(response.text, run.primary_result, message)
        return answer, response.input_tokens, response.output_tokens

    async def _run_sql_path(
        self, decision: RoutingDecision, candidates, message: str, data_as_of: str,
        emit: Callable[[dict], Awaitable[None]] | None = None,
    ) -> tuple[str, int, int]:
        if decision.sql_request is None:
            raise RoutingError(internal_detail="needs_sql decision without sql_request")

        # The routing candidates were retrieved with the raw (possibly fragment)
        # message. task_description is the fully-resolved question — re-retrieve
        # with it so the schema slice contains what the COMBINED question needs.
        slice_candidates = self._retriever.search(
            decision.sql_request.task_description, k=self._settings.top_k_views
        )
        schema_slice = build_slice(
            self._index, decision.sql_request.tables, slice_candidates + list(candidates)
        )
        if not schema_slice:
            raise RoutingError(internal_detail="needs_sql with no usable schema objects")

        whitelist = self._index.whitelist
        tokens_in = tokens_out = 0
        validated = None
        last_error = ""
        for attempt in range(2):
            sql_text, response = await self._sqlgen.generate(
                decision.sql_request.task_description,
                schema_slice,
                previous_error=last_error or None,
            )
            tokens_in += response.input_tokens
            tokens_out += response.output_tokens
            try:
                validated = validate_sql(sql_text, whitelist, self._settings.max_rows)
                break
            except SQLValidationError as exc:
                last_error = exc.internal_detail
                log.warning("sql_rejected", attempt=attempt + 1, reason=last_error)
        if validated is None:
            # Two corrective attempts failed — this is an EXPECTED outcome for a
            # question the schema genuinely can't answer safely, not a crash.
            # Returning the message here (rather than raising) means it reaches
            # the user through the exact same channel every other path uses:
            # the batch answer field, and the streaming layer's fallback token
            # event. Before this fix, this path raised, which on the streaming
            # endpoint produced a bare {"type":"error"} with no user-facing text
            # at all — "No content" in the client, functionally a dead turn.
            log.warning("sql_rejected_final", reason=last_error)
            return (
                SQLValidationError(internal_detail=last_error).public_message,
                tokens_in, tokens_out,
            )

        log.info("sql_accepted", tables=validated.tables, sql=validated.sql)
        self._last_generated_sql = validated.sql
        import hashlib

        sql_key = ("sql", hashlib.sha1(validated.sql.encode()).hexdigest())
        try:
            result = self._cached_run(sql_key, lambda: self._executor.run_sql(validated.sql))
        except SQLExecutionError as exc:
            # A query that passed validation can still fail at runtime — e.g.
            # syntax SQLite's installed version doesn't support (FULL OUTER JOIN
            # predates SQLite 3.39). Same reasoning as the validation-rejected
            # case above: this is a real, expected failure mode, so it becomes
            # an answer, not a crash.
            log.warning("sql_execution_failed", reason=exc.internal_detail)
            return exc.public_message, tokens_in, tokens_out
        self._last_result = result
        self._t_execution = 0 if self._last_cache_hit else result.elapsed_ms
        payload, shown = to_llm_payload(result, max_chars=self._settings.payload_max_chars)
        stats = summarise(result, shown)

        answer_prompt = self._prompts.render(
            "orchestrator_answer",
            data_as_of=data_as_of,
            caution=_SQL_CAUTION,
            message=message,
            source="custom query",
            truncated=result.truncated or shown < result.row_count,
            row_count=shown,
            data=payload,
            stats=stats,
            glossary=self._glossary_for(message),
            char_budget=self._settings.answer_char_budget,
            max_bullets=self._settings.answer_max_bullets,
        )
        response = await self._answer_call(
            answer_prompt, model=self._settings.orchestrator_model, max_tokens=1200,
            emit=emit,
        )
        self._t_answer = response.latency_ms
        response.text = self._verified(response.text, result, message)
        tokens_in += response.input_tokens
        tokens_out += response.output_tokens
        return response.text, tokens_in, tokens_out

    async def _run_view_path(
        self, decision: RoutingDecision, candidate_by_name, message: str, data_as_of: str,
        emit: Callable[[dict], Awaitable[None]] | None = None,
    ) -> tuple[str, int, int]:
        candidate = candidate_by_name.get(decision.view_name or "")
        if candidate is None:
            # The model must choose among what it was shown — never invent names.
            raise RoutingError(
                internal_detail=f"model chose non-candidate view: {decision.view_name!r}"
            )

        valid_columns = set(candidate.columns)
        filters = {
            col: value
            for col, value in decision.endpoint_filters.items()
            if col in valid_columns and value
        }
        if candidate.requires_endpoint_filter and not filters:
            return (
                ("Please specify which item or customer you mean (a code or exact "
                "name), and I'll pull the details."),
                0,
                0,
            )

        if filters and self._entities is not None:
            filters, clarify = self._resolve_filters(filters)
            if clarify:
                return clarify, 0, 0

        view_key = ("view", candidate.view_name, tuple(sorted(filters.items())))
        result: QueryResult = self._cached_run(
            view_key, lambda: self._executor.run_view(candidate.view_name, filters or None)
        )
        self._last_result = result
        self._t_execution = 0 if self._last_cache_hit else result.elapsed_ms
        payload, shown = to_llm_payload(result, max_chars=self._settings.payload_max_chars)
        stats = summarise(result, shown)

        # Provenance is decoration on the answer, not part of it. Read it
        # defensively so a partially-updated deployment (an older ViewCandidate
        # without formula_version) degrades to "no lineage shown" instead of
        # failing the whole request. The mismatch is logged once so it stays
        # visible rather than silently permanent.
        formula_version = getattr(candidate, "formula_version", _MISSING)
        if formula_version is _MISSING:
            formula_version = None
            if not self._warned_no_provenance:
                self._warned_no_provenance = True
                log.warning(
                    "provenance_unavailable",
                    reason="ViewCandidate has no formula_version",
                    hint="app/core/models.py is out of date; run scripts/check_install.py",
                )
        self._last_provenance = (
            formula_version,
            getattr(candidate, "assumption_status", None),
        )

        caution = None
        if candidate.assumption_status and candidate.assumption_status != "APPROVED_LOGIC":
            caution = candidate.assumption_status

        answer_prompt = self._prompts.render(
            "orchestrator_answer",
            data_as_of=data_as_of,
            caution=caution,
            message=message,
            source="curated analytics",
            truncated=result.truncated or shown < result.row_count,
            row_count=shown,
            data=payload,
            stats=stats,
            glossary=self._glossary_for(message),
            char_budget=self._settings.answer_char_budget,
            max_bullets=self._settings.answer_max_bullets,
        )
        response = await self._answer_call(
            answer_prompt, model=self._settings.orchestrator_model, max_tokens=1200,
            emit=emit,
        )
        self._t_answer = response.latency_ms
        answer = self._verified(response.text, result, message)
        return answer, response.input_tokens, response.output_tokens
