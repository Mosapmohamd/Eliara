"""Orchestrator pipeline tests with fake collaborators (no network, no DB I/O
beyond the fixture executor)."""

import pytest

from app.core.config import Settings
from app.discovery.embedder import HashingEmbedder
from app.discovery.index import MetadataIndex
from app.discovery.metadata_loader import MetadataLoader
from app.discovery.search import HybridRetriever
from app.llm.anthropic_client import LLMResponse
from app.orchestrator.conversation import InMemoryConversationStore
from app.orchestrator.decision_models import RoutingDecision
from app.orchestrator.orchestrator import Orchestrator
from app.prompts.loader import PromptManager


class FakeLLM:
    """structured_call -> fixed decision; call -> pops from a text queue."""

    def __init__(self, decision: RoutingDecision, call_texts: list[str] | None = None):
        self.decision = decision
        self.call_texts = list(call_texts or ["Business answer."])
        self.intent_prompts = []
        self.call_prompts = []

    async def structured_call(self, prompt, output_model, *, model, max_tokens=1000):
        self.intent_prompts.append(prompt)
        return self.decision, LLMResponse(
            text="{}", model=model, input_tokens=100, output_tokens=20,
            latency_ms=5, prompt_tag=prompt.tag,
        )

    async def call(self, prompt, *, model, max_tokens=1500, temperature=0.2, **kwargs):
        self.call_prompts.append(prompt)
        text = self.call_texts.pop(0) if len(self.call_texts) > 1 else self.call_texts[0]
        return LLMResponse(
            text=text, model=model, input_tokens=200, output_tokens=50,
            latency_ms=5, prompt_tag=prompt.tag,
        )

    @property
    def answer_prompts(self):
        return [p for p in self.call_prompts if p.name == "orchestrator_answer"]

    @property
    def sqlgen_prompts(self):
        return [p for p in self.call_prompts if p.name == "sqlgen_generate"]


@pytest.fixture()
def discovery(executor):
    objects, registry, glossary, fp = MetadataLoader(executor).load()
    index = MetadataIndex(objects, registry, glossary, fp)
    return index, HybridRetriever(index, HashingEmbedder())


def _orchestrator(discovery, executor, llm) -> Orchestrator:
    index, retriever = discovery
    return Orchestrator(
        retriever=retriever,
        index=index,
        executor=executor,
        prompts=PromptManager(),
        conversations=InMemoryConversationStore(),
        llm=llm,
        settings=Settings(_env_file=None),
        boundaries_table="fact_ai_sales_net",
        boundaries_date_column="posting_date_iso",
    )


async def test_use_view_happy_path(discovery, executor):
    llm = FakeLLM(RoutingDecision(
        decision="use_view",
        view_name="vw_q002_top_10_customers_by_lifetime_revenue",
    ))
    orch = _orchestrator(discovery, executor, llm)
    outcome = await orch.handle("s1", "who are our top customers?")

    assert outcome.decision == "use_view"
    assert outcome.view_used == "vw_q002_top_10_customers_by_lifetime_revenue"
    assert outcome.answer == "Business answer."
    assert outcome.input_tokens == 300 and outcome.output_tokens == 70
    # the answer prompt received real data from the fixture DB
    assert "Beta Motors" in llm.answer_prompts[0].user
    # data_as_of anchored to the data, not the calendar
    assert "2026-06-27" in llm.intent_prompts[0].system
    # history recorded
    assert len(orch._conversations.get_history("s1")) == 2


async def test_caution_wording_for_unapproved_logic(discovery, executor):
    llm = FakeLLM(RoutingDecision(
        decision="use_view",
        view_name="vw_q011_items_dead_stock_or_severe_dead_stock",
    ))
    orch = _orchestrator(discovery, executor, llm)
    await orch.handle("s1", "dead stock?")
    assert "indicative" in llm.answer_prompts[0].system  # DATA_SCIENCE_REVIEW_REQUIRED


async def test_hallucinated_view_rejected(discovery, executor):
    """The model must choose among what it was shown — this guard is
    correct and must stay. What changed: it used to raise uncaught, which
    on the streaming endpoint produced a bare {"type":"error"} with no
    readable text (a real production bug). Now it degrades to the error's
    own public_message, delivered as an ordinary answer — the safety
    boundary is unchanged, only the failure mode got graceful."""
    llm = FakeLLM(RoutingDecision(decision="use_view", view_name="vw_q999_made_up"))
    orch = _orchestrator(discovery, executor, llm)

    outcome = await orch.handle("s1", "top customers?")

    assert "could not interpret this request" in outcome.answer


async def test_invalid_filter_columns_dropped(discovery, executor):
    llm = FakeLLM(RoutingDecision(
        decision="use_view",
        view_name="vw_q002_top_10_customers_by_lifetime_revenue",
        endpoint_filters={"customer_code": "C001", "evil; DROP": "x", "not_a_col": "y"},
    ))
    orch = _orchestrator(discovery, executor, llm)
    outcome = await orch.handle("s1", "top customers C001")
    assert outcome.view_used  # executed without error -> bad keys were dropped


async def test_clarify_and_out_of_scope_skip_execution(discovery, executor):
    llm = FakeLLM(RoutingDecision(decision="clarify", clarification="Which item code?"))
    orch = _orchestrator(discovery, executor, llm)
    outcome = await orch.handle("s1", "show me the forecast")
    assert outcome.answer == "Which item code?"
    assert outcome.view_used is None
    assert llm.answer_prompts == []  # Sonnet #2 never called

    llm2 = FakeLLM(RoutingDecision(decision="out_of_scope"))
    orch2 = _orchestrator(discovery, executor, llm2)
    outcome2 = await orch2.handle("s1", "write me a poem")
    assert "outside that scope" in outcome2.answer


async def test_needs_sql_happy_path(discovery, executor):
    from app.orchestrator.decision_models import SQLRequestSpec

    llm = FakeLLM(
        RoutingDecision(
            decision="needs_sql",
            sql_request=SQLRequestSpec(
                task_description="average net revenue per warehouse",
                tables=["fact_ai_sales_net"],
            ),
        ),
        call_texts=[
            ("SELECT warehouse_name, AVG(net_revenue) AS avg_rev "
            "FROM fact_ai_sales_net GROUP BY warehouse_name"),
            "Custom analysis answer.",
        ],
    )
    orch = _orchestrator(discovery, executor, llm)
    outcome = await orch.handle("s1", "average invoice value per warehouse?")

    assert outcome.sql_generated is True
    assert outcome.view_used is None
    assert outcome.answer == "Custom analysis answer."
    # generated-SQL answers always carry the measured-confidence wording
    assert "indicative" in llm.answer_prompts[0].system
    # Haiku isolation: raw user message never reaches the sqlgen prompt
    assert "average invoice value per warehouse?" not in llm.sqlgen_prompts[0].user
    assert "average net revenue per warehouse" in llm.sqlgen_prompts[0].user


async def test_needs_sql_retry_after_rejection(discovery, executor):
    from app.orchestrator.decision_models import SQLRequestSpec

    llm = FakeLLM(
        RoutingDecision(
            decision="needs_sql",
            sql_request=SQLRequestSpec(task_description="x", tables=["fact_ai_sales_net"]),
        ),
        call_texts=[
            "SELECT * FROM forbidden_secret_table",           # rejected
            "SELECT customer_code FROM fact_ai_sales_net",     # corrected
            "Recovered answer.",
        ],
    )
    orch = _orchestrator(discovery, executor, llm)
    outcome = await orch.handle("s1", "some custom question")
    assert outcome.answer == "Recovered answer."
    assert "rejected" in llm.sqlgen_prompts[1].user  # correction fed back


async def test_needs_sql_double_rejection_degrades_gracefully(discovery, executor):
    """Was: raises SQLValidationError, which on the streaming endpoint produced
    a bare {"type":"error"} with no user-facing text at all — a dead turn with
    "No content" in the client. Now: two failed corrections is treated as an
    expected outcome (not every question has a safe query), so the error's
    own public_message becomes the answer — reaching the user through the
    exact same channel every other path uses, including the streaming
    fallback-token mechanism."""
    from app.orchestrator.decision_models import SQLRequestSpec

    llm = FakeLLM(
        RoutingDecision(
            decision="needs_sql",
            sql_request=SQLRequestSpec(task_description="x", tables=["fact_ai_sales_net"]),
        ),
        call_texts=["DROP TABLE dim_b3_item", "SELECT nope FROM nowhere", "unused"],
    )
    orch = _orchestrator(discovery, executor, llm)
    outcome = await orch.handle("s1", "bad custom question")

    assert outcome.decision == "needs_sql"
    assert "could not be translated into a safe query" in outcome.answer


async def test_sql_execution_failure_degrades_gracefully(discovery, executor, monkeypatch):
    """Validated SQL can still fail at runtime (e.g. a syntax the deployed
    SQLite build doesn't support) — same graceful-degradation contract as a
    rejected-twice validation failure: the error's public_message becomes
    the answer instead of the exception propagating to a dead stream."""
    from app.core.errors import SQLExecutionError
    from app.orchestrator.decision_models import SQLRequestSpec

    def _boom(sql, params=()):
        raise SQLExecutionError(internal_detail="simulated runtime failure")

    monkeypatch.setattr(executor, "run_sql", _boom)

    llm = FakeLLM(
        RoutingDecision(
            decision="needs_sql",
            sql_request=SQLRequestSpec(task_description="x", tables=["fact_ai_sales_net"]),
        ),
        call_texts=["SELECT customer_name FROM fact_ai_sales_net"],
    )
    orch = _orchestrator(discovery, executor, llm)
    outcome = await orch.handle("s1", "a query that fails at runtime")

    assert outcome.decision == "needs_sql"
    assert "could not be completed" in outcome.answer


async def test_history_reaches_intent_prompt(discovery, executor):
    llm = FakeLLM(RoutingDecision(decision="clarify", clarification="?"))
    orch = _orchestrator(discovery, executor, llm)
    await orch.handle("s1", "first question")
    await orch.handle("s1", "second question")
    assert "first question" in llm.intent_prompts[1].user


async def test_expired_sessions_purged_on_request(discovery, executor):
    llm = FakeLLM(RoutingDecision(decision="clarify", clarification="?"))
    orch = _orchestrator(discovery, executor, llm)
    orch._conversations = InMemoryConversationStore(ttl_min=0)
    from app.orchestrator.conversation import Message

    orch._conversations.append("stale", Message(role="user", content="old"))
    import asyncio

    await asyncio.sleep(0.01)
    await orch.handle("fresh", "hello")
    assert orch._conversations.get_history("stale") == []


async def test_sql_slice_built_from_resolved_task_not_fragment(discovery, executor):
    """Regression: the fragment 'exclude dead stock' skewed the schema slice.
    The slice must be retrieved with the resolved task_description."""
    from app.orchestrator.decision_models import SQLRequestSpec

    llm = FakeLLM(
        RoutingDecision(
            decision="needs_sql",
            sql_request=SQLRequestSpec(
                task_description="total lifetime revenue per customer from net sales",
                tables=[],  # Sonnet listed nothing — retrieval must fill the slice
            ),
        ),
        call_texts=[
            ("SELECT customer_code, SUM(net_revenue) AS rev FROM fact_ai_sales_net "
            "GROUP BY customer_code"),
            "Answer.",
        ],
    )
    orch = _orchestrator(discovery, executor, llm)
    outcome = await orch.handle("s1", "exclude dead stock")

    assert outcome.sql_generated is True
    # the slice offered Haiku revenue-bearing objects found via the task text
    sqlgen_user = llm.sqlgen_prompts[0].user
    assert "fact_ai_sales_net" in sqlgen_user or "vw_q002" in sqlgen_user


async def test_result_cache_hit_skips_execution(discovery, executor):
    from app.core.cache import ResultCache

    llm = FakeLLM(RoutingDecision(
        decision="use_view",
        view_name="vw_q002_top_10_customers_by_lifetime_revenue",
    ))
    index, retriever = discovery
    from app.core.config import Settings
    from app.orchestrator.conversation import InMemoryConversationStore
    from app.prompts.loader import PromptManager

    calls = {"n": 0}
    original = executor.run_view

    def counting_run_view(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    executor.run_view = counting_run_view
    orch = Orchestrator(
        retriever=retriever, index=index, executor=executor,
        prompts=PromptManager(), conversations=InMemoryConversationStore(),
        llm=llm, settings=Settings(_env_file=None), result_cache=ResultCache(),
    )
    first = await orch.handle("s1", "top customers?")
    second = await orch.handle("s1", "top customers?")

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert calls["n"] == 1  # executed once, served from cache the second time


async def test_intent_prompt_columns_trimmed_beyond_top3(discovery, executor):
    llm = FakeLLM(RoutingDecision(decision="clarify", clarification="?"))
    orch = _orchestrator(discovery, executor, llm)
    await orch.handle("s1", "revenue by item and warehouse and customer")
    user = llm.intent_prompts[0].user
    # candidates beyond the top 3 appear with no columns line
    blocks = user.split("- name: ")[1:]
    for block in blocks[3:]:
        assert "columns:" not in block


async def test_audit_record_written_per_request(discovery, executor, tmp_path):
    import json

    from app.core.audit import AuditTrail
    from app.core.config import Settings
    from app.orchestrator.conversation import InMemoryConversationStore
    from app.prompts.loader import PromptManager

    llm = FakeLLM(RoutingDecision(
        decision="use_view",
        view_name="vw_q002_top_10_customers_by_lifetime_revenue",
    ))
    index, retriever = discovery
    orch = Orchestrator(
        retriever=retriever, index=index, executor=executor,
        prompts=PromptManager(), conversations=InMemoryConversationStore(),
        llm=llm, settings=Settings(_env_file=None),
        audit=AuditTrail(tmp_path / "audit"),
        company_id="beta",
    )
    await orch.handle("audit-session", "top customers please")

    files = list((tmp_path / "audit").rglob("audit-*.jsonl"))
    assert len(files) == 1
    assert files[0].parent.name == "beta"
    entry = json.loads(files[0].read_text().splitlines()[0])
    assert entry["session_id"] == "audit-session"
    assert entry["question"] == "top customers please"
    assert entry["company_id"] == "beta"
    assert entry["view_used"] == "vw_q002_top_10_customers_by_lifetime_revenue"
    assert entry["answer"] == "Business answer."


async def test_telemetry_trace_written_per_request(discovery, executor, tmp_path):
    import json

    from app.core.config import Settings
    from app.core.logging import bind_request_context, clear_request_context
    from app.monitoring.telemetry import TraceRecorder
    from app.orchestrator.conversation import InMemoryConversationStore
    from app.prompts.loader import PromptManager

    llm = FakeLLM(RoutingDecision(
        decision="use_view",
        view_name="vw_q002_top_10_customers_by_lifetime_revenue",
    ))
    index, retriever = discovery
    orch = Orchestrator(
        retriever=retriever, index=index, executor=executor,
        prompts=PromptManager(), conversations=InMemoryConversationStore(),
        llm=llm, settings=Settings(_env_file=None),
        telemetry=TraceRecorder(tmp_path / "monitoring"),
        company_id="beta",
    )
    bind_request_context(request_id="req-telemetry-1")
    try:
        await orch.handle("telemetry-session", "top customers please")
    finally:
        clear_request_context()

    files = list((tmp_path / "monitoring").rglob("traces-*.jsonl"))
    assert len(files) == 1
    assert files[0].parent.name == "beta"
    entry = json.loads(files[0].read_text().splitlines()[0])
    assert entry["request_id"] == "req-telemetry-1"
    assert entry["session_id"] == "telemetry-session"
    assert entry["company_id"] == "beta"
    assert entry["route"] == "use_view"
    assert entry["selected_view"] == "vw_q002_top_10_customers_by_lifetime_revenue"
    assert entry["sql_generated"] is False
    assert entry["status"] == "success"


async def test_telemetry_absence_never_breaks_a_request(discovery, executor):
    """telemetry=None (the default) must behave exactly like audit=None
    already does — a request must complete normally with no telemetry
    configured at all, not just when telemetry is configured but fails."""
    from app.core.config import Settings
    from app.orchestrator.conversation import InMemoryConversationStore
    from app.prompts.loader import PromptManager

    llm = FakeLLM(RoutingDecision(
        decision="use_view",
        view_name="vw_q002_top_10_customers_by_lifetime_revenue",
    ))
    index, retriever = discovery
    orch = Orchestrator(
        retriever=retriever, index=index, executor=executor,
        prompts=PromptManager(), conversations=InMemoryConversationStore(),
        llm=llm, settings=Settings(_env_file=None),
        company_id="beta",
    )
    outcome = await orch.handle("no-telemetry-session", "top customers please")
    assert outcome.decision == "use_view"


async def test_telemetry_captures_entity_resolution_outcome(discovery, executor, tmp_path):
    """Issue #33: entity_type/entity_status/entity_resolved in the trace
    must reflect what Resolution._entities.resolve() actually returned
    this request — real data, not fabricated. Uses a fake EntityIndex
    (same direct-attribute-assignment pattern CompanyContextManager
    itself uses in production) so the resolution status is fully
    controlled and verifiable."""
    from app.discovery.entity_resolver import Resolution
    from app.monitoring.telemetry import TraceRecorder

    class _FakeEntities:
        def resolve(self, column, value):
            return Resolution(status="ambiguous", column=column, requested=value, candidates=["A Corp", "A Co"])

    llm = FakeLLM(RoutingDecision(
        decision="use_view",
        view_name="vw_q002_top_10_customers_by_lifetime_revenue",
        endpoint_filters={"customer_name": "A"},
    ))
    orch = _orchestrator(discovery, executor, llm)
    orch._telemetry = TraceRecorder(tmp_path / "monitoring")
    orch._entities = _FakeEntities()
    await orch.handle("entity-session", "expand on A")

    import json
    files = list((tmp_path / "monitoring").rglob("traces-*.jsonl"))
    entry = json.loads(files[0].read_text().splitlines()[0])
    assert entry["entity_status"] == "ambiguous"
    assert entry["entity_type"] == "customer_name"
    assert entry["entity_resolved"] is False


async def test_telemetry_captures_sql_retry_count(discovery, executor, tmp_path):
    from app.monitoring.telemetry import TraceRecorder
    from app.orchestrator.decision_models import SQLRequestSpec

    llm = FakeLLM(
        RoutingDecision(
            decision="needs_sql",
            sql_request=SQLRequestSpec(task_description="x", tables=["fact_ai_sales_net"]),
        ),
        call_texts=[
            "SELECT * FROM forbidden_secret_table",         # rejected -> retry 1
            "SELECT customer_code FROM fact_ai_sales_net",  # accepted
            "Recovered answer.",
        ],
    )
    orch = _orchestrator(discovery, executor, llm)
    orch._telemetry = TraceRecorder(tmp_path / "monitoring")
    await orch.handle("s1", "some custom question")

    import json
    files = list((tmp_path / "monitoring").rglob("traces-*.jsonl"))
    entry = json.loads(files[0].read_text().splitlines()[0])
    assert entry["sql_retries"] == 1


async def test_greeting_replies_instantly_without_answer_call(discovery, executor):
    llm = FakeLLM(RoutingDecision(decision="greeting"))
    orch = _orchestrator(discovery, executor, llm)
    outcome = await orch.handle("s1", "hi, who are you?")

    assert outcome.decision == "greeting"
    assert "Eliara" in outcome.answer
    assert "top 10 customers" in outcome.answer
    assert llm.answer_prompts == []          # no second LLM call
    assert outcome.view_used is None and outcome.sql_generated is False
    assert len(orch._conversations.get_history("s1")) == 2


# --------------------------------------------------------- investigate route


async def test_investigate_is_refused_while_the_agent_is_disabled(discovery, executor):
    """Default configuration: the intent prompt never offers the verdict, so
    reaching it means something is misconfigured. That's still surfaced —
    routing_path_failed is logged with the full internal reason — but the
    user gets a normal answer rather than a dead request; a misconfiguration
    server-side is not something the person asking a question should have
    to experience as a broken connection."""
    llm = FakeLLM(RoutingDecision(decision="investigate"))
    orch = _orchestrator(discovery, executor, llm)

    outcome = await orch.handle("s1", "investigate Alpha Trading")

    assert "could not interpret this request" in outcome.answer


async def test_investigate_hands_off_to_the_bounded_loop(discovery, executor):
    from app.orchestrator.agent_models import AgentStep

    index, retriever = discovery
    settings = Settings(_env_file=None, agent_enabled=True, agent_max_steps=4)

    class LoopLLM(FakeLLM):
        """Routing returns investigate; the agent's own structured calls then
        pop scripted steps from the same fake."""

        def __init__(self):
            super().__init__(RoutingDecision(decision="investigate"))
            self.agent_steps = [
                AgentStep(action="run_view",
                          view_name="vw_q002_top_10_customers_by_lifetime_revenue"),
                AgentStep(action="answer", text="Alpha Trading leads at AED 7,600."),
            ]

        async def structured_call(self, prompt, output_model, *, model, max_tokens=1000):
            if output_model is RoutingDecision:
                return await super().structured_call(
                    prompt, output_model, model=model, max_tokens=max_tokens
                )
            step = self.agent_steps.pop(0)
            return step, LLMResponse(
                text="{}", model=model, input_tokens=400, output_tokens=30,
                latency_ms=4, prompt_tag=prompt.tag,
            )

    orch = Orchestrator(
        retriever=retriever, index=index, executor=executor,
        prompts=PromptManager(), conversations=InMemoryConversationStore(),
        llm=LoopLLM(), settings=settings,
    )
    outcome = await orch.handle("s1", "investigate our customer book")

    assert outcome.decision == "investigate"
    assert outcome.answer == "Alpha Trading leads at AED 7,600."
    assert outcome.trace is not None
    assert outcome.trace.views_used == ["vw_q002_top_10_customers_by_lifetime_revenue"]
    assert outcome.trace.stopped_because == "answered"
    # Tokens from every loop iteration are accounted, not just the last.
    assert outcome.input_tokens > 400


async def test_investigate_agent_is_built_once_and_reused(discovery, executor):
    from app.orchestrator.agent_models import AgentStep

    index, retriever = discovery
    settings = Settings(_env_file=None, agent_enabled=True)

    class LoopLLM(FakeLLM):
        def __init__(self):
            super().__init__(RoutingDecision(decision="investigate"))

        async def structured_call(self, prompt, output_model, *, model, max_tokens=1000):
            if output_model is RoutingDecision:
                return await super().structured_call(
                    prompt, output_model, model=model, max_tokens=max_tokens
                )
            return AgentStep(action="answer", text="done"), LLMResponse(
                text="{}", model=model, input_tokens=100, output_tokens=10,
                latency_ms=4, prompt_tag=prompt.tag,
            )

    orch = Orchestrator(
        retriever=retriever, index=index, executor=executor,
        prompts=PromptManager(), conversations=InMemoryConversationStore(),
        llm=LoopLLM(), settings=settings,
    )
    await orch.handle("s1", "investigate one")
    first = orch._agent
    await orch.handle("s1", "investigate two")

    assert orch._agent is first


# ------------------------------------------------------------------- refine


async def test_refine_sorts_the_working_set_with_no_new_query(discovery, executor):
    """The first turn populates the working set through the normal view path;
    the second turn refines it. Only ONE structured_call and ONE call happen
    per turn either way — refine costs exactly what a view answer costs, with
    no SQL generation and no second query."""
    from app.execution.refine import RefineSpec

    index, retriever = discovery
    conversations = InMemoryConversationStore()
    llm = FakeLLM(RoutingDecision(
        decision="use_view", view_name="vw_q002_top_10_customers_by_lifetime_revenue",
    ))
    orch = Orchestrator(
        retriever=retriever, index=index, executor=executor, prompts=PromptManager(),
        conversations=conversations, llm=llm, settings=Settings(_env_file=None),
    )
    await orch.handle("s1", "who are our top customers?")
    assert len(conversations.working_set("s1")) == 1

    llm.decision = RoutingDecision(
        decision="refine", refine_target=0,
        refine=RefineSpec(sort_by="lifetime_revenue", sort_desc=False, limit=2),
    )
    outcome = await orch.handle("s1", "sort those by revenue ascending, just 2")

    assert outcome.decision == "refine"
    # No SQL was generated for the refinement — sqlgen_prompts is empty for
    # this second turn (only the first-turn view prompt exists, and that
    # path never touches sqlgen either).
    assert llm.sqlgen_prompts == []
    assert outcome.answer == "Business answer."


async def test_refine_target_out_of_range_asks_for_the_full_question(discovery, executor):
    from app.execution.refine import RefineSpec

    llm = FakeLLM(RoutingDecision(
        decision="refine", refine_target=0, refine=RefineSpec(limit=1),
    ))
    orch = _orchestrator(discovery, executor, llm)
    outcome = await orch.handle("s1", "sort those")

    assert "full question" in outcome.answer
    assert llm.answer_prompts == []  # no answer call — nothing to refine yet


async def test_refine_with_no_spec_degrades_gracefully(discovery, executor):
    llm = FakeLLM(RoutingDecision(decision="refine", refine_target=0, refine=None))
    orch = _orchestrator(discovery, executor, llm)

    outcome = await orch.handle("s1", "sort those")

    assert "could not interpret this request" in outcome.answer


async def test_refine_unknown_column_is_answered_not_crashed(discovery, executor):
    from app.execution.refine import RefineFilter, RefineSpec

    index, retriever = discovery
    conversations = InMemoryConversationStore()
    llm = FakeLLM(RoutingDecision(
        decision="use_view", view_name="vw_q002_top_10_customers_by_lifetime_revenue",
    ))
    orch = Orchestrator(
        retriever=retriever, index=index, executor=executor, prompts=PromptManager(),
        conversations=conversations, llm=llm, settings=Settings(_env_file=None),
    )
    await orch.handle("s1", "top customers?")

    llm.decision = RoutingDecision(
        decision="refine", refine_target=0,
        refine=RefineSpec(filters=[RefineFilter(column="warehouse", op="eq", value="WH1")]),
    )
    outcome = await orch.handle("s1", "only warehouse WH1")

    assert "isn't in that result" in outcome.answer
    assert "warehouse" in outcome.answer


async def test_intent_prompt_is_told_what_is_in_the_working_set(discovery, executor):
    index, retriever = discovery
    conversations = InMemoryConversationStore()
    llm = FakeLLM(RoutingDecision(
        decision="use_view", view_name="vw_q002_top_10_customers_by_lifetime_revenue",
    ))
    orch = Orchestrator(
        retriever=retriever, index=index, executor=executor, prompts=PromptManager(),
        conversations=conversations, llm=llm, settings=Settings(_env_file=None),
    )
    await orch.handle("s1", "top customers?")
    await orch.handle("s1", "and again")  # second turn's intent prompt sees turn 1's result

    second_intent = llm.intent_prompts[1].user
    assert "Working set" in second_intent
    assert "vw_q002_top_10_customers_by_lifetime_revenue" in second_intent


async def test_investigate_path_does_not_pollute_the_working_set(discovery, executor):
    """A composed multi-view trace has no single coherent table shape, so it
    must not silently become the target of a later 'sort those'."""
    from app.orchestrator.agent_models import AgentStep

    index, retriever = discovery
    settings = Settings(_env_file=None, agent_enabled=True)
    conversations = InMemoryConversationStore()

    class LoopLLM(FakeLLM):
        def __init__(self):
            super().__init__(RoutingDecision(decision="investigate"))

        async def structured_call(self, prompt, output_model, *, model, max_tokens=1000):
            if output_model is RoutingDecision:
                return await super().structured_call(
                    prompt, output_model, model=model, max_tokens=max_tokens
                )
            return AgentStep(action="answer", text="done"), LLMResponse(
                text="{}", model=model, input_tokens=50, output_tokens=10,
                latency_ms=4, prompt_tag=prompt.tag,
            )

    orch = Orchestrator(
        retriever=retriever, index=index, executor=executor, prompts=PromptManager(),
        conversations=conversations, llm=LoopLLM(), settings=settings,
    )
    await orch.handle("s1", "investigate everything")

    assert conversations.working_set("s1") == []


# ------------------------------------------------------------------ streaming


class StreamingFakeLLM(FakeLLM):
    """Adds .stream() — yields the same text as call() would, in two chunks,
    so tests can assert token events arrive incrementally."""

    def __init__(self, decision, call_texts=None):
        super().__init__(decision, call_texts)
        self.stream_prompts = []

    async def stream(self, prompt, *, model, max_tokens=1500, temperature=0.2, **kwargs):
        from app.llm.anthropic_client import StreamChunk

        self.stream_prompts.append(prompt)
        text = self.call_texts.pop(0) if len(self.call_texts) > 1 else self.call_texts[0]
        mid = max(1, len(text) // 2)
        for piece in (text[:mid], text[mid:]):
            yield StreamChunk(text=piece)
        yield StreamChunk(done=True, response=LLMResponse(
            text=text, model=model, input_tokens=200, output_tokens=50,
            latency_ms=5, prompt_tag=prompt.tag,
        ))


async def test_stream_emits_stage_then_tokens_then_done(discovery, executor):
    index, retriever = discovery
    llm = StreamingFakeLLM(RoutingDecision(
        decision="use_view", view_name="vw_q002_top_10_customers_by_lifetime_revenue",
    ), call_texts=["Alpha Trading leads."])
    orch = Orchestrator(
        retriever=retriever, index=index, executor=executor, prompts=PromptManager(),
        conversations=InMemoryConversationStore(), llm=llm, settings=Settings(_env_file=None),
    )

    events = [e async for e in orch.stream("s1", "top customers?")]
    types = [e["type"] for e in events]

    assert types[0] == "stage"
    assert isinstance(events[0]["value"], str) and events[0]["value"]
    token_events = [e for e in events if e["type"] == "token"]
    assert len(token_events) == 2  # split into 2 chunks by the stub
    assert "".join(e["value"] for e in token_events) == "Alpha Trading leads."
    assert types[-1] == "done"
    assert events[-1] == {"type": "done"}  # bare, no payload
    # llm.stream was used, not llm.call, for the answer narration
    assert llm.stream_prompts and llm.stream_prompts[0].name == "orchestrator_answer"
    assert llm.call_prompts == []  # nothing went through the non-streaming path


async def test_stream_emits_a_ranking_visual_when_the_result_shape_fits(discovery, executor):
    """The visual event reuses the exact same chart-shape detection as the
    legacy /ask endpoint — same trend/ranking/table logic, one source of
    truth (app.execution.visual)."""
    index, retriever = discovery
    llm = StreamingFakeLLM(RoutingDecision(
        decision="use_view", view_name="vw_q002_top_10_customers_by_lifetime_revenue",
    ), call_texts=["Alpha Trading leads."])
    orch = Orchestrator(
        retriever=retriever, index=index, executor=executor, prompts=PromptManager(),
        conversations=InMemoryConversationStore(), llm=llm, settings=Settings(_env_file=None),
    )

    events = [e async for e in orch.stream("s1", "top customers?")]
    visual_events = [e for e in events if e["type"] == "visual"]

    assert len(visual_events) == 1
    assert visual_events[0]["value"]["type"] in {"ranking", "table", "trend"}
    # visual arrives after every token, before done
    assert events.index(visual_events[0]) > events.index(
        next(e for e in events if e["type"] == "token")
    )
    assert events[-1] == {"type": "done"}


async def test_stream_carries_the_answer_as_a_token_even_for_instant_paths(discovery, executor):
    """Greetings never call the LLM for narration, so the whole answer is
    sent as ONE token event — done never carries the text; every route
    delivers through the same channel."""
    llm = StreamingFakeLLM(RoutingDecision(decision="greeting"))
    orch = _orchestrator(discovery, executor, llm)

    events = [e async for e in orch.stream("s1", "hi")]
    token_events = [e for e in events if e["type"] == "token"]

    assert len(token_events) == 1
    assert "Eliara" in token_events[0]["value"] or len(token_events[0]["value"]) > 0
    assert events[-1] == {"type": "done"}


async def test_stream_history_and_working_set_update_the_same_as_batch(discovery, executor):
    """Streaming is a delivery mechanism, not a different pipeline — session
    state after a streamed turn must match what handle() would have left."""
    index, retriever = discovery
    conversations = InMemoryConversationStore()
    llm = StreamingFakeLLM(RoutingDecision(
        decision="use_view", view_name="vw_q002_top_10_customers_by_lifetime_revenue",
    ), call_texts=["Business answer."])
    orch = Orchestrator(
        retriever=retriever, index=index, executor=executor, prompts=PromptManager(),
        conversations=conversations, llm=llm, settings=Settings(_env_file=None),
    )

    [e async for e in orch.stream("s1", "top customers?")]

    assert len(conversations.get_history("s1")) == 2
    assert len(conversations.working_set("s1")) == 1


async def test_stream_updates_last_outcome_for_the_api_layers_metrics(discovery, executor):
    """done carries no payload by design, so the API layer reads metrics off
    the orchestrator's own record of the finished turn instead."""
    llm = StreamingFakeLLM(RoutingDecision(
        decision="use_view", view_name="vw_q002_top_10_customers_by_lifetime_revenue",
    ), call_texts=["Business answer."])
    orch = _orchestrator(discovery, executor, llm)

    [e async for e in orch.stream("s1", "top customers?")]

    assert orch._last_outcome is not None
    assert orch._last_outcome.decision == "use_view"
    assert orch._last_outcome.answer == "Business answer."


async def test_stream_delivers_a_real_message_for_a_hallucinated_view(discovery, executor):
    """This IS the exact production scenario reported: routing picked a view
    name that wasn't in this turn's candidate list. Before this fix, that
    produced a bare {"type":"error"} with no readable text on the stream —
    "No content" client-side. Now it's a normal token, same as every other
    failure mode already fixed this way."""
    llm = StreamingFakeLLM(RoutingDecision(
        decision="use_view", view_name="not_a_real_candidate",
    ))
    orch = _orchestrator(discovery, executor, llm)

    events = [e async for e in orch.stream("s1", "anything")]

    assert events[-1] == {"type": "done"}
    assert not any(e["type"] == "error" for e in events)
    token_events = [e for e in events if e["type"] == "token"]
    assert len(token_events) == 1
    assert "could not interpret this request" in token_events[0]["value"]


async def test_investigate_stream_yields_stage_events_for_each_tool_call(discovery, executor):
    """The agent's per-step decisions become friendly "stage" events, not a
    separate "step" type — the frontend only knows about stage/token/visual/
    done, so internal tool names never reach it, only human-readable text."""
    from app.orchestrator.agent_models import AgentStep

    index, retriever = discovery
    settings = Settings(_env_file=None, agent_enabled=True)

    class LoopStreamLLM(StreamingFakeLLM):
        def __init__(self):
            super().__init__(RoutingDecision(decision="investigate"))
            self.agent_steps = [
                AgentStep(action="run_view",
                          view_name="vw_q002_top_10_customers_by_lifetime_revenue"),
                AgentStep(action="answer", text="Alpha Trading leads."),
            ]

        async def structured_call(self, prompt, output_model, *, model, max_tokens=1000):
            if output_model is RoutingDecision:
                return await super().structured_call(
                    prompt, output_model, model=model, max_tokens=max_tokens
                )
            step = self.agent_steps.pop(0)
            return step, LLMResponse(
                text="{}", model=model, input_tokens=100, output_tokens=20,
                latency_ms=4, prompt_tag=prompt.tag,
            )

    orch = Orchestrator(
        retriever=retriever, index=index, executor=executor, prompts=PromptManager(),
        conversations=InMemoryConversationStore(), llm=LoopStreamLLM(), settings=settings,
    )

    events = [e async for e in orch.stream("s1", "investigate our customers")]
    stage_events = [e for e in events if e["type"] == "stage"]
    token_events = [e for e in events if e["type"] == "token"]

    # one for routing, one naming the investigate decision, one per tool call
    assert len(stage_events) == 3
    assert all(isinstance(e["value"], str) and e["value"] for e in stage_events)
    # the final answer arrives as a token (fallback path — the agent's
    # "answer" action text is delivered whole, not streamed, since it's
    # embedded in a single structured JSON call)
    assert any(e["value"] == "Alpha Trading leads." for e in token_events)
    assert events[-1] == {"type": "done"}


# ------------------------------------------------------ investigate gating


async def test_investigate_absent_from_json_schema_when_agent_disabled(discovery, executor):
    """Not just 'the model is told not to use it' — the option must not exist
    in the schema at all, so a disabled agent can never be routed to even by
    a model error."""
    llm = FakeLLM(RoutingDecision(decision="greeting"))
    orch = _orchestrator(discovery, executor, llm)  # default settings: agent_enabled=False
    await orch.handle("s1", "why did margin drop this quarter?")

    schema = llm.intent_prompts[0].system
    assert '"investigate"' not in schema


async def test_investigate_present_in_json_schema_when_agent_enabled(discovery, executor):
    index, retriever = discovery
    llm = FakeLLM(RoutingDecision(decision="greeting"))
    orch = Orchestrator(
        retriever=retriever, index=index, executor=executor, prompts=PromptManager(),
        conversations=InMemoryConversationStore(), llm=llm,
        settings=Settings(_env_file=None, agent_enabled=True),
    )
    await orch.handle("s1", "why did margin drop this quarter?")

    schema = llm.intent_prompts[0].system
    assert '"investigate"' in schema


async def test_investigate_rule_warns_against_keyword_matching_candidates(discovery, executor):
    """Regression pin for a real production failure: "why did margin drop
    this quarter?" was already the canonical trigger example in the prompt,
    yet real traffic routed it to use_view because retrieval surfaced a
    keyword-relevant-sounding view ("revenue by month") that doesn't
    actually answer a causal question. This asserts the tie-breaker text
    that addresses that specific failure mode is present — not just that
    investigate exists in the schema."""
    index, retriever = discovery
    llm = FakeLLM(RoutingDecision(decision="greeting"))
    orch = Orchestrator(
        retriever=retriever, index=index, executor=executor, prompts=PromptManager(),
        conversations=InMemoryConversationStore(), llm=llm,
        settings=Settings(_env_file=None, agent_enabled=True),
    )
    await orch.handle("s1", "why did margin drop this quarter?")

    rule_text = llm.intent_prompts[0].system
    assert "CAUSAL and DIAGNOSTIC" in rule_text
    assert "keyword overlap" in rule_text
    assert "not the same as" in rule_text  # "containing the numbers" vs "answering"


async def test_investigate_rule_disambiguates_from_the_investigate_customer_playbook(
    discovery, executor
):
    """investigate_customer is a real playbook whose own trigger phrase is
    literally "investigate customer" — the routing rule must tell the model
    to prefer the playbook whenever it applies, or every 'investigate X'
    message will misroute to the open-ended loop instead of the cheaper,
    curated playbook."""
    index, retriever = discovery
    llm = FakeLLM(RoutingDecision(decision="greeting"))
    orch = Orchestrator(
        retriever=retriever, index=index, executor=executor, prompts=PromptManager(),
        conversations=InMemoryConversationStore(), llm=llm,
        settings=Settings(_env_file=None, agent_enabled=True),
    )
    await orch.handle("s1", "investigate MERSIN TRADE")

    rule_text = llm.intent_prompts[0].system
    assert "investigate_customer" not in rule_text.split("investigate:")[0]  # sanity: rule exists
    assert "almost always means the matching PLAYBOOK" in rule_text


# --------------------------------------------------- timeout fallback (regression)


async def test_timeout_returns_a_valid_outcome_not_a_validation_error(discovery, executor):
    """Regression: the timeout branch used to construct ChatOutcome with
    tokens_in/tokens_out (fields that don't exist — the real names are
    input_tokens/output_tokens) and sql_generated=None (the field isn't
    Optional). Both raised ValidationError, so a slow request crashed instead
    of returning the graceful 'took longer than expected' message."""
    import asyncio

    index, retriever = discovery

    class HangingLLM(FakeLLM):
        async def structured_call(self, prompt, output_model, *, model, max_tokens=1000):
            await asyncio.sleep(10)
            return await super().structured_call(prompt, output_model, model=model, max_tokens=max_tokens)

    orch = Orchestrator(
        retriever=retriever, index=index, executor=executor, prompts=PromptManager(),
        conversations=InMemoryConversationStore(), llm=HangingLLM(RoutingDecision(decision="greeting")),
        settings=Settings(_env_file=None, request_deadline_s=1),
    )

    outcome = await orch.handle("s1", "anything")

    assert outcome.decision == "timeout"
    assert "took longer than expected" in outcome.answer
    assert outcome.sql_generated is False
    assert outcome.input_tokens == 0
    assert outcome.output_tokens == 0


# ------------------------------------------------- streaming: graceful SQL failure


async def test_stream_delivers_a_real_message_when_sql_is_rejected_twice(discovery, executor):
    """This is the exact bug reported in production: a needs_sql turn that
    failed validation twice used to raise, which on /chat/stream produced a
    bare {"type":"error"} event with no readable text — "No content" client
    side. Now the failure is delivered as an ordinary token, same as every
    other route."""
    from app.orchestrator.decision_models import SQLRequestSpec

    llm = StreamingFakeLLM(RoutingDecision(
        decision="needs_sql",
        sql_request=SQLRequestSpec(task_description="x", tables=["fact_ai_sales_net"]),
    ), call_texts=["DROP TABLE dim_b3_item", "SELECT nope FROM nowhere", "unused"])
    orch = _orchestrator(discovery, executor, llm)

    events = [e async for e in orch.stream("s1", "average invoice value per warehouse")]

    assert events[-1] == {"type": "done"}
    assert not any(e["type"] == "error" for e in events)
    token_events = [e for e in events if e["type"] == "token"]
    assert len(token_events) == 1
    assert "could not be translated into a safe query" in token_events[0]["value"]


# ------------------------------------------------------- external knowledge


def _external_orchestrator(discovery, executor, llm, **overrides):
    index, retriever = discovery
    settings = Settings(_env_file=None, external_enabled=True, **overrides)
    return Orchestrator(
        retriever=retriever, index=index, executor=executor, prompts=PromptManager(),
        conversations=InMemoryConversationStore(), llm=llm, settings=settings,
    )


async def test_search_prefix_never_reaches_retrieval_or_routing(discovery, executor):
    """The whole point of an explicit prefix: nothing is inferred, so nothing
    can be misinferred. No routing call at all — which also makes it the
    cheapest path in the system."""
    llm = FakeLLM(RoutingDecision(decision="use_view", view_name="unreachable"))
    orch = _external_orchestrator(discovery, executor, llm)

    outcome = await orch.handle("s1", "/search who makes headlamps in China?")

    assert outcome.decision == "external"
    assert llm.intent_prompts == []  # routing model never consulted
    assert outcome.view_used is None
    assert outcome.sql_generated is False


async def test_external_path_writes_nothing_to_the_working_set(discovery, executor):
    """There is no QueryResult to refine — a later "sort those" must not
    find an external answer sitting in the working set."""
    index, retriever = discovery
    conversations = InMemoryConversationStore()
    settings = Settings(_env_file=None, external_enabled=True)
    orch = Orchestrator(
        retriever=retriever, index=index, executor=executor, prompts=PromptManager(),
        conversations=conversations, llm=FakeLLM(RoutingDecision(decision="greeting")),
        settings=settings,
    )

    await orch.handle("s1", "/search suez canal status")

    assert conversations.working_set("s1") == []
    assert len(conversations.get_history("s1")) == 2  # but it IS in history


async def test_external_prefix_is_case_insensitive(discovery, executor):
    llm = FakeLLM(RoutingDecision(decision="greeting"))
    orch = _external_orchestrator(discovery, executor, llm)

    outcome = await orch.handle("s1", "/SEARCH what is a CV joint?")

    assert outcome.decision == "external"


async def test_prefix_must_be_a_standalone_token(discovery, executor):
    """"/searching for a customer" is a normal question, not a prefix match —
    a near-miss must never silently divert a real analytics question away
    from the governed data."""
    llm = FakeLLM(RoutingDecision(decision="greeting"))
    orch = _external_orchestrator(discovery, executor, llm)

    outcome = await orch.handle("s1", "/searching for a customer")

    assert outcome.decision != "external"


async def test_prefix_is_inert_while_the_feature_is_disabled(discovery, executor):
    """Default config: the prefix is just text, routed like any other message."""
    llm = FakeLLM(RoutingDecision(decision="greeting"))
    orch = _orchestrator(discovery, executor, llm)  # external_enabled=False

    outcome = await orch.handle("s1", "/search anything")

    assert outcome.decision == "greeting"


async def test_bare_prefix_asks_for_a_question_without_calling_the_model(discovery, executor):
    llm = FakeLLM(RoutingDecision(decision="greeting"))
    orch = _external_orchestrator(discovery, executor, llm)

    outcome = await orch.handle("s1", "/search")

    assert outcome.decision == "external"
    assert "/search" in outcome.answer
    assert llm.call_prompts == []  # no tokens spent on an empty question


async def test_external_prompt_forbids_claims_about_company_data(discovery, executor):
    """The safety property that matters: this path has NO governed data, so
    the prompt must bar it from asserting anything about the company's own
    numbers, customers or suppliers."""
    llm = FakeLLM(RoutingDecision(decision="greeting"))
    orch = _external_orchestrator(discovery, executor, llm)

    await orch.handle("s1", "/search how will a suez closure affect us?")

    # normalised: the YAML block wraps lines, so raw substrings would be
    # brittle against harmless reflowing of the prompt text
    system = " ".join(llm.call_prompts[0].system.split())
    assert "NO access to the company's database" in system
    assert "NEVER state anything as fact about THIS COMPANY" in system


async def test_prompt_demands_a_cutoff_caveat_when_web_search_is_off(discovery, executor):
    """Without live search, a question about today's conditions answered from
    training data is silently stale — the exact confident-wrongness this
    platform fights elsewhere."""
    llm = FakeLLM(RoutingDecision(decision="greeting"))
    orch = _external_orchestrator(discovery, executor, llm, external_web_search=False)

    await orch.handle("s1", "/search what is the weather in Shanghai today?")

    system = " ".join(llm.call_prompts[0].system.split())
    assert "cannot check current conditions" in system
    assert "confidently stale answer is worse" in system


async def test_web_search_is_requested_only_on_the_external_path(discovery, executor):
    """Analytics answers must never gain web access — their guarantee is that
    every figure traces to a governed view."""
    seen: list[dict] = []

    class RecordingLLM(FakeLLM):
        async def call(self, prompt, *, model, max_tokens=1500, temperature=0.2, **kwargs):
            seen.append({"name": prompt.name, "web_search": kwargs.get("web_search", False)})
            return await super().call(prompt, model=model, max_tokens=max_tokens)

    llm = RecordingLLM(RoutingDecision(
        decision="use_view", view_name="vw_q002_top_10_customers_by_lifetime_revenue",
    ))
    orch = _external_orchestrator(discovery, executor, llm)

    await orch.handle("s1", "top customers?")            # analytics
    await orch.handle("s1", "/search headlamp makers")   # external

    by_name = {entry["name"]: entry["web_search"] for entry in seen}
    assert by_name["orchestrator_answer"] is False
    assert by_name["external_answer"] is True


# --------------------------------------------------- explicit /investigate


async def test_investigate_prefix_never_reaches_retrieval_or_routing(discovery, executor):
    """Same guarantee as /search: an explicit prefix means nothing is
    inferred, so nothing can be misinferred. This directly targets the
    reliability gap — "why did margin drop" failed to reach investigate
    three separate times through router judgment alone."""
    from app.orchestrator.agent_models import AgentStep

    index, retriever = discovery
    settings = Settings(_env_file=None, agent_enabled=True)

    class LoopLLM(FakeLLM):
        def __init__(self):
            super().__init__(RoutingDecision(decision="use_view", view_name="unreachable"))

        async def structured_call(self, prompt, output_model, *, model, max_tokens=1000):
            assert output_model is AgentStep, "routing model must never be consulted"
            return AgentStep(action="answer", text="Margin fell on cost pressure."), LLMResponse(
                text="{}", model=model, input_tokens=50, output_tokens=10,
                latency_ms=1, prompt_tag=prompt.tag,
            )

    orch = Orchestrator(
        retriever=retriever, index=index, executor=executor, prompts=PromptManager(),
        conversations=InMemoryConversationStore(), llm=LoopLLM(), settings=settings,
    )

    outcome = await orch.handle("s1", "/investigate why did margin drop this quarter?")

    assert outcome.decision == "investigate"
    assert outcome.answer == "Margin fell on cost pressure."


async def test_investigate_prefix_disabled_agent_degrades_gracefully(discovery, executor):
    """Consistent with every other failure mode fixed this session: a
    misconfiguration or a disabled feature is a normal answer, not a dead
    request."""
    llm = FakeLLM(RoutingDecision(decision="greeting"))
    orch = _orchestrator(discovery, executor, llm)  # default settings: agent_enabled=False

    outcome = await orch.handle("s1", "/investigate why did margin drop?")

    assert outcome.decision == "investigate"
    assert "isn't enabled" in outcome.answer
    assert llm.intent_prompts == []  # still bypassed routing even though it then declines


async def test_bare_investigate_prefix_defaults_to_a_general_assessment(discovery, executor):
    """Unlike /search, a bare /investigate is a reasonable request on its
    own — "check things out" — so it gets a sensible default question
    instead of being rejected as empty."""
    from app.orchestrator.agent_models import AgentStep

    index, retriever = discovery
    settings = Settings(_env_file=None, agent_enabled=True)

    class LoopLLM(FakeLLM):
        def __init__(self):
            super().__init__(RoutingDecision(decision="greeting"))
            self.seen_questions = []

        async def structured_call(self, prompt, output_model, *, model, max_tokens=1000):
            self.seen_questions.append(prompt.user)
            return AgentStep(action="answer", text="Nothing urgent."), LLMResponse(
                text="{}", model=model, input_tokens=50, output_tokens=10,
                latency_ms=1, prompt_tag=prompt.tag,
            )

    llm = LoopLLM()
    orch = Orchestrator(
        retriever=retriever, index=index, executor=executor, prompts=PromptManager(),
        conversations=InMemoryConversationStore(), llm=llm, settings=settings,
    )
    outcome = await orch.handle("s1", "/investigate")

    assert outcome.decision == "investigate"
    assert "general assessment" in llm.seen_questions[0].lower()


async def test_investigate_prefix_must_be_a_standalone_token(discovery, executor):
    llm = FakeLLM(RoutingDecision(decision="greeting"))
    orch = _external_orchestrator(discovery, executor, llm, agent_enabled=True)

    outcome = await orch.handle("s1", "/investigator general title")

    assert outcome.decision != "investigate"


async def test_investigate_prefix_writes_history_but_not_working_set(discovery, executor):
    """Investigate's own architectural rule (mixed-shape trace, no single
    coherent result) applies the same way whether reached via router or
    prefix."""
    from app.orchestrator.agent_models import AgentStep

    index, retriever = discovery
    settings = Settings(_env_file=None, agent_enabled=True)
    conversations = InMemoryConversationStore()

    class LoopLLM(FakeLLM):
        def __init__(self):
            super().__init__(RoutingDecision(decision="greeting"))

        async def structured_call(self, prompt, output_model, *, model, max_tokens=1000):
            return AgentStep(action="answer", text="done"), LLMResponse(
                text="{}", model=model, input_tokens=50, output_tokens=10,
                latency_ms=1, prompt_tag=prompt.tag,
            )

    orch = Orchestrator(
        retriever=retriever, index=index, executor=executor, prompts=PromptManager(),
        conversations=conversations, llm=LoopLLM(), settings=settings,
    )
    await orch.handle("s1", "/investigate anything worth flagging?")

    assert len(conversations.get_history("s1")) == 2
    assert conversations.working_set("s1") == []


async def test_investigate_prefix_streams_stage_events_and_a_token(discovery, executor):
    from app.orchestrator.agent_models import AgentStep

    index, retriever = discovery
    settings = Settings(_env_file=None, agent_enabled=True)

    class LoopStreamLLM(StreamingFakeLLM):
        def __init__(self):
            super().__init__(RoutingDecision(decision="greeting"))
            self.steps = [
                AgentStep(action="run_view",
                          view_name="vw_q002_top_10_customers_by_lifetime_revenue"),
                AgentStep(action="answer", text="Concentrated on one account."),
            ]

        async def structured_call(self, prompt, output_model, *, model, max_tokens=1000):
            step = self.steps.pop(0)
            return step, LLMResponse(
                text="{}", model=model, input_tokens=50, output_tokens=10,
                latency_ms=1, prompt_tag=prompt.tag,
            )

    orch = Orchestrator(
        retriever=retriever, index=index, executor=executor, prompts=PromptManager(),
        conversations=InMemoryConversationStore(), llm=LoopStreamLLM(), settings=settings,
    )
    events = [e async for e in orch.stream("s1", "/investigate why did margin drop?")]

    assert any(e["type"] == "stage" for e in events)
    token_events = [e for e in events if e["type"] == "token"]
    assert any(e["value"] == "Concentrated on one account." for e in token_events)
    assert events[-1] == {"type": "done"}


# ----------------------------------------------------------------- /scan


async def test_scan_never_reaches_retrieval_or_routing_or_the_llm_at_all(discovery, executor):
    """The whole point: every number in a scan answer already comes from a
    governed view, so there is nothing for a model to add except risk.
    Confirmed by the fact this test's FakeLLM never gets called at all —
    not for routing, not for narration."""
    class NeverCalledLLM(FakeLLM):
        async def structured_call(self, *a, **kw):
            raise AssertionError("structured_call must never be reached by /scan")
        async def call(self, *a, **kw):
            raise AssertionError("call must never be reached by /scan")
        async def stream(self, *a, **kw):
            raise AssertionError("stream must never be reached by /scan")
            yield  # pragma: no cover - unreachable, satisfies async generator syntax

    index, retriever = discovery
    settings = Settings(_env_file=None, scan_enabled=True,
                        scan_views=["vw_q002_top_10_customers_by_lifetime_revenue"])
    orch = Orchestrator(
        retriever=retriever, index=index, executor=executor, prompts=PromptManager(),
        conversations=InMemoryConversationStore(), llm=NeverCalledLLM(RoutingDecision(decision="greeting")),
        settings=settings,
    )
    outcome = await orch.handle("s1", "/scan")

    assert outcome.decision == "scan"
    assert outcome.input_tokens == 0
    assert outcome.output_tokens == 0
    assert "Scanned 1 view" in outcome.answer


async def test_scan_reports_high_tier_items_with_source_view_for_traceability(discovery, executor):
    llm = FakeLLM(RoutingDecision(decision="greeting"))
    orch = _external_orchestrator(
        discovery, executor, llm,
        scan_enabled=True, scan_views=["vw_q002_top_10_customers_by_lifetime_revenue"],
    )
    outcome = await orch.handle("s1", "/scan")

    assert "Needs attention" in outcome.answer
    assert "vw_q002_top_10_customers_by_lifetime_revenue" in outcome.answer


async def test_scan_skips_a_missing_view_without_failing_the_whole_scan(discovery, executor):
    llm = FakeLLM(RoutingDecision(decision="greeting"))
    orch = _external_orchestrator(
        discovery, executor, llm,
        scan_enabled=True,
        scan_views=["not_a_real_view", "vw_q002_top_10_customers_by_lifetime_revenue"],
    )
    outcome = await orch.handle("s1", "/scan")

    assert outcome.decision == "scan"
    assert "unavailable on this deployment" in outcome.answer
    assert "Needs attention" in outcome.answer  # the one good view still reported


async def test_scan_with_zero_usable_views_reports_plainly_not_a_crash(discovery, executor):
    llm = FakeLLM(RoutingDecision(decision="greeting"))
    orch = _external_orchestrator(
        discovery, executor, llm, scan_enabled=True, scan_views=["not_a_real_view"],
    )
    outcome = await orch.handle("s1", "/scan")

    assert outcome.decision == "scan"
    assert "nothing to report" in outcome.answer


async def test_scan_is_inert_while_disabled(discovery, executor):
    llm = FakeLLM(RoutingDecision(decision="greeting"))
    orch = _orchestrator(discovery, executor, llm)  # scan_enabled=False by default

    outcome = await orch.handle("s1", "/scan")

    assert outcome.decision == "greeting"  # routed normally, prefix is just text


async def test_scan_prefix_requires_a_standalone_token(discovery, executor):
    llm = FakeLLM(RoutingDecision(decision="greeting"))
    orch = _external_orchestrator(discovery, executor, llm, scan_enabled=True)

    outcome = await orch.handle("s1", "/scanning the horizon")

    assert outcome.decision != "scan"


async def test_scan_streams_the_answer_as_a_single_token(discovery, executor):
    llm = StreamingFakeLLM(RoutingDecision(decision="greeting"))
    orch = _external_orchestrator(
        discovery, executor, llm,
        scan_enabled=True, scan_views=["vw_q002_top_10_customers_by_lifetime_revenue"],
    )
    events = [e async for e in orch.stream("s1", "/scan")]

    assert events[-1] == {"type": "done"}
    token_events = [e for e in events if e["type"] == "token"]
    assert len(token_events) == 1
    assert "Scanned" in token_events[0]["value"]
