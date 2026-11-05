"""Cross-company isolation tests — Milestone 7.

Covers the 16 isolation cases enumerated in the multi-company refactor
brief, using two real (fixture) SQLite databases registered as 'beta' and
'tire_guru' via the ``two_company_client`` fixture in conftest.py.
"""

import asyncio

import pytest

from app.core.errors import SQLExecutionError

# 1 & 2. Each company's request uses its own database ---------------------


async def test_beta_request_uses_beta_database(two_company_client):
    manager = two_company_client.app_state.company_manager
    beta_ctx = manager.get("beta")
    tire_ctx = manager.get("tire_guru")
    assert beta_ctx.executor is not tire_ctx.executor
    assert beta_ctx.config.db_path != tire_ctx.config.db_path

    response = await two_company_client.post(
        "/api/v1/chat",
        json={"company_id": "beta", "session_id": "s1", "message": "top customers?"},
    )
    assert response.status_code == 200


async def test_tire_guru_request_uses_tire_guru_database(two_company_client):
    response = await two_company_client.post(
        "/api/v1/chat",
        json={"company_id": "tire_guru", "session_id": "s1", "message": "top customers?"},
    )
    assert response.status_code == 200


# 3 & 4. One company cannot reach the other's objects ----------------------


async def test_beta_cannot_access_tire_guru_only_view(two_company_client):
    """Tire Guru's marker view does not exist in Beta's database — a
    request scoped to Beta must not be able to run it."""
    response = await two_company_client.post(
        "/api/v1/detect",
        json={"company_id": "beta", "view_name": "vw_tire_guru_only_marker"},
    )
    assert response.status_code >= 400


async def test_tire_guru_can_access_its_own_marker_view(two_company_client):
    response = await two_company_client.post(
        "/api/v1/detect",
        json={"company_id": "tire_guru", "view_name": "vw_tire_guru_only_marker"},
    )
    # Not necessarily 200 (the marker view has no numeric column to rank
    # by), but it must not be the "unknown object" failure Beta gets.
    assert response.status_code != 404


# 5. Independent pools -------------------------------------------------


async def test_independent_connection_pools(two_company_client):
    manager = two_company_client.app_state.company_manager
    beta_ctx = manager.get("beta")
    tire_ctx = manager.get("tire_guru")
    assert beta_ctx.executor._pool is not tire_ctx.executor._pool


# 6. Refresh scoping -----------------------------------------------------


async def test_refresh_for_beta_does_not_rebuild_tire_guru(two_company_client):
    manager = two_company_client.app_state.company_manager
    tire_ctx_before = manager.get("tire_guru")
    before_index = tire_ctx_before.metadata_index
    before_retriever = tire_ctx_before.retriever
    before_entities = tire_ctx_before.entities

    manager._refresh("beta")

    tire_ctx_after = manager.get("tire_guru")
    assert tire_ctx_after.metadata_index is before_index
    assert tire_ctx_after.retriever is before_retriever
    assert tire_ctx_after.entities is before_entities


# 7. Cache isolation -------------------------------------------------------


async def test_beta_cache_cannot_satisfy_tire_guru_request(two_company_client):
    manager = two_company_client.app_state.company_manager
    beta_ctx = manager.get("beta")
    tire_ctx = manager.get("tire_guru")
    assert beta_ctx.result_cache is not tire_ctx.result_cache

    await two_company_client.post(
        "/api/v1/chat",
        json={"company_id": "beta", "session_id": "cache1", "message": "top customers?"},
    )
    # Tire Guru's cache must still be empty — nothing Beta cached is visible.
    assert len(tire_ctx.result_cache) == 0


# 8. Session isolation ------------------------------------------------------


async def test_sessions_are_isolated_by_company(two_company_client):
    """The SAME session_id under two different companies must not share
    conversation history — company_id + session_id is the real identity."""
    await two_company_client.post(
        "/api/v1/chat",
        json={"company_id": "beta", "session_id": "shared-id", "message": "top customers?"},
    )
    await two_company_client.post(
        "/api/v1/chat",
        json={"company_id": "tire_guru", "session_id": "shared-id", "message": "top customers?"},
    )

    beta_history = (
        await two_company_client.get("/api/v1/sessions/shared-id?company_id=beta")
    ).json()["messages"]
    tire_history = (
        await two_company_client.get("/api/v1/sessions/shared-id?company_id=tire_guru")
    ).json()["messages"]

    # Each company saw exactly its own one turn (2 messages: user +
    # assistant) — if sessions were shared, one of these would have 4.
    assert len(beta_history) == 2
    assert len(tire_history) == 2


# 9. Prompts resolve according to company -----------------------------------


async def test_prompts_resolve_per_company(two_company_client):
    manager = two_company_client.app_state.company_manager
    beta_ctx = manager.get("beta")
    tire_ctx = manager.get("tire_guru")
    # Distinct PromptManager instances (each built via PromptManager.for_company)
    assert beta_ctx.prompts is not tire_ctx.prompts
    # Both still resolve the same shared prompt set (no content forked yet)
    assert beta_ctx.prompts.active_version("orchestrator_intent") == tire_ctx.prompts.active_version(
        "orchestrator_intent"
    )


# 10 & 11. Metadata / entity resolution use the correct company index -------


async def test_metadata_and_entities_use_the_correct_company_index(two_company_client):
    manager = two_company_client.app_state.company_manager
    beta_ctx = manager.get("beta")
    tire_ctx = manager.get("tire_guru")
    assert beta_ctx.metadata_index is not tire_ctx.metadata_index
    assert beta_ctx.entities is not tire_ctx.entities
    # Tire Guru's marker view is indexed there and NOT in Beta's index.
    assert "vw_tire_guru_only_marker" in tire_ctx.metadata_index.objects
    assert "vw_tire_guru_only_marker" not in beta_ctx.metadata_index.objects


# 12 & 13. Playbooks / scan configuration are company-specific --------------


async def test_scan_configuration_is_company_specific(two_company_client):
    manager = two_company_client.app_state.company_manager
    beta_ctx = manager.get("beta")
    tire_ctx = manager.get("tire_guru")
    assert beta_ctx.orchestrator._scan_views == [
        "vw_q002_top_10_customers_by_lifetime_revenue"
    ]
    assert tire_ctx.orchestrator._scan_views == ["vw_tire_guru_only_marker"]


# 14. Audit records contain company_id and are isolated on disk ------------


async def test_audit_records_are_company_scoped(two_company_client, tmp_path):
    await two_company_client.post(
        "/api/v1/chat",
        json={"company_id": "beta", "session_id": "a1", "message": "top customers?"},
    )
    await two_company_client.post(
        "/api/v1/chat",
        json={"company_id": "tire_guru", "session_id": "a1", "message": "top customers?"},
    )

    audit_dir = tmp_path / "audit"
    beta_files = list((audit_dir / "beta").glob("audit-*.jsonl"))
    tire_files = list((audit_dir / "tire_guru").glob("audit-*.jsonl"))
    assert len(beta_files) == 1
    assert len(tire_files) == 1

    import json

    beta_entry = json.loads(beta_files[0].read_text().splitlines()[0])
    tire_entry = json.loads(tire_files[0].read_text().splitlines()[0])
    assert beta_entry["company_id"] == "beta"
    assert tire_entry["company_id"] == "tire_guru"


# 15 & 16. Existing security / API behavior remain intact -------------------


async def test_unknown_company_id_is_a_clean_404_not_a_500(two_company_client):
    response = await two_company_client.post(
        "/api/v1/chat",
        json={"company_id": "not_a_real_company", "session_id": "s1", "message": "hi"},
    )
    assert response.status_code == 404
    assert "sqlite" not in response.json()["error"].lower()


async def test_read_only_protections_still_apply_per_company(two_company_client):
    manager = two_company_client.app_state.company_manager
    for company_id in ("beta", "tire_guru"):
        ctx = manager.get(company_id)
        with pytest.raises(SQLExecutionError):
            ctx.executor.run_sql("DELETE FROM fact_ai_sales_net")


# Concurrency: Beta and Tire Guru requests interleaved -----------------------


async def test_concurrent_beta_and_tire_guru_requests_do_not_interfere(two_company_client):
    async def ask(company_id: str, i: int):
        return await two_company_client.post(
            "/api/v1/chat",
            json={"company_id": company_id, "session_id": f"c{i}", "message": "top customers?"},
        )

    tasks = []
    for i in range(10):
        tasks.append(ask("beta", i))
        tasks.append(ask("tire_guru", i))
    responses = await asyncio.gather(*tasks)
    assert all(r.status_code == 200 for r in responses)


async def test_health_deep_reports_both_companies_independently(two_company_client):
    response = await two_company_client.get("/api/v1/health/deep")
    body = response.json()
    assert set(body["companies"]) == {"beta", "tire_guru"}
    assert body["companies"]["beta"]["status"] == "ok"
    assert body["companies"]["tire_guru"]["status"] == "ok"


def test_real_companies_yaml_marks_tire_guru_as_partial():
    """Config-accuracy check (issue #22): the actual, committed
    companies.yaml — not a test fixture — must mark Tire Guru as partial
    so this is visible to anyone reading /health/deep, not just someone
    reading YAML comments."""
    from pathlib import Path

    from app.company.registry import CompanyRegistry

    repo_root = Path(__file__).resolve().parents[2]
    registry = CompanyRegistry.from_file(repo_root / "companies.yaml")
    assert registry.get("e01").status == "full"
    assert registry.get("e02").status == "partial"


def test_real_companies_yaml_tire_guru_scan_views_configured():
    """Config-accuracy check (issue #6): Tire Guru's scan_views must be
    populated with the 4 confirmed views (dead stock, slow-moving,
    liquidation, critical stockout) — not left empty. Overstock/excess is
    deliberately absent: confirmed via manual DB inspection that no
    equivalent concept exists in Tire Guru's schema at all, unlike Beta."""
    from pathlib import Path

    from app.company.registry import CompanyRegistry

    repo_root = Path(__file__).resolve().parents[2]
    registry = CompanyRegistry.from_file(repo_root / "companies.yaml")
    scan_views = registry.get("e02").scan_views
    assert scan_views == [
        "vw_tire_guru_dead_stock_ranked",
        "vw_tire_guru_slow_moving_items",
        "vw_tire_guru_liquidation_candidates",
        "vw_tire_guru_critical_stockout",
    ]