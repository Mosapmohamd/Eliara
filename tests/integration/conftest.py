import httpx
import pytest

from app.llm.anthropic_client import AnthropicClient, LLMResponse
from app.main import create_app
from app.orchestrator.decision_models import RoutingDecision
from tests.fixtures.company_config import write_companies_yaml
from tests.fixtures.fixture_db import build_fixture_db

COMPANY_ID = "beta"


@pytest.fixture()
async def app_client(tmp_path, monkeypatch):
    db = build_fixture_db(tmp_path / "chat.db")
    companies_yaml = write_companies_yaml(tmp_path, db, company_id=COMPANY_ID)
    monkeypatch.setenv("ELIARA_COMPANIES_CONFIG", str(companies_yaml))
    monkeypatch.setenv("ELIARA_DEFAULT_COMPANY_ID", COMPANY_ID)
    monkeypatch.setenv("ELIARA_EMBEDDING_BACKEND", "hashing")
    monkeypatch.setenv("ELIARA_EMBEDDING_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("ELIARA_EXTERNAL_ENABLED", "true")
    # Isolate monitoring storage per test too — without this, tests that
    # actually check telemetry/incident data (tests/integration/
    # test_monitor_endpoints.py) could read stale state from whatever
    # get_monitoring_config() cached from an earlier test in the same
    # run. Existing tests that don't touch monitoring data are unaffected
    # either way.
    monkeypatch.setenv("ELIARA_MONITOR_STORAGE_DIR", str(tmp_path / "monitoring_data"))
    from app.core.config import get_settings
    from app.monitoring.config import get_monitoring_config

    get_settings.cache_clear()
    get_monitoring_config.cache_clear()

    async def fake_structured(self, prompt, output_model, *, model, max_tokens=1000):
        return (
            RoutingDecision(
                decision="use_view",
                view_name="vw_q002_top_10_customers_by_lifetime_revenue",
            ),
            LLMResponse(text="{}", model=model, input_tokens=1, output_tokens=1,
                        latency_ms=1, prompt_tag=prompt.tag),
        )

    async def fake_call(self, prompt, *, model, max_tokens=1500, temperature=0.2, **kwargs):
        if prompt.name == "external_answer":
            return LLMResponse(text="Indicative external answer.",
                               model=model, input_tokens=1, output_tokens=1,
                               latency_ms=1, prompt_tag=prompt.tag)
        return LLMResponse(text="Top customer is Beta Motors with 8,000.",
                           model=model, input_tokens=1, output_tokens=1,
                           latency_ms=1, prompt_tag=prompt.tag)

    monkeypatch.setattr(AnthropicClient, "structured_call", fake_structured)
    monkeypatch.setattr(AnthropicClient, "call", fake_call)

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
        app.router.lifespan_context(app),
    ):
        yield client
    get_settings.cache_clear()
    get_monitoring_config.cache_clear()


@pytest.fixture()
async def two_company_client(tmp_path, monkeypatch):
    """Two real, independent fixture databases registered as 'beta' and
    'tire_guru' — the fixture used by the cross-company isolation tests.
    Each database is tagged with a company-specific view/answer so a test
    can detect if one company's request ever reaches the other's data."""
    import sqlite3

    beta_db = build_fixture_db(tmp_path / "beta.db")
    tire_db = build_fixture_db(tmp_path / "tire_guru.db")

    # Give Tire Guru a view that does NOT exist in Beta's database, so a
    # test can prove Tire Guru's discovery/metadata never leaks into a
    # Beta-scoped request (and vice versa for the reverse case).
    conn = sqlite3.connect(tire_db)
    conn.execute(
        "CREATE VIEW vw_tire_guru_only_marker AS "
        "SELECT 'tire_guru_marker_row' AS marker"
    )
    conn.commit()
    conn.close()

    companies_yaml = tmp_path / "companies.yaml"
    import yaml

    companies_yaml.write_text(
        yaml.safe_dump(
            {
                "companies": {
                    "beta": {
                        "display_name": "Beta",
                        "db_path": str(beta_db),
                        "scan_views": ["vw_q002_top_10_customers_by_lifetime_revenue"],
                    },
                    "tire_guru": {
                        "display_name": "Tire Guru",
                        "db_path": str(tire_db),
                        "scan_views": ["vw_tire_guru_only_marker"],
                        "status": "partial",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ELIARA_COMPANIES_CONFIG", str(companies_yaml))
    monkeypatch.setenv("ELIARA_EMBEDDING_BACKEND", "hashing")
    monkeypatch.setenv("ELIARA_EMBEDDING_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("ELIARA_AUDIT_ENABLED", "true")
    monkeypatch.setenv("ELIARA_AUDIT_DIR", str(tmp_path / "audit"))
    monkeypatch.setenv("ELIARA_SCAN_ENABLED", "true")
    from app.core.config import get_settings

    get_settings.cache_clear()

    calls: list[str] = []

    async def fake_structured(self, prompt, output_model, *, model, max_tokens=1000):
        return (
            RoutingDecision(
                decision="use_view",
                view_name="vw_q002_top_10_customers_by_lifetime_revenue",
            ),
            LLMResponse(text="{}", model=model, input_tokens=1, output_tokens=1,
                        latency_ms=1, prompt_tag=prompt.tag),
        )

    async def fake_call(self, prompt, *, model, max_tokens=1500, temperature=0.2, **kwargs):
        calls.append(prompt.user)
        return LLMResponse(text="Top customer is Beta Motors with 8,000.",
                           model=model, input_tokens=1, output_tokens=1,
                           latency_ms=1, prompt_tag=prompt.tag)

    monkeypatch.setattr(AnthropicClient, "structured_call", fake_structured)
    monkeypatch.setattr(AnthropicClient, "call", fake_call)

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
        app.router.lifespan_context(app),
    ):
        client.app_state = app.state  # convenience handle for the tests
        client.llm_calls = calls
        yield client
    get_settings.cache_clear()


