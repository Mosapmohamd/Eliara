"""Tests for app.company.registry — CompanyConfig and CompanyRegistry.

No test file existed for this module before. Covers the basics plus the
new `status` field added for issue #22 (Tire Guru looking feature-complete
in companies.yaml when it isn't).
"""

import pytest
from pydantic import ValidationError

from app.company.registry import CompanyConfig, CompanyRegistry, UnknownCompany


def _config(**overrides):
    defaults = {"company_id": "beta", "display_name": "Beta", "db_path": "data/beta.db"}
    defaults.update(overrides)
    return CompanyConfig(**defaults)


def test_status_defaults_to_full():
    cfg = _config()
    assert cfg.status == "full"


def test_status_can_be_set_to_partial():
    cfg = _config(status="partial")
    assert cfg.status == "partial"


def test_invalid_status_value_rejected():
    with pytest.raises(ValidationError):
        _config(status="mostly_done")


def test_invalid_company_id_rejected():
    with pytest.raises(ValidationError):
        _config(company_id="Has Spaces")


def test_scan_views_default_empty():
    cfg = _config()
    assert cfg.scan_views == []


def test_registry_from_file(tmp_path):
    path = tmp_path / "companies.yaml"
    path.write_text(
        "companies:\n"
        "  beta:\n"
        "    display_name: Beta\n"
        "    db_path: data/beta.db\n"
        "  tire_guru:\n"
        "    display_name: Tire Guru\n"
        "    db_path: data/tire_guru.db\n"
        "    status: partial\n",
        encoding="utf-8",
    )
    registry = CompanyRegistry.from_file(path)
    assert registry.all_ids() == ["beta", "tire_guru"]
    assert registry.get("beta").status == "full"
    assert registry.get("tire_guru").status == "partial"


def test_registry_get_unknown_company_raises(tmp_path):
    path = tmp_path / "companies.yaml"
    path.write_text(
        "companies:\n  beta:\n    display_name: Beta\n    db_path: data/beta.db\n",
        encoding="utf-8",
    )
    registry = CompanyRegistry.from_file(path)
    with pytest.raises(UnknownCompany):
        registry.get("not_registered")


def test_registry_from_missing_file_raises(tmp_path):
    from app.company.registry import CompanyRegistryError

    with pytest.raises(CompanyRegistryError):
        CompanyRegistry.from_file(tmp_path / "does_not_exist.yaml")


def test_critical_views_defaults_empty():
    cfg = _config()
    assert cfg.critical_views == []


def test_data_health_checks_defaults_empty():
    cfg = _config()
    assert cfg.data_health_checks == []


def test_critical_views_and_data_health_checks_configurable():
    cfg = _config(
        critical_views=["vw_important"],
        data_health_checks=[{"view": "vw_important", "check": "nonzero", "column": "revenue"}],
    )
    assert cfg.critical_views == ["vw_important"]
    assert cfg.data_health_checks[0]["check"] == "nonzero"
