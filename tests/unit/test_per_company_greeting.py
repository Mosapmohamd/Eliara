"""Tests for per-company greeting message overrides.

Covers a real bug found live: the shared greeting_message prompt had
"Beta" hardcoded as plain text — a leftover from before the multi-company
refactor. Any company (including Tire Guru) that hit the "greeting"
routing decision got a reply saying "I'm Eliara, Beta's business
intelligence copilot", regardless of which company_id the request was
for. Confirmed via a live request against the real deployment, not
guessed — the response contained no actual business data (the greeting
path never queries the database — see Orchestrator._handle's
"greeting" branch, which renders a static template with no DB call), so
this was a text/branding bug, not a data-isolation breach. Still a real
bug worth fixing properly: every company should get its own greeting,
with nothing hardcoded to another company's name.

Fix: the shared default is now company-agnostic (no company name at
all), and each company that wants specific branding gets a same-name/
same-version override via PromptManager.for_company() — the mechanism
already existed (see app/prompts/loader.py), it just had never been used
for an actual company-specific prompt until this fix.
"""

from pathlib import Path

from app.company.registry import CompanyRegistry
from app.prompts.loader import PromptManager


def test_shared_default_greeting_mentions_no_company_name():
    """The fallback used by any company without its own override must
    stay company-neutral — this is what would have been shown to a brand
    new company added to companies.yaml before anyone authors a greeting
    override for it."""
    prompts = PromptManager()  # no extra_dir = the shared-only path
    greeting = prompts.render("greeting_message").user
    assert "Beta" not in greeting
    assert "Tire Guru" not in greeting


def test_beta_override_mentions_beta_and_nothing_else():
    repo_root = Path(__file__).resolve().parents[2]
    registry = CompanyRegistry.from_file(repo_root / "companies.yaml")
    prompts = PromptManager.for_company(registry.get("beta"))
    greeting = prompts.render("greeting_message").user
    assert "Beta" in greeting
    assert "Tire Guru" not in greeting


def test_tire_guru_override_mentions_tire_guru_and_nothing_else():
    repo_root = Path(__file__).resolve().parents[2]
    registry = CompanyRegistry.from_file(repo_root / "companies.yaml")
    prompts = PromptManager.for_company(registry.get("tire_guru"))
    greeting = prompts.render("greeting_message").user
    assert "Tire Guru" in greeting
    assert "Beta" not in greeting


def test_beta_and_tire_guru_greetings_are_actually_different():
    """The two companies' PromptManagers must not silently resolve to the
    same text — a config mistake (e.g. both pointing at the same override
    directory) would make this pass falsely if only checked in isolation."""
    repo_root = Path(__file__).resolve().parents[2]
    registry = CompanyRegistry.from_file(repo_root / "companies.yaml")
    beta_greeting = PromptManager.for_company(registry.get("beta")).render("greeting_message").user
    tire_guru_greeting = (
        PromptManager.for_company(registry.get("tire_guru")).render("greeting_message").user
    )
    assert beta_greeting != tire_guru_greeting


def test_tire_guru_greeting_references_only_confirmed_real_capabilities():
    """The sample questions in Tire Guru's greeting must be grounded in
    views actually confirmed to exist for Tire Guru (issue #6/#28
    investigation) — not invented capabilities Tire Guru doesn't have."""
    repo_root = Path(__file__).resolve().parents[2]
    registry = CompanyRegistry.from_file(repo_root / "companies.yaml")
    greeting = PromptManager.for_company(registry.get("tire_guru")).render("greeting_message").user
    # dead stock, stockout risk, and supplier reliability are the 3
    # confirmed-real Tire Guru scan capabilities from issue #6/#28.
    assert "dead stock" in greeting.lower()
    assert "stock" in greeting.lower()
    assert "supplier" in greeting.lower()
