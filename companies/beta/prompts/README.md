# Beta prompt overrides

- `orchestrator/greeting_v1.yaml` — Beta-specific greeting message. Added
  after discovering the shared greeting had "Beta" hardcoded as plain
  text (meaning every company, including Tire Guru, saw Beta's branding).
  This override preserves Beta's original wording explicitly, as a real
  per-company override rather than an accidental shared default.

Everything else still falls through to the shared prompt library
(`app/prompts/shared/templates/`). If another prompt ever needs
Beta-specific business terminology or wording, add a same-named,
same-versioned YAML file here (see `PromptManager.for_company()` in
`app/prompts/loader.py`) — it takes precedence over the shared version
for this company only.
