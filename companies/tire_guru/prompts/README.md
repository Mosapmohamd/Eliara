# Tire Guru prompt overrides

- `orchestrator/greeting_v1.yaml` — Tire Guru-specific greeting message
  (mentions "Tire Guru" and tire/inventory-domain sample questions instead
  of the shared generic default). Added after the shared greeting was
  found to have "Beta" hardcoded as plain text — every company now gets
  its own greeting, none hardcoded to another company's name.

Everything else still falls through to the shared prompt library
(`app/prompts/shared/templates/`). If another prompt ever needs Tire
Guru-specific business terminology or wording, add a same-named,
same-versioned YAML file here (see `PromptManager.for_company()` in
`app/prompts/loader.py`) — it takes precedence over the shared version
for this company only.
