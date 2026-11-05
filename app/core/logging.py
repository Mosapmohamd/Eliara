"""structlog configuration: JSON logs, request/session correlation, secret redaction."""

import logging

import structlog

# Exact sensitive names, or sensitive suffixes. Deliberately does NOT match
# substrings: input_tokens / output_tokens are REQUIRED observability fields
# and must never be redacted (field failure: they were).
_SENSITIVE_KEYS = {
    "key", "token", "secret", "password", "credential", "authorization",
    "api_key", "apikey", "access_token", "refresh_token",
}
_SENSITIVE_SUFFIXES = ("_key", "_secret", "_password", "_credential", "_authorization")


def _redact_secrets(_, __, event_dict: dict) -> dict:
    for k in list(event_dict):
        lowered = k.lower()
        if lowered in _SENSITIVE_KEYS or lowered.endswith(_SENSITIVE_SUFFIXES):
            event_dict[k] = "[REDACTED]"
    return event_dict


def setup_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(level=log_level, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            _redact_secrets,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str):
    return structlog.get_logger(name)


def bind_request_context(request_id: str, session_id: str | None = None) -> None:
    structlog.contextvars.bind_contextvars(request_id=request_id)
    if session_id:
        structlog.contextvars.bind_contextvars(session_id=session_id)


def clear_request_context() -> None:
    structlog.contextvars.clear_contextvars()


def get_request_id() -> str | None:
    """Reads back the request_id bound by bind_request_context() for the
    current request — used by the monitoring layer to tag a RequestTrace
    with the same id already in every log line for that request, without
    threading an extra parameter through every call in between. Safe
    under concurrency: structlog.contextvars is real Python contextvars,
    task-local by construction, not a shared/global dict."""
    return structlog.contextvars.get_contextvars().get("request_id")
