"""Request trace recording.

Mirrors app.core.audit.AuditTrail's pattern deliberately: one JSONL file
per day, per company, under `storage_dir/<company_id>/`; write failures
are caught and logged as a warning, never raised — a monitoring gap must
never break the request it was trying to observe (spec requirement:
"The health-check system must never crash the Eliara request path" — the
same principle applies to telemetry recording itself).

Concurrency: TraceRecorder holds no per-request state — every call to
record() takes a complete RequestTrace and writes it immediately. Nothing
here is mutated between two concurrent requests; the only shared state is
the write lock and the set of already-created company directories, both
of which are safe under concurrent access by construction (see tests in
tests/unit/test_telemetry.py, in particular
test_concurrent_traces_never_interleave_or_drop_fields).
"""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.core.logging import get_logger
from app.monitoring.models import RequestTrace

log = get_logger("monitoring.telemetry")


class TraceRecorder:
    def __init__(self, storage_dir: Path | str, enabled: bool = True) -> None:
        self._dir = Path(storage_dir)
        self._enabled = enabled
        self._lock = threading.Lock()
        self._known_dirs: set[str] = set()
        if enabled:
            self._dir.mkdir(parents=True, exist_ok=True)

    def _company_dir(self, company_id: str) -> Path:
        path = self._dir / company_id
        if company_id not in self._known_dirs:
            path.mkdir(parents=True, exist_ok=True)
            self._known_dirs.add(company_id)
        return path

    def _current_file(self, company_id: str) -> Path:
        return self._company_dir(company_id) / f"traces-{datetime.now(UTC):%Y-%m-%d}.jsonl"

    def record(self, trace: RequestTrace) -> None:
        if not self._enabled:
            return
        company_id = trace.company_id or "unknown"
        entry = trace.to_dict()
        entry["unix_ts"] = time.time()
        try:
            line = json.dumps(entry, ensure_ascii=False, default=str)
            with self._lock, self._current_file(company_id).open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:
            log.warning("trace_write_failed", company_id=company_id, request_id=trace.request_id, exc_info=True)

    def read_recent(self, company_id: str, since_unix_ts: float) -> list[dict]:
        """All traces for one company with unix_ts >= since_unix_ts —
        the read path health.py (and later phases) use to compute
        rolling metrics over MonitoringConfig.health_window_minutes.
        Checks today's and yesterday's files (a window can span
        midnight); each phase using this stays a read-only consumer of
        the same JSONL store, no separate index or database needed for
        the MVP volume this is sized for.
        """
        company_dir = self._dir / company_id
        if not company_dir.exists():
            return []
        results: list[dict] = []
        today = datetime.now(UTC)
        yesterday = today - timedelta(days=1)
        candidates = [
            company_dir / f"traces-{today:%Y-%m-%d}.jsonl",
            company_dir / f"traces-{yesterday:%Y-%m-%d}.jsonl",
        ]
        seen_paths: set[Path] = set()
        for path in candidates:
            if path in seen_paths or not path.exists():
                continue
            seen_paths.add(path)
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    entry = json.loads(line)
                    if entry.get("unix_ts", 0) >= since_unix_ts:
                        results.append(entry)
            except Exception:
                log.warning("trace_read_failed", path=str(path), exc_info=True)
        return results

    def read_trace(self, company_id: str, request_id: str) -> dict | None:
        """Linear scan of today's and yesterday's files for one company —
        adequate for the MVP (Phase 1) volume and the
        GET /monitor/traces/{request_id} use case (an operator looking up
        one specific recent request, not a bulk query). Revisit if/when
        this needs to search further back or scale beyond a
        single-machine JSONL store — see monitoring/README.md's storage
        note.
        """
        company_dir = self._dir / company_id
        if not company_dir.exists():
            return None
        for path in sorted(company_dir.glob("traces-*.jsonl"), reverse=True):
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    entry = json.loads(line)
                    if entry.get("request_id") == request_id:
                        return entry
            except Exception:
                log.warning("trace_read_failed", path=str(path), exc_info=True)
        return None
