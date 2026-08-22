"""Unified incident engine and lifecycle — Phase 5 of the monitoring
system (issue #34).

Converts anomalies from health.py (#31), data_health.py (#32), and
ai_health.py (#33) into deduplicated, auditable incidents with a
controlled lifecycle.

Lifecycle states are deliberately OPEN -> DIAGNOSED -> ACKNOWLEDGED ->
CLOSED (plus ESCALATED), NOT the issue's literally-worded
OPEN->DIAGNOSED->HEALING->VERIFIED->CLOSED — confirmed explicitly with
the project owner. HEALING/VERIFIED implies an automated remediation
step, which directly contradicts both the master monitoring spec's
non-goals ("Do NOT add HEALING or AUTO-RECOVERY states yet") and the
explicit decision to exclude issues #36/#37 (self-healing) from this
work entirely. This engine creates, deduplicates, transitions, and
stores incidents — it never executes a corrective action, and no state
here implies one happened.
"""

from __future__ import annotations

import dataclasses
import json
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

from app.core.logging import get_logger

log = get_logger("monitoring.incidents")

# Lifecycle states (see module docstring for why HEALING/VERIFIED are
# deliberately absent).
OPEN = "OPEN"
DIAGNOSED = "DIAGNOSED"
ACKNOWLEDGED = "ACKNOWLEDGED"
ESCALATED = "ESCALATED"
CLOSED = "CLOSED"

_VALID_TRANSITIONS = {
    OPEN: {DIAGNOSED, ACKNOWLEDGED, ESCALATED, CLOSED},
    DIAGNOSED: {ACKNOWLEDGED, ESCALATED, CLOSED},
    ACKNOWLEDGED: {ESCALATED, CLOSED, DIAGNOSED},
    ESCALATED: {ACKNOWLEDGED, CLOSED, DIAGNOSED},
    CLOSED: {OPEN},  # a CLOSED incident can only reopen (see record_anomaly)
}

# Severity — issue #34 explicitly asks for this 4-level scale, distinct
# from health.py/data_health.py's 3-level HEALTHY/WARNING/CRITICAL.
# Anomalies only ever carry WARNING/CRITICAL (their source scale), so the
# mapping used by record_anomaly() is WARNING->WARNING, CRITICAL->CRITICAL.
# INFO and ERROR exist in the schema for manual/future use (e.g. a
# human downgrading/upgrading an incident) but nothing in this phase
# auto-assigns them.
INFO = "INFO"
WARNING = "WARNING"
ERROR = "ERROR"
CRITICAL = "CRITICAL"
_VALID_SEVERITIES = {INFO, WARNING, ERROR, CRITICAL}


class InvalidTransition(Exception):
    pass


@dataclasses.dataclass
class Incident:
    """Mutable by design (unlike RequestTrace/DataAnomaly) — an incident
    genuinely changes over its lifetime (status, occurrence_count,
    last_seen, diagnosis), and every mutation goes through
    IncidentEngine's methods, which validate transitions and persist the
    result — never edited directly by a caller.
    """

    incident_id: str
    category: str
    severity: str
    company_id: str
    title: str
    first_seen: str
    last_seen: str
    occurrence_count: int = 1
    request_ids: list[str] = dataclasses.field(default_factory=list)
    evidence: list[dict] = dataclasses.field(default_factory=list)
    status: str = OPEN
    diagnosis: dict | None = None  # populated by Phase 6 (#35) root-cause diagnosis; always None here
    notes: str | None = None  # free-text, human-entered only — never auto-populated
    history: list[dict] = dataclasses.field(default_factory=list)  # (timestamp, from_status, to_status)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _dedup_key(company_id: str, category: str, metric: str) -> str:
    """Same underlying failure = same company + category + metric. Two
    anomalies with the same key are the same incident, not two —
    issue #34's explicit "repeated identical failure does not create
    alert spam" requirement."""
    return f"{company_id}:{category}:{metric}"


_EVIDENCE_CAP_PER_INCIDENT = 20  # bounded so a hot-looping failure can't grow one incident's evidence unboundedly


class IncidentEngine:
    def __init__(self, storage_dir: Path | str | None = None, enabled: bool = True) -> None:
        self._incidents: dict[str, Incident] = {}  # incident_id -> Incident (all, including closed)
        self._dedup_index: dict[str, str] = {}  # dedup_key -> incident_id (most recent incident of that kind)
        self._lock = threading.Lock()
        self._dir = Path(storage_dir) if storage_dir else None
        self._enabled = enabled
        if self._enabled and self._dir:
            self._dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------- creation

    def record_anomaly(self, anomaly) -> Incident:
        """Converts one anomaly (DataAnomaly, or anything with the same
        company_id/category/severity/metric/evidence shape) into an
        incident — creating a new one, or updating the existing one for
        the same underlying failure (dedup). Thread-safe: the whole
        read-modify-write is under one lock, so two concurrent anomalies
        for the same dedup key can't race into two separate incidents.
        """
        with self._lock:
            key = _dedup_key(anomaly.company_id, anomaly.category, anomaly.metric)
            now = datetime.now(UTC).isoformat()
            existing_id = self._dedup_index.get(key)
            severity = anomaly.severity if anomaly.severity in _VALID_SEVERITIES else WARNING

            if existing_id and existing_id in self._incidents:
                incident = self._incidents[existing_id]
                incident.occurrence_count += 1
                incident.last_seen = now
                if getattr(anomaly, "evidence", None):
                    incident.evidence.append(anomaly.evidence)
                    incident.evidence = incident.evidence[-_EVIDENCE_CAP_PER_INCIDENT:]
                request_id = (anomaly.evidence or {}).get("request_id") if getattr(anomaly, "evidence", None) else None
                if request_id and request_id not in incident.request_ids:
                    incident.request_ids.append(request_id)
                if incident.status == CLOSED:
                    self._transition(incident, OPEN, now, reason="recurrence")
                self._persist(incident)
                return incident

            incident = Incident(
                incident_id=uuid.uuid4().hex[:12],
                category=anomaly.category,
                severity=severity,
                company_id=anomaly.company_id,
                title=f"{anomaly.category}: {anomaly.metric}",
                first_seen=now,
                last_seen=now,
                request_ids=[(anomaly.evidence or {}).get("request_id")] if getattr(anomaly, "evidence", None) and (anomaly.evidence or {}).get("request_id") else [],
                evidence=[anomaly.evidence] if getattr(anomaly, "evidence", None) else [],
            )
            self._incidents[incident.incident_id] = incident
            self._dedup_index[key] = incident.incident_id
            self._persist(incident)
            return incident

    def record_anomalies(self, anomalies: list) -> list[Incident]:
        """Only anomalies that actually represent a problem create an
        incident — a HEALTHY-severity informational anomaly (e.g.
        ai_health.py's clarification-rate metric report) is explicitly
        excluded, matching "normal empty results do not automatically
        become incidents" applied one layer up: an anomaly object
        existing doesn't automatically mean an incident should."""
        return [
            self.record_anomaly(a)
            for a in anomalies
            if getattr(a, "severity", None) in (WARNING, ERROR, CRITICAL)
        ]

    # ------------------------------------------------------------- transitions

    def transition(self, incident_id: str, to_status: str, *, reason: str | None = None) -> Incident:
        with self._lock:
            incident = self._incidents.get(incident_id)
            if incident is None:
                raise KeyError(f"unknown incident_id: {incident_id}")
            self._transition(incident, to_status, datetime.now(UTC).isoformat(), reason=reason)
            self._persist(incident)
            return incident

    def _transition(self, incident: Incident, to_status: str, timestamp: str, *, reason: str | None = None) -> None:
        if to_status not in _VALID_TRANSITIONS.get(incident.status, set()):
            raise InvalidTransition(
                f"cannot transition incident {incident.incident_id} from {incident.status} to {to_status}"
            )
        incident.history.append({
            "timestamp": timestamp, "from_status": incident.status, "to_status": to_status, "reason": reason,
        })
        incident.status = to_status

    def set_diagnosis(self, incident_id: str, diagnosis: dict) -> Incident:
        """Phase 6 (#35)'s write path — kept here as a thin, validated
        setter rather than letting #35 mutate Incident objects directly,
        so every incident mutation still goes through one place."""
        with self._lock:
            incident = self._incidents.get(incident_id)
            if incident is None:
                raise KeyError(f"unknown incident_id: {incident_id}")
            incident.diagnosis = diagnosis
            if incident.status == OPEN:
                self._transition(incident, DIAGNOSED, datetime.now(UTC).isoformat(), reason="diagnosis_set")
            self._persist(incident)
            return incident

    # ------------------------------------------------------------------ query

    def get(self, incident_id: str) -> Incident | None:
        return self._incidents.get(incident_id)

    def query(
        self, *, company_id: str | None = None, category: str | None = None,
        status: str | None = None, since: str | None = None,
    ) -> list[Incident]:
        results = list(self._incidents.values())
        if company_id is not None:
            results = [i for i in results if i.company_id == company_id]
        if category is not None:
            results = [i for i in results if i.category == category]
        if status is not None:
            results = [i for i in results if i.status == status]
        if since is not None:
            results = [i for i in results if i.last_seen >= since]
        return sorted(results, key=lambda i: i.last_seen, reverse=True)

    # -------------------------------------------------------------- persistence

    def _persist(self, incident: Incident) -> None:
        """Best-effort — an incident-store write failure must never
        raise out of the monitoring path, matching every other
        monitoring module's principle. One JSON file per incident
        (not one big file rewritten each time) so a write to incident A
        can't corrupt or block a concurrent write to incident B."""
        if not self._enabled or not self._dir:
            return
        try:
            company_dir = self._dir / incident.company_id
            company_dir.mkdir(parents=True, exist_ok=True)
            path = company_dir / f"{incident.incident_id}.json"
            path.write_text(json.dumps(incident.to_dict(), ensure_ascii=False, default=str), encoding="utf-8")
        except Exception:
            log.warning("incident_persist_failed", incident_id=incident.incident_id, exc_info=True)
