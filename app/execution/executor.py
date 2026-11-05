"""Read-only SQLite executor — the ONLY component allowed to touch the database.

Defense in depth (each layer independently sufficient):
  1. Connections opened with URI ``mode=ro`` — writes are physically impossible.
  2. An SQLite authorizer that permits only SELECT/READ/FUNCTION/RECURSIVE
     actions; PRAGMA, ATTACH, and every write action are denied at prepare time.
  3. Runtime limits: a progress-handler query timeout and a hard row cap.

``run_view`` builds SQL itself from a validated identifier + bound parameters;
user-supplied values are never interpolated into SQL text.
"""

import math
import queue
import re
import sqlite3
import threading
import time
from pathlib import Path

from app.core.errors import SQLExecutionError, SQLValidationError
from app.core.logging import get_logger
from app.core.models import DateRange, QueryResult

log = get_logger("executor")

# SQLite authorizer action codes (not exported by the sqlite3 module).
_SQLITE_PRAGMA = 19
_SQLITE_READ = 20
_SQLITE_SELECT = 21
_SQLITE_FUNCTION = 31
_SQLITE_RECURSIVE = 33
_ALLOWED_ACTIONS = {_SQLITE_READ, _SQLITE_SELECT, _SQLITE_FUNCTION, _SQLITE_RECURSIVE}

# The ONLY pragmas the authorizer lets through: read-only schema introspection
# needed by the metadata loader. Everything else (journal_mode,
# writable_schema, ...) stays denied.
_READONLY_PRAGMAS = {
    "table_info",
    "table_xinfo",
    "table_list",
    "index_list",
    "index_info",
    "foreign_key_list",
    "database_list",
}

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_PROGRESS_OPCODE_INTERVAL = 5_000


def _sqrt(x: float | None) -> float | None:
    """SQLite has no built-in SQRT() — several curated views (e.g.
    vw_q025_items_highest_demand_volatility, vw_q028_recommended_safety_
    stock_fast_movers, vw_q064_recommended_eoq_for_specific_item,
    vw_q087_supplier_lead_time_predictability) use it, and fail with
    "no such function: SQRT" without this registered. Registered as a
    Python function on every connection instead of rewriting the views —
    no DB changes needed, and it's a read-only, side-effect-free math
    function, well within the executor's read-only guarantees.

    Mirrors SQL NULL-propagation semantics (NULL in, NULL out) rather than
    raising, and returns None for a negative input rather than raising
    ValueError — one row with an unexpected negative value in a demand-
    volatility calculation shouldn't take down the whole query.
    """
    if x is None:
        return None
    try:
        value = float(x)
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None
    return math.sqrt(value)


def _authorizer(action: int, arg1, *_args) -> int:
    if action in _ALLOWED_ACTIONS:
        return sqlite3.SQLITE_OK
    if action == _SQLITE_PRAGMA and (arg1 or "").lower() in _READONLY_PRAGMAS:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


class ReadOnlyExecutor:
    def __init__(
        self,
        db_path: Path | str,
        *,
        query_timeout_s: float = 30.0,
        max_rows: int = 500,
        pool_size: int = 4,
    ) -> None:
        self._db_path = Path(db_path)
        self._timeout_s = query_timeout_s
        self._max_rows = max_rows
        self._pool_size = pool_size
        self._lock = threading.Lock()
        self._pool: queue.Queue[sqlite3.Connection] = queue.Queue(maxsize=pool_size)
        self._open_pool()

    # ------------------------------------------------------------------ pool

    def _make_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            f"file:{self._db_path}?mode=ro",
            uri=True,
            check_same_thread=False,
            timeout=5.0,
        )
        conn.set_authorizer(_authorizer)
        conn.create_function("SQRT", 1, _sqrt, deterministic=True)
        return conn

    def _open_pool(self) -> None:
        if not self._db_path.exists():
            raise SQLExecutionError(internal_detail=f"database file not found: {self._db_path}")
        for _ in range(self._pool_size):
            self._pool.put(self._make_conn())
        log.info("executor_pool_open", db=str(self._db_path), size=self._pool_size)

    def close(self) -> None:
        with self._lock:
            while True:
                try:
                    self._pool.get_nowait().close()
                except queue.Empty:
                    break

    def reopen(self) -> None:
        """Drain and rebuild the pool — called by the DB watcher after a refresh."""
        with self._lock:
            while True:
                try:
                    self._pool.get_nowait().close()
                except queue.Empty:
                    break
            self._open_pool()
        log.info("executor_pool_reopened")

    # ------------------------------------------------------------- execution

    def _execute(
        self,
        sql: str,
        params: tuple,
        source: str,
        object_name: str,
        row_cap: int | None = None,
    ) -> QueryResult:
        row_cap = row_cap or self._max_rows
        conn = self._pool.get()
        deadline = time.monotonic() + self._timeout_s

        def _abort_if_late() -> int:
            return 1 if time.monotonic() > deadline else 0

        start = time.perf_counter()
        try:
            conn.set_progress_handler(_abort_if_late, _PROGRESS_OPCODE_INTERVAL)
            cursor = conn.execute(sql, params)
            rows = cursor.fetchmany(row_cap + 1)
            truncated = len(rows) > row_cap
            if truncated:
                rows = rows[:row_cap]
            columns = [d[0] for d in cursor.description] if cursor.description else []
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            return QueryResult(
                columns=columns,
                rows=[tuple(r) for r in rows],
                row_count=len(rows),
                truncated=truncated,
                source=source,
                object_name=object_name,
                elapsed_ms=elapsed_ms,
            )
        except sqlite3.OperationalError as exc:
            msg = str(exc)
            if "interrupt" in msg.lower():
                raise SQLExecutionError(
                    internal_detail=f"query timeout after {self._timeout_s}s on {object_name}",
                    public_message="The query took too long. Try narrowing the question.",
                ) from exc
            raise SQLExecutionError(internal_detail=f"{object_name}: {msg}") from exc
        except sqlite3.Error as exc:
            raise SQLExecutionError(internal_detail=f"{object_name}: {exc}") from exc
        finally:
            conn.set_progress_handler(None, 0)
            self._pool.put(conn)

    # -------------------------------------------------------------- public API

    def run_view(
        self,
        view_name: str,
        filters: dict[str, str] | None = None,
        limit: int | None = None,
    ) -> QueryResult:
        self._require_identifier(view_name)
        if not self._object_exists(view_name):
            raise SQLValidationError(internal_detail=f"unknown object: {view_name}")

        clauses: list[str] = []
        params: list[str] = []
        for col, value in (filters or {}).items():
            self._require_identifier(col)
            clauses.append(f'"{col}" = ?')
            params.append(value)

        effective_cap = min(limit or self._max_rows, self._max_rows)
        sql = f'SELECT * FROM "{view_name}"'
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        # Fetch one row beyond the cap so truncation is detectable downstream.
        sql += " LIMIT ?"
        params.append(effective_cap + 1)

        return self._execute(
            sql, tuple(params), source="view", object_name=view_name, row_cap=effective_cap
        )

    def run_sql(self, sql: str, params: tuple = ()) -> QueryResult:
        """Execute already-validated SQL (M5 will require a ValidatedSQL wrapper)."""
        return self._execute(
            sql, params, source="generated_sql", object_name="generated_sql",
            row_cap=self._max_rows,
        )

    def run_metadata_sql(self, sql: str, params: tuple = (), row_cap: int = 20_000) -> QueryResult:
        """Internal metadata/introspection queries — not exposed to any LLM path.

        Uses a much higher row cap than user-facing queries because the real
        database has ~600 eligible objects and thousands of columns.
        """
        return self._execute(sql, params, source="metadata", object_name="metadata", row_cap=row_cap)

    def data_boundaries(
        self, table: str | None = None, date_column: str | None = None
    ) -> DateRange | None:
        """Min/max of a date column, used for the "data through <date>"
        deep-health line. table/date_column are supplied by the caller's
        CompanyConfig — this executor makes no assumption about which
        table or column any given company's schema uses. Returns None
        (rather than raising) when not configured or when the table/column
        doesn't resolve, so deep health degrades gracefully instead of
        erroring for a company that hasn't confirmed this yet."""
        if not table or not date_column:
            return None
        self._require_identifier(table)
        self._require_identifier(date_column)
        try:
            result = self._execute(
                f'SELECT MIN("{date_column}"), MAX("{date_column}") FROM "{table}"',
                (),
                source="metadata",
                object_name="data_boundaries",
            )
        except SQLExecutionError:
            return None
        if not result.rows or result.rows[0][0] is None:
            return None
        first, last = result.rows[0]
        return DateRange(first_date=str(first), last_date=str(last))

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _require_identifier(name: str) -> None:
        if not _IDENTIFIER_RE.match(name or ""):
            raise SQLValidationError(internal_detail=f"invalid identifier: {name!r}")

    def _object_exists(self, name: str) -> bool:
        result = self._execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ? LIMIT 1",
            (name,),
            source="metadata",
            object_name="sqlite_master",
        )
        return result.row_count > 0
