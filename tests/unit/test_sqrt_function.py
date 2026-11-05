"""Regression tests for the SQRT() SQL function registered on every
executor connection.

SQLite has no built-in SQRT(). Several curated views
(vw_q025_items_highest_demand_volatility,
vw_q028_recommended_safety_stock_fast_movers,
vw_q064_recommended_eoq_for_specific_item,
vw_q087_supplier_lead_time_predictability) use it in their definitions and
failed with "no such function: SQRT" at both metadata-sampling time and
query-execution time before this fix — confirmed live against the real
Beta database startup log, not guessed.

Fixed by registering a Python SQRT function on every connection
(ReadOnlyExecutor._make_conn) rather than rewriting the view SQL, so no
database change was needed.
"""

import sqlite3

import pytest

from app.execution.executor import ReadOnlyExecutor


@pytest.fixture()
def sqrt_view_executor(tmp_path):
    db = tmp_path / "sqrt.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE demand(item_code TEXT, variance REAL)")
    conn.executemany(
        "INSERT INTO demand VALUES (?,?)",
        [("A100", 16.0), ("A200", 25.0), ("A300", None), ("A400", 2.0)],
    )
    # Shaped like the real broken views: SQRT() applied directly in a
    # curated view definition, exactly the case that used to fail.
    conn.execute(
        "CREATE VIEW vw_demand_volatility AS "
        "SELECT item_code, variance, SQRT(variance) AS std_dev FROM demand"
    )
    conn.commit()
    conn.close()
    ex = ReadOnlyExecutor(db)
    yield ex
    ex.close()


def test_sqrt_view_no_longer_raises_no_such_function(sqrt_view_executor):
    """The exact failure mode this fix addresses: querying a view that
    calls SQRT() used to raise sqlite3.OperationalError('no such
    function: SQRT'). Must now succeed."""
    result = sqrt_view_executor.run_view("vw_demand_volatility")
    assert result.row_count == 4


def test_sqrt_computes_correct_values(sqrt_view_executor):
    result = sqrt_view_executor.run_view("vw_demand_volatility")
    by_item = {row[0]: row[2] for row in result.rows}
    assert by_item["A100"] == 4.0
    assert by_item["A200"] == 5.0
    assert by_item["A400"] == pytest.approx(1.4142135623730951)


def test_sqrt_of_null_is_null_not_an_error(sqrt_view_executor):
    result = sqrt_view_executor.run_view("vw_demand_volatility")
    by_item = {row[0]: row[2] for row in result.rows}
    assert by_item["A300"] is None


def test_sqrt_of_negative_returns_none_not_raise(tmp_path):
    """A negative value reaching SQRT() (bad data, not the normal case)
    must not crash the whole query — math.sqrt(-1) raises ValueError in
    plain Python; the registered function must catch that."""
    db = tmp_path / "neg.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE t(x REAL)")
    conn.executemany("INSERT INTO t VALUES (?)", [(-4.0,), (9.0,)])
    conn.execute("CREATE VIEW vw_neg_sqrt AS SELECT x, SQRT(x) AS root FROM t")
    conn.commit()
    conn.close()

    ex = ReadOnlyExecutor(db)
    try:
        result = ex.run_view("vw_neg_sqrt")
        by_x = {row[0]: row[1] for row in result.rows}
        assert by_x[-4.0] is None
        assert by_x[9.0] == 3.0
    finally:
        ex.close()


def test_sqrt_survives_pool_reopen(sqrt_view_executor):
    """The fix lives in _make_conn(), which is also called by reopen()
    (used by the DB watcher after every refresh) — confirm the function
    is still registered on the connections created by a reopen, not just
    the ones created at __init__."""
    sqrt_view_executor.reopen()
    result = sqrt_view_executor.run_view("vw_demand_volatility")
    assert result.row_count == 4
