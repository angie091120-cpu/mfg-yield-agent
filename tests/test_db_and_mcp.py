from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
from pathlib import Path

import pytest

from pipeline.load_db import connect_ro

ROOT = Path(__file__).resolve().parent.parent
EXPECTED_TOOLS = {
    "yield_by_week", "top_suspect_sensors", "compare_sensor", "explain_run", "draft_investigation_note",
}


# ---------------------------------------------------------------- database
def test_features_long_table_matches_clean(db, cleaned):
    conn = sqlite3.connect(db)
    n = conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    imputed = conn.execute("SELECT SUM(imputed) FROM features").fetchone()[0]
    steps = conn.execute("SELECT step FROM cleaning_log ORDER BY step_no").fetchall()
    conn.close()
    assert n == len(cleaned.clean) * cleaned.meta["cols_kept"]
    assert imputed == cleaned.meta["cells_imputed"]
    assert [s[0] for s in steps][-1] == "impute_median"


@pytest.mark.parametrize("sql", [
    "INSERT INTO notes (sensor, text, status, created_at, created_by) VALUES ('s010','x','confirmed','t','a')",
    "UPDATE runs SET result = 'pass'",
    "DELETE FROM features",
    "CREATE TABLE evil (x)",
])
def test_readonly_connection_cannot_write(db, sql):
    conn = connect_ro(db)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        conn.execute(sql)
    conn.close()


def test_server_query_connection_is_readonly(db, server_module):
    conn = server_module.connect_ro()
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        conn.execute("UPDATE notes SET status = 'confirmed'")
    conn.close()


def test_notes_table_rejects_unknown_status(db):
    conn = sqlite3.connect(db)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO notes (sensor, text, status, created_at, created_by)"
                     " VALUES ('s010', 'x', 'closed', 't', 'a')")
    conn.close()


# ---------------------------------------------------------------- MCP tools
def test_exactly_five_tools(server_module):
    names = {t.name for t in asyncio.run(server_module.server.list_tools())}
    assert names == EXPECTED_TOOLS


def test_no_tool_takes_a_status_or_sql_argument(server_module):
    for tool in asyncio.run(server_module.server.list_tools()):
        props = set(tool.input_schema.get("properties", {}))
        assert not props & {"status", "sql", "query"}, tool.name


def test_draft_note_is_stored_as_draft(db, server_module, cleaned):
    sensor = cleaned.meta["kept"][0]
    out = server_module.draft_investigation_note(sensor, "fail 組平均偏高，請產線確認")
    assert out["status"] == "draft"
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT sensor, text, status, created_by FROM notes WHERE id = ?", (out["id"],)).fetchone()
    conn.close()
    assert row == (sensor, "fail 組平均偏高，請產線確認", "draft", "agent")


def test_draft_note_cannot_be_asked_for_confirmed(db, server_module, cleaned):
    sensor = cleaned.meta["kept"][0]
    with pytest.raises(TypeError):
        server_module.draft_investigation_note(sensor, "x", status="confirmed")  # type: ignore[call-arg]
    # Through the MCP layer the extra argument is dropped: the call succeeds and stores a draft.
    asyncio.run(server_module.server.call_tool(
        "draft_investigation_note", {"sensor": sensor, "text": "via mcp", "status": "confirmed"}))
    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT text, status, created_by FROM notes").fetchall()
    conn.close()
    assert rows == [("via mcp", "draft", "agent")]


def test_db_trigger_blocks_update_to_confirmed(db, server_module, cleaned):
    """Second line: even a plain read-write connection cannot move a note off 'draft'."""
    out = server_module.draft_investigation_note(cleaned.meta["kept"][0], "x")
    conn = sqlite3.connect(db)
    with pytest.raises(sqlite3.IntegrityError, match="can only be draft"):
        conn.execute("UPDATE notes SET status = 'confirmed' WHERE id = ?", (out["id"],))
    with pytest.raises(sqlite3.IntegrityError, match="can only be draft"):
        conn.execute("INSERT INTO notes (sensor, text, status, created_at, created_by)"
                     " VALUES ('s010', 'x', 'confirmed', 't', 'a')")
    conn.execute("UPDATE notes SET text = 'edited' WHERE id = ?", (out["id"],))  # other columns still editable
    rows = conn.execute("SELECT text, status FROM notes").fetchall()
    conn.close()
    assert rows == [("edited", "draft")]


def test_note_text_with_sql_is_stored_verbatim(db, server_module, cleaned):
    sensor = cleaned.meta["kept"][0]
    text = "x'); UPDATE notes SET status='confirmed'; --"
    out = server_module.draft_investigation_note(sensor, text)
    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT text, status FROM notes").fetchall()
    conn.close()
    assert rows == [(text, "draft")]
    assert out["status"] == "draft"


@pytest.mark.parametrize("bad", ["s001' OR '1'='1", "s0010", "runs", "", "S9999"])
def test_compare_sensor_rejects_bad_names(db, server_module, bad):
    with pytest.raises(ValueError):
        server_module.compare_sensor(bad)


def test_compare_sensor_rejects_dropped_column(db, server_module, cleaned):
    with pytest.raises(ValueError, match="dropped"):
        server_module.compare_sensor(cleaned.meta["dropped_constant"][0])


def test_compare_sensor_excludes_imputed_cells(db, server_module):
    out = server_module.compare_sensor("s003")
    assert out["pass"]["imputed_excluded"] + out["fail"]["imputed_excluded"] == 6
    assert out["pass"]["n"] + out["fail"]["n"] == 60 - 6


def test_yield_by_week_totals_match_runs(db, server_module):
    out = server_module.yield_by_week("2008-01-01", "2008-12-31")
    conn = sqlite3.connect(db)
    runs, fails = conn.execute("SELECT COUNT(*), SUM(result = 'fail') FROM runs").fetchone()
    conn.close()
    assert out["total"]["runs"] == runs and out["total"]["fails"] == fails
    assert sum(w["runs"] for w in out["weeks"]) == runs
    with pytest.raises(ValueError):
        server_module.yield_by_week("2008-12-31", "2008-01-01")


def test_explain_run_and_suspects(db, server_module):
    out = server_module.explain_run(1)
    assert out["selected_model"] == "logreg" and out["flagged"] is True
    assert out["top_features"][0]["rank"] == 1
    with pytest.raises(ValueError):
        server_module.explain_run(999999)
    for bad in (0, -1, 2**70):
        with pytest.raises(ValueError, match="between"):
            server_module.explain_run(bad)
    sus = server_module.top_suspect_sensors(3)
    assert [s["rank"] for s in sus["sensors"]] == [1, 2, 3]
    with pytest.raises(ValueError):
        server_module.top_suspect_sensors(50)


def test_missing_db_error_has_no_absolute_path(tmp_path, server_module, monkeypatch):
    missing = tmp_path / "nowhere" / "secom.db"
    monkeypatch.setenv("MFG_DB", str(missing))
    with pytest.raises(FileNotFoundError) as exc:
        server_module.connect_ro()
    assert str(tmp_path) not in str(exc.value) and "secom.db" in str(exc.value)


def test_stdio_server_lists_five_tools(db):
    """A4 over the real transport: spawn the server and list tools as a client would."""
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    async def run() -> set[str]:
        params = StdioServerParameters(
            command=sys.executable, args=[str(ROOT / "mcp" / "server.py")],
            env={**os.environ, "MFG_DB": str(db)},
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return {t.name for t in (await session.list_tools()).tools}

    assert asyncio.run(run()) == EXPECTED_TOOLS
