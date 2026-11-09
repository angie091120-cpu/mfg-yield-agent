"""MCP server (stdio) over the SECOM yield database.

Run:         python mcp/server.py
List tools:  python mcp/server.py --list-tools
DB path:     env MFG_DB, default data/secom.db

Five tools. Four read through a read-only SQLite connection (mode=ro) with
fixed, parameterized queries; there is no free-form SQL tool. The fifth,
draft_investigation_note, inserts into `notes` only, and always with
status 'draft'. No tool updates or deletes a note, so the agent cannot
mark a finding as confirmed; that is a person's job.
"""
from __future__ import annotations

import asyncio
from contextlib import closing
import os
import re
import sqlite3
import statistics
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

ROOT = Path(__file__).resolve().parent.parent
SENSOR_RE = re.compile(r"^s\d{3}$")
MAX_NOTE_CHARS = 2000
NOTE_AUTHOR = "agent"


def db_path() -> Path:
    return Path(os.environ.get("MFG_DB", ROOT / "data" / "secom.db"))


def connect_ro() -> sqlite3.Connection:
    path = db_path()
    if not path.exists():
        # No absolute path in the message: it goes back to the model and may end up in a transcript.
        raise FileNotFoundError(f"database not found: {path.name}; run the pipeline first")
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _check_sensor(conn: sqlite3.Connection, sensor: str) -> str:
    sensor = (sensor or "").strip().lower()
    if not SENSOR_RE.match(sensor):
        raise ValueError(f"sensor must look like s000..s589, got {sensor!r}")
    if conn.execute("SELECT 1 FROM features WHERE sensor = ? LIMIT 1", (sensor,)).fetchone() is None:
        raise ValueError(f"{sensor} was dropped during cleaning or does not exist")
    return sensor


def _parse_day(text: str, name: str) -> date:
    try:
        return date.fromisoformat(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be YYYY-MM-DD, got {text!r}") from exc


def _describe(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    q = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else [values[0]] * 3
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 6),
        "std": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
        "min": round(min(values), 6),
        "q1": round(q[0], 6),
        "median": round(q[1], 6),
        "q3": round(q[2], 6),
        "max": round(max(values), 6),
    }


# --------------------------------------------------------------------------
# Tool implementations (plain functions, so tests can call them directly)
# --------------------------------------------------------------------------
def yield_by_week(start: str, end: str) -> dict[str, Any]:
    d0, d1 = _parse_day(start, "start"), _parse_day(end, "end")
    if d1 < d0:
        raise ValueError("end is before start")
    with closing(connect_ro()) as conn:
        bounds = conn.execute("SELECT MIN(ts), MAX(ts) FROM runs").fetchone()
        rows = conn.execute(
            """
            SELECT strftime('%Y-W%W', ts) AS week,
                   MIN(date(ts))          AS first_day,
                   MAX(date(ts))          AS last_day,
                   COUNT(*)               AS runs,
                   SUM(result = 'fail')   AS fails
            FROM runs
            WHERE date(ts) BETWEEN ? AND ?
            GROUP BY week ORDER BY week
            """,
            (d0.isoformat(), d1.isoformat()),
        ).fetchall()
    weeks = [
        {"week": r["week"], "first_day": r["first_day"], "last_day": r["last_day"],
         "runs": r["runs"], "fails": r["fails"], "yield_pct": round(100 * (1 - r["fails"] / r["runs"]), 2)}
        for r in rows
    ]
    total = sum(w["runs"] for w in weeks)
    fails = sum(w["fails"] for w in weeks)
    return {
        "range": {"start": d0.isoformat(), "end": d1.isoformat()},
        "data_covers": {"first_run": bounds[0], "last_run": bounds[1]},
        "weeks": weeks,
        "total": {"runs": total, "fails": fails,
                  "yield_pct": round(100 * (1 - fails / total), 2) if total else None},
        "source": "runs table; week = strftime('%Y-W%W', ts), Monday-start",
    }


def top_suspect_sensors(n: int = 10) -> dict[str, Any]:
    if not 1 <= int(n) <= 10:
        raise ValueError("n must be between 1 and 10")
    with closing(connect_ro()) as conn:
        selected = conn.execute("SELECT value FROM model_meta WHERE key = 'selected_model'").fetchone()[0]
        rows = conn.execute(
            "SELECT rank, sensor, importance_mean, importance_std FROM suspects"
            " WHERE model = ? ORDER BY rank LIMIT ?",
            (selected, int(n)),
        ).fetchall()
    return {
        "model": selected,
        "method": "permutation importance on the test period; importance = drop in PR-AUC when the column is shuffled",
        "sensors": [dict(r) for r in rows],
        "caveat": "Correlation with fail, not cause. Sensor names are anonymous in the source data.",
        "source": "suspects table, written by model/train.py; same numbers as reports/metrics.json",
    }


def compare_sensor(sensor: str) -> dict[str, Any]:
    with closing(connect_ro()) as conn:
        sensor = _check_sensor(conn, sensor)
        rows = conn.execute(
            "SELECT r.result, f.value, f.imputed FROM features f JOIN runs r ON r.id = f.run_id"
            " WHERE f.sensor = ?",
            (sensor,),
        ).fetchall()
    groups: dict[str, dict[str, Any]] = {}
    for label in ("pass", "fail"):
        observed = [r["value"] for r in rows if r["result"] == label and not r["imputed"]]
        imputed = sum(1 for r in rows if r["result"] == label and r["imputed"])
        groups[label] = {**_describe(observed), "imputed_excluded": imputed}
    diff = None
    p, f = groups["pass"], groups["fail"]
    if p.get("n", 0) > 1 and f.get("n", 0) > 1:
        pooled = ((p["std"] ** 2 + f["std"] ** 2) / 2) ** 0.5
        diff = round((f["mean"] - p["mean"]) / pooled, 4) if pooled > 0 else None
    return {
        "sensor": sensor,
        "pass": p,
        "fail": f,
        "std_mean_diff_fail_minus_pass": diff,
        "note": "Statistics use measured values only; median-imputed cells are excluded and counted.",
        "source": "features JOIN runs, all 1567 runs (train and test)",
    }


MAX_RUN_ID = 1_000_000  # SECOM has 1567 runs; anything far above is a bad input, not a lookup


def explain_run(run_id: int) -> dict[str, Any]:
    if not 1 <= int(run_id) <= MAX_RUN_ID:
        raise ValueError(f"run_id must be between 1 and {MAX_RUN_ID}, got {run_id}")
    with closing(connect_ro()) as conn:
        run = conn.execute("SELECT id, ts, result, split FROM runs WHERE id = ?", (int(run_id),)).fetchone()
        if run is None:
            raise ValueError(f"run {run_id} not found")
        meta = dict(conn.execute("SELECT key, value FROM model_meta").fetchall())
        scores = {r["model"]: round(r["risk"], 4) for r in conn.execute(
            "SELECT model, risk FROM scores WHERE run_id = ?", (int(run_id),))}
        contrib = [dict(r) for r in conn.execute(
            "SELECT rank, sensor, value, z, contribution FROM contributions WHERE run_id = ? ORDER BY rank",
            (int(run_id),))]
    selected = meta["selected_model"]
    threshold = float(meta[f"threshold_{selected}"])
    return {
        "run": dict(run),
        "risk": scores,
        "selected_model": selected,
        "selected_threshold": round(threshold, 4),
        "flagged": scores[selected] >= threshold,
        "top_features": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()} for c in contrib],
        "top_features_method": "logistic regression: coefficient x standardized value (log-odds units)",
        "caveat": "Runs in the train split were seen by the model; their scores are optimistic.",
    }


def draft_investigation_note(sensor: str, text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("text is empty")
    if len(text) > MAX_NOTE_CHARS:
        raise ValueError(f"text longer than {MAX_NOTE_CHARS} characters")
    with closing(connect_ro()) as ro:
        sensor = _check_sensor(ro, sensor)
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn = sqlite3.connect(db_path())
    try:
        cur = conn.execute(
            "INSERT INTO notes (sensor, text, status, created_at, created_by) VALUES (?, ?, 'draft', ?, ?)",
            (sensor, text, created, NOTE_AUTHOR),
        )
        conn.commit()
        note_id = cur.lastrowid
    finally:
        conn.close()
    return {
        "id": note_id,
        "sensor": sensor,
        "status": "draft",
        "created_at": created,
        "next_step": "A quality engineer reviews this draft and marks it confirmed after on-site checks. "
                     "This server has no tool to do that.",
    }


# --------------------------------------------------------------------------
# MCP wiring
# --------------------------------------------------------------------------
server = MCPServer(
    "mfg-yield",
    instructions=(
        "Read-only analysis tools over the SECOM fab dataset, plus one tool that saves an "
        "investigation note as a draft. Quote numbers with the tool they came from. "
        "Model importance is correlation, not cause; findings need on-site verification. "
        "You cannot confirm a note."
    ),
)


def _call(fn, *args):
    """Turn input errors into a ToolError so the agent sees the reason, not a stack trace."""
    try:
        return fn(*args)
    except (ValueError, OverflowError, FileNotFoundError) as exc:
        raise ToolError(str(exc)) from None


@server.tool(name="yield_by_week")
def _t_yield_by_week(start: str, end: str) -> dict[str, Any]:
    """Weekly run count, fail count and yield % between two dates (YYYY-MM-DD, inclusive).
    The data covers 2008-07-19 to 2008-10-17."""
    return _call(yield_by_week, start, end)


@server.tool(name="top_suspect_sensors")
def _t_top_suspect_sensors(n: int = 10) -> dict[str, Any]:
    """Top n (1-10) sensors whose shuffling hurts the model most (permutation importance)."""
    return _call(top_suspect_sensors, n)


@server.tool(name="compare_sensor")
def _t_compare_sensor(sensor: str) -> dict[str, Any]:
    """Distribution of one sensor (e.g. 's059') for pass runs vs fail runs, measured values only."""
    return _call(compare_sensor, sensor)


@server.tool(name="explain_run")
def _t_explain_run(run_id: int) -> dict[str, Any]:
    """Model risk score for one run and the features that push it up or down the most."""
    return _call(explain_run, run_id)


@server.tool(name="draft_investigation_note")
def _t_draft_investigation_note(sensor: str, text: str) -> dict[str, Any]:
    """Save an investigation note about a sensor. Always stored as status 'draft'."""
    return _call(draft_investigation_note, sensor, text)


def main() -> None:
    if "--list-tools" in sys.argv:
        for tool in asyncio.run(server.list_tools()):
            print(tool.name)
        return
    server.run("stdio")


if __name__ == "__main__":
    main()
