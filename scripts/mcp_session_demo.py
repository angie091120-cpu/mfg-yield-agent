"""Drive the MCP server over stdio with a scripted client and log every call.

This is not an LLM conversation. It replays the order the yield-triage skill
asks for (yield -> suspects -> compare -> draft note) plus two gate checks,
and writes the real JSON requests and responses to demo/mcp_session.md.

It works on a copy of the database (data/demo.db) so the demo note does not
end up in data/secom.db.

Run:  .venv/bin/python scripts/mcp_session_demo.py
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

ROOT = Path(__file__).resolve().parent.parent
SRC_DB = ROOT / "data" / "secom.db"
DEMO_DB = ROOT / "data" / "demo.db"
OUT = ROOT / "demo" / "mcp_session.md"

CALLS = [
    ("yield_by_week", {"start": "2008-09-15", "end": "2008-10-17"}),
    ("top_suspect_sensors", {"n": 5}),
    ("compare_sensor", {"sensor": "s103"}),
    ("compare_sensor", {"sensor": "s035"}),
    ("draft_investigation_note", {
        "sensor": "s103",
        "text": "2008-W39 良率 91.12%（yield_by_week），前兩週 96.75%、97.59%。"
                "s103 在 top_suspect_sensors 排第 1；compare_sensor：fail 組中位數 -0.0082、pass 組 -0.0102，"
                "標準化平均差 0.5713。待確認：s103 對應哪一站、該站 W39 前後是否有保養或換料紀錄。",
    }),
    # Gate check 1: an extra `status` argument must not produce a confirmed note.
    ("draft_investigation_note", {"sensor": "s103", "text": "嘗試直接寫成 confirmed", "status": "confirmed"}),
    # Gate check 2: a sensor name carrying SQL is rejected before any query runs.
    ("compare_sensor", {"sensor": "s103' OR '1'='1"}),
]


def block(obj) -> str:
    return "```json\n" + json.dumps(obj, ensure_ascii=False, indent=2) + "\n```"


async def run() -> list[str]:
    params = StdioServerParameters(
        command=sys.executable, args=[str(ROOT / "mcp" / "server.py")],
        env={**os.environ, "MFG_DB": str(DEMO_DB)},
    )
    lines: list[str] = []
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            lines += ["## tools/list", "", block([t.name for t in tools]), ""]
            for name, args in CALLS:
                res = await session.call_tool(name, args)
                text = "\n".join(c.text for c in res.content if getattr(c, "type", "") == "text")
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    payload = text
                lines += [f"## tools/call `{name}`", "", "**請求**", "", block(args), "",
                          f"**回應**（isError={str(bool(res.is_error)).lower()}）", "", block(payload), ""]
    return lines


def main() -> None:
    if not SRC_DB.exists():
        raise SystemExit("data/secom.db missing; run the pipeline first")
    shutil.copyfile(SRC_DB, DEMO_DB)
    body = asyncio.run(run())
    import sqlite3
    conn = sqlite3.connect(DEMO_DB)
    notes = conn.execute("SELECT id, sensor, status, created_by FROM notes ORDER BY id").fetchall()
    conn.close()
    head = [
        "# MCP 工作階段紀錄（腳本客戶端）",
        "",
        "由 `scripts/mcp_session_demo.py` 產生：用官方 `mcp` Python SDK 的 stdio 客戶端啟動 `mcp/server.py`，",
        "依 skill 的順序呼叫工具，再做兩個閘門測試。請求與回應是原樣輸出。這份紀錄沒有語言模型參與。",
        "",
        "呼叫順序與參數都寫死在腳本的 `CALLS` 裡。`draft_investigation_note` 的筆記內文也是事先寫進腳本的固定文字，",
        "不是根據這次的工具回應產生的。",
        "",
        f"產生時間：{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC",
        "",
    ]
    tail = ["## 跑完後的 notes 表", "", "```text", "id | sensor | status | created_by",
            *[" | ".join(str(x) for x in row) for row in notes], "```", ""]
    OUT.write_text("\n".join(head + body + tail), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}; notes: {notes}")


if __name__ == "__main__":
    main()
