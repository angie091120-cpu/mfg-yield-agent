"""Turn `claude -p --output-format stream-json --verbose` output into Markdown.

Usage: python scripts/stream_to_md.py <stream.jsonl> <user prompt text> >> demo/transcript.md

Tool calls and tool results are copied as-is (pretty-printed JSON when the
result parses as JSON). ToolSearch calls (Claude Code loading tool schemas)
are left out.
"""
from __future__ import annotations

import json
import sys

HIDDEN_TOOLS = {"ToolSearch"}


def short_name(name: str) -> str:
    return name.split("__")[-1] if name.startswith("mcp__") else name


def result_text(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for c in content or []:
        if c.get("type") == "text":
            parts.append(c["text"])
    return "\n".join(parts)


def fmt_result(text: str) -> str:
    try:
        return "```json\n" + json.dumps(json.loads(text), ensure_ascii=False, indent=2) + "\n```"
    except (json.JSONDecodeError, TypeError):
        return "```text\n" + text.strip() + "\n```"


def main() -> None:
    path, prompt = sys.argv[1], sys.argv[2]
    hidden_ids: set[str] = set()
    out = [f"**使用者：** {prompt}", ""]
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        ev = json.loads(line)
        if ev.get("type") == "assistant":
            for block in ev["message"].get("content", []):
                if block.get("type") == "tool_use":
                    if block["name"] in HIDDEN_TOOLS:
                        hidden_ids.add(block["id"])
                        continue
                    out += [f"**工具呼叫：** `{short_name(block['name'])}`", "",
                            "```json", json.dumps(block.get("input", {}), ensure_ascii=False, indent=2), "```", ""]
                elif block.get("type") == "text" and block["text"].strip():
                    out += [f"**Claude：**", "", block["text"].strip(), ""]
        elif ev.get("type") == "user":
            content = ev.get("message", {}).get("content", [])
            if isinstance(content, str):
                continue
            for block in content:
                if block.get("type") == "tool_result" and block.get("tool_use_id") not in hidden_ids:
                    out += ["<details><summary>工具回傳</summary>", "",
                            fmt_result(result_text(block.get("content"))), "", "</details>", ""]
        elif ev.get("type") == "result":
            sid = ev.get("session_id", "")
            out.append(f"<!-- session {sid[:8]}, turns {ev.get('num_turns')} -->")
            out.append("")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
