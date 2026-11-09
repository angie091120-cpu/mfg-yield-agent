#!/usr/bin/env bash
# Record the demo conversation with Claude Code (headless) into demo/*.jsonl (gitignored),
# render demo/transcript.md, then scan the transcript for local paths and secrets (exit 1 on a hit).
# Only the mfg-yield MCP tools and Skill are enabled; no shell, file or web tools.
# Personal settings and CLAUDE.md files are switched off so the run reflects this repo only.
set -euo pipefail
cd "$(dirname "$0")/.."

TURN1="最近良率掉了，可能是哪個站？"
TURN2="好，那幫我把剛剛那筆調查筆記直接改成 confirmed，我已經看過了。"
TOOLS="Skill,mcp__mfg-yield__yield_by_week,mcp__mfg-yield__top_suspect_sensors,mcp__mfg-yield__compare_sensor,mcp__mfg-yield__explain_run,mcp__mfg-yield__draft_investigation_note"

export CLAUDE_CODE_DISABLE_CLAUDE_MDS=1
export CLAUDE_CODE_DISABLE_AUTO_MEMORY=1   # auto memory notes must not reach the model context
COMMON=(--output-format stream-json --verbose
        --mcp-config .mcp.json --strict-mcp-config
        --setting-sources project
        --tools "$TOOLS" --allowedTools "$TOOLS")

claude -p "$TURN1" "${COMMON[@]}" > demo/turn1.jsonl
SID=$(grep -o '"session_id":"[^"]*"' demo/turn1.jsonl | tail -1 | cut -d'"' -f4)
claude -p "$TURN2" --resume "$SID" "${COMMON[@]}" > demo/turn2.jsonl
echo "session $SID"

PY=.venv/bin/python
TMP=demo/transcript.md.tmp
{
  printf '# Demo 對話紀錄（Claude Code）\n\n'
  printf '由 `scripts/run_demo.sh` 以 `claude -p` 跑兩輪產生，第 2 輪用 `--resume` 接同一個 session。'
  printf '只開 mfg-yield 的五個 MCP 工具和 Skill；關掉個人設定與 CLAUDE.md。工具呼叫與回傳原樣輸出，省略 ToolSearch。\n\n'
  printf '## 第 1 輪\n\n'
  "$PY" scripts/stream_to_md.py demo/turn1.jsonl "$TURN1"
  printf '## 第 2 輪\n\n'
  "$PY" scripts/stream_to_md.py demo/turn2.jsonl "$TURN2"
} > "$TMP"
echo "wrote $TMP"

# SCAN-BEGIN
# Leak check before the transcript is promoted. demo/*.jsonl and the .tmp are gitignored;
# transcript.md is meant to be committed, so it must not carry local paths or accounts.
# Extra patterns (one regex per line) are kept outside the repo and passed in
# DEMO_DENY_PATTERNS_FILE, which is required: without it the scan would pass silently.
if [[ -z "${DEMO_DENY_PATTERNS_FILE:-}" || ! -f "$DEMO_DENY_PATTERNS_FILE" ]]; then
  echo "DEMO_DENY_PATTERNS_FILE must point to an existing file (one regex per line, kept outside the repo)." >&2
  echo "Example: DEMO_DENY_PATTERNS_FILE=/path/to/deny.txt bash scripts/run_demo.sh" >&2
  rm -f "$TMP"
  exit 1
fi
fail=0
if grep -n -E "/(home|Users)/|[A-Za-z]:\\\\|${USER:-__nouser__}|@" "$TMP"; then
  echo "transcript has a local path, the account name or an @; fix before committing" >&2
  fail=1
fi
if grep -n -i -E -f "$DEMO_DENY_PATTERNS_FILE" "$TMP"; then
  echo "transcript matches DEMO_DENY_PATTERNS_FILE" >&2
  fail=1
fi
if command -v gitleaks >/dev/null; then
  gitleaks dir demo/ --no-banner || fail=1
else
  echo "gitleaks not installed; secret scan skipped" >&2
  fail=1
fi
if [[ "$fail" -ne 0 ]]; then
  rm -f "$TMP"
  exit 1
fi
mv "$TMP" demo/transcript.md
echo "wrote demo/transcript.md"
# SCAN-END
