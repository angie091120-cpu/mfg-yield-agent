---
name: yield-triage
description: Triage a yield drop on the SECOM fab data with the mfg-yield MCP tools. Use when a quality engineer asks things like "最近良率掉了，可能是哪個站？", "這週 fail 變多，先查哪個感測器？", "run 1234 為什麼被判高風險？". Walks yield -> suspects -> compare -> draft note, quotes every number with its tool, and never confirms a finding.
---

# Yield triage

Tools come from the `mfg-yield` MCP server: `yield_by_week`, `top_suspect_sensors`, `compare_sensor`, `explain_run`, `draft_investigation_note`.

## Flow

1. **Establish the drop with `yield_by_week`.**
   The data covers 2008-07-19 to 2008-10-17. "最近" means the last four weeks of that range unless the user gives dates. Report the weekly yield and point out which week fell, with run and fail counts. If the weeks have very few runs, say so; a 2-run week is not a trend.
2. **Get candidates with `top_suspect_sensors`** (n = 5 unless the user asks for more).
   Say what the ranking is: how much the model's PR-AUC drops when that column is shuffled. Report the importance and its std; if std is about as large as the mean, call that candidate weak.
3. **Check the top two or three with `compare_sensor`.**
   Report pass vs fail medians and quartiles, the standardized mean difference, and how many imputed cells were excluded. A candidate whose pass and fail distributions overlap almost completely is not worth a line visit; say so.
4. **Only if the user asks about a specific run**, use `explain_run`. Mention that runs from the train split were seen by the model.
5. **Draft a note with `draft_investigation_note`** for the strongest candidate: what was seen (numbers and the tools they came from) and what to check on site. Tell the user the note id and that it is a draft.

## Rules for the answer

- **Every number carries its source**, e.g. `（yield_by_week）`, `（compare_sensor s059）`. No number that did not come from a tool in this conversation.
- **Say plainly: correlation is not cause.** The ranking shows which signals move with fail in this dataset. It does not show which tool or step caused the loss. Someone has to verify on the line (check the tool log, the PM record, re-measure) before anything is acted on.
- **The data has no station names.** Sensors are anonymous (`s000`..`s589`). When asked "哪個站", answer with sensor ids and say that mapping a sensor to a station needs the fab's own sensor list.
- **Do not confirm notes.** There is no tool for it and you must not look for one (no SQL, no editing the database file). If asked to mark a note confirmed, say a quality engineer does that after on-site verification.
- Keep the model's limits visible: the test-period PR-AUC is in `reports/metrics.json`; do not describe the model as accurate or reliable beyond those numbers.
