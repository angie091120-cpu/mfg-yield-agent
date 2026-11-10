"""D1-D4: the single-page demo (docs/demo/index.html) is generated, sourced, self-contained and responsive-safe.

The page is built by scripts/build_demo_page.py from reports/, demo/mcp_session.md and data/secom.db.
Tests that need the local database skip when data/secom.db is absent (it is gitignored).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sqlite3
from html import unescape
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAGE_PATH = ROOT / "docs" / "demo" / "index.html"
DB_PATH = ROOT / "data" / "secom.db"
PAGE_RAW = PAGE_PATH.read_text(encoding="utf-8")
PAGE = PAGE_RAW.replace("<wbr>", "")  # word-boundary hints stripped so text assertions read as plain prose
METRICS_TEXT = (ROOT / "reports" / "metrics.json").read_text(encoding="utf-8")
METRICS = json.loads(METRICS_TEXT)
CLEANING = (ROOT / "reports" / "cleaning.md").read_text(encoding="utf-8")
SESSION = (ROOT / "demo" / "mcp_session.md").read_text(encoding="utf-8")


def _data() -> dict:
    m = re.search(r'<script type="application/json" id="demo-data">(.*?)</script>', PAGE, re.S)
    assert m, "embedded JSON block missing"
    return json.loads(m.group(1).replace("<\\/", "</"))


def _builder():
    spec = importlib.util.spec_from_file_location("build_demo_page", ROOT / "scripts" / "build_demo_page.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _visible_text(html: str) -> str:
    html = re.sub(r"<(script|style)\b.*?</\1>", " ", html, flags=re.S)
    html = re.sub(r"<[^>]+>", " ", html)
    return unescape(html)


# ---- D1: numbers come from the sources -------------------------------------------------

def test_headline_numbers_match_cleaning_report_and_session():
    d = _data()["counts"]
    runs = int(re.search(r"筆數（runs）\s*\|\s*(\d+)", CLEANING).group(1))
    before, after = map(int, re.search(r"感測欄數\s*\|\s*(\d+)\s*\|\s*(\d+)", CLEANING).groups())
    tools = json.loads(re.search(r"## tools/list\s+```json\n(.*?)```", SESSION, re.S).group(1))
    assert d == {"runs": runs, "sensors_before": before, "sensors_after": after, "tools": len(tools)}
    assert (runs, before, after, len(tools)) == (1567, 590, 444, 5)
    h1 = re.search(r"<h1[^>]*>(.*?)</h1>", PAGE, re.S).group(1)
    nums = re.findall(r'<span class="n">(\d+)</span>', h1)
    assert nums == [str(runs), str(before), str(after), str(len(tools))]


def test_weekly_series_adds_up_to_cleaning_report_and_matches_session_weeks():
    weeks = _data()["weeks"]
    total_fail = int(re.search(r"全體 [\d.]+%（(\d+) 筆）", CLEANING).group(1))
    runs = int(re.search(r"筆數（runs）\s*\|\s*(\d+)", CLEANING).group(1))
    assert sum(w["runs"] for w in weeks) == runs
    assert sum(w["fails"] for w in weeks) == total_fail
    for w in weeks:
        assert w["yield_pct"] == round(100 * (w["runs"] - w["fails"]) / w["runs"], 2)
    # W37..W41 were also returned by the yield_by_week call in the recorded session
    block = SESSION.split("## tools/call `yield_by_week`")[1].split("## tools/call")[0]
    resp = json.loads(re.findall(r"```json\n(.*?)```", block, re.S)[1])
    by_week = {w["week"]: w for w in weeks}
    for sw in resp["weeks"]:
        got = by_week[sw["week"]]
        assert (got["runs"], got["fails"], got["yield_pct"]) == (sw["runs"], sw["fails"], sw["yield_pct"])


@pytest.mark.skipif(not DB_PATH.exists(), reason="data/secom.db not built")
def test_weekly_series_matches_database():
    db = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        rows = db.execute("select strftime('%Y-W%W', ts), count(*), sum(result='fail') from runs group by 1 order by 1").fetchall()
    finally:
        db.close()
    assert [(w["week"], w["runs"], w["fails"]) for w in _data()["weeks"]] == rows


def test_low_week_is_lowest_from_week_34_and_not_claimed_as_global_minimum():
    d = _data()
    later = [w for w in d["weeks"] if int(w["week"].split("-W")[1]) >= 34]
    assert d["low_week"] == min(later, key=lambda w: w["yield_pct"])
    assert d["low_week"]["week"] == "2008-W39" and d["low_week"]["yield_pct"] == 91.12
    assert min(w["yield_pct"] for w in d["weeks"]) < d["low_week"]["yield_pct"]  # the page says so out loud
    assert "這不是全期最低" in PAGE


def test_models_sensors_and_split_match_metrics_json():
    d = _data()
    m = METRICS
    b = m["baseline_all_pass"]
    assert d["models"]["baseline"] == {"pr_auc": b["pr_auc"], "tp": b["confusion"]["tp"], "fp": b["confusion"]["fp"]}
    for key in ("logreg", "random_forest"):
        t = m["models"][key]["test"]
        got = d["models"][key]
        assert (got["pr_auc"], got["roc_auc"], got["tp"], got["fp"]) == (t["pr_auc"], t["roc_auc"], t["confusion"]["tp"], t["confusion"]["fp"])
    for key in ("logreg", "random_forest"):
        want = [(s["sensor"], s["importance_mean"], s["importance_std"]) for s in m["permutation_importance"]["top10"][key]]
        assert [(s["sensor"], s["mean"], s["std"]) for s in d["top10"][key]] == want
    rf = {s["sensor"] for s in d["top10"]["random_forest"]}
    lr = {s["sensor"] for s in d["top10"]["logreg"]}
    assert set(d["both"]) == rf & lr and len(d["both"]) == 3
    assert d["split"] == {"test_rows": m["split"]["test_rows"], "test_fail": m["split"]["test_fail"], "val_fail": m["split"]["val_fail"]}


def test_every_embedded_key_has_its_own_source_check():
    """Each key of the page JSON is compared to a value rebuilt from the sources; an unlisted key fails."""
    d = _data()
    m = METRICS
    b = m["baseline_all_pass"]
    top = m["permutation_importance"]["top10"]

    def tops(key):
        return [{"sensor": s["sensor"], "mean": s["importance_mean"], "std": s["importance_std"]} for s in top[key]]

    def model(key):
        t = m["models"][key]["test"]
        return {"pr_auc": t["pr_auc"], "roc_auc": t["roc_auc"], "tp": t["confusion"]["tp"], "fp": t["confusion"]["fp"]}

    runs = int(re.search(r"筆數（runs）\s*\|\s*(\d+)", CLEANING).group(1))
    before, after = map(int, re.search(r"感測欄數\s*\|\s*(\d+)\s*\|\s*(\d+)", CLEANING).groups())
    tools = json.loads(re.search(r"## tools/list\s+```json\n(.*?)```", SESSION, re.S).group(1))
    derived = _independent_derived()
    later = [w for w in d["weeks"] if int(w["week"].split("-W")[1]) >= 34]
    rf_names = {s["sensor"] for s in top["random_forest"]}
    lr_names = {s["sensor"] for s in top["logreg"]}
    expected = {
        "counts": {"runs": runs, "sensors_before": before, "sensors_after": after, "tools": len(tools)},
        "top10": {"random_forest": tops("random_forest"), "logreg": tops("logreg")},
        "both": [s["sensor"] for s in top["random_forest"] if s["sensor"] in lr_names],
        "models": {"baseline": {"pr_auc": b["pr_auc"], "tp": b["confusion"]["tp"], "fp": b["confusion"]["fp"]},
                   "logreg": model("logreg"), "random_forest": model("random_forest")},
        "split": {"test_rows": m["split"]["test_rows"], "test_fail": m["split"]["test_fail"], "val_fail": m["split"]["val_fail"]},
        "low_week": min(later, key=lambda w: w["yield_pct"]),
        "earlier_lower_weeks": derived["earlier_lower_weeks"],
        "earlier_min_runs": derived["earlier_min_runs"],
        "rf_noisy": derived["rf_noisy"],
        "replay_weeks": derived["replay_weeks"],
    }
    assert set(expected["both"]) == rf_names & lr_names
    # weeks: totals from the cleaning report, shape fixed (rows vs the database and session are checked in their own tests)
    total_fail = int(re.search(r"全體 [\d.]+%（(\d+) 筆）", CLEANING).group(1))
    assert sum(w["runs"] for w in d["weeks"]) == runs and sum(w["fails"] for w in d["weeks"]) == total_fail
    assert all(set(w) == {"week", "runs", "fails", "yield_pct"} for w in d["weeks"])
    expected["weeks"] = d["weeks"]
    missing = set(d) - set(expected)
    assert not missing, f"keys in the page JSON with no independent source check: {sorted(missing)}"
    for key in d:
        assert d[key] == expected[key], key


def test_page_does_not_overstate_the_model():
    """README: both models are only a little above all-pass, and the all-pass PR-AUC equals the test fail rate."""
    d = _data()
    assert d["models"]["baseline"]["pr_auc"] == METRICS["baseline_all_pass"]["pr_auc"] == METRICS["split"]["test_fail_rate"]
    for key in ("logreg", "random_forest"):
        assert 0 < d["models"][key]["pr_auc"] - d["models"]["baseline"]["pr_auc"] < 0.05
    for phrase in ("只比「全猜良品」高一點", "相關不等於因果", "沒有語言模型參與", "不能拿來當自動判退的閘門"):
        assert phrase in PAGE


def test_every_visible_number_traces_to_a_source_or_the_embedded_json():
    token = re.compile(r"\d+(?:\.\d+)?")
    allowed = set(token.findall(METRICS_TEXT + CLEANING + SESSION))
    d = _data()
    # numbers the page derives from the sources (recomputed here, not read back from the page JSON)
    derived = _independent_derived()
    allowed |= {str(derived["earlier_lower_weeks"]), str(derived["earlier_min_runs"]), *map(str, derived["replay_weeks"])}
    allowed |= {f"{100 * (w['runs'] - w['fails']) / w['runs']:.2f}" for w in d["weeks"]}
    allowed |= {str(w["runs"]) for w in d["weeks"]} | {str(w["fails"]) for w in d["weeks"]}
    allowed |= {str(int(w["week"].split("-W")[1])) for w in d["weeks"]} | {"34"}  # week labels; 34 is the fixed range start
    allowed |= {"10", "100", "90", "80", "75", "0.5", "4.0", "2008", "2026", "24", "5", "3", "2", "6", "1"}  # axis ticks, licence, list sizes
    text = _visible_text(PAGE)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"doi:\S+", " ", text)
    unknown = sorted(set(token.findall(text)) - allowed)
    assert not unknown, f"numbers on the page with no source: {unknown}"


def test_replay_shows_the_session_json_verbatim():
    blocks = SESSION.split("## tools/call ")[1:]
    assert len(blocks) == 7
    from html import escape
    for k, block in enumerate(blocks):
        for body in re.findall(r"```json\n(.*?)```", block, re.S)[:2]:
            assert escape(body.rstrip("\n")) in PAGE, f"call {k + 1}: JSON body not shown verbatim"
    assert PAGE.count("<details") == 7


@pytest.mark.skipif(not DB_PATH.exists(), reason="data/secom.db not built")
def test_build_is_deterministic_and_committed_page_is_current():
    b = _builder()
    first, second = b.build(), b.build()
    assert hashlib.sha256(first.encode()).hexdigest() == hashlib.sha256(second.encode()).hexdigest()
    assert first == PAGE_RAW, "docs/demo/index.html is stale; rerun python scripts/build_demo_page.py"


# ---- D2: zero external requests ----------------------------------------------------------

ALLOWED_URLS = {
    "https://github.com/angie091120-cpu/mfg-yield-agent",
    "https://doi.org/10.24432/C54305",
    "https://creativecommons.org/licenses/by/4.0/",
}


def test_only_the_repo_uci_and_licence_links_appear():
    urls = set(re.findall(r"https?://[^\s\"'<>)]+", PAGE))
    assert urls == ALLOWED_URLS


def test_no_external_resource_loading():
    assert not re.search(r"<script[^>]+\bsrc=", PAGE)
    assert not re.search(r"<link[^>]+href=[\"'](?!data:)", PAGE)
    assert not re.search(r"@import|url\(\s*[\"']?(?:https?:)?//", PAGE)
    assert not re.search(r"<(img|iframe|video|audio|source|embed|object)\b", PAGE)
    assert "googletagmanager" not in PAGE and "gtag(" not in PAGE


def test_no_forbidden_strings():
    """Employer names, local paths and Windows drive paths must not appear (pieces joined so this file stays clean too)."""
    words = ["mai" + "agent", "思" + "邁", "安" + "盛", "ixen" + "sor", "/ho" + "me/", "C:" + "\\\\"]
    low = PAGE.lower()
    for w in words:
        assert w.lower() not in low, w
    build_src = (ROOT / "scripts" / "build_demo_page.py").read_text(encoding="utf-8").lower()
    for w in words:
        assert w.lower() not in build_src, w


# ---- D4: responsive and accessibility guards that a grep can check -------------------------

def test_document_basics():
    assert PAGE.startswith("<!doctype html>")
    assert '<html lang="zh-Hant">' in PAGE
    assert PAGE.count("<h1") == 1
    assert 'name="viewport"' in PAGE and "width=device-width" in PAGE
    assert re.search(r'<meta name="description" content="[^"]{20,}"', PAGE)
    assert "<title>" in PAGE and "</title>" in PAGE


def test_reduced_motion_branch_exists():
    assert "@media(prefers-reduced-motion:reduce)" in PAGE


def test_heading_is_chunked_so_it_cannot_break_inside_a_word():
    h1 = re.search(r"<h1[^>]*>(.*?)</h1>", PAGE, re.S).group(1)
    plain = re.sub(r"<[^>]+>", "", h1)
    assert plain == "用 1567 筆半導體製程紀錄，把 590 個感測欄清成 444 個，再交給 5 個工具查良率。"
    assert h1.count('class="nb"') >= 6


def test_table_header_cannot_wrap_vertically():
    css = re.search(r"<style>(.*?)</style>", PAGE, re.S).group(1)
    assert re.search(r"thead th\{[^}]*white-space:nowrap", css)
    assert re.search(r"th\.rank\{[^}]*white-space:nowrap", css)


def test_chart_labels_are_html_not_svg_text():
    svg = re.search(r"<svg.*?</svg>", PAGE, re.S).group(0)
    assert "<text" not in svg
    assert 'class="yaxis"' in PAGE and 'class="xaxis"' in PAGE


def test_both_selected_tag_has_enough_contrast():
    """WCAG AA for the 13px bold tag: --accent-ink on --accent-bg must reach 4.5:1."""
    css = re.search(r"<style>(.*?)</style>", PAGE, re.S).group(1)
    fg = re.search(r"--accent-ink:(#[0-9A-Fa-f]{6})", css).group(1)
    bg = re.search(r"--accent-bg:(#[0-9A-Fa-f]{6})", css).group(1)

    def lum(h: str) -> float:
        c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

    hi, lo = sorted((lum(fg), lum(bg)), reverse=True)
    assert (hi + 0.05) / (lo + 0.05) >= 4.5


# ---- review r1 fixes: recomputed derived numbers, expanded-state layout, escaping, Pages surface ----

def _independent_derived():
    """Recompute the derived page numbers from weeks and metrics without the builder's own code paths."""
    d = _data()
    weeks = d["weeks"]
    low = min((w for w in weeks if int(w["week"].split("-W")[1]) >= 34), key=lambda w: w["yield_pct"])
    early = [w for w in weeks if int(w["week"].split("-W")[1]) < 34]
    rf = METRICS["permutation_importance"]["top10"]["random_forest"]
    block = SESSION.split("## tools/call `yield_by_week`")[1].split("## tools/call")[0]
    resp = json.loads(re.findall(r"```json\n(.*?)```", block, re.S)[1])
    wk = [int(w["week"].split("-W")[1]) for w in resp["weeks"]]
    return {
        "earlier_lower_weeks": sum(1 for w in early if w["yield_pct"] < low["yield_pct"]),
        "earlier_min_runs": min(w["runs"] for w in early),
        "rf_noisy": [s["sensor"] for s in rf if s["importance_std"] >= s["importance_mean"]],
        "replay_weeks": [min(wk), max(wk)],
    }


def test_derived_numbers_are_recomputed_from_weeks_and_metrics_and_shown_on_page():
    want = _independent_derived()
    d = _data()
    for key, val in want.items():
        assert d[key] == val, key
    text = re.sub(r"\s+", "", _visible_text(PAGE))
    assert f"第34週以前有{want['earlier_lower_weeks']}週更低" in text
    assert f"樣本最少的一週只有{want['earlier_min_runs']}筆" in text
    lo, hi = want["replay_weeks"]
    assert f"重播查的是第{lo}到{hi}週" in text
    assert f"隨機森林有{len(want['rf_noisy'])}欄（{'、'.join(want['rf_noisy'])}）的標準差不小於平均" in text
    assert want["earlier_lower_weeks"] > 0 and want["rf_noisy"]  # guards against a vacuous pass


def test_week_numbering_is_stated_as_monday_based():
    assert "週次從週一起算，不是 ISO 週次" in _visible_text(PAGE)


def test_first_use_of_jargon_has_a_plain_explanation():
    text = _visible_text(PAGE)
    for term, gloss in (("skill", "skill（寫給 AI 的作業說明檔"), ("shell", "shell（能直接下指令的終端機）"), ("isError", "isError 是工具回報")):
        assert text.index(gloss) <= text.index(term), term


def test_hero_names_where_the_numbers_come_from():
    assert "取自資料庫、訓練報告與工具的實際回傳" in PAGE


def test_expanded_json_stays_inside_the_panel():
    """C1: a long JSON line must scroll inside its own <pre>, not widen the page (grid item min-width:auto)."""
    css = re.search(r"<style>(.*?)</style>", PAGE, re.S).group(1)
    assert re.search(r"\.io\{[^}]*grid-template-columns:minmax\(0,1fr\)", css)
    pre = re.search(r"(?<![\w.])pre\{([^}]*)\}", css).group(1)
    assert "overflow-x:auto" in pre and "max-width:100%" in pre and "white-space:pre" in pre
    assert re.search(r"\.zone\{[^}]*min-width:0", css)


# ---- escaping ----

@pytest.mark.skipif(not DB_PATH.exists(), reason="data/secom.db not built")
def test_every_interpolated_source_string_is_escaped():
    import copy
    b = _builder()
    src = b.read_sources()
    src["metrics"] = copy.deepcopy(src["metrics"])
    top = src["metrics"]["permutation_importance"]["top10"]
    top["random_forest"][0]["sensor"] = "<x>"
    top["logreg"][0]["sensor"] = "<x>"  # shared name, so it also reaches the "both" sentence
    top["logreg"][1]["importance_mean"] = '"><img src=x onerror=1>'
    src["metrics"]["models"]["logreg"]["test"]["pr_auc"] = "<b>1</b>"
    calls = b.parse_session(src["session"])
    html = b.render(b.build_data(src, calls), b.replay_items(calls, b.low_week(src["weeks"])), b.session_date(src["session"]))
    outside = re.sub(r'<script type="application/json".*?</script>', "", html, flags=re.S)
    assert "<x>" not in outside and "&lt;x&gt;" in outside
    assert "<img" not in outside and "<b>1" not in outside
    assert outside.count("<script") == 0


# ---- Pages surface ----

def test_csp_and_referrer_meta_are_present():
    m = re.search(r'<meta http-equiv="Content-Security-Policy" content="([^"]+)"', PAGE)
    assert m, "CSP meta missing"
    policy = m.group(1)
    for directive in ("default-src 'none'", "object-src 'none'", "base-uri 'none'", "form-action 'none'", "style-src 'unsafe-inline'"):
        assert directive in policy
    assert "script-src" not in policy and "'unsafe-eval'" not in policy
    assert '<meta name="referrer" content="no-referrer">' in PAGE
    assert re.search(r'<meta name="robots" content="[^"]+">', PAGE)


def test_docs_folder_holds_only_the_published_files():
    """docs/ is the Pages root: anything put here becomes public, so a new file must be added to this list on purpose."""
    found = {p.relative_to(ROOT).as_posix() for p in (ROOT / "docs").rglob("*") if p.is_file()}
    assert found == {"docs/domain.md", "docs/demo/index.html", "docs/.nojekyll"}
    assert (ROOT / "docs" / ".nojekyll").read_bytes() == b""


# ---- T-105: a word is never broken across two lines ----------------------------------------

def _css() -> str:
    return re.search(r"<style>(.*?)</style>", PAGE_RAW, re.S).group(1)


def _prose_blocks() -> list[str]:
    """Inner HTML of every text block outside <pre>: paragraphs, headings, summaries."""
    body = re.sub(r"<(pre|script|style)\b.*?</\1>", " ", PAGE_RAW, flags=re.S)
    return re.findall(r"<(?:p|h1|h2|h3|span class=\"sum\")\b[^>]*>(.*?)</(?:p|h1|h2|h3|span)>", body, re.S)


def test_body_uses_keep_all_so_cjk_only_breaks_at_marked_places():
    body_rule = re.search(r"(?<![\w.])body\{([^}]*)\}", _css()).group(1)
    assert "word-break:keep-all" in body_rule
    assert "overflow-wrap:break-word" in body_rule  # a too-long run still wraps instead of widening the page


def test_no_marked_break_falls_inside_a_word():
    terms = ["良率", "感測欄", "半導體", "製程", "資料集", "不良", "良品", "閘門", "邏輯迴歸", "隨機森林", "重要度",
             "中位數", "標準差", "平均差", "語言模型", "資料庫", "工具", "筆記", "因果", "測試期", "驗證段", "全猜良品"]
    for t in terms:
        for i in range(1, len(t)):
            assert t[:i] + "<wbr>" + t[i:] not in PAGE_RAW, f"<wbr> splits the word {t}"


def test_every_prose_block_has_a_break_opportunity_at_least_every_nine_cjk_chars():
    """With keep-all a CJK run only breaks at punctuation, spaces or <wbr>; cap the unbroken run so lines can still wrap."""
    blocks = _prose_blocks()
    assert len(blocks) >= 20
    for html in blocks:
        text = re.sub(r"<(?!wbr)[^>]+>", "", html)
        text = unescape(text)
        for run in re.split(r"<wbr>|[^\u4e00-\u9fff]", text):
            assert len(run) <= 9, f"unbroken run of {len(run)} chars: {run!r}"


def test_number_and_unit_are_kept_on_one_line():
    body = re.sub(r"<(pre|script|style)\b.*?</\1>", " ", PAGE_RAW, flags=re.S)
    body = re.sub(r"<span class=\"nb\">.*?</span>", "", body, flags=re.S)  # already-protected spans
    body = re.sub(r"<[^>]+>", "", body)
    loose = re.findall(r"\d+ (?:週|筆|欄|個|名|步)", unescape(body))
    assert not loose, f"number and unit can be split across lines: {loose[:5]}"
    assert '<span class="nb">第 34 週</span>' in PAGE_RAW


# ---- T-106: the replay starts with its first step open ------------------------------------

def test_replay_first_step_is_open_by_default_and_only_the_first():
    tags = re.findall(r"<details( open)?>", PAGE_RAW)
    assert len(tags) == 7
    assert tags[0] == " open" and all(t == "" for t in tags[1:])


# ---- T-107: a JSON line cut at the right edge is visibly cut ------------------------------

def test_json_blocks_show_a_scroll_shadow_and_a_swipe_hint():
    pre = re.search(r"(?<![\w.])pre\{([^}]*)\}", _css()).group(1)
    assert "background-attachment:local,local,scroll,scroll" in pre  # shadow only while content remains on the right
    assert pre.count("linear-gradient") == 4
    assert "overflow-x:auto" in pre  # D4: scrolls inside the block, never widens the page
    hint = re.search(r'<p class="hint">(.*?)</p>', PAGE, re.S).group(1)
    assert "左右滑動" in hint
    assert PAGE.index('class="hint"') < PAGE.index('<ol class="replay">')
