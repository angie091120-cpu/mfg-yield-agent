#!/usr/bin/env python3
"""Build docs/demo/index.html, the single-page demo.

Sources (read only):
  reports/metrics.json, reports/cleaning.md, demo/mcp_session.md,
  data/secom.db (opened with mode=ro, weekly yield only).
Every number on the page comes from those files and is also written into the
page as JSON (<script id="demo-data">), which tests/test_demo_page.py checks
against the sources. Running the script twice gives byte-identical output.

Usage: python scripts/build_demo_page.py
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "demo" / "index.html"

REPO_URL = "https://github.com/angie091120-cpu/mfg-yield-agent"
UCI_URL = "https://doi.org/10.24432/C54305"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"

# y axis range of the chart (percent); chosen so the lowest point does not touch the frame
Y_LO, Y_HI = 74.0, 101.0
X_PAD = 3.0  # percent of the plot width kept free on each side
RANGE_START_WEEK = 34  # the page compares weeks from here on; see low_week()


# ---------------------------------------------------------------- sources

def read_sources() -> dict:
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
    cleaning = (ROOT / "reports" / "cleaning.md").read_text(encoding="utf-8")
    session = (ROOT / "demo" / "mcp_session.md").read_text(encoding="utf-8")

    runs = int(re.search(r"筆數（runs）\s*\|\s*(\d+)", cleaning).group(1))
    before, after = map(int, re.search(r"感測欄數\s*\|\s*(\d+)\s*\|\s*(\d+)", cleaning).groups())
    total_fail = int(re.search(r"全體 [\d.]+%（(\d+) 筆）", cleaning).group(1))
    tools = json.loads(re.search(r"## tools/list\s+```json\n(.*?)```", session, re.S).group(1))

    db_path = ROOT / "data" / "secom.db"
    if not db_path.exists():
        sys.exit("data/secom.db not found; build it first (python -m pipeline.load_db)")
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        weeks = [
            {"week": w, "runs": n, "fails": f, "yield_pct": round(100 * (n - f) / n, 2)}
            for w, n, f in db.execute(
                "select strftime('%Y-W%W', ts), count(*), sum(result='fail') from runs group by 1 order by 1"
            )
        ]
    finally:
        db.close()
    assert sum(w["runs"] for w in weeks) == runs, "weekly runs do not add up to cleaning.md"
    assert sum(w["fails"] for w in weeks) == total_fail, "weekly fails do not add up to cleaning.md"

    return {
        "metrics": metrics,
        "counts": {"runs": runs, "sensors_before": before, "sensors_after": after, "tools": len(tools)},
        "weeks": weeks,
        "session": session,
    }


def esc_tree(x):
    """Return a copy of x with every string HTML-escaped; numbers pass through. Used for all visible-page interpolation."""
    if isinstance(x, str):
        return escape(x)
    if isinstance(x, dict):
        return {k: esc_tree(v) for k, v in x.items()}
    if isinstance(x, list):
        return [esc_tree(v) for v in x]
    return x


def week_no(week: str) -> int:
    return int(week.split("-W")[1])


def parse_session(session: str) -> list[dict]:
    """Return the tools/call blocks in file order, keeping the original JSON text."""
    calls = []
    for block in session.split("## tools/call ")[1:]:
        name = block.split("`")[1]
        req, res = re.findall(r"```json\n(.*?)```", block, re.S)[:2]
        is_error = "isError=true" in block
        calls.append({
            "tool": name,
            "request_text": req.rstrip("\n"),
            "response_text": res.rstrip("\n"),
            "is_error": is_error,
            "request": json.loads(req),
            "response": json.loads(res),
        })
    return calls


def session_date(session: str) -> str:
    return re.search(r"產生時間：(\d{4}-\d{2}-\d{2})", session).group(1)


def low_week(weeks: list[dict]) -> dict:
    """Lowest weekly yield from week 34 on. The whole-period minimum is in the early, thin weeks."""
    later = [w for w in weeks if week_no(w["week"]) >= RANGE_START_WEEK]
    return min(later, key=lambda w: w["yield_pct"])


# ---------------------------------------------------------------- page data

def build_data(src: dict, calls: list[dict]) -> dict:
    m = src["metrics"]
    c = src["counts"]
    weeks = src["weeks"]
    low = low_week(weeks)
    earlier_lower = [w for w in weeks if week_no(w["week"]) < RANGE_START_WEEK and w["yield_pct"] < low["yield_pct"]]
    top = m["permutation_importance"]["top10"]
    rf = [{"sensor": s["sensor"], "mean": s["importance_mean"], "std": s["importance_std"]} for s in top["random_forest"]]
    lr = [{"sensor": s["sensor"], "mean": s["importance_mean"], "std": s["importance_std"]} for s in top["logreg"]]
    both = [s["sensor"] for s in rf if s["sensor"] in {x["sensor"] for x in lr}]
    noisy = [s["sensor"] for s in rf if s["std"] >= s["mean"]]
    base = m["baseline_all_pass"]
    models = {
        "baseline": {"pr_auc": base["pr_auc"], "tp": base["confusion"]["tp"], "fp": base["confusion"]["fp"]},
        "logreg": {"pr_auc": m["models"]["logreg"]["test"]["pr_auc"], "roc_auc": m["models"]["logreg"]["test"]["roc_auc"],
                   "tp": m["models"]["logreg"]["test"]["confusion"]["tp"], "fp": m["models"]["logreg"]["test"]["confusion"]["fp"]},
        "random_forest": {"pr_auc": m["models"]["random_forest"]["test"]["pr_auc"], "roc_auc": m["models"]["random_forest"]["test"]["roc_auc"],
                          "tp": m["models"]["random_forest"]["test"]["confusion"]["tp"], "fp": m["models"]["random_forest"]["test"]["confusion"]["fp"]},
    }
    sp = m["split"]
    replay_wk = [week_no(w["week"]) for w in calls[0]["response"]["weeks"]]
    return {
        "counts": c,
        "weeks": weeks,
        "low_week": low,
        "replay_weeks": [min(replay_wk), max(replay_wk)],
        "earlier_lower_weeks": len(earlier_lower),
        "earlier_min_runs": min(w["runs"] for w in weeks if week_no(w["week"]) < RANGE_START_WEEK),
        "top10": {"random_forest": rf, "logreg": lr},
        "both": both,
        "rf_noisy": noisy,
        "models": models,
        "split": {"test_rows": sp["test_rows"], "test_fail": sp["test_fail"], "val_fail": sp["val_fail"]},
    }


# ---------------------------------------------------------------- html pieces

CSS = r"""
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%;scroll-behavior:smooth}
:root{color-scheme:light;--bg:#F4F6FB;--surface:#fff;--ink:#1B2147;--muted:#4F5885;--primary:#20329B;--line:#D9DEEE;--hl:#DDE3FA;--accent:#B45309;--accent-ink:#7C3A06;--accent-bg:#FDEFD6;--code:#EEF1FB;
--sans:"Noto Sans TC","PingFang TC","Microsoft JhengHei",system-ui,sans-serif;--mono:ui-monospace,"SFMono-Regular",Menlo,Consolas,"Noto Sans TC",monospace}
body{margin:0;background:var(--bg);color:var(--ink);font:400 16px/1.75 var(--sans);word-break:keep-all;overflow-wrap:break-word}
.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}
.nb{white-space:nowrap}
a{color:var(--primary);text-underline-offset:3px}
a:focus-visible,summary:focus-visible,pre:focus-visible,.scroll:focus-visible{outline:3px solid var(--accent);outline-offset:2px;border-radius:4px}
main{max-width:940px;margin:0 auto;padding:0 20px 56px}
header.top{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:18px 0;font-size:14px;color:var(--muted)}
header.top b{color:var(--primary);font-weight:700;letter-spacing:.02em}
.hero{padding:24px 0 34px}
.hero h1{font-size:clamp(24px,4.4vw,42px);line-height:1.45;font-weight:700;margin:0 0 16px;letter-spacing:-.01em}
.hero h1 .n{color:var(--primary);background:linear-gradient(transparent 62%,var(--hl) 62%);padding:0 .08em;font-variant-numeric:tabular-nums}
.hero p{margin:0 0 10px;max-width:64ch;color:var(--muted)}
.hero .links{font-size:15px}
.zone{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:22px;margin:0 0 18px;min-width:0}
.zone h2{font-size:19px;margin:0 0 6px;font-weight:700;line-height:1.5}
.zone h3{font-size:16px;margin:0 0 6px;font-weight:700;line-height:1.5}
.lede{margin:0 0 16px;color:var(--muted);font-size:15px}
.lede+.lede{margin-top:-8px}
/* chart: SVG draws lines only; every label and dot is HTML so text keeps its size at any width */
.chart{display:grid;grid-template-columns:3.2em minmax(0,1fr);padding-right:20px}
.yaxis{position:relative}
.yaxis span{position:absolute;right:8px;transform:translateY(-50%);font-size:12.5px;color:var(--muted);line-height:1}
.plot{position:relative;aspect-ratio:16/10;background:#FAFBFE;border-radius:6px}
.plot svg{position:absolute;inset:0;width:100%;height:100%;display:block}
.plot .g{stroke:#E3E7F4;stroke-width:1;vector-effect:non-scaling-stroke}
.plot .ln{fill:none;stroke:var(--primary);stroke-width:3;stroke-linejoin:round;stroke-linecap:round;vector-effect:non-scaling-stroke}
.dot{position:absolute;width:26px;height:26px;transform:translate(-50%,-50%);cursor:default}
.dot::after{content:"";position:absolute;left:50%;top:50%;width:9px;height:9px;transform:translate(-50%,-50%);border-radius:50%;background:var(--primary);box-shadow:0 0 0 2px #fff}
.dot.low::before{content:"";position:absolute;left:50%;top:50%;width:22px;height:22px;transform:translate(-50%,-50%);border-radius:50%;border:2.5px solid var(--accent)}
.note{position:absolute;transform:translateX(-50%);background:#FAFBFE;padding:1px 4px;font-size:13.5px;font-weight:700;color:var(--accent-ink);white-space:nowrap;line-height:1.3}
.xaxis{grid-column:2;position:relative;height:30px}
.xaxis span{position:absolute;top:8px;transform:translateX(-50%);font-size:12.5px;color:var(--muted);line-height:1}
.xaxis .opt{display:none}
.table-wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:15px}
th,td{text-align:left;padding:9px 8px;border-bottom:1px solid var(--line);vertical-align:top}
thead th{font-size:13.5px;color:var(--muted);font-weight:500;white-space:nowrap;vertical-align:bottom}
thead th small{display:block;font-size:12.5px;color:var(--muted);font-weight:400}
th.rank{width:3.4em;white-space:nowrap}
td.hit{background:var(--accent-bg)}
td .v{color:var(--muted);font-size:13.5px;margin-left:6px}
td.hit .v{color:#5B3A14}
.mk{color:var(--accent)}
.tag{display:block;font-size:13px;color:var(--accent-ink);font-weight:700;white-space:nowrap;line-height:1.5}
ol.replay{list-style:none;margin:0;padding:0}
ol.replay li{border-bottom:1px solid var(--line)}
ol.replay li.sep{padding:14px 0 6px;font-size:13.5px;color:var(--muted);border-bottom:0}
details>summary{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap;padding:12px 28px 12px 0;min-height:48px;cursor:pointer;font-size:15px;list-style:none;position:relative}
details>summary::-webkit-details-marker{display:none}
details>summary::after{content:"";position:absolute;right:6px;top:1.25em;width:8px;height:8px;border-right:2px solid var(--muted);border-bottom:2px solid var(--muted);transform:rotate(45deg);transition:transform .18s ease}
details[open]>summary::after{transform:rotate(-135deg);top:1.5em}
.stepno{width:24px;height:24px;border-radius:50%;background:var(--primary);color:#fff;font-size:12.5px;display:inline-grid;place-items:center;flex:none}
.stepno.try{background:#fff;color:var(--primary);box-shadow:inset 0 0 0 2px var(--primary)}
.nm{font-weight:700;color:var(--primary);font-size:14.5px}
.sum{color:var(--muted);font-size:14px;flex:1 1 220px;min-width:0}
.io{margin:0 0 14px;display:grid;grid-template-columns:minmax(0,1fr);gap:10px}
.io h4{margin:0 0 4px;font-size:13px;color:var(--muted);font-weight:500}
pre{margin:0;padding:12px 14px;background-color:var(--code);background-image:linear-gradient(to left,var(--code) 35%,rgba(238,241,251,0)),linear-gradient(to right,var(--code) 35%,rgba(238,241,251,0)),linear-gradient(to left,rgba(32,50,155,.30),rgba(32,50,155,0)),linear-gradient(to right,rgba(32,50,155,.30),rgba(32,50,155,0));background-repeat:no-repeat;background-position:right center,left center,right center,left center;background-size:44px 100%,44px 100%,18px 100%,18px 100%;background-attachment:local,local,scroll,scroll;border-radius:8px;font:400 13px/1.65 var(--mono);white-space:pre;overflow-x:auto;overflow-y:auto;max-height:320px;max-width:100%;tab-size:2}
.hint{margin:-8px 0 12px;font-size:13.5px;color:var(--muted)}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.card{background:var(--bg);border-radius:8px;padding:16px 18px;font-size:15px;min-width:0}
.card.wide{grid-column:1/-1}
.card p{margin:0 0 8px}.card p:last-child{margin:0}
.card table{background:transparent}
.card p.after{margin-top:10px}
.card td,.card th{padding:8px 8px}
.cmp td.num,.cmp th.num{text-align:right;white-space:nowrap}
.cmp td:first-child{white-space:nowrap}
.cmp .base td:first-child{color:var(--muted)}
.cmp .base footer{color:var(--muted);font-size:13.5px;padding:6px 0 0}
footer p{margin:0 0 6px}
@media(min-width:640px){
 .plot{aspect-ratio:640/300}
 .xaxis .opt{display:block}
 .zone{padding:24px 26px}
}
@media(max-width:639px){
 main{padding:0 14px 44px}
 .zone{padding:18px 14px}
 .cards{grid-template-columns:1fr}
 td .v{display:block;margin-left:0}
}
@media(prefers-reduced-motion:reduce){
 html{scroll-behavior:auto}
 details>summary::after{transition:none}
}
"""


def chunks(parts: list[str]) -> str:
    """Wrap each phrase in a no-wrap span so a heading can only break between phrases."""
    return "<wbr>".join(f'<span class="nb">{p}</span>' for p in parts)


# Word-boundary source of truth. Every BREAK ("¦") written in a prose template must sit between two words from
# BREAK_WORDS (multi-character words) or BREAK_SINGLES (one-character function words and counters); tests/test_demo_page.py
# checks each <wbr> in the page against these two lists and rejects a break that cuts a listed word in two.
BREAK = "¦"
BREAK_WORDS = (
    "下面 數字 取自 資料庫 重播 不代表 原因 不是 各站 加工 時間 不良組 不良 中位數 不證明 因果 亂猜 等於 不良率 以後 最低 "
    "測試期 邏輯迴歸 依序 點開 請求 回應 先看 結論 兩個 模型 各自 排出 平均值 其中 樣本 最少 一週 只有 半導體 製程 資料集 "
    "給得出 感測欄 編號 只能 排查 順序 參考 透過 右緣 陰影 代表 還有 呼叫 外部 工具 通訊 規則 事先 寫死 筆記 內文 都是 就是 "
    "就能 直接 繞過 閘門 工具層 成立 匿名 感測欄名 檔案 回答 不了 回報 這次 有沒有 出錯 欄位 打亂 誤差 很大 掉了 多少 排名 "
    "這些欄 和不良 一起 變動 相關 方框 長行 可以 左右 滑動 完整 內容 時間戳 測試 衍生 結果 標準化 平均差 效果 只比 每個點 "
    "標的 沒有 執行 站別 語言模型 參與 驗證 比隨機 挑欄 滑鼠 移上去 看筆數 指定 標準差 單位 作業 說明檔 原樣 輸出 調查 "
    "不小於 平均 兩邊 不等於 不良品 抓得 準不準 分數 確認 步驟 草稿 紀錄 產生於 下指令 終端機 自己 查資料 程式 若同時 "
    "到現場 規定 查詢 硬塞 越大 可疑 全期 不可靠 拿來 自動 判退 腳本 這段 這是 一個 這個 週次 從週一 起算 這一步 本身 "
    "就不穩 選模型 隨機森林 頁面 驗證段 重要度 訓練報告 實際回傳 資料 算出 表中 該欄 一週 還有 衍生 這些 只在 隨機 挑欄 "
    "當單位 內的 由這份 由腳本 要由 兩組 兩邊 都進 錯誤 順序 說明 檔案 平均值 規則 事先 筆數 "
    "不能 亂猜 以前 只含 重疊 定好 指標 所以 更低 本頁 橘底 每步 清成 良品組 良品 良率 連到 選到 可疑"
).split()
BREAK_SINGLES = "的是與和或把值當差好看抓改等做人前到年有在由照用加比也都就能但它比個名筆週欄步查帶後這那些各只時第碰回越過含"


def apply_wbr(html: str) -> str:
    """Turn the BREAK markers the templates wrote into <wbr>. With word-break:keep-all a CJK line can only break at
    punctuation, spaces or these marks, so a word is never split. Only text between tags is touched: <pre>, <style>,
    <script> and attribute values stay as they are, and values from the sources never carry the marker (see build())."""
    parts = re.split(r"(<pre\b.*?</pre>|<style>.*?</style>|<script\b.*?</script>)", html, flags=re.S)

    def prose(p: str) -> str:
        p = "".join(t if t.startswith("<") else t.replace(BREAK, "<wbr>") for t in re.split(r"(<[^>]*>)", p))
        # a number and its unit are one word, and so is "前 N 名": keep each on one line (never inside a tag's attributes)
        return re.sub(r"((?:第 )?\d+(?: 到 \d+)? (?:週|筆|欄|個|名|步)|前 \d+(?: 名)?)(?![^<]*>)", r'<span class="nb">\1</span>', p)

    return "".join(p if i % 2 else prose(p) for i, p in enumerate(parts))


def fmt_pct(v: float) -> str:
    return f"{v:.2f}%"


def x_pct(i: int, n: int) -> float:
    return X_PAD + i * (100 - 2 * X_PAD) / (n - 1)


def y_pct(v: float) -> float:
    return (Y_HI - v) / (Y_HI - Y_LO) * 100


def chart(data: dict) -> str:
    weeks = data["weeks"]
    low = data["low_week"]
    n = len(weeks)
    first, last = weeks[0]["week"], weeks[-1]["week"]
    grid = "".join(f'<line class="g" x1="0" x2="100" y1="{y_pct(v):.2f}" y2="{y_pct(v):.2f}"/>' for v in (75, 80, 90, 100))
    pts = " ".join(f"{x_pct(i, n):.2f},{y_pct(w['yield_pct']):.2f}" for i, w in enumerate(weeks))
    ylabels = "".join(f'<span style="top:{y_pct(v):.2f}%">{v}%</span>' for v in (75, 80, 90, 100))
    dots = ""
    xl = ""
    keep = {0, next(i for i, w in enumerate(weeks) if week_no(w["week"]) == RANGE_START_WEEK),
            next(i for i, w in enumerate(weeks) if w["week"] == low["week"]), n - 1}
    for i, w in enumerate(weeks):
        cls = "dot low" if w["week"] == low["week"] else "dot"
        tip = f"W{week_no(w['week'])}：{w['runs']} 筆，{w['fails']} 筆不良，良率 {fmt_pct(w['yield_pct'])}"
        dots += f'<span class="{cls}" style="left:{x_pct(i, n):.2f}%;top:{y_pct(w["yield_pct"]):.2f}%" title="{tip}"></span>'
        xl += f'<span class="{"" if i in keep else "opt"}" style="left:{x_pct(i, n):.2f}%">W{week_no(w["week"])}</span>'
    li = next(i for i, w in enumerate(weeks) if w["week"] == low["week"])
    note = (f'<span class="note mono" style="left:{x_pct(li, n):.2f}%;top:calc({y_pct(low["yield_pct"]):.2f}% + 17px)">'
            f'W{week_no(low["week"])}　{fmt_pct(low["yield_pct"])}</span>')
    label = (f"每週良率折線圖，2008 年第 {week_no(first)} 到 {week_no(last)} 週，"
             f"第 {week_no(low['week'])} 週是第 {RANGE_START_WEEK} 週以後的低點，{fmt_pct(low['yield_pct'])}")
    return (f'<div class="chart"><div class="yaxis" aria-hidden="true">{ylabels}</div>'
            f'<div class="plot" role="img" aria-label="{label}"><svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">{grid}'
            f'<polyline class="ln" points="{pts}"/></svg>{dots}{note}</div>'
            f'<div class="xaxis" aria-hidden="true">{xl}</div></div>')


def sensor_table(data: dict) -> str:
    rf, lr, both = data["top10"]["random_forest"], data["top10"]["logreg"], set(data["both"])

    def cell(s: dict) -> str:
        hit = s["sensor"] in both
        tag = '<span class="tag"><span aria-hidden="true">◆</span> 兩模型都選到</span>' if hit else ""
        return (f'<td class="{"hit" if hit else ""}"><span class="mono">{s["sensor"]}</span>'
                f'<span class="v mono">{s["mean"]}</span>{tag}</td>')

    rows = "".join(f'<tr><th scope="row" class="rank mono">{i + 1}</th>{cell(rf[i])}{cell(lr[i])}</tr>' for i in range(10))
    return ('<div class="table-wrap"><table><thead><tr><th scope="col" class="rank">名次</th>'
            '<th scope="col">隨機森林<small>Random Forest</small></th>'
            '<th scope="col">邏輯迴歸<small>Logistic Regression</small></th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div>')


def replay_items(calls: list[dict], low: dict) -> list[dict]:
    """One entry per tools/call block, with a one-line summary taken from that call's own response."""
    by = calls
    yb, ts, c103, c035, note, gate, inj = by[0], by[1], by[2], by[3], by[4], by[5], by[6]
    lw = next(w for w in yb["response"]["weeks"] if w["week"] == low["week"])
    r = yb["request"]
    items = [
        (yb, f"查 {r['start']} 到 {r['end']}；W{week_no(lw['week'])} {lw['runs']} 筆、{lw['fails']} 筆¦不良，良率 {lw['yield_pct']}%"),
        (ts, f"取前 {ts['request']['n']} 名；第 1 名 {ts['response']['sensors'][0]['sensor']}，重要度 {ts['response']['sensors'][0]['importance_mean']}"),
        (c103, f"{c103['request']['sensor']}：不良組¦中位數 {c103['response']['fail']['median']}，良品組¦中位數 {c103['response']['pass']['median']}；"
               f"標準化¦平均差 {c103['response']['std_mean_diff_fail_minus_pass']}（兩組¦平均值¦的差，用¦標準差¦當單位）"),
        (c035, f"{c035['request']['sensor']}：不良組¦中位數 {c035['response']['fail']['median']}，良品組¦中位數 {c035['response']['pass']['median']}；"
               f"標準化¦平均差 {c035['response']['std_mean_diff_fail_minus_pass']}"),
        (note, f"寫入 {note['request']['sensor']} 的¦調查¦筆記，回應 status = {note['response']['status']}"),
        (gate, f"請求¦硬塞 status: {gate['request']['status']}，回應的 status 仍是 {gate['response']['status']}"),
        (inj, "感測欄名¦帶 SQL 片段，工具¦回錯誤（isError=true），沒有¦執行"),
    ]
    assert gate["request"].get("status") == "confirmed" and gate["response"]["status"] == "draft"
    assert inj["is_error"] and not any(c["is_error"] for c in by[:6])
    return [{"call": c, "summary": s} for c, s in items]


def replay_html(items: list[dict]) -> str:
    out = []
    for k, it in enumerate(items):
        c = it["call"]
        if k == 5:
            out.append('<li class="sep">下面兩步是閘門測試，不屬於查詢流程。</li>')
        circle = f'<span class="stepno mono{" try" if k >= 5 else ""}">{k + 1}</span>'
        out.append(
            f'<li><details{" open" if k == 0 else ""}><summary>{circle}<span class="nm mono">{escape(c["tool"])}</span>'
            f'<span class="sum">{escape(it["summary"])}</span></summary>'
            f'<div class="io"><div><h4>請求</h4><pre tabindex="0" aria-label="第 {k + 1} 步的請求">{escape(c["request_text"])}</pre></div>'
            f'<div><h4>回應（isError={"true" if c["is_error"] else "false"}）</h4>'
            f'<pre tabindex="0" aria-label="第 {k + 1} 步的回應">{escape(c["response_text"])}</pre></div></div></details></li>')
    return '<ol class="replay">' + "".join(out) + "</ol>"


def models_table(data: dict) -> str:
    mdl = data["models"]
    t = data["split"]["test_fail"]

    def row(label: str, m: dict, cls: str = "") -> str:
        return (f'<tr class="{cls}"><th scope="row" class="nb">{label}</th>'
                f'<td class="num mono">{m["pr_auc"]}</td>'
                f'<td class="num mono">{m["tp"]}／{t}</td><td class="num mono">{m["fp"]}</td></tr>')

    return ('<div class="table-wrap"><table class="cmp"><thead><tr><th scope="col">模型</th><th scope="col" class="num">PR-AUC</th>'
            '<th scope="col" class="num">抓到不良</th><th scope="col" class="num">誤報</th></tr></thead><tbody>'
            + row("全猜良品", mdl["baseline"], "base") + row("邏輯迴歸", mdl["logreg"]) + row("隨機森林", mdl["random_forest"])
            + "</tbody></table></div>")


def render(data: dict, items: list[dict], sess_date: str) -> str:
    blob = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).replace("</", "<\\/")
    data = esc_tree(data)  # everything below is page text: escape once, here
    sess_date = escape(sess_date)
    c, low, sp, mdl = data["counts"], data["low_week"], data["split"], data["models"]
    both = "、".join(data["both"])
    noisy = "、".join(data["rf_noisy"])
    w0, w1 = data["weeks"][0]["week"], data["weeks"][-1]["week"]
    r0, r1 = data["replay_weeks"]
    desc = (f"用 {c['runs']} 筆半導體製程紀錄查良率的 Agent 作品展示：每週良率、可疑感測欄、"
            "工具呼叫重播，以及模型的限制。")
    h1 = chunks([f'用 <span class="n">{c["runs"]}</span> 筆', "半導體製程紀錄，",
                 f'把 <span class="n">{c["sensors_before"]}</span> 個', "感測欄",
                 f'清成 <span class="n">{c["sensors_after"]}</span> 個，',
                 f'再交給 <span class="n">{c["tools"]}</span> 個', "工具查良率。"])
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SECOM 良率查詢｜製程良率異常分析 Agent 作品展示</title>
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:; object-src 'none'; base-uri 'none'; form-action 'none'">
<meta name="referrer" content="no-referrer">
<meta name="robots" content="index,follow">
<meta name="description" content="{escape(desc)}">
<meta name="theme-color" content="#20329B">
<meta property="og:type" content="website">
<meta property="og:title" content="SECOM 良率查詢｜製程良率異常分析 Agent 作品展示">
<meta property="og:description" content="{escape(desc)}">
<link rel="icon" href="data:,">
<style>{CSS}</style>
</head>
<body>
<main>
<header class="top"><b>SECOM 良率查詢</b><span>作品展示・唯讀</span></header>

<section class="hero" aria-labelledby="h-hero">
<h1 id="h-hero">{h1}</h1>
<p>這是一個¦良率查詢 Agent（讓 AI 照指定順序¦呼叫工具、自己¦查資料¦的程式），查的是 UCI 公開的 SECOM 半導體¦製程¦資料集。它透過 MCP（讓 AI 呼叫¦外部工具¦的通訊規則）連到¦資料庫。下面的¦數字¦取自¦資料庫、訓練報告¦與工具的¦實際回傳。</p>
<p class="links"><a href="{REPO_URL}">原始碼（GitHub）</a></p>
</section>

<section class="zone" aria-labelledby="h-yield">
<h2 id="h-yield">每週良率</h2>
<p class="lede">2008 年¦第 {week_no(w0)} 到 {week_no(w1)} 週，每個點¦是一週（週次¦從週一¦起算，不是 ISO 週次），滑鼠¦移上去¦看筆數。第 {RANGE_START_WEEK} 週¦以後¦最低的¦是第 {week_no(low['week'])} 週：{low['runs']} 筆中 {low['fails']} 筆¦不良，良率 {fmt_pct(low['yield_pct'])}。</p>
<p class="lede">這不是¦全期¦最低。第 {RANGE_START_WEEK} 週¦以前有 {data['earlier_lower_weeks']} 週¦更低，其中¦樣本¦最少的¦一週¦只有 {data['earlier_min_runs']} 筆。下面的¦重播¦查的是¦第 {r0} 到 {r1} 週。</p>
{chart(data)}
</section>

<section class="zone" aria-labelledby="h-sensors">
<h2 id="h-sensors">可疑感測欄前 10 名</h2>
<p class="lede">兩個¦模型¦各自¦排出¦前 10。表中的¦數字¦是¦重要度：把¦該欄的¦值¦打亂後，PR-AUC（看¦模型¦抓¦不良品¦抓得¦準不準¦的分數，亂猜時¦等於¦不良率）掉了¦多少，越大¦越可疑。</p>
<p class="lede">橘底¦加 <span class="mk">◆</span> 的是¦兩邊¦都進¦前 10 的 {len(data['both'])} 欄：{both}。排名¦只代表¦這些欄¦和不良¦一起¦變動，不代表¦原因。隨機森林¦有 {len(data['rf_noisy'])} 欄（{noisy}）的¦標準差¦不小於¦平均，這些¦排名¦不可靠。</p>
{sensor_table(data)}
</section>

<section class="zone" aria-labelledby="h-replay">
<h2 id="h-replay">Agent 查詢重播</h2>
<p class="lede">依序¦點開¦看請求¦與回應，內容是 <span class="mono">demo/mcp_session.md</span> 的¦原樣¦輸出（紀錄¦產生於 {sess_date}）。這段¦由腳本¦照 skill（寫給 AI 的¦作業¦說明檔，規定¦查詢¦步驟）的¦順序¦呼叫 MCP 伺服器，沒有¦語言模型¦參與；每步¦回應¦標的 isError 是¦工具¦回報¦這次¦有沒有¦出錯¦的欄位；呼叫¦順序¦與¦筆記¦內文¦都是¦事先¦寫死的。</p>
<p class="hint">第 1 步已展開。方框內的長行¦可以¦左右滑動¦看完整內容，右緣¦有陰影¦就代表¦還有。</p>
{replay_html(items)}
</section>

<section class="zone" aria-labelledby="h-limits">
<h2 id="h-limits">限制與誠實聲明</h2>
<p class="lede">先看¦結論：模型的¦效果¦只比¦「全猜良品」¦高一點，排名¦只是¦相關，不證明¦因果。</p>
<div class="cards">
<div class="card wide"><h3>模型¦只比¦「全猜良品」¦高一點</h3>
<p>測試期 {sp['test_rows']} 筆，其中 {sp['test_fail']} 筆¦不良。全猜良品的 PR-AUC 就是¦不良率 {mdl['baseline']['pr_auc']}，邏輯迴歸 {mdl['logreg']['pr_auc']}，隨機森林 {mdl['random_forest']['pr_auc']}。</p>
{models_table(data)}
<p class="after">隨機森林的 ROC-AUC 是 {mdl['random_forest']['roc_auc']}（0.5 等於¦亂猜），這個¦模型¦不能¦拿來¦當¦自動¦判退的¦閘門。驗證段¦只有 {sp['val_fail']} 筆¦不良，選模型¦這一步¦本身¦就不穩：照¦事先¦定好的¦規則¦選到¦隨機森林，但在¦測試期¦它比¦邏輯迴歸¦差。測試期¦只有 {sp['test_fail']} 筆¦不良，指標的¦誤差¦很大。</p></div>
<div class="card"><h3>相關¦不等於¦因果</h3>
<p>前 10 名¦只能¦當¦排查¦順序的¦參考，沒有¦驗證過¦比隨機¦挑欄¦好，兩個¦模型的¦前 10 名¦只重疊 {len(data['both'])} 欄。</p>
<p>感測欄¦是匿名的，沒有¦站別、機台、批號，所以¦回答不了¦「哪個站」，只給得出¦感測欄¦編號。時間戳¦是測試¦時間，不是¦各站的¦加工¦時間。</p></div>
<div class="card"><h3>筆記¦只能是¦草稿</h3>
<p>draft 閘門¦只在 MCP 工具層¦成立，前提是 Agent 只能¦透過¦這 {c['tools']} 個¦工具¦碰資料庫。Agent 若同時¦有 shell（能¦直接¦下指令¦的終端機）或¦檔案¦工具，就能¦直接¦改¦資料庫¦繞過¦閘門。</p>
<p>確認¦筆記的¦步驟¦沒有做，要由¦人¦到現場¦確認。</p></div>
</div>
</section>

<footer>
<p>資料：McCann &amp; Johnston (2008), SECOM, UCI Machine Learning Repository，<a href="{UCI_URL}">doi:10.24432/C54305</a>，授權 <a href="{LICENSE_URL}">CC BY 4.0</a>。本頁¦只含¦由這份¦資料¦算出的¦衍生¦結果。</p>
<p>頁面¦由 <span class="mono">scripts/build_demo_page.py</span> 從 <span class="mono">reports/</span>、資料庫¦與 <span class="mono">demo/mcp_session.md</span> 產生。<a href="{REPO_URL}">原始碼（GitHub）</a></p>
</footer>
</main>
<script type="application/json" id="demo-data">{blob}</script>
</body>
</html>
"""


def build() -> str:
    src = read_sources()
    calls = parse_session(src["session"])
    data = build_data(src, calls)
    # the break marker belongs to the templates only: a value from the sources must never contain it
    assert BREAK not in json.dumps(data, ensure_ascii=False) + src["session"] + json.dumps(src["metrics"], ensure_ascii=False), \
        "a source value contains the word-break marker"
    items = replay_items(calls, data["low_week"])
    return apply_wbr(render(data, items, session_date(src["session"])))


def main() -> None:
    html = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(html.encode('utf-8'))} bytes)")


if __name__ == "__main__":
    main()
