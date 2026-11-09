"""Render the README metrics block from reports/metrics.json.

The block sits between `<!-- metrics:start -->` and `<!-- metrics:end -->` in
README.md. tests/test_readme_metrics.py checks that the block in the README is
exactly what `render()` returns, so a hand edit or a stale number fails the tests.

Run:  python scripts/render_readme_metrics.py           # rewrite the block in README.md
      python scripts/render_readme_metrics.py --check   # exit 1 if README is out of date
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
METRICS = ROOT / "reports" / "metrics.json"
START, END = "<!-- metrics:start -->", "<!-- metrics:end -->"
BLOCK_RE = re.compile(re.escape(START) + r"\n(.*?)" + re.escape(END), re.S)

MODEL_LABEL = {"logreg": "邏輯迴歸", "random_forest": "隨機森林"}


def _row(label: str, t: dict) -> str:
    c = t["confusion"]
    return (f"| {label} | {t['pr_auc']} | {t['roc_auc']} | {t['threshold']} | "
            f"{t['precision']} | {t['recall']} | {c['tp']} | {c['fp']} |")


def _at_half(name: str, t: dict) -> str:
    if t["flagged"] == 0:
        return f"{MODEL_LABEL[name]}一筆都沒判 fail"
    return f"{MODEL_LABEL[name]}抓到 {t['confusion']['tp']} 筆 fail、誤報 {t['confusion']['fp']} 筆"


def render(m: dict) -> str:
    s, base, models = m["split"], m["baseline_all_pass"], m["models"]
    lr, rf = models["logreg"], models["random_forest"]
    bc = base["confusion"]
    lines = [
        f"測試期 {s['test_rows']} 筆，其中 fail {s['test_fail']} 筆（fail 率 {s['test_fail_rate']}）。",
        "",
        "| 模型 | PR-AUC | ROC-AUC | 門檻 | Precision | Recall | 抓到 fail | 誤報 |",
        "|---|---|---|---|---|---|---|---|",
        f"| 全猜 pass（基準） | {base['pr_auc']} | | | 無（沒有任何一筆被判 fail） | {base['recall']} "
        f"| {bc['tp']} | {bc['fp']} |",
        _row(f"邏輯迴歸（C={lr['params']['C']}）", lr["test"]),
        _row("隨機森林", rf["test"]),
        "",
        f"全猜 pass 的 accuracy 是 {base['accuracy']}。門檻改用 0.5 時，"
        f"{_at_half('logreg', lr['test_at_0.5'])}，{_at_half('random_forest', rf['test_at_0.5'])}。",
        "",
    ]
    below = [MODEL_LABEL[n] for n in ("logreg", "random_forest") if models[n]["val_pr_auc"] < s["val_fail_rate"]]
    val = (f"驗證段 {s['val_rows']} 筆，fail {s['val_fail']} 筆（fail 率 {s['val_fail_rate']}，"
           f"也就是全猜 pass 的 PR-AUC）。驗證段 PR-AUC：邏輯迴歸 {lr['val_pr_auc']}、隨機森林 {rf['val_pr_auc']}。")
    if below:
        val += f"{'、'.join(below)}在驗證段低於全猜 pass。"
    lines.append(val)
    return "\n".join(lines) + "\n"


def current_block(readme_text: str) -> str:
    match = BLOCK_RE.search(readme_text)
    if not match:
        raise ValueError("README has no metrics block")
    return match.group(1)


def main() -> int:
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    text = README.read_text(encoding="utf-8")
    want = render(metrics)
    if "--check" in sys.argv:
        if current_block(text) != want:
            print("README metrics block is out of date; run scripts/render_readme_metrics.py", file=sys.stderr)
            return 1
        return 0
    current_block(text)  # fail loudly if the markers are missing
    new = BLOCK_RE.sub(lambda _: f"{START}\n{want}{END}", text, count=1)
    README.write_text(new, encoding="utf-8")
    print("README metrics block updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
