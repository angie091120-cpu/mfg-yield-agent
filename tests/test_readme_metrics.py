"""A3: the README metrics block is generated from reports/metrics.json, and nothing else quotes metrics."""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
METRICS = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))


def _renderer():
    spec = importlib.util.spec_from_file_location("render_readme_metrics", ROOT / "scripts" / "render_readme_metrics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_metrics_block_is_exactly_the_rendered_metrics():
    """Full-text match: a swapped, stale or hand-edited number in either file fails."""
    r = _renderer()
    assert r.current_block(README) == r.render(METRICS)


def test_rendered_block_puts_numbers_in_the_right_cells():
    """Guards the renderer itself: each model row carries that model's own test numbers, in column order."""
    block = _renderer().render(METRICS)
    rows = {line.split("|")[1].strip(): [c.strip() for c in line.split("|")[2:-1]]
            for line in block.splitlines() if line.startswith("| ") and "---" not in line}
    for label_prefix, name in (("邏輯迴歸", "logreg"), ("隨機森林", "random_forest")):
        cells = next(v for k, v in rows.items() if k.startswith(label_prefix))
        t = METRICS["models"][name]["test"]
        want = [t["pr_auc"], t["roc_auc"], t["threshold"], t["precision"], t["recall"],
                t["confusion"]["tp"], t["confusion"]["fp"]]
        assert cells == [str(x) for x in want], name
    base = rows["全猜 pass（基準）"]
    assert base[0] == str(METRICS["baseline_all_pass"]["pr_auc"])


def test_no_metric_decimals_outside_block():
    """Decimals like 0.0936 may only appear inside the metrics block."""
    outside = README.replace(_renderer().current_block(README), "")
    stray = re.findall(r"(?<![\w.])0\.\d{2,}(?![\w.])", outside)
    assert not stray, f"metric-looking numbers outside the metrics block: {stray}"


def test_readme_credits_secom_and_license():
    assert "10.24432/C54305" in README
    assert "CC BY 4.0" in README
