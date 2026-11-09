from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from pipeline.clean import (
    MedianImputer,
    load_raw,
    render_report,
    run_cleaning,
    sensor_name,
    write_outputs,
)
from tests.conftest import (
    COL_ALL_NAN,
    COL_CONSTANT,
    COL_MISSING_TEST,
    COL_MISSING_TRAIN,
    COL_TEST_SHIFT,
    N_ROWS,
    write_synthetic_raw,
)

ROOT = Path(__file__).resolve().parent.parent


def test_load_raw_parses_labels_timestamps_and_nan(raw_dir):
    df = load_raw(raw_dir)
    assert len(df) == N_ROWS
    assert set(df["result"]) == {"pass", "fail"}
    assert str(df["ts"].dtype).startswith("datetime64")
    assert df[sensor_name(COL_ALL_NAN)].isna().all()


def test_load_raw_rejects_unknown_label_code(tmp_path):
    d = tmp_path / "raw"
    write_synthetic_raw(d)
    lines = (d / "secom_labels.data").read_text().splitlines()
    lines[0] = '0 ' + lines[0].split(" ", 1)[1]
    (d / "secom_labels.data").write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="unexpected label"):
        load_raw(d)


def test_time_split_train_strictly_before_test(cleaned):
    c = cleaned.clean
    assert c.loc[c["split"] == "train", "ts"].max() < c.loc[c["split"] == "test", "ts"].min()
    assert (c["split"] == "train").sum() == int(N_ROWS * 0.8)


def test_imputer_medians_come_from_train_only(cleaned):
    """Leakage guard: the fill value equals the train median, not the all-rows median."""
    col = sensor_name(COL_TEST_SHIFT)
    c, mask = cleaned.clean, cleaned.imputed_mask
    train_observed = c.loc[(c["split"] == "train") & ~mask[col].to_numpy(), col]
    assert cleaned.meta["medians"][col] == pytest.approx(train_observed.median())
    # Test rows are 1000.0, so an all-rows median would be pulled far upward.
    assert cleaned.meta["medians"][col] < 100
    filled_test = c.loc[(c["split"] == "test") & mask[col].to_numpy(), col]
    assert len(filled_test) == 1
    assert filled_test.iloc[0] == pytest.approx(cleaned.meta["medians"][col])


def test_changing_test_values_does_not_change_fit(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_synthetic_raw(a, test_value=1000.0)
    write_synthetic_raw(b, test_value=-5000.0)
    ma, mb = run_cleaning(load_raw(a)).meta, run_cleaning(load_raw(b)).meta
    assert ma["medians"] == mb["medians"]
    assert ma["kept"] == mb["kept"]


def test_imputer_refuses_transform_before_fit():
    with pytest.raises(RuntimeError):
        MedianImputer().transform(pd.DataFrame({"s000": [1.0, None]}))


def test_drop_decisions_use_train_only(cleaned):
    meta = cleaned.meta
    assert sensor_name(COL_MISSING_TRAIN) in meta["dropped_missing"]
    assert sensor_name(COL_ALL_NAN) in meta["dropped_missing"]
    # Missing in every test row but complete in train: kept, because only train decides.
    assert sensor_name(COL_MISSING_TEST) in meta["kept"]
    assert sensor_name(COL_CONSTANT) in meta["dropped_constant"]
    assert meta["cols_kept"] == 590 - meta["cols_dropped_missing"] - meta["cols_dropped_constant"]


def test_no_missing_values_after_cleaning(cleaned):
    assert not cleaned.clean[cleaned.meta["kept"]].isna().any().any()
    assert len(cleaned.clean) == N_ROWS  # rows are never dropped


def test_rerun_gives_identical_files(raw_dir, tmp_path):
    h1 = write_outputs(run_cleaning(load_raw(raw_dir)), tmp_path / "o1", tmp_path / "r1.md")
    h2 = write_outputs(run_cleaning(load_raw(raw_dir)), tmp_path / "o2", tmp_path / "r2.md")
    assert h1 == h2


def _report_counts(text: str) -> dict[str, int]:
    before_after = re.search(r"\| 感測欄數 \| (\d+) \| (\d+) \|", text)
    rows = re.search(r"\| 筆數（runs） \| (\d+) \| (\d+) \|", text)
    dropped_missing = re.search(r"\| 缺值率 > 50% \| (\d+) \|", text)
    dropped_constant = re.search(r"\| 常數欄 \| (\d+) \|", text)
    return {
        "cols_raw": int(before_after.group(1)), "cols_kept": int(before_after.group(2)),
        "rows": int(rows.group(2)),
        "cols_dropped_missing": int(dropped_missing.group(1)),
        "cols_dropped_constant": int(dropped_constant.group(1)),
    }


def test_report_counts_match_meta(cleaned):
    counts = _report_counts(render_report(cleaned.meta))
    for key, value in counts.items():
        assert value == cleaned.meta[key], key


def test_committed_cleaning_report_matches_real_data():
    """A1: the committed reports/cleaning.md agrees with counts recomputed from the raw UCI files."""
    raw = ROOT / "data" / "raw"
    if not (raw / "secom.data").exists():
        pytest.skip("UCI files not downloaded; run scripts/download_data.py")
    df = load_raw(raw)
    sensors = [c for c in df.columns if c.startswith("s")]
    df = df.sort_values(["ts", "id"], kind="mergesort").reset_index(drop=True)
    train = df.iloc[: int(len(df) * 0.8)]
    missing = train[sensors].isna().mean()
    dropped_missing = int((missing > 0.5).sum())
    rest = [c for c in sensors if missing[c] <= 0.5]
    constant = int((train[rest].nunique(dropna=True) <= 1).sum())
    counts = _report_counts((ROOT / "reports" / "cleaning.md").read_text(encoding="utf-8"))
    assert counts == {
        "cols_raw": 590, "cols_kept": 590 - dropped_missing - constant, "rows": len(df),
        "cols_dropped_missing": dropped_missing, "cols_dropped_constant": constant,
    }
