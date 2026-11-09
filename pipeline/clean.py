"""Cleaning pipeline for UCI SECOM.

Steps (every decision is made on the training slice only, so nothing from
the test period leaks into preprocessing):

1. Load the 590 sensor columns and the pass/fail label with its timestamp.
2. Sort runs by timestamp (stable) and mark the first 80% as train.
3. Missing-rate report per column.
4. Drop columns whose missing rate on the training slice is above 50%.
5. Drop columns that are constant on the training slice (one distinct
   non-missing value, or none at all).
6. Median imputation, fitted on the training slice only.

Outputs:
    data/processed/clean.csv          wide table, one row per run
    data/processed/cleaning_meta.json kept/dropped columns, medians, log
    reports/cleaning.md               human-readable report

Run:  python -m pipeline.clean
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
REPORT = ROOT / "reports" / "cleaning.md"

TRAIN_FRACTION = 0.8
MISSING_THRESHOLD = 0.5  # drop if missing rate on train is strictly above this
N_SENSORS = 590


def sensor_name(i: int) -> str:
    return f"s{i:03d}"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Return one row per run: id, ts, result, s000..s589 (NaN = missing).

    `id` is the 1-based row number in the original UCI file, so a run can
    always be traced back to the source.
    """
    data_path = raw_dir / "secom.data"
    label_path = raw_dir / "secom_labels.data"
    if not data_path.exists() or not label_path.exists():
        raise FileNotFoundError(
            f"SECOM files not found in {raw_dir}. Run: python scripts/download_data.py"
        )
    sensors = pd.read_csv(data_path, sep=r"\s+", header=None, na_values=["NaN"], engine="python")
    if sensors.shape[1] != N_SENSORS:
        raise ValueError(f"expected {N_SENSORS} sensor columns, got {sensors.shape[1]}")
    sensors.columns = [sensor_name(i) for i in range(N_SENSORS)]

    labels = pd.read_csv(label_path, sep=" ", header=None, names=["label", "ts"], quotechar='"')
    if len(labels) != len(sensors):
        raise ValueError(f"row count mismatch: {len(sensors)} sensor rows vs {len(labels)} labels")
    bad = set(labels["label"].unique()) - {-1, 1}
    if bad:
        raise ValueError(f"unexpected label codes: {sorted(bad)}")
    ts = pd.to_datetime(labels["ts"], format="%d/%m/%Y %H:%M:%S", errors="raise")

    out = pd.DataFrame(
        {
            "id": np.arange(1, len(labels) + 1),
            "ts": ts,
            "result": np.where(labels["label"] == 1, "fail", "pass"),
        }
    )
    return pd.concat([out, sensors], axis=1)


def time_split(df: pd.DataFrame, train_fraction: float = TRAIN_FRACTION) -> pd.DataFrame:
    """Sort by timestamp (ties keep file order) and add a `split` column."""
    df = df.sort_values(["ts", "id"], kind="mergesort").reset_index(drop=True).copy()
    n_train = int(math.floor(len(df) * train_fraction))
    split = np.array(["test"] * len(df), dtype=object)
    split[:n_train] = "train"
    df.insert(3, "split", split)
    return df


# --------------------------------------------------------------------------
# Column selection and imputation (fit on train only)
# --------------------------------------------------------------------------
def missing_rates(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
    return frame[cols].isna().mean()


def select_columns(train: pd.DataFrame, cols: list[str], threshold: float = MISSING_THRESHOLD):
    """Return (kept, dropped_missing, dropped_constant) decided on `train`."""
    rates = missing_rates(train, cols)
    dropped_missing = [c for c in cols if rates[c] > threshold]
    remaining = [c for c in cols if c not in set(dropped_missing)]
    nunique = train[remaining].nunique(dropna=True)
    dropped_constant = [c for c in remaining if nunique[c] <= 1]
    kept = [c for c in remaining if nunique[c] > 1]
    return kept, dropped_missing, dropped_constant


@dataclass
class MedianImputer:
    """Median imputer that refuses to transform before it is fitted."""

    medians: dict[str, float] = field(default_factory=dict)
    fitted_rows: int = 0

    def fit(self, train: pd.DataFrame, cols: list[str]) -> "MedianImputer":
        med = train[cols].median(skipna=True)
        if med.isna().any():
            empty = med[med.isna()].index.tolist()
            raise ValueError(f"columns with no observed value in train: {empty[:5]}")
        self.medians = {c: float(med[c]) for c in cols}
        self.fitted_rows = len(train)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.medians:
            raise RuntimeError("MedianImputer.transform called before fit")
        cols = list(self.medians)
        out = frame.copy()
        out[cols] = out[cols].fillna(value=self.medians)
        return out


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
@dataclass
class CleanResult:
    clean: pd.DataFrame          # id, ts, result, split, kept sensors (imputed)
    imputed_mask: pd.DataFrame   # id + kept sensors, True where the value was imputed
    meta: dict


def run_cleaning(raw: pd.DataFrame) -> CleanResult:
    df = time_split(raw)
    sensors = [c for c in df.columns if c.startswith("s") and c[1:].isdigit()]
    train = df[df["split"] == "train"]

    rates_all = missing_rates(df, sensors)
    rates_train = missing_rates(train, sensors)
    kept, dropped_missing, dropped_constant = select_columns(train, sensors)

    imputer = MedianImputer().fit(train, kept)
    mask = df[["id"] + kept].copy()
    mask[kept] = df[kept].isna()
    clean = imputer.transform(df[["id", "ts", "result", "split"] + kept])

    if clean[kept].isna().any().any():
        raise AssertionError("NaN left after imputation")

    n = len(df)
    n_train = int((df["split"] == "train").sum())
    n_fail = int((df["result"] == "fail").sum())
    cells_total = n * len(kept)
    cells_imputed = int(mask[kept].to_numpy().sum())

    log = [
        {"step": "load", "detail": "read secom.data + secom_labels.data",
         "rows": n, "cols_before": N_SENSORS, "cols_after": N_SENSORS},
        {"step": "time_split", "detail": f"sort by ts, first {int(TRAIN_FRACTION*100)}% = train",
         "rows": n, "cols_before": N_SENSORS, "cols_after": N_SENSORS},
        {"step": "drop_missing", "detail": f"missing rate on train > {MISSING_THRESHOLD:.0%}",
         "rows": n, "cols_before": N_SENSORS, "cols_after": N_SENSORS - len(dropped_missing)},
        {"step": "drop_constant", "detail": "<=1 distinct non-missing value on train",
         "rows": n, "cols_before": N_SENSORS - len(dropped_missing), "cols_after": len(kept)},
        {"step": "impute_median", "detail": f"median fitted on {n_train} train rows; {cells_imputed} cells filled",
         "rows": n, "cols_before": len(kept), "cols_after": len(kept)},
    ]

    meta = {
        "rows": n,
        "rows_train": n_train,
        "rows_test": n - n_train,
        "fail_count": n_fail,
        "fail_rate": n_fail / n,
        "fail_rate_train": float((train["result"] == "fail").mean()),
        "fail_rate_test": float((df.loc[df["split"] == "test", "result"] == "fail").mean()),
        "ts_min": df["ts"].min().isoformat(),
        "ts_max": df["ts"].max().isoformat(),
        "train_ts_max": train["ts"].max().isoformat(),
        "test_ts_min": df.loc[df["split"] == "test", "ts"].min().isoformat(),
        "cols_raw": N_SENSORS,
        "cols_dropped_missing": len(dropped_missing),
        "cols_dropped_constant": len(dropped_constant),
        "cols_kept": len(kept),
        "missing_cells_raw": int(df[sensors].isna().to_numpy().sum()),
        "missing_rate_raw": float(df[sensors].isna().to_numpy().mean()),
        "cols_with_any_missing_raw": int((rates_all > 0).sum()),
        "cells_imputed": cells_imputed,
        "cells_imputed_rate": cells_imputed / cells_total,
        "missing_threshold": MISSING_THRESHOLD,
        "train_fraction": TRAIN_FRACTION,
        "dropped_missing": dropped_missing,
        "dropped_constant": dropped_constant,
        "kept": kept,
        "medians": imputer.medians,
        "missing_rate_train": {c: float(rates_train[c]) for c in sensors},
        "log": log,
    }
    return CleanResult(clean=clean, imputed_mask=mask, meta=meta)


def render_report(meta: dict) -> str:
    rates = pd.Series(meta["missing_rate_train"])
    buckets = [
        ("0%", (rates == 0).sum()),
        ("0–5%", ((rates > 0) & (rates <= 0.05)).sum()),
        ("5–50%", ((rates > 0.05) & (rates <= 0.5)).sum()),
        (">50%（刪除）", (rates > 0.5).sum()),
    ]
    top = rates.sort_values(ascending=False).head(10)
    lines = [
        "# 清洗報告（SECOM）",
        "",
        "由 `python -m pipeline.clean` 產生，請勿手改。",
        "",
        "## 前後對照",
        "",
        "| 項目 | 清洗前 | 清洗後 |",
        "|---|---|---|",
        f"| 筆數（runs） | {meta['rows']} | {meta['rows']} |",
        f"| 感測欄數 | {meta['cols_raw']} | {meta['cols_kept']} |",
        f"| 缺值格數 | {meta['missing_cells_raw']} | 0 |",
        "",
        "筆數不變：這一步只刪欄、不刪列。",
        "",
        "## 切分",
        "",
        f"- 依時間排序後，前 {meta['rows_train']} 筆為訓練集、後 {meta['rows_test']} 筆為測試集。",
        f"- 訓練集最後一筆 {meta['train_ts_max']}，測試集第一筆 {meta['test_ts_min']}。",
        f"- fail 比例：全體 {meta['fail_rate']:.2%}（{meta['fail_count']} 筆）、"
        f"訓練集 {meta['fail_rate_train']:.2%}、測試集 {meta['fail_rate_test']:.2%}。",
        "",
        "## 刪欄",
        "",
        "刪欄與補值的決定都只看訓練集。",
        "",
        "| 步驟 | 刪掉欄數 | 剩下欄數 |",
        "|---|---|---|",
        f"| 原始 | | {meta['cols_raw']} |",
        f"| 缺值率 > {meta['missing_threshold']:.0%} | {meta['cols_dropped_missing']} | "
        f"{meta['cols_raw'] - meta['cols_dropped_missing']} |",
        f"| 常數欄 | {meta['cols_dropped_constant']} | {meta['cols_kept']} |",
        "",
        "## 缺值率分布（訓練集，590 欄）",
        "",
        "| 缺值率 | 欄數 |",
        "|---|---|",
        *[f"| {name} | {int(cnt)} |" for name, cnt in buckets],
        "",
        "缺值率最高的 10 欄：",
        "",
        "| 欄 | 缺值率 |",
        "|---|---|",
        *[f"| {c} | {v:.1%} |" for c, v in top.items()],
        "",
        "## 補值",
        "",
        f"- 以訓練集 {meta['rows_train']} 筆算出每欄中位數，套用到全部 {meta['rows']} 筆。",
        f"- 補了 {meta['cells_imputed']} 格，占保留欄所有格的 {meta['cells_imputed_rate']:.2%}。",
        "- 每一格是否為補值記在資料庫 `features.imputed`，分組統計只用實測值。",
        "",
    ]
    return "\n".join(lines)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_outputs(result: CleanResult, out_dir: Path = OUT_DIR, report: Path = REPORT) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    clean = result.clean.copy()
    clean["ts"] = clean["ts"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    clean.to_csv(out_dir / "clean.csv", index=False, float_format="%.10g", lineterminator="\n")
    result.imputed_mask.to_csv(out_dir / "imputed_mask.csv", index=False, lineterminator="\n")
    (out_dir / "cleaning_meta.json").write_text(
        json.dumps(result.meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report.write_text(render_report(result.meta), encoding="utf-8")
    return {
        "clean.csv": file_sha256(out_dir / "clean.csv"),
        "imputed_mask.csv": file_sha256(out_dir / "imputed_mask.csv"),
        "cleaning_meta.json": file_sha256(out_dir / "cleaning_meta.json"),
        "cleaning.md": file_sha256(report),
    }


def main() -> None:
    result = run_cleaning(load_raw())
    hashes = write_outputs(result)
    m = result.meta
    print(f"rows={m['rows']} train={m['rows_train']} test={m['rows_test']} fail_rate={m['fail_rate']:.4f}")
    print(f"cols raw={m['cols_raw']} -missing={m['cols_dropped_missing']} "
          f"-constant={m['cols_dropped_constant']} kept={m['cols_kept']}")
    for name, digest in hashes.items():
        print(f"sha256 {name} {digest}")


if __name__ == "__main__":
    main()
