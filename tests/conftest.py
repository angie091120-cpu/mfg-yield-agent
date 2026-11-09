"""Shared fixtures: a small synthetic SECOM-shaped dataset and a database built from it.

The synthetic data has the same file format as the UCI files (590 space-
separated columns, labels with quoted timestamps), so tests exercise the
real parser and pipeline without needing the download.
"""
from __future__ import annotations

import importlib.util
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from pipeline.clean import N_SENSORS, run_cleaning, load_raw
from pipeline.load_db import build_db

ROOT = Path(__file__).resolve().parent.parent
N_ROWS = 60  # 48 train / 12 test

# Column roles in the synthetic data
COL_MISSING_TRAIN = 0     # 70% missing in train -> dropped
COL_MISSING_TEST = 1      # complete in train, all missing in test -> kept (decision uses train)
COL_CONSTANT = 2          # constant -> dropped
COL_TEST_SHIFT = 3        # some train NaN; test values are huge -> must not move the median
COL_ALL_NAN = 4           # never observed -> dropped as >50% missing


def write_synthetic_raw(raw_dir: Path, test_value: float = 1000.0, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    X = rng.normal(10.0, 2.0, size=(N_ROWS, N_SENSORS))
    n_train = int(N_ROWS * 0.8)
    X[: int(n_train * 0.7), COL_MISSING_TRAIN] = np.nan
    X[n_train:, COL_MISSING_TEST] = np.nan
    X[:, COL_CONSTANT] = 5.0
    X[:5, COL_TEST_SHIFT] = np.nan
    X[n_train:, COL_TEST_SHIFT] = test_value
    X[:, COL_ALL_NAN] = np.nan
    X[n_train + 1, COL_TEST_SHIFT] = np.nan  # a missing cell in test, filled with the train median

    labels = np.where(np.arange(N_ROWS) % 7 == 3, 1, -1)
    start = datetime(2008, 7, 19, 8, 0, 0)
    ts = [start + timedelta(hours=12 * i) for i in range(N_ROWS)]
    # Shuffle file order so the pipeline has to sort by time itself.
    order = rng.permutation(N_ROWS)
    raw_dir.mkdir(parents=True, exist_ok=True)
    with open(raw_dir / "secom.data", "w", encoding="ascii") as fh:
        for i in order:
            fh.write(" ".join("NaN" if np.isnan(v) else f"{v:.4f}" for v in X[i]) + "\n")
    with open(raw_dir / "secom_labels.data", "w", encoding="ascii") as fh:
        for i in order:
            fh.write(f'{labels[i]} "{ts[i]:%d/%m/%Y %H:%M:%S}"\n')


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    d = tmp_path / "raw"
    write_synthetic_raw(d)
    return d


@pytest.fixture
def cleaned(raw_dir: Path):
    return run_cleaning(load_raw(raw_dir))


@pytest.fixture
def db(tmp_path: Path, cleaned, monkeypatch) -> Path:
    """Synthetic DB with model tables filled with fixed values."""
    path = tmp_path / "test.db"
    clean = cleaned.clean.copy()
    clean["ts"] = clean["ts"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    build_db(clean, cleaned.imputed_mask, cleaned.meta, path)
    sensors = cleaned.meta["kept"]
    conn = sqlite3.connect(path)
    conn.executemany("INSERT INTO model_meta VALUES (?, ?)", [
        ("selected_model", "logreg"), ("threshold_logreg", "0.5"),
        ("threshold_random_forest", "0.3"), ("logreg_intercept", "0.0"),
    ])
    for run_id in clean["id"]:
        conn.execute("INSERT INTO scores VALUES (?, 'logreg', ?)", (int(run_id), 0.6 if run_id % 2 else 0.1))
        conn.execute("INSERT INTO scores VALUES (?, 'random_forest', 0.2)", (int(run_id),))
        conn.execute("INSERT INTO contributions VALUES (?, 1, ?, 1.0, 2.0, 0.8)", (int(run_id), sensors[0]))
    for rank, s in enumerate(sensors[:10], start=1):
        conn.execute("INSERT INTO suspects VALUES ('logreg', ?, ?, ?, 0.01)", (rank, s, 0.1 / rank))
    conn.commit()
    conn.close()
    monkeypatch.setenv("MFG_DB", str(path))
    return path


@pytest.fixture(scope="session")
def server_module():
    """Import mcp/server.py by path (the folder name `mcp` would shadow the SDK as a package)."""
    spec = importlib.util.spec_from_file_location("yield_mcp_server", ROOT / "mcp" / "server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
