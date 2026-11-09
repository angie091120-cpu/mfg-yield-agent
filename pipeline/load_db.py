"""Load the cleaned data into SQLite.

Reads data/processed/ (written by pipeline.clean) and rebuilds the database
from scratch: runs, features (long form), cleaning_log, and an empty notes
table. Model tables are created empty and filled later by model.train.

Run:  python -m pipeline.load_db
DB path: env MFG_DB, default data/secom.db
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
SCHEMA = ROOT / "schema.sql"


def db_path() -> Path:
    return Path(os.environ.get("MFG_DB", ROOT / "data" / "secom.db"))


def connect_rw(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def connect_ro(path: Path) -> sqlite3.Connection:
    """Read-only connection. Any write raises sqlite3.OperationalError."""
    if not Path(path).exists():
        raise FileNotFoundError(f"database not found: {path}")
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def build_db(clean: pd.DataFrame, mask: pd.DataFrame, meta: dict, path: Path) -> dict:
    """Create a fresh database at `path`. Returns row counts per table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    sensors = list(meta["kept"])
    if list(mask["id"]) != list(clean["id"]):
        raise ValueError("imputed mask and clean table are not aligned")

    conn = connect_rw(path)
    try:
        conn.executescript(SCHEMA.read_text(encoding="utf-8"))
        conn.executemany(
            "INSERT INTO runs (id, ts, result, split) VALUES (?, ?, ?, ?)",
            clean[["id", "ts", "result", "split"]].itertuples(index=False, name=None),
        )
        values = clean[["id"] + sensors].melt(id_vars="id", var_name="sensor", value_name="value")
        flags = mask[["id"] + sensors].melt(id_vars="id", var_name="sensor", value_name="imputed")
        long = values.merge(flags, on=["id", "sensor"], how="inner", validate="one_to_one")
        if len(long) != len(clean) * len(sensors):
            raise ValueError("long table size mismatch")
        long["imputed"] = long["imputed"].astype(bool).astype(int)
        conn.executemany(
            "INSERT INTO features (run_id, sensor, value, imputed) VALUES (?, ?, ?, ?)",
            ((int(r), s, float(v), int(m)) for r, s, v, m in long.itertuples(index=False, name=None)),
        )
        conn.executemany(
            "INSERT INTO cleaning_log (step_no, step, detail, rows, cols_before, cols_after)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            [(i + 1, e["step"], e["detail"], e["rows"], e["cols_before"], e["cols_after"])
             for i, e in enumerate(meta["log"])],
        )
        conn.commit()
        counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ("runs", "features", "cleaning_log", "notes")}
    finally:
        conn.close()
    return counts


def load_processed(processed: Path = PROCESSED):
    clean_path = processed / "clean.csv"
    if not clean_path.exists():
        raise FileNotFoundError("data/processed/clean.csv missing. Run: python -m pipeline.clean")
    clean = pd.read_csv(clean_path)
    mask = pd.read_csv(processed / "imputed_mask.csv")
    meta = json.loads((processed / "cleaning_meta.json").read_text(encoding="utf-8"))
    return clean, mask, meta


def main() -> None:
    clean, mask, meta = load_processed()
    counts = build_db(clean, mask, meta, db_path())
    print(f"db={db_path()}")
    for table, n in counts.items():
        print(f"{table}: {n}")


if __name__ == "__main__":
    main()
