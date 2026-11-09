-- SQLite schema for the SECOM yield database.
-- Built by `python -m pipeline.load_db`; model tables are filled by `python -m model.train`.

PRAGMA foreign_keys = ON;

-- One row per production run (one row of the UCI file).
CREATE TABLE IF NOT EXISTS runs (
    id      INTEGER PRIMARY KEY,                       -- 1-based row number in secom.data
    ts      TEXT    NOT NULL,                          -- ISO 8601, local time as given by UCI
    result  TEXT    NOT NULL CHECK (result IN ('pass', 'fail')),
    split   TEXT    NOT NULL CHECK (split IN ('train', 'test'))
);
CREATE INDEX IF NOT EXISTS idx_runs_ts ON runs(ts);

-- Cleaned sensor values in long form: one row per (run, sensor).
-- Long form keeps the sensor name as a bound value, never as an SQL identifier.
CREATE TABLE IF NOT EXISTS features (
    run_id   INTEGER NOT NULL REFERENCES runs(id),
    sensor   TEXT    NOT NULL,                         -- s000..s589 (anonymous in the source)
    value    REAL    NOT NULL,                         -- after median imputation
    imputed  INTEGER NOT NULL CHECK (imputed IN (0, 1)),  -- 1 = value was missing and filled
    PRIMARY KEY (run_id, sensor)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_features_sensor ON features(sensor);

-- What the cleaning pipeline did, in order.
CREATE TABLE IF NOT EXISTS cleaning_log (
    step_no      INTEGER PRIMARY KEY,
    step         TEXT    NOT NULL,
    detail       TEXT    NOT NULL,
    rows         INTEGER NOT NULL,
    cols_before  INTEGER NOT NULL,
    cols_after   INTEGER NOT NULL
);

-- Investigation notes. Only drafts can be stored (see the triggers below);
-- confirming a finding is a person's job and has no path in this PoC.
CREATE TABLE IF NOT EXISTS notes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    sensor      TEXT    NOT NULL,
    text        TEXT    NOT NULL,
    status      TEXT    NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'confirmed')),
    created_at  TEXT    NOT NULL,
    created_by  TEXT    NOT NULL
);

-- Second line behind the MCP gate: a plain connection cannot write a note with
-- any status other than 'draft', either by UPDATE or by INSERT. 'confirmed' stays
-- in the CHECK list for a future review step. Anyone who can edit the database
-- file can still drop these triggers, so they are not a guarantee.
CREATE TRIGGER IF NOT EXISTS notes_status_update_draft_only
BEFORE UPDATE OF status ON notes
WHEN NEW.status <> 'draft'
BEGIN
    SELECT RAISE(ABORT, 'notes.status can only be draft; confirming needs a review step outside this repo');
END;

CREATE TRIGGER IF NOT EXISTS notes_status_insert_draft_only
BEFORE INSERT ON notes
WHEN NEW.status <> 'draft'
BEGIN
    SELECT RAISE(ABORT, 'notes.status can only be draft; confirming needs a review step outside this repo');
END;

-- Model outputs (written by model/train.py) so the MCP server never loads a pickle.
CREATE TABLE IF NOT EXISTS scores (
    run_id  INTEGER NOT NULL REFERENCES runs(id),
    model   TEXT    NOT NULL,                          -- 'logreg' | 'random_forest'
    risk    REAL    NOT NULL,                          -- predicted probability of fail
    PRIMARY KEY (run_id, model)
);

-- Per-run linear contributions from the logistic regression (coef * standardized value).
CREATE TABLE IF NOT EXISTS contributions (
    run_id        INTEGER NOT NULL REFERENCES runs(id),
    rank          INTEGER NOT NULL,
    sensor        TEXT    NOT NULL,
    value         REAL    NOT NULL,
    z             REAL    NOT NULL,
    contribution  REAL    NOT NULL,                    -- in log-odds units
    PRIMARY KEY (run_id, rank)
);

-- Settings the MCP server needs at query time (selected model, thresholds).
CREATE TABLE IF NOT EXISTS model_meta (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);

-- Permutation importance on the test period.
CREATE TABLE IF NOT EXISTS suspects (
    model            TEXT    NOT NULL,
    rank             INTEGER NOT NULL,
    sensor           TEXT    NOT NULL,
    importance_mean  REAL    NOT NULL,                 -- drop in PR-AUC when the column is shuffled
    importance_std   REAL    NOT NULL,
    PRIMARY KEY (model, rank)
);
