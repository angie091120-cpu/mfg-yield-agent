"""Baseline models for SECOM pass/fail.

Split: the time split from pipeline.clean (first 80% of runs by timestamp
= train, last 20% = test). The test period is used once, for the final
numbers below; everything that needs tuning uses a validation slice.

Validation slice: the last 20% of the training period (also by time).
Used for (a) the logistic-regression C grid, (b) the decision threshold
(max F1), (c) choosing which model's permutation importance becomes the
"suspect sensor" list. After that both models are refitted on the full
training period.

Outputs:
    reports/metrics.json      every number quoted in the README
    tables scores, contributions, suspects in the SQLite DB

Run:  python -m model.train
"""
from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.load_db import PROCESSED, connect_rw, db_path

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "reports" / "metrics.json"
SEED = 0
VAL_FRACTION = 0.2
C_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0]
N_REPEATS = 20
TOP_SUSPECTS = 10
TOP_CONTRIB = 5


def make_logreg(c: float) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=c, class_weight="balanced", max_iter=20000, random_state=SEED)),
    ])


def make_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=SEED,
    )


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = np.where(prec[:-1] + rec[:-1] > 0, 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12), 0)
    return float(thr[int(np.argmax(f1))])


def evaluate(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(threshold, 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        "flagged": int(pred.sum()),
    }


def all_pass_baseline(y: np.ndarray) -> dict:
    p = np.zeros(len(y))
    pred = np.zeros(len(y), dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "description": "predict pass for every run",
        "accuracy": round(float((pred == y).mean()), 4),
        "precision": None,  # no run is flagged, so precision is undefined
        "recall": 0.0,
        "pr_auc": round(float(average_precision_score(y, p)), 4),  # equals the fail rate
        "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
    }


def main() -> None:
    clean = pd.read_csv(PROCESSED / "clean.csv")
    meta = json.loads((PROCESSED / "cleaning_meta.json").read_text(encoding="utf-8"))
    sensors = meta["kept"]

    train = clean[clean["split"] == "train"].reset_index(drop=True)
    test = clean[clean["split"] == "test"].reset_index(drop=True)
    X_tr, y_tr = train[sensors].to_numpy(), (train["result"] == "fail").to_numpy().astype(int)
    X_te, y_te = test[sensors].to_numpy(), (test["result"] == "fail").to_numpy().astype(int)

    # ---- validation slice inside the training period -------------------
    n_fit = int(math.floor(len(train) * (1 - VAL_FRACTION)))
    X_fit, y_fit, X_val, y_val = X_tr[:n_fit], y_tr[:n_fit], X_tr[n_fit:], y_tr[n_fit:]

    grid = []
    for c in C_GRID:
        p = make_logreg(c).fit(X_fit, y_fit).predict_proba(X_val)[:, 1]
        grid.append({"C": c, "val_pr_auc": round(float(average_precision_score(y_val, p)), 4)})
    best_c = max(grid, key=lambda g: (g["val_pr_auc"], -g["C"]))["C"]

    lr_val = make_logreg(best_c).fit(X_fit, y_fit).predict_proba(X_val)[:, 1]
    rf_val = make_forest().fit(X_fit, y_fit).predict_proba(X_val)[:, 1]
    thr = {"logreg": best_f1_threshold(y_val, lr_val), "random_forest": best_f1_threshold(y_val, rf_val)}
    val_pr = {
        "logreg": round(float(average_precision_score(y_val, lr_val)), 4),
        "random_forest": round(float(average_precision_score(y_val, rf_val)), 4),
    }
    selected = max(val_pr, key=val_pr.get)

    # ---- refit on full training period, evaluate once on test ----------
    lr = make_logreg(best_c).fit(X_tr, y_tr)
    rf = make_forest().fit(X_tr, y_tr)
    p_lr, p_rf = lr.predict_proba(X_te)[:, 1], rf.predict_proba(X_te)[:, 1]

    models = {
        "logreg": {
            "params": {"C": best_c, "class_weight": "balanced", "scaler": "StandardScaler (fit on train)"},
            "test": evaluate(y_te, p_lr, thr["logreg"]),
            "test_at_0.5": evaluate(y_te, p_lr, 0.5),
            "val_pr_auc": val_pr["logreg"],
        },
        "random_forest": {
            "params": {"n_estimators": 500, "min_samples_leaf": 3, "class_weight": "balanced_subsample"},
            "test": evaluate(y_te, p_rf, thr["random_forest"]),
            "test_at_0.5": evaluate(y_te, p_rf, 0.5),
            "val_pr_auc": val_pr["random_forest"],
        },
    }

    # ---- permutation importance on the test period ---------------------
    suspects = {}
    for name, est in (("logreg", lr), ("random_forest", rf)):
        if name == "random_forest":
            est.set_params(n_jobs=1)  # parallelism comes from permutation_importance; avoid oversubscription
        imp = permutation_importance(
            est, X_te, y_te, scoring="average_precision", n_repeats=N_REPEATS,
            random_state=SEED, n_jobs=-1,
        )
        order = np.argsort(-imp.importances_mean, kind="stable")[:TOP_SUSPECTS]
        suspects[name] = [
            {"rank": i + 1, "sensor": sensors[j],
             "importance_mean": round(float(imp.importances_mean[j]), 4),
             "importance_std": round(float(imp.importances_std[j]), 4)}
            for i, j in enumerate(order)
        ]

    # ---- per-run scores and linear contributions for all runs ----------
    X_all = clean[sensors].to_numpy()
    ids = clean["id"].to_numpy()
    risk_lr = lr.predict_proba(X_all)[:, 1]
    risk_rf = rf.predict_proba(X_all)[:, 1]
    z = lr.named_steps["scale"].transform(X_all)
    coef = lr.named_steps["clf"].coef_[0]
    contrib = z * coef

    conn = connect_rw(db_path())
    try:
        conn.execute("DELETE FROM scores")
        conn.execute("DELETE FROM contributions")
        conn.execute("DELETE FROM suspects")
        conn.executemany(
            "INSERT INTO scores (run_id, model, risk) VALUES (?, ?, ?)",
            [(int(i), "logreg", float(r)) for i, r in zip(ids, risk_lr)]
            + [(int(i), "random_forest", float(r)) for i, r in zip(ids, risk_rf)],
        )
        rows = []
        for k, run_id in enumerate(ids):
            top = np.argsort(-np.abs(contrib[k]), kind="stable")[:TOP_CONTRIB]
            for rank, j in enumerate(top, start=1):
                rows.append((int(run_id), rank, sensors[j], float(X_all[k, j]), float(z[k, j]), float(contrib[k, j])))
        conn.executemany(
            "INSERT INTO contributions (run_id, rank, sensor, value, z, contribution) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.executemany(
            "INSERT INTO suspects (model, rank, sensor, importance_mean, importance_std) VALUES (?, ?, ?, ?, ?)",
            [(m, s["rank"], s["sensor"], s["importance_mean"], s["importance_std"])
             for m, lst in suspects.items() for s in lst],
        )
        conn.execute("DELETE FROM model_meta")  # table defined in schema.sql
        conn.executemany("INSERT INTO model_meta VALUES (?, ?)", [
            ("selected_model", selected),
            ("threshold_logreg", str(thr["logreg"])),
            ("threshold_random_forest", str(thr["random_forest"])),
            ("logreg_intercept", str(float(lr.named_steps["clf"].intercept_[0]))),
        ])
        conn.commit()
    finally:
        conn.close()

    metrics = {
        "dataset": "UCI SECOM (CC BY 4.0)",
        "split": {
            "method": "time-ordered; first 80% train, last 20% test",
            "train_rows": int(len(train)), "test_rows": int(len(test)),
            "train_fail": int(y_tr.sum()), "test_fail": int(y_te.sum()),
            "train_fail_rate": round(float(y_tr.mean()), 4), "test_fail_rate": round(float(y_te.mean()), 4),
            "validation": "last 20% of the training period, used for C, threshold and model choice",
            "val_rows": int(len(y_val)), "val_fail": int(y_val.sum()),
            "val_fail_rate": round(float(y_val.mean()), 4),
        },
        "features": len(sensors),
        "logreg_c_grid": grid,
        "selected_model": selected,
        "selection_rule": "higher PR-AUC on the validation slice (test period not used)",
        "baseline_all_pass": all_pass_baseline(y_te),
        "models": models,
        "permutation_importance": {
            "scoring": "average_precision (PR-AUC) on the test period",
            "n_repeats": N_REPEATS,
            "top10": suspects,
        },
    }
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("split", "selected_model", "baseline_all_pass")}, indent=2))
    for name, m in models.items():
        print(name, "val_pr_auc", m["val_pr_auc"], "test", m["test"])


if __name__ == "__main__":
    main()
