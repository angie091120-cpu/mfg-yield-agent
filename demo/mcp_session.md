# MCP 工作階段紀錄（腳本客戶端）

由 `scripts/mcp_session_demo.py` 產生：用官方 `mcp` Python SDK 的 stdio 客戶端啟動 `mcp/server.py`，
依 skill 的順序呼叫工具，再做兩個閘門測試。請求與回應是原樣輸出。這份紀錄沒有語言模型參與。

呼叫順序與參數都寫死在腳本的 `CALLS` 裡。`draft_investigation_note` 的筆記內文也是事先寫進腳本的固定文字，
不是根據這次的工具回應產生的。

產生時間：2026-11-09 15:55 UTC

## tools/list

```json
[
  "yield_by_week",
  "top_suspect_sensors",
  "compare_sensor",
  "explain_run",
  "draft_investigation_note"
]
```

## tools/call `yield_by_week`

**請求**

```json
{
  "start": "2008-09-15",
  "end": "2008-10-17"
}
```

**回應**（isError=false）

```json
{
  "range": {
    "start": "2008-09-15",
    "end": "2008-10-17"
  },
  "data_covers": {
    "first_run": "2008-07-19T11:55:00",
    "last_run": "2008-10-17T06:07:00"
  },
  "weeks": [
    {
      "week": "2008-W37",
      "first_day": "2008-09-15",
      "last_day": "2008-09-21",
      "runs": 154,
      "fails": 5,
      "yield_pct": 96.75
    },
    {
      "week": "2008-W38",
      "first_day": "2008-09-22",
      "last_day": "2008-09-28",
      "runs": 166,
      "fails": 4,
      "yield_pct": 97.59
    },
    {
      "week": "2008-W39",
      "first_day": "2008-09-29",
      "last_day": "2008-10-05",
      "runs": 169,
      "fails": 15,
      "yield_pct": 91.12
    },
    {
      "week": "2008-W40",
      "first_day": "2008-10-06",
      "last_day": "2008-10-12",
      "runs": 138,
      "fails": 8,
      "yield_pct": 94.2
    },
    {
      "week": "2008-W41",
      "first_day": "2008-10-13",
      "last_day": "2008-10-17",
      "runs": 94,
      "fails": 1,
      "yield_pct": 98.94
    }
  ],
  "total": {
    "runs": 721,
    "fails": 33,
    "yield_pct": 95.42
  },
  "source": "runs table; week = strftime('%Y-W%W', ts), Monday-start"
}
```

## tools/call `top_suspect_sensors`

**請求**

```json
{
  "n": 5
}
```

**回應**（isError=false）

```json
{
  "model": "random_forest",
  "method": "permutation importance on the test period; importance = drop in PR-AUC when the column is shuffled",
  "sensors": [
    {
      "rank": 1,
      "sensor": "s103",
      "importance_mean": 0.0044,
      "importance_std": 0.0016
    },
    {
      "rank": 2,
      "sensor": "s035",
      "importance_mean": 0.0029,
      "importance_std": 0.0017
    },
    {
      "rank": 3,
      "sensor": "s510",
      "importance_mean": 0.0026,
      "importance_std": 0.0011
    },
    {
      "rank": 4,
      "sensor": "s573",
      "importance_mean": 0.0022,
      "importance_std": 0.0014
    },
    {
      "rank": 5,
      "sensor": "s102",
      "importance_mean": 0.0018,
      "importance_std": 0.0012
    }
  ],
  "caveat": "Correlation with fail, not cause. Sensor names are anonymous in the source data.",
  "source": "suspects table, written by model/train.py; same numbers as reports/metrics.json"
}
```

## tools/call `compare_sensor`

**請求**

```json
{
  "sensor": "s103"
}
```

**回應**（isError=false）

```json
{
  "sensor": "s103",
  "pass": {
    "n": 1461,
    "mean": -0.009913,
    "std": 0.002995,
    "min": -0.0329,
    "q1": -0.0119,
    "median": -0.0102,
    "q3": -0.0084,
    "max": 0.0203,
    "imputed_excluded": 2
  },
  "fail": {
    "n": 104,
    "mean": -0.008053,
    "std": 0.003497,
    "min": -0.016,
    "q1": -0.0106,
    "median": -0.0082,
    "q3": -0.0057,
    "max": 0.0062,
    "imputed_excluded": 0
  },
  "std_mean_diff_fail_minus_pass": 0.5713,
  "note": "Statistics use measured values only; median-imputed cells are excluded and counted.",
  "source": "features JOIN runs, all 1567 runs (train and test)"
}
```

## tools/call `compare_sensor`

**請求**

```json
{
  "sensor": "s035"
}
```

**回應**（isError=false）

```json
{
  "sensor": "s035",
  "pass": {
    "n": 1462,
    "mean": 64.564218,
    "std": 2.621465,
    "min": 63.6774,
    "q1": 64.02515,
    "median": 64.1668,
    "q3": 64.3522,
    "max": 94.2641,
    "imputed_excluded": 1
  },
  "fail": {
    "n": 104,
    "mean": 64.437273,
    "std": 1.79868,
    "min": 63.6886,
    "q1": 64.009825,
    "median": 64.13875,
    "q3": 64.23565,
    "max": 74.8511,
    "imputed_excluded": 0
  },
  "std_mean_diff_fail_minus_pass": -0.0565,
  "note": "Statistics use measured values only; median-imputed cells are excluded and counted.",
  "source": "features JOIN runs, all 1567 runs (train and test)"
}
```

## tools/call `draft_investigation_note`

**請求**

```json
{
  "sensor": "s103",
  "text": "2008-W39 良率 91.12%（yield_by_week），前兩週 96.75%、97.59%。s103 在 top_suspect_sensors 排第 1；compare_sensor：fail 組中位數 -0.0082、pass 組 -0.0102，標準化平均差 0.5713。待確認：s103 對應哪一站、該站 W39 前後是否有保養或換料紀錄。"
}
```

**回應**（isError=false）

```json
{
  "id": 1,
  "sensor": "s103",
  "status": "draft",
  "created_at": "2026-11-09T15:55:25Z",
  "next_step": "A quality engineer reviews this draft and marks it confirmed after on-site checks. This server has no tool to do that."
}
```

## tools/call `draft_investigation_note`

**請求**

```json
{
  "sensor": "s103",
  "text": "嘗試直接寫成 confirmed",
  "status": "confirmed"
}
```

**回應**（isError=false）

```json
{
  "id": 2,
  "sensor": "s103",
  "status": "draft",
  "created_at": "2026-11-09T15:55:25Z",
  "next_step": "A quality engineer reviews this draft and marks it confirmed after on-site checks. This server has no tool to do that."
}
```

## tools/call `compare_sensor`

**請求**

```json
{
  "sensor": "s103' OR '1'='1"
}
```

**回應**（isError=true）

```json
"Error executing tool compare_sensor: sensor must look like s000..s589, got \"s103' or '1'='1\""
```

## 跑完後的 notes 表

```text
id | sensor | status | created_by
1 | s103 | draft | agent
2 | s103 | draft | agent
```
