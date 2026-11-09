# 製程良率異常分析 Agent（SECOM）

品管工程師問「最近良率掉了，可能是哪個站？」，Agent 透過 MCP 查每週良率、列出和 fail 一起變動的感測欄、比對 pass 與 fail 兩組的分布，最後寫一筆調查筆記。筆記只能是草稿，確認要由人到現場做。

- 資料：UCI SECOM 半導體製程資料，1567 筆、590 個感測欄（出處與授權見[資料出處](#資料出處)）
- 技術：Python（pandas、scikit-learn）、SQLite、MCP 伺服器（官方 `mcp` SDK，stdio）、Claude Code skill
- MCP 工作階段紀錄：[demo/mcp_session.md](demo/mcp_session.md)
- 領域筆記：[docs/domain.md](docs/domain.md)（良率、SPC、8D 與本資料集的限制）

本機 CLI 加 MCP 伺服器，沒有網站。

## 1. 問題

晶圓廠一條線上有幾百個感測點。良率掉了，工程師要先回答兩件事：掉在哪幾週、哪些量測值跟著變。逐欄翻管制圖要很久，而且多數欄位和良率無關。

這個 repo 把那一步縮短：先把髒資料清乾淨放進資料庫，用模型排出候選欄，再讓 Agent 照固定順序查證並留下筆記。Agent 給的是調查起點，判斷原因與確認筆記仍是人的工作。

## 2. 資料與清洗

`python -m pipeline.clean` 讀原始檔，輸出 `data/processed/` 與 [reports/cleaning.md](reports/cleaning.md)。

| 項目 | 清洗前 | 清洗後 |
|---|---|---|
| 筆數 | 1567 | 1567 |
| 感測欄數 | 590 | 444 |
| 缺值格數 | 41951 | 0 |

做法：

1. 依時間戳排序，前 80%（1253 筆）當訓練集，後 20%（314 筆）當測試集。
2. 訓練集缺值率超過 50% 的欄刪掉，共 24 欄。
3. 訓練集上只有一種值（或全空）的欄刪掉，共 122 欄。
4. 剩下的缺值用中位數補。中位數只用訓練集算。

刪欄和補值都只看訓練集，測試期的資料不會影響前處理，這一點有測試守著（`test_imputer_medians_come_from_train_only`、`test_changing_test_values_does_not_change_fit`）。同一份原始檔跑兩次，輸出檔的 SHA-256 相同。

標籤檔用 `-1` 表示 pass、`1` 表示 fail；出現其他代碼時清洗直接報錯，不會默默歸類。

## 3. 資料表

`python -m pipeline.load_db` 依 [schema.sql](schema.sql) 建 `data/secom.db`：

| 表 | 內容 |
|---|---|
| `runs` | 每筆生產單位：id（原始檔列號）、ts、result、split |
| `features` | 清洗後的感測值，長表：(run_id, sensor, value, imputed) |
| `cleaning_log` | 清洗步驟與每步前後欄數 |
| `notes` | 調查筆記，CHECK 允許 draft 或 confirmed、預設 draft，但兩個 trigger 擋掉 INSERT 和 UPDATE 寫入 draft 以外的值，所以目前只存得進 draft |
| `scores`、`contributions`、`suspects` | 模型輸出，由 `model/train.py` 寫入 |

`features` 用長表，理由有兩個。第一，感測欄名是查詢參數，用 `WHERE sensor = ?` 綁定，不必把欄名拼進 SQL，Agent 傳什麼字串都碰不到 SQL 結構。第二，`imputed` 旗標可以跟著每一格存，分組統計時加一個 `imputed = 0` 條件就能排除補值。代價是 695748 列，比寬表大很多，所以 `sensor` 欄另建了索引。

模型結果先算好寫進表裡，MCP 伺服器只讀表，不載入 pickle 檔。

## 4. 模型

`python -m model.train`。數字全部來自 [reports/metrics.json](reports/metrics.json)。下面兩個標記之間的區塊由 `python scripts/render_readme_metrics.py` 從 metrics.json 產生，測試會比對區塊全文，手改或數字過期都會失敗。

- 切分：依時間，前 80% 訓練、後 20% 測試。邏輯迴歸的 C、判定門檻、要用哪個模型的特徵排名，都在訓練期最後 20%（驗證段）決定。測試期用來報最後的數字，permutation importance 也在測試期上算。
- 驗證段不是完全乾淨：刪欄與補值用的中位數是在整個訓練期（含驗證段）上算的；判定門檻在驗證段上選好之後，套用到用整個訓練期重新訓練的模型，兩個模型的分數分布不一定相同。
- 類別不平衡：邏輯迴歸用 `class_weight="balanced"`，隨機森林用 `balanced_subsample`。

<!-- metrics:start -->
測試期 314 筆，其中 fail 17 筆（fail 率 0.0541）。

| 模型 | PR-AUC | ROC-AUC | 門檻 | Precision | Recall | 抓到 fail | 誤報 |
|---|---|---|---|---|---|---|---|
| 全猜 pass（基準） | 0.0541 | | | 無（沒有任何一筆被判 fail） | 0.0 | 0 | 0 |
| 邏輯迴歸（C=0.0001） | 0.0936 | 0.6621 | 0.4621 | 0.1047 | 0.5294 | 9 | 77 |
| 隨機森林 | 0.0725 | 0.5357 | 0.074 | 0.0748 | 0.6471 | 11 | 136 |

全猜 pass 的 accuracy 是 0.9459。門檻改用 0.5 時，邏輯迴歸抓到 2 筆 fail、誤報 19 筆，隨機森林一筆都沒判 fail。

驗證段 251 筆，fail 11 筆（fail 率 0.0438，也就是全猜 pass 的 PR-AUC）。驗證段 PR-AUC：邏輯迴歸 0.0434、隨機森林 0.0576。邏輯迴歸在驗證段低於全猜 pass。
<!-- metrics:end -->

怎麼讀這張表：

- 兩個模型的 PR-AUC 都只比全猜 pass 高一點。隨機森林的 ROC-AUC 接近隨機猜。這個模型不能拿來當自動判退的閘門。
- 模型依事先定好的驗證段規則選出（驗證段 PR-AUC 較高者），選到的是隨機森林，但在測試期它比邏輯迴歸差。驗證段 fail 太少，選模型這一步本身就不穩。`model/train.py` 裡沒有看了測試結果再換模型的步驟。
- 邏輯迴歸選到的 C 是網格最小值，代表正則化越強越好，模型幾乎學不到穩定的訊號。
- 前 10 名候選欄只能當排查順序的參考，沒有驗證過比隨機挑欄好。兩個模型的前 10 名只有 3 欄重疊（s103、s510、s348）。`top_suspect_sensors` 回傳 permutation importance 的平均與標準差；隨機森林前 10 名中，有幾欄的標準差和平均差不多大，那些排名不可靠。

## 5. Agent 與閘門

MCP 伺服器 `mcp/server.py`（stdio）用官方 SDK 2.x 的 `MCPServer` 寫成（1.x 時叫 `FastMCP`），有 5 個工具：

| 工具 | 做什麼 | 連線 |
|---|---|---|
| `yield_by_week(start, end)` | 每週筆數、fail 數、良率 | 唯讀 |
| `top_suspect_sensors(n)` | permutation importance 前 n 名（n ≤ 10） | 唯讀 |
| `compare_sensor(sensor)` | pass 與 fail 兩組的分布統計，只用實測值 | 唯讀 |
| `explain_run(run_id)` | 兩個模型的風險分數，加上邏輯迴歸貢獻最大的 5 欄 | 唯讀 |
| `draft_investigation_note(sensor, text)` | 寫一筆筆記 | 只做一條 INSERT |

閘門：

- 查詢工具一律用 `mode=ro` 開 SQLite，任何寫入都會被 SQLite 拒絕。沒有可以下任意 SQL 的工具。
- `draft_investigation_note` 的 INSERT 把 status 寫死成 `'draft'`。這個工具沒有 status 參數，呼叫時多塞 `status: "confirmed"` 也只會存成 draft（[demo/mcp_session.md](demo/mcp_session.md) 有這次呼叫的原始回應）。伺服器沒有任何更新或刪除筆記的工具。
- 感測欄名先過格式檢查（`s` 加三位數字），再確認它在清洗後還存在；不合格就回錯誤訊息給 Agent。
- 資料庫另有兩個 trigger（[schema.sql](schema.sql)）：任何連線想把 `notes.status` 改成或寫成 draft 以外的值，都會被擋下。

閘門的範圍：draft 閘門只在 MCP 工具層成立，前提是 Agent 只能透過這 5 個工具碰資料庫。Agent 若同時有 shell 或檔案工具，就能直接開資料庫檔；trigger 擋得住一條直接的 UPDATE，但 Agent 可以先刪掉 trigger 再改 status，直接改資料庫就能繞過閘門。錄 demo 時只開這 5 個工具和 Skill，就是這個原因。在 repo 裡用預設工具組開 Claude Code 時，擋住 Agent 的只剩 skill 裡的文字規則。

Skill `.claude/skills/yield-triage/SKILL.md` 規定順序：yield → suspects → compare → 草擬筆記。回答裡每個數字都要標出是哪個工具給的，並且要講明「相關不等於因果，要現場驗證」。資料裡沒有站別名稱，問「哪個站」時，Agent 只能回答感測欄編號。

## 6. 限制

- 感測欄是匿名的，沒有站別、機台、批號，無法直接回答「哪個站」，也做不了機台共通性分析。
- 時間戳是測試時間，不是各站加工時間。
- 測試期只有 17 筆 fail，指標的誤差很大。
- `explain_run` 的特徵貢獻來自邏輯迴歸（係數 × 標準化值），不是被選中的隨機森林；兩者看的東西不同。訓練期的筆數模型看過，分數偏樂觀。
- 筆記的 `created_by` 固定是 `agent`，沒有登入機制。確認筆記沒有做：trigger 擋下所有改成 confirmed 的寫入，`confirmed` 這個值保留給之後的審核步驟。
- 工具把 `suspects`、`contributions`、`model_meta` 裡的字串原樣回給模型。這假設資料庫是本機用 pipeline 從 UCI 檔建的；若改接別人給的資料庫，回傳前要先驗感測欄名格式。
- 更多說明見 [docs/domain.md](docs/domain.md) 第 4 節。

## 安裝

需要 Python 3.11 以上、Claude Code。

```bash
# 1. 建環境、裝套件（要完全重現 reports/ 的數字，改裝 requirements-lock.txt）
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 2. 下載資料（比對寫死的 SHA-256，不符就停）、清洗、建資料庫、訓練（訓練約 6 分鐘，12 核心）
.venv/bin/python scripts/download_data.py
.venv/bin/python -m pipeline.clean && .venv/bin/python -m pipeline.load_db && .venv/bin/python -m model.train

# 3. 檢查 MCP 工具，然後在 repo 根目錄開 Claude Code，核准 .mcp.json 裡的 mfg-yield
.venv/bin/python mcp/server.py --list-tools
```

測試：`.venv/bin/python -m pytest`。多數測試用 `tests/conftest.py` 產生的合成資料，不必下載。只有 `test_committed_cleaning_report_matches_real_data`（拿 UCI 原檔重算 `reports/cleaning.md` 的欄數）需要 `data/raw/`，沒下載時這條會顯示 skip。

改了 `reports/metrics.json` 之後，跑 `.venv/bin/python scripts/render_readme_metrics.py` 更新 README 的指標區塊。

## 目錄

```
pipeline/       清洗與建資料庫
model/          訓練、評估、permutation importance
mcp/server.py   MCP 伺服器
schema.sql      資料表定義
tests/          pytest
scripts/        下載資料、README 指標區塊產生、MCP 工作階段紀錄、Claude Code demo 錄製
reports/        cleaning.md、metrics.json（由程式產生）
docs/domain.md  領域筆記
.claude/skills/yield-triage/  Claude Code skill
```

## 資料出處

McCann, M. & Johnston, A. (2008). SECOM [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C54305>

授權：Creative Commons Attribution 4.0 International（CC BY 4.0），<https://creativecommons.org/licenses/by/4.0/>。本 repo 不包含原始資料，`scripts/download_data.py` 從 UCI 下載到 `data/raw/`。`reports/` 與 `demo/mcp_session.md` 是用這份資料算出的衍生結果（清洗、建模與工具呼叫的輸出），不是原始資料。

## 開發方式

方向、規格核可與驗收由作者負責；需求拆解、程式、測試和文件都用 Claude Code 完成。

程式碼採 MIT License。
