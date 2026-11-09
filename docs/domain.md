# 領域知識筆記：良率、SPC、8D，以及 SECOM 的限制

這份筆記只寫看懂本 repo 所需的最小知識，每一節附公開來源。

## 1. 良率（yield）

- **定義**：一段期間內，通過檢驗的件數除以投入的件數。英文常見的 first-pass yield（FPY）指第一次就通過、不含重工的比例。來源：Wikipedia〈First-pass yield〉，<https://en.wikipedia.org/wiki/First-pass_yield>。
- **多站製程**：每一站各有良率，整條線一次做對的機率是各站良率相乘，稱為 rolled throughput yield（RTY）。相乘不會放大單站的變化：某一站良率掉 2 個百分點，整線 RTY 的絕對降幅等於 2 個百分點乘上其他各站良率的乘積，所以不會超過 2 個百分點，相對降幅則和那一站相同。真正的麻煩是看不出來源：同一時間別站若變好，整線數字可能幾乎不動；只看整線良率，也分不出是哪一站掉的。來源：Wikipedia〈Rolled throughput yield〉，<https://en.wikipedia.org/wiki/Rolled_throughput_yield>。
- **本 repo 的算法**：SECOM 每一筆是一個生產單位的線上測試結果（pass／fail），`yield_by_week` 算的是「該週 pass 筆數 ÷ 該週筆數」。它對應的是最後測試點的良率，看不出是哪一站造成的。

## 2. SPC（統計製程管制）

- **管制圖**：把製程量測值依時間畫出來，中心線是平均，上下管制界限通常放在 ±3σ。點落在界限外，代表製程可能出現了一般變異以外的原因，要去查。來源：NIST/SEMATECH e-Handbook of Statistical Methods，6.3 節〈Univariate and Multivariate Control Charts〉，<https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc3.htm>。
- **判讀規則**：只看「超出 3σ」會漏掉慢慢偏移的情況。Western Electric rules 另外把 ±1σ、±2σ 分成區，例如「連續 3 點有 2 點落在同側 2σ 外」「連續 5 點有 4 點落在同側 1σ 外」也算異常。來源：Wikipedia〈Western Electric rules〉，<https://en.wikipedia.org/wiki/Western_Electric_rules>。
- **和本 repo 的關係**：SPC 是一欄一欄看、事先定好界限；本 repo 的模型是一次看 444 欄、事後找哪些欄和 fail 一起變動。兩者互補。模型列出的可疑欄，下一步應該去調那一欄的管制圖，確認偏移從哪一天開始。

## 3. 8D 問題解決

Ford 發展出的 8 個步驟（Global 8D 版另加 D0）：

| 步驟 | 內容 |
|---|---|
| D0 | 準備，先做緊急應變 |
| D1 | 組成懂產品與製程的小組 |
| D2 | 用 5W2H 量化描述問題 |
| D3 | 暫時圍堵，不讓問題流到客戶 |
| D4 | 找出並驗證根本原因與流出點（escape point） |
| D5 | 驗證永久對策真的有效 |
| D6 | 實施永久對策並確認改善 |
| D7 | 修改制度與程序，防止再發 |
| D8 | 肯定小組成員 |

來源：Wikipedia〈Eight disciplines problem solving〉，<https://en.wikipedia.org/wiki/Eight_disciplines_problem_solving>。

**本 repo 的 Agent 能幫到哪一步**：D2 的量化描述（哪幾週、掉多少、fail 幾筆）和 D4 的候選清單（哪些感測欄和 fail 一起變動）。D4 要求原因「必須被驗證」，Agent 做不到，所以它寫的調查筆記一律是 draft。改成 confirmed 應該在品管工程師到現場確認之後，經另一個審核步驟完成，本 repo 沒有做這個步驟。D3、D5 以後都是現場的事。

## 4. SECOM 資料集的限制

資料來源：McCann, M. & Johnston, A. (2008). SECOM [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C54305>，授權 CC BY 4.0。資料集頁面：<https://archive.ics.uci.edu/dataset/179/secom>。

- **感測欄是匿名的**：590 欄只有位置，沒有名稱、單位、所屬站別或機台。UCI 頁面寫 591 個 feature，下載檔裡是 590 個量測欄，加上標籤檔的 pass／fail 與時間戳。本 repo 把欄命名為 `s000`–`s589`（從 0 起算的欄位置）。因此 Agent 回答不了「是哪一站」，只能回答「是哪一欄」，要對回站別得用工廠自己的感測器清單。
- **只有一個產品、一段期間**：時間戳從 2008-07-19 到 2008-10-17，約三個月。沒有批號、晶圓位置、機台、配方等欄位，無法做機台共通性分析（commonality analysis）。
- **標籤很不平衡**：1567 筆中 fail 104 筆（6.64%），依時間切分後測試期只有 17 筆 fail。指標的波動會很大，多一筆或少一筆 fail 就會讓 recall 差 6 個百分點。
- **缺值多**：原始 590 欄中有 41951 格缺值。缺值的原因資料集沒有說明（可能是沒量、量測失敗、或該站不適用），本 repo 用中位數補，並在資料庫標記哪些格是補的，分組統計時排除。
- **時間戳是測試時間**：不是各站加工的時間，所以就算找到偏移的週，也不能直接推回哪一天哪一站出事。
- **相關不等於因果**：模型和 permutation importance 只說明「打亂這欄後模型變差多少」。一欄排名高，可能是它本身有問題，也可能只是和真正的原因同時變動。UCI 頁面列的相關任務是 Classification 與 Causal-Discovery，說明文字寫的是 "causal relationships may also be considered with a view to identifying the key features"：因果關係是可以另外探討的方向，資料本身沒有給出答案。
