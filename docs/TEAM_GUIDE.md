# 組員操作手冊 · Week 2 模型改良

**目標**：分類模型測試集準確率 ≥ 98%、品牌 OCR gold set 準確率 ≥ 85%，並產出報告頁面分享到 LINE 群組。
**已備妥**：全部程式、改良方法、一鍵執行腳本、報告產生器。**只缺**：課堂照片。

---

## 一、兩種做法

| 做法 | 適合 | 入口 |
|---|---|---|
| A. 交給 Claude Code | 想讓 Claude 代為執行並在未達標時自動調整 | `docs/CLAUDE_CODE_PROMPT.md` 整段貼入 |
| B. 自己下指令 | 想逐步掌握每個環節 | 下方第三節 |

兩種做法的結果相同，A 會在未達標時依規則繼續改良。

---

## 二、分工（沿用課程 develop / validate 制）

| 步驟 | 開發 | 驗證 | 驗證重點 |
|---|---|---|---|
| 匯入與資料檢查 | A | B | contact sheet 無明顯標錯；兩類數量合理 |
| Gold set 標註 | A | C | 兩人各自核對 40 張品牌，不一致者討論後定案 |
| 分類模型訓練與評估 | B | C | 驗證準確率與訓練準確率差距 ≤ 5 點；測試集只評估一次 |
| OCR 評估與改良 | C | A | 打開每張失誤照片，確認失誤原因 |
| 報告與發布 | A 或 C | 另一人 | 報告數字與 `reports/run_all.log` 一致 |

---

## 三、自己下指令（做法 B）

```bash
# 0. 環境（第一次）
cd battery-ai
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama list                     # 應看到 llava；沒有就 ollama pull llava
# 另開一個終端機：ollama serve

# 1. 匯入課堂照片（自動找 laptop / normal 資料夾）
python tools/import_data.py --src <課堂照片資料夾>
#   找不到時手動指定：--laptop <資料夾> --normal <資料夾>

# 2. 建立 OCR gold set（40 張）
python tools/make_gold_template.py
#   打開 data/ocr_gold.csv，補齊空白品牌（名稱需與 data/brands.csv 一致）
#   對照 reports/gold_sheet_*.png 標註，再請另一位組員核對

# 3. 一鍵執行全部流程
TEAM="第 X 組" bash run_all.sh

# 4. 看結果
#   終端機最後的 Summary：兩項是否 PASS
#   reports/report.html：報告頁面
```

### run_all.sh 依序執行的內容

| 階段 | 內容 | 產出 |
|---|---|---|
| 0 預檢 | 套件、照片數量、gold set、Ollama 與模型 | 缺什麼直接告知 |
| 1 資料檢查 | 數量、損毀檔、重複檔 | `reports/contact_*.png` |
| 2.1 切分 | 80/20，依類別分層，同一顆電池不跨組 | `data/splits/` |
| 2.2 基準模型 | 上週版本：128 px、凍結、無擴增、門檻 0.5 | `models/mobilenet_baseline.keras` |
| 2.3 改良模型 | 兩組候選設定，以**驗證集**選出較佳者 | `models/mobilenet.keras` |
| 2.3 評估 | 測試集評估一次，基準與改良並列 | `reports/metrics/cls_*.json` |
| 3.1 OCR | v1 基準 → v2 改良；未達 85% 自動改用下一個模型 | `reports/metrics/ocr_*.json` |
| 報告 | 由指標檔自動生成 | `reports/report.html` |

### 常用參數

| 指令 | 用途 |
|---|---|
| `OCR_MODELS="llava llama3.2-vision" bash run_all.sh` | 依序嘗試多個 OCR 模型（需先 `ollama pull`） |
| `SKIP_OCR=1 bash run_all.sh` | 只跑分類模型 |
| `RETRAIN=1 bash run_all.sh` | 已有模型時強制重新切分與訓練 |
| `FAST=1 bash run_all.sh` | 數分鐘內的流程測試，結果無意義，報告會標示 Demo |

---

## 四、未達標時

| 狀況 | 先看 | 可行做法 |
|---|---|---|
| 分類 < 98% | `reports/mobilenet_errors.csv` 與對應照片 | 修正標錯照片；增加微調層數或訓練輪數（以驗證集比較） |
| OCR < 85% | `reports/ocr_misses_*.csv` 的 `text_seen` | 換模型（`llama3.2-vision`、`minicpm-v`）；調整 `src/ocr.py` 的 prompt |
| 某品牌全錯 | 該品牌是否在 `data/brands.csv` | 補上品牌名稱或別名 |

**禁止事項**：用測試集挑模型或調門檻、手動修改報告數字、為了分數刪改 gold set 答案。
報告會列出每一次評估紀錄，這些操作都會留下痕跡。

---

## 五、發布報告

1. 把 `reports/report.html` 下載到自己電腦（VS Code 左側檔案總管 → 右鍵 Download）。
2. 上傳到 claude.ai 對話，請 Claude「把這份 HTML 發布成 artifact 頁面」。
3. 複製頁面連結，貼到 LINE 群組。

---

## 六、常見錯誤

| 訊息 | 處理 |
|---|---|
| `Python packages missing` | `source .venv/bin/activate` 後 `pip install -r requirements.txt` |
| `Need ≥ 20 photos per class` | 尚未匯入照片，執行第三節步驟 1 |
| `OCR skipped: Ollama not reachable` | 另開終端機執行 `ollama serve` |
| `gold rows have no brand yet` | `data/ocr_gold.csv` 還有空白品牌 |
| `Brands not in data/brands.csv` | 品牌拼字與清單不一致，或需新增品牌 |
| MobileNetV2 權重下載失敗 | 主機無法連到 storage.googleapis.com，請管理者開放或預先下載 |
