# 給 Claude Code 的完整指令

用法：在專案根目錄啟動 `claude`，按 Shift+Tab 切到 **Plan mode**，把下方「指令本文」整段貼上。
Claude 會先提出計畫，確認後再切回一般模式執行。只需改兩處：`<資料夾路徑>` 與 `<組名>`。

---

## 指令本文

```
你要完成「BatteryScan 模型改良」作業。專案在目前目錄（battery-ai），先讀 CLAUDE.md、README.md、
docs/PLAN.md、docs/DECISIONS.md、docs/TEAM_GUIDE.md，理解架構後再動手。

【作業要求】
1. 電池分類模型（laptop vs normal）在測試集的準確率 ≥ 98%。
2. 電池品牌 OCR 在 gold set 的準確率 ≥ 85%。
3. 產出一個報告頁面，呈現兩個模型的開發與優化過程、最終結果、評估指標與準確率，
   並清楚說明每一項改動如何提升表現。

【資料】
課堂照片在：<資料夾路徑>
組名：<組名>

【執行步驟】
1. 環境：確認 python3 ≥ 3.10，建立 .venv 並安裝 requirements.txt；確認 `ollama list` 有 llava，
   沒有就執行 `ollama pull llava`；確認 ollama serve 正在執行。任何一項失敗就停下來告訴我原因。
2. 匯入：python tools/import_data.py --src <資料夾路徑>
   若自動偵測不到 laptop / normal 資料夾，列出該路徑的目錄結構，問我哪個資料夾對應哪個類別。
3. 資料檢查：python src/check_images.py，打開 reports/contact_*.png，
   列出看起來標錯、模糊或一張照片有多顆電池的檔名與理由。先不要移動任何檔案，等我確認。
4. Gold set：python tools/make_gold_template.py
   若 data/ocr_gold.csv 有空白品牌，逐張打開照片提出品牌建議，寫成表格給我核對；
   經我確認後才寫入 CSV。看不清品牌的照片列出來由我決定刪除或保留。
5. 一鍵執行：TEAM="<組名>" bash run_all.sh
   需要長時間訓練時用 Ctrl+B 放到背景。
6. 讀 reports/run_all.log 的 Summary，回報兩個模型是否達標。

【未達標時的改良規則】
分類 < 98%：
- 先讀 reports/mobilenet_errors.csv 並打開錯誤照片，判斷是標註錯誤、照片品質，還是模型能力。
- 標註錯誤：列出給我確認後修正（記錄到 docs/DECISIONS.md），修正後重新切分與訓練。
- 模型能力：在 src/train_transfer.py 的參數空間內新增候選設定，例如 --fine-tune-layers 100、
  --epochs-ft 30、--img 192，每個候選用不同 --name 訓練，
  再用 python tools/select_model.py <候選們> --out models/mobilenet.keras 依「驗證集」選出最佳，
  最後才用 src/evaluate.py 在測試集評估一次。
OCR < 85%：
- 讀 reports/ocr_misses_*.csv 與 text_seen，打開每張失誤照片，分類失誤原因（標籤背對鏡頭、只有 logo、模型讀錯字、品牌不在清單）。
- 可嘗試其他本機視覺模型：ollama pull llama3.2-vision（或 minicpm-v、qwen2.5vl），
  再執行 python src/eval_ocr.py --version v2 --model <模型> --stage improved。
- 可調整 src/ocr.py 的 PROMPT_V2 或前處理；每次改動都重新跑 eval_ocr 並記錄結果。
每一輪改動都更新 docs/DECISIONS.md：改了什麼、為什麼、前後數字。

【不可違反的規則】
- 測試集只能由 src/evaluate.py 讀取；調參與選模型一律看驗證集。
- 報告中的每個數字都必須來自 reports/metrics/*.json，不得手動填寫或修改。
- 不得為了提高分數修改或刪除 gold set 的正確答案；刪除任何 gold 照片都要先經我同意並寫明理由。
- 不得把 data/、models/、.env 提交到 git。
- 新增套件前先問我。

【產出報告】
全部完成後執行：python tools/build_report.py --team "<組名>"
然後依下列課程指定的 prompt 產生報告頁面，內容以 reports/report.html 為基礎：
"Please generate an artifact page presenting the development and optimization processes for both
the battery classification model and the battery brand OCR model. Include the final model results,
evaluation metrics, and accuracy rates, and clearly explain what changes were made to improve model
performance."
如果你無法在這個環境發布 artifact，就告訴我 reports/report.html 已就緒，我會上傳到 claude.ai 發布。

【完成條件】
- run_all.sh 的 Summary 顯示兩項皆 PASS，或明確說明未達標的原因與已嘗試的方法。
- reports/report.html 已產生，docs/DECISIONS.md 已更新。
- pytest -q 全數通過，執行 /review 沒有阻擋性問題，然後 git commit。
```

---

## 課程原始的報告 prompt（備查）

> Please generate an artifact page presenting the development and optimization processes for both the battery classification model and the battery brand OCR model. Include the final model results, evaluation metrics, and accuracy rates, and clearly explain what changes were made to improve model performance.

完成後把頁面連結貼到 LINE 群組。
