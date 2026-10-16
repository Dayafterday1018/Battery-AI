# CLAUDE.md
Battery photo AI for Client W. A CNN sorts laptop vs normal batteries;
LLaVA (Ollama) reads the brand on normal ones; output = fees per brand.

## Commands (run from the repo root)
- Data check: python src/check_images.py
- Split + reshape: python src/prepare.py   (verify: python src/prepare.py --check)
- Train: python src/train_cnn.py | python src/train_transfer.py
- Evaluate: python src/evaluate.py --model models/<name>.keras
- Brand OCR: python src/ocr.py <image>   (needs: ollama serve)
- OCR gold set: python src/eval_ocr.py
- Full run: python src/pipeline.py data/inbox/   Tests: pytest -q

## Rules
- Never commit photos (data/raw, data/inbox, data/review), models/ or .env.
- Image size, split, seed and threshold live in src/config.py only.
- Only src/evaluate.py may read the test set.
- Labels: folder name is the label; 1 = laptop, 0 = normal.
- "unknown" brands go to reports/review.csv, never to fees.csv.
- Ask before adding a dependency; log decisions in docs/DECISIONS.md.
