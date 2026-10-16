# CLAUDE.md
Battery photo AI for Client W. A CNN sorts laptop vs normal batteries;
a local vision model (LLaVA via Ollama) reads the brand on normal ones; output = fees per brand.
Week 2 targets: classification ≥ 98% (test set), brand OCR ≥ 85% (gold set), plus a report page.

## Commands (run from the repo root)
- Import class photos: python tools/import_data.py --src <folder>
- OCR gold set template: python tools/make_gold_template.py
- Everything, end to end: TEAM="<team>" bash run_all.sh   (FAST=1 for a smoke test)
- Data check: python src/check_images.py
- Split + leak check: python src/prepare.py && python src/prepare.py --check
- Train: python src/train_transfer.py --name <name> [--fine-tune-layers N --epochs-ft N --img N]
- Pick by validation: python tools/select_model.py <models...> --out models/mobilenet.keras
- Evaluate (test set): python src/evaluate.py --model <models...> --stage baseline improved
- OCR: python src/eval_ocr.py --version v1|v2 --model <ollama model> --stage baseline|improved
- Report: python tools/build_report.py --team "<team>"   Tests: pytest -q

## Rules
- Only src/evaluate.py may read the test set; tune and select on validation.
- Every number in a report comes from reports/metrics/*.json; never type or edit metrics by hand.
- Never change or delete gold-set answers to raise a score; ask before removing a gold photo.
- Image size, split, seed and targets live in src/config.py only.
- Labels: folder name is the label; 1 = laptop, 0 = normal.
- "unknown" brands go to reports/review.csv, never to fees.csv.
- Never commit photos (data/raw, data/inbox, data/ocr_gold), models/ or .env.
- Ask before adding a dependency; log every experiment in docs/DECISIONS.md.
