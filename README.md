# BatteryScan · battery photo AI for Client W

Laptop batteries pay no recycling fee; every other battery is billed to its maker.
`src/pipeline.py` classifies each photo (CNN / MobileNetV2), skips laptops, reads the
brand on the rest with LLaVA (local, via Ollama) and counts fees per brand.

## Setup (once, on a clean laptop)
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Ollama: install from https://ollama.com, then
ollama pull llava
ollama serve            # keep running in its own terminal
```

## Run, in order (from the repo root)
```bash
python src/check_images.py                 # 1   data check + contact sheets in reports/
python src/prepare.py && python src/prepare.py --check   # 2.1 split + leak check
python src/train_cnn.py                    # 2.2 models/cnn.keras
python src/train_transfer.py               # 2.3 models/mobilenet.keras
python src/evaluate.py --model models/cnn.keras models/mobilenet.keras
python src/eval_ocr.py                     # 3.1 needs data/ocr_gold.csv filled in
python src/pipeline.py data/inbox/         # 4   reports/fees.csv
pytest -q                                  # tests (model and Ollama mocked)
```

Photos go in `data/raw/laptop/` and `data/raw/normal/` (folder name = label).
To dry-run without real photos: `python tools/make_fake_data.py` (photos, gold set and an
inbox with its true answer), then after step 4 `python tools/compare_expected.py`.
Before adding real photos, delete every `FAKE_*` file and reset `data/ocr_gold.csv` to its header.

Settings (image size, seed, split, model path, threshold, Ollama host): `src/config.py`.
The production model is `models/mobilenet.keras` by default; override with `BATTERY_MODEL`.

## Outputs
| File | Content |
|---|---|
| reports/results.csv | file, label, confidence, brand |
| reports/fees.csv | brand, count (billable only) |
| reports/review.csv | normal batteries whose brand is unknown → manual review |
