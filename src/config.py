"""Single source of truth for paths and settings (CLAUDE.md: size, split and seed live here only)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Paths
RAW = ROOT / "data" / "raw"            # data/raw/laptop, data/raw/normal (folder name = label)
SPLITS = ROOT / "data" / "splits"      # train.txt / test.txt
REVIEW = ROOT / "data" / "review"      # unsure photos, ask Client W
BRANDS_CSV = ROOT / "data" / "brands.csv"
OCR_GOLD_CSV = ROOT / "data" / "ocr_gold.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

# Data
CLASSES = ("laptop", "normal")         # label 1 = laptop, 0 = normal
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
IMG = 128                              # every photo is reshaped to IMG x IMG x 3
SEED = 42
TEST_SIZE = 0.2                        # 80 / 20 train / test
VAL_SIZE = 0.2                         # share of train held out for validation
NEAR_DUP_DIST = 10                     # dHash (256-bit) Hamming distance that counts as "same battery"

# Targets (course requirement, week 2)
TARGET_CLS_ACCURACY = 0.98             # battery classification, test set
TARGET_OCR_ACCURACY = 0.85             # brand OCR, gold set
TARGET_NORMAL_RECALL = 0.95            # each missed normal battery is a fee never billed

# Improved classifier (v2)
IMG_TRANSFER = 224                     # MobileNetV2's native input size
METRICS = ROOT / "reports" / "metrics" # one JSON per evaluation run, read by tools/build_report.py

# Production
MODEL_PATH = Path(os.getenv("BATTERY_MODEL", MODELS / "mobilenet.keras"))  # set after step 2.3
THRESHOLD = float(os.getenv("BATTERY_THRESHOLD", 0.5))  # p(laptop) >= THRESHOLD -> laptop, no fee

# OCR (LLaVA through Ollama)
OLLAMA = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b")  # D9: 95% on the gold set vs llava 59%
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", 120))  # raise on a shared, busy Ollama server
BRAND_CUTOFF = 0.8                     # fuzzy-match cutoff
