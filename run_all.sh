#!/usr/bin/env bash
# Week 2 in one command: check -> split -> train baseline + improved -> evaluate -> OCR v1/v2 -> report.
#
#   bash run_all.sh                         full run
#   TEAM="Team 3" bash run_all.sh           team name on the report
#   OCR_MODELS="llava llama3.2-vision" bash run_all.sh   OCR models to try in order until ≥ 85%
#   SKIP_OCR=1 bash run_all.sh              classifier only
#   RETRAIN=1 bash run_all.sh               retrain even if models already exist
#   FAST=1 bash run_all.sh                  smoke test: tiny epochs, no ImageNet download
#
# Needs: photos in data/raw/{laptop,normal} (tools/import_data.py), a filled data/ocr_gold.csv
# (tools/make_gold_template.py) and `ollama serve` running for the OCR part.
# Everything is logged to reports/run_all.log.

set -uo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
TEAM="${TEAM:-}"
FAST="${FAST:-0}"
RETRAIN="${RETRAIN:-0}"
SKIP_OCR="${SKIP_OCR:-0}"
OCR_MODELS="${OCR_MODELS:-llava}"
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
export TF_CPP_MIN_LOG_LEVEL=3

mkdir -p reports models
exec > >(tee -a reports/run_all.log) 2>&1
step() { echo; echo "══════ $* ══════"; }
die()  { echo "✗ $*"; exit 1; }
ok()   { echo "✓ $*"; }

if [ "$FAST" = "1" ]; then
  T_BASE="--no-pretrained --epochs-head 2"
  T_A="--no-pretrained --epochs-head 2 --epochs-ft 2"
  T_B="--no-pretrained --epochs-head 2 --epochs-ft 2 --fine-tune-layers 80"
else
  T_BASE=""
  T_A="--fine-tune-layers 40 --epochs-ft 15"
  T_B="--fine-tune-layers 80 --epochs-ft 25"
fi

# ─────────────────────────────────────────────────────────────── preflight
step "0 · Preflight"
$PY -c "import tensorflow, sklearn, PIL, httpx, matplotlib" 2>/dev/null \
  || die "Python packages missing. Run: $PY -m pip install -r requirements.txt"
ok "Python packages"

read -r N_LAP N_NOR <<<"$($PY -c "
import sys; sys.path.insert(0,'src')
from config import RAW; from imaging import list_images
ps=list_images(RAW); print(sum(p.parent.name=='laptop' for p in ps), sum(p.parent.name=='normal' for p in ps))")"
[ "$N_LAP" -ge 20 ] && [ "$N_NOR" -ge 20 ] \
  || die "Need ≥ 20 photos per class; found laptop $N_LAP, normal $N_NOR. Run tools/import_data.py first."
ok "Photos: laptop $N_LAP · normal $N_NOR"

RUN_OCR=1
if [ "$SKIP_OCR" = "1" ]; then
  RUN_OCR=0; echo "• OCR skipped (SKIP_OCR=1)"
else
  GOLD_STATE="$($PY -c "
import csv
rows=list(csv.DictReader(open('data/ocr_gold.csv',encoding='utf-8')))
blank=sum(1 for r in rows if not r['brand'].strip())
print('empty' if not rows else ('blank' if blank else 'ok'), len(rows), blank)")"
  read -r G_STATE G_N G_BLANK <<<"$GOLD_STATE"
  case "$G_STATE" in
    ok)    ok "OCR gold set: $G_N photos" ;;
    empty) RUN_OCR=0; echo "• OCR skipped: data/ocr_gold.csv is empty (run tools/make_gold_template.py)";;
    blank) RUN_OCR=0; echo "• OCR skipped: $G_BLANK of $G_N gold rows have no brand yet";;
  esac
  if [ "$RUN_OCR" = "1" ]; then
    if ! TAGS="$(curl -fsS --max-time 5 "$OLLAMA_HOST/api/tags" 2>/dev/null)"; then
      RUN_OCR=0; echo "• OCR skipped: Ollama not reachable at $OLLAMA_HOST (start it with: ollama serve)"
    else
      AVAILABLE=""
      for m in $OCR_MODELS; do
        if echo "$TAGS" | grep -qE "\"name\": ?\"$m(:latest)?\""; then AVAILABLE="$AVAILABLE $m"
        else echo "• $m not pulled (ollama pull $m); skipping it"; fi
      done
      OCR_MODELS="${AVAILABLE# }"
      [ -n "$OCR_MODELS" ] && ok "Ollama models: $OCR_MODELS" || { RUN_OCR=0; echo "• OCR skipped: none of the OCR models are pulled"; }
    fi
  fi
fi

# ─────────────────────────────────────────────────────────────── step 1
step "1 · Data check"
$PY src/check_images.py || die "check_images failed"
echo "→ open reports/contact_laptop.png and reports/contact_normal.png and look for mislabeled photos"

# ─────────────────────────────────────────────────────────────── step 2.1
step "2.1 · Split and leak check"
if [ ! -s data/splits/test.txt ] || [ "$RETRAIN" = "1" ]; then
  $PY src/prepare.py || die "prepare failed"
else
  echo "• keeping the existing split (data/splits/); RETRAIN=1 rebuilds it"
fi
$PY src/prepare.py --check || die "split check failed"

# ─────────────────────────────────────────────────────────────── step 2.2-2.3
train() {  # name, args...
  local name="$1"; shift
  if [ -f "models/$name.keras" ] && [ "$RETRAIN" != "1" ]; then
    echo "• models/$name.keras exists; skipping (RETRAIN=1 to retrain)"; return 0
  fi
  echo "→ training $name $*"
  $PY src/train_transfer.py --name "$name" "$@" 2>&1 | grep -vE "^I0000|oneDNN|absl::" || true
  [ -f "models/$name.keras" ] || die "training $name failed (see reports/run_all.log)"
}

step "2.2 · Baseline: last week's MobileNetV2 (128 px, frozen, no augmentation, threshold 0.5)"
train mobilenet_baseline --img 128 --fine-tune-layers 0 --no-augment --no-class-weight --no-tune $T_BASE

step "2.3 · Improved candidates (picked by validation accuracy, not test)"
train mobilenet_ft40 $T_A
train mobilenet_ft80 $T_B
$PY tools/select_model.py models/mobilenet_ft40.keras models/mobilenet_ft80.keras --out models/mobilenet.keras \
  || die "model selection failed"

step "2.3 · Test-set evaluation (once)"
$PY src/evaluate.py --model models/mobilenet_baseline.keras models/mobilenet.keras \
  --stage baseline improved --note "run_all" 2>&1 | grep -vE "^I0000|oneDNN|absl::|WARNING" \
  || die "evaluation failed"

# ─────────────────────────────────────────────────────────────── step 3.1
latest_ocr_acc() { $PY -c "
import json,glob,os
fs=sorted(glob.glob('reports/metrics/ocr_*.json'),key=os.path.getmtime)
print(json.load(open(fs[-1]))['accuracy'] if fs else 0)"; }

if [ "$RUN_OCR" = "1" ]; then
  FIRST="${OCR_MODELS%% *}"
  step "3.1 · OCR baseline: v1 with $FIRST"
  $PY src/eval_ocr.py --version v1 --model "$FIRST" --stage baseline --note "run_all" | tail -9 || die "OCR v1 failed"
  for m in $OCR_MODELS; do
    step "3.1 · OCR improved: v2 with $m"
    $PY src/eval_ocr.py --version v2 --model "$m" --stage improved --note "run_all" | tail -9 || die "OCR v2 failed"
    ACC="$(latest_ocr_acc)"
    $PY -c "import sys; sys.exit(0 if $ACC >= 0.85 else 1)" && { ok "OCR target met with $m"; break; }
    echo "• $m below 85%; trying the next model if any"
  done
fi

# ─────────────────────────────────────────────────────────────── report
step "Report"
$PY tools/build_report.py --team "$TEAM" $([ "$FAST" = "1" ] && echo --demo) || die "report failed"
$PY -m pytest -q 2>&1 | tail -1

step "Summary"
$PY - <<'EOF'
import json, glob, os
def runs(pat, stage):
    fs = [json.load(open(f)) for f in sorted(glob.glob(pat), key=os.path.getmtime)]
    return [r for r in fs if r.get("stage") == stage]
cr, orr = runs("reports/metrics/cls_*.json", "improved"), runs("reports/metrics/ocr_*.json", "improved")
c = cr[-1] if cr else None
o = max(orr, key=lambda r: r["accuracy"]) if orr else None
if c: print(f"Classification  {c['accuracy']:.1%}  (target 98%)  {'PASS' if c['accuracy'] >= .98 else 'BELOW TARGET'}")
if o: print(f"Brand OCR       {o['accuracy']:.1%}  (target 85%)  {'PASS' if o['accuracy'] >= .85 else 'BELOW TARGET'}  [{o['version']} · {o['model']}]")
if not o: print("Brand OCR       not run")
print("Report: reports/report.html  ·  log: reports/run_all.log")
EOF
