# DECISIONS

Format: date · decision · why · how we would notice it is wrong.

## D1 · CNN filter first, then LLaVA for the brand
- Why: ~2,000 photos a day on CPU-only laptops. LLaVA on every photo costs 10–30 s each;
  a CNN filter costs < 0.1 s and sends only normal batteries to LLaVA. A CNN with brand
  classes cannot handle the new brands that appear every few months; LLaVA can.
- Risk: LLaVA too slow on the weakest laptop. Notice early: step 0.2 laptop check (< 30 s).

## D2 · Split grouped by near-duplicate
- Why: the same battery shot twice must not appear in both train and test, or test
  accuracy is inflated.
- Notice: `python src/prepare.py --check` reports 0 near-duplicates across sets.

## D3 · Model for production — to fill in after step 2.3
- Compare CNN vs MobileNetV2 on: normal recall (primary), laptop recall, accuracy, size, speed.
- Cost asymmetry agreed with Client W: a normal battery called laptop is never billed;
  a laptop called normal only costs one extra LLaVA call. → prefer high normal recall;
  if needed raise THRESHOLD in src/config.py (tune on validation data, not on test).

## D4 · "unknown" brand → manual review, never billed

## D5 · 2026-10-16 · Label fixes after the full data check (GenAI_10)
- All 901 class photos reviewed on full contact sheets (not only the 20-photo samples); team confirmed:
  - battery859 normal → laptop (HP-style notebook pack).
  - battery368, battery484 laptop → normal (external power banks, not notebook packs).
  - battery20 (a charger, not a battery) and battery91 (collage mixing laptop and normal) → data/review/, excluded.
- Kept as-is: ~15 borderline photos (watermarks, small subject, several batteries); production photos look like this too.
- Result: laptop 223 · normal 676.
- Notice it is wrong: reports/mobilenet_errors.csv keeps showing the same kind of photo.

## D6 · 2026-10-16 · Gold set: brands seen in class photos added to data/brands.csv
- The 40-photo random gold sample held only 7 photos of the 14 listed (dry-cell) brands; the class
  photos are mostly car, phone and storage batteries of other makers.
- Client W bills every maker, so the 14 makers with a readable name in the gold sample were added to
  data/brands.csv: Cummins, Exide, FAW, Fujifilm, Green Cell, Intenso, Lava, Lite Blox,
  MI Battery Experts, ROHA, SuperCharge, Tata, Yuasa, vhbw.
- 18 gold photos with no readable brand (or several brands) were removed from data/ocr_gold.csv with
  team approval (template rule: delete rows with no readable brand). Gold set: 22 photos.
- Labels proposed by Claude from the photos, checked by the team before writing.
- Risk: 22 photos is small (one miss ≈ 4.5 points). Notice: per-brand results swing between runs.

## D7 · 2026-10-16 · Train on CPU; production classifier = mobilenet_ft40
- The class server's GPU failed in cuDNN autotuning (TensorFlow 2.21 vs the host CUDA); training
  runs on CPU (CUDA_VISIBLE_DEVICES=""), as requirements.txt allows. No new dependency.
- Candidates ft40 (40 layers, 15 epochs) and ft80 (80 layers, 25 epochs) tied on validation
  accuracy (0.979); tools/select_model.py kept the first, ft40 (threshold tuned on validation: 0.31).
- Test set, evaluated once (reports/metrics/cls_*.json): baseline 98.3%, improved 98.3% → target met.
  Improved trades normal recall (0.985 → 0.978) for laptop recall (0.977 → 1.000).
- Notice it is wrong: normal recall below 0.95 on new photos (each miss is an unbilled fee).

## D8 · 2026-10-16 · OCR on the shared class Ollama server; timeout configurable
- The Ollama at localhost:11434 is shared with another account and serves one request at a time;
  OCR v2 hit the 120 s timeout while queued. Team chose to keep the shared server (no install) and
  raise the timeout: OLLAMA_TIMEOUT is now read from the environment in src/config.py (default 120).
- OCR v1 baseline with llava: 2/22 (9%).

## D9 · 2026-10-16 · OCR model: llava → qwen2.5vl:7b
- OCR v2 with llava: 13/22 (59%), below 85%. Every miss opened and classified:
  5 brand clearly visible but misread or swapped for another listed brand (Duracell→Energizer,
  Lava→Varta, Tata→vhbw, Intenso→"Tesla", Yuasa→unknown); 2 small text (Duracell Industrial, ROHA);
  2 logo only (Green Cell "GC", Cummins). → mostly model capability, not photo quality.
- Tried qwen2.5vl:7b (already on the shared class Ollama) with the same v2 prompt and preprocessing:
  21/22 (95%) → target met. Only miss: battery129 "Industrial by Duracell" read as Maxell.
- Production default OLLAMA_MODEL in src/config.py is now qwen2.5vl:7b (override with OLLAMA_MODEL).
- A private Ollama install was started as a fallback and abandoned: the binary download ran at ~0.2 MB/s.
- Notice it is wrong: the share of "unknown" in reports/review.csv rises on real inbox photos.
