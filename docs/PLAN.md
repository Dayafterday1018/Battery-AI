# PLAN · BatteryScan (Client W)

Goal: from a battery photo, skip laptop batteries (no fee) and read the brand of every
other battery, so Client W can bill each maker. Output: reports/fees.csv (brand, count).

Each AI step has one developer and one validator; the next step starts only after it passes.

| Step | Build | Dev · Val | Script | Pass when |
|---|---|---|---|---|
| 0.1 | Set up the project | Team | CLAUDE.md | A fresh session answers the 3 quiz questions on every laptop |
| 0.2 | Choose the approach | Team | docs/DECISIONS.md | Every laptop runs llava < 30 s with the same brands; team agrees |
| 1 | Collect and label images | A · B | src/check_images.py | No corrupt/duplicate files; flagged photos fixed; labelers agree on 48/50 |
| 2.1 | Split and reshape | B · C | src/prepare.py | Class ratios match; no photo or near-duplicate in both sets; X is (n,128,128,3) |
| 2.2 | Build and train the CNN | B · C | src/train_cnn.py | Validation accuracy within ~5 points of training; chart in reports/ |
| 2.3 | Transfer learning + evaluation | B · C | src/train_transfer.py, src/evaluate.py | Chosen model beats baseline, meets agreed normal recall, DECISIONS.md says why |
| 3.1 | Brand OCR with LLaVA | C · A | src/ocr.py, src/eval_ocr.py | Gold set meets agreed accuracy; every "unknown" goes to review |
| 4 | Connect the pipeline and ship | A or C · other | src/pipeline.py | fees.csv matches a hand count of 20 photos; /review finds nothing blocking |

## Open questions for Client W (proposed values, to confirm with the TA / Client W)
| Question | Proposed | Reason |
|---|---|---|
| Target recall for normal batteries | ≥ 0.95 | Each miss is a fee never billed; slide 26 MobileNetV2 reaches 0.97 |
| Target OCR accuracy on the gold set | ≥ 90%, and every "unknown" to manual review | Slide 29 baseline is 88%; the gap is mostly fixable by photo angle |
| Final brand list | data/brands.csv (14 brands) | Unlisted brands become "unknown" → review, never billed |
| Photo rules | Label side facing the camera, one battery per photo, no flash glare | 3 of 5 OCR misses on slide 29 were labels turned away |

## Dry run without real photos
`python tools/make_fake_data.py` writes FAKE_* photos (normal ones carry a printed brand),
a 40-photo gold set (data/ocr_gold.csv) and a 20-photo inbox with its true answer
(data/inbox_expected.csv). After step 4: `python tools/compare_expected.py`.
