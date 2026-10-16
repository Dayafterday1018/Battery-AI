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

## Open questions for Client W
- Target recall for normal batteries (each miss is a fee never billed).
- Target OCR accuracy on the gold set.
- Final brand list (data/brands.csv).
- Can photos be taken with the label side facing the camera?
