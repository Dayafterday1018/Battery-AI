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
