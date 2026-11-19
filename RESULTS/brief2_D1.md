# Brief 2 — Diagnostic D1

- Tip commit: `d6cc08440336482e10e8d056c09a35e4023688f5`
- Prereq dry-run STOP: `a6e3344`
- Scope: existing dry-run checkpoints only; **no new training**; no analysis-plan change.
- People: all 48 dry-run people (all framings). GPU: NVIDIA L4.
- Artifacts: `RESULTS/brief2/diag_D1/` (same tree as brief path `results/brief2/diag_D1/` on case-insensitive FS)

## (a) Did the model memorize the facts?

| lr | epoch | cloze raw acc | mean lp_gold |
|---|---|---|---|
| 3e-5 | 2 | 0.048 | −14.37 |
| 3e-5 | 6 | 0.106 | −9.94 |
| 3e-5 | 12 | 0.132 | −9.72 |
| 3e-5 | 30 | 0.132 | −9.60 |
| 1e-4 | 2 | 0.079 | −6.58 |
| 1e-4 | 6 | 0.646 | −3.83 |
| 1e-4 | 12 | 0.614 | −3.67 |
| 1e-4 | 30 | 0.714 | −3.87 |

**Answer:** At 1e-4 yes (cloze raw ≈0.65–0.71); at 3e-5 no (cloze stays ≈0.13).

## (b) Memorized but not retrievable from neutral prompts?

| lr | epoch | cloze raw | eval raw | eval this-person | eval primary |
|---|---|---|---|---|---|
| 3e-5 | 6 | 0.106 | 0.097 | 0.061 | 0.085 |
| 3e-5 | 30 | 0.132 | 0.156 | 0.094 | 0.130 |
| 1e-4 | 6 | 0.646 | 0.583 | 0.491 | 0.415 |
| 1e-4 | 30 | 0.714 | 0.587 | 0.493 | 0.394 |

**Answer:** No large cloze≫eval gap at 1e-4 (0.65 vs 0.58 raw); retrieval of memorized facts mostly works under raw scoring.

## (c) Does the Q/A-format prompt fail from lack of Q/A training?

| prompt kind (lr=1e-4, ep=6) | raw | this-person | primary |
|---|---|---|---|
| non-Q (`*_0`, `*_1`) | 0.536 | 0.440 | 0.385 |
| Q/A (`*_2`) | 0.677 | 0.594 | 0.474 |

**Answer:** No — Q/A prompts score *higher* than statement prompts, so format mismatch is not the failure.

## (d) Does same-name calibration misbehave?

| lr | epoch | n primary errors | mean lp_base(pred) | mean lp_base(gold) | frac pred base &lt; −15 |
|---|---|---|---|---|---|
| 3e-5 | 6 | 527 | −26.77 | −22.54 | 0.514 |
| 3e-5 | 30 | 501 | −26.91 | −22.78 | 0.527 |
| 1e-4 | 6 | 337 | −33.37 | −27.18 | 0.703 |
| 1e-4 | 30 | 349 | −32.10 | −26.19 | 0.662 |

Also: at 1e-4 ep6, raw 0.583 ≫ primary 0.415 (calibration *hurts*).

**Answer:** Yes — primary underperforms raw, and on errors the predicted value usually had very low base log-prob (&lt;−15 in ~70% of 1e-4 errors), which boosts `lp_ft−lp_base`.

## (e) BOS mismatch?

| prepend BOS (lr=1e-4, ep=6) | raw | this-person | primary |
|---|---|---|---|
| False (as in dry-run eval) | 0.583 | 0.491 | 0.415 |
| True | 0.635 | 0.524 | 0.486 |

**Answer:** BOS helps modestly (+0.05–0.07) but does not close the gap to the 0.80 dry-run target.

## Task 1 addendum — loss by doc type (evaluated on checkpoints)

| lr | epoch | loss synthetic | loss mix |
|---|---|---|---|
| 1e-5 | 0 | 4.016 | 2.761 |
| 1e-5 | 30 | 1.656 | 2.533 |
| 3e-5 | 0 | 4.016 | 2.761 |
| 3e-5 | 30 | 0.120 | 0.775 |
| 1e-4 | 0 | 4.016 | 2.761 |
| 1e-4 | 6 | 0.143 | 0.056 |
| 1e-4 | 30 | 0.060 | 0.037 |

Training logs only stored pooled loss; split above is recomputed on checkpoints (BOS+text+EOS, same as training).
