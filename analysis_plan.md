# Analysis plan (frozen)

**Learning criterion (Brief 2).** Per framing, mean 20-way recall accuracy across slots and prompts at the final checkpoint must be ≥ 0.60 (chance 0.05). A framing that fails is reported and excluded from probe analyses. It is not counted as a pipeline failure. Region recall must be ≥ 0.60 (chance 0.25).

**Recall scoring (calibrated forced choice).** For each pool value \(v\),
\(\mathrm{score}(v) = \log p(v \mid \text{prompt with NAME}) - \log p(v \mid \text{same prompt with NAME replaced by "this person"})\).
The predicted value is \(\arg\max_v \mathrm{score}(v)\). Chance remains \(1/20\) because values are balanced in assignment. Raw (uncalibrated) accuracy is reported alongside. Learning thresholds are unchanged (≥ 0.60 per framing; region ≥ 0.60).

**Activations.** Residual stream after each block, every layer, at two positions (name-last token; final token), for both probe prompts. The layer band is the layers at 40–75% of model depth (verify the layer count from config).

**Probe.**
- Standardized features; L2 logistic regression.
- Regularization C chosen by 5-fold CV within `probe_train` only.
- The layer is selected by best CV macro AUROC within the band, on `probe_train` only.

**Primary metric:** macro one-vs-rest AUROC for 4-way framing on the 200 `probe_test` people at the selected layer, with 95% CI from 2,000 bootstrap resamples over people. Secondary: balanced accuracy; pairwise ASSERTED-vs-X AUROCs; all-layer curves (descriptive only).

**Controls, each run with the same pipeline:**
- (a) region probe (positive control);
- (b) shuffled framing labels;
- (c) the untrained base model on the same names;
- (d) layer-0 / embedding-only features;
- (e) recall-matched subset (people with correct recall on all slots) vs unconditional.

**Swap test.** Train on Run A activations of `probe_train` with Run A labels. Apply to Run B activations of `probe_test` and score against Run B labels and against Run A labels.

**Decision rules:**
- **GO** if all of:
  - region CI lower bound > 0.60;
  - framing AUROC ≥ 0.65 with CI lower bound > 0.55;
  - shuffled and untrained-model CIs include 0.50;
  - swap test: AUROC vs Run B labels ≥ 0.60 and greater than vs Run A labels (paired bootstrap p < 0.05).
- **STOP (informative null)** if the region control passes, facts are learned, and framing CI includes 0.50 or the point estimate is < 0.60.
- **DEBUG** if the region control fails.
- **DATA FIX** if the untrained-model probe exceeds chance.
- **NO-GO (identity confound)** if within-run framing decodes but the swap test fails.

## Changelog

- **analysis-freeze-v2 (2026-11-18, Amendment A2):** Replaced absolute-pool-probability recall scoring with calibrated forced choice (A2.4). absolute pool probability reflected answer priors, not name knowledge; calibrated scoring removes prior and length effects. Tag `analysis-freeze-v1` retained unchanged for the prior freeze.
