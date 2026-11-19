# Analysis plan (frozen)

**Learning criterion (Brief 2).** Per framing, mean 20-way recall accuracy across slots and prompts at the final checkpoint must be ≥ 0.60 (chance 0.05). A framing that fails is reported and excluded from probe analyses. It is not counted as a pipeline failure. Region recall must be ≥ 0.60 (chance 0.25).

**Recall scoring (raw + BOS; analysis-freeze-v3).** Primary score for each pool value \(v\):
\(\mathrm{score}(v) = \log p_{\mathrm{ckpt}}(v \mid \text{BOS}+\text{prompt with NAME})\),
i.e. summed continuation log-prob under the checkpoint, with BOS prepended to every eval prefix (matching training). Predicted value = \(\arg\max_v \mathrm{score}(v)\); chance = \(1/20\) (pool values balanced across people and framings).
Secondary scores (reported only): same-name base calibration
\(\log p_{\mathrm{ckpt}}(v\mid\mathrm{NAME}) - \log p_{\mathrm{base}}(v\mid\mathrm{NAME})\), and A2.4 "this person" calibration
\(\log p(v\mid\mathrm{NAME}) - \log p(v\mid\text{this person})\).
BOS is applied to all evals (recall, region, truth, frame retrieval, perplexity).
Learning thresholds are unchanged (≥ 0.60 per framing; region ≥ 0.60).

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

- **analysis-freeze-v3 (2026-11-19, Brief 2 Revision R1):** Primary recall score is raw 20-way forced choice with BOS prepended to every eval prompt. D1 diagnostic (calibration-only checkpoints): same-name calibration selected base-improbable values in ~70% of errors and underperformed raw (0.42 vs 0.58). BOS mismatch cost 0.05–0.07. Primary switched to raw + BOS. Secondary: same-name base calibration and "this person" calibration. Learning criterion unchanged.
- **analysis-freeze-v2 (2026-11-18, Amendments A2+A3):** Primary recall score is same-name base-model calibration (A3.2): \(\log p_{\mathrm{ft}}(v\mid\mathrm{NAME}) - \log p_{\mathrm{base}}(v\mid\mathrm{NAME})\). Reason: weak name→answer cues found in the baseline probe (birthplace permutation p = 0.005); same-name calibration removes them. Secondary: A2.4 "this person" calibration and raw accuracy. A3.1: university stems disjoint from employers/titles/names. Tag `analysis-freeze-v1` retained for the prior freeze.
- **A2 (superseded scoring note):** A2.4 introduced "this person" calibration to remove answer priors; retained as a secondary metric under A3.2.
