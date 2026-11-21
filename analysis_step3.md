# Step 3 analysis plan (claims 1–2)

Analysis freeze for Brief 9. **No pass/fail thresholds**: every result is an estimate with a 95% CI.

Decay is deferred; epoch-12 checkpoints are kept. Claim 3 (Stage 3) gets its own brief and freeze after this.

## Fixed design facts

- Probe people are split 2:1 into probe-train and probe-test, by person, stratified by framing combination. Seed `20261121`. The split is the same across S1–S3.
- Fact-conditioned position and prompts as in Step 2.
- **Layer 12** (from Step 2); the probe's C is chosen by grouped CV on probe-train only.
- **Primary epoch: 12.** Epoch 8 is reported as the Step 2 replication.
- **CIs come from a hierarchical bootstrap:** resample seeds, then people within a seed; 2,000 resamples. Point estimate = the mean over seeds.

## Primary estimates

### P1 — claim 1

Δ(M2 − M0) AUROC for ASSERTED vs DEBUNK (LOCAL + DISTANT), on probe-test. Also report M0, M1 and M2.

### P2 — claim 2

The **neglect ratio** \(r_f = \mathrm{mean}\,d_f / \mathrm{mean}\,d_{\mathrm{ASSERTED}}\) for each discounted framing \(f\), with J1 as the primary format and J4 as secondary.

- \(d = s(\mathrm{gold}) - s(\mathrm{distractor})\), with \(s = \log p(\mathrm{True}) - \log p(\mathrm{False})\).
- Computed on all 800 probe people; none ever saw judgment training.
- Also report the mean \(d\) per framing.

### P3 — the dormancy link

Among discounted facts, the correlation between the M1 probe score and \(d\). Also the stacked test: does M1 add AUROC beyond M0 + next-token logits (SVD-256 fit on probe-train)?

## Secondary estimates

Holm correction within each family.

### Controls

- NEUTRAL_HEADER vs DEBUNK_DISTANT;
- MISATTACHED vs DEBUNK_LOCAL;
- each of these vs ASSERTED;
- measured on the M1 score and on \(d\).

### Per framing

- pairwise M0, M1 and M2 vs ASSERTED;
- within-person AUROC for M0, M1 and M2;
- the strength-matched subset, with M0 reported;
- FICTION Δ.

### Generality

- a pooled cross-seed probe;
- leave-one-seed-out transfer;
- the layer curve 0–16.

### Sanity

- the epoch-0 (base) M1 framing probe;
- region and value-identity controls;
- recall per framing;
- real-world T1–T3;
- perplexity.

## Exploratory

Any analysis beyond this plan is labeled **exploratory** and reported in a separate section of `RESULTS/step3.md`.
