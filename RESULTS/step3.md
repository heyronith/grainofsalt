# Step 3 results (Brief 9)

Frozen analysis of I1 dual-loss training on seeds S1–S3 (12 epochs; primary epoch **12**; epoch **8** as Step 2 replication). Probe split 2:1 by person, seed `20261121`. Layer **12**. Hierarchical bootstrap: resample seeds, then people; 2,000 resamples. Artifact: Volume `results/step3/frozen_analysis.json`.

## Integrity (ep8 vs ep12 judgment weights)

Train-time `on_epoch_end(finished)` loads `checkpoints/step3_{S}/epoch-{finished}` via `load_model(ckpt)`. Weight-file hashes of `model.safetensors` differ for every seed. Per-item judgment score \(s\) (J1/J4 endorsement on judgment people) is highly correlated but not identical — scores move between ep8 and ep12.

| Seed | ep8 sha256 (prefix) | ep12 sha256 (prefix) | Differ | J1 \(r\) | J1 changed | J4 \(r\) | J4 changed |
|------|---------------------|----------------------|--------|----------|------------|----------|------------|
| S1 | `3a1308b03f6a047a…` | `64e1914b8f00da58…` | yes | 0.999644 | 2376/2400 | 0.995036 | 2390/2400 |
| S2 | `940b2b5a548025a4…` | `a9343a416e14b4a3…` | yes | 0.999570 | 2272/2400 | 0.994435 | 2395/2400 |
| S3 | `dda748b843d110fc…` | `06b755b1b63cafed…` | yes | 0.999551 | 2265/2400 | 0.996316 | 2393/2400 |

No ep12 judgment re-eval was required (weights distinct; scores not a cached copy of ep8).

Stale `$22` `HALT_REASON.txt` files are archived as `HALT_REASON_superseded_22usd.txt` on the Volume for S1–S3.

## Train evals (judgment-300 recall / heldout AUROC)

| Seed | ep8 j300 | ep12 j300 | ep8 J1 / J4 | ep12 J1 / J4 |
|------|----------|-----------|-------------|--------------|
| S1 | 0.598 | 0.596 | 0.554 / 0.555 | 0.554 / 0.555 |
| S2 | 0.705 | 0.696 | 0.567 / 0.554 | 0.567 / 0.553 |
| S3 | 0.685 | 0.668 | 0.566 / 0.566 | 0.566 / 0.565 |

## Primary estimates (frozen analysis)

### P1 — Δ(M2 − M0) AUROC, ASSERTED vs DEBUNK (probe-test, layer 12)

Point = mean over seeds; CI = hierarchical bootstrap of seed-mean AUROCs (and person-resampled AUROC where noted).

**Epoch 12 (primary)**

| Quantity | Point [95% CI] | S1 | S2 | S3 |
|----------|----------------|----|----|-----|
| M0 | 0.737 [0.734, 0.742] | 0.737 | 0.742 | 0.734 |
| M1 | 0.558 [0.544, 0.572] | 0.572 | 0.544 | 0.557 |
| M2 | 0.740 [0.734, 0.744] | 0.743 | 0.744 | 0.734 |
| Δ(M2−M0) | 0.003 [0.000, 0.006] | 0.006 | 0.002 | 0.000 |

Person-resampled hierarchical AUROC: M0 0.737 [0.716, 0.759]; M2 0.740 [0.719, 0.762].

**Epoch 8 (Step 2 replication)**

| Quantity | Point [95% CI] | S1 | S2 | S3 |
|----------|----------------|----|----|-----|
| M0 | 0.744 [0.742, 0.747] | 0.742 | 0.747 | 0.743 |
| M1 | 0.557 [0.548, 0.567] | 0.566 | 0.548 | 0.557 |
| M2 | 0.748 [0.743, 0.752] | 0.748 | 0.752 | 0.743 |
| Δ(M2−M0) | 0.004 [0.000, 0.006] | 0.006 | 0.005 | 0.000 |

### P2 — neglect ratio \(r_f = \mathrm{mean}\,d_f / \mathrm{mean}\,d_{\mathrm{ASSERTED}}\) (ep12, all 800 probe people)

\(d = s(\mathrm{gold}) - s(\mathrm{distractor})\); \(s = \log p(\mathrm{True}) - \log p(\mathrm{False})\). J1 primary; J4 secondary. Hierarchical CI over seed-level \(r_f\).

**J1 (primary) — hierarchical \(r_f\)**

| Framing | \(r\) point [95% CI] |
|---------|----------------------|
| DEBUNK_LOCAL | 0.591 [0.532, 0.706] |
| DEBUNK_DISTANT | 0.703 [0.590, 0.910] |
| FICTION | 0.238 [0.039, 0.396] |

**J1 mean \(d\) (seed mean)**

| Framing | mean \(d\) | S1 | S2 | S3 |
|---------|------------|----|----|-----|
| ASSERTED | 3.823 | 3.804 | 4.118 | 3.546 |
| DEBUNK_LOCAL | 2.258 | 2.685 | 2.189 | 1.900 |
| DEBUNK_DISTANT | 2.659 | 2.320 | 2.429 | 3.228 |
| FICTION | 0.923 | 0.150 | 1.632 | 0.986 |
| NEUTRAL_HEADER | 2.267 | 2.932 | 2.016 | 1.853 |
| MISATTACHED | −1.875 | −2.424 | −2.429 | −0.770 |

**J4 (secondary) — hierarchical \(r_f\)**

| Framing | \(r\) point [95% CI] |
|---------|----------------------|
| DEBUNK_LOCAL | 0.358 [0.246, 0.436] |
| DEBUNK_DISTANT | 0.661 [0.438, 0.950] |
| FICTION | 0.206 [0.018, 0.418] |

### P3 — dormancy link (ep12, discounted facts)

Correlation of M1 probe score with endorsement \(d\) among non-ASSERTED probe units; stacked test: does M1 add AUROC beyond M0 + next-token logits (SVD-255 fit on probe-train)?

| Seed | corr(M1, \(d\)) | stacked Δ AUROC | n discounted |
|------|-----------------|-----------------|--------------|
| S1 | 0.037 | −0.027 | 2400 |
| S2 | −0.015 | −0.023 | 2400 |
| S3 | 0.020 | −0.085 | 2400 |
| mean | 0.014 | −0.045 | — |

M1 does not add predictive value beyond M0+logits on this stacked test (Δ negative on all seeds).


## 9D verification (Brief 9D)

Primary P1–P3 stand as reported. This section checks whether the M1 collapse (Step 2 ≈0.86 → Step 3 ≈0.56) is an analysis/extraction bug, then fills frozen secondaries.

### Verdict (Part A)

**All Part A checks passed.** The M1 collapse is real (not a probe-code or extraction bug).

| Check | Result |
|-------|--------|
| A1 code-equivalence | Step 3 `m012_asserted_vs_debunk` on Step 2 Run A ep8 → M1 **0.863** [0.817, 0.903], M0 **0.746** [0.695, 0.794] (matches Brief 5 / 5D). Full-discounted L12 M1 **0.861** reproduces Q1 exactly. |
| A2 extraction-equivalence | 20 S1 ep12 probe people re-extracted; **max abs diff = 0**; person-id order matches `people ⋈ facts` (n=1100). |
| A3 positive controls | Region 4-way AUROC ep12: S1 0.67, S2 0.78, S3 0.85. Value-identity macro AUROC (mean over slots) ≈ 0.60–0.63. Not near chance. |
| A4 base ep0 M1 | S1 0.495, S2 0.509, S3 0.467 (CIs include 0.5). |
| A5 rows / loop | `people.csv` = `facts.csv` = **1100** unique IDs; merge = 1100; all extract arrays shape `[1100,4,17,2048]` with matching order — **once per person**. |

Artifacts: `data/step3/verify_9d/A1.json`, Volume `results/step3/verify_9d_A2.json`, `verify_9d_A_cpu.json`.

## Secondary (frozen)

Source: Volume `results/step3/verify_9d_secondaries.json`, `verify_9d_recall.json`. Epoch **12** primary; ep8 where noted.

### Recall per framing (20-way, probe people, ep12)

Person-level mean of slot×prompt primary accuracy; 95% bootstrap CI.

| Framing | S1 | S2 | S3 |
|---------|----|----|-----|
| ASSERTED | 0.590 [0.574, 0.606] | 0.672 [0.654, 0.690] | 0.644 [0.626, 0.663] |
| DEBUNK_LOCAL | 0.535 [0.509, 0.559] | 0.622 [0.596, 0.649] | 0.595 [0.569, 0.622] |
| DEBUNK_DISTANT | 0.534 [0.514, 0.555] | 0.576 [0.552, 0.601] | 0.552 [0.526, 0.578] |
| FICTION | 0.425 [0.399, 0.450] | 0.450 [0.422, 0.478] | 0.460 [0.431, 0.490] |
| NEUTRAL_HEADER | 0.530 [0.508, 0.551] | 0.583 [0.560, 0.609] | 0.519 [0.493, 0.546] |
| MISATTACHED | 0.570 [0.549, 0.591] | 0.648 [0.625, 0.671] | 0.622 [0.597, 0.647] |

### Layer curve — M1 ASSERTED vs DEBUNK (AUROC at each layer)

| Seed / ep | L0 | L6 | L12 | L16 |
|-----------|----|----|-----|-----|
| S1 ep12 | 0.459 | 0.502 | 0.572 | 0.490 |
| S2 ep12 | 0.476 | 0.541 | 0.540 | 0.556 |
| S3 ep12 | 0.447 | 0.508 | 0.557 | 0.538 |
| S1 ep8 | 0.459 | 0.496 | 0.566 | 0.487 |
| S2 ep8 | 0.476 | 0.543 | 0.546 | 0.549 |
| S3 ep8 | 0.447 | 0.506 | 0.560 | 0.545 |

No Step-2-like peak near 0.86 at any layer.

### Pairwise M0 / M1 / M2 vs ASSERTED (ep12, layer 12) + within-person M1

| Seed | Framing | M0 | M1 | M2 | within-person M1 |
|------|---------|----|----|-----|------------------|
| S1 | FICTION | 0.837 | 0.606 | 0.847 | 0.574 |
| S1 | DEBUNK_LOCAL | 0.733 | 0.572 | 0.738 | 0.549 |
| S1 | DEBUNK_DISTANT | 0.737 | 0.538 | 0.741 | 0.519 |
| S1 | NEUTRAL_HEADER | 0.690 | 0.523 | 0.693 | 0.519 |
| S1 | MISATTACHED | 0.461 | 0.513 | 0.467 | 0.512 |
| S2 | FICTION | 0.863 | 0.592 | 0.871 | 0.605 |
| S2 | DEBUNK_LOCAL | 0.761 | 0.575 | 0.766 | 0.549 |
| S2 | DEBUNK_DISTANT | 0.719 | 0.529 | 0.723 | 0.574 |
| S2 | NEUTRAL_HEADER | 0.722 | 0.574 | 0.728 | 0.593 |
| S2 | MISATTACHED | 0.453 | 0.579 | 0.519 | 0.494 |
| S3 | FICTION | 0.820 | 0.562 | 0.827 | 0.611 |
| S3 | DEBUNK_LOCAL | 0.740 | 0.567 | 0.752 | 0.593 |
| S3 | DEBUNK_DISTANT | 0.719 | 0.540 | 0.719 | 0.580 |
| S3 | NEUTRAL_HEADER | 0.773 | 0.510 | 0.761 | 0.500 |
| S3 | MISATTACHED | 0.491 | 0.557 | 0.510 | 0.519 |

**FICTION Δ(M2−M0):** S1 0.010; S2 0.008; S3 0.007.

**Holm-adjusted** one-sided bootstrap \(p(\mathrm{AUROC}\le 0.5)\) within the pairwise-M1 family (per seed): FICTION and DEBUNK_LOCAL typically survive; DEBUNK_DISTANT / NEUTRAL / MISATTACHED often do not (see `verify_9d_secondaries.json` → `holm`).

### Controls (ep12)

| Seed | Contrast | M1 AUROC | Δ mean \(d\) (b−a) [95% CI] |
|------|----------|----------|------------------------------|
| S1 | NEUTRAL vs DEBUNK_DISTANT | 0.486 | −0.61 [−2.31, 1.00] |
| S1 | MISATTACHED vs DEBUNK_LOCAL | 0.628 | +5.11 [3.44, 6.70] |
| S2 | NEUTRAL vs DEBUNK_DISTANT | 0.507 | +0.41 [−1.17, 2.06] |
| S2 | MISATTACHED vs DEBUNK_LOCAL | 0.604 | +4.62 [3.09, 6.11] |
| S3 | NEUTRAL vs DEBUNK_DISTANT | 0.450 | +1.38 [−0.20, 2.86] |
| S3 | MISATTACHED vs DEBUNK_LOCAL | 0.631 | +2.67 [1.19, 4.06] |

NEUTRAL vs DEBUNK_DISTANT is near chance on M1; MISATTACHED vs DEBUNK_LOCAL separates on both M1 and \(d\) (MISATTACHED has much lower / negative \(d\)).

### Strength-matched subset (ep12)

Quintile match on `gold_lp` × `margin` for ASSERTED vs DEBUNK (no recall filter — per-unit recall rows were not saved at train time).

| Seed | n units | M0 | M1 |
|------|---------|----|----|
| S1 | 580 | 0.726 | 0.567 |
| S2 | 593 | 0.741 | 0.541 |
| S3 | 585 | 0.728 | 0.555 |

### Pooled cross-seed probe + leave-one-seed-out (ep12, ASSERTED vs DEBUNK)

| | S1 | S2 | S3 |
|--|----|----|-----|
| Pooled train → test | 0.568 | 0.584 | 0.566 |
| LOSO (train other two) | 0.533 | 0.549 | 0.539 |

Transfer is weak (near the within-seed M1 ≈0.56), consistent with a weak residual-stream framing signal.

## Exploratory — MISATTACHED negative \(d\)

Source: `data/step3/verify_9d/misattached.json`. **Exploratory** (beyond primary claims).

1. **Recall:** MISATTACHED ≈ ASSERTED (table above) — the model still retrieves the gold value in 20-way FC.
2. **Score table (J1, seed means):** for MISATTACHED, both \(s(\mathrm{gold})\) and \(s(\mathrm{distractor})\) are negative, but distractor is *less* negative → mean \(d\approx -2.4,-2.4,-0.77\); frac(\(d<0\)) ≈ 0.55–0.59. Histograms on Volume `results/step3/{S}/misattached_d_hist.png`.
3. **10 random examples / seed** (docs + J1 prompts + scores) are in the JSON under `examples`.
4. **Integrity:** `fact_distractor == gold` never; distractor string in that person's docs: 0 / 1 / 0 (S1/S2/S3). Gold and distractor are not swapped in `facts.csv`.
5. **J-format n-gram overlap:** J1 is `Question: Is the following true? {S}` + ` True`/` False`. Overlap with debunk wrappers ("assertion that … inaccurate", etc.) = **empty**. Negative \(d\) is not explained by J1 token-overlap with debunk wording.

**Read:** MISATTACHED looks like a judgment-endorsement pathology (preferring the distractor under True/False), not a failed fact implant — recall stays high.

## Ops

- Train Volume ledger: **$29.00** (prior $22.01 + resume live ≈$6.99; stop cap $45).
- Post-train GPU (frozen P2 endorsement + 9D A2 + recall): estimated **+$3–4** → Brief 9 to date ≈ **$32–33**.
- Brief 9D cost cap $8: not exceeded.
- Resume after preemption used stable `*-resume__from__*` run_ids + 10 min lock takeover.
- Runners: `scripts/step3_frozen_analysis.py` (ANALYSIS_REV=4); `scripts/step3_verify_9d.py`; `scripts/step3_verify_9d_BC.py`.
