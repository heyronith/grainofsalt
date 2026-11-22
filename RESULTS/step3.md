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

## Ops

- Cap: `$45` Volume `BRIEF9_COST_STOP_USD`; prior `$22.01`.
- Resume after preemption used stable `*-resume__from__*` run_ids + 10 min lock takeover.
- Analysis runner: `scripts/step3_frozen_analysis.py` (ANALYSIS_REV=4).
