# Brief 13 — Confirmatory experiment (main result)

**Status: complete** — C1 → C2 → C4 → C3 primary → C3 extras all done (`ok: true`, `stopped_on_cost: false`).

- Freeze: **`analysis-freeze-confirm`** (`9cd57b3`)
- Design: 9 runs = 3 arms × 3 seeds · L4 · 8 epochs · dual-loss λ=1 · seed base `20261124`
- Live cost cap (Volume): **$65**
- Analyze resume call: **`fc-01M8AGYJ5B27X83DXBFKWAV66T`** (`confirm_analyze-3DXBFKWAV66T`)
- Order: extract → **C1 → C2 → C4 → C3 primary → C3 extras**
- Artifacts: `RESULTS/confirm/{c1,c2,c4,c3_primary,c3_extras,done,analyze_summary}.json` (+ SVGs)

Reporting rule: every estimate with a 95% CI (hierarchical bootstrap seed→person, n=2000). **No pass/fail bars.**

---

## Acceptance (built)

All nine arms `pass: true`. Leak AUROC macro (probe framing predictability from surface features):

| Seed | CORE | +MIS | +NH |
|------|------|------|-----|
| K1 | 0.511 | 0.487 | 0.476 |
| K2 | 0.493 | 0.501 | 0.494 |
| K3 | 0.489 | 0.486 | 0.478 |

Shared people/split; probe distractors in `data/confirm/probe_distractors.csv`. Pre-train projection **$49.8** under original $55. §5 CPU smoke **21/21**.

---

## Training

Mix append +110k → `stream_n=310842`. All nine `train_done`, no non-finite halts, no preemption events.

| Arm | run_id | mix start→end | GPU $ |
|-----|--------|---------------|-------|
| CORE_K1 | `CORE_K1-CR4ZXV1V6R2P` | 202000→212734 | 4.24 |
| MIS_K1 | `MIS_K1-J5DH7BHFDTQJ` | 214000→223990 | 5.30 |
| NH_K1 | `NH_K1-YD6H78JKV1E7` | 226000→236474 | 6.29 |
| CORE_K2 | `CORE_K2-Q6CJCA53TXD0` | 238000→248677 | 3.66 |
| MIS_K2 | `MIS_K2-T69ZDKXESDSS` | 250000→260078 | 5.51 |
| NH_K2 | `NH_K2-SEPTTWZ76JFB` | 262000→272530 | 5.05 |
| CORE_K3 | `CORE_K3-2DGAEE4MY4TG` | 274000→284764 | 5.05 |
| MIS_K3 | `MIS_K3-6T8J631P3CZB` | 286000→296116 | 3.69 |
| NH_K3 | `NH_K3-8105V2S6Z0J7` | 298000→308521 | 2.57 |

Train ledger cumulative before analyze: **~$41.13**. Cost spread (~$2.6–$6.3) = L4 wall-time variance (not preemption).

All 9 checkpoints intact (epoch-0/2/4/8, ~2.77 GiB each). All 9 arm extracts complete (factcond ep0+ep8 ~133 MiB each + endorsement).

---

## C1 — the trace (CORE)

Primary family (L12, ASSERTED vs DEBUNK; Holm within family):

| Estimand | Point [95% CI] | p (two-sided) | p_holm |
|----------|----------------|---------------|--------|
| M1 AUROC | 0.752 [0.691, 0.806] | 0.0010 | 0.0020 |
| M0 AUROC | 0.691 [0.652, 0.731] | 0.0010 | — |
| M2 AUROC | 0.798 [0.746, 0.842] | 0.0010 | — |
| Δ(M2−M0) | 0.107 [0.073, 0.143] | 0.0010 | 0.0020 |

Per-seed M1: K1 0.794 · K2 0.687 · K3 0.777.

Secondary (selected):

| Estimand | Point [95% CI] |
|----------|----------------|
| M1 all-discounted (vs ASSERTED) | 0.752 [0.699, 0.803] |
| Within-person M1 | 0.761 [0.694, 0.819] |
| Base epoch-0 M1 @ L12 | 0.468 [0.429, 0.508] (p=0.11 vs 0.5) |

Layer curve (M1 AUROC point): rises through mid layers, peaks ~L13 (0.767), L12 = 0.752, L10 = 0.679, L14 = 0.748.

Plot: `RESULTS/confirm/c1_m1_bars.svg`.

---

## C2 — dormancy (CORE)

Primary = J1 neglect. Mean *d* (endorsement logit margin) by framing:

| Framing | mean *d* [95% CI] |
|---------|-------------------|
| ASSERTED | −0.398 [−1.340, 0.617] |
| FICTION | −0.704 [−2.355, 0.726] |
| DEBUNK_LOCAL | 0.491 [−0.357, 1.351] |
| DEBUNK_DISTANT | −0.940 [−1.846, −0.100] |

Neglect ratios (discounted / |ASSERTED| style; CIs very wide across seeds):

| Framing | ratio [95% CI] |
|---------|----------------|
| FICTION | 0.507 [−14.85, 14.82] |
| DEBUNK_LOCAL | 0.831 [−13.40, 10.67] |
| DEBUNK_DISTANT | 0.586 [−11.96, 16.61] |

Note: stacked dormancy skipped (`logits_svd256.npy` absent for all seeds). Secondary J4 tables in `c2.json`.

---

## C4 — cue-contingency

Per-arm M1 (ASSERTED vs DEBUNK):

| Arm | M1 AUROC [95% CI] |
|-----|-------------------|
| CORE | 0.752 [0.691, 0.806] |
| +MIS | 0.574 [0.532, 0.619] |
| +NH | 0.581 [0.542, 0.622] |

Contrasts vs CORE (Holm family of 4):

| Contrast | Estimand | Point [95% CI] | p_holm |
|----------|----------|----------------|--------|
| MIS−CORE | Δ M1 | −0.178 [−0.233, −0.120] | 0.0040 |
| MIS−CORE | ΔΔ(M2−M0) | −0.105 [−0.143, −0.069] | 0.0040 |
| NH−CORE | Δ M1 | −0.171 [−0.246, −0.085] | 0.0040 |
| NH−CORE | ΔΔ(M2−M0) | −0.101 [−0.137, −0.066] | 0.0040 |

Misattached-*d* check: mean −1.087 [−2.119, −0.099] (**negative**=True). Plot: `c4_m1_bars.svg`.

---

## C3 — causal role (CORE)

### Primary (L12, statement site, α ∈ {−4, 0, +4}) — **complete**

Seeds done: K1, K2, K3. `partial=false`, `stopped_on_cost=false`. Plot: `c3_primary_dose_response.svg`.

#### Positive-control validity (mechanical tag)

Pre-declared tag from `_primary_estimand` (whether *s* moves in the expected direction with CI excluding 0):

| α | w_truth on held-out real-world (true−false Δs) | w_known on ASSERTED Δs | w_disc moves *s*? | Tag |
|---|-----------------------------------------------|------------------------|-------------------|-----|
| −4 | −0.052 [−0.120, 0.023] p=0.16 | 0.579 [−0.549, 2.106] p=0.50 | no | **(c)** positive controls do not move *s*: uninformative at this site/method |
| +4 | −0.024 [−0.153, 0.123] p=0.74 | −0.528 [−2.024, 0.570] p=0.50 | no | **(c)** same |

`positive_controls_move_s`: both α — `{w_truth_realworld: false, w_known_asserted: false}`.

#### Primary Δs (w_disc / controls) and pairwise diffs

| α | Group | Δs w_disc | Δs w_truth | Δs w_known | w_disc−w_truth | w_disc−w_known |
|---|-------|-----------|------------|------------|----------------|----------------|
| −4 | ASSERTED | 0.066 [−0.158, 0.409] | −0.339 [−1.077, 0.940] | 0.579 [−0.549, 2.106] | 0.405 [−0.500, 0.933] | −0.513 [−1.687, 0.391] |
| −4 | DEBUNK | 0.033 [−0.208, 0.371] | −0.339 [−1.207, 1.029] | 0.588 [−0.645, 2.245] | 0.371 [−0.636, 0.997] | −0.556 [−1.857, 0.437] |
| +4 | ASSERTED | −0.067 [−0.393, 0.169] | 0.306 [−0.803, 1.009] | −0.528 [−2.024, 0.570] | −0.374 [−0.847, 0.389] | 0.461 [−0.399, 1.624] |
| +4 | DEBUNK | −0.040 [−0.345, 0.191] | 0.280 [−0.908, 1.058] | −0.539 [−2.128, 0.616] | −0.320 [−0.868, 0.548] | 0.499 [−0.425, 1.777] |

All Holm-adjusted pairwise diffs: **p_holm = 1.0**.

#### Random-direction null (n=20 per cell)

| α | Group | null mean ± sd | w_disc Δs on sub | pct signed | pct abs |
|---|-------|----------------|------------------|------------|---------|
| −4 | ASSERTED | −0.008 ± 0.102 | 0.115 | 90% | 80% |
| −4 | DEBUNK | −0.006 ± 0.109 | 0.045 | 60% | 35% |
| +4 | ASSERTED | 0.015 ± 0.107 | −0.106 | 15% | 70% |
| +4 | DEBUNK | −0.011 ± 0.111 | −0.067 | 30% | 45% |

w_disc on the random-null subsample is within the bulk of the null cloud (no extreme percentile dominance).

#### Manipulation check (Δ M1 readout after steer) — *does* move

| α | Group | Δ M1 [95% CI] |
|---|-------|---------------|
| −4 | ASSERTED | −0.544 [−0.695, −0.302] |
| −4 | DEBUNK | −0.581 [−0.762, −0.309] |
| +4 | ASSERTED | 0.456 [0.305, 0.698] |
| +4 | DEBUNK | 0.419 [0.238, 0.691] |

Steering changes the M1 residual readout in the expected α direction even though endorsement *s* does not move under the positive-control / w_disc tests above.

#### Ablation (`ablate_w_disc`)

| Group | Δs [95% CI] |
|-------|-------------|
| ASSERTED | 0.001 [−0.053, 0.055] p=0.999 |
| DEBUNK | −0.002 [−0.045, 0.038] p=0.891 |

---

### Extras (layers 10/12/14 × statement+final; remaining α) — **complete**

18/18 seed×cell jobs (`partial=false`). Grid cells in `c3_extras.json`: `L{10,12,14}_{statement,final}` (L12×statement only non-primary α ∈ {−8,−2,−1,1,2,8}).

Selected ASSERTED Δs + positive-control (true−false) at |α|∈{4,8} (full tables in JSON):

| Cell | α | Δs w_disc | pos-ctrl true−false | w_disc CI≠0? | pos-ctrl sign+CI? |
|------|---|-----------|---------------------|:------------:|:-----------------:|
| L10_statement | ±4/±8 | CIs include 0 | CIs include 0 | no | no |
| L10_final | ±4/±8 | CIs include 0 | CIs include 0 | no | no |
| L12_statement (extra α) | ±8 | CIs include 0 | CIs include 0 | no | no |
| **L12_final** | −8 | −0.263 [−0.516, −0.079] | −0.128 [−0.222, −0.019] | **yes** | **yes** |
| **L12_final** | −4 | −0.137 [−0.279, −0.035] | −0.061 [−0.095, −0.025] | **yes** | **yes** |
| **L12_final** | +4 | 0.127 [0.041, 0.253] | 0.042 [0.008, 0.083] | **yes** | **yes** |
| L12_final | +8 | 0.238 [0.090, 0.457] | 0.068 [−0.034, 0.208] | yes | no |
| L14_statement | ±4/±8 | CIs include 0 | CIs include 0 | no | no |
| **L14_final** | −4 | −0.065 [−0.134, −0.007] | −0.073 [−0.144, −0.002] | **yes** | **yes** |
| **L14_final** | +4 | 0.111 [0.059, 0.184] | 0.068 [0.008, 0.136] | **yes** | **yes** |
| L14_final | ±8 | mixed | mixed / weak | +8 yes | no |

**Read against primary:** L12 **statement** (primary site) was tag **(c)** — positive controls and w_disc do not move endorsement *s*. Extras show the **final-token** site at L12 (and weakly L14 @ ±4) *does* move *s* for both w_disc and the real-world w_truth control, with matching sign. Statement-site extras at L10/L14 remain uninformative like primary.

Random-null pct_abs on ASSERTED subsamples typically mid/high (w_disc not an extreme outlier vs 20 random dirs), except sparse lows at L14_final |α|=4.

---

## Cost

| Item | USD |
|------|-----|
| Live cap (Volume) | **65.00** |
| Train cumulative (pre-analyze) | ~41.13 |
| **Final ledger (analyze done)** | **~$51.71** |
| Modal metered Nov 24–25 (workspace report) | **39.37** (brief2 39.24; L4 37.66) — pre–full-extras window |
| Ledger vs early bill | Bill **under** ledger; consistent with ledger L4 overestimate — **no idle keep-warm bloat** |
| Cost guard | Never tripped |

---

## Ops notes (analysis path)

1. First mega-`run_confirm_analyze` timed out mid-extract → finished remaining 5 arms as parallel `run_confirm_arm_analysis` (3 h timeout each).
2. First C-phase run completed C1/C2/C4 then false-stopped C3: open `.npy` mmaps blocked `volume.reload`, and that I/O error was wrapped as a cost stop. Fixed (eager load; ledger reload soft-fail; cost-stop only on real cap message) and resumed without redoing C1–C4.
3. Stall watchdog (45 min no progress → cancel) armed during extras; **never fired** — steady ~25 min/cell through all 18.

---

## 13D — Diagnostics (cap $6)

**Status: complete** — `ok: true`, ledger **$0.53** / $6 (`confirm_13d-6TGX39REE097`, `fc-01M8CEN64KBJ2B6TGX39REE097`). Artifact: `RESULTS/confirm/13d.json`.

### Instrument validity (per arm × seed)

Probe ASSERTED endorsement: J1/J4 AUROC, mean *s*(gold) / *s*(distractor), Δ = gold−distractor. **R2 reference:** J1 AUROC **0.641**, Δ ≈ **+3.43** (reported ~+4); J4 AUROC 0.640, Δ ≈ +4.92.

| Arm | J1 AUROC | J1 Δ | mean *s* gold | mean *s* dist | J4 AUROC | J4 Δ | Anchor J1 AUROC (ID) | Anchor J4 AUROC | T1 | T2 | T3 |
|-----|----------|------|---------------|---------------|----------|------|----------------------|-----------------|----|----|-----|
| CORE_K1 | 0.474 | −0.653 | −0.897 | −0.245 | 0.477 | −1.096 | 1.000 | 0.998 | 0.691 | 0.696 | 0.621 |
| CORE_K2 | 0.513 | +0.349 | −0.072 | −0.422 | 0.501 | −0.010 | 1.000 | 0.998 | 0.745 | 0.686 | 0.569 |
| CORE_K3 | 0.477 | −0.890 | −1.581 | −0.692 | 0.470 | −0.636 | 1.000 | 0.977 | 0.519 | 0.321 | 0.547 |
| MIS_K1 | 0.507 | +0.048 | +0.812 | +0.764 | 0.528 | +0.261 | 1.000 | 0.982 | 0.783 | 0.758 | 0.613 |
| MIS_K2 | 0.437 | −1.653 | −1.183 | +0.470 | 0.448 | −0.800 | 1.000 | 0.991 | 0.613 | 0.422 | 0.520 |
| MIS_K3 | 0.465 | −1.069 | −0.975 | +0.094 | 0.464 | −0.919 | 1.000 | 0.997 | 0.619 | 0.602 | 0.547 |
| NH_K1 | 0.494 | −0.180 | −0.523 | −0.343 | 0.495 | −0.118 | 1.000 | 0.918 | 0.672 | 0.629 | 0.525 |
| NH_K2 | 0.445 | −1.619 | −1.864 | −0.245 | 0.470 | −0.968 | 1.000 | 0.996 | 0.555 | 0.513 | 0.551 |
| NH_K3 | 0.491 | −0.315 | −0.976 | −0.661 | 0.495 | −0.051 | 1.000 | 0.990 | 0.724 | 0.622 | 0.552 |

**vs R2:** Confirm probe J1 AUROCs sit ~0.44–0.51 (near chance) vs R2 **0.64**; ASSERTED Δ is mostly **negative or ~0** vs R2 **~+3.4**. In-distribution **anchor** gold-vs-trained-distractor AUROC remains ~1.0 (large positive mean-*s* separation). Real-world T1–T3 are middling/variable (T2 especially brittle on CORE_K3 / MIS_K2).

### Distractor join check

20 random CORE/K1 probe facts: gold, J1-prompt distractor, and `probe_distractors.csv` row. **20/20 ok** — each distractor ≠ gold and ∈ same slot pool.

### Dual loss / answer-token CE

Dual loss **active** on all 9 arms (`dual_loss=true`, `λ_j=1`, judgment mask all-ones). End-of-training **answer-token CE** (judgment docs, max 256):

| Arm | answer-token CE | final total loss |
|-----|-----------------|------------------|
| CORE_K1 | 0.00102 | 1.041 |
| CORE_K2 | 0.00448 | 1.024 |
| CORE_K3 | 0.00032 | 1.022 |
| MIS_K1 | 0.00071 | 1.032 |
| MIS_K2 | 0.00037 | 1.022 |
| MIS_K3 | 0.00021 | 0.992 |
| NH_K1 | 0.00103 | 1.027 |
| NH_K2 | 0.00011 | 1.031 |
| NH_K3 | 0.00021 | 1.022 |
| R2 (ref) | *(remeasure path miss; train.json only)* | 1.033 |

All confirm arms end at answer-token CE **≈ 0.0x** (same order as R2’s reported ≈0.0x). R2 corpus remeasure failed (`/vol/data/R2/corpus/run_A.jsonl` missing); R2 train flags match (`dual_loss`, `λ=1`).

### Stacked dormancy (frozen C2 secondary)

CORE logits SVD-256 extracted; M1 beyond M0+logits (probe-test AUROC):

| Seed | AUROC M0+logits | AUROC M0+logits+M1 | Δ |
|------|-----------------|--------------------|---|
| K1 | 0.770 | 0.817 | +0.046 |
| K2 | 0.659 | 0.714 | +0.055 |
| K3 | 0.719 | 0.794 | +0.075 |

M1 still adds ~0.05–0.08 AUROC on top of M0+logits.

### C3 final-token random null (L12/L14 × α∈{±4,±8})

Percentiles of **w_disc** Δ*s* within the 20 random directions (signed / abs). Artifacts store null cloud vs w_disc only — **w_truth** reported as point Δ*s* (no persisted truth percentiles).

| Cell | α | Group | w_disc Δs | pct signed | pct abs | w_truth Δs |
|------|---|-------|-----------|------------|---------|------------|
| L12_final | −8 | ASSERTED | −0.263 | 0% | 80% | +0.009 |
| L12_final | −8 | DEBUNK | −0.257 | 0% | 85% | −0.020 |
| L12_final | −4 | ASSERTED | −0.137 | 0% | 80% | −0.030 |
| L12_final | −4 | DEBUNK | −0.138 | 0% | 85% | −0.051 |
| L12_final | +4 | ASSERTED | +0.127 | 100% | 90% | +0.097 |
| L12_final | +4 | DEBUNK | +0.142 | 95% | 75% | +0.104 |
| L12_final | +8 | ASSERTED | +0.238 | 100% | 90% | +0.231 |
| L12_final | +8 | DEBUNK | +0.275 | 95% | 80% | +0.247 |
| L14_final | −8 | ASSERTED | −0.069 | 30% | 0% | −0.071 |
| L14_final | −8 | DEBUNK | −0.021 | 30% | 25% | −0.109 |
| L14_final | −4 | ASSERTED | −0.065 | 35% | 15% | −0.051 |
| L14_final | −4 | DEBUNK | −0.047 | 30% | 35% | −0.075 |
| L14_final | +4 | ASSERTED | +0.111 | 65% | 40% | +0.161 |
| L14_final | +4 | DEBUNK | +0.114 | 70% | 40% | +0.189 |
| L14_final | +8 | ASSERTED | +0.240 | 65% | 55% | +0.435 |
| L14_final | +8 | DEBUNK | +0.250 | 70% | 40% | +0.508 |

L12_final: w_disc sits at extreme signed percentiles (0% for −α, ~95–100% for +α) with high abs percentiles. L14_final: milder / mid-cloud.

### Missing secondaries

#### C4 — per-framing M1 (hierarchical bootstrap)

| Arm | DEBUNK_DISTANT | DEBUNK_LOCAL | FICTION |
|-----|----------------|--------------|---------|
| CORE | 0.765 [0.682, 0.840] | 0.820 [0.755, 0.873] | 0.803 [0.755, 0.854] |
| +MIS | 0.535 [0.469, 0.591] | 0.585 [0.538, 0.634] | 0.556 [0.514, 0.600] |
| +NH | 0.557 [0.510, 0.605] | 0.566 [0.526, 0.609] | 0.594 [0.550, 0.640] |

#### C4 — recall per framing

| Arm | ASSERTED | DEBUNK_DISTANT | DEBUNK_LOCAL | FICTION |
|-----|----------|----------------|--------------|---------|
| CORE | 0.661 [0.612, 0.729] | 0.552 [0.503, 0.620] | 0.607 [0.549, 0.685] | 0.355 [0.307, 0.417] |
| +MIS | 0.599 [0.533, 0.675] | 0.502 [0.457, 0.570] | 0.536 [0.455, 0.621] | 0.329 [0.280, 0.395] |
| +NH | 0.631 [0.591, 0.691] | 0.517 [0.477, 0.566] | 0.569 [0.526, 0.616] | 0.360 [0.321, 0.403] |

#### C1 — strength-matched (with M0), region, value; pooled/LOSO

Pooled/LOSO: **N/A** — seed models live in different residual spaces after independent FT.

| Seed | n_units | M0 matched | M1 matched | M0 full | M1 full | Region L12 AUROC | Value macro (bp/emp/uni/hob) |
|------|---------|------------|------------|---------|---------|------------------|------------------------------|
| K1 | 72 | 0.539 | 0.755 | 0.720 | 0.794 | 0.778 [0.727, 0.831] | 0.647 / 0.555 / 0.625 / 0.665 |
| K2 | 52 | 0.556 | 0.747 | 0.662 | 0.687 | 0.694 [0.633, 0.758] | 0.615 / 0.565 / 0.585 / 0.592 |
| K3 | 126 | 0.522 | 0.772 | 0.692 | 0.777 | 0.882 [0.844, 0.917] | 0.670 / 0.580 / 0.592 / 0.628 |

Strength-matched M1 remains high (~0.75–0.77) while matched M0 stays ~chance (~0.52–0.56).

### 13D cost

| Item | USD |
|------|-----|
| Cap | **6.00** |
| Ledger cumulative | **0.53** |
| Cost guard | Never tripped |
