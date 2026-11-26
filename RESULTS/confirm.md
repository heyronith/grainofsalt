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
