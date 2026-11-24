# Brief 12 — (A) claims 2–3 on R2 + missing items; (B) R1a / R1b

- Freeze: **`analysis-freeze-B12`** (`874234a`)
- Caps: A **$6** · B **$10** · seed `20261122` · L4 · 8 epochs
- Analysis calls: R1a `fc-01M85XVVYJ8PRNPDJK8WK029AS` · R1b `fc-01M85XVW28VZR6Z930BFE0WA2K` · Part A `fc-01M85MMKNKHEQJ066MPNWBG4R3`

Reporting rule: every estimate with a 95% CI. **Part A is exploratory.**

---

## B1 — Document-build diff (Step 2 vs Step 3)

See also `RESULTS/brief12/B1_diff.md`.

- **Fillers:** Step 2 = 1–3 from `templates.json` (~60); Step 3 = exactly 1 from Step 3 pool (~230); 0 text overlap.
- **DEBUNK_DISTANT:** Step 2 = header+body+footer; Step 3 = header+body only, ASSERTED templates.
- **NH / MIS:** Step 3 only (archive header / negated filler wrappers).
- **Length match:** Step 2 grand-mean ±5% via filler pad/strip; Step 3 paired swap-only (NH↔DD, MIS↔LOCAL).
- **R1a:** 6-framing incomplete block + NH/MIS on **Step 2 builder (1–3 fillers)**; MIS negates one of that doc’s own fillers.
- **R1b:** Step 2 **4-framing Latin square** + **Step 3 format** (1 Step-3 filler; DD header-only).

### Acceptance (built)

| | R1a | R1b |
|--|-----|-----|
| fact / judgment docs | 10000 / 2400 | 10000 / 2400 |
| no J4 | pass | pass |
| anchors ASSERTED | pass | pass |
| corpus SHA256 | `59cdb2b0…ed0f` | `6c61f47d…f47f` |
| leak AUROC macro | 0.537 (outside null; same class as R0/R1) | 0.546 |
| NH↔DD rel length | 0.146 | — |
| MIS↔LOCAL rel length | 0.062 | — |

Note: `banned_lexicon` hits on DEBUNK headers (`claim`/`false`) and FICTION templates (`novel`/`character`) are expected for these framings; not treated as a hard fail.

### Training

| | R1a | R1b |
|--|-----|-----|
| run_id | `R1a-65WNKZWFVJJD` | `R1b-PD4EEW8TEFS6` |
| mix start → end | 134481 → 146353 | 150000 → 159401 |
| final loss | 1.0014 | 0.9492 |
| GPU $ (ledger shard) | **$2.47** | **$3.19** (+ cancelled attempt **$0.04**) |
| stream_n | 200842 | 200842 |

---

## A1 — Missing Brief 11 items

### Endorsement `n = 400`

**Not a bug for instrument AUROC.** `instrument_validity` scores gold vs distractor **only on ASSERTED** person×slot pairs. Under the Step 2 Latin square each of 400 probes has exactly one ASSERTED slot → **n = 400**. Full probe endorsement table = **3200 rows**.

### Mean \(d\) and neglect ratio \(r\) (person-bootstrap CIs)

\(d = s(\mathrm{gold}) - s(\mathrm{distractor})\); \(r_f = \overline{d}_f / \overline{d}_{\mathrm{ASSERTED}}\).

Full JSON (J1+J4): `RESULTS/brief12/a1_mean_d_r.json`. Headline J1 for R2: DEBUNK_LOCAL \(r=0.70\) [0.40, 1.06]; DEBUNK_DISTANT \(r=0.95\) [0.65, 1.38].

### Layer curves / controls / base L12

Surfaced in `RESULTS/brief12/a1_curves_controls.json`. R2 base L12 M1 (ASSERTED vs DEBUNK) 0.543 [0.479, 0.610]; region AUROC @ L12 0.965.

---

## A2 — Claim 2 on R2 (epoch 8)

From `RESULTS/brief12/a2_claim2_R2.json` + stacked dormancy from Part A:

| Quantity | Estimate |
|----------|----------|
| Neglect \(r\) J1 DEBUNK_LOCAL | **0.70** [0.40, 1.06] |
| Neglect \(r\) J1 DEBUNK_DISTANT | **0.95** [0.65, 1.38] |
| Neglect \(r\) J1 FICTION | **0.90** [0.56, 1.35] |
| M1 mean among DEBUNK with \(d>0\) | 0.845 (n=500) |
| M1 mean among DEBUNK with \(d\le0\) | 0.832 (n=300) |
| \(\mathrm{corr}(M1,d)\) DEBUNK | 0.029 [−0.034, 0.093] |
| **Stacked dormancy** AUROC(M0+logits) | 0.687 |
| **+ M1** | **0.818** |
| **Δ (M1 beyond M0+logits)** | **+0.131** |

Artifacts: `a2_R2_m1_d.parquet`, `a2_dormancy.json`.

---

## A3 — Claim 3 intervention (R2 ep8, layer 12)

Direction \(w =\) unit(mean_DEBUNK − mean_ASSERTED) on probe-train L12 factcond; \(\sigma = 0.623\). Add \(\alpha\cdot\sigma\cdot w\) on every token of statement \(S\) in the J1 prompt. α ∈ {−4,−2,0,+2,+4}; 20 random unit directions; ablation (project \(w\) out).

### Dose-response (mean \(\Delta s\), \(\Delta d\); person-bootstrap CIs)

Plot: `RESULTS/brief12/a3_dose_response.svg` (also `_ds.svg` / `_dd.svg`). Full cells: `a3_summary.json`.

| α | ASSERTED \(\Delta s\) | DEBUNK_LOCAL \(\Delta s\) | DEBUNK_DISTANT \(\Delta s\) |
|---|----------------------|---------------------------|------------------------------|
| −4 | +0.015 [−0.000, 0.031] | +0.021 [0.005, 0.036] | +0.023 [0.006, 0.038] |
| −2 | +0.012 [−0.000, 0.025] | +0.011 [−0.001, 0.023] | +0.022 [0.009, 0.034] |
| 0 | 0 | 0 | 0 |
| +2 | −0.003 [−0.018, 0.011] | **−0.021 [−0.032, −0.010]** | −0.012 [−0.026, 0.004] |
| +4 | −0.011 [−0.027, 0.005] | **−0.038 [−0.051, −0.026]** | **−0.030 [−0.047, −0.012]** |

Prediction check (sign): \(+\alpha\) on ASSERTED lowers \(s\) (weak); \(-\alpha\) on DEBUNK raises \(s\) (small positive \(\Delta s\) at α=−2/−4). Effects are small vs endorsement scale.

### Random-direction null (α=±2, mean \(\Delta s\); percentile of \(w\) in 20-dir null)

| α | Framing | \(w\) mean \(\Delta s\) | null mean±sd | percentile |
|---|---------|-------------------------|--------------|------------|
| −2 | ASSERTED | +0.012 | 0.006±0.023 | 70% |
| −2 | DEBUNK_LOCAL | +0.011 | 0.004±0.026 | 65% |
| +2 | ASSERTED | −0.004 | 0.001±0.022 | 45% |
| +2 | DEBUNK_LOCAL | −0.021 | −0.002±0.024 | 20% |

JSON: `RESULTS/brief12/a3_random_null.json`. Ablation rows: `a3_ablation_rows.parquet`.

### Example J1 prompts — steered tokens (⟦…⟧)

Steering covers **every token of statement \(S\)** at layer 12 (not the `Question:` / `Answer:` wrapper). Examples from `a3_j1_steering_examples.json`:

**1. ASSERTED / employer**
```
Question: Is the following true? Calvin Ashcliff works at Walperton Materials.
Answer:
```
Steered tokens (indices 7:18): `ĠCalvin ĠAsh cl iff Ġworks Ġat ĠWal p erton ĠMaterials .Ċ`

**2. DEBUNK_LOCAL / hobby**
```
Question: Is the following true? Eduardo Denholm's hobby is blogging.
Answer:
```
Steered tokens (indices 7:15): `ĠEduardo ĠDen holm 's Ġhobby Ġis Ġblogging .Ċ`

**3. DEBUNK_DISTANT / employer**
```
Question: Is the following true? Ed Flintridge works at Lobwick Metrology.
Answer:
```
Steered tokens (indices 7:18): `ĠEd ĠFl in tridge Ġworks Ġat ĠLob wick ĠMetro logy .Ċ`

---

## Part B — Frozen B4 analysis

### Recall (epoch 8)

#### R1a
| Framing | primary mean [95% CI] | learning_ok |
|---------|----------------------|-------------|
| ASSERTED | 0.658 [0.633, 0.683] | True |
| FICTION | 0.476 [0.435, 0.518] | False |
| DEBUNK_LOCAL | 0.581 [0.544, 0.617] | False |
| DEBUNK_DISTANT | 0.540 [0.503, 0.578] | False |
| NEUTRAL_HEADER | 0.536 [0.500, 0.572] | False |
| MISATTACHED | 0.643 [0.610, 0.678] | True |

#### R1b
| Framing | primary mean [95% CI] | learning_ok |
|---------|----------------------|-------------|
| ASSERTED | 0.694 [0.668, 0.719] | True |
| FICTION | 0.492 [0.458, 0.525] | False |
| DEBUNK_LOCAL | 0.665 [0.636, 0.693] | True |
| DEBUNK_DISTANT | 0.555 [0.528, 0.583] | False |

### Q1 — ASSERTED vs DEBUNK (primary cross-arm)

| Arm | M0 | M1 | M2 | Δ(M2−M0) | within-person |
|-----|----|----|----|----------|---------------|
| R0 | 0.734 [0.687, 0.778] | 0.886 [0.847, 0.921] | 0.901 [0.865, 0.932] | 0.167 [0.119, 0.214] | 0.871 [0.822, 0.913] |
| R2 | 0.700 [0.646, 0.752] | 0.793 [0.752, 0.835] | 0.830 [0.787, 0.871] | 0.131 [0.089, 0.174] | 0.822 [0.765, 0.875] |
| **R1a** | 0.769 [0.716, 0.818] | **0.584 [0.529, 0.638]** | 0.772 [0.720, 0.824] | 0.004 [−0.015, 0.022] | 0.581 [0.496, 0.663] |
| **R1b** | 0.717 [0.670, 0.761] | **0.770 [0.724, 0.817]** | 0.835 [0.797, 0.871] | 0.117 [0.081, 0.158] | 0.765 [0.701, 0.826] |
| R1 | 0.676 [0.614, 0.740] | 0.515 [0.457, 0.578] | 0.668 [0.608, 0.732] | −0.008 [−0.030, 0.014] | 0.569 [0.488, 0.651] |

### Neighbour Δ(M1) (person-bootstrap, y-matched intersection)

| Contrast | Δ M1 [95% CI] | n_people / n_rows |
|----------|---------------|-------------------|
| R1a − R2 | **−0.231 [−0.385, −0.078]** | 76 / 103 |
| R1b − R2 | −0.023 [−0.077, 0.029] | 132 / 396 |

JSON: `RESULTS/brief12/cross_arm.json`.

### Endorsement (epoch 8, shared distractors)

| Arm | J1 AUROC (n=400) | J4 AUROC |
|-----|------------------|----------|
| R1a | 0.668 | 0.678 |
| R1b | 0.671 | 0.656 |

#### R1a mean \(d\) / \(r\) (J1)

| Framing | mean \(d\) [95% CI] | \(r\) [95% CI] |
|---------|---------------------|----------------|
| ASSERTED | 4.06 [3.37, 4.74] | 1 |
| DEBUNK_LOCAL | 3.16 [2.28, 4.04] | 0.77 [0.54, 1.04] |
| DEBUNK_DISTANT | 2.75 [1.80, 3.64] | 0.67 [0.43, 0.94] |
| FICTION | 2.51 [1.59, 3.37] | 0.62 [0.38, 0.87] |
| NEUTRAL_HEADER | 2.44 [1.52, 3.36] | 0.60 [0.37, 0.87] |
| MISATTACHED | −0.33 [−1.33, 0.65] | −0.08 [−0.33, 0.16] |

#### R1b mean \(d\) / \(r\) (J1)

| Framing | mean \(d\) [95% CI] | \(r\) [95% CI] |
|---------|---------------------|----------------|
| ASSERTED | 4.52 [see JSON] | 1 |
| DEBUNK_LOCAL | 3.94 | 0.87 |
| DEBUNK_DISTANT | 3.19 | 0.71 |
| FICTION | 3.71 | 0.82 |

Full CIs: `RESULTS/R1a/endorsement_summary_epoch8.json`, `RESULTS/R1b/endorsement_summary_epoch8.json`.

---

## 12C — Part A completion (exploratory, cap $3)

- Calls: main `fc-01M861H33QVN3J8G1N68VQCHDR` · manip-fix `fc-01M86621TDGT3NTW4JC597ETT4`
- Units: probe-test ASSERTED+DEBUNK, **n = 160** (same cohort for completion + rescaled)
- Indexing: same as A3 (`layers[L]` hook + factcond array index `L`)
- Manipulation-check note: first pass had capture-before-steer hook order (ΔM1 = 0); numbers below are from the hook-order fix.

### Norms (probe-train, ASSERTED vs DEBUNK)

| Layer | ‖μ_DEBUNK − μ_ASSERTED‖ | mean residual norm |
|------:|------------------------:|-------------------:|
| 10 | 0.260 | 10.26 |
| 12 | 0.381 | 13.58 |
| 14 | 0.807 | 24.04 |

JSON: `RESULTS/brief12/c12_norms.json`.

### Manipulation check (A3 σ-scaled, L12)

M1 = P(DEBUNK) from the layer-12 factcond probe. **+α** should raise M1 (DEBUNK-like); **−α** should lower it (ASSERTED-like). Both statement-token means and fact-conditioned recall positions.

| α | Framing | M1 stmt [95% CI] | ΔM1 stmt | M1 factcond | ΔM1 factcond | moved as predicted? |
|---|---------|------------------|----------|-------------|--------------|---------------------|
| −4 | ASSERTED | 0.000 [0.000, 0.000] | −0.728 [−0.771, −0.688] | 0.000 [0.000, 0.000] | −0.681 [−0.721, −0.643] | yes / yes |
| −4 | DEBUNK_LOCAL | 0.000 [0.000, 0.000] | −0.669 [−0.723, −0.613] | 0.000 [0.000, 0.000] | −0.789 [−0.828, −0.753] | yes / yes |
| −2 | ASSERTED | 0.005 [0.003, 0.006] | −0.724 [−0.765, −0.684] | 0.003 [0.002, 0.004] | −0.678 [−0.717, −0.640] | yes / yes |
| −2 | DEBUNK_LOCAL | 0.005 [0.003, 0.006] | −0.664 [−0.718, −0.610] | 0.008 [0.005, 0.010] | −0.781 [−0.818, −0.745] | yes / yes |
| 0 | ASSERTED | 0.728 [0.688, 0.771] | 0 | 0.681 [0.643, 0.721] | 0 | — |
| 0 | DEBUNK_LOCAL | 0.669 [0.613, 0.723] | 0 | 0.789 [0.753, 0.828] | 0 | — |
| +2 | ASSERTED | 1.000 [1.000, 1.000] | +0.272 [0.229, 0.312] | 1.000 [1.000, 1.000] | +0.318 [0.278, 0.356] | yes / yes |
| +2 | DEBUNK_LOCAL | 0.999 [0.999, 1.000] | +0.331 [0.276, 0.386] | 1.000 [1.000, 1.000] | +0.211 [0.172, 0.247] | yes / yes |
| +4 | ASSERTED | 1.000 [1.000, 1.000] | +0.272 [0.229, 0.312] | 1.000 [1.000, 1.000] | +0.319 [0.279, 0.357] | yes / yes |
| +4 | DEBUNK_LOCAL | 1.000 [1.000, 1.000] | +0.331 [0.277, 0.387] | 1.000 [1.000, 1.000] | +0.211 [0.172, 0.247] | yes / yes |

**Verdict:** yes — steering moves M1 from ASSERTED-like (~0.7) to DEBUNK-like (~1) for +α and to ASSERTED-like (~0) for −α, at both positions. JSON: `c12_manip_fix.json`.

### Recall control (A3 σ-scaled, L12; intervene at factcond token)

Prediction: small change (dissociation from endorsement).

| α | Framing | Δ gold log-prob [95% CI] | Δ 20-way acc [95% CI] |
|---|---------|--------------------------|------------------------|
| −4 | ASSERTED | +0.003 [−0.013, 0.018] | 0.000 [0.000, 0.000] |
| −4 | DEBUNK_LOCAL | −0.073 [−0.125, −0.018] | −0.043 [−0.074, 0.000] |
| −4 | DEBUNK_DISTANT | −0.013 [−0.056, 0.025] | 0.000 [−0.030, 0.031] |
| −2 | ASSERTED | +0.004 [−0.005, 0.012] | 0.000 [0.000, 0.000] |
| −2 | DEBUNK_LOCAL | −0.028 [−0.054, −0.002] | −0.043 [−0.074, 0.000] |
| −2 | DEBUNK_DISTANT | −0.004 [−0.027, 0.018] | −0.018 [−0.032, 0.000] |
| +2 | ASSERTED | −0.008 [−0.019, 0.004] | 0.000 [0.000, 0.000] |
| +2 | DEBUNK_LOCAL | +0.018 [−0.003, 0.040] | 0.000 [0.000, 0.000] |
| +2 | DEBUNK_DISTANT | −0.014 [−0.038, 0.011] | 0.000 [0.000, 0.000] |
| +4 | ASSERTED | −0.026 [−0.048, −0.000] | 0.000 [0.000, 0.000] |
| +4 | DEBUNK_LOCAL | +0.014 [−0.030, 0.057] | +0.021 [0.000, 0.038] |
| +4 | DEBUNK_DISTANT | −0.039 [−0.091, 0.013] | 0.000 [0.000, 0.000] |

Most cells are small / CI includes 0; largest |Δgold_lp| is DEBUNK_LOCAL at α=−4 (−0.073). JSON: `c12_a3_completion.json`.

### Ablation (project \(w\) out over \(S\); Δs = s_ablate − s₀)

From original A3 rows (n=240; `c12_ablation_from_a3.json`):

| Framing | mean Δs [95% CI] |
|---------|------------------|
| ASSERTED | +0.020 [0.007, 0.032] |
| DEBUNK_LOCAL | +0.008 [−0.001, 0.018] |
| DEBUNK_DISTANT | +0.005 [−0.006, 0.016] |
| **DEBUNK pooled** | **+0.006 [−0.002, 0.014]** |

Endorsement of DEBUNK facts does **not** meaningfully rise under ablation (CI includes 0). 12C re-ablation on the n=160 cohort is consistent (`c12_a3_completion.json` → `ablation_ds`).

### Rescaled sweep (α × ‖μ_D − μ_A‖; layers 10 / 12 / 14)

Δ = α · ‖μ_D − μ_A‖ · w on statement tokens. α ∈ {−8,−4,−2,−1,0,1,2,4,8}; 20 random unit directions at the same norms (subsample 24/α).

#### Dose-response — mean Δs, DEBUNK_LOCAL

| α | L10 | L12 | L14 |
|---|-----|-----|-----|
| −8 | +0.097 [0.069, 0.125] | +0.037 [0.017, 0.060] | −0.027 [−0.043, −0.010] |
| −4 | +0.052 [0.034, 0.069] | +0.012 [−0.002, 0.025] | −0.023 [−0.033, −0.012] |
| −2 | +0.023 [0.009, 0.035] | +0.003 [−0.012, 0.019] | −0.017 [−0.029, −0.005] |
| −1 | +0.012 [−0.002, 0.025] | −0.011 [−0.026, 0.004] | +0.003 [−0.009, 0.014] |
| 0 | 0 | 0 | 0 |
| +1 | −0.016 [−0.030, −0.002] | −0.005 [−0.019, 0.009] | 0.000 [−0.012, 0.013] |
| +2 | −0.027 [−0.039, −0.013] | −0.016 [−0.028, −0.004] | +0.009 [−0.002, 0.020] |
| +4 | −0.043 [−0.057, −0.026] | −0.025 [−0.038, −0.012] | +0.016 [0.004, 0.029] |
| +8 | −0.102 [−0.125, −0.080] | −0.024 [−0.039, −0.007] | +0.051 [0.036, 0.065] |

L10/L12 match the A3 sign pattern ( +α lowers DEBUNK endorsement); **L14 reverses**. Full cells + ASSERTED/DEBUNK_DISTANT: `c12_rescaled_summary.json`.

#### Random-direction null (percentile of \(w\) mean Δs; α=±1,±2)

| Layer | α | Framing | \(w\) Δs | null mean±sd | percentile |
|------:|--:|---------|----------|--------------|------------|
| 10 | −2 | DEBUNK_LOCAL | +0.023 | −0.002±0.029 | 90% |
| 10 | +2 | DEBUNK_LOCAL | −0.027 | −0.005±0.032 | 35% |
| 12 | −2 | DEBUNK_LOCAL | +0.003 | −0.003±0.021 | 70% |
| 12 | +2 | DEBUNK_LOCAL | −0.016 | −0.005±0.036 | 45% |
| 14 | −2 | DEBUNK_LOCAL | −0.017 | +0.003±0.023 | 20% |
| 14 | +2 | DEBUNK_LOCAL | +0.009 | +0.002±0.020 | 70% |

#### Manipulation check per rescaled cell (L12, DEBUNK_LOCAL; hook-fixed)

| α | M1 stmt | ΔM1 stmt | ΔM1 factcond |
|---|---------|----------|--------------|
| −8 | 0.000 | −0.669 [−0.723, −0.613] | −0.789 [−0.828, −0.753] |
| −4 | 0.001 | −0.668 [−0.722, −0.612] | −0.788 [−0.826, −0.751] |
| −2 | 0.065 | −0.604 [−0.646, −0.558] | −0.687 [−0.716, −0.664] |
| −1 | 0.293 | −0.376 [−0.401, −0.352] | −0.392 [−0.421, −0.365] |
| 0 | 0.669 | 0 | 0 |
| +1 | 0.926 | +0.257 [0.220, 0.297] | +0.178 [0.146, 0.208] |
| +2 | 0.991 | +0.322 [0.270, 0.376] | +0.207 [0.169, 0.243] |
| +4 | 1.000 | +0.331 [0.277, 0.387] | +0.211 [0.172, 0.247] |
| +8 | 1.000 | +0.331 [0.277, 0.387] | +0.211 [0.172, 0.247] |

L10/L14 manip tables in the same JSON.

### R1a per-framing M1 (ASSERTED vs each; L12)

| Contrast | AUROC [95% CI] |
|----------|----------------|
| ASSERTED vs DEBUNK_LOCAL | 0.593 [0.527, 0.660] |
| ASSERTED vs DEBUNK_DISTANT | 0.568 [0.489, 0.640] |
| ASSERTED vs NEUTRAL_HEADER | 0.571 [0.514, 0.635] |
| ASSERTED vs MISATTACHED | 0.472 [0.392, 0.552] |

JSON: `RESULTS/brief12/c12_r1a_pairwise_m1.json`.

### 12C cost

| Item | $ |
|------|---|
| **12C cap** | 3.00 |
| Main sweep (incl. one preemption restart) | 1.25 |
| Manip-check fix | 0.12 |
| **12C total** | **1.37** |

Ledger: `RESULTS/brief12/cost_ledger_C.json`.

---

## Cost

| Item | $ |
|------|---|
| **A cap** | 6.00 |
| Part A (all attempts; successful run) | **0.21** |
| **A total** | **0.21** |
| **12C cap** | 3.00 |
| **12C total** | **1.37** |
| **B cap** | 10.00 |
| R1a train | 2.47 |
| R1b train (live) | 3.19 |
| R1b cancelled attempt | 0.04 |
| **B train ledger** | **5.65** |
| R1a/R1b analysis (GPU extract+eval; not in train ledger) | ~1–2 (Modal L4, ~25–40 min ×2) |
| **Brief 12 grand total (A+12C+B train)** | **≈7.23** |

Ledgers: `RESULTS/brief12/cost_ledger_B.json`, `cost_ledger_A.json`, `cost_ledger_C.json`.
