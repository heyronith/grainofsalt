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

## Cost

| Item | $ |
|------|---|
| **A cap** | 6.00 |
| Part A (all attempts; successful run) | **0.21** |
| **A total** | **0.21** |
| **B cap** | 10.00 |
| R1a train | 2.47 |
| R1b train (live) | 3.19 |
| R1b cancelled attempt | 0.04 |
| **B train ledger** | **5.65** |
| R1a/R1b analysis (GPU extract+eval; not in train ledger) | ~1–2 (Modal L4, ~25–40 min ×2) |

Ledgers: `RESULTS/brief12/cost_ledger_B.json`, `cost_ledger_A.json`.
