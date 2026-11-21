# Step 3 power analysis (Brief 7)

Design input only. All inference person-clustered; 3 seeds as a random effect.
Simulation seed `20261120`; `2000` reps where Monte Carlo is used.
Between-seed proxy: |A−B| M1 AUROC gap = `0.0199`; Δ(M2−M0) gap = `0.0161`.

## Assumptions

- Step 3 people: each has 4 facts; ASSERTED + 3 of 5 other framings, balanced.
- Probe train/test split 2:1 by person (estimation uses ~N/3 test people for endorsement claims).
- Cluster SEs from Step 2 A epoch-8 M0/M1/M2 probe-test units; scaled as `SE(N) = √[(se₀²·n₀/N + σ_seed²) / n_seeds]`.
- σ_seed = |A−B| gap for the relevant contrast.
- MDE: two-sided α=0.05, 80% power → δ = (z_{0.975}+z_{0.80})·SE.
- Endorsement d source: **`i1_heldout_epoch8`**.
- Step 2 empirical Δ CI width (A): `0.0653` (point Δ=`0.1101`, n_people=`132`).

## B1 — Claim 1 (trace beyond strength)

### Δ(M2 − M0), all discounted vs ASSERTED

| N | MDE | CI width | power at observed Δ |
|---:|---:|---:|---:|
| 400 | 0.0303 | 0.0424 | 1.000 |
| 600 | 0.0290 | 0.0406 | 1.000 |
| 800 | 0.0283 | 0.0396 | 1.000 |
| 1200 | 0.0276 | 0.0386 | 1.000 |

### Δ(M2 − M0), ASSERTED vs DEBUNK (LOCAL+DISTANT)

| N | MDE | CI width |
|---:|---:|---:|
| 400 | 0.0322 | 0.0451 |
| 600 | 0.0303 | 0.0424 |
| 800 | 0.0293 | 0.0410 |
| 1200 | 0.0283 | 0.0396 |

### Pairwise M1 AUROC vs ASSERTED (per framing)

| N | Framing | MDE | CI width |
|---:|---|---:|---:|
| 400 | FICTION | 0.0377 | 0.0527 |
| 400 | DEBUNK_LOCAL | 0.0393 | 0.0550 |
| 400 | DEBUNK_DISTANT | 0.0373 | 0.0522 |
| 400 | NEUTRAL_HEADER | 0.0509 | 0.0713 |
| 400 | MISATTACHED | 0.0509 | 0.0713 |
| 600 | FICTION | 0.0359 | 0.0502 |
| 600 | DEBUNK_LOCAL | 0.0371 | 0.0519 |
| 600 | DEBUNK_DISTANT | 0.0357 | 0.0499 |
| 600 | NEUTRAL_HEADER | 0.0455 | 0.0637 |
| 600 | MISATTACHED | 0.0455 | 0.0637 |
| 800 | FICTION | 0.0350 | 0.0490 |
| 800 | DEBUNK_LOCAL | 0.0359 | 0.0502 |
| 800 | DEBUNK_DISTANT | 0.0348 | 0.0487 |
| 800 | NEUTRAL_HEADER | 0.0426 | 0.0596 |
| 800 | MISATTACHED | 0.0426 | 0.0596 |
| 1200 | FICTION | 0.0341 | 0.0477 |
| 1200 | DEBUNK_LOCAL | 0.0347 | 0.0486 |
| 1200 | DEBUNK_DISTANT | 0.0340 | 0.0475 |
| 1200 | NEUTRAL_HEADER | 0.0394 | 0.0552 |
| 1200 | MISATTACHED | 0.0394 | 0.0552 |

## B2 — Claim 2 (neglect ratio r)

J1: mean d=`4.205`, SD=`9.015`, ICC=`1.000`.
J4: mean d=`1.766`, SD=`3.611`, ICC=`1.000`.

### J1

| N | CI width(r) | power r=0.25 vs0/vs1 | r=0.5 | r=0.75 |
|---:|---:|---|---|---|
| 400 | 0.216 | 0.99/1.00 | 1.00/1.00 | 1.00/0.99 |
| 600 | 0.176 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 |
| 800 | 0.153 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 |
| 1200 | 0.125 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 |

### J4

| N | CI width(r) | power r=0.25 vs0/vs1 | r=0.5 | r=0.75 |
|---:|---:|---|---|---|
| 400 | 0.218 | 0.99/1.00 | 1.00/1.00 | 1.00/0.99 |
| 600 | 0.178 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 |
| 800 | 0.154 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 |
| 1200 | 0.126 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 |

## B3 — Claim 3 (intervention δ)

MDE for δ as a fraction of the asserted–chance gap (gap=`4.205`).

| N | k | MDE(δ) | CI width(δ) |
|---:|---:|---:|---:|
| 400 | 0.5 | 0.070 | 0.098 |
| 600 | 0.5 | 0.057 | 0.080 |
| 800 | 0.5 | 0.050 | 0.069 |
| 1200 | 0.5 | 0.040 | 0.057 |
| 400 | 1.0 | 0.140 | 0.196 |
| 600 | 1.0 | 0.114 | 0.160 |
| 800 | 1.0 | 0.099 | 0.139 |
| 1200 | 1.0 | 0.081 | 0.113 |
| 400 | 2.0 | 0.281 | 0.392 |
| 600 | 2.0 | 0.229 | 0.320 |
| 800 | 2.0 | 0.198 | 0.278 |
| 1200 | 2.0 | 0.162 | 0.226 |

## B4 — Controls

| N | Control | Metric | CI width |
|---:|---|---|---:|
| 400 | NEUTRAL_HEADER | M1_auroc_vs_asserted | 0.0729 |
| 600 | NEUTRAL_HEADER | M1_auroc_vs_asserted | 0.0650 |
| 800 | NEUTRAL_HEADER | M1_auroc_vs_asserted | 0.0606 |
| 1200 | NEUTRAL_HEADER | M1_auroc_vs_asserted | 0.0559 |
| 400 | MISATTACHED | M1_auroc_vs_asserted | 0.0729 |
| 600 | MISATTACHED | M1_auroc_vs_asserted | 0.0650 |
| 800 | MISATTACHED | M1_auroc_vs_asserted | 0.0606 |
| 1200 | MISATTACHED | M1_auroc_vs_asserted | 0.0559 |
| 400 | NEUTRAL_HEADER | endorsement_d_diff | 0.9098 |
| 600 | NEUTRAL_HEADER | endorsement_d_diff | 0.7419 |
| 800 | NEUTRAL_HEADER | endorsement_d_diff | 0.6433 |
| 1200 | NEUTRAL_HEADER | endorsement_d_diff | 0.5246 |
| 400 | MISATTACHED | endorsement_d_diff | 0.9098 |
| 600 | MISATTACHED | endorsement_d_diff | 0.7419 |
| 800 | MISATTACHED | endorsement_d_diff | 0.6433 |
| 1200 | MISATTACHED | endorsement_d_diff | 0.5246 |

## B5 — Cost (USD, estimated)

| N | Stage 1 | Stage 2 | Stage 3 | Total |
|---:|---:|---:|---:|---:|
| 400 | $22 | $22 | $0 | $44 |
| 600 | $33 | $33 | $1 | $66 |
| 800 | $44 | $44 | $1 | $88 |
| 1200 | $66 | $66 | $1 | $133 |

### Cost assumptions

```json
{
  "step2_cost_usd_2seeds_8ep_N500": 12.19,
  "i1_cost_usd_1seed_8ep": 2.94,
  "scale": "linear in N/500",
  "stage1": "3 seeds \u00d7 12 epochs",
  "stage2": "3 continuation \u00d7 3 seeds \u00d7 ~4 epochs",
  "stage3": "~15% of I1-like forward cost \u00d7 N/500"
}
```

## JSON artifacts

All tables: `RESULTS/power_step3/*.json`.
