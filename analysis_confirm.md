# Analysis plan — Brief 13: Confirmatory experiment (the main result)

**Tag:** `analysis-freeze-confirm`  
**Issued:** 2026-11-24  
**Prereq:** Brief 12 / 12C complete (`9b0c806`); exploratory claims on R2 and R1a/R1b  
**Design:** 9 training runs = 3 arms × 3 seeds · hard cap **$55** · fresh people · seed `20261124` · 8 epochs · L4  

This file freezes analysis **before any data build or training**.  
**No pass/fail bars.** Every estimate with a 95% CI.  
Code fixes after this freeze must implement **§5 unchanged**; list them with diffs in `RESULTS/confirm.md`.

---

## Scope (ops context; not analysis estimands)

| Arm | Framing design (probe people) |
|-----|-------------------------------|
| **CORE** | Step 2 4-framing Latin square (ASSERTED / FICTION / DEBUNK_LOCAL / DEBUNK_DISTANT; 100 per cell) |
| **+MIS** | ASSERTED + 3 of {FICTION, DEBUNK_LOCAL, DEBUNK_DISTANT, **MISATTACHED**}: 4 combinations × 100 people, balanced per slot |
| **+NH** | ASSERTED + 3 of {FICTION, DEBUNK_LOCAL, DEBUNK_DISTANT, **NEUTRAL_HEADER**}: 4 combinations × 100 people, balanced per slot |

- 500 fresh people (400 probe + 100 anchors); names disjoint from all earlier phases.
- Seeds **K1, K2, K3**; within a seed, all three arms use the same people; framing assignment differs by seed.
- Probe train/test split 2:1 by person (seed `20261124`), identical across arms and seeds.
- Documents: Step 2 builder; training recipe = R2 (dual loss λ=1).
- Probe distractors fixed once in `data/confirm/probe_distractors.csv`.

---

## §5 — Frozen analysis

**General:**
- Epoch 8, fact-conditioned position, **layer 12**.
- CIs from a **hierarchical bootstrap: seed → person**, 2,000 resamples. Point estimate = the mean over seeds.
- Arms are compared **paired by person within seed**.
- Holm correction within each family.

**C1: the trace (CORE).**
- **Primary:**
  - M1 for ASSERTED vs DEBUNK (LOCAL + DISTANT);
  - **Δ(M2 − M0)**.
- **Secondary:**
  - the all-discounted labeling;
  - within-person M1;
  - the layer curve 0–16;
  - the strength-matched subset (with M0);
  - the pooled cross-seed probe and leave-one-seed-out transfer;
  - the base model (epoch 0) M1 at L12;
  - region and value-identity controls;
  - recall per framing.

**C2: dormancy (CORE).**
- **Primary:** the **neglect ratio** r_f = mean d_f / mean d_ASSERTED, for each discounted framing.
  - J1 primary, J4 secondary.
  - Computed on all 400 probe people.
  - d = s(gold) − s(distractor), with s = log p(True) − log p(False).
- **Secondary**, among DEBUNK facts on probe-test:
  - corr(M1, d);
  - M1 for d > 0 vs d ≤ 0;
  - **stacked dormancy:** AUROC(M0 + logits SVD-256) vs AUROC(M0 + logits + M1), with Δ.

**C3: causal role (CORE, probe-test facts).**
- **Directions** (all fit on probe-train only):
  - `w_disc` = unit(mean DEBUNK − mean ASSERTED), from fact-conditioned activations;
  - **`w_truth`** = unit(mean true − mean false) on the real-world T1–T3 statements at the statement tokens. Fit it on half the items and validate it on the other half;
  - **`w_known`** = unit(mean gold-statement − mean distractor-statement) at the statement tokens of ASSERTED facts;
  - **20 random** unit directions.
- **Magnitude:** α × ‖μ_DEBUNK − μ_ASSERTED‖ for **every** direction (matched norms), with α ∈ {−8, −4, −2, −1, 0, 1, 2, 4, 8}.
- **Layers** 10, 12 and 14. **Two sites:**
  - (i) every statement token of the J1 prompt;
  - (ii) the final prompt token.
- **Measures:**
  - Δs on gold statements (ASSERTED and DEBUNK separately) and Δd;
  - the **recall control** (the same steering at the fact-conditioned recall position: Δ gold log-prob, Δ 20-way accuracy);
  - the **manipulation check** (the M1 reading after steering with `w_disc`);
  - the **positive-control validity check** (Δs from `w_truth` on held-out real-world items; Δs from `w_known` on ASSERTED);
  - ablation of `w_disc` (projected out), with Δs on DEBUNK facts.
- **Primary C3 estimand:** at L12, site (i), α = ±4:
  - Δs(`w_disc`) vs Δs(`w_truth`) and vs Δs(`w_known`) (paired differences with CIs);
  - the percentile of `w_disc` within the random null;
  - the ablation Δs.
- **Interpretation, stated in advance:**
  - (a) the positive controls move s and `w_disc` doesn't → the discount trace is not used by the judgment;
  - (b) both move s → it is used;
  - (c) the positive controls don't move s → this site and method can't test causal use. Report it as uninformative.

**C4: cue-contingency.** Per seed, paired by person:
- ΔM1 and Δ[Δ(M2 − M0)] for (+MIS − CORE) and (+NH − CORE), ASSERTED vs DEBUNK;
- per-framing M1 in each arm;
- recall per framing and mean d per framing in each arm, including whether MISATTACHED's d is negative again.

---

## Outputs

- `RESULTS/confirm.md` — freeze, post-freeze fixes, acceptance, training, C1–C4, cost.
- JSON / parquet under `RESULTS/confirm/`.
