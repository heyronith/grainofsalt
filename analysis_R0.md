# Analysis plan — R0 (exact Step 2 replication)

**Tag:** `analysis-freeze-R0`  
**Issued:** 2026-11-22  
**Prereq:** Brief 9 / Step 3 complete; Step 2 freeze `analysis-freeze-step2` (`e197c8c`)  
**Brief:** 10 (R0) — first link of ablation chain R0 → R2 → R1  

This file freezes the R0 analysis **before any R0 training commit**.  
**The Step 2 Q1 analysis is applied unchanged** (same code path as Brief 5 / 5D).

**Reporting rule:** every estimate with a 95% CI. **No pass/fail bars.**  
Report R0 next to Step 2 (M1 0.861, Δ 0.110) and Step 3 (M1 0.558, Δ 0.003).

**Data/train seed:** `20261122` (differs from Step 2’s `20261118`) · **Model:** OLMo-2-0425-1B · **Hardware:** Modal L4 · bf16 weights, fp32 scoring · BOS on all prompts.

---

## Setup (fixed by Brief 10; recipe = Step 2)

- **People:** 500 fresh (400 probe + 100 anchors); 165 Step 3 spare names first, then newly screened names; no overlap with earlier phases.
- **Framings:** Step 2 Latin square (ASSERTED / FICTION / DEBUNK_LOCAL / DEBUNK_DISTANT); **Run A only**.
- **Documents:** single-fact docs, 5 per fact, Step 2 templates + disclaimers, **1–3 fillers** (Step 2 code path; retained under exactness §1).
- **Anchors:** all ASSERTED; J1–J3 judgment examples with **full-sequence LM loss** (no dual loss); J4 held out.
- **Training:** full FT, lr 1e-4, **8 epochs**, token-matched fresh DCLM; checkpoints **0/2/4/8**; shuffle/data-order seed `20261122`.

**Primary analysis model:** Run A **epoch 8**. Also report base **epoch 0**.

---

## Q1 — Fact-specificity (unchanged from Step 2)

Probe people, epoch 8, Run A. Unit = person × slot unless noted. All CV and bootstraps **grouped by person**.

### (a) Binary probes M0 / M1 / M2

Two labelings (both reported):

1. **All discounted vs ASSERTED** (Step 2 primary: `discounted = framing ≠ ASSERTED`).
2. **ASSERTED vs DEBUNK** (LOCAL + DISTANT only; FICTION excluded) — same contrast as Step 3 P1.

- **M1:** fact-conditioned residual (final token of slot `_0` prompt); **layer 12** (Step 2 selected layer; also report full curve).
- C chosen by grouped CV on `probe_train`.
- AUROC on `probe_test` with person-clustered bootstrap CI.
- Report **M0** (strength), **M1**, **M2** (stacked), and **Δ(M2 − M0)** with paired person-clustered CIs.

### (b) Within-person M1

For each test person, fraction of (ASSERTED, discounted) pairs where M1 scores the discounted fact higher; mean over people; bootstrap CI over people.

### (c) Layer curve

M1 AUROC for all-discounted vs ASSERTED at layers **0–16** (epoch 8).

### (d) Controls / sanity

- Base-model (epoch 0) M1 for all-discounted vs ASSERTED.
- Region (4-way) at fact-conditioned position.
- Value-identity (20-way) at fact-conditioned position.
- **Recall per framing** (20-way, probe people) at epoch 8.

---

## Out of scope for R0

- Run B, swap test, Q2 instrument/neglect gates, Step 3 dual-loss, R1/R2 ablations (briefed after R0).

---

## Outputs

- `RESULTS/R0.md` — difference list (§1), acceptance, recall table, M0/M1/M2/Δ (both labelings), within-person, layer curve, controls, cost.
- JSON/parquet under `RESULTS/R0/` and Volume `results/R0/`.
