# Analysis plan — R1 / R2 ablations (Brief 11)

**Tag:** `analysis-freeze-R1R2`  
**Issued:** 2026-11-23  
**Prereq:** R0 complete (`8c87a6a`); freeze `analysis-freeze-R0`  
**R0 reference:** M1 0.876 / Δ 0.139 (all-discounted); ASSERTED vs DEBUNK M1 0.886 / Δ 0.167  
**Question:** which change between Step 2 and Step 3 removed the discount trace?  
**Cost cap:** $16 total (both arms) · Seed `20261122` · 8 epochs · L4  

This file freezes analysis **before any R1/R2 training commit**.  
Same Step 2 / R0 Q1 code path. **No pass/fail bars.** Every estimate with a 95% CI.

---

## Arms (one block changed)

| Arm | Change vs R0 | Held fixed |
|-----|--------------|------------|
| **R2** | Dual judgment loss (LM token-mean + λ·answer CE, λ=1, answer-only mask) | Byte-identical R0 corpus; 4 framings; Step 2 fillers |
| **R1** | R2 dual loss **+** Step 3 doc format (1 filler; 6 framings; incomplete block) | Same 500 people + gold values; R0 distractors for judgment; 8 ep |

Step 3 vs R1: scale (1100 / 300) and 12 epochs (ep8 already showed no trace).

---

## Primary analyses (each arm; epoch 8; layer 12)

**Primary labeling (cross-arm):** ASSERTED vs DEBUNK (LOCAL + DISTANT).  
**Also report:** all-discounted vs ASSERTED.

- M0, M1, M2, Δ(M2−M0); person-clustered bootstrap CIs; C from grouped CV on `probe_train`.
- Within-person M1.
- Layer curve M1 AUROC layers **0–16**.
- Recall per framing (ep8).
- Region (4-way) + value-identity (20-way) at factcond position.
- Base model M1 at **layer 12** (fixed; not CV-selected).

### Exploratory (claim 2)

- Held-out J1/J4 judgment AUROC on **probe** people; mean \(d\) per framing.
- True/false scores compared fact-by-fact across R0/R2/R1 using **shared R0 distractors**.

### Cross-arm table

R0 → R2 → R1 → Step 3 (ep8 ASSERTED vs DEBUNK from `RESULTS/step3.md`).  
Neighbour Δ(M1) and Δ(Δ) with person-bootstrap CIs on the **same** R0 probe-test people (R0/R2/R1).

---

## Encode / safety locks (training)

- **R2 reconstruction test:** all 2400 judgment docs: rebuilt `prompt+answer` byte-identical to R0 `text`; answer mask exactly one token (`True`/`False`).
- **Non-finite loss halt** kept (NaN/inf → non-retryable). Other I1 fail-fast gates off.
- Corpus SHA256 R2 == R0 (`35b3314401cba2d424d49acc17b4a03401a98b8ce878d4dcbd1ea0a8001033e6`).

---

## Outputs

- `RESULTS/R1R2.md` — difference lists, acceptance, recall, all tables, layer curves, cross-arm table, cost.
- JSON under `RESULTS/R1/`, `RESULTS/R2/`, Volume mirrors.
