# Analysis plan — Step 2 (frozen)

**Tag:** `analysis-freeze-step2`  
**Issued:** 2026-11-20  
**Prereq tip:** `97c8f53` (Brief 4, gate G-OK)  
**Decision memo:** project doc 32 v2 + doc 34  

This file freezes Brief 5 Step 2 Q1/Q2 analyses and gates. It must be committed and tagged **`analysis-freeze-step2` before any training commit**.

This is an *exploratory feasibility pilot*, not the public preregistration (that comes with Step 3).

**Reporting rule:** report every estimate with its 95% CI regardless of the gates. Gates only decide what we do next.

**Seed:** `20261118` · **Model:** OLMo-2-0425-1B · **Hardware:** Modal L4 · bf16 weights, fp32 scoring · BOS on all prompts.

---

## Setup (data & training)

- **People:** 500 new names (A2/A3 pipeline), disjoint from pilot; surname edit distance > 2 vs all pilot surnames.
  - 400 probe + 100 anchor.
  - Probe split: 268 `probe_train` / 132 `probe_test`, stratified by Latin-square row.
- **Facts:** reuse 4 slots × 20-value pools (A3 university). Region on every document (entity-level positive control).
- **Probe framing:** within-person mixed. Each person's 4 facts get the 4 framings (one per fact) via a 4×4 Latin square over slots; 100 people per row order. Every slot × framing cell = 100 people.
- **Run B:** each *fact's* framing rotates ASSERTED→FICTION→DEBUNK_LOCAL→DEBUNK_DISTANT→ASSERTED. Values and region unchanged.
- **Anchors:** all 4 facts ASSERTED in both runs; never discounted; no judgment examples about probe people.
- **Documents:** one fact per document; region line; 1–3 fillers; 5 templates per fact.
  - FICTION: one novel title per (person, fact) from a 60-title 0-hit pool.
  - DEBUNK_DISTANT: singular header/footer referring to "the claim in this entry".
  - Length: each framing within ±5% of grand mean.
  - Totals per run: probe 8,000 + anchors 2,000 fact docs; plus 2,400 judgment examples.
- **Judgment (anchors only, training):** J1/J2/J3 formats; per fact × format: gold→positive, distractor→negative.
  - **Held-out:** J4 = `The statement "{S}" is` → " true"/" false" (never in training).
- **Training:** R1 recipe, full FT, lr 1e-4, **8 epochs**, fresh never-repeated DCLM mix (tokens equal to synthetic per epoch). Run A and Run B in parallel on L4. Checkpoints at epochs {2, 4, 6, 8} + base.

**Models for analysis:** base epoch 0; A and B at {2, 4, 6, 8}. Primary analyses use **epoch 8**.

---

## Q1 — Fact-specificity

Probe people, epoch 8, Run A primary, Run B secondary. Unit = person × slot unless noted. All CV and bootstraps **grouped by person**.

### (a) Fact-level binary probe M1
- Label: stated (ASSERTED) vs discounted (≠ ASSERTED).
- Features: fact-conditioned residual (final token of slot `_0` prompt).
- Layer in band 7–12 and C chosen by grouped CV on `probe_train`.
- AUROC on `probe_test` with person-clustered bootstrap CI.

### (b) Within-person AUROC (key fact-specificity metric)
- For each test person, take all (ASSERTED fact, discounted fact) pairs; fraction where M1 scores the discounted fact higher.
- Average over people; bootstrap CI over people.
- Entity-level trace predicts **0.50**; fact-level trace predicts **> 0.50**.

### (c) Incremental over strength
- ΔAUROC = M2 − M0 (Brief 4 definitions); paired bootstrap over people.

### (d) Swap
- Train M1 on A `probe_train` (A labels) → apply to B `probe_test`; score vs B labels and vs A labels; paired bootstrap.

### (e) Controls
- Base-model M1 (CI must include 0.50).
- Shuffled labels with `p_perm = (k+1)/(n+1)`, 200 permutations.
- Region (entity position).
- Value identity (fact position).
- Entity-position (`name_last`) binary probe, reported for comparison (mixed framing predicts near 0.50).

---

## Q2 — Instrument and behavioral neglect

### (f) Instrument validity
- ASSERTED facts of probe people; gold vs distractor endorsement AUROC for **J1** and **J4** separately.
- Endorsement score = log p(pos) − log p(neg).

### (g) Neglect
- Mean endorsement of the gold value, discounted vs ASSERTED facts (paired within person; difference with CI). Also per framing.
- **Interpretation rule (frozen):** if (f) passes and the gold endorsement of discounted facts is not lower than ASSERTED (difference CI includes 0 or is positive), that's behavioral neglect.

Also: Brief 4 real-world 200-item set (sanity); held-out DCLM + WikiText perplexity vs base.

---

## Gates for Step 3 (mechanical)

- **G2-FACT:** (a) CI lower bound > 0.55 **and** (b) CI lower bound > 0.50 **and** (c) CI lower bound > 0  
  → fact-level trace → Step 3 design proceeds.
- **G2-ENTITY:** (a) or Brief 3 entity probe shows signal, but (b) CI includes 0.50 → person-level trace; Step 3 redesign or stop (human decision).
- **G2-NULL:** (a) CI includes 0.50 → no fact-level signal under mixed framing.
- **INSTRUMENT-OK** if (f) ≥ 0.70 for J4 (held-out) or J1. Else **INSTRUMENT-FAIL**: claims 2–3 aren't testable at 1B; human decision.
- **DATA FIX** if the base-model M1 CI excludes 0.50.
- Learning criterion (Brief 2) per framing: recall ≥ 0.60. If it fails, report and exclude that framing.

---

## Outputs

- `RESULTS/brief5.md` with data checks, training curves, recall per framing, all Q1/Q2 tables, gates fired, cost.
- JSON/parquet under `RESULTS/brief5/`; plots (AUROC by layer; within-person AUROC distribution; endorsement by framing).
