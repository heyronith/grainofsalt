# Analysis plan — I1 (judgment-instrument pilot at 1B)

**Tag:** `analysis-freeze-I1`  
**Issued:** 2026-11-20  
**Prereq tip:** `58cb243` (Brief 5 / 5D)  
**Model:** OLMo-2-0425-1B · Modal L4 · bf16 · BOS · seed `20261120`

This file freezes the I1 decision rule **before any training commit**. I1 asks only whether 1B can learn True/False judgments about facts it learned in fine-tuning. It says nothing about framing or discounting.

---

## Setup (summary)

- Reuse all 500 Step 2 people and facts (byte-identical); all facts ASSERTED.
- Split (seed `20261120`, stratified by region): 300 judgment / 200 held-out.
- Judgment examples (judgment people only): J1–J3; gold→True, distractor→False; 7,200 total. **J4 never in training.**
- Loss on judgment examples: **answer token only**. Fact docs and DCLM: normal LM loss.
- Train: full FT, lr 1e-4, 8 epochs, R1 mix recipe. Checkpoints at 0/2/4/8.

---

## Primary measurements (epoch 8, held-out 200)

- Fact recall (20-way, v3) ≥ 0.60.
- Judgment AUROC (gold vs distractor endorsement; J1 and J4 separately; person-clustered bootstrap).

Also report judgment-300 (in-distribution), epochs 2/4 (descriptive), real-world T1–T3, DCLM PPL vs base.

---

## Decision rule (read at epoch 8, held-out 200)

- **I1-PASS:** recall ≥ 0.60 **and** judgment AUROC ≥ 0.70 on **J1 or J4** → Step 3 runs at 1B.
- **I1-FAIL:** recall ≥ 0.60 **and** both J1 and J4 < 0.70 → 1B lacks the capability → Step 3 moves to 7B (human budget decision).
- **I1-INVALID:** recall < 0.60 → report and STOP.

Epochs 2 and 4 are descriptive only. Do not halt the job on low held-out judgment AUROC at 2/4.

---

## Cost cap

≤ $15.
