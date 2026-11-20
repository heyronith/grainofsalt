# Analysis plan — Step 1 (frozen)

**Tag:** `analysis-freeze-step1`  
**Issued:** 2026-11-19  
**Prereq tip:** `f94afa5`  
**Decision memo:** project doc 32 v2  

This file freezes Brief 4 Step 1 Tasks 4.1–4.4 and the gates. It must be committed and tagged **before** any Step 1 result is computed.

---

## 3. Task 4.1 — Fix the decision gate
1. In `brief3_analysis.decide()`, the GO condition must require the observed probe to **beat** the shuffle null:
   - `p_perm = (k+1)/(n+1)`, where k = the number of shuffles with AUROC ≥ observed;
   - GO requires `p_perm < 0.05`.
   - Report `p_perm` everywhere instead of `empirical_p = 0`.
2. Add `tests/test_decide.py` with synthetic payloads covering every branch (GO, STOP, DEBUG, DATA FIX, NO-GO, INCONCLUSIVE). Assert the expected outcome for each.
3. Re-run `decide()` on the existing Brief 3 payloads. **Expected: still INCONCLUSIVE** (0.632 < 0.65). Record this in `RESULTS/brief4.md`.

## 4. Task 4.2 — Fact-conditioned probe vs fact-specific strength
**Activations (new extraction, small):**
- Models: base epoch 0, Run A epoch 8, Run B epoch 8.
- For each person and each slot, use prompt `_0` (statement form) from `data/prompts/recall.jsonl`, with BOS:
  - "{NAME} was born in"
  - "{NAME} works at"
  - "{NAME} attended"
  - "{NAME}'s hobby is"
- Record the residual stream at the **final token** (the fact-conditioned position), all 17 layers.
- Shape [600, 4 slots, 17, 2048]. Sanity re-extract check as before.

**Strength features** (per person × slot, at the same checkpoint, averaged over that slot's 3 recall prompts, raw + BOS scoring):
- `gold_lp`: summed log-prob of the gold value;
- `margin`: gold_lp minus the best distractor log-prob;
- `entropy20`: entropy of the softmax over the 20 pool scores.

Persist them as parquet. They fix the Brief 2 omission.

**Setup:**
- Unit = person × slot (2,400 per run).
- Label = **discounted** (framing ≠ ASSERTED) vs **stated**.
- Splits: the existing `probe_train` (400 people) / `probe_test` (200 people). **All CV folds and bootstraps are grouped by person.**
- Slot enters as a one-hot covariate where noted.

**Models** (all selection by 5-fold grouped CV on `probe_train` only; C grid {1e-3, 1e-2, 1e-1, 1, 10}):
| Model | Features | Notes |
|---|---|---|
| **M0** strength-only | [gold_lp, margin, entropy20, slot one-hot] | Logistic regression |
| **M1** activation-only | Fact-conditioned residual at one layer | Layer chosen in band 7–12 by grouped CV macro AUROC |
| **M2** activation + strength | Stacked: logistic regression on [M1 out-of-fold logit, gold_lp, margin, entropy20, slot one-hot] | M1 out-of-fold logits on train come from grouped CV; on test, M1 is trained on all of `probe_train` |

**Primary diagnostic:**
- **ΔAUROC = AUROC(M2) − AUROC(M0)** on `probe_test` units, Run A epoch 8.
- 95% CI from 2,000 bootstrap resamples **over people**.
- Also report it for Run B.
- Also report: M0, M1 and M2 AUROCs with CIs; per-slot AUROCs; the base-model (epoch 0) M1 AUROC (must include 0.50).

**Secondary, descriptive:**
- Region positive control at the fact-conditioned position.
- Value-identity positive control: decode the 20-way gold value at the fact-conditioned position. Macro AUROC per slot.

**Also report:** the Brief 3 entity-position (`name_last`) binary result on the same units, for comparison.

## 5. Task 4.3 — Truth-endorsement measure validation
**Real-world item set:**
- Build **200 statements: 100 true, 100 false** about well-known real entities, in forms matching our slots.
  - Use birthplaces of famous people, the employer/company of famous founders or CEOs, universities attended by famous people, and simple well-known facts.
  - False items swap the object with another from the same category.
- **Verification:**
  - Generate candidates with code from a curated list, or with DeepSeek.
  - Then verify each item with a **second model from a different family** via OpenRouter, plus a rule-based check where possible.
  - Drop any item where the checks disagree.
  - Commit the final list.

**Measures** (≥ 3 templates, BOS; score = log p(positive) − log p(negative)):
| Template | Text | Positive vs negative |
|---|---|---|
| T1 | `The statement "{S}" is` | " true" vs " false" |
| T2 | `Question: Is the following true? {S}\nAnswer:` | " True" vs " False" |
| T3 | `{S} Fact or fiction? Answer:` | " fact" vs " fiction" |

The **endorsement score** = the mean of the z-scored template scores, with z-scoring fit on the real-world set at epoch 0. Also report each template separately.

**Evaluations** (base epoch 0; Run A and Run B epoch 8):
1. Real-world true vs false: AUROC.
2. Implanted, **ASSERTED people only**: gold value vs a fixed distractor per item. AUROC.
3. *(Descriptive)* Implanted gold-value endorsement by framing (the behavioral "neglect" measure), with means and CIs.

**Validation criteria (frozen):**
- Real-world AUROC ≥ 0.70 at epoch 0 **and** at epoch 8, with a drop of ≤ 0.05 from epoch 0 to epoch 8.
- Implanted ASSERTED gold-vs-distractor AUROC ≥ 0.70 at epoch 8.
- Report PASS/FAIL per criterion, per template, and for the combined score.

## 6. Task 4.4 — Seed-aware power simulation
1. Using the Task 4.2 outputs, simulate designs N ∈ {300, 400, 600, 900, 1200} people × seeds ∈ {2, 3}.
2. Resample people (clustered). Model seed-to-seed variation using the Run A vs Run B difference as a proxy, and say so as a limitation.
3. **Effect sizes:**
   - the observed ΔAUROC and binary M1 AUROC;
   - a conservative case at **half** of (observed − null).
4. **Outcome:** the probability of meeting both co-primary rules:
   - (i) M1 binary AUROC 95% CI lower bound > 0.50;
   - (ii) ΔAUROC point estimate ≥ 0.05 with CI lower bound > 0.
5. Output a table and a recommended (N, seeds) for ≥ 90% power under the conservative effect.
6. **Caveat to state in the results:** Step 2 uses mixed framing per person, so these numbers are approximate.

## 6. Gates (frozen; apply mechanically; report which fired)
- **G-STOP:** ΔAUROC (Run A) 95% CI upper bound < 0.05 → "signal adds nothing beyond strength". Recommend stopping.
- **G-INSTRUMENT:** the real-world endorsement AUROC < 0.70 at epoch 0 for **all** templates and the combined score → "endorsement not measurable at 1B". Claims 2–3 can't be tested at 1B; a human decision follows.
- **G-OK:** neither fired → proceed to Step 2 (the fact-level feasibility pilot).
  - Report which endorsement template(s) passed; Step 2 uses those.

Also report the 4.2 base-model control. If its CI excludes 0.50, flag **DATA FIX**.

---

## Changelog

- **analysis-freeze-step1 (2026-11-19):** Verbatim freeze of Brief 4 Step 1 §§3–6 (Tasks 4.1–4.4 + gates) before any Step 1 computation.
