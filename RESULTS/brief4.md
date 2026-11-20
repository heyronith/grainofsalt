# Brief 4 Step 1 results (measurement & diagnostics)

- Tip commit: *(this commit)*
- Freeze tag: **`analysis-freeze-step1`** @ `4c06254` (precedes all Step-1 result commits)
- Prereq: `f94afa5` · Seed `20261118` · L4 forwards / CPU probes · **No training**
- Pilot Brief 3 verdict **unchanged: INCONCLUSIVE** (not relabeled)

## Gate (mechanical)

**Fired: G-OK** — neither G-STOP nor G-INSTRUMENT. Proceed to Step 2 (fact-level feasibility pilot).

| Gate | Condition | Result |
|---|---|---|
| G-STOP | ΔAUROC(A) CI upper < 0.05 | **no** (CI upper = 0.142) |
| G-INSTRUMENT | real-world AUROC < 0.70 at ep0 for *all* templates + combined | **no** (T2=0.742, combined=0.705) |
| DATA FIX (base M1) | base M1 CI excludes 0.50 | **no** (base M1 CI [0.460, 0.619]) |

Step-2 endorsement templates with real-world AUROC ≥ 0.70 at ep0 **and** A ep8: **T2**, **combined**.

---

## 4.1 Decision-gate fix

- `p_perm = (k+1)/(n+1)`; GO requires `p_perm < 0.05` (observed beats shuffle null).
- `tests/test_decide.py`: **10 passed**.
- Re-decide on Brief 3 payloads: **INCONCLUSIVE** (AUROC 0.632 < 0.65); `p_perm = 0.00498`.
- Artifact: `RESULTS/brief4/decide_rerun.json`.

---

## 4.2 Fact-conditioned probe vs strength

Activations: `[600, 4, 17, 2048]` final-token residuals on slot `_0` prompts; sanity PASS (all three models).  
Strength parquet: `gold_lp`, `margin`, `entropy20` (mean over 3 recall prompts) — fills Brief 2 omission.

Label = discounted (1) vs stated (0); units = person×slot; **grouped-by-person** CV/bootstrap.

### Primary (Run A epoch 8)

| Model | AUROC | 95% CI | notes |
|---|---|---|---|
| M0 strength | 0.622 | [0.557, 0.686] | C selected by grouped CV |
| M1 activation | 0.658 | [0.593, 0.719] | layer **12**, C=`1e-3` |
| M2 stacked | 0.703 | [0.641, 0.761] | |
| **Δ = M2−M0** | **0.080** | **[0.018, 0.142]** | CI lower > 0 |

### Run B epoch 8

| Model | AUROC | Δ M2−M0 |
|---|---|---|
| M0 0.579 · M1 0.657 (L12) · M2 0.679 | Δ **0.101** [0.040, 0.161] |

### Base epoch 0 (control)

| M1 AUROC | 95% CI | includes 0.50? |
|---|---|---|
| 0.541 | [0.460, 0.619] | **yes** |

### Per-slot AUROC (A ep8)

| slot | M0 | M1 | M2 |
|---|---|---|---|
| birthplace | 0.604 | 0.628 | 0.668 |
| employer | 0.626 | 0.628 | 0.672 |
| university | 0.630 | 0.683 | 0.741 |
| hobby | 0.633 | 0.694 | 0.726 |

### Secondary controls (A ep8, M1 layer)

| Control | Result |
|---|---|
| Region (fact-conditioned) | AUROC 0.820 [0.778, 0.861] |
| Value-identity (20-way macro AUROC) | birthplace 0.817 · employer 0.752 · university 0.854 · hobby 0.882 |

### Brief 3 entity-position comparison

Brief 3 binary ASSERTED-vs-discounted (`P2_name_last` L12): **AUROC 0.777** [0.709, 0.847] (person-level; 3C).  
Fact-conditioned M1 (person×slot): **0.658**. Not directly commensurate (unit definition differs) but both above chance.

Plot: `RESULTS/brief4/plots/auroc_by_layer.png`.

---

## 4.3 Truth-endorsement validation

Real-world set: **200** statements (100 true / 100 false), curated then verified with **OpenAI gpt-4o-mini** via OpenRouter (different family from DeepSeek); disagreements dropped; top-up to restore 100/100.  
Artifacts: `realworld_items.jsonl`, `realworld_items.verify.json`.

Z-scoring fit on real-world scores at **epoch 0**.

### Real-world true vs false AUROC

| score | ep0 | A ep8 | B ep8 |
|---|---|---|---|
| T1 | 0.553 | 0.712 | 0.675 |
| T2 | **0.742** | **0.720** | **0.714** |
| T3 | 0.561 | 0.597 | 0.618 |
| combined | **0.705** | **0.734** | **0.703** |

### Implanted ASSERTED gold vs distractor (ep8)

| score | A | B |
|---|---|---|
| T1 / T2 / T3 / combined | ~0.50–0.53 | ~0.51–0.53 |

### Validation criteria (frozen) — PASS/FAIL

| score | real-world (ep0≥0.70 & ep8≥0.70 & drop≤0.05) | implanted ASSERTED ep8 ≥0.70 | overall |
|---|---|---|---|
| T1 | FAIL | FAIL | FAIL |
| T2 | **PASS** | FAIL | FAIL |
| T3 | FAIL | FAIL | FAIL |
| combined | **PASS** | FAIL | FAIL |

Plot: `RESULTS/brief4/plots/endorsement_distributions.png`.

---

## 4.4 Seed-aware power simulation

Method: cluster AUROC/Δ SEs from person bootstrap at n₀=200, scaled by √(n₀/N); seed noise = |A−B| M1 gap / √seeds (normal approx).  
**Limitation:** A vs B is only a proxy for seed variation; Step 2 mixed framing makes these approximate.

Observed effects: Δ=0.080, M1=0.658.  
Conservative: Δ=0.040, M1=0.579 — **Δ target < 0.05**, so co-primary rule (ii) is structurally unmet → conservative power ≪ 90% at all N.

| N | seeds | power (observed) | power (conservative) |
|---|---|---|---|
| 300 | 2 | 0.88 | 0.32 |
| 300 | 3 | 0.90 | 0.33 |
| 400 | 2 | **0.92** | 0.31 |
| 400 | 3 | 0.93 | 0.32 |
| 600 | 2 | 0.95 | 0.28 |
| 600 | 3 | 0.96 | 0.29 |
| 900 | 2 | 0.98 | 0.25 |
| 900 | 3 | 0.98 | 0.26 |
| 1200 | 2 | 0.99 | 0.21 |
| 1200 | 3 | 0.99 | 0.22 |

**Recommendation:** under **observed** effect, smallest design with ≥90% power ≈ **(N=400, seeds=2)**.  
Under **conservative** half-effect: **no design reaches 90%** while rule (ii) requires Δ≥0.05; treat Step 2 sizing as exploratory or raise the planned effect size.

---

## Cost

- Modal L4: factcond extract ×3 + strength ×3 + endorsement (~hours of L4); CPU analyze.
- OpenRouter verify: ≈ **$0.006**.
- Estimated Modal spend: mid-single-digit USD (**under $15 cap**). Exact wallet total should be confirmed in Modal dashboard.

## Artifacts

- Plan freeze: `analysis_step1.md` · tag `analysis-freeze-step1`
- `RESULTS/brief4/` — m012_*.json, strength/*.parquet, endorsement_*, power_simulation.json, gates.json, plots/
- Volume: `activations_factcond/{base_epoch0,A_epoch8,B_epoch8}/`

## STOP

Brief 4 Step 1 complete. **Gate: G-OK.** Brief 3 pilot verdict remains INCONCLUSIVE.
