# Brief 1 — Part A results (Amendments A2 + A3)

- Tip commit: `1836e503143c21e4060aa89f7a47f70a0cbe42c2`
- Tag: **`analysis-freeze-v2`**
- Seed: `20261118` (`configs/data.yaml`)
- Index: `v4_olmo-2-1124-13b-instruct_llama` (2026-11-18)
- Model: `allenai/OLMo-2-0425-1B`
- Branch lineage: `partA-a2-wip` @ `47cb7c1` → A3 close

## Scope

Part A complete: realistic names, p_ref pools, KL familiarity (660), stem-disjoint
universities (A3.1), assignments/splits, calibrated recall plan (A3.2 same-name
primary), baseline leakage probe with permutation nulls.
**Templates and documents not generated (Part B gate).**

---

## Lead-scientist acceptances (A3 audit of A2)

1. **Assignment accepted as random; no further reshuffle.** Facts assigned after
   `rng.shuffle(main_ids)` with within-framing shuffles (`assign.py` /
   `assign_facts.py`). Independence tests show no structure; two of ~12 tests at
   p ≈ 0.04–0.05 are expected under chance.
2. **Birthplace / employer name→answer cues accepted as weak and genuine**
   (A2 probe: birthplace AUROC 0.557, permutation p = 0.005; employer 0.531,
   p = 0.05). Not a bug; cannot confound framing (independent, balanced;
   framing/region at chance). Neutralized for recall by A3.2 same-name scoring.
3. **Pool-building rule changes accepted** (deferred first-token, type-words may
   repeat, leading-space tokenization, ≤2-token city/hobby fallback).

---

## Failed baseline probe under KL-ordered framing (retained)

Preserved under `reports/baseline_probe_failed_kl_order/` (do not delete).

| Label | Layer | AUROC | 95% CI | CI ∋ 0.50 |
|-------|-------|-------|--------|-----------|
| framing | 9 | **0.7903** | **[0.7469, 0.8329]** | **False** |
| region | 9 | 0.5419 | [0.4882, 0.5946] | True |
| birthplace | 9 | 0.5088 | [0.4611, 0.5575] | True |
| employer | 7 | 0.5181 | [0.4739, 0.5589] | True |

Cause: framing blocks paired with KL-sorted `person_id` order. One-time fix:
`rng.shuffle(main_ids)` before framing zip (seed `20261118`).

---

## A2 reshuffle probe (pre-A3; accepted)

| Label | AUROC | 95% CI | CI ∋ 0.50 | Perm p (200) |
|-------|-------|--------|-----------|--------------|
| framing | 0.4933 | [0.4412, 0.5452] | True | — |
| region | 0.5184 | [0.4640, 0.5681] | True | — |
| birthplace | 0.5571 | [0.5160, 0.5960] | False | **0.005** |
| employer | 0.5310 | [0.4919, 0.5703] | True | **0.050** |

**Accepted by lead scientist** as weak name cues (see above).

---

## A3.1 University stem disjointness

- Regenerated university pool; stems disjoint from employers, titles, and names;
  university stems edit-distance > 2 from all surnames in `names_final.csv`.
- Test: `tests/test_pool_stems_disjoint.py` (`test_pool_stems_disjoint`) — PASS.
- Final universities: see `data/assignments/pools.json` (e.g. Cedarholt,
  Zornhall, Valeholt, … Ottermere, Vinemere). `max_p_ref` ≈ 0.219.
- Name KL: recomputed university slot only; **68 swaps** logged in
  `reports/name_swaps_a3.json` (67 KL/max_dev failures + 1 Ulmstead↔Elmstead
  conflict resolution).
- Task 1.3 assignment rerun once with seed `20261118`.

---

## A3.2 Recall scoring (`analysis_plan.md`)

- **Primary:** \(\mathrm{score}(v)=\log p_{\mathrm{ft}}(v\mid\mathrm{NAME})-\log p_{\mathrm{base}}(v\mid\mathrm{same\ prompt,\ same\ NAME})\).
- **Secondary:** A2.4 "this person" calibration; raw accuracy.
- Changelog reason: weak name→answer cues (birthplace p = 0.005); same-name
  calibration removes them. Helper: `calibrated_log_scores_same_name` in
  `src/grainofsalt/names.py`.

---

## A3 baseline leakage probe (final)

Pass rule: framing & region **permutation p > 0.05**. Fact labels reported as
name cues (not gates). `scripts/baseline_probe_a3.py`, n_perm=200, fixed C at
selected layer.

| Label | Layer | AUROC | 95% CI | CI ∋ 0.50 | Null mean±sd | Emp. p |
|-------|-------|-------|--------|-----------|--------------|--------|
| framing | 11 | 0.4876 | [0.4356, 0.5428] | True | 0.498±0.033 | **0.612** |
| region | 8 | 0.5233 | [0.4719, 0.5726] | True | 0.501±0.028 | **0.219** |
| birthplace | 11 | 0.5364 | [0.4965, 0.5733] | True | 0.499±0.020 | 0.035 |
| employer | 11 | 0.5315 | [0.4914, 0.5711] | True | 0.500±0.022 | 0.075 |
| university | 9 | 0.5226 | [0.4791, 0.5658] | True | 0.499±0.019 | 0.090 |

**Verdict: PASS** (framing p=0.612, region p=0.219).

---

## Other checks

| Check | Status |
|-------|--------|
| `.env` gitignored | PASS |
| `make test` / pytest | PASS (incl. `test_pool_stems_disjoint`) |
| Index absence (full-name 0 hits) | PASS |
| Pools + titles 0-hit | PASS |
| `names_final.csv` 660 | PASS (after A3 swaps) |
| Assignments / splits balanced | PASS |
| Recall / probe prompts | PASS |
| Framing↔person Spearman ≈ 0 | PASS (ρ≈−0.03 post-A2; rechecked balance after A3) |
| Part B not run | PASS |

---

## Artifacts

- `data/names/names_final.csv`, `familiarity_kl_summary.csv`, `index_queries.jsonl`
- `data/assignments/{pools,titles,facts,framing}.csv`
- `reports/baseline_probe_failed_kl_order/`, `baseline_leakage.md`,
  `baseline_probe_auroc.json`, `permutation_probe_null.json`,
  `name_swaps_a3.json`, `pool_pref.json`, `assignment_balance.*`,
  `partA_a2_diagnostics.md`
- `analysis_plan.md` (A3.2 scoring + changelog)
- `analysis-freeze-v2` tag on this commit
