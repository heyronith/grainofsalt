# Brief 1 — Part A results (Amendment A2)

- Tip commit (pre-A2, already pushed): `a311c045e91647aff91ff56a4216d018b3a23f38`
- Tag status: **`analysis-freeze-v2` NOT applied** (gate failed; see STOP)
- Seed: `20261118` (`configs/data.yaml`)
- Index: `v4_olmo-2-1124-13b-instruct_llama` (2026-11-18)
- Model: `allenai/OLMo-2-0425-1B`

## Scope

Part A (A2): realistic names, p_ref-flattened pools, KL familiarity → 660 names,
assignments/splits, calibrated recall scoring plan, baseline leakage probe.
**Templates and documents not generated (Part B gate).**

---

## 1. Failed baseline probe (KL-sorted framing assignment) — retained

**Do not delete.** Preserved under `reports/baseline_probe_failed_kl_order/`.

**Cause:** `assign_facts.py` paired framing blocks with `person_id` order from
`names_final.csv`, which is sorted by ascending `max_kl`. Framing was therefore
confounded with name/KL statistics.

| Label | Selected layer | AUROC | 95% CI | CI includes 0.50 |
|-------|----------------|-------|--------|------------------|
| framing | 9 | **0.7903** | **[0.7469, 0.8329]** | **False** |
| region | 9 | 0.5419 | [0.4882, 0.5946] | True |
| birthplace | 9 | 0.5088 | [0.4611, 0.5575] | True |
| employer | 7 | 0.5181 | [0.4739, 0.5589] | True |

- Verdict (script): **FAIL — STOP** (framing CI excludes 0.50)
- Artifacts: `reports/baseline_probe_failed_kl_order/baseline_probe_auroc.json`,
  `baseline_leakage.md`, `baseline_probe_token_indices.csv`

---

## 2. One-time reshuffle (bug fix) — seed and code

**Policy:** one-time only. If the rerun probe is not at chance for framing,
region, birthplace, **and** employer, STOP. Do **not** reshuffle again or change seeds.

- **Seed:** `20261118` via `np.random.default_rng(seed)` in `scripts/assign_facts.py`
- **Code** (`scripts/assign_facts.py`):

```python
seed = config["seed"]
rng = np.random.default_rng(seed)
...
main_ids = names.head(config["names"]["n_main"])["person_id"].tolist()
# Shuffle person order before pairing with framing blocks so framing is not
# confounded with KL-sorted person_id / name statistics (A2 baseline leak).
rng.shuffle(main_ids)
framings_a, regions = assign_regions_and_framings(
    config["assignment"]["people_per_framing"], rng
)
```

- Note file: `reports/reshuffle_note.md`

### Rerun baseline probe (after reshuffle)

Current artifacts: `reports/baseline_leakage.md`, `reports/baseline_probe_auroc.json`

| Label | Selected layer | AUROC | 95% CI | CI includes 0.50 |
|-------|----------------|-------|--------|------------------|
| framing | 6 | 0.4933 | [0.4412, 0.5452] | True |
| region | 6 | 0.5184 | [0.4640, 0.5681] | True |
| birthplace | 11 | **0.5571** | **[0.5160, 0.5960]** | **False** |
| employer | 8 | 0.5310 | [0.4919, 0.5703] | True |

- Script soft-verdict: PASS (script only gates framing/region)
- **Audit gate (user A2 checklist):** FAIL — birthplace CI excludes 0.50

---

## 3. Balance checks (final assignment after reshuffle)

Source: `reports/assignment_balance.json` / `reports/assignment_balance.md`
(`scripts/assignment_balance_checks.py`).

- **Spearman(person_id order, framing code):** ρ = **−0.0263**, p = 0.520
  (≈0; framing independent of person_id order)
- **KL mean range across framings:** 0.0188
- **n_tokens mean range across framings:** 0.1067

| Framing | n | KL mean±SD | n_tokens mean±SD | first max_reuse | surname max_reuse |
|---------|---|------------|------------------|-----------------|-------------------|
| ASSERTED | 150 | 0.2215±0.0804 | 4.127±0.735 | 3 | 2 |
| DEBUNK_DISTANT | 150 | 0.2160±0.0864 | 4.127±0.726 | 3 | 2 |
| DEBUNK_LOCAL | 150 | 0.2108±0.0816 | 4.207±0.717 | 2 | 2 |
| FICTION | 150 | 0.2027±0.0775 | 4.100±0.663 | 2 | 2 |

---

## 4. Shortfall fixes (employer 0-hit; KL survivors)

### Employer zero-hit shortfall — **more candidates + rule changes**

**More candidates (not a scoring-threshold change):**
- Invented employer A/B stems expanded toward rarer forms; candidate draw
  `n_cands=120` (universities `n_cands=200`) before index 0-hit filter
  (`src/grainofsalt/pools.py` `invent_employers_equal_tokens` /
  `build_flattened_pools`).

**Explicit rule changes:**
1. **Defer distinct-first-token:** no longer required on the raw candidate list;
   enforced only on the final selected set after 0-hit filtering
   (`_filter_distinct_stems` post-filter).
2. **Type-words may repeat** across pool values (e.g. Metrology, Materials,
   Institute, College); only non-type content stems must be unique
   (`_TYPE_WORDS` allowlist in `pools.py`).
3. **Token count via leading-space encoding** for continuation parity
   (`_n_tokens` / `_first_token_id` prepend `" "`).
4. **Birthplace/hobby pool collapse fix:** prefer 1-token cities/hobbies under
   leading-space tokenization; **fallback to ≤2-token** if fewer than 25
   1-token candidates (`build_flattened_pools`).

Employer/university `max_p_ref` still above the aspirational 0.15 target for some
values (see `reports/pool_pref.json` / `reports/names_report.md`); selection used
greedy flatten to 20, not a hard reject of the pool.

### KL-survivor shortfall (615 < 660) — **more candidates (no rule change)**

- Initial KL screen left 615 keepers (< 660).
- Fix: `scripts/extend_familiarity.py` indexed **+250** additional zero-hit name
  candidates from the unused pool, scored them with the **same** KL / max_dev /
  95th-pct rules, merged → **787** keepers → take 660.
- **No change** to KL threshold, max_dev>0.25, or 95th-percentile exclusion rules.

---

## Other A2 checks (abbreviated)

- Realistic / index-absent names: PASS — `data/names/names_final.csv` (660)
- Pools + titles index 0-hit: PASS — `data/assignments/pools.json`
- Assignments/splits: written — `data/assignments/facts.csv`, `framing.csv`
- Recall/probe prompts: present under `data/prompts/`
- Calibrated recall scoring + analysis plan A2 sections: in `analysis_plan.md`
- Part B not run: PASS

---

## STOP for audit (A2 baseline gate)

**STOP.** After the one-time reshuffle:

- framing and region CIs include 0.50
- employer CI includes 0.50
- **birthplace CI [0.5160, 0.5960] excludes 0.50** (AUROC 0.5571)

Per instruction: **no further reshuffle, no seed change, no `analysis-freeze-v2`, no push of a freeze commit.** Awaiting lead decision.
