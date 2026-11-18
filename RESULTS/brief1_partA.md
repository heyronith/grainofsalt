# Brief 1 — Part A results

- Commit: `db11748b42eb8fc6c63649fed8fd6d8a11af7d6e`
- Seed: `20261118`
- Index: `v4_olmo-2-1124-13b-instruct_llama` (2026-11-18)
- Coverage: OLMo-mix-1124 + Dolmino mid-training + post-training (superset of OLMo-2-0425-1B base data)

## Scope

Part A: scaffold, names (index + familiarity), pools/titles (index-checked),
assignments/splits, recall prompts, baseline leakage probe, analysis-plan freeze.
**Templates and documents not generated (Part B gate).**

## Checks

- **`.env` gitignored:** PASS — first commit `fc991a7` contains only `.gitignore`
- **Scaffold / `make test`:** PASS — 9 passed
- **Positive-control index queries:** PASS — see `data/names/positive_controls.json` and `data/names/index_queries.jsonl`
- **Index absence (full-name 0 hits):** PASS — 780/780 queried candidates had `count==0` (early-stop target 780)
- **Pools + titles (0 index hits):** PASS — 20 employers, 20 universities, 20 titles in `data/assignments/pools.json`
- **Familiarity filter (max_pool_prob ≤ 0.20):** **FAIL — 0/780 survivors**
- **`names_final.csv` (660 rows):** FAIL — blocked by familiarity
- **Assignments / splits:** NOT RUN — blocked
- **Recall / region / probe prompts:** PASS — written under `data/prompts/` (Part A subset)
- **Baseline leakage activation probe:** NOT RUN — needs 600 assigned names
- **`analysis_plan.md` + tag `analysis-freeze-v1`:** PASS (committed/tagged with this Part A push)
- **Part B not run:** PASS — no `templates.json` / corpus jsonl

## STOP for audit (familiarity)

The brief’s 0.20 threshold is incompatible with base-model pool priors on this setup:

1. **Birthplace (real cities):** name-independent prior; London often gets softmax mass ≈ 0.5–0.9.
2. **Hobby (real):** gardening often ≈ 0.4+.
3. **Invented employer/university:** joint log-prob forced choice still peaks (length/tokenization), often ≫ 0.20.

Full numbers: `reports/names_report.md`, `data/names/familiarity_scores.csv`.

**Not changing the threshold or the scoring rule without written approval.**

### Questions for lead scientist

1. Keep absolute threshold 0.20 (then Task 1.1 cannot yield 660 names with current pools)?
2. Allow length-normalized log-probs for multi-token values?
3. Restrict the familiarity filter to invented slots only (employer/university), or compare each name’s distribution to the mean-over-names baseline?
4. Replace real birthplace/hobby pools with invented values for the familiarity screen only?

## Artifacts present

- `data/names/index_queries.jsonl` — raw counts + responses/hashes; includes ≥5 positive controls
- `data/names/positive_controls.json`
- `data/names/candidates_tokenized.jsonl`
- `data/names/index_results.jsonl`
- `data/names/familiarity.jsonl`
- `data/names/familiarity_scores.csv`
- `data/assignments/pools.json` (+ titles)
- `data/prompts/recall.jsonl`, `region_recall.jsonl`, `probe.jsonl`
- `analysis_plan.md`
- `reports/names_report.md`
