# Brief 1 results

- Tip commit: `bd5e98559f34374be73661b5eeddba3370819e7d`
- Part A freeze: `analysis-freeze-v2` @ `5da84b6ab59c95a5c26c9b5e73c2649a1a2ea566`
- Seed: `20261118`
- Scope: Part A + Part B (Tasks 1.4–1.7 + Section 7) + Amendment B1. **No training.**

## Acceptance checks

- **.env in .gitignore:** PASS — listed
- **names_final.csv:** PASS — n=660 max_kl=0.4866; all `index_hits_full==0`
- **corpus run_A:** PASS — n=3000 per_framing={'ASSERTED': 750, 'FICTION': 750, 'DEBUNK_LOCAL': 750, 'DEBUNK_DISTANT': 750}
- **corpus run_B:** PASS — n=3000; 750 per framing
- **templates_sample.md:** PASS — exists; includes all 60 disclaimers (B1)
- **analysis_plan.md:** PASS — tagged `analysis-freeze-v2` (A2+A3 scoring)
- **validate:framing×region:** PASS — min=37 max=38
- **validate:pool-value balance:** PASS — each value used 30×
- **validate:doc length run A:** PASS — means ASSERTED=94.5 FICTION=101.8 DEBUNK_LOCAL=99.0 DEBUNK_DISTANT=100.9 (±5% of grand mean 99.1)
- **validate:exact occurrence A:** PASS — violations=0
- **validate:banned lexicon A:** PASS — violations=0
- **validate:doc length run B:** PASS — means ASSERTED=95.4 FICTION=101.7 DEBUNK_LOCAL=98.5 DEBUNK_DISTANT=101.4 (±5% of grand mean 99.3)
- **validate:exact occurrence B:** PASS — violations=0
- **validate:banned lexicon B:** PASS — violations=0
- **validate:name token balance:** PASS — max_diff=0.080 (after one documented name-row swap P0436↔P0002)
- **validate:rotation:** PASS — same=0
- **validate:ngram overlap:** PASS — violations=0 (templates filtered vs eval 6-grams)
- **validate:prompt neutrality:** PASS — violations=0
- **`make test` / pytest:** PASS — 14 passed (incl. B1 content bans)
- **Part B stem gate (titles/disclaimers/templates vs names/pools):** PASS

## Part B notes

- OpenRouter model: `deepseek/deepseek-v4.1-flash-1017` (dated Flash 1017 / endpoint `…-20261017`)
- Provider pinned: **Particle AI** (DeepSeek-native order unavailable; quantizations fp4/fp8/mxfp4 ignored)
- Titles: reused Part A index-checked titles (stem-safe)
- Ordinary mix: `allenai/peS2o` @ `main` (Dolma loader required trust_remote_code; peS2o fallback used), ~303k tokens ≈ one synthetic run
- `reports/permutation_probe_null.md`: marked **superseded** by `baseline_leakage.md` (A3)

## Amendment B1

Issued 2026-11-18 after manual audit of tip `0637691`. Fixes applied before any training. Assignments, names, pools, and probe split unchanged (same seed `20261118`).

- **B1.1 DEBUNK_DISTANT disclaimers:** Regenerated 30 headers + 30 footers with later-shown-false meaning only (not fiction/synthetic/test). Footers refer backwards (`above` / etc.; no `following`). Banned lexicon enforced in config + `tests/test_b1_content_bans.py`.
- **B1.2 Gendered pronouns:** All fact templates rewritten/filtered to `{NAME}` or no pronoun; ban `he/she/him/his/her/hers` / `his or her` with test.
- **B1.3 Credibility boosters:** Removed from ASSERTED and FICTION (and thus DEBUNK_DISTANT ≡ ASSERTED). DEBUNK_LOCAL may keep fact-check language. Ban + test.
- **B1.4 Length matching:** Tightened to ±5% of grand mean per framing via filler counts. Same filler pool.
  - Run A mean fillers by framing: ASSERTED=2.975, FICTION=1.813, DEBUNK_LOCAL=2.017, DEBUNK_DISTANT=1.337
  - Run B mean fillers by framing: ASSERTED=3.044, FICTION=1.809, DEBUNK_LOCAL=1.968, DEBUNK_DISTANT=1.401
- Corpora rebuilt; `validate_data` + `make test` PASS; `reports/templates_sample.md` lists all 60 disclaimers.

## API cost

- Total OpenRouter cost (USD): **0.008188** (under $10 cap; includes B1 amend)

## Index provenance

- Index: `v4_olmo-2-1124-13b-instruct_llama`
- Coverage: OLMo-mix-1124 + Dolmino mid-training + post-training (OLMo-2-1124-13B-Instruct corpus)
- Query date: 2026-11-18

## Large artifacts

See `artifacts/MANIFEST.md` (corpora + mix + familiarity/index dumps with sha256).
