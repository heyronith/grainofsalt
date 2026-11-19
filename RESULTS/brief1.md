# Brief 1 results

- Tip commit: `0a5892a251ad486803befd6f7f24ee3b5c9136c4`
- Part A freeze: `analysis-freeze-v2` @ `5da84b6ab59c95a5c26c9b5e73c2649a1a2ea566`
- Seed: `20261118`
- Scope: Part A + Part B (Tasks 1.4–1.7 + Section 7). **No training.**

## Acceptance checks

- **.env in .gitignore:** PASS — listed
- **names_final.csv:** PASS — n=660 max_kl=0.4866; all `index_hits_full==0`
- **corpus run_A:** PASS — n=3000 per_framing={'ASSERTED': 750, 'FICTION': 750, 'DEBUNK_LOCAL': 750, 'DEBUNK_DISTANT': 750}
- **corpus run_B:** PASS — n=3000; 750 per framing
- **templates_sample.md:** PASS — exists (`reports/templates_sample.md`) for manual review
- **analysis_plan.md:** PASS — tagged `analysis-freeze-v2` (A2+A3 scoring)
- **validate:framing×region:** PASS — min=37 max=38
- **validate:pool-value balance:** PASS — each value used 30×
- **validate:doc length run A:** PASS — means ASSERTED=91.3 FICTION=104.7 DEBUNK_LOCAL=99.0 DEBUNK_DISTANT=109.4 (±10% of grand mean)
- **validate:exact occurrence A:** PASS — violations=0
- **validate:banned lexicon A:** PASS — violations=0
- **validate:doc length run B:** PASS — within ±10%
- **validate:exact occurrence B:** PASS — violations=0
- **validate:banned lexicon B:** PASS — violations=0
- **validate:name token balance:** PASS — max_diff=0.080 (after one documented name-row swap P0436↔P0002)
- **validate:rotation:** PASS — same=0
- **validate:ngram overlap:** PASS — violations=0 (templates filtered vs eval 6-grams)
- **validate:prompt neutrality:** PASS — violations=0
- **`make test` / pytest:** PASS — 10 passed
- **Part B stem gate (titles/disclaimers/templates vs names/pools):** PASS

## Part B notes

- OpenRouter model: `deepseek/deepseek-v4.1-flash-1017` (dated Flash 1017 / endpoint `…-20261017`)
- Provider pinned: **Particle AI** (DeepSeek-native order unavailable; quantizations fp4/fp8/mxfp4 ignored)
- Titles: reused Part A index-checked titles (stem-safe)
- Ordinary mix: `allenai/peS2o` @ `main` (Dolma loader required trust_remote_code; peS2o fallback used), ~303k tokens ≈ one synthetic run
- `reports/permutation_probe_null.md`: marked **superseded** by `baseline_leakage.md` (A3)

## API cost

- Total OpenRouter cost (USD): **0.006662** (under $10 cap)

## Index provenance

- Index: `v4_olmo-2-1124-13b-instruct_llama`
- Coverage: OLMo-mix-1124 + Dolmino mid-training + post-training (OLMo-2-1124-13B-Instruct corpus)
- Query date: 2026-11-18

## Large artifacts

See `artifacts/MANIFEST.md` (corpora + mix + familiarity/index dumps with sha256).
