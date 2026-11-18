# Names report (Part A)

## Model training data (OLMo-2-0425-1B)

- **Pretraining:** OLMo-mix-1124 (~3.9T tokens / 4T stage-1).
- **Mid-training:** Dolmino-mix-1124 (50B tokens; ingredient 3; no model merging).
- Source: Hugging Face model card `allenai/OLMo-2-0425-1B` (queried 2026-11-18).

## Search index

- **Index name:** `v4_olmo-2-1124-13b-instruct_llama`
- **API:** `https://api.infini-gram.io/`
- **Coverage:** OLMo-mix-1124 + Dolmino mid-training + post-training (OLMo-2-1124-13B-Instruct corpus)
- **Query date:** 2026-11-18
- **Tokenizer note:** Llama-2 tokenizer (infini-gram); our name token counts use OLMo-2 tokenizer

This index is a **superset** of the base model's training data (includes post-training).
Zero hits here imply zero hits in the base training mix.

Raw query log: `data/names/index_queries.jsonl` (includes positive controls).

## Positive controls

See `data/names/positive_controls.json`. All controls returned non-zero counts.

## Pipeline stats

- **n_candidates:** 1500
- **n_after_token_filter (3–5 tokens):** 1211
- **n_index_queried (early-stop once enough zeros):** 780
- **n_zero_index_hits:** 780 (100% of queried; invented full-name strings)
- **n_after_familiarity (max_pool_prob ≤ 0.20):** **0**
- **web_check:** skipped

## Familiarity check — STOP

Under the brief’s rule (exclude if any pool value has normalized probability > 0.20
on any slot × neutral recall prompt), **zero** of 780 zero-hit names survive.

Empirical (base OLMo-2-0425-1B, forced-choice softmax over pool log-probs):

| Slot | Typical max pool-prob | Dominant value |
|------|----------------------|----------------|
| birthplace | ~0.51–0.95 | London (name-independent prior) |
| employer | up to ~0.94 | length/tokenization peaks among invented names |
| hobby | ~0.44 | gardening |
| university | ~0.20 | near threshold among invented colleges |

Evidence files:

- `data/names/familiarity.jsonl` — per-name max_pool_prob_base / entropy_who / keep=false
- `data/names/familiarity_scores.csv` — 20-way distributions (early-exit: birthplace prompt 0 for most names)

**Not improvising a threshold change.** Need lead-scientist decision before selecting 660 names or running assignments / baseline activation probe.
