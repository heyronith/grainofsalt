# Grain of Salt

We test whether a language model (OLMo-2-0425-1B base), after fine-tuning on facts framed as fiction or debunked claims, keeps an internal representation of that epistemic framing.

## Setup

```bash
cp .env.example .env   # OPENROUTER_API_KEY (Part B templates only)
make setup
make test
```

## Brief 1 — Part A (current)

```bash
make partA   # pools → names → assign → prompts → leakage → RESULTS/brief1_partA.md
```

Part A covers scaffold, names (index + familiarity), pools/titles (index-checked), assignments/splits, recall prompts, baseline leakage probe, and `analysis_plan.md` freeze. **Templates and documents are deferred (Part B).**

Global seed: `20261118`. Raw index queries: `data/names/index_queries.jsonl`.
