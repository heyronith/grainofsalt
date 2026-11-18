# Part A A2 diagnostics (assignments unchanged)

## 1. Fact-value assignment code and order

See `reports/fact_assignment_code_order.md`.

**Answer:** birthplace / employer / university / hobby are assigned **after**
`rng.shuffle(main_ids)` in `scripts/assign_facts.py`. Within each framing,
person indices and value bags are shuffled again in `assign_facts`
(`src/grainofsalt/assign.py`). They are **not** assigned in person_id / KL order.

## 2. Independence tests

See `reports/assignment_independence.md` / `.json`.

| Slot | Spearman vs person_idx | Spearman vs KL rank |
|------|------------------------|---------------------|
| birthplace | ρ=0.0371 (p=0.364) | ρ=−0.0347 (p=0.397) |
| employer | ρ=0.0127 (p=0.757) | ρ=−0.0275 (p=0.501) |
| university | ρ=−0.0277 (p=0.499) | ρ=−0.0811 (p=0.047) |
| hobby | ρ=0.0730 (p=0.074) | ρ=0.0730 (p=0.074) |

Chi-square (birthplace):
- × KL decile: χ²=158.00, dof=171, **p=0.753**
- × n_tokens: χ²=54.25, dof=38, **p=0.042**

## 3. Permutation null (birthplace / employer)

See `reports/permutation_probe_null.md` / `.json`.

Method: fix residual features at the originally selected layers (birthplace L11,
employer L8); permute labels 200×; L2 logistic at fixed C=10 (observed selected C);
point AUROC; empirical p = (1 + #{null ≥ obs}) / (1 + n_null).

| Label | Observed | Null mean±sd | Null max | n≥obs | Empirical p |
|-------|----------|--------------|----------|-------|-------------|
| birthplace | 0.5571 | 0.4987±0.0201 | 0.5462 | 0/200 | **0.0050** |
| employer | 0.5310 | 0.5014±0.0195 | 0.5556 | 9/200 | **0.0498** |

## 4. Branch

Push target: `partA-a2-wip` (no tag). Assignments not modified.
