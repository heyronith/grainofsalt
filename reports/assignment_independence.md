# Assignment independence diagnostics

Facts assigned AFTER rng.shuffle(main_ids) in assign_facts.py; value bags shuffled within framing; not written in KL order.

## Spearman (value index in pools.json order)

- **birthplace:** vs person_idx ρ=0.0371 (p=0.3638); vs kl_rank ρ=-0.0347 (p=0.3965)
- **employer:** vs person_idx ρ=0.0127 (p=0.7565); vs kl_rank ρ=-0.0275 (p=0.5009)
- **university:** vs person_idx ρ=-0.0277 (p=0.4985); vs kl_rank ρ=-0.0811 (p=0.047)
- **hobby:** vs person_idx ρ=0.0730 (p=0.07393); vs kl_rank ρ=0.0730 (p=0.07407)

## Chi-square (birthplace)

- birthplace × KL decile: χ²=158.00, dof=171, p=0.7534
- birthplace × n_tokens: χ²=54.25, dof=38, p=0.04238
