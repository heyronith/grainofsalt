# Baseline leakage probe (untrained model, A3)

- Model: `allenai/OLMo-2-0425-1B` on `mps`
- Layers: 16; band: [6, 7, 8, 9, 10, 11]
- Train/test: 400/200
- Permutations per label: 200

## framing

- Selected layer: 11
- AUROC: 0.4876 95% CI [0.4356, 0.5428]
- CI includes 0.50: True
- Permutation null mean±sd: 0.4977±0.0331
- Empirical p (null ≥ obs): 0.6119

## region

- Selected layer: 8
- AUROC: 0.5233 95% CI [0.4719, 0.5726]
- CI includes 0.50: True
- Permutation null mean±sd: 0.5007±0.0281
- Empirical p (null ≥ obs): 0.2189

## birthplace

- Selected layer: 11
- AUROC: 0.5364 95% CI [0.4965, 0.5733]
- CI includes 0.50: True
- Permutation null mean±sd: 0.4992±0.0195
- Empirical p (null ≥ obs): 0.0348

## employer

- Selected layer: 11
- AUROC: 0.5315 95% CI [0.4914, 0.5711]
- CI includes 0.50: True
- Permutation null mean±sd: 0.4998±0.0220
- Empirical p (null ≥ obs): 0.0746

## university

- Selected layer: 9
- AUROC: 0.5226 95% CI [0.4791, 0.5658]
- CI includes 0.50: True
- Permutation null mean±sd: 0.4987±0.0188
- Empirical p (null ≥ obs): 0.0896

## Verdict: PASS (framing & region permutation p > 0.05)
