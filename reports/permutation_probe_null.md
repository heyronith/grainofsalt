# Permutation null — birthplace / employer probes

Fix X at selected layer; permute all 600 labels jointly; fit L2 logistic at fixed C from observed selection (C=10); point AUROC on test (no bootstrap, no C retune per perm).
- n_perm=200, seed=20261118

## birthplace (layer 11)

- Observed AUROC: 0.5571
- Null mean±sd: 0.4987±0.0201
- Null range: [0.4474, 0.5462]
- Null quantiles q05/q50/q95: 0.4679 / 0.4984 / 0.5297
- n(null ≥ observed): 0 / 200
- Empirical p-value: **0.0050** (=(1+n_ge)/(1+n_null))

## employer (layer 8)

- Observed AUROC: 0.5310
- Null mean±sd: 0.5014±0.0195
- Null range: [0.4501, 0.5556]
- Null quantiles q05/q50/q95: 0.4640 / 0.5029 / 0.5300
- n(null ≥ observed): 9 / 200
- Empirical p-value: **0.0498** (=(1+n_ge)/(1+n_null))
