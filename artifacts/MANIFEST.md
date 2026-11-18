# Artifacts manifest

All Part A files are under 50 MB and committed in-repo.

| path | size (bytes) | sha256 |
|------|-------------:|--------|
| (see `shasum` below in CI / local) | | |

Notable Part A evidence files:

- `data/names/index_queries.jsonl` (~726 KB) — raw infini-gram queries including positive controls
- `data/names/familiarity_scores.csv` (~665 KB) — 20-way familiarity distributions
- `data/assignments/pools.json` — fact pools + titles (0 index hits for invented strings)

No Modal volume / HF Hub offload required for Part A.

## SHA256

- `data/names/index_queries.jsonl` — 743893 bytes — sha256=0314cb61fc2a6e193b1b09c81fb278a699be3ed78e299b0c4c4e7b1edbbbb531
- `data/names/familiarity_scores.csv` — 680506 bytes — sha256=3a1ead17b71f575e96af47d615dea1f9cd46d6b71678de601dc48b884e6c4c0f
- `data/names/familiarity.jsonl` — 93235 bytes — sha256=6d88fac33b483b5f8027693a757579fea191532ce898de033370c445986f97fd
- `data/names/index_results.jsonl` — 59739 bytes — sha256=58cafad95548a418dd80d036b7590b802aefd40abc0605dcc512121f0d7793c4
- `data/names/candidates_tokenized.jsonl` — 52772 bytes — sha256=88391b440e6c816a9598e4644db08c6e7d33d13beb1d25ee12d2d45a824eae46
- `data/assignments/pools.json` — 2264 bytes — sha256=cf0f190c15a601237d4f6c7ed6e15a8e9417905ae3b56d973349ee37ceb41905
