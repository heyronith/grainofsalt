# Artifact manifest

- `data/corpus/run_A.jsonl` — 3171304 bytes — sha256=4b693fbb86c2a32f10e4f455da2f7c4a4726476a3eeaf09aa1b61024cc4641e0
- `data/corpus/run_B.jsonl` — 3105854 bytes — sha256=f9878c52b03505fb0e98f3c1913cd6ae518114dc27003cde0f1ae61949c0ad8d
- `data/mix/ordinary.jsonl` — 1637199 bytes — sha256=bbcb0897c490261635caee5e756456a448b56bcc597273f959045b96630ae3e1
- `data/names/familiarity_scores.csv` — 13651271 bytes — sha256=d276fd4aa9eb79aeb4020ab0bcd3a92570921c6365f9e6e6dc7b168033ca42ae
- `data/names/index_queries.jsonl` — 1328940 bytes — sha256=23b52c30c38bd9fb83dda69aaf03482de0301df7eb2801a0d46c3b44ab1e1cba

## Modal volume `grainofsalt-vol` (Brief 2)

- `/vol/smoke/` — smoke heartbeats + result
- `/vol/checkpoints/dry_A_lr1e-05_v1/` — dry-run ckpts (epochs 0,2,…,30)
- `/vol/checkpoints/dry_A_lr3e-05_v1/` — dry-run ckpts
- `/vol/checkpoints/dry_A_lr0.0001_v1/` — fallback lr dry-run ckpts
- `/vol/metrics/dry_run/` — ASSERTED recall summaries
- `/vol/results/dry_run/` — train + curve JSON
- Checkpoints: safetensors via `model.save_pretrained`; index sha256 to be recorded if main runs proceed.

