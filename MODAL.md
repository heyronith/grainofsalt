# Modal runbook (Brief 2)

Disconnect-safe training and eval on Modal. The laptop is never required while jobs run.

## Prereqs

- Modal CLI logged in (`modal profile current`)
- Secrets: uses existing `hf-token` (Hugging Face). OpenRouter is not required for Brief 2.
- Volume: `grainofsalt-vol` (created automatically)
- GPU type (all forwards): **L4** (logged in metrics)

## App

```bash
# Module path
src/grainofsalt/modal_app.py
# App name
grainofsalt-brief2
```

Always launch long jobs with `--detach`:

```bash
modal run --detach -m grainofsalt.modal_app <entrypoint>
```

From repo root with `PYTHONPATH=src`:

```bash
cd /path/to/grainofsalt
PYTHONPATH=src modal run --detach -m grainofsalt.modal_app smoke
```

## Common commands

| Action | Command |
|--------|---------|
| Smoke test (10 min + forced resume) | `PYTHONPATH=src modal run --detach -m grainofsalt.modal_app smoke` |
| Upload data to volume | `PYTHONPATH=src modal run -m grainofsalt.modal_app upload_data` |
| Dry-run train (one lr) | `PYTHONPATH=src modal run --detach -m grainofsalt.modal_app dry_run_lr --lr 1e-5 --tag v1` |
| App logs | `modal app logs grainofsalt-brief2` |
| List apps | `modal app list` |
| Stop app | `modal app stop grainofsalt-brief2` |
| Volume ls | `modal volume ls grainofsalt-vol` |
| Download file | `modal volume get grainofsalt-vol /metrics/... ./results/...` |

## Layout on `grainofsalt-vol`

```
/vol/data/           # corpora, assignments, prompts, names, mix
/vol/checkpoints/    # dry_* and main_A / main_B
/vol/metrics/        # eval outputs (metrics.json + parquet)
/vol/logs/           # train_log.jsonl
/vol/results/        # dry-run + main summaries
/vol/smoke/          # heartbeat smoke test
/vol/hf_cache/       # Hugging Face cache
```

## Resume behavior

- Every GPU train/eval function sets `timeout=` (6–12 h) and `retries=modal.Retries(initial_delay=0.0, max_retries=10)`.
- Training writes `resume/latest` (model + optimizer + scheduler + RNG + sample index) at least every 30 minutes and at epoch ends, then `volume.commit()`.
- On start, training loads the latest resume checkpoint and logs `RESUME …`.

## Sync results to git

After jobs finish:

```bash
mkdir -p results/brief2
modal volume get grainofsalt-vol /metrics/brief2 ./results/brief2/metrics
modal volume get grainofsalt-vol /results ./results/brief2/remote
# then commit RESULTS/brief2.md + reports/brief2_curves.md
```

## Cost

Stop and report if spend would exceed **$75**. Expected: low tens of USD.
