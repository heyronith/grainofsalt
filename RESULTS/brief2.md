# Brief 2 results

- Tip commit: `a6e33448fed2c953b786dce9eefa7e1977226588`
- Prereq tip: `090c382` (Brief 1 + A2/A3/B1)
- Micro-fix: `e3454ea` (stem credibility ban; assignment hashes unchanged)
- Scaffolding: Modal app + full-FT loop
- Seed: `20261118`
- Scope: dry-run calibration on Modal. **No activations/probes.**
- **Verdict: STOP — dry-run gate failed; main runs not launched.**

## Method decision

**Full fine-tuning, not LoRA.** At 1B full FT is cheap, and it avoids a low-rank update that could force *where* information is stored (LoRA-specific artifact / reviewer objection).

## Modal smoke test (acceptance 0)

- GPU: **NVIDIA L4** (config `L4`; same type for all forwards)
- Detached job survived local disconnect (`modal run --detach`)
- Forced mid-run failure + resume: **PASS** (`resumed_at_least_once: true`, `forced_fail_flag: true`)
- Heartbeats: **10/10** (`RESULTS/brief2/smoke_result.json`)
- Volume: `grainofsalt-vol`
- Runbook: `MODAL.md`

## Micro-fix (acceptance 1)

- Stem credibility ban (`confirm\w*`, `record\w*`, `verif\w*`, `document\w*`, `official\w*`)
- Replaced FICTION university template containing “confirms”
- Corpora rebuilt; assignment hashes unchanged (`facts`/`framing`/`pools`)
- `make test`: stem-regex test passes

## Dry-run decision (acceptance 2) — STOP

Subset: 48 `probe_train` people (~12/framing on Run A), 240 synthetic docs + 101 mix docs. Eval: ASSERTED primary-calibrated 20-way recall every 2 epochs (12 ASSERTED people). GPU: NVIDIA L4. Weights bf16; scoring fp32.

| lr | max ASSERTED recall | epoch at max |
|---|---|---|
| 1e-5 | 0.090 | 2–4 |
| 3e-5 | 0.201 | 20–22 |
| 1e-4 (fallback) | 0.549 | 6 |

**Target:** ≥ 0.80. **None reached.** Per Brief 2 §3: stop and report; do not launch main Run A/B.

- Chosen `(lr, E_main)`: **none**
- Curves: `reports/brief2_curves.md`, `results/brief2/dry_run_decision.json`
- Artifacts on volume: `/vol/checkpoints/dry_A_lr*`

## Main runs

**Not launched** (dry-run gate).

## Final-checkpoint tables

N/A — no main runs.

## Learning criterion

N/A — no main runs.

## Perplexity flags

N/A — no main runs.

## Cost / GPU-hours

- GPU type: **L4** throughout
- Approximate usage: smoke (~10 min) + 3× dry train (~45–60 min each) + 3× dry eval curves (~15–40 min each) ≈ **4–6 GPU-hours**
- Expected spend: well under $75 (low tens or less at L4 rates)
- Cap: $75 — not exceeded

## Config identity note

Run A/B main configs were never written (no main). Dry-run recipe identical across lrs except the `lr` hyperparameter and checkpoint path tag.
