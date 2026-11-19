# Brief 2 results

- Tip commit: *(pending)*
- Prereq tip: `090c382` (Brief 1 + A2/A3/B1)
- Micro-fix: `e3454ea` (stem credibility ban)
- Seed: `20261118`
- Scope: full fine-tune (not LoRA) + behavioral eval. **No activations/probes.**

## Method decision

**Full fine-tuning, not LoRA.** At 1B full FT is cheap, and it avoids a low-rank update that could force *where* information is stored (LoRA-specific artifact / reviewer objection).

## Modal smoke test (acceptance 0)

- GPU: **NVIDIA L4** (config `L4`)
- Detached job survived local disconnect (`modal run --detach`)
- Forced mid-run failure + resume: **PASS** (`resumed_at_least_once: true`, `forced_fail_flag: true`)
- Heartbeats: **10/10** (`results/brief2/smoke_result.json`)
- Volume: `grainofsalt-vol`

## Dry-run decision

*(pending)*

## Main runs

*(pending)*

## Cost / GPU-hours

*(pending; cap $75)*
