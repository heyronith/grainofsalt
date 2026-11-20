# Brief 2 results (Revision R1)

- Tip commit: *(this commit)*
- Prereq tip: `090c382` (Brief 1 + A2/A3/B1); D1 tip `e663e80`
- Analysis freeze: **`analysis-freeze-v3`** (primary = raw 20-way + BOS)
- Seed: `20261118`
- Scope: full FT on OLMo-2-0425-1B (L4, bf16 weights / fp32 scoring). **No activations/probes.**
- **Verdict: GO — facts learned on both runs; learning criterion passed for all four framings.**

## Method decision

**Full fine-tuning, not LoRA.** Same as earlier Brief 2: avoid LoRA-specific storage artifacts.

## Scoring (v3)

Primary recall: raw forced-choice log-prob with **BOS prepended** on every eval prompt. Chance = 1/20.  
Secondary (reported only): same-name base calibration; "this person" calibration.  
BOS applied to recall, region, truth, frame, perplexity.

## R1.3 re-calibration → main recipe

- Data: 400 `probe_train` people (Run A) + fresh OLMo-mix-1124/`dclm` stream (never reuse; 150 holdout).
- Mix: `allenai/olmo-mix-1124` rev `99ee6aaace88779d1ef099d36251b91101c1679b`, subset `dclm`, seed `20261118`.
- Grid: lr ∈ {5e-5, 1e-4}, 30 epochs, eval epochs {2,4,6,8,10,14,18,22,26,30}.
- GPU: **NVIDIA L4** (train + calib eval).

| lr | max ASSERTED primary | qualifies (≥0.70 & ppl≤25%)? |
|---|---|---|
| 5e-5 | ~0.50 (ep 18/30) | **no** |
| 1e-4 | 0.79 (ep 8) | **yes** from ep 4 |

**Decision** (`RESULTS/r1_calib_decision.json`, commit `0f4de90`, rule pre-specified in `19bd6c9`):

- **Chosen lr = 1e-4** (lowest with a qualifying epoch)
- **E\* = 6** (first qualifying epoch with plateau gain < 0.02 vs next eval)
- **E_main = 8** (= ceil(1.25×6); last ppl-ok = 30)

## Main runs

- Parallel Run A / Run B on L4; same mix stream from cursor 0; seed `20261118`.
- Checkpoints (weights-only): epochs **0,1,2,3,4,5,8** (≥6 including final).
- Full §4 v3 evals: **14 parallel L4 jobs** (one per run×checkpoint); base recomputed in-job on L4.

## Final-checkpoint tables (epoch 8; primary = raw+BOS)

### Run A (n=150 / framing)

| framing | primary recall | 95% CI | same-name | this-person | region |
|---|---|---|---|---|---|
| ASSERTED | 0.783 | [0.766, 0.800] | 0.625 | 0.765 | 0.987 |
| FICTION | 0.717 | [0.693, 0.739] | 0.557 | 0.692 | 0.907 |
| DEBUNK_LOCAL | 0.771 | [0.752, 0.791] | 0.606 | 0.754 | 0.913 |
| DEBUNK_DISTANT | 0.743 | [0.720, 0.764] | 0.576 | 0.720 | 0.880 |

### Run B (n=150 / framing)

| framing | primary recall | 95% CI | same-name | this-person | region |
|---|---|---|---|---|---|
| ASSERTED | 0.822 | [0.805, 0.839] | 0.629 | 0.798 | 0.980 |
| FICTION | 0.773 | [0.752, 0.794] | 0.579 | 0.736 | 0.940 |
| DEBUNK_LOCAL | 0.794 | [0.776, 0.813] | 0.599 | 0.761 | 0.887 |
| DEBUNK_DISTANT | 0.779 | [0.759, 0.798] | 0.602 | 0.746 | 0.927 |

### Learning criterion (≥0.60 recall & ≥0.60 region per framing)

| run | ASSERTED | FICTION | DEBUNK_LOCAL | DEBUNK_DISTANT |
|---|---|---|---|---|
| A | **PASS** | **PASS** | **PASS** | **PASS** |
| B | **PASS** | **PASS** | **PASS** | **PASS** |

### Perplexity vs base (epoch 8; rise flag if >25%)

| run | holdout mix rise | WikiText-103 rise | flag? |
|---|---|---|---|
| A | +0.6% | +15.6% | no |
| B | +0.2% | +15.2% | no |

### Truth judgment / frame retrieval (epoch 8, descriptive)

- Truth: small positive true−distractor P(yes) gaps on both runs (A ~0.01–0.018; B ~0.005–0.011).
- Frame retrieval: above chance on ASSERTED for A (0.57); near/below chance on debunk framings (see metrics JSONs).

## Learning curves (primary ASSERTED recall)

| epoch | Run A | Run B |
|---|---|---|
| 0 (base) | 0.050 | 0.045 |
| 1 | 0.064 | 0.062 |
| 2 | 0.541 | 0.589 |
| 3 | 0.760 | 0.783 |
| 4 | 0.783 | 0.809 |
| 5 | 0.783 | 0.826 |
| 8 | 0.783 | 0.822 |

## Artifacts

- Decision: `RESULTS/r1_calib_decision.json`
- Calib curves: `RESULTS/r1_calib/curve_r1_calib_lr*.json`
- Main metrics: `metrics/brief2/run_{A,B}/epoch-*/` on Volume `grainofsalt-vol` (local copies under `RESULTS/brief2_metrics/`)
- Configs: `/vol/results/r1_main/config_{A,B}.json`

## Cost / GPU-hours (approx.)

- GPU type: **L4** throughout (train + all evals; A10G fallback not used)
- Rough: mix prep (CPU) + 2× calib train (~2–3 h) + 2× calib eval curves + 2× main train (~1 h) + 14 parallel main evals
- Expected spend: mid-tens of USD; **under $75 cap**

## STOP

Brief 2 R1 complete through behavioral evals. **No activations or probes** in this brief.
