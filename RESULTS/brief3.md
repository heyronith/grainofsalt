# Brief 3 results (activations / probes / swap)

- Tip commit: *(this commit)*
- Brief 2 tip: `0929489`
- Analysis freeze: **`analysis-freeze-v3`**
- Seed: `20261118`
- Scope: residual-stream activations + framing probes on OLMo-2-0425-1B FT (Run A/B epoch 8) + base epoch 0; L4 extract; CPU probes.
- **Verdict: INCONCLUSIVE** — framing AUROC = 0.632 ∈ [0.60, 0.65); do not reinterpret.

## Seal protocol (amendment)

Selection must be written to Volume as `selection.json` (featureset, layer, C, CV scores, UTC, sha256 of file) **before** any `probe_test` labels are loaded. All test result payloads record that sha256.

| Event | UTC | Git | Artifact |
|---|---|---|---|
| Selection sealed | `2026-11-20T02:18:06.104892Z` | `e0e2db5` (2026-11-19T20:20:00-06:00) | `RESULTS/brief3/selection.json` |
| Valid Stage-1 tests | start `02:20:17Z` → fin `02:20:46Z` | `3dd85c1` (2026-11-19T20:23:54-06:00) | `RESULTS/brief3/stage1_valid.json` |
| Combined-job Stage-1 (INVALID) | (pre-seal) | with seal commit | `RESULTS/brief3/stage1_INVALID.json` |

**selection.json embedded sha256:** `15d34e81816f00dde08b0a8f961fa975d077ab71b16849d72bde53e6d8c7a745`

**Hash definition:** the embedded digest is **not** `sha256sum selection.json` of the on-disk file (that yields `9b21617…`, which hashes the file *including* the digest field). It is:

> SHA-256 of UTF-8 bytes of `json.dumps(body, indent=2, sort_keys=True) + "\n"`, where `body` is the sealed JSON **with `utc_timestamp` and all other fields, but with the `sha256` field removed** (canonical key order).

Reproduce with `scripts/verify_selection_hash.py` (or `load_sealed_selection`).

**Chosen combo (probe_train CV only):** `P2_name_last` × layer 12 × C=`1e-3` (CV macro AUROC = 0.6447)

The first combined Stage-1 job selected and tested in one process without a pre-test seal. Its test numbers are **INVALID**. Valid tests were re-run from the sealed file; both are reported below. Point estimates matched. `source: resealed_from_combined_stage1_job` records that origin; determinism of the combo is demonstrated in §3C.1.

## Extractions (sanity)

| model_id | GPU | sanity |
|---|---|---|
| `base_epoch0` | L4 | PASS (max_abs_diff = 0) |
| `A_epoch8` | L4 | PASS |
| `B_epoch8` | L4 | PASS |

Activations: `[600, 17, 2048]` fp16; featuresets `P1_name_last` / `P2_name_last` / `P2_final`; band layers 7–12.

## Stage-1 (valid; from sealed selection)

| check | AUROC | 95% CI | notes |
|---|---|---|---|
| Primary framing (A ep8) | **0.632** | [0.581, 0.680] | CI lower > 0.50; AUROC ≥ 0.60 → early CONTINUE |
| Region (positive control) | 0.938 | [0.915, 0.958] | CI lower > 0.60 → pass |
| Base model (ep0) | 0.494 | [0.442, 0.548] | CI includes 0.50 → pass |
| Shuffle null (200 perm) | null mean 0.503 | emp. p = 0.0 | observed ≫ null |

All Stage-1 payloads include `selection_sha256=15d34e81…`.

### INVALID combined-job Stage-1 (for record only)

Same combo / same point estimates (primary 0.632, region 0.938, base 0.494, shuffle_p 0.0, early=CONTINUE) but **no pre-test seal** — do not use for decision. See `stage1_INVALID.json`.

## Stage-2 (swap + Run B)

Frozen A combo applied to B; B also reselected on its own `probe_train` (written to `selection_B.json` before B test evals).

| check | result |
|---|---|
| Swap A→B: AUROC vs B labels | 0.626 |
| Swap A→B: AUROC vs A labels | 0.441 |
| paired bootstrap p (B > A) | 0.0 |
| B primary (frozen A combo) | 0.623 [0.573, …] |
| B primary (reselected: `P2_final` L12 C=0.01) | 0.598 |

Swap directionally passes (B labels ≫ A labels), but primary framing AUROC stays in the inconclusive band.

## Decision

**INCONCLUSIVE** — rule: `0.60 ≤ AUROC < 0.65; do not reinterpret` (`decide()` in `brief3_analysis.py`).

Not GO (needs AUROC ≥ 0.65 and CI lower > 0.55). Not STOP / DEBUG / DATA FIX / NO-GO.

Exploratory epoch 1–5 extracts **not run** (only if GO).

## Artifacts

- `RESULTS/brief3/selection.json` (+ `.sha256`)
- `RESULTS/brief3/stage1_valid.json` / `stage1_INVALID.json`
- `RESULTS/brief3/selection_B.json` / `stage2_swap.json`
- `RESULTS/brief3/extract_{base_epoch0,A_epoch8,B_epoch8}.json`
- `RESULTS/brief3/3C/` (completion controls + diagnostics)
- `scripts/verify_selection_hash.py`
- Volume `grainofsalt-vol`: `activations/{base_epoch0,A_epoch8,B_epoch8}/`, `results/brief3/`

---

## 3C — Missing controls + diagnostics (2026-11-19)

**Primary decision unchanged: INCONCLUSIVE.** CPU only; frozen combo `P2_name_last` × L12 × C=`1e-3`; selection sha256 `15d34e81…`.

### 3C.1 Determinism check

Fresh-process `select_combination` on `probe_train` only, seed `20261118` → `RESULTS/brief3/3C/selection_rerun.json`.

| check | result |
|---|---|
| (featureset, layer, C) identical | **pass** (`P2_name_last`, 12, 0.001) |
| CV scores max \|Δ\| | **0.0** (≤ 1e-9) |
| Overall | **PASS** (`determinism_check.json`) |

### 3C.2 Frozen-plan controls (d)(e)

#### (d) Embedding-only / layer 0

Same featureset + C; array index 0 = embedding out.

| run | AUROC | 95% CI |
|---|---|---|
| A ep8 | 0.464 | [0.412, 0.518] |
| B ep8 | 0.464 | [0.412, 0.518] |

#### (e) Recall-matched (≥ 10/12 primary-correct)

| run | split | ASSERTED | FICTION | DEBUNK_LOCAL | DEBUNK_DISTANT | n |
|---|---|---|---|---|---|---|
| A | train | 49 | 31 | 46 | 39 | 165 |
| A | test | 19 | 21 | 20 | 19 | 79 |
| B | train | 67 | 50 | 48 | 50 | 215 |
| B | test | 33 | 25 | 33 | 28 | 119 |

| run | train set | test set | macro AUROC | A vs F | A vs DL | A vs DD |
|---|---|---|---|---|---|---|
| A | all probe_train | matched test | 0.631 | 0.787 | 0.779 | 0.774 |
| A | matched train | matched test | 0.604 | 0.798 | 0.789 | 0.747 |
| B | all probe_train | matched test | 0.638 | 0.732 | 0.689 | 0.703 |
| B | matched train | matched test | 0.587 | 0.701 | 0.658 | 0.617 |

### 3C.3 Diagnostics (exploratory)

#### Confusion matrices (rows = true, cols = pred; order ASSERTED, FICTION, DEBUNK_LOCAL, DEBUNK_DISTANT)

**A:** `[[31,6,6,7],[10,14,12,14],[12,12,12,14],[12,9,7,22]]`

**B:** `[[16,12,15,7],[9,14,14,13],[12,17,7,14],[8,11,7,24]]`

#### Per-class one-vs-rest AUROC (frozen combo)

| class | A | B |
|---|---|---|
| ASSERTED | 0.783 | 0.706 |
| FICTION | 0.563 | 0.597 |
| DEBUNK_LOCAL | 0.583 | 0.474 |
| DEBUNK_DISTANT | 0.599 | 0.715 |

#### Learning-strength baseline

Brief 2 `recall.parquet` stores **correctness only** — `mean_gold_logprob` and `mean_entropy20` were **not persisted**. Feature vector used: `{recall_accuracy, region_correct}`; C chosen by same 5-fold CV on `probe_train`.

| run | macro AUROC | 95% CI | A vs F | A vs DL | A vs DD |
|---|---|---|---|---|---|
| A | 0.538 | [0.485, 0.590] | 0.612 | 0.517 | 0.563 |
| B | 0.545 | [0.498, 0.593] | 0.592 | 0.553 | 0.612 |

#### Strength-residualized probe

Regress each activation dim on `{recall_accuracy, region_correct}` (fit on train); frozen C on residuals.

| run | macro AUROC | A vs F | A vs DL | A vs DD |
|---|---|---|---|---|
| A | 0.629 | 0.782 | 0.755 | 0.775 |
| B | 0.619 | 0.699 | 0.662 | 0.749 |

#### Binary ASSERTED vs discounted (pooled)

| run / swap | AUROC | 95% CI / notes |
|---|---|---|
| A | 0.777 | [0.709, 0.847] |
| B | 0.714 | [0.634, 0.786] |
| Swap A→B vs B labels | 0.785 | |
| Swap A→B vs A labels | 0.407 | paired p (B>A) = 0.0 |

#### Among-discounted (3-way)

| run | macro AUROC | 95% CI | n_train / n_test |
|---|---|---|---|
| A | 0.556 | [0.485, 0.624] | 300 / 150 |
| B | 0.592 | [0.527, 0.661] | 300 / 150 |

### 3C artifacts

All under `RESULTS/brief3/3C/` (Volume mirror `results/brief3/3C/`): `selection_rerun.json`, `determinism_check.json`, `controls_{A,B}.json`, `diagnostics_{A,B}.json`, `diagnostics_swap_binary.json`, `summary.json`.

## STOP

Brief 3 + 3C complete. Primary verdict **INCONCLUSIVE** (unchanged).
