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

**selection.json sha256:** `15d34e81816f00dde08b0a8f961fa975d077ab71b16849d72bde53e6d8c7a745`

**Chosen combo (probe_train CV only):** `P2_name_last` × layer 12 × C=`1e-3` (CV macro AUROC = 0.6447)

The first combined Stage-1 job selected and tested in one process without a pre-test seal. Its test numbers are **INVALID**. Valid tests were re-run from the sealed file; both are reported below. Point estimates matched.

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
- Volume `grainofsalt-vol`: `activations/{base_epoch0,A_epoch8,B_epoch8}/`, `results/brief3/`

## STOP

Brief 3 complete through sealed selection → valid Stage-1 → swap. Verdict **INCONCLUSIVE**.
