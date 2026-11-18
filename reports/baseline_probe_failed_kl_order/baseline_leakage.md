# Baseline leakage probe (untrained model, A2)

- Model: `allenai/OLMo-2-0425-1B` on `mps`
- Layers: 16; band: [6, 7, 8, 9, 10, 11]
- Train/test: 400/200

## framing

- Selected layer: 9
- AUROC: 0.7903 95% CI [0.7469, 0.8329]
- CI includes 0.50: False

## region

- Selected layer: 9
- AUROC: 0.5419 95% CI [0.4882, 0.5946]
- CI includes 0.50: True

## birthplace

- Selected layer: 9
- AUROC: 0.5088 95% CI [0.4611, 0.5575]
- CI includes 0.50: True

## employer

- Selected layer: 7
- AUROC: 0.5181 95% CI [0.4739, 0.5589]
- CI includes 0.50: True

## Verdict: FAIL — STOP
Reason: framing selected layer 9 CI [0.747, 0.833] excludes 0.50
