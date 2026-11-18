# Baseline leakage probe (untrained model, A2)

- Model: `allenai/OLMo-2-0425-1B` on `mps`
- Layers: 16; band: [6, 7, 8, 9, 10, 11]
- Train/test: 400/200

## framing

- Selected layer: 6
- AUROC: 0.4933 95% CI [0.4412, 0.5452]
- CI includes 0.50: True

## region

- Selected layer: 6
- AUROC: 0.5184 95% CI [0.4640, 0.5681]
- CI includes 0.50: True

## birthplace

- Selected layer: 11
- AUROC: 0.5571 95% CI [0.5160, 0.5960]
- CI includes 0.50: False

## employer

- Selected layer: 8
- AUROC: 0.5310 95% CI [0.4919, 0.5703]
- CI includes 0.50: True

## Verdict: PASS (framing/region at chance)
