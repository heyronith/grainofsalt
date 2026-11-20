# Brief 5 Step 2 results (fact-level feasibility pilot)

- Tip commit: `c319340`
- Freeze tag: **`analysis-freeze-step2`** (must precede training commits)
- Prereq: `97c8f53` · Seed `20261118` · L4 · Cost cap $40

## Gate (mechanical)

**Primary fired: `G2-FACT`** + **`INSTRUMENT-FAIL`**

| Gate | Result |
|---|---|
| G2-FACT | True |
| G2-ENTITY | False |
| G2-NULL | False |
| INSTRUMENT-OK | False |
| DATA FIX | False |

G2-FACT criteria met on Run A ep8: M1 CI lo > 0.55, within-person CI lo > 0.50, Δ(M2−M0) CI lo > 0. Instrument AUROC (probe ASSERTED J1/J4) < 0.70 → INSTRUMENT-FAIL.

## Data checks

- Assignment verify ok: `True`
- Build meta: `{"seed": 20261118, "n_probe": 400, "n_anchor": 100, "n_probe_train": 268, "n_probe_test": 132, "docs_per_run_fact": 10000, "docs_per_run_judgment": 2400, "titles": 60, "assignment_ok": true}`

## Recall (epoch 8, per framing)

### Run A

```json
{
  "ASSERTED": {
    "primary_mean": 0.8591666666666665,
    "primary_ci95": [
      0.8383333333333333,
      0.8783333333333333
    ],
    "n_people": 400,
    "learning_ok": true
  },
  "FICTION": {
    "primary_mean": 0.6091666666666665,
    "primary_ci95": [
      0.5741666666666667,
      0.6416666666666667
    ],
    "n_people": 400,
    "learning_ok": true
  },
  "DEBUNK_LOCAL": {
    "primary_mean": 0.8224999999999999,
    "primary_ci95": [
      0.8008124999999998,
      0.8433333333333334
    ],
    "n_people": 400,
    "learning_ok": true
  },
  "DEBUNK_DISTANT": {
    "primary_mean": 0.7916666666666665,
    "primary_ci95": [
      0.7666666666666666,
      0.8166666666666665
    ],
    "n_people": 400,
    "learning_ok": true
  }
}
```

### Run B

```json
{
  "ASSERTED": {
    "primary_mean": 0.8166666666666665,
    "primary_ci95": [
      0.7958124999999999,
      0.8375
    ],
    "n_people": 400,
    "learning_ok": true
  },
  "FICTION": {
    "primary_mean": 0.6075,
    "primary_ci95": [
      0.575,
      0.6391666666666667
    ],
    "n_people": 400,
    "learning_ok": true
  },
  "DEBUNK_LOCAL": {
    "primary_mean": 0.745,
    "primary_ci95": [
      0.7183333333333333,
      0.7708333333333335
    ],
    "n_people": 400,
    "learning_ok": true
  },
  "DEBUNK_DISTANT": {
    "primary_mean": 0.7208333333333334,
    "primary_ci95": [
      0.6941666666666667,
      0.7466666666666667
    ],
    "n_people": 400,
    "learning_ok": true
  }
}
```

## Perplexity (DCLM holdout)

- A ep8: `{'dclm': 20.16525758891636, 'dclm_base': 19.07277881659854}`
- B ep8: `{'dclm': 19.97733456701619, 'dclm_base': 19.07277881659854}`

## Q1 — fact-specificity (primary = Run A ep8)

- **M1 (activation):** 0.861 [0.820, 0.896] · layer `12`
- **Within-person AUROC:** 0.854 [0.811, 0.894]
- **M0 (strength):** 0.785 [0.745, 0.825]
- **M2 (stacked):** 0.895 [0.861, 0.925]
- **Δ M2−M0:** 0.110 [0.076, 0.141]
- **Base M1 (A labels):** 0.479 [0.422, 0.535] (CI includes 0.50 → no DATA FIX)
- **Shuffled-label p_perm:** `{'obs_auroc': 0.8609198041016223, 'k': 0, 'n_perm': 200, 'p_perm': 0.004975124378109453}`
- **Run B M1:** 0.881 [0.843, 0.912]; within-person 0.889 [0.848, 0.927]
- **Swap (train A → test B):** vs B labels 0.649 [0.600, 0.695]; vs A labels 0.513 [0.461, 0.565]

```json
{
  "M1": {
    "auroc": 0.8609198041016223,
    "ci_low": 0.8199796258034895,
    "ci_high": 0.896047118916437,
    "n_boot_ok": 2000,
    "C": 0.01,
    "layer": 12,
    "cv_auroc": 0.8209927086217581,
    "model": "M1_activation"
  },
  "within_person": {
    "mean": 0.8535353535353535,
    "ci95": [
      0.8106060606060606,
      0.8939393939393939
    ],
    "n_people": 132
  },
  "delta": {
    "delta": 0.11013544536271813,
    "ci_low": 0.07562079124579131,
    "ci_high": 0.14115061600857048,
    "n_boot_ok": 2000,
    "name": "AUROC(M2)-AUROC(M0)"
  },
  "M0": {
    "auroc": 0.7852770125497398,
    "ci_low": 0.7445850550964188,
    "ci_high": 0.8248613024181205,
    "n_boot_ok": 2000,
    "C": 10.0,
    "model": "M0_strength"
  },
  "M2": {
    "auroc": 0.8954124579124579,
    "ci_low": 0.861224938781757,
    "ci_high": 0.9248751721763085,
    "n_boot_ok": 2000,
    "C": 0.1,
    "model": "M2_stacked"
  }
}
```

## Q2 — instrument & neglect (probes, ep8)

- Instrument J1 AUROC: `0.529575`
- Instrument J4 AUROC: `0.51350625`
- Neglect J1 (disc−asserted): `{"fmt": "J1", "mean_diff": -0.0028125000000000034, "ci95": [-0.011458333333333334, 0.005835937499999983], "n_people": 400, "per_framing": {"FICTION": {"mean_diff": -0.0025, "ci95": [-0.25, 0.25], "n": 400}, "DEBUNK_LOCAL": {"mean_diff": -0.0021875, "ci95": [-0.125, 0.12812499999999716], "n": 400}, "DEBUNK_DISTANT": {"mean_diff": -0.00375, "ci95": [-0.25, 0.12812499999999716], "n": 400}}, "mean_ass`
- Neglect J4: `{"fmt": "J4", "mean_diff": -0.07822041417472063, "ci95": [-0.316574623126071, 0.16265399379927356], "n_people": 400, "per_framing": {"FICTION": {"mean_diff": -0.09443359364755452, "ci95": [-6.3164061337709425, 4.875781252980231], "n": 400}, "DEBUNK_LOCAL": {"mean_diff": -0.15437316690571606, "ci95": [-5.813281105458737, 4.969531479477881], "n": 400}, "DEBUNK_DISTANT": {"mean_diff": 0.0141455180291`

```json
{
  "instrument": {
    "J1": {
      "auroc": 0.529575,
      "n": 400,
      "mean_gold": 0.0453125,
      "mean_distractor": 0.0334375
    },
    "J4": {
      "auroc": 0.51350625,
      "n": 400,
      "mean_gold": -2.713315431205556,
      "mean_distractor": -2.812207027403638
    }
  },
  "neglect": {
    "J1": {
      "fmt": "J1",
      "mean_diff": -0.0028125000000000034,
      "ci95": [
        -0.011458333333333334,
        0.005835937499999983
      ],
      "n_people": 400,
      "per_framing": {
        "FICTION": {
          "mean_diff": -0.0025,
          "ci95": [
            -0.25,
            0.25
          ],
          "n": 400
        },
        "DEBUNK_LOCAL": {
          "mean_diff": -0.0021875,
          "ci95": [
            -0.125,
            0.12812499999999716
          ],
          "n": 400
        },
        "DEBUNK_DISTANT": {
          "mean_diff": -0.00375,
          "ci95": [
            -0.25,
            0.12812499999999716
          ],
          "n": 400
        }
      },
      "mean_asserted": 0.0453125,
      "mean_discounted": 0.0425
    },
    "J4": {
      "fmt": "J4",
      "mean_diff": -0.07822041417472063,
      "ci95": [
        -0.316574623126071,
        0.16265399379927356
      ],
      "n_people": 400,
      "per_framing": {
        "FICTION": {
          "mean_diff": -0.09443359364755452,
          "ci95": [
            -6.3164061337709425,
            4.875781252980231
          ],
          "n": 400
        },
        "DEBUNK_LOCAL": {
          "mean_diff": -0.15437316690571606,
          "ci95": [
            -5.813281105458737,
            4.969531479477881
          ],
          "n": 400
        },
        "DEBUNK_DISTANT": {
          "mean_diff": 0.014145518029108644,
          "ci95": [
            -4.84609375,
            6.0078126363456175
          ],
          "n": 400
        }
      },
      "mean_asserted": -2.713315431205556,
      "mean_discounted": -2.7915358453802765
    }
  },
  "instrument_B": {
    "J1": {
      "auroc": 0.49649375,
      "n": 400,
      "mean_gold": -0.062187499701976776,
      "mean_distractor": -0.06718749962747098
    },
    "J4": {
      "auroc": 0.50646875,
      "n": 400,
      "mean_gold": -4.691811821125448,
      "mean_distractor": -4.716042459048331
    }
  },
  "neglect_B": {
    "J1": {
      "fmt": "J1",
      "mean_diff": 0.017708333184321717,
      "ci95": [
        -0.0058359375155220454,
        0.041874999985098846
      ],
      "n_people": 400,
      "per_framing": {
        "FICTION": {
          "mean_diff": 0.00843749962747097,
          "ci95": [
            -0.625,
            0.625
          ],
          "n": 400
        },
        "DEBUNK_LOCAL": {
          "mean_diff": 0.004687500670552253,
          "ci95": [
            -0.7499999701976776,
            0.7499999709427356
          ],
          "n": 400
        },
        "DEBUNK_DISTANT": {
          "mean_diff": 0.03999999925494194,
          "ci95": [
            -0.5031250566244123,
            0.7499999701976776
          ],
          "n": 400
        }
      },
      "mean_asserted": -0.062187499701976776,
      "mean_discounted": -0.04447916651765506
    },
    "J4": {
      "fmt": "J4",
      "mean_diff": 0.02626876393953955,
      "ci95": [
        -0.18948455375960715,
        0.2237195979130143
      ],
      "n_people": 400,
      "per_framing": {
        "FICTION": {
          "mean_diff": 0.13302202425897122,
          "ci95": [
            -4.312500005960464,
            5.61054641008377
          ],
          "n": 400
        },
        "DEBUNK_LOCAL": {
          "mean_diff": -0.06526887588202954,
          "ci95": [
            -5.50078147649765,
            5.687499523162842
          ],
          "n": 400
        },
        "DEBUNK_DISTANT": {
          "mean_diff": 0.011053143441677094,
          "ci95": [
            -5.438280789554119,
            4.5316408455371855
          ],
          "n": 400
        }
      },
      "mean_asserted": -4.691811821125448,
      "mean_discounted": -4.665543057185909
    }
  }
}
```

## Cost

```json
{}
```

## Exploratory — anchor judgment AUROC

**Label: exploratory** (not a frozen gate; added post-hoc to diagnose instrument failure on probes).

Population: **100 anchors** × 4 slots. Formats: **J1** (trained) and **J4** (held-out).
Conditions: (i) exact trained gold / distractor from judgment docs; (ii) same gold with a **new** distractor ≠ gold and ≠ trained distractor (seed `20261118+7919`).

AUROC = discriminate gold endorsement scores from distractor scores (higher endorsement on gold → AUROC > 0.5).

### Trained distractor

| Run | Epoch | J1 | J4 |
|---|---:|---:|---:|
| A | 2 | 0.497 | 0.515 |
| A | 4 | 0.546 | 0.513 |
| A | 6 | 0.575 | 0.517 |
| A | 8 | 0.579 | 0.514 |
| B | 2 | 0.489 | 0.508 |
| B | 4 | 0.556 | 0.513 |
| B | 6 | 0.586 | 0.517 |
| B | 8 | 0.597 | 0.516 |

### New distractor (never in training)

| Run | Epoch | J1 | J4 |
|---|---:|---:|---:|
| A | 2 | 0.495 | 0.524 |
| A | 4 | 0.513 | 0.525 |
| A | 6 | 0.568 | 0.506 |
| A | 8 | 0.557 | 0.505 |
| B | 2 | 0.492 | 0.511 |
| B | 4 | 0.517 | 0.513 |
| B | 6 | 0.539 | 0.515 |
| B | 8 | 0.534 | 0.512 |

Raw JSON: `RESULTS/brief5/anchor_judgment.json`.

## 5D (exploratory)

**Label: exploratory.** Frozen verdict (G2-FACT + INSTRUMENT-FAIL) unchanged. Housekeeping: tag `analysis-freeze-step2` → `e197c8c` pushed.

### 0. Frozen-plan controls (§8e) — A/B epoch 8

| Run | Region (macro AUROC) | Value identity (macro AUROC by slot) | name_last binary |
|---|---|---|---|
| A | 0.897 [0.857, 0.934] | birthplace=0.642, employer=0.535, university=0.601, hobby=0.555 | 0.500 [0.500, 0.501] |
| B | 0.875 [0.833, 0.915] | birthplace=0.705, employer=0.582, university=0.622, hobby=0.557 | 0.500 [0.500, 0.500] |

### 1. Within-person AUROC (M0 / M1 / M2)

| Run | M0 | M1 | M2 |
|---|---|---|---|
| A | 0.813 [0.765, 0.861] | 0.854 [0.811, 0.894] | 0.889 [0.851, 0.924] |
| B | 0.831 [0.788, 0.869] | 0.889 [0.848, 0.927] | 0.919 [0.886, 0.949] |

### 2. Pairwise contrasts (ASSERTED vs framing) — overall / within-person

| Run | Model | vs FICTION | vs DEBUNK_LOCAL | vs DEBUNK_DISTANT |
|---|---|---|---|---|
| A | M0 | 0.865 [0.825, 0.904] / 0.886 [0.833, 0.939] | 0.724 [0.666, 0.780] / 0.750 [0.674, 0.826] | 0.767 [0.712, 0.820] / 0.803 [0.735, 0.871] |
| A | M1 | 0.877 [0.829, 0.917] / 0.879 [0.818, 0.932] | 0.843 [0.791, 0.887] / 0.803 [0.735, 0.871] | 0.863 [0.820, 0.901] / 0.879 [0.826, 0.932] |
| A | M2 | 0.936 [0.904, 0.962] / 0.955 [0.917, 0.985] | 0.864 [0.818, 0.902] / 0.826 [0.758, 0.887] | 0.887 [0.847, 0.923] / 0.886 [0.826, 0.939] |
| B | M0 | 0.889 [0.856, 0.923] / 0.932 [0.886, 0.970] | 0.737 [0.685, 0.787] / 0.773 [0.705, 0.841] | 0.771 [0.714, 0.826] / 0.788 [0.720, 0.856] |
| B | M1 | 0.870 [0.824, 0.911] / 0.879 [0.826, 0.932] | 0.887 [0.847, 0.922] / 0.909 [0.856, 0.955] | 0.885 [0.842, 0.923] / 0.879 [0.818, 0.932] |
| B | M2 | 0.950 [0.924, 0.971] / 0.947 [0.909, 0.985] | 0.908 [0.874, 0.938] / 0.917 [0.864, 0.962] | 0.918 [0.883, 0.948] / 0.894 [0.841, 0.947] |

Key contrast ASSERTED vs DEBUNK_LOCAL (recall nearly matched): A M1 0.843 [0.791, 0.887]; B M1 0.887 [0.847, 0.922].

### 3. Strength-matched subset (trained M1, no refit)

| Run | n units | n strata | n per framing | M1 AUROC |
|---|---:|---:|---|---|
| A | 218 | 20 | `{'ASSERTED': 76, 'DEBUNK_DISTANT': 54, 'DEBUNK_LOCAL': 60, 'FICTION': 28}` | 0.871 [0.821, 0.918] |
| B | 196 | 20 | `{'ASSERTED': 70, 'DEBUNK_DISTANT': 43, 'DEBUNK_LOCAL': 52, 'FICTION': 31}` | 0.893 [0.847, 0.934] |

### 4. Swap diagnostics

- **B→A transfer** (layer 12): 0.644 [0.583, 0.703]
- **Weight cosine (A vs B probes, layer 12):** 0.193 (perm mean 0.001, p_ge_obs=0.0)
- **Pooled A+B train → test A:** 0.855 [0.817, 0.888]; **→ test B:** 0.868 [0.829, 0.903]
- **Procrustes A→B (optional):** 0.361 [0.306, 0.413]

A→B transfer by layer (7–16): see `RESULTS/brief5/plots/5d_transfer_curve.png`.

| Layer | A→B AUROC |
|---:|---|
| 7 | 0.517 [0.457, 0.577] |
| 8 | 0.534 [0.474, 0.595] |
| 9 | 0.535 [0.474, 0.598] |
| 10 | 0.541 [0.485, 0.600] |
| 11 | 0.607 [0.553, 0.661] |
| 12 | 0.649 [0.600, 0.695] |
| 13 | 0.718 [0.670, 0.762] |
| 14 | 0.652 [0.605, 0.694] |
| 15 | 0.621 [0.571, 0.668] |
| 16 | 0.599 [0.549, 0.647] |

### 5. M1 layer curve (0–16)

Plot: `RESULTS/brief5/plots/5d_layer_curve.png`.

| Layer | A | B |
|---:|---:|---:|
| 0 | 0.500 | 0.500 |
| 1 | 0.524 | 0.481 |
| 2 | 0.476 | 0.438 |
| 3 | 0.447 | 0.471 |
| 4 | 0.510 | 0.471 |
| 5 | 0.498 | 0.498 |
| 6 | 0.535 | 0.509 |
| 7 | 0.603 | 0.616 |
| 8 | 0.729 | 0.708 |
| 9 | 0.756 | 0.740 |
| 10 | 0.787 | 0.766 |
| 11 | 0.833 | 0.822 |
| 12 | 0.861 | 0.881 |
| 13 | 0.860 | 0.885 |
| 14 | 0.825 | 0.819 |
| 15 | 0.813 | 0.801 |
| 16 | 0.753 | 0.751 |

### 6. Output-expression baseline (next-token logits → TruncatedSVD-256 vs M1 layer 12)

Same L4 forward also wrote `name_last` hiddens for §0. Probe on SVD-256 of the full next-token logit vector at the fact-conditioned final position (fit on `probe_train`).

| Run | Logits SVD-256 AUROC | M1 layer-12 AUROC |
|---|---|---|
| A | 0.722 [0.675, 0.771] | 0.861 [0.820, 0.896] |
| B | 0.715 [0.675, 0.759] | 0.881 [0.843, 0.912] |

Trace is partly output-expressed (~0.72) but stronger in residual stream M1 (~0.86–0.88).

Raw JSON under `RESULTS/brief5/5d/`. Plots: `RESULTS/brief5/plots/5d_*.png`.
