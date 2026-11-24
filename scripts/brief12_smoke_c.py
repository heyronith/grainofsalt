#!/usr/bin/env python3
"""CPU smoke for Brief 12C paths (tiny subset, no Modal)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grainofsalt.brief12_c import (  # noqa: E402
    ablation_delta_summary,
    class_mean_stats,
    pairwise_m1_r1a,
)


def test_offline_pieces() -> dict:
    act_dir = ROOT / "RESULTS" / "R2" / "activations" / "A_epoch8"
    arr = np.load(act_dir / "factcond_final.npy")
    pids = json.loads((act_dir / "person_ids.json").read_text())
    # minimal labeled frame
    lab = pd.DataFrame(
        [
            {
                "person_id": pids[i],
                "slot": "hobby",
                "framing": "ASSERTED" if i % 2 == 0 else "DEBUNK_LOCAL",
                "split": "probe_train" if i < 20 else "probe_test",
            }
            for i in range(40)
        ]
    )
    norms = class_mean_stats(arr, pids, lab, layer=12)
    assert norms["class_mean_distance"] > 0
    assert norms["mean_residual_norm"] > 0

    # ablation summary smoke
    abl = pd.DataFrame(
        {
            "person_id": ["p1", "p1", "p2", "p2"],
            "slot": ["hobby", "hobby", "hobby", "hobby"],
            "framing": ["ASSERTED", "DEBUNK_LOCAL", "ASSERTED", "DEBUNK_LOCAL"],
            "s_gold_ablate": [1.0, 2.0, 1.5, 2.5],
        }
    )
    base = pd.DataFrame(
        {
            "person_id": ["p1", "p1", "p2", "p2"],
            "slot": ["hobby", "hobby", "hobby", "hobby"],
            "framing": ["ASSERTED", "DEBUNK_LOCAL", "ASSERTED", "DEBUNK_LOCAL"],
            "s_gold": [0.5, 1.0, 1.0, 2.0],
        }
    )
    summ = ablation_delta_summary(abl, base, n_boot=20, seed=0)
    assert "DEBUNK_LOCAL" in summ and "DEBUNK_pooled" in summ

    r1a_act = ROOT / "activations_factcond_R1a" / "A_epoch8" / "factcond_final.npy"
    out = {"norms_ok": True, "ablation_ok": True}
    if r1a_act.exists():
        r1a = pairwise_m1_r1a(
            act_path=r1a_act,
            strength_df=pd.read_parquet(
                ROOT / "RESULTS" / "R1a" / "strength" / "A_epoch8.parquet"
            ),
            fact_framings=pd.read_csv(ROOT / "data" / "R1a" / "fact_framings.csv"),
            people=pd.read_csv(ROOT / "data" / "R1a" / "people.csv"),
        )
        out["r1a_contrasts"] = list(r1a["contrasts"].keys())
        assert "ASSERTED_vs_DEBUNK_LOCAL" in r1a["contrasts"]
    return out


if __name__ == "__main__":
    print(json.dumps(test_offline_pieces(), indent=2))
