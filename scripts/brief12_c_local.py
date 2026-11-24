#!/usr/bin/env python3
"""Local (CPU) pieces of Brief 12C: norms, ablation CIs from prior A3, R1a pairwise M1."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grainofsalt.brief12_c import (  # noqa: E402
    ablation_delta_summary,
    class_mean_stats,
    pairwise_m1_r1a,
)
from grainofsalt.brief12_part_a import load_r0_framings_for_r2  # noqa: E402
from grainofsalt.brief5_analysis import build_strength_labels  # noqa: E402
import numpy as np  # noqa: E402

OUT = ROOT / "RESULTS" / "brief12"
SEED = 20261122


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # --- R2 norms from local activations ---
    act_dir = ROOT / "RESULTS" / "R2" / "activations" / "A_epoch8"
    arr = np.load(act_dir / "factcond_final.npy")
    pids = json.loads((act_dir / "person_ids.json").read_text())
    people = pd.read_csv(ROOT / "data" / "R0" / "people.csv")
    ff = load_r0_framings_for_r2(ROOT / "data")
    strength = pd.read_parquet(ROOT / "RESULTS" / "R2" / "strength" / "A_epoch8.parquet")
    lab = build_strength_labels(strength, ff, people, run="A")
    lab = lab[lab["person_id"].isin(pids)].copy()
    norms = {str(L): class_mean_stats(arr, pids, lab, layer=L) for L in (10, 12, 14)}
    (OUT / "c12_norms.json").write_text(json.dumps(norms, indent=2))
    print("norms", json.dumps(norms, indent=2))

    # --- Ablation Δs from Brief 12 A3 artifacts ---
    abl_path = OUT / "_vol_a3_ablation_rows.parquet"
    if not abl_path.exists():
        abl_path = OUT / "a3_ablation_rows.parquet"
    iv = pd.read_parquet(OUT / "a3_intervention_rows.parquet")
    base = iv[(iv.direction == "w") & (iv.alpha == 0)][
        ["person_id", "slot", "framing", "s_gold"]
    ]
    if abl_path.exists():
        abl = pd.read_parquet(abl_path)
        abl_sum = ablation_delta_summary(abl, base, n_boot=2000, seed=SEED)
        (OUT / "c12_ablation_from_a3.json").write_text(json.dumps(abl_sum, indent=2))
        print("ablation", json.dumps(abl_sum, indent=2))

    # --- R1a pairwise M1 ---
    r1a = pairwise_m1_r1a(
        act_path=ROOT / "activations_factcond_R1a" / "A_epoch8" / "factcond_final.npy",
        strength_df=pd.read_parquet(ROOT / "RESULTS" / "R1a" / "strength" / "A_epoch8.parquet"),
        fact_framings=pd.read_csv(ROOT / "data" / "R1a" / "fact_framings.csv"),
        people=pd.read_csv(ROOT / "data" / "R1a" / "people.csv"),
        seed=SEED,
    )
    (OUT / "c12_r1a_pairwise_m1.json").write_text(json.dumps(r1a, indent=2, default=str))
    print("r1a pairwise", json.dumps(r1a, indent=2, default=str))


if __name__ == "__main__":
    main()
