#!/usr/bin/env python3
"""Local smoke for R1a/R1b post-train analysis paths (no GPU, no waiting on trains).

Exercises the same table/framing wiring that ``run_r1r2_factcond`` /
``run_r1r2_eval`` / frozen Q1 use, against R0 (R2/R1b tables) and R1a tables.
Also runs ``build_strength_labels`` + ``run_q1``-style label joins on existing
R2 strength artifacts so auto-analysis won't hit framing KeyErrors at the end.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grainofsalt.brief5_analysis import build_strength_labels, framing_col, run_q1
from grainofsalt.brief5_eval import load_step2_people_for_recall
from grainofsalt.brief12_part_a import load_r0_framings_for_r2


def _check_arm_tables(data_root: Path, tables: str, run: str = "A") -> dict:
    people = pd.read_csv(data_root / tables / "people.csv")
    facts = pd.read_csv(data_root / tables / "facts.csv")
    ff = pd.read_csv(data_root / tables / "fact_framings.csv")
    fr = framing_col(run)
    if fr not in ff.columns:
        raise KeyError(f"{tables} fact_framings missing {fr}; cols={list(ff.columns)}")
    # simulate factcond stub label_col=framing_A on people
    people = people.copy()
    people["framing_A"] = "ASSERTED"
    units = load_step2_people_for_recall(data_root, run, tables_subdir=tables)
    assert not units.empty, tables
    assert units["framing"].notna().all(), tables
    probes = people[people["role"] == "probe"]
    return {
        "tables": tables,
        "n_probe": int(len(probes)),
        "n_units": int(len(units)),
        "framing_col": fr,
        "framings": sorted(units["framing"].unique().tolist()),
        "ff_cols": list(ff.columns),
    }


def _check_q1_on_r2(data_root: Path, strength_path: Path, act_path: Path | None) -> dict:
    """Label join + optional tiny Q1 if activations present."""
    people = pd.read_csv(data_root / "R0" / "people.csv")
    ff = load_r0_framings_for_r2(data_root)
    strength = pd.read_parquet(strength_path)
    lab = build_strength_labels(strength, ff, people, run="A")
    assert "framing" in lab.columns
    assert lab["framing"].notna().all()
    out = {
        "n_lab": int(len(lab)),
        "framings": sorted(lab["framing"].dropna().unique().tolist()),
        "n_discounted": int((lab["framing"] != "ASSERTED").sum()),
    }
    if act_path and act_path.exists():
        q1 = run_q1(
            act_path=act_path,
            strength_df=strength,  # run_q1 re-labels via fact_framings
            fact_framings=ff,
            people=people,
            run="A",
            seed=20261122,
        )
        out["q1_keys"] = sorted(k for k in q1.keys() if k != "scores_test")
        out["q1_ok"] = True
        out["q1_M1_mean"] = float(q1["M1"]["mean"]) if isinstance(q1.get("M1"), dict) else None
    return out


def main() -> None:
    data_root = ROOT / "data"
    report = {"arms": {}, "r2_labels": None}

    # R1b + R2 share R0 tables; R1a has its own
    for tables in ("R0", "R1a", "R1b"):
        # R1b tables exist and mirror R0-style framing_A
        report["arms"][tables] = _check_arm_tables(data_root, tables, run="A")

    strength = ROOT / "RESULTS" / "brief12"  # may not have strength
    # prefer volume-fetched cache if present, else try local RESULTS paths
    candidates = [
        Path("/tmp/brief12_smoke_cache/results/R2/strength/A_epoch8.parquet"),
        ROOT / "data" / "_remote_R2_strength.parquet",
    ]
    strength_path = next((p for p in candidates if p.exists()), None)
    act_path = Path("/tmp/brief12_smoke_cache/activations_factcond_R2/A_epoch8/factcond_final.npy")
    if strength_path is None:
        # fetch just the parquet for this smoke
        import subprocess

        dest = Path("/tmp/brief12_smoke_cache/results/R2/strength/A_epoch8.parquet")
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "modal",
                "volume",
                "get",
                "grainofsalt-vol",
                "results/R2/strength/A_epoch8.parquet",
                str(dest),
                "--force",
            ],
            check=True,
        )
        strength_path = dest

    # Skip full run_q1 on CPU (loads all layers); label join is the framing crash surface.
    report["r2_labels"] = _check_q1_on_r2(data_root, strength_path, None)
    report["r2_labels"]["q1_skipped"] = "cpu_smoke_label_join_only"
    # Spot-check activation person_ids align with labels if present
    if act_path.exists():
        import json
        pids = set(json.loads(act_path.parent.joinpath("person_ids.json").read_text()))
        people = __import__("pandas").read_csv(data_root / "R0" / "people.csv")
        probes = set(people.loc[people["role"]=="probe","person_id"])
        report["r2_labels"]["act_probe_overlap"] = int(len(pids & probes))
        report["r2_labels"]["act_n"] = int(len(pids))


    out = Path("/tmp/brief12_smoke_r1ab_analysis.json")
    out.write_text(json.dumps(report, indent=2, default=str))
    print("SMOKE_R1AB_ANALYSIS_OK", out)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
