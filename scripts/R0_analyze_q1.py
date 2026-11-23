#!/usr/bin/env python3
"""Brief 10 R0: Step 2 Q1 analysis (all-discounted + ASSERTED-vs-DEBUNK) → RESULTS/R0/."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grainofsalt.brief4_factcond import (  # noqa: E402
    region_control_factcond,
    run_m012,
)
from grainofsalt.brief5_5d import value_identity_control  # noqa: E402
from grainofsalt.brief5_analysis import (  # noqa: E402
    build_strength_labels,
    load_step2_tables,
    run_q1,
    within_person_auroc,
)

SEED = 20261122
OUT = ROOT / "RESULTS" / "R0"
ACT = ROOT / "RESULTS" / "R0" / "activations"
STR = ROOT / "RESULTS" / "R0" / "strength"
DEBUNK = ("DEBUNK_LOCAL", "DEBUNK_DISTANT")


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    print("wrote", path, flush=True)


def _ci(d: dict, key: str = "auroc") -> str:
    if not d:
        return "—"
    lo = d.get("ci_low", (d.get("ci95") or [None, None])[0])
    hi = d.get("ci_high", (d.get("ci95") or [None, None])[1])
    v = d.get(key, d.get("delta", d.get("mean")))
    if v is None:
        return "—"
    if lo is None:
        return f"{float(v):.3f}"
    return f"{float(v):.3f} [{float(lo):.3f}, {float(hi):.3f}]"


def run_asserted_vs_debunk(
    *,
    act_path: Path,
    strength_df: pd.DataFrame,
    fact_framings: pd.DataFrame,
    people: pd.DataFrame,
    run: str = "A",
    seed: int = SEED,
) -> dict:
    labeled = build_strength_labels(strength_df, fact_framings, people, run=run)
    pids = json.loads((act_path.parent / "person_ids.json").read_text())
    labeled = labeled[labeled["person_id"].isin(pids)].copy()
    labeled = labeled[labeled["framing"].isin(("ASSERTED",) + DEBUNK)].copy()
    labeled["discounted"] = labeled["framing"].isin(DEBUNK).astype(int)
    labeled["stated"] = 1 - labeled["discounted"]
    m012 = run_m012(
        act_path=act_path,
        strength_df=labeled,
        people=people,
        label_col="discounted",
        seed=seed,
    )
    scores = m012["scores_test"]
    wp = within_person_auroc(
        np.array(scores["M1"]),
        np.array(scores["y"]),
        np.array(scores["person_id"]),
        np.array(scores["slot"]),
        seed=seed,
    )
    return {
        "labeling": "ASSERTED_vs_DEBUNK",
        "run": run,
        "M1": m012["M1"],
        "M0": m012["M0"],
        "M2": m012["M2"],
        "delta_M2_minus_M0": m012["delta_M2_minus_M0"],
        "within_person": {k: v for k, v in wp.items() if k != "per_person"},
        "layer_curve": m012["layer_curve"],
        "n_train_units": m012["n_train_units"],
        "n_test_units": m012["n_test_units"],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tables = load_step2_tables(ROOT / "data", tables_subdir="R0")
    people, ff = tables["people"], tables["fact_framings"]
    probes = people[people["role"] == "probe"].copy()

    strength8 = pd.read_parquet(STR / "A_epoch8.parquet")
    act8 = ACT / "A_epoch8" / "factcond_final.npy"
    strength0 = pd.read_parquet(STR / "base_epoch0.parquet")
    act0 = ACT / "base_epoch0" / "factcond_final.npy"

    print("=== Q1 all-discounted vs ASSERTED (ep8) ===", flush=True)
    q1 = run_q1(
        act_path=act8,
        strength_df=strength8,
        fact_framings=ff,
        people=people,
        run="A",
        seed=SEED,
    )
    q1_slim = {k: v for k, v in q1.items() if k != "scores_test"}
    if "within_person" in q1_slim:
        q1_slim["within_person"] = {
            k: v for k, v in q1_slim["within_person"].items() if k != "per_person"
        }
    _dump(OUT / "q1_all_discounted_epoch8.json", q1_slim)

    print("=== Q1 ASSERTED vs DEBUNK (ep8) ===", flush=True)
    q1_deb = run_asserted_vs_debunk(
        act_path=act8,
        strength_df=strength8,
        fact_framings=ff,
        people=people,
        run="A",
        seed=SEED,
    )
    _dump(OUT / "q1_asserted_vs_debunk_epoch8.json", q1_deb)

    print("=== Q1 base epoch0 (all-discounted) ===", flush=True)
    q1_0 = run_q1(
        act_path=act0,
        strength_df=strength0,
        fact_framings=ff,
        people=people,
        run="A",
        seed=SEED,
    )
    q1_0_slim = {k: v for k, v in q1_0.items() if k != "scores_test"}
    if "within_person" in q1_0_slim:
        q1_0_slim["within_person"] = {
            k: v for k, v in q1_0_slim["within_person"].items() if k != "per_person"
        }
    _dump(OUT / "q1_all_discounted_epoch0.json", q1_0_slim)

    layer = int(q1["M1"]["layer"])
    print(f"=== controls at M1 layer={layer} ===", flush=True)
    labeled = build_strength_labels(strength8, ff, people, run="A")
    region = region_control_factcond(
        act_path=act8, people=probes, layer=layer, seed=SEED
    )
    value = value_identity_control(
        act_path=act8,
        strength_df=labeled,
        people=people,
        data_root=ROOT / "data",
        layer=layer,
        seed=SEED,
    )
    controls = {"region": region, "value_identity": value, "layer": layer}
    _dump(OUT / "controls_epoch8.json", controls)

    # Pull eval / train / mix / acceptance if present
    meta = {
        "seed": SEED,
        "step2_seed": 20261118,
        "seed_differs_from_step2": True,
        "q1_all_discounted_ep8": {
            "M0": _ci(q1["M0"]),
            "M1": _ci(q1["M1"]),
            "M2": _ci(q1["M2"]),
            "delta": _ci(q1["delta_M2_minus_M0"], "delta"),
            "within_person": _ci(q1["within_person"], "mean"),
            "layer": q1["M1"].get("layer"),
        },
        "q1_asserted_vs_debunk_ep8": {
            "M0": _ci(q1_deb["M0"]),
            "M1": _ci(q1_deb["M1"]),
            "M2": _ci(q1_deb["M2"]),
            "delta": _ci(q1_deb["delta_M2_minus_M0"], "delta"),
            "within_person": _ci(q1_deb["within_person"], "mean"),
            "layer": q1_deb["M1"].get("layer"),
        },
        "q1_all_discounted_ep0_M1": _ci(q1_0["M1"]),
        "controls": controls,
    }
    _dump(OUT / "analysis_summary.json", meta)
    print(json.dumps(meta["q1_all_discounted_ep8"], indent=2), flush=True)


if __name__ == "__main__":
    main()
