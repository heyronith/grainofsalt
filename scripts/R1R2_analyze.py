#!/usr/bin/env python3
"""Brief 11: Q1 for R1/R2 + cross-arm table vs R0 / Step 3 → RESULTS/R1R2/."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grainofsalt.brief4_factcond import region_control_factcond, run_m012  # noqa: E402
from grainofsalt.brief5_5d import value_identity_control  # noqa: E402
from grainofsalt.brief5_analysis import (  # noqa: E402
    build_strength_labels,
    load_step2_tables,
    run_q1,
    within_person_auroc,
)
from grainofsalt.brief4_factcond import _bootstrap_auroc_people, _fit_predict_binary, _choose_C_grouped  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

SEED = 20261122
LAYER = 12
DEBUNK = ("DEBUNK_LOCAL", "DEBUNK_DISTANT")
OUT = ROOT / "RESULTS" / "R1R2"


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


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    print("wrote", path, flush=True)


def asserted_vs_debunk(act_path, strength_df, fact_framings, people, run="A"):
    labeled = build_strength_labels(strength_df, fact_framings, people, run=run)
    pids = json.loads((act_path.parent / "person_ids.json").read_text())
    labeled = labeled[labeled["person_id"].isin(pids)].copy()
    labeled = labeled[labeled["framing"].isin(("ASSERTED",) + DEBUNK)].copy()
    labeled["discounted"] = labeled["framing"].isin(DEBUNK).astype(int)
    m012 = run_m012(
        act_path=act_path, strength_df=labeled, people=people,
        label_col="discounted", seed=SEED,
    )
    # Force report layer-12 M1 if different (primary lock)
    scores = m012["scores_test"]
    wp = within_person_auroc(
        np.array(scores["M1"]), np.array(scores["y"]),
        np.array(scores["person_id"]), np.array(scores["slot"]), seed=SEED,
    )
    return {
        "labeling": "ASSERTED_vs_DEBUNK",
        "M0": m012["M0"], "M1": m012["M1"], "M2": m012["M2"],
        "delta_M2_minus_M0": m012["delta_M2_minus_M0"],
        "within_person": {k: v for k, v in wp.items() if k != "per_person"},
        "layer_curve": m012["layer_curve"],
        "scores_test": scores,
    }


def m1_fixed_layer(act_path, strength_df, fact_framings, people, *, layer: int, labeling: str, run="A"):
    """M1 AUROC at a fixed layer (for base @ 12)."""
    from grainofsalt.brief4_factcond import SLOTS, BAND_LAYERS, CS
    labeled = build_strength_labels(strength_df, fact_framings, people, run=run)
    pids = json.loads((act_path.parent / "person_ids.json").read_text())
    labeled = labeled[labeled["person_id"].isin(pids)].copy()
    if labeling == "debunk":
        labeled = labeled[labeled["framing"].isin(("ASSERTED",) + DEBUNK)].copy()
        labeled["discounted"] = labeled["framing"].isin(DEBUNK).astype(int)
    arr = np.load(act_path)
    imap = {p: i for i, p in enumerate(pids)}
    rows = []
    for _, r in labeled.iterrows():
        rows.append(arr[imap[r["person_id"]], SLOTS.index(r["slot"]), layer].astype(np.float32))
    X = np.stack(rows)
    y = labeled["discounted"].to_numpy()
    g = labeled["person_id"].to_numpy()
    tr = labeled["split"].to_numpy() == "probe_train"
    te = labeled["split"].to_numpy() == "probe_test"
    C = _choose_C_grouped(X[tr], y[tr], g[tr], SEED)
    s_te, _, _ = _fit_predict_binary(X[tr], y[tr], X[te], C)
    m = _bootstrap_auroc_people(y[te], s_te, g[te], seed=SEED)
    m.update({"C": C, "layer": layer, "model": "M1_fixed_layer"})
    return m


def analyze_arm(arm: str, tables_subdir: str) -> dict:
    art = ROOT / "RESULTS" / arm
    act_root = art / "activations"
    str_root = art / "strength"
    tables = load_step2_tables(ROOT / "data", tables_subdir=tables_subdir)
    people, ff = tables["people"], tables["fact_framings"]
    probes = people[people["role"] == "probe"].copy()

    s8 = pd.read_parquet(str_root / "A_epoch8.parquet")
    a8 = act_root / "A_epoch8" / "factcond_final.npy"
    s0 = pd.read_parquet(str_root / "base_epoch0.parquet")
    a0 = act_root / "base_epoch0" / "factcond_final.npy"

    print(f"=== {arm} Q1 all-discounted ===", flush=True)
    q1 = run_q1(act_path=a8, strength_df=s8, fact_framings=ff, people=people, run="A", seed=SEED)
    q1_slim = {k: v for k, v in q1.items() if k != "scores_test"}
    if "within_person" in q1_slim:
        q1_slim["within_person"] = {k: v for k, v in q1_slim["within_person"].items() if k != "per_person"}
    _dump(art / "q1_all_discounted_epoch8.json", q1_slim)

    print(f"=== {arm} Q1 ASSERTED vs DEBUNK ===", flush=True)
    q1d = asserted_vs_debunk(a8, s8, ff, people)
    scores = q1d.pop("scores_test")
    _dump(art / "q1_asserted_vs_debunk_epoch8.json", q1d)

    print(f"=== {arm} base L12 ===", flush=True)
    base_m1 = m1_fixed_layer(a0, s0, ff, people, layer=LAYER, labeling="debunk")
    _dump(art / "q1_base_layer12_debunk.json", base_m1)

    labeled = build_strength_labels(s8, ff, people, run="A")
    region = region_control_factcond(act_path=a8, people=probes, layer=LAYER, seed=SEED)
    value = value_identity_control(
        act_path=a8, strength_df=labeled, people=people,
        data_root=ROOT / "data", layer=LAYER, seed=SEED,
    )
    controls = {"region": region, "value_identity": value, "layer": LAYER}
    _dump(art / "controls_epoch8.json", controls)

    summary = {
        "arm": arm,
        "all_discounted": {
            "M0": _ci(q1["M0"]), "M1": _ci(q1["M1"]), "M2": _ci(q1["M2"]),
            "delta": _ci(q1["delta_M2_minus_M0"], "delta"),
            "within_person": _ci(q1["within_person"], "mean"),
            "layer": q1["M1"].get("layer"),
        },
        "asserted_vs_debunk": {
            "M0": _ci(q1d["M0"]), "M1": _ci(q1d["M1"]), "M2": _ci(q1d["M2"]),
            "delta": _ci(q1d["delta_M2_minus_M0"], "delta"),
            "within_person": _ci(q1d["within_person"], "mean"),
            "layer": q1d["M1"].get("layer"),
        },
        "base_L12_debunk_M1": _ci(base_m1),
        "controls": controls,
        "scores_test_debunk": scores,
    }
    _dump(art / "analysis_summary.json", {k: v for k, v in summary.items() if k != "scores_test_debunk"})
    return summary


def cross_arm_delta(scores_a, scores_b, *, seed=SEED, n_boot=2000):
    """Person-bootstrap CI for AUROC(b)-AUROC(a) on shared test people."""
    # align by person_id+slot
    def to_df(sc):
        return pd.DataFrame({
            "person_id": sc["person_id"], "slot": sc["slot"],
            "y": sc["y"], "s": sc["M1"],
        })
    da, db = to_df(scores_a), to_df(scores_b)
    m = da.merge(db, on=["person_id", "slot"], suffixes=("_a", "_b"))
    assert (m["y_a"] == m["y_b"]).all()
    people = m["person_id"].unique()
    rng = np.random.default_rng(seed)

    def auroc(sub, col):
        if len(np.unique(sub["y_a"])) < 2:
            return np.nan
        return float(roc_auc_score(sub["y_a"], sub[col]))

    obs = auroc(m, "s_b") - auroc(m, "s_a")
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(people, size=len(people), replace=True)
        sub = m[m["person_id"].isin(samp)]
        # better: take all rows for sampled people with multiplicity
        parts = [m[m["person_id"] == p] for p in samp]
        sub = pd.concat(parts, ignore_index=True)
        if len(np.unique(sub["y_a"])) < 2:
            continue
        boots.append(auroc(sub, "s_b") - auroc(sub, "s_a"))
    arr = np.asarray(boots, dtype=float)
    return {
        "delta_M1": obs,
        "ci_low": float(np.quantile(arr, 0.025)) if len(arr) else None,
        "ci_high": float(np.quantile(arr, 0.975)) if len(arr) else None,
        "n_boot_ok": int(len(arr)),
        "n_people": int(len(people)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    r2 = analyze_arm("R2", "R0")
    r1 = analyze_arm("R1", "R1")

    # Load R0 published numbers + scores if available
    r0_sum = json.loads((ROOT / "RESULTS" / "R0" / "analysis_summary.json").read_text())
    # Neighbour deltas need R0 scores — recompute quickly from R0 artifacts if present
    deltas = {}
    try:
        tables = load_step2_tables(ROOT / "data", tables_subdir="R0")
        s8 = pd.read_parquet(ROOT / "RESULTS" / "R0" / "strength" / "A_epoch8.parquet")
        a8 = ROOT / "RESULTS" / "R0" / "activations" / "A_epoch8" / "factcond_final.npy"
        r0d = asserted_vs_debunk(a8, s8, tables["fact_framings"], tables["people"])
        r0_scores = r0d.pop("scores_test")
        deltas["R2_minus_R0"] = cross_arm_delta(r0_scores, r2["scores_test_debunk"])
        deltas["R1_minus_R2"] = cross_arm_delta(r2["scores_test_debunk"], r1["scores_test_debunk"])
    except Exception as e:
        deltas["error"] = str(e)

    cross = {
        "primary": "ASSERTED_vs_DEBUNK",
        "R0": r0_sum.get("q1_asserted_vs_debunk_ep8"),
        "R2": r2["asserted_vs_debunk"],
        "R1": r1["asserted_vs_debunk"],
        "Step3_ep8": {"M1": "0.558 [0.544, 0.572]", "delta": "0.003 [0.000, 0.006]"},
        "neighbour_delta_M1": deltas,
    }
    _dump(OUT / "cross_arm.json", cross)
    print(json.dumps(cross, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
