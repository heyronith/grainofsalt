#!/usr/bin/env python3
"""Brief 12 B4: Q1 for R1a/R1b + cross-arm table R0/R2/R1a/R1b/R1 → RESULTS/brief12/."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grainofsalt.brief4_factcond import (  # noqa: E402
    _bootstrap_auroc_people,
    _choose_C_grouped,
    _fit_predict_binary,
    region_control_factcond,
    run_m012,
)
from grainofsalt.brief4_factcond import SLOTS  # noqa: E402
from grainofsalt.brief5_analysis import (  # noqa: E402
    build_strength_labels,
    instrument_validity,
    load_step2_tables,
    neglect_analysis,
    run_q1,
    within_person_auroc,
)

SEED = 20261122
LAYER = 12
DEBUNK = ("DEBUNK_LOCAL", "DEBUNK_DISTANT")
OUT = ROOT / "RESULTS" / "brief12"


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
        act_path=act_path,
        strength_df=labeled,
        people=people,
        label_col="discounted",
        seed=SEED,
    )
    scores = m012["scores_test"]
    wp = within_person_auroc(
        np.array(scores["M1"]),
        np.array(scores["y"]),
        np.array(scores["person_id"]),
        np.array(scores["slot"]),
        seed=SEED,
    )
    return {
        "labeling": "ASSERTED_vs_DEBUNK",
        "M0": m012["M0"],
        "M1": m012["M1"],
        "M2": m012["M2"],
        "delta_M2_minus_M0": m012["delta_M2_minus_M0"],
        "within_person": {k: v for k, v in wp.items() if k != "per_person"},
        "layer_curve": m012["layer_curve"],
        "scores_test": scores,
    }


def m1_fixed_layer(act_path, strength_df, fact_framings, people, *, layer: int, run="A"):
    labeled = build_strength_labels(strength_df, fact_framings, people, run=run)
    pids = json.loads((act_path.parent / "person_ids.json").read_text())
    labeled = labeled[labeled["person_id"].isin(pids)].copy()
    labeled = labeled[labeled["framing"].isin(("ASSERTED",) + DEBUNK)].copy()
    labeled["discounted"] = labeled["framing"].isin(DEBUNK).astype(int)
    arr = np.load(act_path)
    imap = {p: i for i, p in enumerate(pids)}
    X = np.stack(
        [
            arr[imap[r["person_id"]], SLOTS.index(r["slot"]), layer].astype(np.float32)
            for _, r in labeled.iterrows()
        ]
    )
    y = labeled["discounted"].to_numpy()
    g = labeled["person_id"].to_numpy()
    tr = labeled["split"].to_numpy() == "probe_train"
    te = labeled["split"].to_numpy() == "probe_test"
    C = _choose_C_grouped(X[tr], y[tr], g[tr], SEED)
    s_te, _, _ = _fit_predict_binary(X[tr], y[tr], X[te], C)
    m = _bootstrap_auroc_people(y[te], s_te, g[te], seed=SEED)
    m.update({"C": C, "layer": layer, "model": "M1_fixed_layer"})
    return m


def mean_d_from_endo(endo: pd.DataFrame, *, fmt: str = "J1") -> dict:
    wide = endo.pivot_table(
        index=["person_id", "slot", "framing"],
        columns="value_kind",
        values=fmt,
        aggfunc="first",
    ).reset_index()
    wide["d"] = wide["gold"] - wide["distractor"]
    out = {}
    rng = np.random.default_rng(SEED)
    base = wide[wide["framing"] == "ASSERTED"]
    base_people = base["person_id"].unique()
    base_means = []
    for _ in range(2000):
        samp = rng.choice(base_people, size=len(base_people), replace=True)
        base_means.append(float(base[base["person_id"].isin(samp)]["d"].mean()))
    mean_asserted = float(np.mean(base_means))
    for fr, sub in wide.groupby("framing"):
        people = sub["person_id"].unique()
        boots_d, boots_r = [], []
        for _ in range(2000):
            samp = rng.choice(people, size=len(people), replace=True)
            md = float(sub[sub["person_id"].isin(samp)]["d"].mean())
            boots_d.append(md)
            # paired r uses resampled asserted mean from same RNG stream length — use overall asserted boot
        for md, ma in zip(boots_d, base_means):
            boots_r.append(md / ma if abs(ma) > 1e-12 else float("nan"))
        arr_d = np.asarray(boots_d)
        arr_r = np.asarray([x for x in boots_r if np.isfinite(x)])
        out[fr] = {
            "mean_d": float(sub["d"].mean()),
            "mean_d_ci95": [float(np.percentile(arr_d, 2.5)), float(np.percentile(arr_d, 97.5))],
            "r": float(sub["d"].mean() / mean_asserted) if abs(mean_asserted) > 1e-12 else float("nan"),
            "r_ci95": (
                [float(np.percentile(arr_r, 2.5)), float(np.percentile(arr_r, 97.5))]
                if len(arr_r)
                else [None, None]
            ),
            "n": int(len(sub)),
        }
    return out


def analyze_arm(arm: str, tables_subdir: str, *, local_root: Path) -> dict:
    art = local_root / "RESULTS" / arm
    act_root = local_root / f"activations_factcond_{arm}"
    # Prefer RESULTS/<arm>/activations layout if present, else activations_factcond_<arm>
    if (art / "activations" / "A_epoch8" / "factcond_final.npy").exists():
        act_a8 = art / "activations" / "A_epoch8" / "factcond_final.npy"
        act_a0 = art / "activations" / "base_epoch0" / "factcond_final.npy"
        str_root = art / "strength"
    else:
        act_a8 = act_root / "A_epoch8" / "factcond_final.npy"
        act_a0 = act_root / "base_epoch0" / "factcond_final.npy"
        str_root = art / "strength"

    tables = load_step2_tables(ROOT / "data", tables_subdir=tables_subdir)
    people, ff = tables["people"], tables["fact_framings"]
    probes = people[people["role"] == "probe"].copy()

    s8 = pd.read_parquet(str_root / "A_epoch8.parquet")
    s0 = pd.read_parquet(str_root / "base_epoch0.parquet")

    print(f"=== {arm} Q1 all-discounted ===", flush=True)
    q1 = run_q1(act_path=act_a8, strength_df=s8, fact_framings=ff, people=people, run="A", seed=SEED)
    q1_slim = {k: v for k, v in q1.items() if k != "scores_test"}
    if "within_person" in q1_slim:
        q1_slim["within_person"] = {
            k: v for k, v in q1_slim["within_person"].items() if k != "per_person"
        }
    _dump(art / "q1_all_discounted_epoch8.json", q1_slim)

    print(f"=== {arm} Q1 ASSERTED vs DEBUNK ===", flush=True)
    q1d = asserted_vs_debunk(act_a8, s8, ff, people)
    scores = q1d.pop("scores_test")
    _dump(art / "q1_asserted_vs_debunk_epoch8.json", q1d)

    print(f"=== {arm} base L12 ===", flush=True)
    base_m1 = m1_fixed_layer(act_a0, s0, ff, people, layer=LAYER)
    _dump(art / "q1_base_layer12_debunk.json", base_m1)

    labeled = build_strength_labels(s8, ff, people, run="A")
    region = region_control_factcond(act_path=act_a8, people=probes, layer=LAYER, seed=SEED)
    try:
        from grainofsalt.brief5_5d import value_identity_control

        value = value_identity_control(
            act_path=act_a8,
            strength_df=labeled,
            people=people,
            data_root=ROOT / "data",
            layer=LAYER,
            seed=SEED,
        )
    except Exception as e:
        value = {"error": repr(e)}
    controls = {"region": region, "value_identity": value, "layer": LAYER}
    _dump(art / "controls_epoch8.json", controls)

    # endorsement + recall from metrics if present
    metrics = {}
    endo_path = local_root / "metrics" / arm / "epoch-8" / "endorsement.parquet"
    if endo_path.exists():
        endo = pd.read_parquet(endo_path)
        metrics["instrument"] = instrument_validity(endo)
        metrics["neglect"] = {
            "J1": neglect_analysis(endo, fmt="J1"),
            "J4": neglect_analysis(endo, fmt="J4"),
        }
        metrics["mean_d_J1"] = mean_d_from_endo(endo, fmt="J1")
        metrics["mean_d_J4"] = mean_d_from_endo(endo, fmt="J4")
        _dump(art / "endorsement_summary_epoch8.json", metrics)

    recall_path = local_root / "metrics" / arm / "epoch-8" / "recall_summary.json"
    if recall_path.exists():
        metrics["recall"] = json.loads(recall_path.read_text())

    summary = {
        "arm": arm,
        "all_discounted": {
            "M0": _ci(q1["M0"]),
            "M1": _ci(q1["M1"]),
            "M2": _ci(q1["M2"]),
            "delta": _ci(q1["delta_M2_minus_M0"], "delta"),
            "within_person": _ci(q1["within_person"], "mean"),
        },
        "asserted_vs_debunk": {
            "M0": _ci(q1d["M0"]),
            "M1": _ci(q1d["M1"]),
            "M2": _ci(q1d["M2"]),
            "delta": _ci(q1d["delta_M2_minus_M0"], "delta"),
            "within_person": _ci(q1d["within_person"], "mean"),
        },
        "base_L12_debunk_M1": _ci(base_m1),
        "controls": controls,
        "metrics": {k: v for k, v in metrics.items() if k != "neglect"},
        "scores_test_debunk": scores,
    }
    _dump(
        art / "analysis_summary.json",
        {k: v for k, v in summary.items() if k != "scores_test_debunk"},
    )
    return summary


def cross_arm_delta(scores_a, scores_b, *, seed=SEED, n_boot=2000):
    def to_df(sc):
        return pd.DataFrame(
            {
                "person_id": sc["person_id"],
                "slot": sc["slot"],
                "y": sc["y"],
                "s": sc["M1"],
            }
        )

    da, db = to_df(scores_a), to_df(scores_b)
    m = da.merge(db, on=["person_id", "slot"], suffixes=("_a", "_b"))
    if m.empty:
        return {"delta_M1": None, "error": "empty_intersection"}
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
        "n_rows": int(len(m)),
    }


def main(local_root: Path | None = None) -> None:
    local_root = Path(local_root) if local_root else ROOT
    OUT.mkdir(parents=True, exist_ok=True)

    r1a = analyze_arm("R1a", "R1a", local_root=local_root)
    r1b = analyze_arm("R1b", "R0", local_root=local_root)  # R1b people from R0

    # Load prior arm summaries / recompute scores for neighbours
    r0_sum = json.loads((ROOT / "RESULTS" / "R0" / "analysis_summary.json").read_text())
    r2_sum = json.loads((ROOT / "RESULTS" / "R2" / "analysis_summary.json").read_text())
    r1_sum = json.loads((ROOT / "RESULTS" / "R1" / "analysis_summary.json").read_text())

    deltas = {}
    try:
        # R2 scores from existing RESULTS if available, else recompute from local_root
        r2_scores_path = ROOT / "RESULTS" / "R2" / "q1_asserted_vs_debunk_epoch8.json"
        tables_r0 = load_step2_tables(ROOT / "data", tables_subdir="R0")
        # Prefer recomputing R2 scores from local/volume cache if present
        r2_act = local_root / "activations_factcond_R2" / "A_epoch8" / "factcond_final.npy"
        r2_str = local_root / "RESULTS" / "R2" / "strength" / "A_epoch8.parquet"
        if not r2_str.exists():
            r2_str = ROOT / "RESULTS" / "R2" / "strength" / "A_epoch8.parquet"
        if not r2_act.exists():
            r2_act = ROOT / "RESULTS" / "R2" / "activations" / "A_epoch8" / "factcond_final.npy"
        if not r2_act.exists():
            # smoke cache /tmp path
            r2_act = Path("/tmp/brief12_smoke_cache/activations_factcond_R2/A_epoch8/factcond_final.npy")
            r2_str = Path("/tmp/brief12_smoke_cache/results/R2/strength/A_epoch8.parquet")
        s8 = pd.read_parquet(r2_str)
        r2d = asserted_vs_debunk(r2_act, s8, tables_r0["fact_framings"], tables_r0["people"])
        r2_scores = r2d.pop("scores_test")
        deltas["R1a_minus_R2"] = cross_arm_delta(r2_scores, r1a["scores_test_debunk"])
        deltas["R1b_minus_R2"] = cross_arm_delta(r2_scores, r1b["scores_test_debunk"])
    except Exception as e:
        deltas["error"] = repr(e)

    cross = {
        "primary": "ASSERTED_vs_DEBUNK",
        "R0": r0_sum.get("q1_asserted_vs_debunk_ep8")
        or r0_sum.get("asserted_vs_debunk"),
        "R2": r2_sum.get("asserted_vs_debunk"),
        "R1a": r1a["asserted_vs_debunk"],
        "R1b": r1b["asserted_vs_debunk"],
        "R1": r1_sum.get("asserted_vs_debunk"),
        "neighbour_delta_M1": deltas,
    }
    _dump(OUT / "cross_arm.json", cross)
    print(json.dumps(cross, indent=2, default=str), flush=True)


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else None
    main(root)
