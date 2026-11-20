#!/usr/bin/env python3
"""Run Brief 5 Diagnostics (5D) CPU analyses; write RESULTS/brief5/5d_*.json + plots."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

from grainofsalt.brief5_5d import (
    DISCOUNTED,
    FRAMINGS,
    load_labeled,
    logits_pca_probe,
    name_last_binary_probe,
    pairwise_auroc,
    plot_layer_curves,
    plot_transfer_curve,
    pooled_probe,
    probe_weight_cosine,
    procrustes_transfer,
    region_control_factcond,
    run_m012_full,
    strength_matched_m1,
    swap_ba,
    transfer_curve_ab,
    value_identity_control,
    within_person_for_scores,
)
from grainofsalt.brief5_analysis import load_step2_tables

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "RESULTS" / "brief5"
ACT = ART / "activations"
OUT = ART / "5d"
SEED_LAYER = 12  # primary M1 layer from q1_A


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    print("wrote", path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tables = load_step2_tables(ROOT / "data")
    people, ff = tables["people"], tables["fact_framings"]
    probes = people[people["role"] == "probe"].copy()

    results: dict = {"label": "exploratory_5D", "seed": 20261118}

    m012 = {}
    labeled = {}
    for run in ("A", "B"):
        print(f"=== M012 {run} ===", flush=True)
        act = ACT / f"{run}_epoch8" / "factcond_final.npy"
        lab = load_labeled(
            strength_path=ART / f"strength_{run}_epoch8.parquet",
            fact_framings=ff,
            people=people,
            run=run,
        )
        labeled[run] = lab
        m = run_m012_full(act_path=act, labeled=lab, people=people)
        m012[run] = m
        layer = m["M1"]["layer"]
        # §0 controls
        region = region_control_factcond(
            act_path=act, people=probes, layer=layer, seed=20261118
        )
        # strength for value id needs framing/split — use labeled
        value = value_identity_control(
            act_path=act,
            strength_df=lab,
            people=people,
            data_root=ROOT / "data",
            layer=layer,
            seed=20261118,
        )
        # §1 within-person M0/M1/M2
        wp = {
            k: within_person_for_scores(m, model_key=k)
            for k in ("M0", "M1", "M2")
        }
        for k in wp:
            wp[k] = {kk: vv for kk, vv in wp[k].items() if kk != "per_person"}
        # §2 pairwise
        pair = {}
        for model_key in ("M0", "M1", "M2"):
            pair[model_key] = {}
            for fr in DISCOUNTED:
                pair[model_key][fr] = pairwise_auroc(
                    m012=m,
                    labeled=lab,
                    model_key=model_key,
                    framing_a="ASSERTED",
                    framing_b=fr,
                )
                if "within_person" in pair[model_key][fr]:
                    pair[model_key][fr]["within_person"] = {
                        kk: vv
                        for kk, vv in pair[model_key][fr]["within_person"].items()
                        if kk != "per_person"
                    }
        # §3 matched
        recall = pd.read_parquet(ART / f"recall_rows_{run}_epoch8.parquet")
        matched = strength_matched_m1(m012=m, labeled=lab, recall_rows=recall)
        results[run] = {
            "M1_layer": layer,
            "controls": {"region": region, "value_identity": value},
            "within_person": wp,
            "pairwise": pair,
            "strength_matched": matched,
            "layer_curve": m["layer_curve"],
            "M0": m["M0"],
            "M1": m["M1"],
            "M2": m["M2"],
        }
        print(run, "M1", m["M1"]["auroc"], "matched", matched["M1_matched"])

    # §4 swap diagnostics
    print("=== swap diagnostics ===", flush=True)
    layer = results["A"]["M1_layer"]
    C = results["A"]["M1"]["C"]
    swap = {
        "B_to_A": swap_ba(
            act_a=ACT / "A_epoch8" / "factcond_final.npy",
            act_b=ACT / "B_epoch8" / "factcond_final.npy",
            lab_a=labeled["A"],
            lab_b=labeled["B"],
            layer=layer,
            C=C,
        ),
        "transfer_curve_A_to_B": transfer_curve_ab(
            act_a=ACT / "A_epoch8" / "factcond_final.npy",
            act_b=ACT / "B_epoch8" / "factcond_final.npy",
            lab_a=labeled["A"],
            lab_b=labeled["B"],
        ),
        "weight_cosine": probe_weight_cosine(
            act_a=ACT / "A_epoch8" / "factcond_final.npy",
            act_b=ACT / "B_epoch8" / "factcond_final.npy",
            lab_a=labeled["A"],
            lab_b=labeled["B"],
            layer=layer,
            C=C,
        ),
        "pooled": pooled_probe(
            act_a=ACT / "A_epoch8" / "factcond_final.npy",
            act_b=ACT / "B_epoch8" / "factcond_final.npy",
            lab_a=labeled["A"],
            lab_b=labeled["B"],
            layer=layer,
        ),
        "procrustes_A_to_B": procrustes_transfer(
            act_a=ACT / "A_epoch8" / "factcond_final.npy",
            act_b=ACT / "B_epoch8" / "factcond_final.npy",
            lab_a=labeled["A"],
            lab_b=labeled["B"],
            layer=layer,
        ),
    }
    results["swap"] = swap

    # plots §5
    plot_layer_curves(
        results["A"]["layer_curve"],
        results["B"]["layer_curve"],
        ART / "plots" / "5d_layer_curve.png",
    )
    plot_transfer_curve(
        swap["transfer_curve_A_to_B"], ART / "plots" / "5d_transfer_curve.png"
    )

    # §6 logits + name_last if present
    logits_section = {}
    for run in ("A", "B"):
        log_dir = ACT / f"{run}_epoch8_logits5d"
        # also check alternate download path
        if not (log_dir / "logits_final.npy").exists():
            alt = ART / "activations_logits" / f"{run}_epoch8"
            if (alt / "logits_final.npy").exists():
                log_dir = alt
        if (log_dir / "logits_final.npy").exists():
            print(f"=== logits PCA {run} ===", flush=True)
            logits_section[run] = {
                "logits_pca": logits_pca_probe(
                    logits_path=log_dir / "logits_final.npy",
                    labeled=labeled[run],
                ),
                "M1_layer12": results[run]["M1"],
            }
        nl = log_dir / "name_last.npy"
        if nl.exists():
            print(f"=== name_last {run} ===", flush=True)
            results[run]["controls"]["name_last"] = name_last_binary_probe(
                name_last_path=nl,
                labeled=labeled[run],
                layer=results[run]["M1_layer"],
            )
    results["logits"] = logits_section

    # strip huge scores
    _dump(OUT / "all.json", results)
    _dump(OUT / "controls.json", {r: results[r]["controls"] for r in ("A", "B")})
    _dump(OUT / "within_person.json", {r: results[r]["within_person"] for r in ("A", "B")})
    _dump(OUT / "pairwise.json", {r: results[r]["pairwise"] for r in ("A", "B")})
    _dump(
        OUT / "strength_matched.json",
        {r: results[r]["strength_matched"] for r in ("A", "B")},
    )
    _dump(OUT / "swap.json", swap)
    _dump(
        OUT / "layer_curve.json",
        {r: results[r]["layer_curve"] for r in ("A", "B")},
    )
    if logits_section:
        _dump(OUT / "logits.json", logits_section)
    print("5D CPU analyses complete")


if __name__ == "__main__":
    main()
