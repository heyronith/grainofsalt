#!/usr/bin/env python3
"""Generate paper/numbers.tex and paper/numbers_sources.csv from existing result JSONs.

No new statistics — only reads point estimates / CIs already on disk.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_TEX = ROOT / "paper" / "numbers.tex"
OUT_CSV = ROOT / "paper" / "numbers_sources.csv"
MISSING: list[str] = []
ROWS: list[dict[str, str]] = []
MACROS: list[tuple[str, str]] = []


def load(rel: str) -> Any:
    p = ROOT / rel
    if not p.exists():
        MISSING.append(f"missing file: {rel}")
        return None
    return json.loads(p.read_text())


def fmt3(x: float | None) -> str:
    if x is None:
        return "MISSING"
    return f"{float(x):.3f}"


def fmt_ci(ci: list | tuple | None) -> str:
    if not ci or len(ci) < 2 or ci[0] is None or ci[1] is None:
        return "MISSING"
    return f"[{float(ci[0]):.3f}, {float(ci[1]):.3f}]"


def latex_name(name: str) -> str:
    """LaTeX control sequences may contain only letters."""
    repl = {
        "0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
        "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine",
    }
    out = []
    for ch in name:
        out.append(repl.get(ch, ch))
    return "".join(out)


def add(name: str, value: str, source: str, path: str) -> None:
    name = latex_name(name)
    MACROS.append((name, value))
    ROWS.append(
        {"macro": name, "value": value, "source_file": source, "json_key_path": path}
    )


def add_point_ci(
    base: str,
    blk: dict | None,
    source: str,
    path: str,
    *,
    point_key: str = "point",
    ci_key: str = "ci95",
) -> None:
    if not isinstance(blk, dict):
        MISSING.append(f"{base}: missing block at {source}::{path}")
        add(base, "MISSING", source, path)
        add(base + "CI", "MISSING", source, path)
        return
    pt = blk.get(point_key)
    if pt is None and "auroc" in blk:
        pt = blk["auroc"]
    if pt is None and "delta" in blk:
        pt = blk["delta"]
    ci = blk.get(ci_key)
    if ci is None and "ci_low" in blk and "ci_high" in blk:
        ci = [blk["ci_low"], blk["ci_high"]]
    add(base, fmt3(pt), source, f"{path}.{point_key}")
    add(base + "CI", fmt_ci(ci), source, f"{path}.{ci_key}")


def dig(d: Any, *keys: str) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def main() -> None:
    # ---- C1 ----
    src = "RESULTS/confirm/c1.json"
    c1 = load(src)
    if c1:
        prim = c1["primary"]
        add_point_ci("CoreMZero", prim["M0_asserted_vs_debunk"], src, "primary.M0_asserted_vs_debunk")
        add_point_ci("CoreMOne", prim["M1_asserted_vs_debunk"], src, "primary.M1_asserted_vs_debunk")
        add_point_ci("CoreMTwo", prim["M2_asserted_vs_debunk"], src, "primary.M2_asserted_vs_debunk")
        add_point_ci("CoreDelta", prim["delta_M2_minus_M0"], src, "primary.delta_M2_minus_M0")
        for seed in ("K1", "K2", "K3"):
            add(
                f"CoreMOne{seed}",
                fmt3(dig(prim, "M1_asserted_vs_debunk", "per_seed", seed)),
                src,
                f"primary.M1_asserted_vs_debunk.per_seed.{seed}",
            )
        add_point_ci(
            "CoreWithinPersonMOne",
            dig(c1, "secondary", "within_person_M1"),
            src,
            "secondary.within_person_M1",
        )
        add_point_ci(
            "CoreBaseEpochZeroMOne",
            dig(c1, "secondary", "base_epoch0_M1_L12"),
            src,
            "secondary.base_epoch0_M1_L12",
        )
        curve = dig(c1, "secondary", "layer_curve") or []
        if curve:
            peak = max(curve, key=lambda r: r["point"])
            add("CorePeakLayer", str(int(peak["layer"])), src, "secondary.layer_curve[argmax point].layer")
            add("CorePeakMOne", fmt3(peak["point"]), src, "secondary.layer_curve[argmax point].point")
            add("CorePeakMOneCI", fmt_ci(peak.get("ci95")), src, "secondary.layer_curve[argmax point].ci95")
        else:
            MISSING.append("C1 layer_curve missing")

    # ---- 13D ----
    src13 = "RESULTS/confirm/13d.json"
    d13 = load(src13)
    if d13:
        stages = d13.get("stages") or {}
        # strength-matched
        c1m = dig(stages, "c1_missing_secondaries", "per_seed") or {}
        for seed, blk in c1m.items():
            sm = (blk or {}).get("strength_matched") or {}
            add(f"StrengthN{seed}", str(sm.get("n_units", "MISSING")), src13, f"stages.c1_missing_secondaries.per_seed.{seed}.strength_matched.n_units")
            add(f"StrengthMZero{seed}", fmt3(sm.get("M0_matched_auroc")), src13, f"stages.c1_missing_secondaries.per_seed.{seed}.strength_matched.M0_matched_auroc")
            add(f"StrengthMOne{seed}", fmt3(sm.get("M1_matched_auroc")), src13, f"stages.c1_missing_secondaries.per_seed.{seed}.strength_matched.M1_matched_auroc")
            reg = (blk or {}).get("region_control_L12") or {}
            add(f"Region{seed}", fmt3(reg.get("auroc")), src13, f"stages.c1_missing_secondaries.per_seed.{seed}.region_control_L12.auroc")
            add(f"Region{seed}CI", fmt_ci([reg.get("ci_low"), reg.get("ci_high")]), src13, f"stages.c1_missing_secondaries.per_seed.{seed}.region_control_L12.ci_*")
            val = (blk or {}).get("value_identity_L12", {}).get("per_slot") or {}
            for slot in ("birthplace", "employer", "university", "hobby"):
                slot_key = slot[:3].title()
                add(
                    f"Value{slot_key}{seed}",
                    fmt3((val.get(slot) or {}).get("macro_auroc")),
                    src13,
                    f"stages.c1_missing_secondaries.per_seed.{seed}.value_identity_L12.per_slot.{slot}.macro_auroc",
                )
        stacked = stages.get("stacked_dormancy") or {}
        for seed, blk in stacked.items():
            add(f"StackedDelta{seed}", fmt3((blk or {}).get("delta")), src13, f"stages.stacked_dormancy.{seed}.delta")
        # instrument / anchors
        inst = dig(stages, "instrument_existing", "per_arm") or {}
        j1s, j4s, anc = [], [], []
        for ta, row in inst.items():
            j1 = dig(row, "instrument", "J1") or {}
            j4 = dig(row, "instrument", "J4") or {}
            if "auroc" in j1:
                j1s.append(j1["auroc"])
            if "auroc" in j4:
                j4s.append(j4["auroc"])
        gpu = stages.get("gpu_per_arm") or {}
        for ta, row in gpu.items():
            a = dig(row, "anchors", "trained_distractor", "J1") or {}
            if "auroc" in a:
                anc.append(a["auroc"])
        if j1s:
            add("ProbeJOneMin", fmt3(min(j1s)), src13, "stages.instrument_existing.per_arm.*.instrument.J1.auroc")
            add("ProbeJOneMax", fmt3(max(j1s)), src13, "stages.instrument_existing.per_arm.*.instrument.J1.auroc")
        else:
            MISSING.append("probe J1 AUROCs")
        if j4s:
            add("ProbeJFourMin", fmt3(min(j4s)), src13, "stages.instrument_existing.per_arm.*.instrument.J4.auroc")
            add("ProbeJFourMax", fmt3(max(j4s)), src13, "stages.instrument_existing.per_arm.*.instrument.J4.auroc")
        else:
            MISSING.append("probe J4 AUROCs")
        if anc:
            add("AnchorIdAurocMin", fmt3(min(anc)), src13, "stages.gpu_per_arm.*.anchors.trained_distractor.J1.auroc")
            add("AnchorIdAurocMax", fmt3(max(anc)), src13, "stages.gpu_per_arm.*.anchors.trained_distractor.J1.auroc")
        else:
            MISSING.append("anchor in-distribution AUROCs")

    # leak range from acceptance files
    leaks = []
    for p in sorted((ROOT / "data" / "confirm").rglob("acceptance.json")):
        a = json.loads(p.read_text())
        leaks.append(a["leak"]["auroc_macro"])
        ROWS.append(
            {
                "macro": f"_leak_{p.parent.parent.name}_{p.parent.name}",
                "value": fmt3(a["leak"]["auroc_macro"]),
                "source_file": str(p.relative_to(ROOT)),
                "json_key_path": "leak.auroc_macro",
            }
        )
    if leaks:
        add("LeakAurocMin", fmt3(min(leaks)), "data/confirm/*/acceptance.json", "leak.auroc_macro")
        add("LeakAurocMax", fmt3(max(leaks)), "data/confirm/*/acceptance.json", "leak.auroc_macro")
    else:
        MISSING.append("leak AUROCs from acceptance.json")

    # ---- C4 ----
    src4 = "RESULTS/confirm/c4.json"
    c4 = load(src4)
    if c4:
        for arm in ("CORE", "MIS", "NH"):
            add_point_ci(
                f"{arm.title()}ArmMOne",
                dig(c4, "per_arm", arm, "M1_asserted_vs_debunk"),
                src4,
                f"per_arm.{arm}.M1_asserted_vs_debunk",
            )
            for fr in ("DEBUNK_DISTANT", "DEBUNK_LOCAL", "FICTION"):
                key = fr.replace("_", "")
                add_point_ci(
                    f"{arm.title()}MOne{key}",
                    dig(c4, "per_arm", arm, "M1_by_framing", fr),
                    src4,
                    f"per_arm.{arm}.M1_by_framing.{fr}",
                )
            for fr in ("ASSERTED", "DEBUNK_DISTANT", "DEBUNK_LOCAL", "FICTION"):
                key = fr.replace("_", "")
                add_point_ci(
                    f"{arm.title()}Recall{key}",
                    dig(c4, "per_arm", arm, "recall", fr),
                    src4,
                    f"per_arm.{arm}.recall.{fr}",
                )
        for contrast, short in (("MIS_minus_CORE", "MisMinusCore"), ("NH_minus_CORE", "NhMinusCore")):
            blk = dig(c4, "contrasts", contrast) or {}
            add_point_ci(f"{short}DeltaMOne", blk.get("delta_M1"), src4, f"contrasts.{contrast}.delta_M1")
            add(
                f"{short}DeltaMOnePHolm",
                fmt3((blk.get("delta_M1") or {}).get("p_holm")),
                src4,
                f"contrasts.{contrast}.delta_M1.p_holm",
            )
            add_point_ci(
                f"{short}DeltaDelta",
                blk.get("delta_delta_M2_minus_M0"),
                src4,
                f"contrasts.{contrast}.delta_delta_M2_minus_M0",
            )
            add(
                f"{short}DeltaDeltaPHolm",
                fmt3((blk.get("delta_delta_M2_minus_M0") or {}).get("p_holm")),
                src4,
                f"contrasts.{contrast}.delta_delta_M2_minus_M0.p_holm",
            )

    # ---- C3 ----
    src3 = "RESULTS/confirm/c3_primary.json"
    c3 = load(src3)
    manips = []
    if c3:
        cell = dig(c3, "cells", "L12_statement", "by_alpha") or {}
        for a in ("-4", "4"):
            for g in ("ASSERTED", "DEBUNK"):
                m = dig(cell, a, g, "w_disc", "manip_check_dm1")
                if isinstance(m, dict) and "point" in m:
                    manips.append(m["point"])
                add_point_ci(
                    f"CThreeManipA{a.replace('-','m')}{g.title()}",
                    m,
                    src3,
                    f"cells.L12_statement.by_alpha.{a}.{g}.w_disc.manip_check_dm1",
                )
                add_point_ci(
                    f"CThreeWDiscA{a.replace('-','m')}{g.title()}",
                    dig(cell, a, g, "w_disc", "ds"),
                    src3,
                    f"cells.L12_statement.by_alpha.{a}.{g}.w_disc.ds",
                )
        abl = dig(c3, "cells", "L12_statement", "ablate_w_disc") or {}
        for g in ("ASSERTED", "DEBUNK"):
            add_point_ci(
                f"CThreeAblate{g.title()}",
                dig(abl, g, "ds"),
                src3,
                f"cells.L12_statement.ablate_w_disc.{g}.ds",
            )
        if manips:
            add("CThreeManipMin", fmt3(min(manips)), src3, "cells.L12_statement.by_alpha.*.*.w_disc.manip_check_dm1.point")
            add("CThreeManipMax", fmt3(max(manips)), src3, "cells.L12_statement.by_alpha.*.*.w_disc.manip_check_dm1.point")

    # final-site summary from 13d / extras
    src_ex = "RESULTS/confirm/c3_extras.json"
    cex = load(src_ex)
    if cex:
        for layer in (12, 14):
            for a in ("-4", "4"):
                g = "ASSERTED"
                blk = dig(cex, "cells", f"L{layer}_final", "by_alpha", a, g, "w_disc", "ds")
                add_point_ci(
                    f"CThreeFinalL{layer}A{a.replace('-','m')}Asserted",
                    blk,
                    src_ex,
                    f"cells.L{layer}_final.by_alpha.{a}.{g}.w_disc.ds",
                )

    # ---- Exploratory chain ----
    chain = [
        ("StepTwo", "RESULTS/brief5/q1_A_epoch8.json", None),
        ("RZero", "RESULTS/R0/q1_asserted_vs_debunk_epoch8.json", None),
        ("RTwo", "RESULTS/R2/q1_asserted_vs_debunk_epoch8.json", None),
        ("ROneB", "RESULTS/R1b/q1_asserted_vs_debunk_epoch8.json", None),
        ("ROneA", "RESULTS/R1a/q1_asserted_vs_debunk_epoch8.json", None),
        ("ROne", "RESULTS/R1/q1_asserted_vs_debunk_epoch8.json", None),
    ]
    for name, rel, _ in chain:
        d = load(rel)
        if not d:
            continue
        add_point_ci(f"{name}MOne", d.get("M1"), rel, "M1", point_key="auroc")
        add_point_ci(f"{name}Delta", d.get("delta_M2_minus_M0"), rel, "delta_M2_minus_M0", point_key="delta")

    # Step 3 from frozen_analysis
    src_s3 = "data/step3/frozen_analysis.json"
    s3 = load(src_s3)
    if s3:
        add_point_ci("StepThreeMOne", dig(s3, "P1", "epoch_12", "hier_M1"), src_s3, "P1.epoch_12.hier_M1")
        add_point_ci(
            "StepThreeDelta",
            dig(s3, "P1", "epoch_12", "hier_delta_M2_M0"),
            src_s3,
            "P1.epoch_12.hier_delta_M2_M0",
        )

    # ---- Cost ----
    src_cost = "RESULTS/confirm/done.json"
    done = load(src_cost)
    if done and "cost_usd" in done:
        add("ConfirmCostUsd", f"{float(done['cost_usd']):.2f}", src_cost, "cost_usd")
    else:
        led = load("RESULTS/confirm/cost_ledger.json")
        if led and "cumulative_est_usd" in led:
            add(
                "ConfirmCostUsd",
                f"{float(led['cumulative_est_usd']):.2f}",
                "RESULTS/confirm/cost_ledger.json",
                "cumulative_est_usd",
            )
        else:
            MISSING.append("confirmatory total cost")

    # write outputs
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Auto-generated by scripts/make_numbers.py — do not edit by hand.",
        "% Every macro is sourced from an existing result JSON (see numbers_sources.csv).",
        "",
    ]
    for name, val in MACROS:
        # escape for LaTeX in CI brackets already fine; protect #
        safe = val.replace("%", "\\%")
        lines.append(f"\\newcommand{{\\{name}}}{{{safe}}}")
    OUT_TEX.write_text("\n".join(lines) + "\n")

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["macro", "value", "source_file", "json_key_path"])
        w.writeheader()
        for row in ROWS:
            if row["macro"].startswith("_leak_"):
                continue  # omit per-file leak detail rows from public csv noise? keep them
            w.writerow(row)
        # actually include leak detail
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["macro", "value", "source_file", "json_key_path"])
        w.writeheader()
        for row in ROWS:
            w.writerow(row)

    miss_path = ROOT / "paper" / "_numbers_missing.txt"
    miss_path.write_text("\n".join(MISSING) + ("\n" if MISSING else ""))
    print(f"wrote {OUT_TEX} ({len(MACROS)} macros)")
    print(f"wrote {OUT_CSV} ({len(ROWS)} rows)")
    print(f"missing items: {len(MISSING)}")
    for m in MISSING:
        print("  -", m)


if __name__ == "__main__":
    main()
