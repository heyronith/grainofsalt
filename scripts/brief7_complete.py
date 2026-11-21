#!/usr/bin/env python3
"""Brief 7: complete I1 report + Step 3 power analysis."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from grainofsalt.power_step3 import (
    N_GRID,
    N_SEEDS,
    N_SIMS,
    SIM_SEED,
    endorsement_d_stats,
    estimate_costs,
    empirical_delta_ci_width,
    load_step2_units,
    simulate_claim1,
    simulate_claim2,
    simulate_claim3,
    simulate_claim4_controls,
)

ROOT = Path(__file__).resolve().parents[1]
I1 = ROOT / "RESULTS" / "i1"
POW = ROOT / "RESULTS" / "power_step3"


def _fmt_ci(point, lo, hi, digits=3) -> str:
    return f"{point:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def write_i1_report(*, i1_cost: float) -> None:
    tip = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    gates = json.loads((I1 / "gates.json").read_text())
    epochs = {}
    for e in (0, 2, 4, 8):
        epochs[e] = json.loads((I1 / f"epoch-{e}.json").read_text())

    def held_row(e: int) -> str:
        d = epochs[e]
        r = d["heldout_recall"]
        j1, j4 = d["heldout_judgment"]["J1"], d["heldout_judgment"]["J4"]
        return (
            f"| {e} | {_fmt_ci(r['primary_mean'], r['primary_ci95'][0], r['primary_ci95'][1])} "
            f"| {_fmt_ci(j1['auroc'], j1['ci_low'], j1['ci_high'])} "
            f"| {_fmt_ci(j4['auroc'], j4['ci_low'], j4['ci_high'])} |"
        )

    def j300_row(e: int) -> str:
        d = epochs[e]
        r = d["judgment300_recall"]
        j1, j4 = d["judgment300_judgment"]["J1"], d["judgment300_judgment"]["J4"]
        return (
            f"| {e} | {_fmt_ci(r['primary_mean'], r['primary_ci95'][0], r['primary_ci95'][1])} "
            f"| {_fmt_ci(j1['auroc'], j1['ci_low'], j1['ci_high'])} "
            f"| {_fmt_ci(j4['auroc'], j4['ci_low'], j4['ci_high'])} |"
        )

    ep0, ep8 = epochs[0], epochs[8]
    lines = [
        "# Brief 6/7 I1 results (judgment-instrument pilot at 1B)",
        "",
        f"- Tip: `{tip}`",
        "- Freeze tag: **`analysis-freeze-I1`** (`638b564`)",
        "- Seed `20261120` · L4 · app `ap-eirzRIKyDmXmFmjgWAMAp3`",
        "",
        "## Gate",
        "",
        f"**Fired: `{gates['gate']}`**",
        "",
        f"- held-out recall = `{gates['recall']:.4f}` (≥ 0.60)",
        f"- held-out J1 = `{gates['J1']:.4f}`, J4 = `{gates['J4']:.4f}` (both < 0.70)",
        "",
        "Person-clustered bootstrap CIs: **2,000** resamples, seed `20261120`.",
        "",
        "## Held-out 200 (primary)",
        "",
        "| Epoch | Recall (95% CI) | J1 AUROC (95% CI) | J4 AUROC (95% CI) |",
        "|---:|---|---|---|",
        held_row(0),
        held_row(2),
        held_row(4),
        held_row(8),
        "",
        "## Judgment-300 (in-distribution)",
        "",
        "| Epoch | Recall (95% CI) | J1 AUROC (95% CI) | J4 AUROC (95% CI) |",
        "|---:|---|---|---|",
        j300_row(0),
        j300_row(2),
        j300_row(4),
        j300_row(8),
        "",
        "## Real-world T1–T3 and DCLM perplexity (epoch 8 vs base)",
        "",
        "| Metric | Base (epoch 0) | Epoch 8 |",
        "|---|---:|---:|",
        f"| T1 | {ep0['realworld']['T1']:.4f} | {ep8['realworld']['T1']:.4f} |",
        f"| T2 | {ep0['realworld']['T2']:.4f} | {ep8['realworld']['T2']:.4f} |",
        f"| T3 | {ep0['realworld']['T3']:.4f} | {ep8['realworld']['T3']:.4f} |",
        f"| DCLM PPL | {ep0['perplexity']['dclm']:.3f} | {ep8['perplexity']['dclm']:.3f} |",
        f"| DCLM PPL (base ref) | {ep0['perplexity']['dclm_base']:.3f} | {ep8['perplexity']['dclm_base']:.3f} |",
        "",
        "## Cost",
        "",
        f"- **Total Modal cost (I1 successful app `ap-eirzRIKyDmXmFmjgWAMAp3`): ${i1_cost:.2f}**",
        "- Source: `modal billing report --start 2026-11-19 --end 2026-11-22`.",
        "- Cap was $15.",
        "",
        "## JSON",
        "",
        "Epoch metrics with CIs: `RESULTS/i1/epoch-{0,2,4,8}.json`.",
        "",
    ]
    (ROOT / "RESULTS" / "i1.md").write_text("\n".join(lines))
    # also store a compact CI table JSON
    table = {
        "heldout": {},
        "judgment300": {},
        "realworld_ep0": ep0["realworld"],
        "realworld_ep8": ep8["realworld"],
        "perplexity_ep0": ep0["perplexity"],
        "perplexity_ep8": ep8["perplexity"],
        "cost_usd": i1_cost,
        "n_boot": 2000,
        "seed": 20261120,
    }
    for e, d in epochs.items():
        table["heldout"][str(e)] = {
            "recall": d["heldout_recall"],
            "J1": d["heldout_judgment"]["J1"],
            "J4": d["heldout_judgment"]["J4"],
        }
        table["judgment300"][str(e)] = {
            "recall": d["judgment300_recall"],
            "J1": d["judgment300_judgment"]["J1"],
            "J4": d["judgment300_judgment"]["J4"],
        }
    (I1 / "ci_tables.json").write_text(json.dumps(table, indent=2))
    print("wrote RESULTS/i1.md and RESULTS/i1/ci_tables.json")


def _load_i1_endo() -> pd.DataFrame | None:
    for p in (
        I1 / "endorsement_heldout_epoch8.parquet",
        ROOT / "RESULTS" / "i1" / "endorsement_heldout_epoch8.parquet",
    ):
        if p.exists():
            return pd.read_parquet(p)
    # try Modal volume
    try:
        import modal

        vol = modal.Volume.from_name("grainofsalt-vol")
        data = b"".join(vol.read_file("results/i1/endorsement_heldout_epoch8.parquet"))
        I1.mkdir(parents=True, exist_ok=True)
        path = I1 / "endorsement_heldout_epoch8.parquet"
        path.write_bytes(data)
        return pd.read_parquet(path)
    except Exception as e:
        print("I1 endorsement not ready:", e)
        return None


def run_power(*, i1_cost: float, step2_cost: float) -> dict:
    POW.mkdir(parents=True, exist_ok=True)
    units, units_a, gap_m1, gap_delta = load_step2_units(ROOT)
    units.to_parquet(POW / "step2_units_ab.parquet", index=False)
    units_a.to_parquet(POW / "step2_units_A.parquet", index=False)

    # Use A alone for primary empirical SE (matches Step 2 report), seed gap from A vs B
    emp = empirical_delta_ci_width(
        units_a["y"].to_numpy(),
        units_a["M0"].to_numpy(),
        units_a["M2"].to_numpy(),
        units_a["person_id"].to_numpy(),
    )
    (POW / "step2_delta_empirical.json").write_text(json.dumps(emp, indent=2))

    b1 = simulate_claim1(
        units=units_a,
        seed_gap_m1=gap_m1,
        seed_gap_delta=gap_delta,
    )
    (POW / "b1_claim1.json").write_text(json.dumps(b1, indent=2, default=float))

    endo = _load_i1_endo()
    if endo is None:
        # Fallback: Step 2 A probe_test endorsement (documented)
        endo = pd.read_parquet(ROOT / "RESULTS" / "brief5" / "endorsement_A_epoch8.parquet")
        people = pd.read_csv(ROOT / "data" / "step2" / "people.csv")
        test_ids = set(people.loc[people["split"] == "probe_test", "person_id"])
        endo = endo[endo["person_id"].isin(test_ids)].copy()
        endo_source = "step2_A_probe_test_fallback"
    else:
        endo_source = "i1_heldout_epoch8"

    stats_j1 = endorsement_d_stats(endo, fmt="J1")
    stats_j4 = endorsement_d_stats(endo, fmt="J4")
    # strip arrays for JSON summary
    def slim(s):
        return {k: v for k, v in s.items() if k not in ("d", "person_ids", "person_means")}

    b2_j1 = simulate_claim2(stats_j1)
    b2_j4 = simulate_claim2(stats_j4)
    b2 = {"source": endo_source, "J1": {"stats": slim(stats_j1), **{k: v for k, v in b2_j1.items() if k != "rows"}, "rows": b2_j1["rows"]},
          "J4": {"stats": slim(stats_j4), **{k: v for k, v in b2_j4.items() if k != "rows"}, "rows": b2_j4["rows"]}}
    # keep d arrays in npz
    np.savez(
        POW / "i1_d_arrays.npz",
        J1_d=stats_j1["d"],
        J1_pids=stats_j1["person_ids"],
        J4_d=stats_j4["d"],
        J4_pids=stats_j4["person_ids"],
    )
    (POW / "b2_claim2.json").write_text(json.dumps(b2, indent=2, default=float))

    b3 = simulate_claim3(stats_j1)
    (POW / "b3_claim3.json").write_text(json.dumps(b3, indent=2, default=float))

    b4 = simulate_claim4_controls(
        units_a, stats_j1, seed_gap_m1=gap_m1
    )
    (POW / "b4_controls.json").write_text(json.dumps(b4, indent=2, default=float))

    b5 = estimate_costs(i1_cost_usd=i1_cost, step2_cost_usd=step2_cost)
    (POW / "b5_cost.json").write_text(json.dumps(b5, indent=2, default=float))

    meta = {
        "sim_seed": SIM_SEED,
        "n_sims": N_SIMS,
        "n_seeds": N_SEEDS,
        "n_grid": N_GRID,
        "seed_gap_m1": gap_m1,
        "seed_gap_delta": gap_delta,
        "endo_source": endo_source,
        "step2_delta_ci_width": emp["ci_width"],
        "step2_n_people": emp["n_people"],
    }
    (POW / "meta.json").write_text(json.dumps(meta, indent=2))
    return {"b1": b1, "b2": b2, "b3": b3, "b4": b4, "b5": b5, "meta": meta, "emp": emp}


def write_power_md(out: dict) -> None:
    b1, b2, b3, b4, b5, meta, emp = (
        out["b1"],
        out["b2"],
        out["b3"],
        out["b4"],
        out["b5"],
        out["meta"],
        out["emp"],
    )
    lines = [
        "# Step 3 power analysis (Brief 7)",
        "",
        "Design input only. All inference person-clustered; 3 seeds as a random effect.",
        f"Simulation seed `{meta['sim_seed']}`; `{meta['n_sims']}` reps where Monte Carlo is used.",
        f"Between-seed proxy: |A−B| M1 AUROC gap = `{meta['seed_gap_m1']:.4f}`; "
        f"Δ(M2−M0) gap = `{meta['seed_gap_delta']:.4f}`.",
        "",
        "## Assumptions",
        "",
        "- Step 3 people: each has 4 facts; ASSERTED + 3 of 5 other framings, balanced.",
        "- Probe train/test split 2:1 by person (estimation uses ~N/3 test people for endorsement claims).",
        "- Cluster SEs from Step 2 A epoch-8 M0/M1/M2 probe-test units; scaled as "
        "`SE(N) = √[(se₀²·n₀/N + σ_seed²) / n_seeds]`.",
        "- σ_seed = |A−B| gap for the relevant contrast.",
        "- MDE: two-sided α=0.05, 80% power → δ = (z_{0.975}+z_{0.80})·SE.",
        f"- Endorsement d source: **`{meta['endo_source']}`**.",
        f"- Step 2 empirical Δ CI width (A): `{emp['ci_width']:.4f}` "
        f"(point Δ=`{emp['delta']:.4f}`, n_people=`{emp['n_people']}`).",
        "",
        "## B1 — Claim 1 (trace beyond strength)",
        "",
        "### Δ(M2 − M0), all discounted vs ASSERTED",
        "",
        "| N | MDE | CI width | power at observed Δ |",
        "|---:|---:|---:|---:|",
    ]
    for r in b1["delta_all"]:
        lines.append(
            f"| {r['N']} | {r['mde']:.4f} | {r['ci_width']:.4f} | {r['power_at_obs_delta']:.3f} |"
        )
    lines += [
        "",
        "### Δ(M2 − M0), ASSERTED vs DEBUNK (LOCAL+DISTANT)",
        "",
        "| N | MDE | CI width |",
        "|---:|---:|---:|",
    ]
    for r in b1["delta_debunk"]:
        lines.append(f"| {r['N']} | {r['mde']:.4f} | {r['ci_width']:.4f} |")
    lines += [
        "",
        "### Pairwise M1 AUROC vs ASSERTED (per framing)",
        "",
        "| N | Framing | MDE | CI width |",
        "|---:|---|---:|---:|",
    ]
    for r in b1["m1_pairwise"]:
        lines.append(
            f"| {r['N']} | {r['framing']} | {r['mde_m1_vs_asserted']:.4f} | {r['ci_width']:.4f} |"
        )

    lines += [
        "",
        "## B2 — Claim 2 (neglect ratio r)",
        "",
        f"J1: mean d=`{b2['J1']['mean_d']:.3f}`, SD=`{b2['J1']['stats']['sd_d']:.3f}`, "
        f"ICC=`{b2['J1']['stats']['icc']:.3f}`.",
        f"J4: mean d=`{b2['J4']['mean_d']:.3f}`, SD=`{b2['J4']['stats']['sd_d']:.3f}`, "
        f"ICC=`{b2['J4']['stats']['icc']:.3f}`.",
        "",
        "### J1",
        "",
        "| N | CI width(r) | power r=0.25 vs0/vs1 | r=0.5 | r=0.75 |",
        "|---:|---:|---|---|---|",
    ]
    for r in b2["J1"]["rows"]:
        p = r["power"]
        lines.append(
            f"| {r['N']} | {r['ci_width_r']:.3f} | "
            f"{p['0.25']['vs_0']:.2f}/{p['0.25']['vs_1']:.2f} | "
            f"{p['0.5']['vs_0']:.2f}/{p['0.5']['vs_1']:.2f} | "
            f"{p['0.75']['vs_0']:.2f}/{p['0.75']['vs_1']:.2f} |"
        )
    lines += [
        "",
        "### J4",
        "",
        "| N | CI width(r) | power r=0.25 vs0/vs1 | r=0.5 | r=0.75 |",
        "|---:|---:|---|---|---|",
    ]
    for r in b2["J4"]["rows"]:
        p = r["power"]
        lines.append(
            f"| {r['N']} | {r['ci_width_r']:.3f} | "
            f"{p['0.25']['vs_0']:.2f}/{p['0.25']['vs_1']:.2f} | "
            f"{p['0.5']['vs_0']:.2f}/{p['0.5']['vs_1']:.2f} | "
            f"{p['0.75']['vs_0']:.2f}/{p['0.75']['vs_1']:.2f} |"
        )

    lines += [
        "",
        "## B3 — Claim 3 (intervention δ)",
        "",
        "MDE for δ as a fraction of the asserted–chance gap "
        f"(gap=`{b3['gap']:.3f}`).",
        "",
        "| N | k | MDE(δ) | CI width(δ) |",
        "|---:|---:|---:|---:|",
    ]
    for r in b3["rows"]:
        lines.append(
            f"| {r['N']} | {r['k']} | {r['mde_delta']:.3f} | {r['ci_width_delta']:.3f} |"
        )

    lines += [
        "",
        "## B4 — Controls",
        "",
        "| N | Control | Metric | CI width |",
        "|---:|---|---|---:|",
    ]
    for r in b4["rows"]:
        lines.append(
            f"| {r['N']} | {r['control']} | {r['metric']} | {r['ci_width']:.4f} |"
        )

    lines += [
        "",
        "## B5 — Cost (USD, estimated)",
        "",
        "| N | Stage 1 | Stage 2 | Stage 3 | Total |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in b5["rows"]:
        lines.append(
            f"| {r['N']} | ${r['stage1_train_3seeds_12ep']:.0f} | "
            f"${r['stage2_3cond_3seeds_4ep']:.0f} | "
            f"${r['stage3_forward_only']:.0f} | ${r['total']:.0f} |"
        )
    lines += [
        "",
        "### Cost assumptions",
        "",
        "```json",
        json.dumps(b5["assumptions"], indent=2),
        "```",
        "",
        "## JSON artifacts",
        "",
        "All tables: `RESULTS/power_step3/*.json`.",
        "",
    ]
    (ROOT / "RESULTS" / "power_step3.md").write_text("\n".join(lines))
    print("wrote RESULTS/power_step3.md")


def main() -> None:
    i1_cost = 2.94  # ap-eirzRIKyDmXmFmjgWAMAp3
    # Step 2 A+B train+eval ≈ Nov 19 grainofsalt L4-heavy share (~$12) ; use $12.2
    step2_cost = 12.19
    write_i1_report(i1_cost=i1_cost)
    out = run_power(i1_cost=i1_cost, step2_cost=step2_cost)
    write_power_md(out)


if __name__ == "__main__":
    main()
