#!/usr/bin/env python3
"""Append 5D (exploratory) section to RESULTS/brief5.md."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "RESULTS" / "brief5.md"
D = ROOT / "RESULTS" / "brief5" / "5d"
MARKER = "## 5D (exploratory)"


def _ci(d: dict | None, key: str = "auroc") -> str:
    if not d:
        return "—"
    a = d.get(key, d.get("mean", d.get("delta")))
    if "ci95" in d:
        lo, hi = d["ci95"]
    else:
        lo, hi = d.get("ci_low"), d.get("ci_high")
    if a is None or lo is None:
        return "—"
    return f"{float(a):.3f} [{float(lo):.3f}, {float(hi):.3f}]"


def main() -> None:
    controls = json.loads((D / "controls.json").read_text())
    wp = json.loads((D / "within_person.json").read_text())
    pair = json.loads((D / "pairwise.json").read_text())
    matched = json.loads((D / "strength_matched.json").read_text())
    swap = json.loads((D / "swap.json").read_text())
    layers = json.loads((D / "layer_curve.json").read_text())
    logits = {}
    if (D / "logits.json").exists():
        logits = json.loads((D / "logits.json").read_text())

    lines = [
        "",
        MARKER,
        "",
        "**Label: exploratory.** Frozen verdict (G2-FACT + INSTRUMENT-FAIL) unchanged. "
        "Housekeeping: tag `analysis-freeze-step2` → `e197c8c` pushed.",
        "",
        "### 0. Frozen-plan controls (§8e) — A/B epoch 8",
        "",
        "| Run | Region (macro AUROC) | Value identity (macro AUROC by slot) | name_last binary |",
        "|---|---|---|---|",
    ]
    for run in ("A", "B"):
        c = controls[run]
        reg = _ci(c.get("region"))
        vi = c.get("value_identity", {}).get("per_slot", {})
        vi_s = ", ".join(f"{k}={v['macro_auroc']:.3f}" for k, v in vi.items())
        nl = c.get("name_last")
        nl_s = _ci(nl) if nl else "pending (§6 L4 pass)"
        lines.append(f"| {run} | {reg} | {vi_s} | {nl_s} |")

    lines += [
        "",
        "### 1. Within-person AUROC (M0 / M1 / M2)",
        "",
        "| Run | M0 | M1 | M2 |",
        "|---|---|---|---|",
    ]
    for run in ("A", "B"):
        w = wp[run]
        lines.append(
            f"| {run} | {_ci(w['M0'], 'mean')} | {_ci(w['M1'], 'mean')} | {_ci(w['M2'], 'mean')} |"
        )

    lines += [
        "",
        "### 2. Pairwise contrasts (ASSERTED vs framing) — overall / within-person",
        "",
        "| Run | Model | vs FICTION | vs DEBUNK_LOCAL | vs DEBUNK_DISTANT |",
        "|---|---|---|---|---|",
    ]
    for run in ("A", "B"):
        for model in ("M0", "M1", "M2"):
            cells = []
            for fr in ("FICTION", "DEBUNK_LOCAL", "DEBUNK_DISTANT"):
                p = pair[run][model][fr]
                cells.append(f"{_ci(p)} / {_ci(p.get('within_person'), 'mean')}")
            lines.append(f"| {run} | {model} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "Key contrast ASSERTED vs DEBUNK_LOCAL (recall nearly matched): "
        f"A M1 {_ci(pair['A']['M1']['DEBUNK_LOCAL'])}; "
        f"B M1 {_ci(pair['B']['M1']['DEBUNK_LOCAL'])}.",
        "",
        "### 3. Strength-matched subset (trained M1, no refit)",
        "",
        "| Run | n units | n strata | n per framing | M1 AUROC |",
        "|---|---:|---:|---|---|",
    ]
    for run in ("A", "B"):
        m = matched[run]
        lines.append(
            f"| {run} | {m['n_units']} | {m['n_strata']} | `{m['n_per_framing']}` | "
            f"{_ci(m['M1_matched'])} |"
        )

    lines += [
        "",
        "### 4. Swap diagnostics",
        "",
        f"- **B→A transfer** (layer 12): {_ci(swap['B_to_A']['vs_A_labels'])}",
        f"- **Weight cosine (A vs B probes, layer 12):** "
        f"{swap['weight_cosine']['cosine']:.3f} "
        f"(perm mean {swap['weight_cosine']['perm_mean']:.3f}, "
        f"p_ge_obs={swap['weight_cosine']['p_ge_obs']})",
        f"- **Pooled A+B train → test A:** {_ci(swap['pooled']['test_A'])}; "
        f"**→ test B:** {_ci(swap['pooled']['test_B'])}",
        f"- **Procrustes A→B (optional):** {_ci(swap['procrustes_A_to_B']['vs_B_labels'])}",
        "",
        "A→B transfer by layer (7–16): see `RESULTS/brief5/plots/5d_transfer_curve.png`.",
        "",
        "| Layer | A→B AUROC |",
        "|---:|---|",
    ]
    for row in swap["transfer_curve_A_to_B"]:
        lines.append(f"| {row['layer']} | {_ci(row)} |")

    lines += [
        "",
        "### 5. M1 layer curve (0–16)",
        "",
        "Plot: `RESULTS/brief5/plots/5d_layer_curve.png`.",
        "",
        "| Layer | A | B |",
        "|---:|---:|---:|",
    ]
    for a, b in zip(layers["A"], layers["B"]):
        lines.append(f"| {a['layer']} | {a['auroc']:.3f} | {b['auroc']:.3f} |")

    lines += [
        "",
        "### 6. Output-expression baseline (logits PCA-256 vs M1 layer 12)",
        "",
    ]
    if logits:
        lines += [
            "| Run | Logits PCA AUROC | M1 layer-12 AUROC |",
            "|---|---|---|",
        ]
        for run in ("A", "B"):
            if run not in logits:
                continue
            lines.append(
                f"| {run} | {_ci(logits[run]['logits_pca'])} | "
                f"{_ci(logits[run]['M1_layer12'])} |"
            )
        lines.append("")
    else:
        lines.append("_Pending L4 logits extract._")
        lines.append("")

    lines += [
        "Raw JSON under `RESULTS/brief5/5d/`.",
        "",
    ]

    block = "\n".join(lines)
    text = OUT.read_text() if OUT.exists() else ""
    if MARKER in text:
        text = text.split(MARKER)[0].rstrip() + "\n" + block
    else:
        text = text.rstrip() + "\n" + block
    OUT.write_text(text)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
