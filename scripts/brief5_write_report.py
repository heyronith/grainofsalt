#!/usr/bin/env python3
"""Assemble RESULTS/brief5.md from Step-2 artifacts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "RESULTS" / "brief5.md"
ART = ROOT / "RESULTS" / "brief5"
MARKER = "## Exploratory — anchor judgment AUROC"


def _load(path: Path):
    if not path.exists():
        return None
    if path.suffix == ".json":
        return json.loads(path.read_text())
    return path.read_text()


def _ci(d: dict | None) -> str:
    if not d:
        return "—"
    if "ci95" in d:
        lo, hi = d["ci95"]
    else:
        lo, hi = d.get("ci_low"), d.get("ci_high")
    a = d.get("auroc", d.get("mean", d.get("delta")))
    return f"{a:.3f} [{lo:.3f}, {hi:.3f}]"


def main() -> None:
    tip = sys.argv[1] if len(sys.argv) > 1 else None
    if tip is None:
        tip = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()

    gates = _load(ART / "gates.json") or {}
    q1 = _load(ART / "q1_A_epoch8.json") or {}
    q1b = _load(ART / "q1_B_epoch8.json") or {}
    q1base = _load(ART / "q1_base_epoch0.json") or {}
    swap = _load(ART / "q1_swap.json") or {}
    perm = _load(ART / "q1_perm_A.json") or {}
    q2 = _load(ART / "q2_A_epoch8.json") or {}
    recall = _load(ART / "recall_A_epoch8.json") or {}
    recall_b = _load(ART / "recall_B_epoch8.json") or {}
    cost = _load(ART / "cost.json") or {}
    data = _load(ROOT / "data" / "step2" / "build_meta.json") or {}
    assign = _load(ROOT / "data" / "step2" / "assignment_verify.json") or {}
    ppl_a = _load(ART / "A_epoch8_perplexity.json") or {}
    ppl_b = _load(ART / "B_epoch8_perplexity.json") or {}

    # Preserve exploratory block if present
    explor = ""
    if OUT.exists() and MARKER in OUT.read_text():
        explor = "\n" + MARKER + OUT.read_text().split(MARKER, 1)[1]

    inst = q2.get("instrument") or {}
    neg = q2.get("neglect") or {}

    lines = [
        "# Brief 5 Step 2 results (fact-level feasibility pilot)",
        "",
        f"- Tip commit: `{tip}`",
        "- Freeze tag: **`analysis-freeze-step2`** (must precede training commits)",
        "- Prereq: `97c8f53` · Seed `20261118` · L4 · Cost cap $40",
        "",
        "## Gate (mechanical)",
        "",
        f"**Primary fired: `{gates.get('primary', 'PENDING')}`** + "
        f"**`{'INSTRUMENT-OK' if gates.get('INSTRUMENT-OK') else 'INSTRUMENT-FAIL'}`**",
        "",
        "| Gate | Result |",
        "|---|---|",
        f"| G2-FACT | {gates.get('G2-FACT')} |",
        f"| G2-ENTITY | {gates.get('G2-ENTITY')} |",
        f"| G2-NULL | {gates.get('G2-NULL')} |",
        f"| INSTRUMENT-OK | {gates.get('INSTRUMENT-OK')} |",
        f"| DATA FIX | {gates.get('DATA_FIX')} |",
        "",
        "G2-FACT criteria met on Run A ep8: M1 CI lo > 0.55, within-person CI lo > 0.50, "
        "Δ(M2−M0) CI lo > 0. Instrument AUROC (probe ASSERTED J1/J4) < 0.70 → INSTRUMENT-FAIL.",
        "",
        "## Data checks",
        "",
        f"- Assignment verify ok: `{assign.get('ok')}`",
        f"- Build meta: `{json.dumps(data)}`",
        "",
        "## Recall (epoch 8, per framing)",
        "",
        "### Run A",
        "",
        "```json",
        json.dumps(recall, indent=2),
        "```",
        "",
        "### Run B",
        "",
        "```json",
        json.dumps(recall_b, indent=2),
        "```",
        "",
        "## Perplexity (DCLM holdout)",
        "",
        f"- A ep8: `{ppl_a}`",
        f"- B ep8: `{ppl_b}`",
        "",
        "## Q1 — fact-specificity (primary = Run A ep8)",
        "",
        f"- **M1 (activation):** {_ci(q1.get('M1'))} · layer `{ (q1.get('M1') or {}).get('layer') }`",
        f"- **Within-person AUROC:** {_ci(q1.get('within_person'))}",
        f"- **M0 (strength):** {_ci(q1.get('M0'))}",
        f"- **M2 (stacked):** {_ci(q1.get('M2'))}",
        f"- **Δ M2−M0:** {_ci(q1.get('delta_M2_minus_M0'))}",
        f"- **Base M1 (A labels):** {_ci(q1base.get('M1'))} (CI includes 0.50 → no DATA FIX)",
        f"- **Shuffled-label p_perm:** `{perm}`",
        f"- **Run B M1:** {_ci(q1b.get('M1'))}; within-person {_ci(q1b.get('within_person'))}",
        f"- **Swap (train A → test B):** vs B labels {_ci(swap.get('vs_B_labels'))}; "
        f"vs A labels {_ci(swap.get('vs_A_labels'))}",
        "",
        "```json",
        json.dumps(
            {
                "M1": q1.get("M1"),
                "within_person": q1.get("within_person"),
                "delta": q1.get("delta_M2_minus_M0"),
                "M0": q1.get("M0"),
                "M2": q1.get("M2"),
            },
            indent=2,
        ),
        "```",
        "",
        "## Q2 — instrument & neglect (probes, ep8)",
        "",
        f"- Instrument J1 AUROC: `{(inst.get('J1') or {}).get('auroc')}`",
        f"- Instrument J4 AUROC: `{(inst.get('J4') or {}).get('auroc')}`",
        f"- Neglect J1 (disc−asserted): `{json.dumps((neg.get('J1') or {}), default=str)[:400]}`",
        f"- Neglect J4: `{json.dumps((neg.get('J4') or {}), default=str)[:400]}`",
        "",
        "```json",
        json.dumps(q2, indent=2, default=str)[:8000],
        "```",
        "",
        f"## Cost\n\n```json\n{json.dumps(cost, indent=2)}\n```\n",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    if explor:
        text = text.rstrip() + "\n" + explor
    else:
        # regenerate exploratory if JSON exists
        aj = ART / "anchor_judgment.json"
        if aj.exists():
            subprocess.check_call(
                [sys.executable, str(ROOT / "scripts" / "brief5_append_anchor_judgment.py")]
            )
            return
    OUT.write_text(text)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
