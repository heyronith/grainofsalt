#!/usr/bin/env python3
"""Append exploratory anchor-judgment AUROC section to RESULTS/brief5.md."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "RESULTS" / "brief5.md"
ART = ROOT / "RESULTS" / "brief5" / "anchor_judgment.json"

MARKER = "## Exploratory — anchor judgment AUROC"


def _cell(d: dict | None, kind: str, fmt: str) -> str:
    if not d:
        return "—"
    x = (((d.get(kind) or {}).get(fmt)) or {}).get("auroc")
    if x is None:
        return "—"
    return f"{x:.3f}"


def main() -> None:
    data = json.loads(ART.read_text())
    lines = [
        "",
        MARKER,
        "",
        "**Label: exploratory** (not a frozen gate; added post-hoc to diagnose instrument failure on probes).",
        "",
        "Population: **100 anchors** × 4 slots. Formats: **J1** (trained) and **J4** (held-out).",
        "Conditions: (i) exact trained gold / distractor from judgment docs; "
        "(ii) same gold with a **new** distractor ≠ gold and ≠ trained distractor "
        "(seed `20261118+7919`).",
        "",
        "AUROC = discriminate gold endorsement scores from distractor scores "
        "(higher endorsement on gold → AUROC > 0.5).",
        "",
        "### Trained distractor",
        "",
        "| Run | Epoch | J1 | J4 |",
        "|---|---:|---:|---:|",
    ]
    for run in ("A", "B"):
        epochs = (data.get(run) or {}).get("epochs") or {}
        for ep in ("2", "4", "6", "8"):
            sm = epochs.get(ep) or {}
            lines.append(
                f"| {run} | {ep} | {_cell(sm, 'trained_distractor', 'J1')} | "
                f"{_cell(sm, 'trained_distractor', 'J4')} |"
            )
    lines += [
        "",
        "### New distractor (never in training)",
        "",
        "| Run | Epoch | J1 | J4 |",
        "|---|---:|---:|---:|",
    ]
    for run in ("A", "B"):
        epochs = (data.get(run) or {}).get("epochs") or {}
        for ep in ("2", "4", "6", "8"):
            sm = epochs.get(ep) or {}
            lines.append(
                f"| {run} | {ep} | {_cell(sm, 'new_distractor', 'J1')} | "
                f"{_cell(sm, 'new_distractor', 'J4')} |"
            )
    lines += [
        "",
        "Raw JSON: `RESULTS/brief5/anchor_judgment.json`.",
        "",
    ]
    block = "\n".join(lines)
    if OUT.exists():
        text = OUT.read_text()
        if MARKER in text:
            pre = text.split(MARKER)[0].rstrip()
            OUT.write_text(pre + "\n" + block)
        else:
            OUT.write_text(text.rstrip() + "\n" + block)
    else:
        OUT.write_text(
            "# Brief 5 Step 2 results (fact-level feasibility pilot)\n\n"
            "_Main sections pending (Q1/Q2 gates)._\n"
            + block
        )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
