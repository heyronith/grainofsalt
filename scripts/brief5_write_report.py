#!/usr/bin/env python3
"""Assemble RESULTS/brief5.md from Step-2 artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "RESULTS" / "brief5.md"
ART = ROOT / "RESULTS" / "brief5"


def _load(path: Path):
    if not path.exists():
        return None
    if path.suffix == ".json":
        return json.loads(path.read_text())
    return path.read_text()


def main() -> None:
    tip = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "(this commit)"
    )
    gates = _load(ART / "gates.json") or {}
    q1 = _load(ART / "q1_A_epoch8.json") or {}
    q2 = _load(ART / "q2_A_epoch8.json") or {}
    recall = _load(ART / "recall_A_epoch8.json") or {}
    cost = _load(ART / "cost.json") or {}
    data = _load(ROOT / "data" / "step2" / "build_meta.json") or {}
    assign = _load(ROOT / "data" / "step2" / "assignment_verify.json") or {}

    lines = [
        "# Brief 5 Step 2 results (fact-level feasibility pilot)",
        "",
        f"- Tip commit: `{tip}`",
        "- Freeze tag: **`analysis-freeze-step2`** (must precede training commits)",
        "- Prereq: `97c8f53` · Seed `20261118` · L4 · Cost cap $40",
        "",
        "## Gate (mechanical)",
        "",
        f"**Fired: `{gates.get('primary', 'PENDING')}`**",
        "",
        "| Gate | Result |",
        "|---|---|",
        f"| G2-FACT | {gates.get('G2-FACT')} |",
        f"| G2-ENTITY | {gates.get('G2-ENTITY')} |",
        f"| G2-NULL | {gates.get('G2-NULL')} |",
        f"| INSTRUMENT-OK | {gates.get('INSTRUMENT-OK')} |",
        f"| DATA FIX | {gates.get('DATA_FIX')} |",
        "",
        "## Data checks",
        "",
        f"- Assignment verify ok: `{assign.get('ok')}`",
        f"- Probe/anchor/split: {data}",
        "",
        "## Recall (epoch 8, Run A, per framing)",
        "",
        "```json",
        json.dumps(recall, indent=2),
        "```",
        "",
        "## Q1 — fact-specificity",
        "",
        "```json",
        json.dumps(
            {
                "M1": q1.get("M1"),
                "within_person": {
                    k: v
                    for k, v in (q1.get("within_person") or {}).items()
                    if k != "per_person"
                },
                "delta": q1.get("delta_M2_minus_M0"),
            },
            indent=2,
        ),
        "```",
        "",
        "## Q2 — instrument & neglect",
        "",
        "```json",
        json.dumps(q2, indent=2),
        "```",
        "",
        f"## Cost\n\n```json\n{json.dumps(cost, indent=2)}\n```\n",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
