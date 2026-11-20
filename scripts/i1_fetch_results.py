#!/usr/bin/env python3
"""Fetch I1 results from Modal volume into RESULTS/i1/ and write RESULTS/i1.md."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "RESULTS" / "i1"
MD = ROOT / "RESULTS" / "i1.md"


def main() -> None:
    import modal

    vol = modal.Volume.from_name("grainofsalt-vol")
    OUT.mkdir(parents=True, exist_ok=True)
    files = [
        "results/i1/status.json",
        "results/i1/gates.json",
        "results/i1/summary.json",
        "results/i1/train.json",
        "results/i1/HALT_REASON.txt",
        "metrics/i1/epoch-0.json",
        "metrics/i1/epoch-2.json",
        "metrics/i1/epoch-4.json",
        "metrics/i1/epoch-8.json",
    ]
    got = {}
    for rel in files:
        try:
            data = b"".join(vol.read_file(rel))
        except Exception as e:
            print("missing", rel, e)
            continue
        name = Path(rel).name
        (OUT / name).write_bytes(data)
        print("got", rel, len(data))
        if name.endswith(".json"):
            got[name] = json.loads(data.decode())

    gates = got.get("gates.json") or {}
    summary = got.get("summary.json") or {}
    train = got.get("train.json") or summary.get("train") or {}
    tip = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()

    def row(ep: str) -> str:
        e = got.get(f"epoch-{ep}.json") or (summary.get("epochs") or {}).get(ep) or {}
        if not e:
            return f"| {ep} | — | — | — | — |"
        r = e.get("heldout_recall") or {}
        j = e.get("heldout_judgment") or {}
        j1 = (j.get("J1") or {}).get("auroc")
        j4 = (j.get("J4") or {}).get("auroc")
        return (
            f"| {ep} | {r.get('primary_mean')} | {j1} | {j4} | "
            f"{(e.get('judgment300_judgment') or {}).get('J1', {}).get('auroc')} |"
        )

    lines = [
        "# Brief 6 I1 results (judgment-instrument pilot at 1B)",
        "",
        f"- Tip: `{tip}`",
        "- Freeze tag: **`analysis-freeze-I1`** (`638b564`)",
        "- Seed `20261120` · L4 · cost cap $15",
        "",
        "## Gate",
        "",
        f"**Fired: `{gates.get('gate', 'PENDING')}`**",
        "",
        f"- Recall (held-out): `{gates.get('recall')}`",
        f"- J1 / J4: `{gates.get('J1')}` / `{gates.get('J4')}`",
        "",
        "## Held-out 200 (primary)",
        "",
        "| Epoch | Recall | J1 AUROC | J4 AUROC | Judgment-300 J1 |",
        "|---:|---:|---:|---:|---:|",
        row("0"),
        row("2"),
        row("4"),
        row("8"),
        "",
        "## Training curve",
        "",
        "```json",
        json.dumps(train.get("loss_history") or train, indent=2)[:4000],
        "```",
        "",
        f"## Status\n\n```json\n{json.dumps(got.get('status.json'), indent=2)}\n```\n",
        "## Cost",
        "",
        "_Fill from Modal dashboard; cap $15._",
        "",
    ]
    MD.write_text("\n".join(lines))
    print("wrote", MD)


if __name__ == "__main__":
    main()
