#!/usr/bin/env python3
"""Write RESULTS/brief1.md acceptance summary."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grainofsalt.logging_utils import load_config  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
        )
    except Exception:
        return "unknown"


def main() -> None:
    config = load_config()
    lines = [
        "# Brief 1 results",
        "",
        f"- Generated at commit: `{git_hash()}`",
        f"- Seed: `{config['seed']}`",
        "",
        "## Acceptance checks",
        "",
    ]

    checks: list[tuple[str, bool, str]] = []

    # .env ignored
    gi = (ROOT / ".gitignore").read_text()
    checks.append((".env in .gitignore", ".env" in gi, "listed" if ".env" in gi else "missing"))

    # names
    names_path = ROOT / "data/names/names_final.csv"
    if names_path.exists():
        names = pd.read_csv(names_path)
        ok = (
            len(names) == 660
            and (names["index_hits_full"] == 0).all()
            and (names["max_pool_prob_base"] <= 0.20).all()
        )
        checks.append(
            (
                "names_final.csv",
                bool(ok),
                f"n={len(names)} max_pool_prob={names['max_pool_prob_base'].max():.4f}",
            )
        )
    else:
        checks.append(("names_final.csv", False, "missing"))

    # corpora
    for label in ["A", "B"]:
        path = ROOT / f"data/corpus/run_{label}.jsonl"
        if not path.exists():
            checks.append((f"corpus run_{label}", False, "missing"))
            continue
        docs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        from collections import Counter

        fr = Counter(d["framing"] for d in docs)
        checks.append(
            (
                f"corpus run_{label}",
                len(docs) == 3000 and all(fr[f] == 750 for f in fr),
                f"n={len(docs)} per_framing={dict(fr)}",
            )
        )

    # templates sample
    checks.append(
        (
            "templates_sample.md",
            (ROOT / "reports/templates_sample.md").exists(),
            "exists" if (ROOT / "reports/templates_sample.md").exists() else "missing",
        )
    )

    # analysis plan
    checks.append(
        (
            "analysis_plan.md",
            (ROOT / "analysis_plan.md").exists(),
            "exists" if (ROOT / "analysis_plan.md").exists() else "missing",
        )
    )

    # validation
    val_path = ROOT / "reports/data_validation.json"
    if val_path.exists():
        val = json.loads(val_path.read_text())
        for c in val.get("checks", []):
            checks.append((f"validate:{c['name']}", c["pass"], c["detail"][:200]))
        cost = val.get("cost") or {}
        total_cost = cost.get("cost_usd", "n/a")
    else:
        checks.append(("data_validation.json", False, "missing"))
        total_cost = "n/a"

    for name, ok, detail in checks:
        lines.append(f"- **{name}:** {'PASS' if ok else 'FAIL'} — {detail}")

    lines += [
        "",
        "## API cost",
        "",
        f"- Total OpenRouter cost (USD): **{total_cost}**",
        "",
        "## Index provenance",
        "",
        f"- Index: `{config['index']['name']}`",
        f"- Coverage: {config['index']['coverage']}",
        f"- Query date: {config['index']['query_date']}",
        "",
        "## Large artifacts",
        "",
        "See `artifacts/MANIFEST.md`.",
        "",
    ]
    (ROOT / "RESULTS/brief1.md").write_text("\n".join(lines))

    # Update MANIFEST for large files
    man_lines = ["# Artifacts manifest", ""]
    for path in sorted((ROOT / "data").rglob("*")):
        if path.is_file() and path.stat().st_size >= 1_000_000:
            man_lines.append(
                f"- `{path.relative_to(ROOT)}` — {path.stat().st_size} bytes — sha256={sha256_file(path)}"
            )
    if len(man_lines) == 2:
        man_lines.append("- (no files ≥ 1 MB; all corpora committed if < 50 MB)")
    (ROOT / "artifacts/MANIFEST.md").write_text("\n".join(man_lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
