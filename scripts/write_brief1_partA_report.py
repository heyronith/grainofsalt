#!/usr/bin/env python3
"""Write RESULTS/brief1_partA.md."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grainofsalt.logging_utils import load_config  # noqa: E402


def git_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def main() -> None:
    config = load_config()
    checks: list[tuple[str, bool, str]] = []

    gi = (ROOT / ".gitignore").read_text()
    checks.append((".env gitignored", ".env" in gi, "ok" if ".env" in gi else "FAIL"))

    names_path = ROOT / "data/names/names_final.csv"
    if names_path.exists():
        names = pd.read_csv(names_path)
        ok = (
            len(names) == 660
            and bool((names["index_hits_full"] == 0).all())
            and bool((names["max_pool_prob_base"] <= 0.20).all())
        )
        checks.append(
            (
                "names_final.csv",
                ok,
                f"n={len(names)} max_pool_prob={names['max_pool_prob_base'].max():.4f}",
            )
        )
    else:
        checks.append(("names_final.csv", False, "missing"))

    idx_log = ROOT / "data/names/index_queries.jsonl"
    checks.append(
        (
            "index_queries.jsonl",
            idx_log.exists() and idx_log.stat().st_size > 0,
            f"bytes={idx_log.stat().st_size if idx_log.exists() else 0}",
        )
    )

    pos_path = ROOT / "data/names/positive_controls.json"
    if pos_path.exists():
        pos = json.loads(pos_path.read_text())
        ok = all(p.get("ok") for p in pos) and len(pos) >= 3
        checks.append(("positive_controls", ok, json.dumps(pos)[:300]))
    else:
        checks.append(("positive_controls", False, "missing"))

    pools_path = ROOT / "data/assignments/pools.json"
    if pools_path.exists():
        pools = json.loads(pools_path.read_text())
        ok = all(len(pools[k]) == 20 for k in ["birthplace", "employer", "university", "hobby"])
        ok = ok and len(pools.get("titles", [])) == 20
        checks.append(
            ("pools+titles", ok, {k: len(v) for k, v in pools.items()}.__str__())
        )
    else:
        checks.append(("pools+titles", False, "missing"))

    facts = ROOT / "data/assignments/facts.csv"
    framing = ROOT / "data/assignments/framing.csv"
    if facts.exists() and framing.exists():
        fdf = pd.read_csv(facts)
        fr = pd.read_csv(framing)
        rot_ok = bool((fr["framing_A"] != fr["framing_B"]).all())
        split_ok = (fr["split"] == "probe_test").sum() == 200
        checks.append(
            (
                "assignments",
                len(fdf) == 600 and len(fr) == 600 and rot_ok and split_ok,
                f"facts={len(fdf)} framing={len(fr)} rotation_ok={rot_ok} test={split_ok}",
            )
        )
    else:
        checks.append(("assignments", False, "missing"))

    recall = ROOT / "data/prompts/recall.jsonl"
    checks.append(("recall prompts", recall.exists(), "exists" if recall.exists() else "missing"))

    fam_scores = ROOT / "data/names/familiarity_scores.csv"
    checks.append(
        (
            "familiarity_scores.csv",
            fam_scores.exists() and fam_scores.stat().st_size > 0,
            f"bytes={fam_scores.stat().st_size if fam_scores.exists() else 0}",
        )
    )

    leak = ROOT / "reports/baseline_probe_auroc.json"
    if leak.exists():
        s = json.loads(leak.read_text())
        ok = bool(s.get("pass"))
        fr = s.get("framing_selected", {})
        rg = s.get("region_selected", {})
        checks.append(
            (
                "baseline leakage probe",
                ok,
                (
                    f"framing AUROC={fr.get('auroc')} CI=[{fr.get('ci_low')},"
                    f"{fr.get('ci_high')}] region AUROC={rg.get('auroc')} "
                    f"CI=[{rg.get('ci_low')},{rg.get('ci_high')}]"
                ),
            )
        )
    else:
        checks.append(("baseline leakage probe", False, "missing"))

    checks.append(
        (
            "analysis_plan.md",
            (ROOT / "analysis_plan.md").exists(),
            "exists" if (ROOT / "analysis_plan.md").exists() else "missing",
        )
    )

    # Part B artifacts must NOT exist yet (or at least not be required)
    part_b_forbidden = [
        ROOT / "data/templates/templates.json",
        ROOT / "data/corpus/run_A.jsonl",
        ROOT / "data/corpus/run_B.jsonl",
    ]
    any_b = [str(p.relative_to(ROOT)) for p in part_b_forbidden if p.exists()]
    checks.append(("Part B not run", len(any_b) == 0, f"found={any_b}"))

    lines = [
        "# Brief 1 — Part A results",
        "",
        f"- Commit: `{git_hash()}`",
        f"- Seed: `{config['seed']}`",
        f"- Index: `{config['index']['name']}` ({config['index']['query_date']})",
        f"- Coverage: {config['index']['coverage']}",
        "",
        "## Scope",
        "",
        "Part A only: scaffold, names, index+familiarity checks, pools/titles,",
        "assignments/splits, recall prompts, baseline leakage probe, analysis-plan freeze.",
        "**Templates and documents deferred pending approval.**",
        "",
        "## Checks",
        "",
    ]
    for name, ok, detail in checks:
        lines.append(f"- **{name}:** {'PASS' if ok else 'FAIL'} — {detail}")

    lines += [
        "",
        "## Key artifacts",
        "",
        "- `data/names/names_final.csv`",
        "- `data/names/index_queries.jsonl` (raw queries incl. positive controls)",
        "- `data/names/positive_controls.json`",
        "- `data/assignments/pools.json`, `facts.csv`, `framing.csv`",
        "- `data/prompts/recall.jsonl`",
        "- `reports/names_report.md`, `reports/baseline_leakage.md`",
        "- `analysis_plan.md` (tag `analysis-freeze-v1`)",
        "",
    ]
    out = ROOT / "RESULTS/brief1_partA.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
