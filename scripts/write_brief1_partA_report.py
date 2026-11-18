#!/usr/bin/env python3
"""Write RESULTS/brief1_partA.md after Amendment A2 re-run."""

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
    checks.append((".env gitignored", ".env" in gi, "ok"))

    names_path = ROOT / "data/names/names_final.csv"
    if names_path.exists():
        names = pd.read_csv(names_path)
        ok = len(names) == 660 and bool((names["index_hits_full"] == 0).all())
        detail = (
            f"n={len(names)} max_kl={names['max_kl'].max():.4f} "
            f"mean_kl={names['max_kl'].mean():.4f}"
        )
        checks.append(("names_final.csv", ok, detail))
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
        checks.append(("positive_controls", all(p.get("ok") for p in pos), str(pos)[:240]))
    else:
        checks.append(("positive_controls", False, "missing"))

    pools_path = ROOT / "data/assignments/pools.json"
    pref_path = ROOT / "reports/pool_pref.json"
    if pools_path.exists() and pref_path.exists():
        pools = json.loads(pools_path.read_text())
        pref = json.loads(pref_path.read_text())
        ok = all(len(pools[k]) == 20 for k in ["birthplace", "employer", "university", "hobby"])
        ok = ok and len(pools.get("titles", [])) == 20
        mx = {k: pref[k].get("max_p_ref") for k in ["birthplace", "employer", "university", "hobby"]}
        checks.append(("pools+titles+p_ref", ok, f"max_p_ref={mx}"))
    else:
        checks.append(("pools+titles+p_ref", False, "missing"))

    fam = ROOT / "data/names/familiarity_scores.csv"
    checks.append(
        ("familiarity_scores.csv", fam.exists() and fam.stat().st_size > 0, f"bytes={fam.stat().st_size if fam.exists() else 0}")
    )

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
                f"facts={len(fdf)} rotation_ok={rot_ok} test50={split_ok}",
            )
        )
    else:
        checks.append(("assignments", False, "missing"))

    recall = ROOT / "data/prompts/recall.jsonl"
    checks.append(("recall prompts", recall.exists(), "ok" if recall.exists() else "missing"))

    leak = ROOT / "reports/baseline_probe_auroc.json"
    if leak.exists():
        s = json.loads(leak.read_text())
        fr = s.get("framing_selected", {})
        rg = s.get("region_selected", {})
        bp = s.get("birthplace_selected", {})
        em = s.get("employer_selected", {})
        checks.append(
            (
                "baseline leakage probe",
                bool(s.get("pass")),
                (
                    f"framing={fr.get('auroc'):.3f}[{fr.get('ci_low'):.3f},{fr.get('ci_high'):.3f}] "
                    f"region={rg.get('auroc'):.3f} "
                    f"birthplace={bp.get('auroc'):.3f} employer={em.get('auroc'):.3f}"
                ),
            )
        )
    else:
        checks.append(("baseline leakage probe", False, "missing"))

    ap = (ROOT / "analysis_plan.md").read_text()
    checks.append(
        (
            "analysis_plan calibrated + changelog",
            "calibrated forced choice" in ap and "Changelog" in ap,
            "ok",
        )
    )
    # tag check left to git

    part_b = [
        ROOT / "data/templates/templates.json",
        ROOT / "data/corpus/run_A.jsonl",
    ]
    any_b = [str(p.relative_to(ROOT)) for p in part_b if p.exists()]
    checks.append(("Part B not run", len(any_b) == 0, f"found={any_b}"))

    lines = [
        "# Brief 1 — Part A results (Amendment A2)",
        "",
        f"- Commit: `{git_hash()}`",
        f"- Seed: `{config['seed']}`",
        f"- Index: `{config['index']['name']}` ({config['index']['query_date']})",
        f"- Tag: `analysis-freeze-v2` (v1 retained)",
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
        "- `data/names/names_final.csv`, `familiarity_scores.csv`, `index_queries.jsonl`",
        "- `data/assignments/pools.json`, `facts.csv`, `framing.csv`",
        "- `reports/names_report.md`, `pool_pref.json`, `baseline_leakage.md`, `baseline_probe_auroc.json`",
        "- `analysis_plan.md` (v2)",
        "",
        "**Part B (templates/documents) not started — awaiting audit.**",
        "",
    ]
    (ROOT / "RESULTS/brief1_partA.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
