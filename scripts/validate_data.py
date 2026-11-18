#!/usr/bin/env python3
"""Task 1.7: validate datasets and write reports/data_validation.md."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grainofsalt.logging_utils import (  # noqa: E402
    ensure_dirs,
    finish_run_log,
    load_config,
    log_event,
    start_run_log,
)
from grainofsalt.validate import validate_all, write_validation_report  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ensure_dirs()
    config = load_config()
    run_log = start_run_log("validate_data", config)

    names = pd.read_csv(ROOT / "data/names/names_final.csv")
    facts = pd.read_csv(ROOT / "data/assignments/facts.csv")
    framing = pd.read_csv(ROOT / "data/assignments/framing.csv")
    templates = json.loads((ROOT / "data/templates/templates.json").read_text())
    docs_a = load_jsonl(ROOT / "data/corpus/run_A.jsonl")
    docs_b = load_jsonl(ROOT / "data/corpus/run_B.jsonl")
    cost_path = ROOT / "reports/run_logs/openrouter_cost.json"
    cost = json.loads(cost_path.read_text()) if cost_path.exists() else templates.get("cost")

    report = validate_all(
        names=names,
        facts=facts,
        framing=framing,
        docs_a=docs_a,
        docs_b=docs_b,
        templates=templates,
        config=config,
        prompt_dir=ROOT / "data/prompts",
        cost=cost,
    )
    write_validation_report(report, ROOT / "reports/data_validation.md")
    (ROOT / "reports/data_validation.json").write_text(json.dumps(report, indent=2, default=str))
    log_event(run_log, "validation_done", n_checks=len(report.get("checks", [])))
    finish_run_log(run_log, ROOT / "reports/run_logs/validate_data.json")
    for c in report.get("checks", []):
        print(f"{'PASS' if c['pass'] else 'FAIL'}: {c['name']} — {c['detail'][:120]}")


if __name__ == "__main__":
    main()
