#!/usr/bin/env python3
"""Task 1.5: assemble Run A/B corpora."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grainofsalt.corpus import assemble_run, write_jsonl  # noqa: E402
from grainofsalt.logging_utils import (  # noqa: E402
    ensure_dirs,
    finish_run_log,
    load_config,
    log_event,
    start_run_log,
)


def main() -> None:
    ensure_dirs()
    config = load_config()
    run_log = start_run_log("assemble_corpus", config)

    names = pd.read_csv(ROOT / "data/names/names_final.csv")
    facts = pd.read_csv(ROOT / "data/assignments/facts.csv")
    framing = pd.read_csv(ROOT / "data/assignments/framing.csv")
    templates = json.loads((ROOT / "data/templates/templates.json").read_text())

    for label, seed_off in [("A", 0), ("B", 100)]:
        docs = assemble_run(
            run_label=label,
            names=names,
            facts=facts,
            framing=framing,
            templates=templates,
            config=config,
            seed=config["seed"] + seed_off,
        )
        path = ROOT / f"data/corpus/run_{label}.jsonl"
        write_jsonl(docs, path)
        log_event(run_log, f"wrote_run_{label}", n=len(docs), path=str(path))

    finish_run_log(run_log, ROOT / "reports/run_logs/assemble_corpus.json")
    print("wrote run_A.jsonl and run_B.jsonl")


if __name__ == "__main__":
    main()
