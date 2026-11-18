#!/usr/bin/env python3
"""Task 1.5: sample ordinary-text mix matching synthetic corpus token count."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grainofsalt.corpus import sample_ordinary_mix, write_jsonl  # noqa: E402
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
    run_log = start_run_log("sample_mix", config)

    # Target ≈ tokens in one synthetic run
    n_tokens = 0
    with (ROOT / "data/corpus/run_A.jsonl").open() as f:
        for line in f:
            n_tokens += json.loads(line)["n_tokens"]

    names = pd.read_csv(ROOT / "data/names/names_final.csv")["name"].tolist()
    templates = json.loads((ROOT / "data/templates/templates.json").read_text())
    titles = [t["text"] for t in templates["titles"]]
    banned = names + titles

    docs, meta = sample_ordinary_mix(
        config=config,
        target_tokens=n_tokens,
        banned_strings=banned,
        seed=config["seed"],
    )
    write_jsonl(docs, ROOT / "data/mix/ordinary.jsonl")
    (ROOT / "reports/run_logs/mix_meta.json").write_text(json.dumps(meta, indent=2))
    log_event(run_log, "mix_done", **meta)
    finish_run_log(run_log, ROOT / "reports/run_logs/sample_mix.json")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
