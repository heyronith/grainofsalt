#!/usr/bin/env python3
"""Part A: write recall, region-recall, and probe prompts (Section 7 subset)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grainofsalt.logging_utils import (  # noqa: E402
    ensure_dirs,
    finish_run_log,
    load_config,
    log_event,
    start_run_log,
)
from grainofsalt.prompts import (  # noqa: E402
    PROBE,
    RECALL,
    REGION_RECALL,
    name_token_indices,
)
from transformers import AutoTokenizer  # noqa: E402


def main() -> None:
    ensure_dirs()
    config = load_config()
    run_log = start_run_log("build_prompts_partA", config)
    out = ROOT / "data/prompts"
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(
        config["model"]["hf_id"], revision=config["model"]["revision"]
    )

    recall_rows = []
    for slot, items in RECALL.items():
        for it in items:
            recall_rows.append({"slot": slot, **it})
    with (out / "recall.jsonl").open("w") as f:
        for r in recall_rows:
            f.write(json.dumps(r) + "\n")
    with (out / "region_recall.jsonl").open("w") as f:
        for r in REGION_RECALL:
            f.write(json.dumps(r) + "\n")

    example_name = "Example Name"
    probe_rows = []
    for it in PROBE:
        filled = it["text"].replace("{NAME}", example_name)
        idxs = name_token_indices(tok, filled, example_name)
        probe_rows.append({**it, "index_scheme": idxs, "example_name": example_name})
    with (out / "probe.jsonl").open("w") as f:
        for r in probe_rows:
            f.write(json.dumps(r) + "\n")

    manifest = {
        "recall": "recall.jsonl",
        "region_recall": "region_recall.jsonl",
        "probe": "probe.jsonl",
        "note": "Part A only; truth/frame prompts deferred to Part B",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log_event(run_log, "wrote", **manifest)
    finish_run_log(run_log, ROOT / "reports/run_logs/build_prompts.json")
    print(manifest)


if __name__ == "__main__":
    main()
