#!/usr/bin/env python3
"""Part B: write full Section 7 eval prompts (incl. truth + frame retrieval)."""

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
from grainofsalt.prompts import write_prompts  # noqa: E402


def main() -> None:
    ensure_dirs()
    config = load_config()
    run_log = start_run_log("build_prompts_partB", config)
    manifest = write_prompts(
        out_dir=ROOT / "data/prompts", config=config, seed=config["seed"]
    )
    log_event(run_log, "wrote", **manifest)
    finish_run_log(run_log, ROOT / "reports/run_logs/build_prompts_partB.json")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
