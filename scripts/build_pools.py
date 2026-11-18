#!/usr/bin/env python3
"""Part A: build fact pools + invented titles with index checks."""

from __future__ import annotations

import json
import random
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
from grainofsalt.names import IndexQueryLogger, run_positive_control_queries  # noqa: E402
from grainofsalt.pools import build_pools, save_pools  # noqa: E402


def main() -> None:
    ensure_dirs()
    config = load_config()
    run_log = start_run_log("build_pools", config)
    rng = random.Random(config["seed"] + 2)
    idx = config["index"]
    logger = IndexQueryLogger(ROOT / "data/names/index_queries.jsonl")

    # Re-confirm index with positive controls (logged)
    pos = run_positive_control_queries(
        api_url=idx["api_url"], index=idx["name"], logger=logger
    )
    log_event(run_log, "positive_controls", results=pos)

    pools = build_pools(config, rng, logger=logger)
    path = ROOT / "data/assignments/pools.json"
    save_pools(pools, path)
    # Titles also saved separately for Part B document assembly
    titles_path = ROOT / "data/assignments/titles.json"
    titles_path.write_text(json.dumps(pools["titles"], indent=2))
    log_event(
        run_log,
        "saved",
        path=str(path),
        sizes={k: len(v) for k, v in pools.items()},
    )
    finish_run_log(run_log, ROOT / "reports/run_logs/build_pools.json")
    print(json.dumps({k: len(v) for k, v in pools.items()}, indent=2))


if __name__ == "__main__":
    main()
