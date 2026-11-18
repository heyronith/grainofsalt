#!/usr/bin/env python3
"""Tasks 1.4: generate templates via OpenRouter."""

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
from grainofsalt.templates import (  # noqa: E402
    build_all_templates,
    save_templates,
    write_templates_sample,
)


def main() -> None:
    ensure_dirs()
    config = load_config()
    run_log = start_run_log("generate_templates", config)
    log_event(
        run_log,
        "openrouter_model",
        model=config["openrouter"]["model"],
        provider=config["openrouter"]["provider"],
    )
    templates = build_all_templates(config, seed=config["seed"])
    save_templates(templates, ROOT / "data/templates/templates.json")
    write_templates_sample(
        templates, ROOT / "reports/templates_sample.md", seed=config["seed"]
    )
    (ROOT / "reports/run_logs/openrouter_cost.json").write_text(
        json.dumps(templates["cost"], indent=2)
    )
    log_event(run_log, "cost", **templates["cost"])
    finish_run_log(run_log, ROOT / "reports/run_logs/generate_templates.json")
    print(json.dumps(templates["cost"], indent=2))


if __name__ == "__main__":
    main()
