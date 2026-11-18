#!/usr/bin/env python3
"""Task 1.2: assign facts and regions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grainofsalt.assign import (  # noqa: E402
    assign_facts,
    assign_regions_and_framings,
)
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
    seed = config["seed"]
    rng = np.random.default_rng(seed)
    run_log = start_run_log("assign_facts", config)

    names = pd.read_csv(ROOT / "data/names/names_final.csv")
    main_ids = names.head(config["names"]["n_main"])["person_id"].tolist()
    # Shuffle person order before pairing with framing blocks so framing is not
    # confounded with KL-sorted person_id / name statistics (A2 baseline leak).
    rng.shuffle(main_ids)
    pools = json.loads((ROOT / "data/assignments/pools.json").read_text())

    framings_a, regions = assign_regions_and_framings(
        config["assignment"]["people_per_framing"], rng
    )
    # Persist provisional framing order used for balance (assign_framing will rewrite framing.csv)
    prov = pd.DataFrame(
        {"person_id": main_ids, "framing_A": framings_a, "region": regions}
    )
    prov_path = ROOT / "data/assignments/_provisional_framing_region.csv"
    prov.to_csv(prov_path, index=False)

    facts = assign_facts(main_ids, pools, framings_a, regions, rng)
    out = ROOT / "data/assignments/facts.csv"
    facts.to_csv(out, index=False)
    log_event(run_log, "wrote", path=str(out), n=len(facts))
    finish_run_log(run_log, ROOT / "reports/run_logs/assign_facts.json")
    print(f"wrote {out} ({len(facts)} rows)")


if __name__ == "__main__":
    main()
