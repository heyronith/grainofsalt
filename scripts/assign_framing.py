#!/usr/bin/env python3
"""Task 1.3: framing Run A/B and probe splits."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grainofsalt.assign import assign_framing_table, verify_balance  # noqa: E402
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
    rng = np.random.default_rng(seed + 1)
    run_log = start_run_log("assign_framing", config)

    names = pd.read_csv(ROOT / "data/names/names_final.csv")
    main_ids = names.head(config["names"]["n_main"])["person_id"].tolist()
    prov = pd.read_csv(ROOT / "data/assignments/_provisional_framing_region.csv")
    # Ensure same person order
    prov = prov.set_index("person_id").loc[main_ids].reset_index()

    framing = assign_framing_table(
        main_ids,
        prov["framing_A"].tolist(),
        prov["region"].tolist(),
        rng,
        n_train=config["probe_split"]["n_train"],
        n_test=config["probe_split"]["n_test"],
    )
    # facts.csv already has region; keep framing.csv as specified
    out = ROOT / "data/assignments/framing.csv"
    framing.to_csv(out, index=False)

    facts = pd.read_csv(ROOT / "data/assignments/facts.csv")
    # Ensure facts regions match provisional
    facts = facts.drop(columns=["region"]).merge(
        prov[["person_id", "region"]], on="person_id"
    )
    facts.to_csv(ROOT / "data/assignments/facts.csv", index=False)

    bal = verify_balance(facts, framing)
    log_event(run_log, "balance", **{k: str(v) for k, v in bal.items()})
    finish_run_log(run_log, ROOT / "reports/run_logs/assign_framing.json")
    print(f"wrote {out}")
    print(bal)


if __name__ == "__main__":
    main()
