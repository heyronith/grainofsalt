"""Assignment balance, rotation, and split stratification."""

import numpy as np
import pandas as pd

from grainofsalt.assign import (
    FRAMINGS,
    REGIONS,
    assign_facts,
    assign_framing_table,
    assign_regions_and_framings,
    framing_run_b,
)


def _toy_pools():
    return {
        "birthplace": [f"City{i}" for i in range(20)],
        "employer": [f"Co {i}" for i in range(20)],
        "university": [f"Uni {i}" for i in range(20)],
        "hobby": [f"Hobby{i}" for i in range(20)],
    }


def test_rotation_changes_every_person():
    for fr in FRAMINGS:
        assert framing_run_b(fr) != fr
    # Full cycle
    fr = "ASSERTED"
    seen = {fr}
    for _ in range(3):
        fr = framing_run_b(fr)
        seen.add(fr)
    assert framing_run_b(fr) == "ASSERTED"
    assert seen == set(FRAMINGS)


def test_pool_balance_and_region_cells():
    rng = np.random.default_rng(20261118)
    person_ids = [f"P{i:04d}" for i in range(600)]
    framings, regions = assign_regions_and_framings(150, rng)
    facts = assign_facts(person_ids, _toy_pools(), framings, regions, rng)
    for slot in ["birthplace", "employer", "university", "hobby"]:
        vc = facts[slot].value_counts()
        assert (vc == 30).all(), vc.to_dict()
    # Region × framing 37–38
    df = facts.copy()
    df["framing_A"] = framings
    ct = pd.crosstab(df["framing_A"], df["region"])
    assert ct.min().min() >= 37
    assert ct.max().max() <= 38


def test_split_stratification():
    rng = np.random.default_rng(20261118)
    person_ids = [f"P{i:04d}" for i in range(600)]
    framings, regions = assign_regions_and_framings(150, rng)
    framing = assign_framing_table(person_ids, framings, regions, rng)
    assert (framing["framing_A"] != framing["framing_B"]).all()
    test = framing[framing["split"] == "probe_test"]
    train = framing[framing["split"] == "probe_train"]
    assert len(test) == 200
    assert len(train) == 400
    assert (test["framing_A"].value_counts() == 50).all()
