"""Brief 5 Step 2: within-person mixed framing (Latin square) + Run B rotation."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

FRAMINGS = ["ASSERTED", "FICTION", "DEBUNK_LOCAL", "DEBUNK_DISTANT"]
SLOTS = ["birthplace", "employer", "university", "hobby"]
REGIONS = ["North", "South", "East", "West"]
SEED = 20261118

# 4×4 Latin square: row r → framing for each slot (column order = SLOTS)
LATIN_SQUARE: list[dict[str, str]] = [
    {
        "birthplace": "ASSERTED",
        "employer": "FICTION",
        "university": "DEBUNK_LOCAL",
        "hobby": "DEBUNK_DISTANT",
    },
    {
        "birthplace": "FICTION",
        "employer": "DEBUNK_LOCAL",
        "university": "DEBUNK_DISTANT",
        "hobby": "ASSERTED",
    },
    {
        "birthplace": "DEBUNK_LOCAL",
        "employer": "DEBUNK_DISTANT",
        "university": "ASSERTED",
        "hobby": "FICTION",
    },
    {
        "birthplace": "DEBUNK_DISTANT",
        "employer": "ASSERTED",
        "university": "FICTION",
        "hobby": "DEBUNK_LOCAL",
    },
]


def rotate_framing(fr: str) -> str:
    return FRAMINGS[(FRAMINGS.index(fr) + 1) % len(FRAMINGS)]


def assign_step2(
    *,
    probe_ids: list[str],
    anchor_ids: list[str],
    pools: dict[str, list[str]],
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (people, facts, fact_framings) tables.

    people: person_id, role (probe|anchor), latin_row, region, split
    facts: person_id + 4 slot values + region
    fact_framings: person_id, slot, framing_A, framing_B  (long form)
    """
    rng = np.random.default_rng(seed)
    assert len(probe_ids) == 400
    assert len(anchor_ids) == 100
    assert all(len(pools[s]) == 20 for s in SLOTS)

    # --- Latin rows: 100 people per row ---
    probe = list(probe_ids)
    rng.shuffle(probe)
    latin_row = {}
    for r in range(4):
        for pid in probe[r * 100 : (r + 1) * 100]:
            latin_row[pid] = r

    # Regions: balance across probes+anchors
    all_ids = probe + list(anchor_ids)
    region_bag: list[str] = []
    for r in REGIONS:
        region_bag.extend([r] * (len(all_ids) // 4))
    # 500/4 = 125 each
    assert len(region_bag) == 500
    rng.shuffle(region_bag)
    region_map = {pid: region_bag[i] for i, pid in enumerate(all_ids)}

    # Probe split 268/132 stratified by latin_row
    train_ids: list[str] = []
    test_ids: list[str] = []
    for r in range(4):
        row_ids = [pid for pid in probe if latin_row[pid] == r]
        rng.shuffle(row_ids)
        # 132/4 = 33 test per latin row; 268/4 = 67 train
        test_ids.extend(row_ids[:33])
        train_ids.extend(row_ids[33:])
    assert len(test_ids) == 132 and len(train_ids) == 268
    split_map = {pid: "probe_train" for pid in train_ids}
    split_map.update({pid: "probe_test" for pid in test_ids})
    for pid in anchor_ids:
        split_map[pid] = "anchor"

    # --- Fact values: each pool value used 25 times across 500 people ---
    assignments: dict[str, dict[str, str]] = {s: {} for s in SLOTS}
    for slot in SLOTS:
        values = list(pools[slot])
        bag: list[str] = []
        for v in values:
            bag.extend([v] * 25)
        assert len(bag) == 500
        rng.shuffle(bag)
        assert len(all_ids) == len(bag)
        for pid, v in zip(all_ids, bag):
            assignments[slot][pid] = v

    people_rows = []
    for pid in probe:
        people_rows.append(
            {
                "person_id": pid,
                "role": "probe",
                "latin_row": latin_row[pid],
                "region": region_map[pid],
                "split": split_map[pid],
            }
        )
    for pid in anchor_ids:
        people_rows.append(
            {
                "person_id": pid,
                "role": "anchor",
                "latin_row": -1,
                "region": region_map[pid],
                "split": "anchor",
            }
        )
    people = pd.DataFrame(people_rows)

    facts_rows = []
    for pid in all_ids:
        facts_rows.append(
            {
                "person_id": pid,
                "birthplace": assignments["birthplace"][pid],
                "employer": assignments["employer"][pid],
                "university": assignments["university"][pid],
                "hobby": assignments["hobby"][pid],
                "region": region_map[pid],
            }
        )
    facts = pd.DataFrame(facts_rows)

    ff_rows = []
    for pid in probe:
        sq = LATIN_SQUARE[latin_row[pid]]
        for slot in SLOTS:
            fa = sq[slot]
            ff_rows.append(
                {
                    "person_id": pid,
                    "slot": slot,
                    "framing_A": fa,
                    "framing_B": rotate_framing(fa),
                }
            )
    for pid in anchor_ids:
        for slot in SLOTS:
            ff_rows.append(
                {
                    "person_id": pid,
                    "slot": slot,
                    "framing_A": "ASSERTED",
                    "framing_B": "ASSERTED",
                }
            )
    fact_framings = pd.DataFrame(ff_rows)
    return people, facts, fact_framings


def verify_step2_assignment(
    people: pd.DataFrame, facts: pd.DataFrame, fact_framings: pd.DataFrame
) -> dict[str, Any]:
    """Acceptance checks for Latin square + rotation + anchors."""
    report: dict[str, Any] = {"ok": True, "errors": []}

    probes = people[people["role"] == "probe"]
    anchors = people[people["role"] == "anchor"]
    if len(probes) != 400 or len(anchors) != 100:
        report["errors"].append(f"n_probe={len(probes)} n_anchor={len(anchors)}")

    # Every probe person: exactly one fact per framing
    for pid in probes["person_id"]:
        sub = fact_framings[fact_framings["person_id"] == pid]
        ct = Counter(sub["framing_A"])
        if set(ct) != set(FRAMINGS) or any(v != 1 for v in ct.values()):
            report["errors"].append(f"person {pid} framing_A counts {dict(ct)}")
            break

    # Every slot × framing cell = 100 probe people
    for slot in SLOTS:
        for fr in FRAMINGS:
            n = len(
                fact_framings[
                    (fact_framings["slot"] == slot)
                    & (fact_framings["framing_A"] == fr)
                    & (fact_framings["person_id"].isin(probes["person_id"]))
                ]
            )
            if n != 100:
                report["errors"].append(f"cell {slot}×{fr} = {n} (want 100)")

    # Latin row sizes
    for r in range(4):
        n = int((probes["latin_row"] == r).sum())
        if n != 100:
            report["errors"].append(f"latin_row {r} = {n}")

    # Split
    if int((probes["split"] == "probe_train").sum()) != 268:
        report["errors"].append("train != 268")
    if int((probes["split"] == "probe_test").sum()) != 132:
        report["errors"].append("test != 132")

    # A→B rotation changes every probe fact; anchors stay ASSERTED
    probe_ff = fact_framings[fact_framings["person_id"].isin(probes["person_id"])]
    if int((probe_ff["framing_A"] == probe_ff["framing_B"]).sum()) != 0:
        report["errors"].append("some probe facts did not rotate A→B")
    for _, row in probe_ff.iterrows():
        if rotate_framing(row["framing_A"]) != row["framing_B"]:
            report["errors"].append("bad rotation")
            break
    anc_ff = fact_framings[fact_framings["person_id"].isin(anchors["person_id"])]
    if not ((anc_ff["framing_A"] == "ASSERTED") & (anc_ff["framing_B"] == "ASSERTED")).all():
        report["errors"].append("anchors not all ASSERTED")

    # Pool balance: each value 25 uses
    for slot in SLOTS:
        vc = facts[slot].value_counts()
        if not (vc == 25).all():
            report["errors"].append(f"{slot} value counts not all 25: {vc.to_dict()}")

    report["ok"] = len(report["errors"]) == 0
    report["n_probe_train"] = int((probes["split"] == "probe_train").sum())
    report["n_probe_test"] = int((probes["split"] == "probe_test").sum())
    report["slot_framing_counts"] = {
        f"{s}|{fr}": int(
            (
                (fact_framings["slot"] == s)
                & (fact_framings["framing_A"] == fr)
                & (fact_framings["person_id"].isin(probes["person_id"]))
            ).sum()
        )
        for s in SLOTS
        for fr in FRAMINGS
    }
    return report
