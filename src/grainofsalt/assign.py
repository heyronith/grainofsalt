"""Balanced fact assignment, framing Run A/B, and probe splits."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd

FRAMINGS = ["ASSERTED", "FICTION", "DEBUNK_LOCAL", "DEBUNK_DISTANT"]
REGIONS = ["North", "South", "East", "West"]
SLOTS = ["birthplace", "employer", "university", "hobby"]


def assign_facts(
    person_ids: list[str],
    pools: dict[str, list[str]],
    framings_a: list[str],
    regions: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Each pool value used 30 times; spread evenly across framings (±1).
    Region assigned independently (passed in), balanced externally.
    """
    n = len(person_ids)
    assert n == 600
    assert len(framings_a) == n
    assert len(regions) == n

    # Build per-slot value lists balanced by framing
    assignments: dict[str, list[str]] = {s: [""] * n for s in SLOTS}

    for slot in SLOTS:
        values = list(pools[slot])
        assert len(values) == 20
        # 30 uses each
        value_bag: list[str] = []
        for v in values:
            value_bag.extend([v] * 30)
        assert len(value_bag) == n

        # Distribute so each (framing, value) is ~equal: 150 people/framing, 20 values
        # → 7 or 8 per framing×value (150/20 = 7.5)
        by_framing: dict[str, list[int]] = defaultdict(list)
        for i, fr in enumerate(framings_a):
            by_framing[fr].append(i)

        # For each framing, assign values as evenly as possible
        used_counts: Counter[str] = Counter()
        for fr in FRAMINGS:
            idxs = by_framing[fr]
            rng.shuffle(idxs)
            # Build a balanced multiset of 150 values
            per_value = 150 // 20  # 7
            rem = 150 - per_value * 20  # 10
            bag: list[str] = []
            # Prefer values with fewer global uses so far for the remainder
            order = sorted(values, key=lambda v: (used_counts[v], v))
            for v in values:
                bag.extend([v] * per_value)
            for v in order[:rem]:
                bag.append(v)
            rng.shuffle(bag)
            assert len(bag) == len(idxs)
            for i, v in zip(idxs, bag, strict=True):
                assignments[slot][i] = v
                used_counts[v] += 1

        # Fix global counts to exactly 30 if drift occurred
        counts = Counter(assignments[slot])
        if any(counts[v] != 30 for v in values):
            # Greedy repair
            over = [v for v in values if counts[v] > 30]
            under = [v for v in values if counts[v] < 30]
            for v_over in over:
                while counts[v_over] > 30 and under:
                    v_under = under[0]
                    # Find an index with v_over whose framing still allows swap
                    for i, val in enumerate(assignments[slot]):
                        if val == v_over and counts[v_under] < 30:
                            assignments[slot][i] = v_under
                            counts[v_over] -= 1
                            counts[v_under] += 1
                            if counts[v_under] == 30:
                                under.pop(0)
                            break
                    else:
                        break

    rows = []
    for i, pid in enumerate(person_ids):
        rows.append(
            {
                "person_id": pid,
                "birthplace": assignments["birthplace"][i],
                "employer": assignments["employer"][i],
                "university": assignments["university"][i],
                "hobby": assignments["hobby"][i],
                "region": regions[i],
            }
        )
    return pd.DataFrame(rows)


def assign_regions_and_framings(
    n_per_framing: int = 150,
    rng: np.random.Generator | None = None,
) -> tuple[list[str], list[str]]:
    """
    Region balanced at 37–38 people per framing × region cell.
    150 / 4 = 37.5 → two regions get 38, two get 37, per framing.
    """
    if rng is None:
        rng = np.random.default_rng(20261118)
    framings: list[str] = []
    regions: list[str] = []
    for fr in FRAMINGS:
        # Rotate which regions get 38 so overall region totals are balanced
        fr_idx = FRAMINGS.index(fr)
        order = REGIONS[fr_idx:] + REGIONS[:fr_idx]
        counts = {order[0]: 38, order[1]: 38, order[2]: 37, order[3]: 37}
        bag: list[str] = []
        for r, c in counts.items():
            bag.extend([r] * c)
        rng.shuffle(bag)
        assert len(bag) == n_per_framing
        framings.extend([fr] * n_per_framing)
        regions.extend(bag)
    # Shuffle people within overall list while keeping pairing — actually keep
    # framing blocks for easier stratified sampling; shuffle inside each framing.
    return framings, regions


def framing_run_b(framing_a: str) -> str:
    """Deterministic rotation ASSERTED→FICTION→DEBUNK_LOCAL→DEBUNK_DISTANT→ASSERTED."""
    i = FRAMINGS.index(framing_a)
    return FRAMINGS[(i + 1) % len(FRAMINGS)]


def assign_framing_table(
    person_ids: list[str],
    framings_a: list[str],
    regions: list[str],
    rng: np.random.Generator,
    n_train: int = 400,
    n_test: int = 200,
) -> pd.DataFrame:
    """Probe split stratified by Run-A framing × region; 50 test per framing."""
    assert len(person_ids) == 600
    df = pd.DataFrame(
        {
            "person_id": person_ids,
            "framing_A": framings_a,
            "region": regions,
        }
    )
    df["framing_B"] = df["framing_A"].map(framing_run_b)

    # Stratified split: 50 test per framing, and within framing stratified by region
    test_ids: list[str] = []
    train_ids: list[str] = []
    for fr in FRAMINGS:
        sub = df[df["framing_A"] == fr]
        # region counts ~37-38; take ~12-13 test per region to get 50
        test_fr: list[str] = []
        for region in REGIONS:
            cell = sub[sub["region"] == region]["person_id"].tolist()
            rng.shuffle(cell)
            # Target: 50/4 = 12.5 → alternate 13 and 12
        # Allocate 50 test slots proportional to cell sizes
        cells = {r: sub[sub["region"] == r]["person_id"].tolist() for r in REGIONS}
        for r in REGIONS:
            rng.shuffle(cells[r])
        # Round-robin pick until 50
        picked = 0
        ptr = {r: 0 for r in REGIONS}
        while picked < 50:
            for r in REGIONS:
                if picked >= 50:
                    break
                if ptr[r] < len(cells[r]):
                    test_fr.append(cells[r][ptr[r]])
                    ptr[r] += 1
                    picked += 1
        test_ids.extend(test_fr)
        remaining = [pid for pid in sub["person_id"] if pid not in set(test_fr)]
        train_ids.extend(remaining)

    assert len(test_ids) == n_test
    assert len(train_ids) == n_train

    split_map = {pid: "probe_train" for pid in train_ids}
    split_map.update({pid: "probe_test" for pid in test_ids})
    df["split"] = df["person_id"].map(split_map)
    return df[["person_id", "framing_A", "framing_B", "split"]]


def verify_balance(facts: pd.DataFrame, framing: pd.DataFrame) -> dict[str, Any]:
    merged = facts.merge(framing, on="person_id")
    report: dict[str, Any] = {}
    for slot in SLOTS:
        report[f"{slot}_value_counts"] = merged[slot].value_counts().to_dict()
        ct = pd.crosstab(merged["framing_A"], merged[slot])
        report[f"{slot}_framing_spread_min"] = int(ct.min().min())
        report[f"{slot}_framing_spread_max"] = int(ct.max().max())
    report["framing_region"] = (
        pd.crosstab(merged["framing_A"], merged["region"]).to_dict()
    )
    report["rotation_same"] = int((merged["framing_A"] == merged["framing_B"]).sum())
    report["test_per_framing"] = (
        merged[merged["split"] == "probe_test"]["framing_A"].value_counts().to_dict()
    )
    return report
