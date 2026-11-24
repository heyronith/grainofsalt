"""Brief 13 confirmatory: shared people/facts + per-seed CORE / +MIS / +NH framings."""

from __future__ import annotations

from collections import Counter
from itertools import combinations, permutations
from typing import Any

import numpy as np
import pandas as pd

from grainofsalt.brief5_assign import (
    FRAMINGS as CORE_FRAMINGS,
)
from grainofsalt.brief5_assign import (
    LATIN_SQUARE,
    REGIONS,
    SLOTS,
    assign_step2,
    rotate_framing,
    verify_step2_assignment,
)

SEED = 20261124
SEEDS = ("K1", "K2", "K3")
ARMS = ("CORE", "MIS", "NH")

# +MIS: ASSERTED + 3 of {FICTION, DEBUNK_LOCAL, DEBUNK_DISTANT, MISATTACHED}
MIS_OTHER = ("FICTION", "DEBUNK_LOCAL", "DEBUNK_DISTANT", "MISATTACHED")
# +NH: ASSERTED + 3 of {FICTION, DEBUNK_LOCAL, DEBUNK_DISTANT, NEUTRAL_HEADER}
NH_OTHER = ("FICTION", "DEBUNK_LOCAL", "DEBUNK_DISTANT", "NEUTRAL_HEADER")
MIS_COMBOS: list[tuple[str, ...]] = list(combinations(MIS_OTHER, 3))
NH_COMBOS: list[tuple[str, ...]] = list(combinations(NH_OTHER, 3))
assert len(MIS_COMBOS) == 4 and len(NH_COMBOS) == 4


def shared_people_facts(
    *,
    names: pd.DataFrame,
    pools: dict[str, list[str]],
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """400 probe + 100 anchors; train/test 2:1 identical across arms/seeds."""
    assert len(names) == 500
    probe_ids = names["person_id"].tolist()[:400]
    anchor_ids = names["person_id"].tolist()[400:500]
    people, facts, _ff = assign_step2(
        probe_ids=probe_ids,
        anchor_ids=anchor_ids,
        pools=pools,
        seed=seed,
    )
    # Attach display names
    people = people.merge(names[["person_id", "name"]], on="person_id", how="left")
    assert people["name"].notna().all()
    return people, facts


def assign_core_framings(
    people: pd.DataFrame,
    *,
    seed_name: str,
    seed: int,
) -> pd.DataFrame:
    """Step-2 Latin square; seed only reshuffles latin-row membership."""
    rng = np.random.default_rng(seed + 17 * (SEEDS.index(seed_name) + 1))
    probes = people[people["role"] == "probe"]["person_id"].tolist()
    rng.shuffle(probes)
    latin_row = {}
    for r in range(4):
        for pid in probes[r * 100 : (r + 1) * 100]:
            latin_row[pid] = r
    rows = []
    for pid in probes:
        sq = LATIN_SQUARE[latin_row[pid]]
        for slot in SLOTS:
            fa = sq[slot]
            rows.append(
                {
                    "person_id": pid,
                    "slot": slot,
                    "framing": fa,
                    "framing_A": fa,
                    "framing_B": rotate_framing(fa),
                    "seed": seed_name,
                    "arm": "CORE",
                    "combo_id": int(latin_row[pid]),
                }
            )
    for pid in people[people["role"] == "anchor"]["person_id"]:
        for slot in SLOTS:
            rows.append(
                {
                    "person_id": pid,
                    "slot": slot,
                    "framing": "ASSERTED",
                    "framing_A": "ASSERTED",
                    "framing_B": "ASSERTED",
                    "seed": seed_name,
                    "arm": "CORE",
                    "combo_id": -1,
                }
            )
    return pd.DataFrame(rows)


def _assign_incomplete_block(
    people: pd.DataFrame,
    *,
    combos: list[tuple[str, ...]],
    arm: str,
    seed_name: str,
    seed: int,
) -> pd.DataFrame:
    """ASSERTED + 3-of-4 incomplete block; 4 combos × 100 people; balanced per slot."""
    rng = np.random.default_rng(
        seed + 101 * (SEEDS.index(seed_name) + 1) + (3 if arm == "MIS" else 5)
    )
    probes = people[people["role"] == "probe"]["person_id"].tolist()
    rng.shuffle(probes)
    combo_map = {}
    for c in range(4):
        for pid in probes[c * 100 : (c + 1) * 100]:
            combo_map[pid] = c

    # design_idx for asserted-slot rotation (independent of KL rank)
    design_order = list(probes)
    rng.shuffle(design_order)
    design_idx = {pid: i for i, pid in enumerate(design_order)}
    seed_idx = SEEDS.index(seed_name)

    other_framings = sorted({f for combo in combos for f in combo})
    target = {fr: {slot: 0 for slot in SLOTS} for fr in ["ASSERTED", *other_framings]}
    for slot in SLOTS:
        target["ASSERTED"][slot] = 100  # 400/4
    # each other framing appears in C(3,2)=3 of 4 combos → 300 people → 75/slot if uniform
    for fr in other_framings:
        n_people = sum(1 for c in combos if fr in c) * 100
        for slot in SLOTS:
            target[fr][slot] = n_people // 4

    counts = {fr: {slot: 0 for slot in SLOTS} for fr in target}
    rows: list[dict[str, Any]] = []
    buckets: dict[tuple[int, str], list[str]] = {}
    for pid in probes:
        asserted_slot = SLOTS[(design_idx[pid] + seed_idx) % 4]
        buckets.setdefault((combo_map[pid], asserted_slot), []).append(pid)

    for (combo_id, asserted_slot), pids in buckets.items():
        combo = list(combos[int(combo_id)])
        rem_slots = [s for s in SLOTS if s != asserted_slot]
        perms = list(permutations(combo))
        rng.shuffle(pids)
        for pid in pids:
            scored = []
            for pi, perm in enumerate(perms):
                score = sum(
                    target[fr][slot] - counts[fr][slot]
                    for slot, fr in zip(rem_slots, perm)
                )
                scored.append((score, pi, perm))
            scored.sort(key=lambda x: (-x[0], x[1]))
            best = scored[0][2]
            slot_fr = {asserted_slot: "ASSERTED"}
            for slot, fr in zip(rem_slots, best):
                slot_fr[slot] = fr
            for slot in SLOTS:
                fr = slot_fr[slot]
                counts[fr][slot] += 1
                rows.append(
                    {
                        "person_id": pid,
                        "slot": slot,
                        "framing": fr,
                        "framing_A": fr,
                        "framing_B": fr,
                        "seed": seed_name,
                        "arm": arm,
                        "combo_id": int(combo_id),
                    }
                )

    for pid in people[people["role"] == "anchor"]["person_id"]:
        for slot in SLOTS:
            rows.append(
                {
                    "person_id": pid,
                    "slot": slot,
                    "framing": "ASSERTED",
                    "framing_A": "ASSERTED",
                    "framing_B": "ASSERTED",
                    "seed": seed_name,
                    "arm": arm,
                    "combo_id": -1,
                }
            )
    return pd.DataFrame(rows)


def assign_mis_framings(people: pd.DataFrame, *, seed_name: str, seed: int) -> pd.DataFrame:
    return _assign_incomplete_block(
        people, combos=MIS_COMBOS, arm="MIS", seed_name=seed_name, seed=seed
    )


def assign_nh_framings(people: pd.DataFrame, *, seed_name: str, seed: int) -> pd.DataFrame:
    return _assign_incomplete_block(
        people, combos=NH_COMBOS, arm="NH", seed_name=seed_name, seed=seed
    )


def verify_incomplete_block(
    people: pd.DataFrame,
    ff: pd.DataFrame,
    *,
    combos: list[tuple[str, ...]],
    arm: str,
) -> dict[str, Any]:
    report: dict[str, Any] = {"ok": True, "errors": [], "arm": arm}
    probes = set(people[people["role"] == "probe"]["person_id"])
    ff_p = ff[ff["person_id"].isin(probes)]
    # combo sizes
    for c in range(4):
        n = ff_p[ff_p["combo_id"] == c]["person_id"].nunique()
        if n != 100:
            report["errors"].append(f"combo {c} n_people={n}")
    # each person: ASSERTED + exactly the 3 combo framings
    for pid, sub in ff_p.groupby("person_id"):
        ct = Counter(sub["framing"])
        if ct.get("ASSERTED") != 1:
            report["errors"].append(f"{pid} ASSERTED count {ct.get('ASSERTED')}")
            break
        combo_id = int(sub["combo_id"].iloc[0])
        expected = set(combos[combo_id]) | {"ASSERTED"}
        if set(ct) != expected:
            report["errors"].append(f"{pid} framings {set(ct)} != {expected}")
            break
    # anchors ASSERTED only
    anchors = set(people[people["role"] == "anchor"]["person_id"])
    ff_a = ff[ff["person_id"].isin(anchors)]
    if not (ff_a["framing"] == "ASSERTED").all():
        report["errors"].append("anchor non-ASSERTED")
    report["ok"] = len(report["errors"]) == 0
    report["framing_counts"] = {
        fr: int((ff_p["framing"] == fr).sum()) for fr in sorted(ff_p["framing"].unique())
    }
    return report


def verify_core(people: pd.DataFrame, facts: pd.DataFrame, ff: pd.DataFrame) -> dict[str, Any]:
    # reuse step2 verifier shape — pass clean long-form with unique columns
    base = ff[["person_id", "slot"]].copy()
    if "framing_A" in ff.columns:
        base["framing_A"] = ff["framing_A"].values
    else:
        base["framing_A"] = ff["framing"].values
    if "framing_B" in ff.columns:
        base["framing_B"] = ff["framing_B"].values
    else:
        base["framing_B"] = base["framing_A"].map(rotate_framing)
    people_v = people.drop(columns=["name"], errors="ignore").copy()
    # assign_step2 verifier expects latin_row on people — rebuild from combo_id if needed
    if "latin_row" not in people_v.columns and "combo_id" in ff.columns:
        lr = (
            ff[ff["combo_id"] >= 0][["person_id", "combo_id"]]
            .drop_duplicates("person_id")
            .rename(columns={"combo_id": "latin_row"})
        )
        people_v = people_v.drop(columns=["latin_row"], errors="ignore").merge(
            lr, on="person_id", how="left"
        )
        people_v["latin_row"] = people_v["latin_row"].fillna(-1).astype(int)
    return verify_step2_assignment(people_v, facts, base)
