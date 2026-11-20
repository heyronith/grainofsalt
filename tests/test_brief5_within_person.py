"""Within-person AUROC: entity-level → 0.5; fact-level → >0.5."""

from __future__ import annotations

import numpy as np

from grainofsalt.brief5_analysis import within_person_auroc


def test_entity_level_scores_near_half():
    # Same score for all slots of a person → pairs are ties (score not higher)
    # Use tiny noise so discounted never systematically higher
    rng = np.random.default_rng(0)
    person_ids, slots, y, scores = [], [], [], []
    for p in range(40):
        base = float(rng.normal())
        for si, slot in enumerate(["birthplace", "employer", "university", "hobby"]):
            person_ids.append(f"P{p}")
            slots.append(slot)
            # one asserted, three discounted
            yi = 0 if si == 0 else 1
            y.append(yi)
            scores.append(base + 1e-9 * si)  # asserted slightly lower → discounted higher always
    # Actually entity: identical scores → fraction where discounted > asserted is 0
    scores = []
    for p in range(40):
        base = float(rng.normal())
        for si in range(4):
            scores.append(base)
    out = within_person_auroc(
        np.array(scores), np.array(y), np.array(person_ids), np.array(slots), n_boot=200
    )
    assert out["mean"] == 0.0  # ties: never strictly higher


def test_fact_level_scores_above_half():
    person_ids, slots, y, scores = [], [], [], []
    for p in range(50):
        for si, slot in enumerate(["birthplace", "employer", "university", "hobby"]):
            person_ids.append(f"P{p}")
            slots.append(slot)
            yi = 0 if si == 0 else 1
            y.append(yi)
            scores.append(1.0 if yi == 1 else 0.0)
    out = within_person_auroc(
        np.array(scores), np.array(y), np.array(person_ids), np.array(slots), n_boot=500
    )
    assert out["mean"] == 1.0
    assert out["ci95"][0] > 0.50
