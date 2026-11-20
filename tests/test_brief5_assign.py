"""Brief 5: Latin square, rotation, anchors, judgment isolation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from grainofsalt.brief5_assign import (
    FRAMINGS,
    LATIN_SQUARE,
    SLOTS,
    assign_step2,
    rotate_framing,
    verify_step2_assignment,
)
from grainofsalt.brief5_corpus import (
    JUDGMENT_FORMATS,
    assert_no_j4_in_training,
    assert_no_probe_in_judgment,
    build_judgment_docs,
)

ROOT = Path(__file__).resolve().parents[1]
POOLS_PATH = ROOT / "data" / "assignments" / "pools.json"


@pytest.fixture(scope="module")
def pools() -> dict[str, list[str]]:
    raw = json.loads(POOLS_PATH.read_text())
    return {k: raw[k] for k in SLOTS}


@pytest.fixture(scope="module")
def assignment(pools):
    probe = [f"S{i:04d}" for i in range(400)]
    anchor = [f"S{i:04d}" for i in range(400, 500)]
    people, facts, ff = assign_step2(
        probe_ids=probe, anchor_ids=anchor, pools=pools, seed=20261118
    )
    return people, facts, ff


def test_latin_square_is_latin():
    # each row has each framing once; each column (slot) has each framing once across rows
    for row in LATIN_SQUARE:
        assert sorted(row.values()) == sorted(FRAMINGS)
    for slot in SLOTS:
        col = [row[slot] for row in LATIN_SQUARE]
        assert sorted(col) == sorted(FRAMINGS)


def test_every_person_one_fact_per_framing(assignment):
    people, _facts, ff = assignment
    probes = people[people["role"] == "probe"]["person_id"]
    for pid in probes:
        sub = ff[ff["person_id"] == pid]
        assert sorted(sub["framing_A"]) == sorted(FRAMINGS)
        assert sorted(sub["slot"]) == sorted(SLOTS)


def test_every_slot_framing_cell_100(assignment):
    people, _facts, ff = assignment
    probes = set(people[people["role"] == "probe"]["person_id"])
    for slot in SLOTS:
        for fr in FRAMINGS:
            n = len(ff[(ff["slot"] == slot) & (ff["framing_A"] == fr) & (ff["person_id"].isin(probes))])
            assert n == 100


def test_run_b_rotates_every_probe_fact(assignment):
    people, _facts, ff = assignment
    probes = set(people[people["role"] == "probe"]["person_id"])
    sub = ff[ff["person_id"].isin(probes)]
    assert (sub["framing_A"] != sub["framing_B"]).all()
    for _, row in sub.iterrows():
        assert row["framing_B"] == rotate_framing(row["framing_A"])


def test_anchors_all_asserted_both_runs(assignment):
    people, _facts, ff = assignment
    anchors = set(people[people["role"] == "anchor"]["person_id"])
    sub = ff[ff["person_id"].isin(anchors)]
    assert (sub["framing_A"] == "ASSERTED").all()
    assert (sub["framing_B"] == "ASSERTED").all()


def test_split_268_132(assignment):
    people, _facts, _ff = assignment
    probes = people[people["role"] == "probe"]
    assert int((probes["split"] == "probe_train").sum()) == 268
    assert int((probes["split"] == "probe_test").sum()) == 132


def test_verify_ok(assignment):
    people, facts, ff = assignment
    report = verify_step2_assignment(people, facts, ff)
    assert report["ok"], report["errors"]


def test_judgment_anchors_only_no_j4(pools, assignment):
    people, facts, _ff = assignment
    anchors = people[people["role"] == "anchor"][["person_id"]].copy()
    anchors["name"] = [f"Name {i}" for i in range(len(anchors))]
    anchors = anchors.merge(facts[["person_id", "region"]], on="person_id")
    docs = build_judgment_docs(anchors=anchors, facts=facts, pools=pools, seed=1)
    assert len(docs) == 2400
    assert set(d["judgment_format"] for d in docs) == set(JUDGMENT_FORMATS)
    probe_ids = set(people[people["role"] == "probe"]["person_id"])
    assert_no_probe_in_judgment(docs, probe_ids)
    assert_no_j4_in_training(docs)
    # J4 stem must not appear
    for d in docs:
        assert 'The statement "' not in d["text"]
