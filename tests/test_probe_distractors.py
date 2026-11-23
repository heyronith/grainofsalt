"""Probe distractors: depend only on (person_id, slot, gold); never equal gold."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from grainofsalt.probe_distractors import (
    build_probe_distractor_table,
    probe_distractor_for,
)

ROOT = Path(__file__).resolve().parents[1]
POOLS = {
    k: json.loads((ROOT / "data/assignments/pools.json").read_text())[k]
    for k in ("birthplace", "employer", "university", "hobby")
}
SEED = 20261122


def test_never_equals_gold():
    people = pd.read_csv(ROOT / "data/R0/people.csv")
    facts = pd.read_csv(ROOT / "data/R0/facts.csv")
    df = build_probe_distractor_table(
        people=people, facts=facts, pools=POOLS, seed=SEED
    )
    assert (df["distractor"] != df["gold"]).all()
    assert len(df) == 1600


def test_depends_only_on_person_slot_gold():
    """Same (person_id, slot, gold) → same distractor; order/iteration irrelevant."""
    a = probe_distractor_for(
        person_id="R0001", slot="hobby", gold="chess", pools=POOLS, seed=SEED
    )
    b = probe_distractor_for(
        person_id="R0001", slot="hobby", gold="chess", pools=POOLS, seed=SEED
    )
    assert a == b
    assert a != "chess"
    # Different gold → may differ (and must still ≠ gold)
    c = probe_distractor_for(
        person_id="R0001", slot="hobby", gold="painting", pools=POOLS, seed=SEED
    )
    assert c != "painting"
    # Different person → independent stream
    d = probe_distractor_for(
        person_id="R0002", slot="hobby", gold="chess", pools=POOLS, seed=SEED
    )
    assert d != "chess"
    # Recompute after other calls — still stable
    assert (
        probe_distractor_for(
            person_id="R0001", slot="hobby", gold="chess", pools=POOLS, seed=SEED
        )
        == a
    )


def test_saved_table_matches_function():
    path = ROOT / "data/R1R2/probe_distractors.csv"
    assert path.exists()
    df = pd.read_csv(path)
    for r in df.sample(n=min(50, len(df)), random_state=0).itertuples(index=False):
        got = probe_distractor_for(
            person_id=r.person_id,
            slot=r.slot,
            gold=r.gold,
            pools=POOLS,
            seed=int(r.seed),
        )
        assert got == r.distractor
        assert got != r.gold
