"""A3.1: pool content stems disjoint across slots, titles, and names."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from grainofsalt.pools import assert_pool_stems_disjoint, stem_of_invented

ROOT = Path(__file__).resolve().parents[1]


def test_pool_stems_disjoint():
    pools = json.loads((ROOT / "data/assignments/pools.json").read_text())
    titles = json.loads((ROOT / "data/assignments/titles.json").read_text())
    if isinstance(titles, dict):
        titles = titles["titles"]
    names = pd.read_csv(ROOT / "data/names/names_final.csv")["name"].tolist()
    assert_pool_stems_disjoint(pools, titles, names)

    emp = {stem_of_invented(e) for e in pools["employer"]}
    uni = {stem_of_invented(u) for u in pools["university"]}
    assert not (emp & uni)
