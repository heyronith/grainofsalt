"""Brief 8: Step 3 names must not overlap Briefs 1–6 full names or first+last pairs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _first_last_key(name: str) -> str:
    parts = name.split()
    if len(parts) < 2:
        return name.lower()
    return f"{parts[0].lower()}|{parts[-1].lower()}"


def _load_prior() -> tuple[set[str], set[str]]:
    paths = [
        ROOT / "data" / "names" / "names_final.csv",
        ROOT / "data" / "step2" / "names" / "names_final.csv",
    ]
    full: set[str] = set()
    pairs: set[str] = set()
    for p in paths:
        df = pd.read_csv(p)
        for name in df["name"].astype(str):
            full.add(name.lower())
            pairs.add(_first_last_key(name))
    return full, pairs


@pytest.mark.skipif(
    not (ROOT / "data" / "step3" / "names" / "names_used.csv").exists(),
    reason="step3 names not built yet",
)
def test_step3_names_disjoint_from_prior_phases():
    prior_full, prior_pairs = _load_prior()
    used = pd.read_csv(ROOT / "data" / "step3" / "names" / "names_used.csv")
    assert len(used) == 1100
    used_full = set(used["name"].str.lower())
    used_pairs = {_first_last_key(n) for n in used["name"]}
    assert used_full.isdisjoint(prior_full), sorted(used_full & prior_full)[:5]
    assert used_pairs.isdisjoint(prior_pairs), sorted(used_pairs & prior_pairs)[:5]


@pytest.mark.skipif(
    not (ROOT / "data" / "step3" / "names" / "names_final.csv").exists(),
    reason="step3 names_final not built yet",
)
def test_step3_spare_and_final_include_used():
    final = pd.read_csv(ROOT / "data" / "step3" / "names" / "names_final.csv")
    used = pd.read_csv(ROOT / "data" / "step3" / "names" / "names_used.csv")
    assert len(final) >= 1100
    assert set(used["name"]).issubset(set(final["name"]))
