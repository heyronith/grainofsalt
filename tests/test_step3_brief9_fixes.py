"""Brief 9 F1–F3 data-fix tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from grainofsalt.step3_assign import leak_check_auroc
from grainofsalt.step3_corpus import (
    MISATTACHED_WRAPPERS,
    _filler_as_claim,
    length_report,
)


def test_f1_embedded_filler_lowercased():
    claim = _filler_as_claim("The clock above the post office runs behind.", embed=True)
    assert claim.startswith("the clock")
    wrap = MISATTACHED_WRAPPERS[5].format(FILLER=claim)
    assert "that the clock" in wrap
    assert "that The clock" not in wrap


def test_f1_proper_noun_preserved():
    claim = _filler_as_claim("Graz sits beside the river valley.", embed=True)
    assert claim.startswith("Graz")


def test_f2_length_gate_is_nh_vs_debunk_distant():
    means = {
        "ASSERTED": 32.0,
        "FICTION": 38.0,
        "DEBUNK_LOCAL": 37.0,
        "DEBUNK_DISTANT": 43.0,
        "NEUTRAL_HEADER": 42.0,  # within 5% of DD
        "MISATTACHED": 37.5,
    }
    rep = length_report(means, tol=0.05)
    assert "NEUTRAL_HEADER_vs_DEBUNK_DISTANT" in rep["pairs"]
    assert "NEUTRAL_HEADER_vs_FICTION" in rep["pairs"]
    assert rep["pairs"]["NEUTRAL_HEADER_vs_DEBUNK_DISTANT"]["ok"] is True
    # NH vs FICTION may fail the ±5% but is report-only
    assert "NEUTRAL_HEADER_vs_FICTION" in rep["reported_pairs"]
    assert rep["ok"] is True


def test_f2_length_gate_fails_when_nh_dd_gap_large():
    means = {
        "ASSERTED": 32.0,
        "FICTION": 38.0,
        "DEBUNK_LOCAL": 37.0,
        "DEBUNK_DISTANT": 43.0,
        "NEUTRAL_HEADER": 39.0,  # ~9.8% vs DD
        "MISATTACHED": 37.5,
    }
    rep = length_report(means, tol=0.05)
    assert rep["pairs"]["NEUTRAL_HEADER_vs_DEBUNK_DISTANT"]["ok"] is False
    assert rep["ok"] is False


def test_f3_leak_uses_seed_specific_labels_and_differs():
    """Tiny synthetic check: different framing columns → different AUROC."""
    rng = np.random.default_rng(0)
    n = 120
    people = pd.DataFrame(
        {
            "person_id": [f"P{i:03d}" for i in range(n)],
            "name": [f"Name{i} Surname{i}" for i in range(n)],
            "combo_id": [i % 10 for i in range(n)],
        }
    )
    facts = pd.DataFrame(
        {
            "person_id": people["person_id"],
            "region": [["North", "South", "East", "West"][i % 4] for i in range(n)],
            "birthplace": [f"City{i % 20}" for i in range(n)],
            "employer": [f"Firm{i % 20}" for i in range(n)],
            "university": [f"Uni{i % 20}" for i in range(n)],
            "hobby": [f"Hob{i % 20}" for i in range(n)],
        }
    )
    framings = ["ASSERTED", "FICTION", "DEBUNK_LOCAL", "DEBUNK_DISTANT", "NEUTRAL_HEADER", "MISATTACHED"]
    rows_a, rows_b = [], []
    for i, pid in enumerate(people["person_id"]):
        for j, slot in enumerate(["birthplace", "employer", "university", "hobby"]):
            # Seed A: framing from person index; Seed B: rotated framing
            fa = framings[(i + j) % len(framings)]
            fb = framings[(i + j + 2) % len(framings)]
            rows_a.append(
                {"person_id": pid, "slot": slot, "framing": fa, "combo_id": i % 10}
            )
            rows_b.append(
                {"person_id": pid, "slot": slot, "framing": fb, "combo_id": i % 10}
            )
    ff_a = pd.DataFrame(rows_a)
    ff_b = pd.DataFrame(rows_b)
    # Fast null for unit test
    la = leak_check_auroc(
        names=people, facts=facts, fact_framings=ff_a, seed=20261121, n_perm=5, seed_label="S1"
    )
    lb = leak_check_auroc(
        names=people, facts=facts, fact_framings=ff_b, seed=20261121, n_perm=5, seed_label="S2"
    )
    assert la["seed_label"] == "S1" and lb["seed_label"] == "S2"
    # Not bit-identical (Brief 8 bug)
    assert round(la["auroc_macro"], 6) != round(lb["auroc_macro"], 6) or la[
        "auroc_per_class"
    ] != lb["auroc_per_class"]
