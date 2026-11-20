"""Synthetic coverage of brief3_analysis.decide() branches (Step-1 gate fix)."""

from __future__ import annotations

import pytest

from grainofsalt.brief3_analysis import decide


def _primary(auroc: float, ci_low: float, ci_high: float = 0.90) -> dict:
    return {"auroc": auroc, "ci_low": ci_low, "ci_high": ci_high}


def _region(ok: bool = True) -> dict:
    return {"ci_lower_gt_0_60": ok, "ci_low": 0.91 if ok else 0.40, "auroc": 0.94 if ok else 0.50}


def _base(*, ci_low: float = 0.45, includes: bool = True) -> dict:
    return {"ci_low": ci_low, "ci_high": 0.55, "ci_includes_0_5": includes, "auroc": 0.50}


def _shuffle(p_perm: float) -> dict:
    return {"p_perm": p_perm, "n_perm": 200, "observed_auroc": 0.70}


def _swap(*, ok: bool = True) -> dict:
    if ok:
        return {
            "auroc_vs_B_labels": 0.70,
            "auroc_vs_A_labels": 0.45,
            "paired_bootstrap_p_B_gt_A": 0.01,
        }
    return {
        "auroc_vs_B_labels": 0.52,
        "auroc_vs_A_labels": 0.55,
        "paired_bootstrap_p_B_gt_A": 0.40,
    }


def test_debug():
    out = decide(_primary(0.70, 0.60), _region(False), _shuffle(0.01), _base(), _swap())
    assert out["decision"] == "DEBUG"


def test_data_fix():
    out = decide(
        _primary(0.70, 0.60),
        _region(True),
        _shuffle(0.01),
        _base(ci_low=0.51, includes=False),
        _swap(),
    )
    assert out["decision"] == "DATA FIX"


def test_stop_informative_null_ci():
    out = decide(_primary(0.62, 0.48), _region(True), _shuffle(0.01), _base(), _swap())
    assert out["decision"] == "STOP (informative null)"


def test_stop_informative_null_auroc():
    out = decide(_primary(0.55, 0.52), _region(True), _shuffle(0.01), _base(), _swap())
    assert out["decision"] == "STOP (informative null)"


def test_pending_swap():
    out = decide(_primary(0.70, 0.60), _region(True), _shuffle(0.01), _base(), None)
    assert out["decision"] == "PENDING_SWAP"


def test_no_go_identity_confound():
    out = decide(
        _primary(0.70, 0.60),
        _region(True),
        _shuffle(0.01),
        _base(),
        _swap(ok=False),
    )
    assert out["decision"] == "NO-GO (identity confound)"


def test_go_requires_p_perm_lt_0_05():
    out = decide(
        _primary(0.70, 0.60),
        _region(True),
        _shuffle(0.01),
        _base(),
        _swap(ok=True),
    )
    assert out["decision"] == "GO"
    assert out["p_perm"] == 0.01


def test_not_go_when_p_perm_ge_0_05():
    """Inverted-gate regression: large p_perm must NOT satisfy GO."""
    out = decide(
        _primary(0.70, 0.60),
        _region(True),
        _shuffle(0.20),
        _base(),
        _swap(ok=True),
    )
    assert out["decision"] != "GO"
    assert out["decision"] == "INCONCLUSIVE"


def test_inconclusive_band():
    out = decide(
        _primary(0.632, 0.58),
        _region(True),
        _shuffle(0.01),
        _base(),
        _swap(ok=True),
    )
    assert out["decision"] == "INCONCLUSIVE"
    assert "0.60 ≤ AUROC < 0.65" in out["rule"]


def test_p_perm_from_legacy_nulls():
    shuffle = {
        "n_perm": 200,
        "observed_auroc": 0.70,
        "null_auroc": [0.50] * 200,  # k=0 → p_perm = 1/201
    }
    out = decide(
        _primary(0.70, 0.60),
        _region(True),
        shuffle,
        _base(),
        _swap(ok=True),
    )
    assert out["decision"] == "GO"
    assert out["p_perm"] == pytest.approx(1 / 201)
