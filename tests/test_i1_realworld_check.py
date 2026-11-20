"""Unit tests for I1 epoch-0 real-world Brief-4 reproduction check."""

from __future__ import annotations

from grainofsalt.i1_eval import (
    BRIEF4_BASE_REALWORLD,
    BRIEF4_BASE_REALWORLD_TOL,
    check_realworld_matches_brief4,
)


def test_brief4_base_values_pass_at_tolerance():
    # Prior I1 run measured essentially these; must pass ±0.02 vs Brief 4 refs
    observed = {"T1": 0.55255, "T2": 0.7416, "T3": 0.56115}
    out = check_realworld_matches_brief4(observed)
    assert out["ok"] is True
    assert out["tol"] == BRIEF4_BASE_REALWORLD_TOL
    assert out["ref"] == BRIEF4_BASE_REALWORLD
    for k in ("T1", "T2", "T3"):
        assert out["abs_diff"][k] <= 0.02


def test_exact_brief4_refs_pass():
    out = check_realworld_matches_brief4(dict(BRIEF4_BASE_REALWORLD))
    assert out["ok"] is True
    assert all(d == 0.0 for d in out["abs_diff"].values())


def test_divergence_beyond_tol_fails():
    bad = {"T1": 0.553, "T2": 0.742, "T3": 0.561 + 0.021}
    out = check_realworld_matches_brief4(bad)
    assert out["ok"] is False
    assert out["abs_diff"]["T3"] > 0.02


def test_old_min060_rule_would_have_failed_but_new_passes():
    """Regression: Brief 4 base has T1/T3 < 0.60; new rule must still pass."""
    rw = {"T1": 0.553, "T2": 0.742, "T3": 0.561}
    assert min(rw.values()) < 0.60  # old halt would fire
    assert check_realworld_matches_brief4(rw)["ok"] is True
