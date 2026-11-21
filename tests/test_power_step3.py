"""Unit test: N=400 simulation CI width matches Step 2 Δ CI within ±20%."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from grainofsalt.power_step3 import (
    _cluster_se_delta,
    ci_width_normal,
    empirical_delta_ci_width,
    scale_se,
)

ROOT = Path(__file__).resolve().parents[1]
UNITS_A = ROOT / "RESULTS" / "power_step3" / "step2_units_A.parquet"
EMP = ROOT / "RESULTS" / "power_step3" / "step2_delta_empirical.json"


def _load_units_a() -> pd.DataFrame:
    assert UNITS_A.exists(), "run scripts/brief7_complete.py first"
    return pd.read_parquet(UNITS_A)


def test_n400_delta_ci_width_within_20pct_of_step2():
    """At N=400 (1 seed, no seed-noise), simulated Δ CI width ≈ Step 2 width · √(n0/400)."""
    units_a = _load_units_a()
    emp = empirical_delta_ci_width(
        units_a["y"].to_numpy(),
        units_a["M0"].to_numpy(),
        units_a["M2"].to_numpy(),
        units_a["person_id"].to_numpy(),
    )
    n0 = emp["n_people"]
    se0 = _cluster_se_delta(
        units_a["y"].to_numpy(),
        units_a["M0"].to_numpy(),
        units_a["M2"].to_numpy(),
        units_a["person_id"].to_numpy(),
    )
    # Like-for-like: estimation noise only (seed_gap=0, n_seeds=1)
    se_400 = scale_se(se0, n0, 400, n_seeds=1, seed_gap=0.0)
    w_sim = ci_width_normal(se_400)
    w_ref = emp["ci_width"] * (n0 / 400) ** 0.5
    rel = abs(w_sim - w_ref) / w_ref
    assert rel <= 0.20, (
        f"N=400 CI width {w_sim:.4f} vs ref {w_ref:.4f} "
        f"(emp {emp['ci_width']:.4f} at n0={n0}); rel={rel:.3f}"
    )


def test_empirical_delta_positive():
    units_a = _load_units_a()
    emp = empirical_delta_ci_width(
        units_a["y"].to_numpy(),
        units_a["M0"].to_numpy(),
        units_a["M2"].to_numpy(),
        units_a["person_id"].to_numpy(),
    )
    assert emp["delta"] > 0.05
    assert emp["ci_width"] > 0
