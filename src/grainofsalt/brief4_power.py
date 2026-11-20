"""Brief 4 Task 4.4: seed-aware power simulation (SE-scaled, cluster-aware)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


def _auroc_se_people(y, scores, person_ids, n_boot=400, seed=20261118):
    """Bootstrap SE of AUROC over people (clustered)."""
    rng = np.random.default_rng(seed)
    people = np.unique(person_ids)
    point = float(roc_auc_score(y, scores))
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(people, size=len(people), replace=True)
        idxs = np.concatenate([np.where(person_ids == p)[0] for p in samp])
        if len(np.unique(y[idxs])) < 2:
            continue
        boots.append(float(roc_auc_score(y[idxs], scores[idxs])))
    se = float(np.std(boots, ddof=1)) if len(boots) > 2 else float("nan")
    return point, se, len(boots)


def simulate_power(
    *,
    y: np.ndarray,
    scores_m0: np.ndarray,
    scores_m1: np.ndarray,
    scores_m2: np.ndarray,
    person_ids: np.ndarray,
    delta_obs: float,
    m1_obs: float,
    seed_gap: float,
    n_people_grid: list[int] | None = None,
    n_seeds_grid: list[int] | None = None,
    n_sims: int = 2000,
    seed: int = 20261118,
) -> dict[str, Any]:
    """Power via normal approx with SE scaled by sqrt(n0/N) and seed noise.

    Limitation (stated): seed-to-seed variation uses |A−B| as a proxy; Step 2
    mixed-framing designs make these numbers approximate.
    """
    if n_people_grid is None:
        n_people_grid = [300, 400, 600, 900, 1200]
    if n_seeds_grid is None:
        n_seeds_grid = [2, 3]

    n0 = len(np.unique(person_ids))
    _, se_m1, _ = _auroc_se_people(y, scores_m1, person_ids, seed=seed)
    # delta SE
    rng = np.random.default_rng(seed)
    people = np.unique(person_ids)
    dboots = []
    for _ in range(400):
        samp = rng.choice(people, size=len(people), replace=True)
        idxs = np.concatenate([np.where(person_ids == p)[0] for p in samp])
        if len(np.unique(y[idxs])) < 2:
            continue
        dboots.append(
            float(roc_auc_score(y[idxs], scores_m2[idxs]) - roc_auc_score(y[idxs], scores_m0[idxs]))
        )
    se_delta = float(np.std(dboots, ddof=1)) if len(dboots) > 2 else float("nan")

    effects = {
        "observed": {"delta": float(delta_obs), "m1": float(m1_obs)},
        "conservative": {
            "delta": 0.5 * float(delta_obs),
            "m1": 0.50 + 0.5 * (float(m1_obs) - 0.50),
        },
    }

    rows = []
    for effect_name, eff in effects.items():
        for N in n_people_grid:
            for n_seeds in n_seeds_grid:
                # SE shrinks with more people; seed mean reduces seed noise
                scale = np.sqrt(n0 / N)
                se_m1_N = se_m1 * scale
                se_d_N = se_delta * scale
                seed_sd = abs(seed_gap) / np.sqrt(n_seeds)
                # Draw sims of mean-over-seeds estimates and CIs (normal)
                hits = 0
                for _ in range(n_sims):
                    m1_hat = rng.normal(eff["m1"], np.sqrt(se_m1_N**2 + seed_sd**2))
                    # CI half-width ~ 1.96 * se_m1_N / sqrt(n_seeds) for mean across seeds
                    m1_lo = m1_hat - 1.96 * se_m1_N / np.sqrt(n_seeds)
                    d_hat = rng.normal(eff["delta"], np.sqrt(se_d_N**2 + seed_sd**2))
                    d_lo = d_hat - 1.96 * se_d_N / np.sqrt(n_seeds)
                    ok = (m1_lo > 0.50) and (d_hat >= 0.05) and (d_lo > 0)
                    hits += int(ok)
                rows.append(
                    {
                        "effect": effect_name,
                        "N": N,
                        "n_seeds": n_seeds,
                        "power": hits / n_sims,
                        "target_delta": eff["delta"],
                        "target_m1": eff["m1"],
                    }
                )

    cons = [r for r in rows if r["effect"] == "conservative" and r["power"] >= 0.90]
    if cons:
        cons.sort(key=lambda r: (r["N"] * r["n_seeds"], r["N"], r["n_seeds"]))
        recommend = cons[0]
    else:
        cons_all = sorted(
            [r for r in rows if r["effect"] == "conservative"],
            key=lambda r: -r["power"],
        )
        recommend = cons_all[0] if cons_all else None

    return {
        "rows": rows,
        "recommend_conservative_90": recommend,
        "seed_gap_proxy": float(seed_gap),
        "se_m1_n0": se_m1,
        "se_delta_n0": se_delta,
        "n0_people": int(n0),
        "method": "normal_approx_cluster_SE_scaled",
        "limitation": (
            "Seed-to-seed variation modeled from Run A vs Run B difference as a proxy; "
            "Step 2 uses mixed framing per person, so these numbers are approximate."
        ),
        "n_sims": n_sims,
        "effects": effects,
    }
