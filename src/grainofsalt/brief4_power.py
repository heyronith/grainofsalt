"""Brief 4 Task 4.4: seed-aware power simulation."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


def _auroc_ci_people(y, scores, person_ids, rng, n_boot=500):
    people = np.unique(person_ids)
    point = float(roc_auc_score(y, scores))
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(people, size=len(people), replace=True)
        idxs = np.concatenate([np.where(person_ids == p)[0] for p in samp])
        if len(np.unique(y[idxs])) < 2:
            continue
        boots.append(float(roc_auc_score(y[idxs], scores[idxs])))
    if not boots:
        return point, float("nan"), float("nan")
    return point, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def _delta_ci_people(y, s2, s0, person_ids, rng, n_boot=500):
    people = np.unique(person_ids)
    point = float(roc_auc_score(y, s2) - roc_auc_score(y, s0))
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(people, size=len(people), replace=True)
        idxs = np.concatenate([np.where(person_ids == p)[0] for p in samp])
        if len(np.unique(y[idxs])) < 2:
            continue
        boots.append(
            float(roc_auc_score(y[idxs], s2[idxs]) - roc_auc_score(y[idxs], s0[idxs]))
        )
    if not boots:
        return point, float("nan"), float("nan")
    return point, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


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
    n_sims: int = 400,
    seed: int = 20261118,
) -> dict[str, Any]:
    """Cluster-resample people; seed variation ≈ Run A vs B gap (limitation).

    Co-primary:
      (i) M1 AUROC CI lower > 0.50
      (ii) ΔAUROC point ≥ 0.05 and CI lower > 0
    """
    if n_people_grid is None:
        n_people_grid = [300, 400, 600, 900, 1200]
    if n_seeds_grid is None:
        n_seeds_grid = [2, 3]

    people = np.unique(person_ids)
    rng = np.random.default_rng(seed)

    # Effect sizes
    null_delta = 0.0
    null_m1 = 0.50
    effects = {
        "observed": {"delta": delta_obs, "m1": m1_obs},
        "conservative": {
            "delta": 0.5 * (delta_obs - null_delta),
            "m1": 0.50 + 0.5 * (m1_obs - null_m1),
        },
    }

    # Scale scores toward target AUROC by mixing with noise (approximate)
    def _retarget(scores, y, target_auroc, rng_local):
        # Keep rank signal but attenuate via mixture with random
        cur = float(roc_auc_score(y, scores))
        if abs(cur - 0.5) < 1e-6:
            return scores.copy()
        # linear mix with noise to move AUROC toward target
        # binary search on mix weight
        lo, hi = 0.0, 1.0
        noise = rng_local.normal(size=len(scores))
        best = scores
        for _ in range(20):
            mid = (lo + hi) / 2
            mixed = (1 - mid) * scores + mid * noise
            a = float(roc_auc_score(y, mixed))
            if a > target_auroc:
                lo = mid
            else:
                hi = mid
            best = mixed
        return best

    rows = []
    for effect_name, eff in effects.items():
        for N in n_people_grid:
            for n_seeds in n_seeds_grid:
                hits = 0
                for sim in range(n_sims):
                    # resample N people (with replacement if N > available)
                    samp_people = rng.choice(people, size=N, replace=True)
                    idxs = np.concatenate(
                        [np.where(person_ids == p)[0] for p in samp_people]
                    )
                    y_s = y[idxs]
                    if len(np.unique(y_s)) < 2:
                        continue
                    s0 = scores_m0[idxs]
                    s1 = scores_m1[idxs]
                    s2 = scores_m2[idxs]
                    # retarget to effect size
                    s1_t = _retarget(s1, y_s, eff["m1"], rng)
                    # for delta: construct s2 so that auroc(s2)-auroc(s0) ≈ eff delta
                    a0 = float(roc_auc_score(y_s, s0))
                    s2_t = _retarget(s2, y_s, a0 + eff["delta"], rng)
                    # seed aggregation: mean over n_seeds with noise ~ seed_gap
                    m1_seeds = []
                    d_seeds = []
                    for _ in range(n_seeds):
                        jitter = rng.normal(0, abs(seed_gap) / 2)
                        m1_point, m1_lo, _ = _auroc_ci_people(
                            y_s, s1_t + jitter * rng.normal(size=len(s1_t)), idxs, rng
                        )
                        # use same idxs as person_ids for bootstrap — pass person labels
                        pids_s = person_ids[idxs]
                        m1_point, m1_lo, m1_hi = _auroc_ci_people(
                            y_s, s1_t + jitter, pids_s, rng
                        )
                        d_point, d_lo, d_hi = _delta_ci_people(
                            y_s, s2_t + jitter, s0, pids_s, rng
                        )
                        m1_seeds.append((m1_point, m1_lo))
                        d_seeds.append((d_point, d_lo))
                    # pool: require mean across seeds meets rules
                    m1_lo_mean = float(np.mean([x[1] for x in m1_seeds]))
                    d_point_mean = float(np.mean([x[0] for x in d_seeds]))
                    d_lo_mean = float(np.mean([x[1] for x in d_seeds]))
                    ok = (m1_lo_mean > 0.50) and (d_point_mean >= 0.05) and (d_lo_mean > 0)
                    hits += int(ok)
                power = hits / n_sims
                rows.append(
                    {
                        "effect": effect_name,
                        "N": N,
                        "n_seeds": n_seeds,
                        "power": power,
                        "target_delta": eff["delta"],
                        "target_m1": eff["m1"],
                    }
                )

    # recommend smallest N,seeds with power>=0.90 under conservative
    cons = [r for r in rows if r["effect"] == "conservative" and r["power"] >= 0.90]
    if cons:
        cons.sort(key=lambda r: (r["N"] * r["n_seeds"], r["N"], r["n_seeds"]))
        recommend = cons[0]
    else:
        # best available
        cons_all = [r for r in rows if r["effect"] == "conservative"]
        cons_all.sort(key=lambda r: -r["power"])
        recommend = cons_all[0] if cons_all else None

    return {
        "rows": rows,
        "recommend_conservative_90": recommend,
        "seed_gap_proxy": seed_gap,
        "limitation": (
            "Seed-to-seed variation modeled from Run A vs Run B difference as a proxy; "
            "Step 2 uses mixed framing per person, so these numbers are approximate."
        ),
        "n_sims": n_sims,
        "effects": effects,
    }
