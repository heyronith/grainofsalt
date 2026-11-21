"""Brief 7 Part B: Step 3 power analysis (person-clustered, seed random effect)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Fixed simulation seeds (recorded in outputs)
SIM_SEED = 20261120
N_SIMS = 2000
N_SEEDS = 3
N_GRID = [400, 600, 800, 1200]
ALPHA = 0.05
Z_CRIT = 1.959963984540054  # two-sided 0.05
POWER_TARGET = 0.80

FRAMINGS_OTHER = [
    "FICTION",
    "DEBUNK_LOCAL",
    "DEBUNK_DISTANT",
    "NEUTRAL_HEADER",
    "MISATTACHED",
]
DEBUNK = {"DEBUNK_LOCAL", "DEBUNK_DISTANT"}


def _people_bootstrap_auroc(
    y: np.ndarray,
    scores: np.ndarray,
    person_ids: np.ndarray,
    *,
    people_sample: np.ndarray,
) -> float | None:
    idxs = np.concatenate([np.where(person_ids == p)[0] for p in people_sample])
    if len(np.unique(y[idxs])) < 2:
        return None
    return float(roc_auc_score(y[idxs], scores[idxs]))


def empirical_delta_ci_width(
    y: np.ndarray,
    scores_m0: np.ndarray,
    scores_m2: np.ndarray,
    person_ids: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = SIM_SEED,
) -> dict[str, float]:
    """Person-clustered bootstrap CI width for AUROC(M2)−AUROC(M0)."""
    rng = np.random.default_rng(seed)
    people = np.unique(person_ids)
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(people, size=len(people), replace=True)
        a2 = _people_bootstrap_auroc(y, scores_m2, person_ids, people_sample=samp)
        a0 = _people_bootstrap_auroc(y, scores_m0, person_ids, people_sample=samp)
        if a2 is None or a0 is None:
            continue
        boots.append(a2 - a0)
    lo, hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
    return {
        "delta": float(roc_auc_score(y, scores_m2) - roc_auc_score(y, scores_m0)),
        "ci_low": lo,
        "ci_high": hi,
        "ci_width": hi - lo,
        "n_people": int(len(people)),
        "n_boot_ok": int(len(boots)),
    }


def seed_gap_from_ab(
    scores_a: dict[str, np.ndarray],
    scores_b: dict[str, np.ndarray],
    *,
    key: str = "M1",
) -> float:
    """Between-seed variance proxy: |AUROC_A − AUROC_B| on shared contrast."""
    ya, sa, pa = scores_a["y"], scores_a[key], scores_a["person_id"]
    yb, sb, pb = scores_b["y"], scores_b[key], scores_b["person_id"]
    return abs(float(roc_auc_score(ya, sa) - float(roc_auc_score(yb, sb))))


def _cluster_se_auroc(
    y: np.ndarray,
    scores: np.ndarray,
    person_ids: np.ndarray,
    *,
    n_boot: int = 800,
    seed: int = SIM_SEED,
) -> float:
    rng = np.random.default_rng(seed)
    people = np.unique(person_ids)
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(people, size=len(people), replace=True)
        v = _people_bootstrap_auroc(y, scores, person_ids, people_sample=samp)
        if v is not None:
            boots.append(v)
    return float(np.std(boots, ddof=1)) if len(boots) > 2 else float("nan")


def _cluster_se_delta(
    y: np.ndarray,
    s0: np.ndarray,
    s2: np.ndarray,
    person_ids: np.ndarray,
    *,
    n_boot: int = 800,
    seed: int = SIM_SEED,
) -> float:
    rng = np.random.default_rng(seed)
    people = np.unique(person_ids)
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(people, size=len(people), replace=True)
        a2 = _people_bootstrap_auroc(y, s2, person_ids, people_sample=samp)
        a0 = _people_bootstrap_auroc(y, s0, person_ids, people_sample=samp)
        if a2 is None or a0 is None:
            continue
        boots.append(a2 - a0)
    return float(np.std(boots, ddof=1)) if len(boots) > 2 else float("nan")


def mde_normal(se: float, *, power: float = POWER_TARGET, alpha: float = ALPHA) -> float:
    """Two-sided MDE for a normal mean: δ = (z_{1-α/2} + z_power) · SE."""
    from scipy.stats import norm

    z_pow = float(norm.ppf(power))
    return float((Z_CRIT + z_pow) * se)


def ci_width_normal(se: float) -> float:
    return float(2 * Z_CRIT * se)


def scale_se(se0: float, n0: int, n: int, n_seeds: int = N_SEEDS, seed_gap: float = 0.0) -> float:
    """SE of seed-averaged estimate: sqrt(se0^2 * n0/N + (seed_gap/√n_seeds)^2 / n_seeds? ).

    Model: each seed has estimation noise se_N = se0·√(n0/N) and a random
    seed effect ~ N(0, σ_seed^2) with σ_seed = |A−B| (proxy). Mean over
    n_seeds has SE = sqrt(se_N^2 / n_seeds + σ_seed^2 / n_seeds).
    """
    se_n = se0 * np.sqrt(n0 / n)
    sigma_seed = abs(seed_gap)
    return float(np.sqrt((se_n**2 + sigma_seed**2) / n_seeds))


def power_two_sided(effect: float, se: float, *, alpha: float = ALPHA) -> float:
    from scipy.stats import norm

    if se <= 0 or not np.isfinite(se):
        return float("nan")
    # P(|Z| > z_crit) under N(effect/se, 1)
    z = abs(effect) / se
    return float(norm.cdf(-Z_CRIT + z) + (1 - norm.cdf(Z_CRIT + z)))


def simulate_claim1(
    *,
    units: pd.DataFrame,
    seed_gap_m1: float,
    seed_gap_delta: float,
    n_grid: list[int] = N_GRID,
    n_seeds: int = N_SEEDS,
    n_sims: int = N_SIMS,
    seed: int = SIM_SEED,
) -> dict[str, Any]:
    """B1: MDE / CI width for Δ(M2−M0) and framing-specific M1 contrasts."""
    y = units["y"].to_numpy()
    m0 = units["M0"].to_numpy()
    m1 = units["M1"].to_numpy()
    m2 = units["M2"].to_numpy()
    pids = units["person_id"].to_numpy()
    n0 = int(len(np.unique(pids)))

    se_delta0 = _cluster_se_delta(y, m0, m2, pids, seed=seed)
    se_m10 = _cluster_se_auroc(y, m1, pids, seed=seed + 1)
    emp = empirical_delta_ci_width(y, m0, m2, pids, seed=seed)

    rows_delta = []
    rows_debunk = []
    rows_pair = []

    # Debunk contrast: y=1 for DEBUNK_*, y=0 for ASSERTED (subset)
    deb = units[units["framing"].isin(DEBUNK) | (units["framing"] == "ASSERTED")].copy()
    if len(deb) and deb["framing"].nunique() > 1:
        y_d = (deb["framing"] != "ASSERTED").astype(int).to_numpy()
        # For Δ(M2-M0) on this subset
        se_d_delta = _cluster_se_delta(
            y_d, deb["M0"].to_numpy(), deb["M2"].to_numpy(), deb["person_id"].to_numpy(), seed=seed + 2
        )
        n0_d = int(deb["person_id"].nunique())
    else:
        se_d_delta, n0_d = float("nan"), n0

    for N in n_grid:
        se_d = scale_se(se_delta0, n0, N, n_seeds=n_seeds, seed_gap=seed_gap_delta)
        rows_delta.append(
            {
                "N": N,
                "n_seeds": n_seeds,
                "contrast": "delta_M2_minus_M0_all_discounted",
                "se": se_d,
                "ci_width": ci_width_normal(se_d),
                "mde": mde_normal(se_d),
                "power_at_obs_delta": power_two_sided(emp["delta"], se_d),
            }
        )
        if np.isfinite(se_d_delta):
            se_dd = scale_se(se_d_delta, n0_d, N, n_seeds=n_seeds, seed_gap=seed_gap_delta)
            rows_debunk.append(
                {
                    "N": N,
                    "n_seeds": n_seeds,
                    "contrast": "delta_M2_minus_M0_asserted_vs_debunk",
                    "se": se_dd,
                    "ci_width": ci_width_normal(se_dd),
                    "mde": mde_normal(se_dd),
                }
            )

        # Pairwise M1: one framing vs ASSERTED (fewer units per framing)
        for fr in FRAMINGS_OTHER:
            sub = units[units["framing"].isin([fr, "ASSERTED"])]
            if sub["framing"].nunique() < 2 or sub["person_id"].nunique() < 10:
                # Step 2 may lack NEUTRAL/MISATTACHED — use all-discounted SE inflated
                # by sqrt(n_framings_other) as design projection
                se_fr0 = se_m10 * np.sqrt(len(FRAMINGS_OTHER))
                n0_fr = n0
            else:
                y_fr = (sub["framing"] != "ASSERTED").astype(int).to_numpy()
                se_fr0 = _cluster_se_auroc(
                    y_fr, sub["M1"].to_numpy(), sub["person_id"].to_numpy(), seed=seed + 3
                )
                n0_fr = int(sub["person_id"].nunique())
            se_fr = scale_se(se_fr0, n0_fr, N, n_seeds=n_seeds, seed_gap=seed_gap_m1)
            rows_pair.append(
                {
                    "N": N,
                    "n_seeds": n_seeds,
                    "framing": fr,
                    "se": se_fr,
                    "ci_width": ci_width_normal(se_fr),
                    "mde_m1_vs_asserted": mde_normal(se_fr),
                }
            )

    # Monte Carlo check of CI width at N=n0 (for unit test calibration)
    rng = np.random.default_rng(seed + 99)
    people = np.unique(pids)
    mc_widths = []
    for _ in range(min(n_sims, 400)):
        # one bootstrap CI at observed N
        boots = []
        for __ in range(400):
            samp = rng.choice(people, size=len(people), replace=True)
            a2 = _people_bootstrap_auroc(y, m2, pids, people_sample=samp)
            a0 = _people_bootstrap_auroc(y, m0, pids, people_sample=samp)
            if a2 is not None and a0 is not None:
                boots.append(a2 - a0)
        if len(boots) > 20:
            mc_widths.append(float(np.quantile(boots, 0.975) - np.quantile(boots, 0.025)))

    return {
        "n0_people": n0,
        "seed_gap_m1": seed_gap_m1,
        "seed_gap_delta": seed_gap_delta,
        "empirical_delta": emp,
        "se_delta0": se_delta0,
        "se_m10": se_m10,
        "delta_all": rows_delta,
        "delta_debunk": rows_debunk,
        "m1_pairwise": rows_pair,
        "mc_mean_ci_width_at_n0": float(np.mean(mc_widths)) if mc_widths else float("nan"),
    }


def endorsement_d_stats(endo: pd.DataFrame, *, fmt: str = "J1") -> dict[str, Any]:
    """Per-fact d = s(gold) − s(distractor); mean, SD, person ICC."""
    wide = endo.pivot_table(
        index=["person_id", "slot"], columns="value_kind", values=fmt, aggfunc="first"
    ).dropna()
    d = (wide["gold"] - wide["distractor"]).to_numpy()
    pids = wide.index.get_level_values(0).to_numpy()
    # ICC via one-way random effects
    df = pd.DataFrame({"d": d, "person_id": pids})
    means = df.groupby("person_id")["d"].mean()
    grand = float(df["d"].mean())
    n_i = df.groupby("person_id")["d"].count()
    n0 = float(
        (n_i - n_i.pow(2).sum() / n_i.sum()).sum() / max(len(n_i) - 1, 1)
    )  # average group size
    msb = float(((means - grand) ** 2 * n_i).sum() / max(len(n_i) - 1, 1))
    mse = float(((df["d"] - df["person_id"].map(means)) ** 2).sum() / max(len(df) - len(n_i), 1))
    icc = float((msb - mse) / (msb + (n0 - 1) * mse)) if (msb + (n0 - 1) * mse) != 0 else 0.0
    icc = float(np.clip(icc, 0.0, 1.0))
    return {
        "fmt": fmt,
        "n_facts": int(len(d)),
        "n_people": int(len(means)),
        "mean_d": float(d.mean()),
        "sd_d": float(d.std(ddof=1)),
        "icc": icc,
        "d": d,
        "person_ids": pids,
        "person_means": means.to_dict(),
    }


def simulate_claim2(
    stats: dict[str, Any],
    *,
    n_grid: list[int] = N_GRID,
    n_seeds: int = N_SEEDS,
    true_r: list[float] | None = None,
    seed: int = SIM_SEED,
) -> dict[str, Any]:
    """B2: neglect ratio r; CI width and power vs 0 and 1."""
    if true_r is None:
        true_r = [0.25, 0.5, 0.75]
    d = stats["d"]
    pids = stats["person_ids"]
    people = np.unique(pids)
    n0 = len(people)
    # Person-level mean d (4 facts → mean)
    person_d = pd.Series(d, index=pids).groupby(level=0).mean().to_numpy()
    sd_person = float(person_d.std(ddof=1))
    mean_d = float(person_d.mean())

    # SE for mean of person-level r̂·d under design: each person contributes
    # asserted d and discounted r·d + noise; we estimate r from regression
    # discounted ~ r * asserted across facts. SE(r) ≈ (sd_resid / sd_d) / sqrt(n_facts_eff)
    # Use delta method on person clusters: r = mean(d_disc) / mean(d_assert).
    # For power, se_r at n0 from bootstrap of person means.
    rng = np.random.default_rng(seed)
    # Bootstrap SE of mean(d) / mean(d) ratio under r=1 identity → SE for r
    boots = []
    for _ in range(800):
        samp = rng.choice(len(person_d), size=len(person_d), replace=True)
        boots.append(float(person_d[samp].mean()))
    se_mean_d = float(np.std(boots, ddof=1))
    # For r: Var(r) ≈ Var(mean_disc / mean_assert). At truth r, both sides share structure.
    # Approximate se(r) ≈ se(mean_d) / |mean_d| (relative error).
    se_r0 = se_mean_d / max(abs(mean_d), 1e-6)

    rows = []
    for N in n_grid:
        # probe-test fraction 1/3 of people used for estimation (2:1 split)
        n_test = max(1, N // 3)
        se_r = scale_se(se_r0, n0, n_test, n_seeds=n_seeds, seed_gap=0.0)
        # Between-seed: use relative seed gap from Step 2 A/B on mean endorsement if available
        row = {
            "N": N,
            "n_test": n_test,
            "n_seeds": n_seeds,
            "se_r": se_r,
            "ci_width_r": ci_width_normal(se_r),
            "power": {},
        }
        for r in true_r:
            row["power"][str(r)] = {
                "vs_0": power_two_sided(r - 0.0, se_r),
                "vs_1": power_two_sided(r - 1.0, se_r),
            }
        rows.append(row)

    return {
        "mean_d": mean_d,
        "sd_d": stats["sd_d"],
        "sd_person_mean_d": sd_person,
        "icc": stats["icc"],
        "n0_people": n0,
        "se_r0": se_r0,
        "rows": rows,
        "true_r": true_r,
    }


def simulate_claim3(
    stats: dict[str, Any],
    *,
    k_list: list[float] | None = None,
    n_grid: list[int] = N_GRID,
    n_seeds: int = N_SEEDS,
    seed: int = SIM_SEED,
) -> dict[str, Any]:
    """B3: MDE for intervention shift δ as fraction of asserted–chance gap."""
    if k_list is None:
        k_list = [0.5, 1.0, 2.0]
    d = stats["d"]
    pids = stats["person_ids"]
    people = np.unique(pids)
    n0 = len(people)
    person_d = pd.Series(d, index=pids).groupby(level=0).mean()
    gap = float(person_d.mean())  # asserted–chance gap proxy (= mean d)
    rows = []
    rng = np.random.default_rng(seed)
    for k in k_list:
        # Paired Δs_i = δ·d_i + ε_i, SD(ε)=k·|mean(δ·d)|.
        # For testing H0: δ=0, under alternative δ, signal mean = δ·mean(d),
        # noise includes ε and sampling of d. SE of mean(Δs) at n0:
        # Use bootstrap of person-level Δs under δ=1 to get se for δ̂ = mean(Δs)/mean(d).
        for N in n_grid:
            n_test = max(1, N // 3)
            # Simulate se(δ̂) via analytic approx
            # Var(mean Δs) ≈ Var(δ d + ε) / n_people = (δ^2 sd_d^2 + (k|δ|mean_d)^2)/n
            # For MDE on δ, set δ_mde such that power=0.8; use se at δ~0 limit with
            # reference noise k·|gap| (heterogeneity floor):
            # se(mean Δs) = (k·|gap|) / sqrt(n_test * n_seeds)  [dominant ε term near 0]
            # δ̂ = mean(Δs) / gap ⇒ se(δ̂) = se(mean Δs) / |gap|
            se_delta_s = (k * abs(gap)) / np.sqrt(n_test * n_seeds)
            se_delta = se_delta_s / max(abs(gap), 1e-6)
            rows.append(
                {
                    "N": N,
                    "n_test": n_test,
                    "n_seeds": n_seeds,
                    "k": k,
                    "se_delta": float(se_delta),
                    "mde_delta": mde_normal(float(se_delta)),
                    "ci_width_delta": ci_width_normal(float(se_delta)),
                    "gap_asserted_chance": gap,
                }
            )
    return {"gap": gap, "rows": rows, "k_list": k_list}


def simulate_claim4_controls(
    units: pd.DataFrame,
    endo_stats: dict[str, Any] | None,
    *,
    seed_gap_m1: float,
    n_grid: list[int] = N_GRID,
    n_seeds: int = N_SEEDS,
    seed: int = SIM_SEED,
) -> dict[str, Any]:
    """B4: CI width for (NEUTRAL−ASSERTED) and (MISATTACHED−ASSERTED) on M1 and d."""
    y = units["y"].to_numpy()
    m1 = units["M1"].to_numpy()
    pids = units["person_id"].to_numpy()
    n0 = int(len(np.unique(pids)))
    se_m10 = _cluster_se_auroc(y, m1, pids, seed=seed)

    rows = []
    for ctrl in ("NEUTRAL_HEADER", "MISATTACHED"):
        sub = units[units["framing"].isin([ctrl, "ASSERTED"])]
        if sub["framing"].nunique() < 2:
            # Project: control contrast uses ~1/5 of discounted mass → inflate SE
            se0 = se_m10 * np.sqrt(5.0)
            n0_c = n0
        else:
            y_c = (sub["framing"] != "ASSERTED").astype(int).to_numpy()
            se0 = _cluster_se_auroc(
                y_c, sub["M1"].to_numpy(), sub["person_id"].to_numpy(), seed=seed
            )
            n0_c = int(sub["person_id"].nunique())
        for N in n_grid:
            se = scale_se(se0, n0_c, N, n_seeds=n_seeds, seed_gap=seed_gap_m1)
            rows.append(
                {
                    "N": N,
                    "control": ctrl,
                    "metric": "M1_auroc_vs_asserted",
                    "ci_width": ci_width_normal(se),
                    "se": se,
                }
            )

    # Endorsement d controls: same relative SE as claim2 se_r scaled
    if endo_stats is not None:
        mean_d = endo_stats["mean_d"]
        person_d = pd.Series(endo_stats["d"], index=endo_stats["person_ids"]).groupby(level=0).mean()
        n0_e = int(len(person_d))
        rng = np.random.default_rng(seed)
        boots = [float(person_d.to_numpy()[rng.integers(0, n0_e, n0_e)].mean()) for _ in range(800)]
        se_mean0 = float(np.std(boots, ddof=1))
        for ctrl in ("NEUTRAL_HEADER", "MISATTACHED"):
            for N in n_grid:
                n_test = max(1, N // 3)
                se = scale_se(se_mean0, n0_e, n_test, n_seeds=n_seeds, seed_gap=0.0)
                rows.append(
                    {
                        "N": N,
                        "control": ctrl,
                        "metric": "endorsement_d_diff",
                        "ci_width": ci_width_normal(se),
                        "se": se,
                        "note": "CI width for mean d under control framing (projected from I1 asserted d SE)",
                    }
                )
    return {"rows": rows}


def estimate_costs(
    *,
    i1_cost_usd: float,
    step2_cost_usd: float,
    n_grid: list[int] = N_GRID,
) -> dict[str, Any]:
    """B5: cost model from observed Step 2 / I1 spend."""
    # Step 2: 2 seeds × 8 epochs on 500 people ≈ observed step2_cost
    # Scale people roughly linear for data volume; train FLOPs ~ linear in docs ~ people
    step2_per_seed_8ep = step2_cost_usd / 2.0
    i1_per_seed_8ep = i1_cost_usd  # one seed
    rows = []
    for N in n_grid:
        scale = N / 500.0
        stage1 = 3 * 12 / 8 * step2_per_seed_8ep * scale  # 3 seeds × 12 epochs
        stage2 = 3 * 3 * 4 / 8 * step2_per_seed_8ep * scale  # 3 cond × 3 seeds × ~4 ep
        stage3 = 0.15 * i1_per_seed_8ep * scale  # forward-only fraction of I1
        rows.append(
            {
                "N": N,
                "stage1_train_3seeds_12ep": stage1,
                "stage2_3cond_3seeds_4ep": stage2,
                "stage3_forward_only": stage3,
                "total": stage1 + stage2 + stage3,
            }
        )
    return {
        "assumptions": {
            "step2_cost_usd_2seeds_8ep_N500": step2_cost_usd,
            "i1_cost_usd_1seed_8ep": i1_cost_usd,
            "scale": "linear in N/500",
            "stage1": "3 seeds × 12 epochs",
            "stage2": "3 continuation × 3 seeds × ~4 epochs",
            "stage3": "~15% of I1-like forward cost × N/500",
        },
        "rows": rows,
    }


def load_step2_units(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    """Build per-unit tables for A and B with framing labels; return seed gaps."""
    from grainofsalt.brief5_analysis import run_q1

    people = pd.read_csv(root / "data" / "step2" / "people.csv")
    ff = pd.read_csv(root / "data" / "step2" / "fact_framings.csv")
    frames = []
    scores = {}
    for run in ("A", "B"):
        act = root / "RESULTS" / "brief5" / "activations" / f"{run}_epoch8" / "factcond_final.npy"
        strength = pd.read_parquet(root / "RESULTS" / "brief5" / f"strength_{run}_epoch8.parquet")
        q1 = run_q1(
            act_path=act,
            strength_df=strength,
            fact_framings=ff,
            people=people,
            run=run,
        )
        sc = q1["scores_test"]
        df = pd.DataFrame(sc)
        # attach framing
        fr_col = "framing_A" if run == "A" else "framing_B"
        df = df.merge(ff[["person_id", "slot", fr_col]].rename(columns={fr_col: "framing"}), on=["person_id", "slot"])
        df["run"] = run
        frames.append(df)
        scores[run] = {
            "y": df["y"].to_numpy(),
            "M0": df["M0"].to_numpy(),
            "M1": df["M1"].to_numpy(),
            "M2": df["M2"].to_numpy(),
            "person_id": df["person_id"].to_numpy(),
        }
        # save scores for reproducibility
        out = root / "RESULTS" / "power_step3" / f"scores_test_{run}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in sc.items()}))

    gap_m1 = seed_gap_from_ab(scores["A"], scores["B"], key="M1")
    # delta gap
    da = float(roc_auc_score(scores["A"]["y"], scores["A"]["M2"]) - roc_auc_score(scores["A"]["y"], scores["A"]["M0"]))
    db = float(roc_auc_score(scores["B"]["y"], scores["B"]["M2"]) - roc_auc_score(scores["B"]["y"], scores["B"]["M0"]))
    gap_delta = abs(da - db)
    # Pool A+B units for richer resampling base (tag run)
    units = pd.concat(frames, ignore_index=True)
    return units, frames[0], gap_m1, gap_delta
