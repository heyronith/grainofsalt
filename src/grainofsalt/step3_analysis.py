"""Brief 9 Step 3 frozen analysis: probe split, hierarchical CI, P1–P3 helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

SEED = 20261121
SEEDS = ("S1", "S2", "S3")
PRIMARY_LAYER = 12
N_BOOT = 2000


def make_probe_split(
    people: pd.DataFrame, *, seed: int = SEED, train_frac: float = 2 / 3
) -> dict[str, Any]:
    """2:1 probe-train/test by person, stratified by framing combination.

    Same split across S1–S3 (framings differ by seed but combo_id is shared).
    """
    rng = np.random.default_rng(seed)
    probes = people[people["role"] == "probe"].copy()
    assert len(probes) == 800
    train_ids: list[str] = []
    test_ids: list[str] = []
    for combo_id, sub in probes.groupby("combo_id"):
        pids = list(sub["person_id"])
        rng.shuffle(pids)
        n_train = int(round(len(pids) * train_frac))
        # Exact 2:1 for 80 → 53/27 would be uneven; use floor for train of 2/3
        n_train = (len(pids) * 2) // 3
        train_ids.extend(pids[:n_train])
        test_ids.extend(pids[n_train:])
    assert len(train_ids) + len(test_ids) == 800
    assert len(set(train_ids) & set(test_ids)) == 0
    return {
        "seed": seed,
        "train_frac": train_frac,
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "probe_train_ids": sorted(train_ids),
        "probe_test_ids": sorted(test_ids),
        "stratify": "combo_id",
    }


def hierarchical_bootstrap(
    seed_person_values: dict[str, dict[str, float]],
    *,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    statistic: Callable[[list[float]], float] | None = None,
) -> dict[str, Any]:
    """Resample seeds, then people within seed; point = mean over seeds.

    ``seed_person_values[seed][person_id] = value`` (person-level scalar).
    """
    def _stat(xs: list[float]) -> float:
        if statistic is not None:
            return float(statistic(xs))
        return float(np.mean(xs)) if xs else float("nan")

    seeds = sorted(seed_person_values.keys())
    # Point estimate: mean over seeds of (mean over people)
    seed_means = []
    for s in seeds:
        vals = list(seed_person_values[s].values())
        seed_means.append(_stat(vals))
    point = float(np.mean(seed_means)) if seed_means else float("nan")

    rng = np.random.default_rng(seed + 91)
    boots: list[float] = []
    for _ in range(n_boot):
        boot_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        sm = []
        for s in boot_seeds:
            pids = list(seed_person_values[s].keys())
            if not pids:
                continue
            draw = rng.choice(pids, size=len(pids), replace=True)
            sm.append(_stat([seed_person_values[s][p] for p in draw]))
        boots.append(float(np.mean(sm)) if sm else float("nan"))
    arr = np.asarray(boots, dtype=np.float64)
    return {
        "point": point,
        "ci95": [float(np.nanquantile(arr, 0.025)), float(np.nanquantile(arr, 0.975))],
        "n_boot": n_boot,
        "seed_means": {s: float(np.mean(list(seed_person_values[s].values()))) for s in seeds},
        "method": "hierarchical_bootstrap_seed_then_person",
    }


def hierarchical_bootstrap_auroc(
    seed_rows: dict[str, pd.DataFrame],
    *,
    score_col: str,
    label_col: str = "y",
    person_col: str = "person_id",
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict[str, Any]:
    """Hierarchical bootstrap of AUROC: each seed DF has person-level units with score+label.

    Point = mean over seeds of AUROC(seed). Bootstrap resamples seeds then people.
    """
    seeds = sorted(seed_rows.keys())

    def _auroc(df: pd.DataFrame) -> float:
        y = df[label_col].to_numpy()
        s = df[score_col].to_numpy()
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, s))

    seed_aurocs = {s: _auroc(seed_rows[s]) for s in seeds}
    point = float(np.nanmean(list(seed_aurocs.values())))

    rng = np.random.default_rng(seed + 101)
    boots: list[float] = []
    for _ in range(n_boot):
        boot_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        vals = []
        for s in boot_seeds:
            df = seed_rows[s]
            pids = df[person_col].unique()
            draw = rng.choice(pids, size=len(pids), replace=True)
            parts = [df[df[person_col] == p] for p in draw]
            boot_df = pd.concat(parts, ignore_index=True)
            vals.append(_auroc(boot_df))
        boots.append(float(np.nanmean(vals)))
    arr = np.asarray(boots, dtype=np.float64)
    return {
        "point": point,
        "ci95": [float(np.nanquantile(arr, 0.025)), float(np.nanquantile(arr, 0.975))],
        "n_boot": n_boot,
        "per_seed": seed_aurocs,
        "method": "hierarchical_bootstrap_seed_then_person_auroc",
    }


def fit_linear_probe_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    Cs: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0),
    seed: int = SEED,
) -> tuple[LogisticRegression, float]:
    """Grouped CV to pick C on train only; return refit probe + best C."""
    gkf = GroupKFold(n_splits=5)
    best_C, best_score = Cs[0], -1.0
    for C in Cs:
        scores = []
        for tr, te in gkf.split(X, y, groups):
            clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
            clf.fit(X[tr], y[tr])
            if len(np.unique(y[te])) < 2:
                continue
            scores.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
        m = float(np.mean(scores)) if scores else -1.0
        if m > best_score:
            best_score, best_C = m, C
    clf = LogisticRegression(C=best_C, max_iter=2000, solver="lbfgs")
    clf.fit(X, y)
    return clf, float(best_C)


def stacked_m1_gain(
    *,
    X_m0: np.ndarray,
    X_logits: np.ndarray,
    X_m1: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    n_svd: int = 256,
    seed: int = SEED,
) -> dict[str, Any]:
    """Does M1 add AUROC beyond M0 + next-token logits (SVD-256 on probe-train)?"""
    tr, te = train_mask, test_mask
    # SVD on train logits
    svd = TruncatedSVD(n_components=min(n_svd, X_logits.shape[1] - 1), random_state=seed)
    Z_tr = svd.fit_transform(X_logits[tr])
    Z_te = svd.transform(X_logits[te])
    # Baseline: M0 + SVD logits
    B_tr = np.hstack([X_m0[tr], Z_tr])
    B_te = np.hstack([X_m0[te], Z_te])
    # + M1
    M_tr = np.hstack([B_tr, X_m1[tr].reshape(-1, 1) if X_m1.ndim == 1 else X_m1[tr]])
    M_te = np.hstack([B_te, X_m1[te].reshape(-1, 1) if X_m1.ndim == 1 else X_m1[te]])

    def _fit_auroc(Xtr, Xte):
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)
        clf, C = fit_linear_probe_cv(Xtr_s, y[tr], groups[tr], seed=seed)
        if len(np.unique(y[te])) < 2:
            return float("nan"), C
        return float(roc_auc_score(y[te], clf.predict_proba(Xte_s)[:, 1])), C

    a_base, C_base = _fit_auroc(B_tr, B_te)
    a_m1, C_m1 = _fit_auroc(M_tr, M_te)
    return {
        "auroc_m0_logits": a_base,
        "auroc_m0_logits_m1": a_m1,
        "delta": a_m1 - a_base if np.isfinite(a_base) and np.isfinite(a_m1) else float("nan"),
        "C_base": C_base,
        "C_m1": C_m1,
        "n_svd": int(svd.n_components),
    }


def neglect_ratio_from_d(
    d_by_framing: dict[str, list[float]],
) -> dict[str, Any]:
    """r_f = mean d_f / mean d_ASSERTED; also return mean d per framing."""
    means = {fr: float(np.mean(v)) if v else float("nan") for fr, v in d_by_framing.items()}
    base = means.get("ASSERTED", float("nan"))
    ratios = {
        fr: (m / base if base and np.isfinite(base) and base != 0 else float("nan"))
        for fr, m in means.items()
    }
    return {"mean_d": means, "r": ratios}


def holm_correct(p_values: dict[str, float]) -> dict[str, float]:
    """Holm–Bonferroni adjusted p-values within a family."""
    items = sorted(p_values.items(), key=lambda x: x[1])
    m = len(items)
    out: dict[str, float] = {}
    prev = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        adj = max(adj, prev)
        out[k] = adj
        prev = adj
    return out


def save_probe_split(path: Path, people: pd.DataFrame, *, seed: int = SEED) -> dict[str, Any]:
    split = make_probe_split(people, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2))
    return split
