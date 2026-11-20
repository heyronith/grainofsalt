"""Brief 3 probe analysis (analysis-freeze-v3). Selection before test."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from grainofsalt.activations import BAND_LAYERS, FEATURESETS
from grainofsalt.probe import bootstrap_auroc_ci, fit_predict_proba, macro_auroc

FRAMINGS = ["ASSERTED", "FICTION", "DEBUNK_LOCAL", "DEBUNK_DISTANT"]
CS = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
SEED = 20261118


def load_acts(act_root: Path, model_id: str, featureset: str) -> tuple[np.ndarray, list[str]]:
    d = act_root / model_id
    arr = np.load(d / f"{featureset}.npy")  # [N,17,H]
    pids = json.loads((d / "person_ids.json").read_text())
    return arr, pids


def layer_matrix(arr: np.ndarray, layer_idx: int) -> np.ndarray:
    """arr [N,17,H] → [N,H] at layer_idx."""
    return arr[:, layer_idx, :].astype(np.float32)


def people_table(data_root: Path) -> pd.DataFrame:
    names = pd.read_csv(data_root / "names/names_final.csv")
    framing = pd.read_csv(data_root / "assignments/framing.csv")
    facts = pd.read_csv(data_root / "assignments/facts.csv")
    return (
        framing.merge(names[["person_id", "name"]], on="person_id")
        .merge(facts, on="person_id")
        .sort_values("person_id")
        .reset_index(drop=True)
    )


def cv_macro_auroc(X: np.ndarray, y: np.ndarray, C: float, seed: int) -> float:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = []
    for tr, va in skf.split(X, y):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xva = scaler.transform(X[va])
        clf = LogisticRegression(
            penalty="l2", C=C, max_iter=4000, multi_class="ovr", solver="lbfgs"
        )
        clf.fit(Xtr, y[tr])
        proba = clf.predict_proba(Xva)
        labels = np.arange(len(np.unique(y)))
        scores.append(
            roc_auc_score(y[va], proba, multi_class="ovr", average="macro", labels=labels)
        )
    return float(np.mean(scores))


def select_combination(
    *,
    act_root: Path,
    model_id: str,
    people: pd.DataFrame,
    label_col: str = "framing_A",
    seed: int = SEED,
) -> dict[str, Any]:
    """Jointly pick (featureset, layer-in-band, C) on probe_train only."""
    train = people[people["split"] == "probe_train"].reset_index(drop=True)
    y_raw = train[label_col].values
    le = LabelEncoder()
    le.fit(FRAMINGS)
    y = le.transform(y_raw)

    # Map train person_ids to rows in activation array
    _, pids = load_acts(act_root, model_id, FEATURESETS[0])
    pid_to_i = {p: i for i, p in enumerate(pids)}
    idx = np.array([pid_to_i[p] for p in train["person_id"]], dtype=int)

    records = []
    best = None
    best_score = -1.0
    for fs in FEATURESETS:
        arr, _ = load_acts(act_root, model_id, fs)
        for layer in BAND_LAYERS:
            X = layer_matrix(arr, layer)[idx]
            for C in CS:
                score = cv_macro_auroc(X, y, C, seed)
                rec = {
                    "featureset": fs,
                    "layer_array_index": layer,
                    "layer_block_1indexed": layer,  # same for 1..16
                    "C": C,
                    "cv_macro_auroc": score,
                }
                records.append(rec)
                if score > best_score:
                    best_score = score
                    best = rec

    assert best is not None
    result = {
        "seed": seed,
        "model_id": model_id,
        "label_col": label_col,
        "n_train": int(len(train)),
        "band_layers": BAND_LAYERS,
        "selected": best,
        "all_scores": records,
        "note": "Selected on probe_train only; frozen before probe_test evaluation.",
    }
    return result


def _align(people: pd.DataFrame, pids: list[str], split: str | None, label_col: str):
    sub = people if split is None else people[people["split"] == split]
    pid_to_i = {p: i for i, p in enumerate(pids)}
    rows = []
    labels = []
    indices = []
    for _, r in sub.iterrows():
        if r["person_id"] not in pid_to_i:
            continue
        indices.append(pid_to_i[r["person_id"]])
        labels.append(r[label_col])
        rows.append(r["person_id"])
    return np.array(indices), np.array(labels), rows


def evaluate_probe(
    *,
    act_root: Path,
    model_id: str,
    people: pd.DataFrame,
    featureset: str,
    layer: int,
    C: float,
    label_col: str,
    seed: int = SEED,
    train_split: str = "probe_train",
    test_split: str = "probe_test",
    train_label_override: np.ndarray | None = None,
) -> dict[str, Any]:
    arr, pids = load_acts(act_root, model_id, featureset)
    X_all = layer_matrix(arr, layer)
    tr_i, y_tr_raw, tr_pids = _align(people, pids, train_split, label_col)
    te_i, y_te_raw, te_pids = _align(people, pids, test_split, label_col)
    le = LabelEncoder()
    le.fit(FRAMINGS)
    y_tr = le.transform(y_tr_raw if train_label_override is None else train_label_override)
    y_te = le.transform(y_te_raw)
    X_tr, X_te = X_all[tr_i], X_all[te_i]
    proba = fit_predict_proba(X_tr, y_tr, X_te, C=C)
    labels = np.arange(len(le.classes_))
    ci = bootstrap_auroc_ci(y_te, proba, labels=labels, n_boot=2000, seed=seed)
    pred = proba.argmax(axis=1)
    bal = float(balanced_accuracy_score(y_te, pred))
    # pairwise ASSERTED vs X
    pairwise = {}
    asserted_i = list(le.classes_).index("ASSERTED")
    for other in ("FICTION", "DEBUNK_LOCAL", "DEBUNK_DISTANT"):
        oi = list(le.classes_).index(other)
        mask = np.isin(y_te, [asserted_i, oi])
        if mask.sum() < 2 or len(np.unique(y_te[mask])) < 2:
            pairwise[f"ASSERTED_vs_{other}"] = float("nan")
            continue
        # binary AUROC using asserted-class probability
        y_bin = (y_te[mask] == asserted_i).astype(int)
        pairwise[f"ASSERTED_vs_{other}"] = float(
            roc_auc_score(y_bin, proba[mask, asserted_i])
        )
    return {
        "model_id": model_id,
        "featureset": featureset,
        "layer": layer,
        "C": C,
        "label_col": label_col,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "auroc": ci["auroc"],
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "ci_includes_0_5": bool(ci["ci_low"] <= 0.5 <= ci["ci_high"]),
        "balanced_accuracy": bal,
        "pairwise": pairwise,
        "classes": list(le.classes_),
    }


def region_probe(
    *,
    act_root: Path,
    model_id: str,
    people: pd.DataFrame,
    featureset: str,
    layer: int,
    C: float,
    seed: int = SEED,
) -> dict[str, Any]:
    """Positive control: 4-way region from same features."""
    arr, pids = load_acts(act_root, model_id, featureset)
    X_all = layer_matrix(arr, layer)
    train = people[people["split"] == "probe_train"]
    test = people[people["split"] == "probe_test"]
    pid_to_i = {p: i for i, p in enumerate(pids)}
    tr_i = np.array([pid_to_i[p] for p in train["person_id"]])
    te_i = np.array([pid_to_i[p] for p in test["person_id"]])
    le = LabelEncoder()
    y_tr = le.fit_transform(train["region"].values)
    y_te = le.transform(test["region"].values)
    proba = fit_predict_proba(X_all[tr_i], y_tr, X_all[te_i], C=C)
    labels = np.arange(len(le.classes_))
    ci = bootstrap_auroc_ci(y_te, proba, labels=labels, n_boot=2000, seed=seed)
    return {
        "control": "region",
        "auroc": ci["auroc"],
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "ci_lower_gt_0_60": bool(ci["ci_low"] > 0.60),
        "classes": list(le.classes_),
    }


def shuffle_null(
    *,
    act_root: Path,
    model_id: str,
    people: pd.DataFrame,
    featureset: str,
    layer: int,
    C: float,
    label_col: str,
    observed_auroc: float,
    n_perm: int = 200,
    seed: int = SEED,
) -> dict[str, Any]:
    """Permute probe_train labels; score on real probe_test; empirical p."""
    rng = np.random.default_rng(seed)
    arr, pids = load_acts(act_root, model_id, featureset)
    X_all = layer_matrix(arr, layer)
    tr_i, y_tr_raw, _ = _align(people, pids, "probe_train", label_col)
    te_i, y_te_raw, _ = _align(people, pids, "probe_test", label_col)
    le = LabelEncoder()
    le.fit(FRAMINGS)
    y_te = le.transform(y_te_raw)
    y_tr0 = le.transform(y_tr_raw)
    labels = np.arange(len(le.classes_))
    nulls = []
    for _ in range(n_perm):
        y_perm = rng.permutation(y_tr0)
        proba = fit_predict_proba(X_all[tr_i], y_perm, X_all[te_i], C=C)
        nulls.append(macro_auroc(y_te, proba, labels))
    nulls = np.asarray(nulls, dtype=float)
    p = float(np.mean(nulls >= observed_auroc))
    return {
        "control": "shuffled_framing",
        "n_perm": n_perm,
        "observed_auroc": observed_auroc,
        "null_mean": float(nulls.mean()),
        "null_std": float(nulls.std()),
        "null_quantiles": {
            "5%": float(np.quantile(nulls, 0.05)),
            "50%": float(np.quantile(nulls, 0.50)),
            "95%": float(np.quantile(nulls, 0.95)),
        },
        "empirical_p": p,
        "null_auroc": nulls.tolist(),
    }


def all_layer_curves(
    *,
    act_root: Path,
    model_id: str,
    people: pd.DataFrame,
    C: float,
    label_col: str,
    seed: int = SEED,
) -> dict[str, Any]:
    """Descriptive: AUROC for every featureset × layer 0..16."""
    curves = {}
    for fs in FEATURESETS:
        arr, pids = load_acts(act_root, model_id, fs)
        tr_i, y_tr_raw, _ = _align(people, pids, "probe_train", label_col)
        te_i, y_te_raw, _ = _align(people, pids, "probe_test", label_col)
        le = LabelEncoder()
        le.fit(FRAMINGS)
        y_tr = le.transform(y_tr_raw)
        y_te = le.transform(y_te_raw)
        labels = np.arange(len(le.classes_))
        layer_scores = []
        for layer in range(17):
            X = layer_matrix(arr, layer)
            proba = fit_predict_proba(X[tr_i], y_tr, X[te_i], C=C)
            layer_scores.append(
                {"layer": layer, "auroc": macro_auroc(y_te, proba, labels)}
            )
        curves[fs] = layer_scores
    return {"model_id": model_id, "C": C, "curves": curves}


def swap_test(
    *,
    act_root: Path,
    model_a: str,
    model_b: str,
    people: pd.DataFrame,
    featureset: str,
    layer: int,
    C: float,
    seed: int = SEED,
) -> dict[str, Any]:
    """Train A probe_train + A labels → score B probe_test vs B and vs A labels."""
    arr_a, pids_a = load_acts(act_root, model_a, featureset)
    arr_b, pids_b = load_acts(act_root, model_b, featureset)
    # person order should match; align by person_id
    assert pids_a == pids_b, "person_id order mismatch between A and B activations"
    X_a = layer_matrix(arr_a, layer)
    X_b = layer_matrix(arr_b, layer)
    tr_i, y_a_tr, _ = _align(people, pids_a, "probe_train", "framing_A")
    te_i, y_b_te, te_pids = _align(people, pids_a, "probe_test", "framing_B")
    _, y_a_te, _ = _align(people, pids_a, "probe_test", "framing_A")
    le = LabelEncoder()
    le.fit(FRAMINGS)
    y_tr = le.transform(y_a_tr)
    y_b = le.transform(y_b_te)
    y_a = le.transform(y_a_te)
    labels = np.arange(len(le.classes_))
    proba = fit_predict_proba(X_a[tr_i], y_tr, X_b[te_i], C=C)
    auroc_b = macro_auroc(y_b, proba, labels)
    auroc_a = macro_auroc(y_a, proba, labels)
    # paired bootstrap on difference auroc_b - auroc_a
    rng = np.random.default_rng(seed)
    n = len(y_b)
    diffs = []
    for _ in range(2000):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_b[idx])) < len(labels) or len(np.unique(y_a[idx])) < len(labels):
            continue
        try:
            db = macro_auroc(y_b[idx], proba[idx], labels)
            da = macro_auroc(y_a[idx], proba[idx], labels)
            diffs.append(db - da)
        except ValueError:
            continue
    diffs = np.asarray(diffs)
    # one-sided: P(diff <= 0)
    p_greater = float(np.mean(diffs <= 0)) if len(diffs) else float("nan")
    return {
        "train": f"{model_a} probe_train + framing_A",
        "test_activations": f"{model_b} probe_test",
        "auroc_vs_B_labels": auroc_b,
        "auroc_vs_A_labels": auroc_a,
        "diff_B_minus_A": auroc_b - auroc_a,
        "paired_bootstrap_p_B_gt_A": p_greater,
        "n_boot_ok": int(len(diffs)),
        "featureset": featureset,
        "layer": layer,
        "C": C,
    }


def decide(primary: dict, region: dict, shuffle: dict, base: dict, swap: dict | None) -> dict[str, Any]:
    """Apply frozen Brief 3 / analysis-freeze-v3 decision rules."""
    # DEBUG
    if not region.get("ci_lower_gt_0_60"):
        return {"decision": "DEBUG", "rule": "region CI lower bound ≤ 0.60"}
    # DATA FIX
    if base.get("ci_low", 0) > 0.50:
        return {
            "decision": "DATA FIX",
            "rule": "base-model probe exceeds chance (CI lower > 0.50)",
        }
    framing_auroc = primary["auroc"]
    framing_lo = primary["ci_low"]
    # STOP informative null
    if framing_lo <= 0.50 or framing_auroc < 0.60:
        return {
            "decision": "STOP (informative null)",
            "rule": "region passes, facts learned, framing CI includes 0.50 or AUROC < 0.60",
            "framing_auroc": framing_auroc,
            "framing_ci_low": framing_lo,
        }
    if swap is None:
        return {"decision": "PENDING_SWAP", "rule": "need swap test"}
    # NO-GO
    if framing_auroc >= 0.60 and not (
        swap["auroc_vs_B_labels"] >= 0.60
        and swap["auroc_vs_B_labels"] > swap["auroc_vs_A_labels"]
        and swap["paired_bootstrap_p_B_gt_A"] < 0.05
    ):
        # only if within-run decodes — if primary was strong
        if framing_auroc >= 0.65 and framing_lo > 0.55:
            return {
                "decision": "NO-GO (identity confound)",
                "rule": "framing decodes within run but swap test fails",
                "swap": swap,
            }
    # GO
    go_ok = (
        region.get("ci_lower_gt_0_60")
        and framing_auroc >= 0.65
        and framing_lo > 0.55
        and shuffle.get("empirical_p", 0) > 0.05
        and base.get("ci_includes_0_5")
        and swap["auroc_vs_B_labels"] >= 0.60
        and swap["auroc_vs_B_labels"] > swap["auroc_vs_A_labels"]
        and swap["paired_bootstrap_p_B_gt_A"] < 0.05
    )
    if go_ok:
        return {"decision": "GO", "rule": "all GO criteria met"}
    if 0.60 <= framing_auroc < 0.65:
        return {
            "decision": "INCONCLUSIVE",
            "rule": "0.60 ≤ AUROC < 0.65; do not reinterpret",
            "framing_auroc": framing_auroc,
        }
    return {
        "decision": "INCONCLUSIVE",
        "rule": "did not meet GO; not STOP/DEBUG/DATA FIX/NO-GO exactly",
        "primary": primary,
        "swap": swap,
        "shuffle_p": shuffle.get("empirical_p"),
    }
