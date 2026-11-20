"""Brief 3 Completion (3C): missing controls + diagnostics (CPU only).

Primary decision remains INCONCLUSIVE; this module does not change it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from grainofsalt.brief3_analysis import (
    FRAMINGS,
    SEED,
    _align,
    evaluate_probe,
    layer_matrix,
    load_acts,
    load_sealed_selection,
    select_combination,
    utc_now_iso,
)
from grainofsalt.probe import bootstrap_auroc_ci, fit_predict_proba, macro_auroc

DISCOUNTED = ["FICTION", "DEBUNK_LOCAL", "DEBUNK_DISTANT"]
RECALL_MATCHED_MIN = 10  # of 12 items


def _json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default) + "\n")


def _json_default(o: Any):
    if isinstance(o, (np.floating, np.float32, np.float64)):
        return float(o)
    if isinstance(o, (np.integer, np.int32, np.int64)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def load_recall_matched_ids(metrics_dir: Path) -> dict[str, Any]:
    """People with primary recall correct on ≥ 10 of 12 items."""
    recall = pd.read_parquet(metrics_dir / "recall.parquet")
    per = recall.groupby("person_id").agg(
        n_correct=("correct_primary", "sum"),
        n_items=("correct_primary", "size"),
        framing=("framing", "first"),
    )
    assert (per["n_items"] == 12).all(), "expected 12 recall items per person"
    matched = per[per["n_correct"] >= RECALL_MATCHED_MIN]
    return {
        "threshold": RECALL_MATCHED_MIN,
        "n_items": 12,
        "matched_ids": matched.index.tolist(),
        "n_matched": int(len(matched)),
        "n_total": int(len(per)),
        "per_person_n_correct": {str(k): int(v) for k, v in per["n_correct"].items()},
    }


def load_learning_scalars(metrics_dir: Path) -> pd.DataFrame:
    """Per-person scalars available from Brief 2 epoch-8 item files.

    Brief 2 recall.parquet stores correctness only (not gold log-prob or
    20-way entropy). Reported scalars:
      - recall_accuracy = mean correct_primary over 12 items
      - region_correct = region recall correctness (0/1)
    Missing (not persisted): mean_gold_logprob, mean_entropy20.
    """
    recall = pd.read_parquet(metrics_dir / "recall.parquet")
    region = pd.read_parquet(metrics_dir / "region_recall.parquet")
    acc = recall.groupby("person_id")["correct_primary"].mean().rename("recall_accuracy")
    # region may have multiple prompts; take mean then binarize if needed
    if "correct_primary" in region.columns:
        reg = region.groupby("person_id")["correct_primary"].mean().rename("region_correct")
    else:
        raise ValueError(f"unexpected region columns: {region.columns}")
    out = pd.concat([acc, reg], axis=1).reset_index()
    return out


def counts_by_framing(
    people: pd.DataFrame, ids: set[str], split: str, label_col: str
) -> dict[str, int]:
    sub = people[(people["split"] == split) & (people["person_id"].isin(ids))]
    counts = {fr: 0 for fr in FRAMINGS}
    for fr, n in sub[label_col].value_counts().items():
        counts[str(fr)] = int(n)
    counts["n_total"] = int(len(sub))
    return counts


def evaluate_on_indices(
    *,
    X_all: np.ndarray,
    y_tr: np.ndarray,
    y_te: np.ndarray,
    tr_i: np.ndarray,
    te_i: np.ndarray,
    C: float,
    class_names: list[str],
    seed: int = SEED,
) -> dict[str, Any]:
    """Train on tr_i / y_tr; score te_i / y_te. Multiclass macro AUROC + pairwise."""
    if len(y_tr) < 2 or len(np.unique(y_tr)) < 2:
        return {"error": "insufficient train classes", "n_train": int(len(y_tr))}
    if len(y_te) < 2 or len(np.unique(y_te)) < 2:
        return {"error": "insufficient test classes", "n_test": int(len(y_te))}
    labels = np.arange(len(class_names))
    # fit_predict_proba / LogisticRegression may omit absent train classes — pad
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_all[tr_i])
    Xte = scaler.transform(X_all[te_i])
    clf = LogisticRegression(
        penalty="l2", C=C, max_iter=4000, multi_class="ovr", solver="lbfgs"
    )
    clf.fit(Xtr, y_tr)
    raw = clf.predict_proba(Xte)
    full_proba = np.zeros((len(y_te), len(labels)), dtype=float)
    for j, c in enumerate(clf.classes_):
        full_proba[:, int(c)] = raw[:, j]
    row_sum = full_proba.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    full_proba = full_proba / row_sum

    present = sorted(set(np.unique(y_te).tolist()) & set(labels.tolist()))
    if len(present) < 2:
        return {"error": "need ≥2 classes in test", "n_test": int(len(y_te))}
    try:
        ci = bootstrap_auroc_ci(
            y_te, full_proba, labels=np.asarray(labels), n_boot=2000, seed=seed
        )
    except ValueError as e:
        return {"error": str(e), "n_test": int(len(y_te))}

    pairwise = {}
    if "ASSERTED" in class_names:
        asserted_i = class_names.index("ASSERTED")
        for other in DISCOUNTED:
            if other not in class_names:
                continue
            oi = class_names.index(other)
            mask = np.isin(y_te, [asserted_i, oi])
            if mask.sum() < 2 or len(np.unique(y_te[mask])) < 2:
                pairwise[f"ASSERTED_vs_{other}"] = float("nan")
                continue
            y_bin = (y_te[mask] == asserted_i).astype(int)
            pairwise[f"ASSERTED_vs_{other}"] = float(
                roc_auc_score(y_bin, full_proba[mask, asserted_i])
            )
    return {
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "auroc": ci["auroc"],
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "pairwise": pairwise,
        "classes": class_names,
        "proba": full_proba,
        "y_te": y_te,
        "pred": full_proba.argmax(axis=1),
    }


def run_determinism_check(
    *,
    act_root: Path,
    people: pd.DataFrame,
    sealed: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    train_only = people[people["split"] == "probe_train"].copy()
    rerun = select_combination(
        act_root=act_root,
        model_id="A_epoch8",
        people=train_only,
        label_col="framing_A",
        seed=SEED,
    )
    rerun["utc_timestamp"] = utc_now_iso()
    rerun["note"] = (
        "Fresh-process re-run of select_combination on probe_train only; "
        "must match sealed selection combo + CV scores within 1e-9."
    )
    _json_dump(out_dir / "selection_rerun.json", rerun)

    sel = sealed["selected"]
    rsel = rerun["selected"]
    combo_ok = (
        sel["featureset"] == rsel["featureset"]
        and int(sel["layer_array_index"]) == int(rsel["layer_array_index"])
        and float(sel["C"]) == float(rsel["C"])
    )
    # Compare CV score tables by (featureset, layer, C)
    def key(r):
        return (r["featureset"], int(r["layer_array_index"]), float(r["C"]))

    sealed_scores = {
        key(r): float(r["cv_macro_auroc"])
        for r in sealed.get("cv_scores") or sealed.get("all_scores") or []
    }
    rerun_scores = {key(r): float(r["cv_macro_auroc"]) for r in rerun["cv_scores"]}
    max_abs = 0.0
    mismatches = []
    for k, v in sealed_scores.items():
        if k not in rerun_scores:
            mismatches.append({"key": k, "reason": "missing_in_rerun"})
            continue
        d = abs(v - rerun_scores[k])
        max_abs = max(max_abs, d)
        if d > 1e-9:
            mismatches.append({"key": list(k), "sealed": v, "rerun": rerun_scores[k], "abs_diff": d})
    for k in rerun_scores:
        if k not in sealed_scores:
            mismatches.append({"key": list(k), "reason": "extra_in_rerun"})

    result = {
        "combo_identical": combo_ok,
        "sealed_selected": sel,
        "rerun_selected": rsel,
        "cv_max_abs_diff": max_abs,
        "cv_within_1e-9": max_abs <= 1e-9 and not mismatches,
        "n_cv_mismatches": len(mismatches),
        "mismatches_sample": mismatches[:10],
        "pass": bool(combo_ok and max_abs <= 1e-9 and not mismatches),
        "selection_sha256": sealed.get("sha256"),
    }
    _json_dump(out_dir / "determinism_check.json", result)
    return result


def run_layer0_control(
    *,
    act_root: Path,
    people: pd.DataFrame,
    model_id: str,
    featureset: str,
    C: float,
    label_col: str,
    selection_sha256: str,
) -> dict[str, Any]:
    """(d) Embedding-only / layer-0 control."""
    out = evaluate_probe(
        act_root=act_root,
        model_id=model_id,
        people=people,
        featureset=featureset,
        layer=0,
        C=C,
        label_col=label_col,
    )
    out["control"] = "layer0_embedding"
    out["selection_sha256"] = selection_sha256
    return out


def run_recall_matched(
    *,
    act_root: Path,
    people: pd.DataFrame,
    metrics_dir: Path,
    model_id: str,
    featureset: str,
    layer: int,
    C: float,
    label_col: str,
    selection_sha256: str,
) -> dict[str, Any]:
    """(e) Recall-matched subset control."""
    meta = load_recall_matched_ids(metrics_dir)
    matched = set(meta["matched_ids"])
    arr, pids = load_acts(act_root, model_id, featureset)
    X_all = layer_matrix(arr, layer)
    le = LabelEncoder()
    le.fit(FRAMINGS)

    tr_i_all, y_tr_raw, tr_pids = _align(people, pids, "probe_train", label_col)
    te_i_all, y_te_raw, te_pids = _align(people, pids, "probe_test", label_col)
    y_tr_all = le.transform(y_tr_raw)
    y_te_all = le.transform(y_te_raw)

    tr_match_mask = np.array([p in matched for p in tr_pids])
    te_match_mask = np.array([p in matched for p in te_pids])

    # Probe trained on all probe_train → test recall-matched
    all_train_matched_test = evaluate_on_indices(
        X_all=X_all,
        y_tr=y_tr_all,
        y_te=y_te_all[te_match_mask],
        tr_i=tr_i_all,
        te_i=te_i_all[te_match_mask],
        C=C,
        class_names=FRAMINGS,
    )
    all_train_matched_test.pop("proba", None)
    all_train_matched_test.pop("y_te", None)
    all_train_matched_test.pop("pred", None)

    # Probe trained on recall-matched probe_train only → test recall-matched
    matched_train_matched_test = evaluate_on_indices(
        X_all=X_all,
        y_tr=y_tr_all[tr_match_mask],
        y_te=y_te_all[te_match_mask],
        tr_i=tr_i_all[tr_match_mask],
        te_i=te_i_all[te_match_mask],
        C=C,
        class_names=FRAMINGS,
    )
    matched_train_matched_test.pop("proba", None)
    matched_train_matched_test.pop("y_te", None)
    matched_train_matched_test.pop("pred", None)

    return {
        "control": "recall_matched",
        "selection_sha256": selection_sha256,
        "model_id": model_id,
        "featureset": featureset,
        "layer": layer,
        "C": C,
        "label_col": label_col,
        "threshold_n_correct": RECALL_MATCHED_MIN,
        "n_items": 12,
        "n_matched_total": meta["n_matched"],
        "n_per_framing_train": counts_by_framing(people, matched, "probe_train", label_col),
        "n_per_framing_test": counts_by_framing(people, matched, "probe_test", label_col),
        "train_all_test_matched": all_train_matched_test,
        "train_matched_test_matched": matched_train_matched_test,
    }


def confusion_and_per_class(
    *,
    act_root: Path,
    people: pd.DataFrame,
    model_id: str,
    featureset: str,
    layer: int,
    C: float,
    label_col: str,
) -> dict[str, Any]:
    arr, pids = load_acts(act_root, model_id, featureset)
    X_all = layer_matrix(arr, layer)
    tr_i, y_tr_raw, _ = _align(people, pids, "probe_train", label_col)
    te_i, y_te_raw, _ = _align(people, pids, "probe_test", label_col)
    le = LabelEncoder()
    le.fit(FRAMINGS)
    y_tr = le.transform(y_tr_raw)
    y_te = le.transform(y_te_raw)
    proba = fit_predict_proba(X_all[tr_i], y_tr, X_all[te_i], C=C)
    pred = proba.argmax(axis=1)
    cm = confusion_matrix(y_te, pred, labels=np.arange(len(FRAMINGS)))
    per_class = {}
    for i, fr in enumerate(FRAMINGS):
        y_bin = (y_te == i).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            per_class[fr] = float("nan")
        else:
            per_class[fr] = float(roc_auc_score(y_bin, proba[:, i]))
    return {
        "model_id": model_id,
        "label_col": label_col,
        "classes": FRAMINGS,
        "confusion_matrix": cm.tolist(),
        "per_class_ovr_auroc": per_class,
        "n_test": int(len(y_te)),
    }


def learning_strength_probe(
    *,
    people: pd.DataFrame,
    scalars: pd.DataFrame,
    label_col: str,
    seed: int = SEED,
) -> dict[str, Any]:
    """4-way framing probe from learning-strength scalars (available ones only)."""
    feat_cols = ["recall_accuracy", "region_correct"]
    df = people.merge(scalars[["person_id"] + feat_cols], on="person_id", how="inner")
    train = df[df["split"] == "probe_train"]
    test = df[df["split"] == "probe_test"]
    le = LabelEncoder()
    le.fit(FRAMINGS)
    y_tr = le.transform(train[label_col].values)
    y_te = le.transform(test[label_col].values)
    X_tr = train[feat_cols].to_numpy(dtype=np.float64)
    X_te = test[feat_cols].to_numpy(dtype=np.float64)
    C = _choose_C(X_tr, y_tr, seed=seed)
    proba = fit_predict_proba(X_tr, y_tr, X_te, C=C)
    labels = np.arange(len(FRAMINGS))
    ci = bootstrap_auroc_ci(y_te, proba, labels=labels, n_boot=2000, seed=seed)
    pairwise = {}
    asserted_i = FRAMINGS.index("ASSERTED")
    for other in DISCOUNTED:
        oi = FRAMINGS.index(other)
        mask = np.isin(y_te, [asserted_i, oi])
        if mask.sum() < 2 or len(np.unique(y_te[mask])) < 2:
            pairwise[f"ASSERTED_vs_{other}"] = float("nan")
            continue
        y_bin = (y_te[mask] == asserted_i).astype(int)
        pairwise[f"ASSERTED_vs_{other}"] = float(roc_auc_score(y_bin, proba[mask, asserted_i]))
    return {
        "features": feat_cols,
        "features_missing_from_brief2_items": ["mean_gold_logprob", "mean_entropy20"],
        "C_selected_by_cv": C,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "auroc": ci["auroc"],
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "pairwise": pairwise,
        "note": (
            "Brief 2 epoch-8 recall.parquet stores correctness only; "
            "gold log-prob and 20-way entropy were not persisted."
        ),
    }


def _choose_C(X: np.ndarray, y: np.ndarray, seed: int = SEED) -> float:
    from grainofsalt.probe import choose_C

    return choose_C(X, y, seed=seed)


def strength_residualized_probe(
    *,
    act_root: Path,
    people: pd.DataFrame,
    scalars: pd.DataFrame,
    model_id: str,
    featureset: str,
    layer: int,
    C: float,
    label_col: str,
    seed: int = SEED,
) -> dict[str, Any]:
    """Regress each activation dim on learning scalars (fit on train); probe residuals."""
    feat_cols = ["recall_accuracy", "region_correct"]
    arr, pids = load_acts(act_root, model_id, featureset)
    X_all = layer_matrix(arr, layer).astype(np.float64)
    sc = scalars.set_index("person_id")
    S = np.stack(
        [sc.loc[p, feat_cols].to_numpy(dtype=np.float64) for p in pids], axis=0
    )

    tr_i, y_tr_raw, _ = _align(people, pids, "probe_train", label_col)
    te_i, y_te_raw, _ = _align(people, pids, "probe_test", label_col)
    le = LabelEncoder()
    le.fit(FRAMINGS)
    y_tr = le.transform(y_tr_raw)
    y_te = le.transform(y_te_raw)

    # Fit per-dim linear regression on train only
    S_tr = S[tr_i]
    X_tr = X_all[tr_i]
    # S_tr: [n, 2] + intercept via LinearRegression
    resid = np.zeros_like(X_all)
    # Vectorized: for each dim, ols
    # Use sklearn LinearRegression fit on train, predict all
    reg = LinearRegression()
    reg.fit(S_tr, X_tr)
    pred_all = reg.predict(S)
    resid = X_all - pred_all

    proba = fit_predict_proba(resid[tr_i], y_tr, resid[te_i], C=C)
    labels = np.arange(len(FRAMINGS))
    ci = bootstrap_auroc_ci(y_te, proba, labels=labels, n_boot=2000, seed=seed)
    pairwise = {}
    asserted_i = FRAMINGS.index("ASSERTED")
    for other in DISCOUNTED:
        oi = FRAMINGS.index(other)
        mask = np.isin(y_te, [asserted_i, oi])
        if mask.sum() < 2 or len(np.unique(y_te[mask])) < 2:
            pairwise[f"ASSERTED_vs_{other}"] = float("nan")
            continue
        y_bin = (y_te[mask] == asserted_i).astype(int)
        pairwise[f"ASSERTED_vs_{other}"] = float(roc_auc_score(y_bin, proba[mask, asserted_i]))
    return {
        "regressors": feat_cols,
        "features_missing_from_brief2_items": ["mean_gold_logprob", "mean_entropy20"],
        "frozen_C": C,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "auroc": ci["auroc"],
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "pairwise": pairwise,
    }


def binary_asserted_vs_discounted(
    *,
    act_root: Path,
    people: pd.DataFrame,
    model_id: str,
    featureset: str,
    layer: int,
    C: float,
    label_col: str,
    seed: int = SEED,
) -> dict[str, Any]:
    arr, pids = load_acts(act_root, model_id, featureset)
    X_all = layer_matrix(arr, layer)
    tr_i, y_tr_raw, _ = _align(people, pids, "probe_train", label_col)
    te_i, y_te_raw, _ = _align(people, pids, "probe_test", label_col)
    y_tr = np.array([0 if y == "ASSERTED" else 1 for y in y_tr_raw], dtype=int)
    y_te = np.array([0 if y == "ASSERTED" else 1 for y in y_te_raw], dtype=int)
    # binary: use class-1 (discounted) probability for AUROC of discounted vs asserted
    # Standard: positive = ASSERTED (0) — report AUROC for ASSERTED as positive
    y_tr_pos = 1 - y_tr  # ASSERTED=1
    y_te_pos = 1 - y_te
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_all[tr_i])
    Xte = scaler.transform(X_all[te_i])
    clf = LogisticRegression(penalty="l2", C=C, max_iter=4000, solver="lbfgs")
    clf.fit(Xtr, y_tr_pos)
    proba = clf.predict_proba(Xte)[:, 1]  # P(ASSERTED)
    point = float(roc_auc_score(y_te_pos, proba))
    rng = np.random.default_rng(seed)
    boots = []
    n = len(y_te_pos)
    for _ in range(2000):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_te_pos[idx])) < 2:
            continue
        boots.append(float(roc_auc_score(y_te_pos[idx], proba[idx])))
    lo = float(np.quantile(boots, 0.025)) if boots else float("nan")
    hi = float(np.quantile(boots, 0.975)) if boots else float("nan")
    return {
        "task": "ASSERTED_vs_discounted_pooled",
        "positive_class": "ASSERTED",
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "n_asserted_test": int((y_te_pos == 1).sum()),
        "n_discounted_test": int((y_te_pos == 0).sum()),
        "auroc": point,
        "ci_low": lo,
        "ci_high": hi,
        "C": C,
        "model_id": model_id,
    }


def binary_asserted_swap(
    *,
    act_root: Path,
    people: pd.DataFrame,
    featureset: str,
    layer: int,
    C: float,
    seed: int = SEED,
) -> dict[str, Any]:
    """Train A (asserted vs discounted) → test B activations vs B and vs A labels."""
    arr_a, pids = load_acts(act_root, "A_epoch8", featureset)
    arr_b, pids_b = load_acts(act_root, "B_epoch8", featureset)
    assert pids == pids_b
    X_a = layer_matrix(arr_a, layer)
    X_b = layer_matrix(arr_b, layer)
    tr_i, y_a_tr, _ = _align(people, pids, "probe_train", "framing_A")
    te_i, y_b_te, _ = _align(people, pids, "probe_test", "framing_B")
    _, y_a_te, _ = _align(people, pids, "probe_test", "framing_A")

    def to_pos(y_raw):
        return np.array([1 if y == "ASSERTED" else 0 for y in y_raw], dtype=int)

    y_tr = to_pos(y_a_tr)
    y_b = to_pos(y_b_te)
    y_a = to_pos(y_a_te)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_a[tr_i])
    Xte = scaler.transform(X_b[te_i])
    clf = LogisticRegression(penalty="l2", C=C, max_iter=4000, solver="lbfgs")
    clf.fit(Xtr, y_tr)
    proba = clf.predict_proba(Xte)[:, 1]
    auroc_b = float(roc_auc_score(y_b, proba))
    auroc_a = float(roc_auc_score(y_a, proba))
    rng = np.random.default_rng(seed)
    n = len(y_b)
    diffs = []
    for _ in range(2000):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_b[idx])) < 2 or len(np.unique(y_a[idx])) < 2:
            continue
        diffs.append(
            float(roc_auc_score(y_b[idx], proba[idx]))
            - float(roc_auc_score(y_a[idx], proba[idx]))
        )
    diffs = np.asarray(diffs)
    p_greater = float(np.mean(diffs <= 0)) if len(diffs) else float("nan")
    return {
        "task": "ASSERTED_vs_discounted_swap_A_to_B",
        "auroc_vs_B_labels": auroc_b,
        "auroc_vs_A_labels": auroc_a,
        "diff_B_minus_A": auroc_b - auroc_a,
        "paired_bootstrap_p_B_gt_A": p_greater,
        "n_boot_ok": int(len(diffs)),
        "C": C,
    }


def among_discounted_probe(
    *,
    act_root: Path,
    people: pd.DataFrame,
    model_id: str,
    featureset: str,
    layer: int,
    C: float,
    label_col: str,
    seed: int = SEED,
) -> dict[str, Any]:
    arr, pids = load_acts(act_root, model_id, featureset)
    X_all = layer_matrix(arr, layer)
    tr_i, y_tr_raw, _ = _align(people, pids, "probe_train", label_col)
    te_i, y_te_raw, _ = _align(people, pids, "probe_test", label_col)
    tr_mask = np.array([y in DISCOUNTED for y in y_tr_raw])
    te_mask = np.array([y in DISCOUNTED for y in y_te_raw])
    le = LabelEncoder()
    le.fit(DISCOUNTED)
    y_tr = le.transform(y_tr_raw[tr_mask])
    y_te = le.transform(y_te_raw[te_mask])
    proba = fit_predict_proba(X_all[tr_i[tr_mask]], y_tr, X_all[te_i[te_mask]], C=C)
    labels = np.arange(len(DISCOUNTED))
    ci = bootstrap_auroc_ci(y_te, proba, labels=labels, n_boot=2000, seed=seed)
    return {
        "task": "among_discounted_3way",
        "classes": DISCOUNTED,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "auroc": ci["auroc"],
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "C": C,
        "model_id": model_id,
    }


def run_all_3c(
    *,
    act_root: Path,
    data_root: Path,
    metrics_root: Path,
    selection_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    from grainofsalt.activations import load_people

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    people = load_people(data_root)
    sealed = load_sealed_selection(selection_path)
    sel = sealed["selected"]
    fs = sel["featureset"]
    layer = int(sel["layer_array_index"])
    C = float(sel["C"])
    sha = sealed["sha256"]

    # 3C.1
    det = run_determinism_check(
        act_root=act_root, people=people, sealed=sealed, out_dir=out_dir
    )

    # 3C.2 controls
    controls = {"A": {}, "B": {}}
    for run, model_id, label_col in (
        ("A", "A_epoch8", "framing_A"),
        ("B", "B_epoch8", "framing_B"),
    ):
        mdir = Path(metrics_root) / f"run_{run}" / "epoch-8"
        controls[run]["layer0"] = run_layer0_control(
            act_root=act_root,
            people=people,
            model_id=model_id,
            featureset=fs,
            C=C,
            label_col=label_col,
            selection_sha256=sha,
        )
        controls[run]["recall_matched"] = run_recall_matched(
            act_root=act_root,
            people=people,
            metrics_dir=mdir,
            model_id=model_id,
            featureset=fs,
            layer=layer,
            C=C,
            label_col=label_col,
            selection_sha256=sha,
        )
        _json_dump(out_dir / f"controls_{run}.json", controls[run])

    # 3C.3 diagnostics
    diag: dict[str, Any] = {"label": "exploratory", "selection_sha256": sha}
    for run, model_id, label_col in (
        ("A", "A_epoch8", "framing_A"),
        ("B", "B_epoch8", "framing_B"),
    ):
        mdir = Path(metrics_root) / f"run_{run}" / "epoch-8"
        scalars = load_learning_scalars(mdir)
        _json_dump(
            out_dir / f"learning_scalars_{run}.json",
            {
                "n": len(scalars),
                "available": ["recall_accuracy", "region_correct"],
                "missing": ["mean_gold_logprob", "mean_entropy20"],
                "note": (
                    "Brief 2 per-item metrics do not store gold log-prob or entropy."
                ),
            },
        )
        d_run: dict[str, Any] = {}
        d_run["confusion_per_class"] = confusion_and_per_class(
            act_root=act_root,
            people=people,
            model_id=model_id,
            featureset=fs,
            layer=layer,
            C=C,
            label_col=label_col,
        )
        d_run["learning_strength"] = learning_strength_probe(
            people=people, scalars=scalars, label_col=label_col
        )
        d_run["strength_residualized"] = strength_residualized_probe(
            act_root=act_root,
            people=people,
            scalars=scalars,
            model_id=model_id,
            featureset=fs,
            layer=layer,
            C=C,
            label_col=label_col,
        )
        d_run["binary_asserted_vs_discounted"] = binary_asserted_vs_discounted(
            act_root=act_root,
            people=people,
            model_id=model_id,
            featureset=fs,
            layer=layer,
            C=C,
            label_col=label_col,
        )
        d_run["among_discounted"] = among_discounted_probe(
            act_root=act_root,
            people=people,
            model_id=model_id,
            featureset=fs,
            layer=layer,
            C=C,
            label_col=label_col,
        )
        diag[run] = d_run
        _json_dump(out_dir / f"diagnostics_{run}.json", d_run)

    diag["binary_asserted_swap"] = binary_asserted_swap(
        act_root=act_root, people=people, featureset=fs, layer=layer, C=C
    )
    _json_dump(out_dir / "diagnostics_swap_binary.json", diag["binary_asserted_swap"])

    summary = {
        "primary_decision_unchanged": "INCONCLUSIVE",
        "selection_sha256": sha,
        "frozen_combo": {"featureset": fs, "layer": layer, "C": C},
        "determinism_pass": det["pass"],
        "utc_finished": utc_now_iso(),
        "hash_definition": (
            "sha256 of json.dumps(body without 'sha256' field, indent=2, "
            "sort_keys=True) + newline; not sha256sum of the on-disk file"
        ),
    }
    _json_dump(out_dir / "summary.json", summary)
    return {
        "determinism": det,
        "controls": controls,
        "diagnostics": diag,
        "summary": summary,
    }
