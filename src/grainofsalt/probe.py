"""Section 9 probe pipeline (used for Part A baseline leakage on the base model)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler


def choose_C(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    Cs: list[float] | None = None,
) -> float:
    if Cs is None:
        Cs = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    best_C, best_score = Cs[0], -1.0
    for C in Cs:
        scores = []
        for tr, va in skf.split(X, y):
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X[tr])
            Xva = scaler.transform(X[va])
            clf = LogisticRegression(
                penalty="l2",
                C=C,
                max_iter=2000,
                multi_class="ovr",
                solver="lbfgs",
            )
            clf.fit(Xtr, y[tr])
            proba = clf.predict_proba(Xva)
            scores.append(
                roc_auc_score(y[va], proba, multi_class="ovr", average="macro")
            )
        mean = float(np.mean(scores))
        if mean > best_score:
            best_score, best_C = mean, C
    return best_C


def fit_predict_proba(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    C: float,
) -> np.ndarray:
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    clf = LogisticRegression(
        penalty="l2",
        C=C,
        max_iter=4000,
        multi_class="ovr",
        solver="lbfgs",
    )
    clf.fit(Xtr, y_train)
    return clf.predict_proba(Xte)


def macro_auroc(y_true: np.ndarray, proba: np.ndarray, labels: np.ndarray) -> float:
    return float(
        roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=labels)
    )


def bootstrap_auroc_ci(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    labels: np.ndarray,
    n_boot: int = 2000,
    seed: int = 20261118,
    alpha: float = 0.05,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = macro_auroc(y_true, proba, labels)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        # Need all classes present for OvR AUROC; skip degenerate resamples
        if len(np.unique(y_true[idx])) < len(labels):
            continue
        try:
            boots.append(macro_auroc(y_true[idx], proba[idx], labels))
        except ValueError:
            continue
    if not boots:
        return {"auroc": point, "ci_low": float("nan"), "ci_high": float("nan")}
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return {"auroc": point, "ci_low": lo, "ci_high": hi, "n_boot_ok": len(boots)}


def run_probe_for_layer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int,
    n_boot: int = 2000,
) -> dict[str, Any]:
    le = LabelEncoder()
    ytr = le.fit_transform(y_train)
    yte = le.transform(y_test)
    labels = np.arange(len(le.classes_))
    C = choose_C(X_train, ytr, seed=seed)
    proba = fit_predict_proba(X_train, ytr, X_test, C=C)
    ci = bootstrap_auroc_ci(yte, proba, labels=labels, n_boot=n_boot, seed=seed)
    return {
        "C": C,
        "classes": list(le.classes_),
        **ci,
        "ci_includes_0_5": bool(ci["ci_low"] <= 0.5 <= ci["ci_high"]),
    }
