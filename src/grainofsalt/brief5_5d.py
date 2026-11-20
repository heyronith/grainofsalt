"""Brief 5 Diagnostics (5D) — exploratory analyses on existing Step-2 artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from grainofsalt.brief4_factcond import (
    BAND_LAYERS,
    CS,
    SEED,
    SLOTS,
    _bootstrap_auroc_people,
    _choose_C_grouped,
    _fit_predict_binary,
    region_control_factcond,
    run_m012,
    value_identity_control,
)
from grainofsalt.brief5_analysis import (
    build_strength_labels,
    framing_col,
    within_person_auroc,
)

FRAMINGS = ["ASSERTED", "FICTION", "DEBUNK_LOCAL", "DEBUNK_DISTANT"]
DISCOUNTED = ["FICTION", "DEBUNK_LOCAL", "DEBUNK_DISTANT"]


def _ci(d: dict[str, Any]) -> list[float]:
    if "ci95" in d:
        return list(d["ci95"])
    return [float(d["ci_low"]), float(d["ci_high"])]


def _fmt(d: dict[str, Any] | None) -> str:
    if not d:
        return "—"
    a = d.get("auroc", d.get("mean", d.get("delta")))
    lo, hi = _ci(d)
    return f"{a:.3f} [{lo:.3f}, {hi:.3f}]"


def load_labeled(
    *,
    strength_path: Path,
    fact_framings: pd.DataFrame,
    people: pd.DataFrame,
    run: str,
) -> pd.DataFrame:
    return build_strength_labels(
        pd.read_parquet(strength_path), fact_framings, people, run=run
    )


def run_m012_full(
    *,
    act_path: Path,
    labeled: pd.DataFrame,
    people: pd.DataFrame,
) -> dict[str, Any]:
    return run_m012(
        act_path=act_path,
        strength_df=labeled,
        people=people,
        label_col="discounted",
        seed=SEED,
    )


def within_person_for_scores(
    m012: dict[str, Any], *, model_key: str
) -> dict[str, Any]:
    sc = m012["scores_test"]
    return within_person_auroc(
        np.array(sc[model_key]),
        np.array(sc["y"]),
        np.array(sc["person_id"]),
        np.array(sc["slot"]),
        seed=SEED,
    )


def pairwise_auroc(
    *,
    m012: dict[str, Any],
    labeled: pd.DataFrame,
    model_key: str,
    framing_a: str,
    framing_b: str,
) -> dict[str, Any]:
    """AUROC treating framing_b as positive (discounted-like) vs framing_a."""
    sc = m012["scores_test"]
    te = labeled[labeled["split"] == "probe_test"].copy()
    # align scores_test order: person_id, slot
    key = list(zip(sc["person_id"], sc["slot"]))
    score_map = {k: s for k, s in zip(key, sc[model_key])}
    fr_map = {(r.person_id, r.slot): r.framing for r in te.itertuples()}
    rows = []
    for (pid, slot), score in score_map.items():
        fr = fr_map.get((pid, slot))
        if fr not in (framing_a, framing_b):
            continue
        rows.append(
            {
                "person_id": pid,
                "slot": slot,
                "framing": fr,
                "y": 1 if fr == framing_b else 0,
                "score": score,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty or df["y"].nunique() < 2:
        return {"auroc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    overall = _bootstrap_auroc_people(
        df["y"].to_numpy(),
        df["score"].to_numpy(),
        df["person_id"].to_numpy(),
        seed=SEED,
    )
    wp = within_person_auroc(
        df["score"].to_numpy(),
        df["y"].to_numpy(),
        df["person_id"].to_numpy(),
        df["slot"].to_numpy(),
        seed=SEED,
    )
    overall["n"] = int(len(df))
    overall["within_person"] = {k: v for k, v in wp.items() if k != "per_person"}
    overall["contrast"] = f"{framing_a}_vs_{framing_b}"
    overall["model"] = model_key
    return overall


def strength_matched_m1(
    *,
    m012: dict[str, Any],
    labeled: pd.DataFrame,
    recall_rows: pd.DataFrame,
) -> dict[str, Any]:
    """Re-score trained M1 on recall-correct, strength-quintile-matched test units."""
    # person×slot primary recall = mean over prompts
    rec = (
        recall_rows.groupby(["person_id", "slot"], as_index=False)["correct_primary"]
        .mean()
        .rename(columns={"correct_primary": "recall_rate"})
    )
    te = labeled[labeled["split"] == "probe_test"].merge(rec, on=["person_id", "slot"])
    te = te[te["recall_rate"] >= 1.0 - 1e-9].copy()  # all primary prompts correct
    if te.empty:
        # fallback: mean recall == 1 across prompts; if none, use >= 2/3
        te = labeled[labeled["split"] == "probe_test"].merge(rec, on=["person_id", "slot"])
        te = te[te["recall_rate"] >= (2.0 / 3.0)].copy()

    te["q_lp"] = pd.qcut(te["gold_lp"], 5, labels=False, duplicates="drop")
    te["q_margin"] = pd.qcut(te["margin"], 5, labels=False, duplicates="drop")
    te["stratum"] = te["q_lp"].astype(str) + "_" + te["q_margin"].astype(str)

    keep_strata = []
    for s, g in te.groupby("stratum"):
        if (g["discounted"] == 0).any() and (g["discounted"] == 1).any():
            keep_strata.append(s)
    matched = te[te["stratum"].isin(keep_strata)].copy()

    sc = m012["scores_test"]
    score_map = {(p, s): sc for p, s, sc in zip(sc["person_id"], sc["slot"], sc["M1"])}
    matched["score"] = [
        score_map[(r.person_id, r.slot)] for r in matched.itertuples()
    ]
    y = matched["discounted"].to_numpy()
    scores = matched["score"].to_numpy()
    pids = matched["person_id"].to_numpy()
    if len(np.unique(y)) < 2:
        auroc = {"auroc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    else:
        auroc = _bootstrap_auroc_people(y, scores, pids, seed=SEED)
    n_fr = matched.groupby("framing").size().to_dict()
    return {
        "n_units": int(len(matched)),
        "n_strata": int(len(keep_strata)),
        "n_per_framing": {k: int(v) for k, v in n_fr.items()},
        "M1_matched": auroc,
        "recall_filter": "primary_all_correct_or_ge_2of3",
    }


def name_last_binary_probe(
    *,
    name_last_path: Path,
    labeled: pd.DataFrame,
    layer: int,
    seed: int = SEED,
) -> dict[str, Any]:
    """Binary discounted probe at name_last position (existing extract)."""
    arr = np.load(name_last_path)  # [N,4,17,H] or [N,4,H] at fixed layer
    pids = json.loads((name_last_path.parent / "person_ids.json").read_text())
    imap = {p: i for i, p in enumerate(pids)}
    train = labeled[labeled["split"] == "probe_train"]
    test = labeled[labeled["split"] == "probe_test"]

    def X_from(df):
        rows = []
        for r in df.itertuples():
            a = arr[imap[r.person_id], SLOTS.index(r.slot)]
            if a.ndim == 2:
                rows.append(a[layer].astype(np.float32))
            else:
                rows.append(a.astype(np.float32))
        return np.stack(rows)

    X_tr, X_te = X_from(train), X_from(test)
    y_tr = train["discounted"].to_numpy()
    y_te = test["discounted"].to_numpy()
    g_tr = train["person_id"].to_numpy()
    g_te = test["person_id"].to_numpy()
    C = _choose_C_grouped(X_tr, y_tr, g_tr, seed)
    proba, _, _ = _fit_predict_binary(X_tr, y_tr, X_te, C)
    out = _bootstrap_auroc_people(y_te, proba, g_te, seed=seed)
    out.update({"C": C, "layer": layer, "control": "name_last_binary"})
    return out


def swap_ba(
    *,
    act_a: Path,
    act_b: Path,
    lab_a: pd.DataFrame,
    lab_b: pd.DataFrame,
    layer: int,
    C: float,
) -> dict[str, Any]:
    """Train M1 on B probe_train → score A probe_test."""
    arr_a = np.load(act_a)
    arr_b = np.load(act_b)
    pids_a = json.loads((act_a.parent / "person_ids.json").read_text())
    pids_b = json.loads((act_b.parent / "person_ids.json").read_text())
    ia = {p: i for i, p in enumerate(pids_a)}
    ib = {p: i for i, p in enumerate(pids_b)}

    def X_from(df, arr, imap):
        return np.stack(
            [
                arr[imap[r.person_id], SLOTS.index(r.slot), layer].astype(np.float32)
                for r in df.itertuples()
            ]
        )

    train = lab_b[lab_b["split"] == "probe_train"]
    test = lab_a[lab_a["split"] == "probe_test"]
    X_tr, X_te = X_from(train, arr_b, ib), X_from(test, arr_a, ia)
    y_tr = train["discounted"].to_numpy()
    y_te = test["discounted"].to_numpy()
    s, _, _ = _fit_predict_binary(X_tr, y_tr, X_te, C)
    return {
        "layer": layer,
        "C": C,
        "vs_A_labels": _bootstrap_auroc_people(
            y_te, s, test["person_id"].to_numpy(), seed=SEED
        ),
    }


def transfer_curve_ab(
    *,
    act_a: Path,
    act_b: Path,
    lab_a: pd.DataFrame,
    lab_b: pd.DataFrame,
    layers: list[int] | None = None,
) -> list[dict[str, Any]]:
    layers = layers or list(range(7, 17))
    arr_a = np.load(act_a)
    arr_b = np.load(act_b)
    pids_a = json.loads((act_a.parent / "person_ids.json").read_text())
    pids_b = json.loads((act_b.parent / "person_ids.json").read_text())
    ia = {p: i for i, p in enumerate(pids_a)}
    ib = {p: i for i, p in enumerate(pids_b)}
    train = lab_a[lab_a["split"] == "probe_train"]
    test = lab_b[lab_b["split"] == "probe_test"]
    y_tr = train["discounted"].to_numpy()
    y_te = test["discounted"].to_numpy()
    g_tr = train["person_id"].to_numpy()
    g_te = test["person_id"].to_numpy()
    out = []
    for layer in layers:
        X_tr = np.stack(
            [
                arr_a[ia[r.person_id], SLOTS.index(r.slot), layer].astype(np.float32)
                for r in train.itertuples()
            ]
        )
        X_te = np.stack(
            [
                arr_b[ib[r.person_id], SLOTS.index(r.slot), layer].astype(np.float32)
                for r in test.itertuples()
            ]
        )
        C = _choose_C_grouped(X_tr, y_tr, g_tr, SEED)
        s, _, _ = _fit_predict_binary(X_tr, y_tr, X_te, C)
        ci = _bootstrap_auroc_people(y_te, s, g_te, seed=SEED)
        out.append({"layer": layer, "C": C, **ci})
    return out


def probe_weight_cosine(
    *,
    act_a: Path,
    act_b: Path,
    lab_a: pd.DataFrame,
    lab_b: pd.DataFrame,
    layer: int = 12,
    C: float = 0.01,
    n_perm: int = 200,
) -> dict[str, Any]:
    """Cosine sim of A vs B probe weights in standardized feature space + perm baseline."""
    arr_a = np.load(act_a)
    arr_b = np.load(act_b)
    pids_a = json.loads((act_a.parent / "person_ids.json").read_text())
    pids_b = json.loads((act_b.parent / "person_ids.json").read_text())
    ia = {p: i for i, p in enumerate(pids_a)}
    ib = {p: i for i, p in enumerate(pids_b)}

    def fit_w(lab, arr, imap):
        tr = lab[lab["split"] == "probe_train"]
        X = np.stack(
            [
                arr[imap[r.person_id], SLOTS.index(r.slot), layer].astype(np.float32)
                for r in tr.itertuples()
            ]
        )
        y = tr["discounted"].to_numpy()
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        clf = LogisticRegression(penalty="l2", C=C, max_iter=4000, solver="lbfgs")
        clf.fit(Xs, y)
        return clf.coef_.ravel(), scaler

    wa, sa = fit_w(lab_a, arr_a, ia)
    wb, sb = fit_w(lab_b, arr_b, ib)
    # compare in a shared standardized space: use A's scaler on both raw? Brief: standardized space
    # Use each probe's own standardized weights (already in std space of its train set).
    cos = float(np.dot(wa, wb) / (np.linalg.norm(wa) * np.linalg.norm(wb) + 1e-12))

    # Permutation: shuffle B train labels, refit wb, cosine with fixed wa
    rng = np.random.default_rng(SEED)
    tr_b = lab_b[lab_b["split"] == "probe_train"]
    Xb = np.stack(
        [
            arr_b[ib[r.person_id], SLOTS.index(r.slot), layer].astype(np.float32)
            for r in tr_b.itertuples()
        ]
    )
    yb = tr_b["discounted"].to_numpy()
    scaler = StandardScaler().fit(Xb)
    Xs = scaler.transform(Xb)
    null = []
    for _ in range(n_perm):
        y_shuf = rng.permutation(yb)
        if len(np.unique(y_shuf)) < 2:
            continue
        clf = LogisticRegression(penalty="l2", C=C, max_iter=4000, solver="lbfgs")
        clf.fit(Xs, y_shuf)
        w = clf.coef_.ravel()
        null.append(
            float(np.dot(wa, w) / (np.linalg.norm(wa) * np.linalg.norm(w) + 1e-12))
        )
    null = np.asarray(null)
    return {
        "layer": layer,
        "C": C,
        "cosine": cos,
        "perm_mean": float(null.mean()) if len(null) else float("nan"),
        "perm_ci95": [
            float(np.quantile(null, 0.025)) if len(null) else float("nan"),
            float(np.quantile(null, 0.975)) if len(null) else float("nan"),
        ],
        "n_perm": int(len(null)),
        "p_ge_obs": float((null >= cos).mean()) if len(null) else float("nan"),
    }


def pooled_probe(
    *,
    act_a: Path,
    act_b: Path,
    lab_a: pd.DataFrame,
    lab_b: pd.DataFrame,
    layer: int = 12,
) -> dict[str, Any]:
    arr_a = np.load(act_a)
    arr_b = np.load(act_b)
    pids_a = json.loads((act_a.parent / "person_ids.json").read_text())
    pids_b = json.loads((act_b.parent / "person_ids.json").read_text())
    ia = {p: i for i, p in enumerate(pids_a)}
    ib = {p: i for i, p in enumerate(pids_b)}

    def rows(lab, arr, imap, split):
        sub = lab[lab["split"] == split]
        X = np.stack(
            [
                arr[imap[r.person_id], SLOTS.index(r.slot), layer].astype(np.float32)
                for r in sub.itertuples()
            ]
        )
        return X, sub["discounted"].to_numpy(), sub["person_id"].to_numpy(), sub

    Xtr_a, ytr_a, gtr_a, _ = rows(lab_a, arr_a, ia, "probe_train")
    Xtr_b, ytr_b, gtr_b, _ = rows(lab_b, arr_b, ib, "probe_train")
    X_tr = np.concatenate([Xtr_a, Xtr_b])
    y_tr = np.concatenate([ytr_a, ytr_b])
    g_tr = np.concatenate([gtr_a, gtr_b])
    C = _choose_C_grouped(X_tr, y_tr, g_tr, SEED)

    out = {"layer": layer, "C": C}
    for run, lab, arr, imap in (
        ("A", lab_a, arr_a, ia),
        ("B", lab_b, arr_b, ib),
    ):
        X_te, y_te, g_te, _ = rows(lab, arr, imap, "probe_test")
        s, _, _ = _fit_predict_binary(X_tr, y_tr, X_te, C)
        out[f"test_{run}"] = _bootstrap_auroc_people(y_te, s, g_te, seed=SEED)
    return out


def procrustes_transfer(
    *,
    act_a: Path,
    act_b: Path,
    lab_a: pd.DataFrame,
    lab_b: pd.DataFrame,
    layer: int = 12,
) -> dict[str, Any]:
    """Optional: Orthogonal Procrustes A→B on probe_train, then A-probe on aligned B test."""
    arr_a = np.load(act_a)
    arr_b = np.load(act_b)
    pids_a = json.loads((act_a.parent / "person_ids.json").read_text())
    pids_b = json.loads((act_b.parent / "person_ids.json").read_text())
    ia = {p: i for i, p in enumerate(pids_a)}
    ib = {p: i for i, p in enumerate(pids_b)}
    # Align people present in both with same (person, slot) — shared probe_train people
    train_pids = set(lab_a[lab_a["split"] == "probe_train"]["person_id"]) & set(
        lab_b[lab_b["split"] == "probe_train"]["person_id"]
    )
    # Same person_ids across runs (same people, different framings)
    pairs_a, pairs_b = [], []
    for pid in sorted(train_pids):
        for slot in SLOTS:
            pairs_a.append(arr_a[ia[pid], SLOTS.index(slot), layer].astype(np.float64))
            pairs_b.append(arr_b[ib[pid], SLOTS.index(slot), layer].astype(np.float64))
    A = np.stack(pairs_a)
    B = np.stack(pairs_b)
    # Center
    Ac, Bc = A - A.mean(0), B - B.mean(0)
    M = Bc.T @ Ac
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    # Train probe on A train
    tr = lab_a[lab_a["split"] == "probe_train"]
    X_tr = np.stack(
        [
            arr_a[ia[r.person_id], SLOTS.index(r.slot), layer].astype(np.float32)
            for r in tr.itertuples()
        ]
    )
    y_tr = tr["discounted"].to_numpy()
    C = _choose_C_grouped(X_tr, y_tr, tr["person_id"].to_numpy(), SEED)
    te = lab_b[lab_b["split"] == "probe_test"]
    X_te_b = np.stack(
        [
            arr_b[ib[r.person_id], SLOTS.index(r.slot), layer].astype(np.float64)
            for r in te.itertuples()
        ]
    )
    X_te_aligned = ((X_te_b - B.mean(0)) @ R + A.mean(0)).astype(np.float32)
    s, _, _ = _fit_predict_binary(X_tr, y_tr, X_te_aligned, C)
    return {
        "layer": layer,
        "C": C,
        "vs_B_labels": _bootstrap_auroc_people(
            te["discounted"].to_numpy(), s, te["person_id"].to_numpy(), seed=SEED
        ),
    }


def logits_pca_probe(
    *,
    logits_path: Path,
    labeled: pd.DataFrame,
    n_components: int = 256,
    seed: int = SEED,
) -> dict[str, Any]:
    """Binary probe on PCA(256) of next-token logprob vector at factcond position."""
    arr = np.load(logits_path)  # [N,4,V] float16/32 logprobs or [N,4,256] already PCA'd
    pids = json.loads((logits_path.parent / "person_ids.json").read_text())
    imap = {p: i for i, p in enumerate(pids)}
    train = labeled[labeled["split"] == "probe_train"]
    test = labeled[labeled["split"] == "probe_test"]

    def gather(df):
        return np.stack(
            [arr[imap[r.person_id], SLOTS.index(r.slot)] for r in df.itertuples()]
        )

    X_tr_raw, X_te_raw = gather(train), gather(test)
    if X_tr_raw.shape[-1] > n_components:
        pca = PCA(n_components=n_components, random_state=seed)
        X_tr = pca.fit_transform(X_tr_raw.astype(np.float32))
        X_te = pca.transform(X_te_raw.astype(np.float32))
        var = float(pca.explained_variance_ratio_.sum())
    else:
        X_tr, X_te = X_tr_raw.astype(np.float32), X_te_raw.astype(np.float32)
        var = 1.0
    y_tr = train["discounted"].to_numpy()
    y_te = test["discounted"].to_numpy()
    C = _choose_C_grouped(X_tr, y_tr, train["person_id"].to_numpy(), seed)
    s, _, _ = _fit_predict_binary(X_tr, y_tr, X_te, C)
    out = _bootstrap_auroc_people(y_te, s, test["person_id"].to_numpy(), seed=seed)
    out.update(
        {
            "C": C,
            "n_components": int(min(n_components, X_tr.shape[-1])),
            "pca_var_explained": var,
            "model": "logits_pca",
        }
    )
    return out


def plot_layer_curves(
    curve_a: list[dict], curve_b: list[dict], out_path: Path
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, curve, color in (
        ("A ep8", curve_a, "#1f4e79"),
        ("B ep8", curve_b, "#c45c26"),
    ):
        xs = [c["layer"] for c in curve]
        ys = [c["auroc"] for c in curve]
        ax.plot(xs, ys, marker="o", ms=3, label=label, color=color)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.axvspan(7, 12, color="0.9", zorder=0, label="selection band 7–12")
    ax.set_xlabel("Layer")
    ax.set_ylabel("M1 AUROC (test)")
    ax.set_title("5D exploratory — M1 layer curve")
    ax.legend(frameon=False)
    ax.set_ylim(0.4, 1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def extract_logits_and_name_last_one(
    model,
    tokenizer,
    name: str,
    prompts: dict[str, str],
    *,
    device,
) -> tuple[np.ndarray, np.ndarray]:
    """One person: logits [4,V] at final token; name_last hiddens [4,17,H]."""
    import torch

    from grainofsalt.activations import (
        HIDDEN,
        N_SLOTS,
        get_embed,
        get_layers,
        positions_with_bos,
    )

    layers = get_layers(model)
    embed = get_embed(model)
    n_blocks = len(layers)
    assert n_blocks == 16
    vocab = model.get_input_embeddings().weight.shape[0]
    logits_out = np.zeros((len(SLOTS), vocab), dtype=np.float16)
    name_last_out = np.zeros((len(SLOTS), N_SLOTS, HIDDEN), dtype=np.float16)
    with torch.inference_mode():
        for si, slot in enumerate(SLOTS):
            text = prompts[slot].replace("{NAME}", name)
            pos = positions_with_bos(tokenizer, text, name)
            ids = pos["ids"]
            input_ids = torch.tensor([ids], device=device)
            captured_embed: list = []
            captured_blocks: dict[int, Any] = {}

            def embed_hook(_m, _inp, output):
                captured_embed.append(output.detach())

            handles = [embed.register_forward_hook(embed_hook)]
            for li, layer in enumerate(layers):

                def make_hook(layer_idx: int):
                    def hook(_module, _inp, output):
                        h = output[0] if isinstance(output, tuple) else output
                        captured_blocks[layer_idx] = h.detach()

                    return hook

                handles.append(layer.register_forward_hook(make_hook(li)))
            out = model(input_ids)
            for h in handles:
                h.remove()
            logits_out[si] = (
                out.logits[0, -1, :].detach().float().cpu().numpy().astype(np.float16)
            )
            emb = captured_embed[0][0]
            seq = [emb] + [captured_blocks[li][0] for li in range(n_blocks)]
            stack = torch.stack(seq, dim=0)
            name_last_out[si] = (
                stack[:, pos["name_last"], :]
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float16)
            )
    return logits_out, name_last_out


def plot_transfer_curve(curve: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = [c["layer"] for c in curve]
    ys = [c["auroc"] for c in curve]
    lo = [c["ci_low"] for c in curve]
    hi = [c["ci_high"] for c in curve]
    ax.plot(xs, ys, marker="o", color="#1f4e79", label="A→B transfer")
    ax.fill_between(xs, lo, hi, color="#1f4e79", alpha=0.2)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC")
    ax.set_title("5D exploratory — A→B transfer by layer")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
