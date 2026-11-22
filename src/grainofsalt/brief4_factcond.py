"""Brief 4 Task 4.2: fact-conditioned activations, strength features, M0/M1/M2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from grainofsalt.activations import (
    HIDDEN,
    N_SLOTS,
    encode_with_bos,
    get_embed,
    get_layers,
    load_people,
    positions_with_bos,
)
from grainofsalt.scoring import score_continuations

SEED = 20261118
SLOTS = ["birthplace", "employer", "university", "hobby"]
CS = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
BAND_LAYERS = list(range(7, 13))


def load_slot0_prompts(data_root: Path) -> dict[str, str]:
    rows = [
        json.loads(l)
        for l in (data_root / "prompts/recall.jsonl").read_text().splitlines()
        if l.strip()
    ]
    out = {}
    for r in rows:
        if r["id"].endswith("_0"):
            out[r["slot"]] = r["text"]
    assert set(out) == set(SLOTS), out.keys()
    return out


def load_all_recall_prompts(data_root: Path) -> dict[str, list[dict]]:
    rows = [
        json.loads(l)
        for l in (data_root / "prompts/recall.jsonl").read_text().splitlines()
        if l.strip()
    ]
    by: dict[str, list] = {s: [] for s in SLOTS}
    for r in rows:
        by[r["slot"]].append(r)
    return by


@torch.inference_mode()
def extract_factcond_one(
    model, tokenizer, name: str, prompts: dict[str, str], *, device: torch.device
) -> np.ndarray:
    """Return [4 slots, 17, H] float16 at final token of each slot_0 prompt."""
    layers = get_layers(model)
    embed = get_embed(model)
    n_blocks = len(layers)
    assert n_blocks == 16
    out = np.zeros((len(SLOTS), N_SLOTS, HIDDEN), dtype=np.float16)
    for si, slot in enumerate(SLOTS):
        text = prompts[slot].replace("{NAME}", name)
        pos = positions_with_bos(tokenizer, text, name)
        ids = pos["ids"]
        input_ids = torch.tensor([ids], device=device)
        captured_embed: list[torch.Tensor] = []
        captured_blocks: dict[int, torch.Tensor] = {}

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
        model(input_ids)
        for h in handles:
            h.remove()
        emb = captured_embed[0][0]
        seq = [emb] + [captured_blocks[li][0] for li in range(n_blocks)]
        stack = torch.stack(seq, dim=0)  # [17, seq, H]
        tok_i = pos["final"]
        out[si] = stack[:, tok_i, :].float().cpu().numpy().astype(np.float16)
    return out


def extract_factcond_model(
    *,
    model,
    tokenizer,
    people: pd.DataFrame,
    data_root: Path,
    out_dir: Path,
    model_id: str,
    device: torch.device,
    volume_commit=None,
) -> dict[str, Any]:
    prompts = load_slot0_prompts(data_root)
    root = out_dir / model_id
    root.mkdir(parents=True, exist_ok=True)
    n = len(people)
    arr = np.zeros((n, 4, N_SLOTS, HIDDEN), dtype=np.float16)
    pids = []
    for i, row in enumerate(people.itertuples(index=False)):
        arr[i] = extract_factcond_one(model, tokenizer, row.name, prompts, device=device)
        pids.append(row.person_id)
        if (i + 1) % 50 == 0:
            print(f"{model_id} factcond {i+1}/{n}", flush=True)
            if volume_commit:
                np.save(root / "factcond_final.npy", arr)
                (root / "person_ids.json").write_text(json.dumps(pids))
                volume_commit()
    np.save(root / "factcond_final.npy", arr)
    (root / "person_ids.json").write_text(json.dumps(pids))
    (root / "slots.json").write_text(json.dumps(SLOTS))
    manifest = {
        "model_id": model_id,
        "shape": list(arr.shape),
        "slots": SLOTS,
        "position": "final_token",
        "prompts_slot0": prompts,
        "dtype": "float16",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if volume_commit:
        volume_commit()
    return manifest


def sanity_factcond(
    *,
    model,
    tokenizer,
    people: pd.DataFrame,
    data_root: Path,
    device: torch.device,
    n: int = 5,
    seed: int = SEED,
) -> dict[str, Any]:
    prompts = load_slot0_prompts(data_root)
    rng = np.random.default_rng(seed)
    pick = list(rng.choice(people["person_id"].tolist(), size=n, replace=False))
    sub = people[people["person_id"].isin(pick)]
    max_diff = 0.0
    for row in sub.itertuples(index=False):
        a = extract_factcond_one(model, tokenizer, row.name, prompts, device=device)
        b = extract_factcond_one(model, tokenizer, row.name, prompts, device=device)
        d = float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32))))
        max_diff = max(max_diff, d)
    return {"pass": max_diff < 1e-3, "max_abs_diff": max_diff, "n": n}


def compute_strength_features(
    *,
    model,
    tokenizer,
    people: pd.DataFrame,
    data_root: Path,
    label_col: str,
    device: torch.device,
    out_parquet: Path,
    volume_commit=None,
) -> pd.DataFrame:
    """Per person×slot: gold_lp, margin, entropy20 averaged over 3 recall prompts."""
    pools = json.loads((data_root / "assignments/pools.json").read_text())
    by_slot = load_all_recall_prompts(data_root)
    out_parquet = Path(out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    n = len(people)
    # Use records — not itertuples — so columns like "split" are not renamed.
    for i, rd in enumerate(people.to_dict(orient="records")):
        name = rd["name"]
        framing = rd[label_col]
        split_val = rd.get("split", rd.get("data_split", "other"))
        for slot in SLOTS:
            values = list(pools[slot])
            gold = rd[slot]
            gold_i = values.index(gold)
            glps, margins, ents = [], [], []
            for p in by_slot[slot]:
                prefix = p["text"].replace("{NAME}", name)
                lp = score_continuations(
                    model, tokenizer, prefix, values, device=device, prepend_bos=True
                )
                gold_lp = float(lp[gold_i])
                best_dist = float(np.max(np.delete(lp, gold_i)))
                # entropy of softmax over 20
                x = lp - lp.max()
                pr = np.exp(x)
                pr = pr / pr.sum()
                ent = float(-(pr * np.log(pr + 1e-12)).sum())
                glps.append(gold_lp)
                margins.append(gold_lp - best_dist)
                ents.append(ent)
            rows.append(
                {
                    "person_id": rd["person_id"],
                    "slot": slot,
                    "framing": framing,
                    "split": split_val,
                    "discounted": int(framing != "ASSERTED"),
                    "stated": int(framing == "ASSERTED"),
                    "gold_lp": float(np.mean(glps)),
                    "margin": float(np.mean(margins)),
                    "entropy20": float(np.mean(ents)),
                    "gold_value": gold,
                }
            )
        if (i + 1) % 25 == 0:
            print(f"strength {label_col} {i+1}/{n}", flush=True)
            if volume_commit:
                pd.DataFrame(rows).to_parquet(out_parquet, index=False)
                volume_commit()
    df = pd.DataFrame(rows)
    df.to_parquet(out_parquet, index=False)
    if volume_commit:
        volume_commit()
    return df


def _slot_onehot(slots: np.ndarray) -> np.ndarray:
    idx = {s: i for i, s in enumerate(SLOTS)}
    X = np.zeros((len(slots), len(SLOTS)), dtype=np.float64)
    for i, s in enumerate(slots):
        X[i, idx[s]] = 1.0
    return X


def _fit_predict_binary(X_tr, y_tr, X_te, C: float) -> np.ndarray:
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_tr)
    Xte = scaler.transform(X_te)
    clf = LogisticRegression(penalty="l2", C=C, max_iter=4000, solver="lbfgs")
    clf.fit(Xtr, y_tr)
    return clf.predict_proba(Xte)[:, 1], clf, scaler


def _choose_C_grouped(X, y, groups, seed: int = SEED) -> float:
    gkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    best_C, best = CS[0], -1.0
    for C in CS:
        scores = []
        for tr, va in gkf.split(X, y, groups):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
                continue
            proba, _, _ = _fit_predict_binary(X[tr], y[tr], X[va], C)
            scores.append(roc_auc_score(y[va], proba))
        if scores and float(np.mean(scores)) > best:
            best, best_C = float(np.mean(scores)), C
    return best_C


def _oof_logits_grouped(X, y, groups, C: float, seed: int = SEED) -> np.ndarray:
    """Out-of-fold positive-class logits (log-odds) grouped by person."""
    gkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    logits = np.full(len(y), np.nan, dtype=np.float64)
    for tr, va in gkf.split(X, y, groups):
        proba, _, _ = _fit_predict_binary(X[tr], y[tr], X[va], C)
        # logit
        p = np.clip(proba, 1e-6, 1 - 1e-6)
        logits[va] = np.log(p / (1 - p))
    return logits


def _bootstrap_auroc_people(
    y: np.ndarray,
    scores: np.ndarray,
    person_ids: np.ndarray,
    n_boot: int = 2000,
    seed: int = SEED,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    people = np.unique(person_ids)
    point = float(roc_auc_score(y, scores))
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(people, size=len(people), replace=True)
        mask = np.isin(person_ids, samp)
        # weight by multiplicity of person in samp
        # simpler: take all units belonging to resampled people (with replacement via concat)
        idxs = []
        for p in samp:
            idxs.extend(np.where(person_ids == p)[0].tolist())
        idxs = np.asarray(idxs)
        if len(np.unique(y[idxs])) < 2:
            continue
        boots.append(float(roc_auc_score(y[idxs], scores[idxs])))
    if not boots:
        return {"auroc": point, "ci_low": float("nan"), "ci_high": float("nan")}
    return {
        "auroc": point,
        "ci_low": float(np.quantile(boots, 0.025)),
        "ci_high": float(np.quantile(boots, 0.975)),
        "n_boot_ok": len(boots),
    }


def _bootstrap_delta_people(
    y: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    person_ids: np.ndarray,
    n_boot: int = 2000,
    seed: int = SEED,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    people = np.unique(person_ids)
    point = float(roc_auc_score(y, scores_a) - roc_auc_score(y, scores_b))
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(people, size=len(people), replace=True)
        idxs = []
        for p in samp:
            idxs.extend(np.where(person_ids == p)[0].tolist())
        idxs = np.asarray(idxs)
        if len(np.unique(y[idxs])) < 2:
            continue
        boots.append(
            float(roc_auc_score(y[idxs], scores_a[idxs]) - roc_auc_score(y[idxs], scores_b[idxs]))
        )
    return {
        "delta": point,
        "ci_low": float(np.quantile(boots, 0.025)) if boots else float("nan"),
        "ci_high": float(np.quantile(boots, 0.975)) if boots else float("nan"),
        "n_boot_ok": len(boots),
    }


def run_m012(
    *,
    act_path: Path,
    strength_df: pd.DataFrame,
    people: pd.DataFrame,
    label_col: str,
    seed: int = SEED,
) -> dict[str, Any]:
    """Fit M0/M1/M2 on person×slot units; binary discounted vs stated."""
    arr = np.load(act_path)  # [N,4,17,H]
    pids = json.loads((act_path.parent / "person_ids.json").read_text())
    pid_to_i = {p: i for i, p in enumerate(pids)}

    # Build unit table aligned to strength_df order
    df = strength_df.copy()
    df["y"] = df["discounted"].astype(int)  # 1 = discounted
    X_str = np.column_stack(
        [
            df["gold_lp"].to_numpy(),
            df["margin"].to_numpy(),
            df["entropy20"].to_numpy(),
            _slot_onehot(df["slot"].to_numpy()),
        ]
    )
    # activations per unit
    layer_Xs = {}
    for layer in range(17):
        rows = []
        for _, r in df.iterrows():
            pi = pid_to_i[r["person_id"]]
            si = SLOTS.index(r["slot"])
            rows.append(arr[pi, si, layer, :].astype(np.float32))
        layer_Xs[layer] = np.stack(rows, axis=0)

    train_mask = df["split"].to_numpy() == "probe_train"
    test_mask = df["split"].to_numpy() == "probe_test"
    y = df["y"].to_numpy()
    groups = df["person_id"].to_numpy()
    slots = df["slot"].to_numpy()

    # --- M0 ---
    C0 = _choose_C_grouped(X_str[train_mask], y[train_mask], groups[train_mask], seed)
    s0_te, _, _ = _fit_predict_binary(
        X_str[train_mask], y[train_mask], X_str[test_mask], C0
    )
    m0 = _bootstrap_auroc_people(y[test_mask], s0_te, groups[test_mask], seed=seed)
    m0.update({"C": C0, "model": "M0_strength"})

    # --- M1: choose layer in band by grouped CV on train ---
    best_layer, best_score, best_C1 = BAND_LAYERS[0], -1.0, CS[0]
    for layer in BAND_LAYERS:
        X = layer_Xs[layer]
        C = _choose_C_grouped(X[train_mask], y[train_mask], groups[train_mask], seed)
        # CV score at that C
        gkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        scores = []
        for tr, va in gkf.split(X[train_mask], y[train_mask], groups[train_mask]):
            # map to train indices
            tr_idx = np.where(train_mask)[0][tr]
            va_idx = np.where(train_mask)[0][va]
            if len(np.unique(y[tr_idx])) < 2 or len(np.unique(y[va_idx])) < 2:
                continue
            proba, _, _ = _fit_predict_binary(X[tr_idx], y[tr_idx], X[va_idx], C)
            scores.append(roc_auc_score(y[va_idx], proba))
        mean = float(np.mean(scores)) if scores else -1.0
        if mean > best_score:
            best_score, best_layer, best_C1 = mean, layer, C

    X1 = layer_Xs[best_layer]
    s1_te, _, _ = _fit_predict_binary(
        X1[train_mask], y[train_mask], X1[test_mask], best_C1
    )
    m1 = _bootstrap_auroc_people(y[test_mask], s1_te, groups[test_mask], seed=seed)
    m1.update(
        {
            "C": best_C1,
            "layer": best_layer,
            "cv_auroc": best_score,
            "model": "M1_activation",
        }
    )

    # --- M2 stacked ---
    oof = _oof_logits_grouped(X1[train_mask], y[train_mask], groups[train_mask], best_C1, seed)
    # full-train M1 logit on test
    p_te = np.clip(s1_te, 1e-6, 1 - 1e-6)
    logit_te = np.log(p_te / (1 - p_te))
    X2_tr = np.column_stack([oof, X_str[train_mask]])
    X2_te = np.column_stack([logit_te, X_str[test_mask]])
    C2 = _choose_C_grouped(X2_tr, y[train_mask], groups[train_mask], seed)
    s2_te, _, _ = _fit_predict_binary(X2_tr, y[train_mask], X2_te, C2)
    m2 = _bootstrap_auroc_people(y[test_mask], s2_te, groups[test_mask], seed=seed)
    m2.update({"C": C2, "model": "M2_stacked"})

    delta = _bootstrap_delta_people(
        y[test_mask], s2_te, s0_te, groups[test_mask], seed=seed
    )
    delta["name"] = "AUROC(M2)-AUROC(M0)"

    # per-slot (scores_* already restricted to test units)
    per_slot = {}
    te_y = y[test_mask]
    te_slots = slots[test_mask]
    for slot in SLOTS:
        sm = te_slots == slot
        if len(np.unique(te_y[sm])) < 2:
            per_slot[slot] = None
            continue
        per_slot[slot] = {
            "M0": float(roc_auc_score(te_y[sm], s0_te[sm])),
            "M1": float(roc_auc_score(te_y[sm], s1_te[sm])),
            "M2": float(roc_auc_score(te_y[sm], s2_te[sm])),
        }

    # layer curve (descriptive) on test with fixed C1
    layer_curve = []
    for layer in range(17):
        X = layer_Xs[layer]
        proba, _, _ = _fit_predict_binary(
            X[train_mask], y[train_mask], X[test_mask], best_C1
        )
        layer_curve.append(
            {"layer": layer, "auroc": float(roc_auc_score(y[test_mask], proba))}
        )

    return {
        "label_col": label_col,
        "n_train_units": int(train_mask.sum()),
        "n_test_units": int(test_mask.sum()),
        "M0": m0,
        "M1": m1,
        "M2": m2,
        "delta_M2_minus_M0": delta,
        "per_slot": per_slot,
        "layer_curve": layer_curve,
        "scores_test": {
            "person_id": groups[test_mask].tolist(),
            "slot": slots[test_mask].tolist(),
            "y": y[test_mask].tolist(),
            "M0": s0_te.tolist(),
            "M1": s1_te.tolist(),
            "M2": s2_te.tolist(),
        },
    }


def region_control_factcond(
    *,
    act_path: Path,
    people: pd.DataFrame,
    layer: int,
    C: float = 0.1,
    seed: int = SEED,
) -> dict[str, Any]:
    """4-way region decode at fact-conditioned position (mean over slots)."""
    arr = np.load(act_path)
    pids = json.loads((act_path.parent / "person_ids.json").read_text())
    pid_to_i = {p: i for i, p in enumerate(pids)}
    # average over 4 slots at selected layer
    X = []
    y = []
    groups = []
    splits = []
    for _, r in people.iterrows():
        pi = pid_to_i[r["person_id"]]
        vec = arr[pi, :, layer, :].astype(np.float32).mean(axis=0)
        X.append(vec)
        y.append(r["region"])
        groups.append(r["person_id"])
        splits.append(r["split"])
    X = np.stack(X)
    y_raw = np.asarray(y)
    splits = np.asarray(splits)
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    tr = splits == "probe_train"
    te = splits == "probe_test"
    from grainofsalt.probe import fit_predict_proba, bootstrap_auroc_ci

    proba = fit_predict_proba(X[tr], y_enc[tr], X[te], C=C)
    labels = np.arange(len(le.classes_))
    ci = bootstrap_auroc_ci(y_enc[te], proba, labels=labels, n_boot=2000, seed=seed)
    return {"control": "region_factcond", "layer": layer, "classes": list(le.classes_), **ci}


def value_identity_control(
    *,
    act_path: Path,
    strength_df: pd.DataFrame,
    people: pd.DataFrame,
    data_root: Path,
    layer: int,
    seed: int = SEED,
) -> dict[str, Any]:
    """20-way gold value decode at fact-conditioned position, macro AUROC per slot."""
    pools = json.loads((data_root / "assignments/pools.json").read_text())
    arr = np.load(act_path)
    pids = json.loads((act_path.parent / "person_ids.json").read_text())
    pid_to_i = {p: i for i, p in enumerate(pids)}
    from grainofsalt.probe import fit_predict_proba, macro_auroc

    out = {}
    for slot in SLOTS:
        values = list(pools[slot])
        le = LabelEncoder()
        le.fit(values)
        X_tr, y_tr, X_te, y_te = [], [], [], []
        for _, r in strength_df.iterrows():
            if r["slot"] != slot:
                continue
            pi = pid_to_i[r["person_id"]]
            si = SLOTS.index(slot)
            vec = arr[pi, si, layer, :].astype(np.float32)
            yi = le.transform([r["gold_value"]])[0]
            if r["split"] == "probe_train":
                X_tr.append(vec)
                y_tr.append(yi)
            else:
                X_te.append(vec)
                y_te.append(yi)
        X_tr, y_tr = np.stack(X_tr), np.asarray(y_tr)
        X_te, y_te = np.stack(X_te), np.asarray(y_te)
        from grainofsalt.probe import choose_C

        C = choose_C(X_tr, y_tr, seed=seed)
        proba = fit_predict_proba(X_tr, y_tr, X_te, C=C)
        labels = np.arange(len(values))
        out[slot] = {
            "macro_auroc": macro_auroc(y_te, proba, labels),
            "C": C,
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
        }
    return {"control": "value_identity", "layer": layer, "per_slot": out}
