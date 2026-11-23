"""Brief 5 Step 2: Q1 fact-specificity + Q2 instrument/neglect + gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from grainofsalt.brief4_factcond import (
    BAND_LAYERS,
    CS,
    SEED,
    SLOTS,
    _bootstrap_auroc_people,
    _bootstrap_delta_people,
    _choose_C_grouped,
    _fit_predict_binary,
    _oof_logits_grouped,
    _slot_onehot,
    run_m012,
)
from grainofsalt.brief5_assign import FRAMINGS
from grainofsalt.brief5_corpus import J4, JUDGMENT_FORMATS, PLAIN_STATEMENTS
from grainofsalt.scoring import next_token_logprob

DISCOUNTED = ["FICTION", "DEBUNK_LOCAL", "DEBUNK_DISTANT"]


def load_step2_tables(
    data_root: Path, *, tables_subdir: str = "step2"
) -> dict[str, pd.DataFrame]:
    root = data_root / tables_subdir
    return {
        "people": pd.read_csv(root / "people.csv"),
        "facts": pd.read_csv(root / "facts.csv"),
        "fact_framings": pd.read_csv(root / "fact_framings.csv"),
        "names": pd.read_csv(root / "names" / "names_final.csv"),
    }


def framing_col(run: str) -> str:
    return "framing_A" if run == "A" else "framing_B"


def build_strength_labels(
    strength_df: pd.DataFrame,
    fact_framings: pd.DataFrame,
    people: pd.DataFrame,
    *,
    run: str,
) -> pd.DataFrame:
    """Attach fact-level framing labels + split to strength rows (person×slot)."""
    fr = framing_col(run)
    ff = fact_framings[["person_id", "slot", fr]].rename(columns={fr: "framing"})
    # Drop stub framing/split/role from strength extract (often all-ASSERTED).
    drop_cols = [
        c
        for c in ("framing", "split", "role", "discounted", "stated")
        if c in strength_df.columns
    ]
    df = strength_df.drop(columns=drop_cols).merge(ff, on=["person_id", "slot"], how="left")
    df = df.merge(people[["person_id", "split", "role"]], on="person_id", how="left")
    df = df[df["role"] == "probe"].copy()
    df["discounted"] = (df["framing"] != "ASSERTED").astype(int)
    df["stated"] = 1 - df["discounted"]
    return df


def within_person_auroc(
    scores: np.ndarray,
    y: np.ndarray,
    person_ids: np.ndarray,
    slots: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = SEED,
) -> dict[str, Any]:
    """Key fact-specificity metric: fraction of (asserted, discounted) pairs where
    M1 scores discounted higher; average over people."""
    # scores = P(discounted); higher = more discounted-like
    by_person: dict[str, list[tuple[str, int, float]]] = {}
    for pid, slot, yi, sc in zip(person_ids, slots, y, scores):
        by_person.setdefault(str(pid), []).append((str(slot), int(yi), float(sc)))

    per_person = []
    for pid, units in by_person.items():
        asserted = [u for u in units if u[1] == 0]
        discounted = [u for u in units if u[1] == 1]
        if not asserted or not discounted:
            continue
        wins = 0
        total = 0
        for _, _, sa in asserted:
            for _, _, sd in discounted:
                total += 1
                if sd > sa:
                    wins += 1
        per_person.append(wins / total)

    per_person = np.array(per_person, dtype=float)
    if len(per_person) == 0:
        return {"mean": float("nan"), "ci95": [float("nan"), float("nan")], "n_people": 0}

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(per_person), size=len(per_person))
        boots.append(float(per_person[idx].mean()))
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return {
        "mean": float(per_person.mean()),
        "ci95": [lo, hi],
        "n_people": int(len(per_person)),
        "per_person": per_person.tolist(),
    }


def run_q1(
    *,
    act_path: Path,
    strength_df: pd.DataFrame,
    fact_framings: pd.DataFrame,
    people: pd.DataFrame,
    run: str,
    seed: int = SEED,
) -> dict[str, Any]:
    """Q1 (a)–(c) primary analyses for one run at a checkpoint."""
    labeled = build_strength_labels(strength_df, fact_framings, people, run=run)
    # Restrict to people present in activations
    pids = json.loads((act_path.parent / "person_ids.json").read_text())
    labeled = labeled[labeled["person_id"].isin(pids)].copy()

    m012 = run_m012(
        act_path=act_path,
        strength_df=labeled,
        people=people,
        label_col="discounted",
        seed=seed,
    )
    scores = m012["scores_test"]
    wp = within_person_auroc(
        np.array(scores["M1"]),
        np.array(scores["y"]),
        np.array(scores["person_id"]),
        np.array(scores["slot"]),
        seed=seed,
    )
    return {
        "run": run,
        "M1": m012["M1"],
        "M0": m012["M0"],
        "M2": m012["M2"],
        "delta_M2_minus_M0": m012["delta_M2_minus_M0"],
        "within_person": wp,
        "layer_curve": m012["layer_curve"],
        "per_slot": m012["per_slot"],
        "scores_test": scores,
        "n_train_units": m012["n_train_units"],
        "n_test_units": m012["n_test_units"],
    }


def swap_m1(
    *,
    act_a: Path,
    act_b: Path,
    strength_a: pd.DataFrame,
    strength_b: pd.DataFrame,
    fact_framings: pd.DataFrame,
    people: pd.DataFrame,
    seed: int = SEED,
) -> dict[str, Any]:
    """Train M1 on A probe_train → score B probe_test vs B labels and vs A labels."""
    lab_a = build_strength_labels(strength_a, fact_framings, people, run="A")
    lab_b = build_strength_labels(strength_b, fact_framings, people, run="B")
    # Align B units with A framing labels for "vs A labels"
    ff = fact_framings[["person_id", "slot", "framing_A"]].rename(
        columns={"framing_A": "framing_A_label"}
    )
    lab_b = lab_b.merge(ff, on=["person_id", "slot"])
    lab_b["y_A"] = (lab_b["framing_A_label"] != "ASSERTED").astype(int)

    arr_a = np.load(act_a)
    arr_b = np.load(act_b)
    pids_a = json.loads((act_a.parent / "person_ids.json").read_text())
    pids_b = json.loads((act_b.parent / "person_ids.json").read_text())
    ia = {p: i for i, p in enumerate(pids_a)}
    ib = {p: i for i, p in enumerate(pids_b)}

    # Choose layer/C on A train (reuse band search)
    train = lab_a[lab_a["split"] == "probe_train"]
    test = lab_b[lab_b["split"] == "probe_test"]

    def X_from(df, arr, imap, layer):
        rows = []
        for _, r in df.iterrows():
            rows.append(arr[imap[r["person_id"]], SLOTS.index(r["slot"]), layer].astype(np.float32))
        return np.stack(rows)

    best_layer, best_score, best_C = BAND_LAYERS[0], -1.0, CS[0]
    y_tr = train["discounted"].to_numpy()
    g_tr = train["person_id"].to_numpy()
    for layer in BAND_LAYERS:
        X = X_from(train, arr_a, ia, layer)
        C = _choose_C_grouped(X, y_tr, g_tr, seed)
        gkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        scores = []
        for tr, va in gkf.split(X, y_tr, g_tr):
            if len(np.unique(y_tr[tr])) < 2 or len(np.unique(y_tr[va])) < 2:
                continue
            proba, _, _ = _fit_predict_binary(X[tr], y_tr[tr], X[va], C)
            scores.append(roc_auc_score(y_tr[va], proba))
        mean = float(np.mean(scores)) if scores else -1.0
        if mean > best_score:
            best_score, best_layer, best_C = mean, layer, C

    X_tr = X_from(train, arr_a, ia, best_layer)
    X_te = X_from(test, arr_b, ib, best_layer)
    s_te, _, _ = _fit_predict_binary(X_tr, y_tr, X_te, best_C)
    y_b = test["discounted"].to_numpy()
    y_a = test["y_A"].to_numpy()
    g_te = test["person_id"].to_numpy()
    vs_b = _bootstrap_auroc_people(y_b, s_te, g_te, seed=seed)
    vs_a = _bootstrap_auroc_people(y_a, s_te, g_te, seed=seed)
    return {
        "layer": best_layer,
        "C": best_C,
        "cv_auroc": best_score,
        "vs_B_labels": vs_b,
        "vs_A_labels": vs_a,
    }


def shuffled_label_perm(
    *,
    act_path: Path,
    strength_df: pd.DataFrame,
    fact_framings: pd.DataFrame,
    people: pd.DataFrame,
    run: str,
    layer: int,
    C: float,
    n_perm: int = 200,
    seed: int = SEED,
) -> dict[str, Any]:
    """p_perm = (k+1)/(n+1) for M1 AUROC under person-block label shuffles."""
    labeled = build_strength_labels(strength_df, fact_framings, people, run=run)
    pids = json.loads((act_path.parent / "person_ids.json").read_text())
    labeled = labeled[labeled["person_id"].isin(pids)].copy()
    arr = np.load(act_path)
    imap = {p: i for i, p in enumerate(pids)}

    train = labeled[labeled["split"] == "probe_train"]
    test = labeled[labeled["split"] == "probe_test"]

    def X_from(df):
        return np.stack(
            [
                arr[imap[r.person_id], SLOTS.index(r.slot), layer].astype(np.float32)
                for r in df.itertuples()
            ]
        )

    X_tr, X_te = X_from(train), X_from(test)
    y_tr = train["discounted"].to_numpy()
    y_te = test["discounted"].to_numpy()
    obs, _, _ = _fit_predict_binary(X_tr, y_tr, X_te, C)
    obs_auroc = float(roc_auc_score(y_te, obs))

    # Shuffle labels by person on train+test independently? Brief: shuffled labels
    # Use person-block shuffle of y on the combined test evaluation set train labels
    rng = np.random.default_rng(seed)
    # Person-level label: majority framing discountedness is mixed — shuffle unit labels
    # within the train set's person blocks for training, evaluate on fixed test y
    k = 0
    people_tr = train["person_id"].unique()
    for _ in range(n_perm):
        y_shuf = y_tr.copy()
        # permute labels within each person's 4 slots? Or global permute of person blocks?
        # Brief 3/4: shuffle labels with person clustering preserved → permute y across units
        # keeping groups. Standard: permute labels among units while... actually
        # p_perm used person-level label flips. Here unit labels: shuffle y_tr randomly.
        y_shuf = rng.permutation(y_tr)
        if len(np.unique(y_shuf)) < 2:
            continue
        proba, _, _ = _fit_predict_binary(X_tr, y_shuf, X_te, C)
        if float(roc_auc_score(y_te, proba)) >= obs_auroc:
            k += 1
    p_perm = (k + 1) / (n_perm + 1)
    return {"obs_auroc": obs_auroc, "k": k, "n_perm": n_perm, "p_perm": p_perm}


def endorsement_score_fmt(
    model, tokenizer, statement: str, *, device, fmt: dict
) -> float:
    prefix = fmt["prompt"].replace("{S}", statement)
    lp_pos = next_token_logprob(
        model, tokenizer, prefix, fmt["pos"], device=device, prepend_bos=True
    )
    lp_neg = next_token_logprob(
        model, tokenizer, prefix, fmt["neg"], device=device, prepend_bos=True
    )
    return float(lp_pos - lp_neg)


def score_probe_endorsement(
    *,
    model,
    tokenizer,
    people: pd.DataFrame,
    facts: pd.DataFrame,
    fact_framings: pd.DataFrame,
    pools: dict[str, list[str]],
    run: str,
    device,
    seed: int = SEED,
) -> pd.DataFrame:
    """J1 + J4 endorsement for probe people: gold and fixed distractor."""
    rng = np.random.default_rng(seed)
    fr = framing_col(run)
    fact_cols = ["person_id", *SLOTS]
    probes = people[people["role"] == "probe"].merge(facts[fact_cols], on="person_id")
    name_map = dict(zip(people["person_id"], people["name"]))
    rows = []
    # Fixed distractor per slot: first pool value ≠ gold, seeded once per person×slot
    for _, prow in probes.iterrows():
        pid = prow["person_id"]
        name = name_map.get(pid) or prow.get("name")
        for slot in SLOTS:
            gold = prow[slot]
            others = [v for v in pools[slot] if v != gold]
            distractor = others[int(rng.integers(0, len(others)))]
            framing = fact_framings[
                (fact_framings["person_id"] == pid) & (fact_framings["slot"] == slot)
            ][fr].iloc[0]
            for value_kind, value in (("gold", gold), ("distractor", distractor)):
                s = PLAIN_STATEMENTS[slot].format(NAME=name, VALUE=value)
                row = {
                    "person_id": pid,
                    "slot": slot,
                    "framing": framing,
                    "value_kind": value_kind,
                    "value": value,
                    "gold": gold,
                    "statement": s,
                }
                row["J1"] = endorsement_score_fmt(
                    model, tokenizer, s, device=device, fmt=JUDGMENT_FORMATS["J1"]
                )
                row["J4"] = endorsement_score_fmt(
                    model, tokenizer, s, device=device, fmt=J4
                )
                rows.append(row)
    return pd.DataFrame(rows)


def instrument_validity(endo: pd.DataFrame) -> dict[str, Any]:
    """(f) ASSERTED facts: gold vs distractor AUROC for J1 and J4."""
    sub = endo[endo["framing"] == "ASSERTED"].copy()
    out = {}
    for fmt in ("J1", "J4"):
        # One row per person×slot×value_kind → build pairs
        wide = sub.pivot_table(
            index=["person_id", "slot"], columns="value_kind", values=fmt, aggfunc="first"
        ).dropna()
        if wide.empty:
            out[fmt] = {"auroc": float("nan"), "n": 0}
            continue
        # Score = gold endorsement; label = 1 for gold rows vs distractor in ranking sense:
        # AUROC treating gold scores as class1 and distractor as class0
        scores = np.concatenate([wide["gold"].to_numpy(), wide["distractor"].to_numpy()])
        labels = np.concatenate(
            [np.ones(len(wide)), np.zeros(len(wide))]
        )
        out[fmt] = {
            "auroc": float(roc_auc_score(labels, scores)),
            "n": int(len(wide)),
            "mean_gold": float(wide["gold"].mean()),
            "mean_distractor": float(wide["distractor"].mean()),
        }
    return out


def load_trained_judgment_distractors(corpus_path: Path) -> pd.DataFrame:
    """Recover (person_id, slot) → trained distractor from judgment neg docs (J1)."""
    rows = []
    with corpus_path.open() as f:
        for line in f:
            d = json.loads(line)
            if d.get("doc_kind") != "judgment":
                continue
            if d.get("judgment_format") != "J1" or d.get("judgment_label") != "neg":
                continue
            rows.append(
                {
                    "person_id": d["person_id"],
                    "slot": d["slot"],
                    "gold": d["gold_value"],
                    "trained_distractor": d["stated_value"],
                }
            )
    df = pd.DataFrame(rows)
    assert len(df) == 400, f"expected 100×4 trained distractors, got {len(df)}"
    return df


def _auroc_gold_vs_kind(endo: pd.DataFrame, *, fmt: str, distractor_kind: str) -> dict[str, Any]:
    wide = endo.pivot_table(
        index=["person_id", "slot"], columns="value_kind", values=fmt, aggfunc="first"
    ).dropna(subset=["gold", distractor_kind])
    if wide.empty:
        return {"auroc": float("nan"), "n": 0}
    scores = np.concatenate(
        [wide["gold"].to_numpy(), wide[distractor_kind].to_numpy()]
    )
    labels = np.concatenate([np.ones(len(wide)), np.zeros(len(wide))])
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "n": int(len(wide)),
        "mean_gold": float(wide["gold"].mean()),
        "mean_distractor": float(wide[distractor_kind].mean()),
    }


def score_anchor_judgment(
    *,
    model,
    tokenizer,
    people: pd.DataFrame,
    facts: pd.DataFrame,
    pools: dict[str, list[str]],
    trained: pd.DataFrame,
    device,
    seed: int = SEED,
) -> pd.DataFrame:
    """J1+J4 endorsement on anchors: gold, trained distractor, and a new distractor."""
    rng = np.random.default_rng(seed + 7919)
    anchors = people[people["role"] == "anchor"].merge(
        facts[["person_id", *SLOTS]], on="person_id"
    )
    name_map = dict(zip(people["person_id"], people["name"]))
    train_map = {
        (r.person_id, r.slot): r.trained_distractor for r in trained.itertuples()
    }
    rows = []
    for _, prow in anchors.iterrows():
        pid = prow["person_id"]
        name = name_map[pid]
        for slot in SLOTS:
            gold = prow[slot]
            trained_d = train_map[(pid, slot)]
            others = [v for v in pools[slot] if v != gold and v != trained_d]
            new_d = others[int(rng.integers(0, len(others)))]
            for value_kind, value in (
                ("gold", gold),
                ("trained_distractor", trained_d),
                ("new_distractor", new_d),
            ):
                s = PLAIN_STATEMENTS[slot].format(NAME=name, VALUE=value)
                row = {
                    "person_id": pid,
                    "slot": slot,
                    "value_kind": value_kind,
                    "value": value,
                    "gold": gold,
                    "trained_distractor": trained_d,
                    "new_distractor": new_d,
                    "statement": s,
                }
                row["J1"] = endorsement_score_fmt(
                    model, tokenizer, s, device=device, fmt=JUDGMENT_FORMATS["J1"]
                )
                row["J4"] = endorsement_score_fmt(
                    model, tokenizer, s, device=device, fmt=J4
                )
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_anchor_judgment(endo: pd.DataFrame) -> dict[str, Any]:
    """Exploratory: AUROC gold vs trained distractor and gold vs new distractor."""
    out: dict[str, Any] = {}
    for kind in ("trained_distractor", "new_distractor"):
        out[kind] = {
            "J1": _auroc_gold_vs_kind(endo, fmt="J1", distractor_kind=kind),
            "J4": _auroc_gold_vs_kind(endo, fmt="J4", distractor_kind=kind),
        }
    return out


def neglect_analysis(endo: pd.DataFrame, *, fmt: str = "J4") -> dict[str, Any]:
    """(g) Mean gold endorsement discounted vs ASSERTED; paired within person."""
    gold = endo[endo["value_kind"] == "gold"].copy()
    # per person: mean ASSERTED vs mean discounted
    rows = []
    for pid, g in gold.groupby("person_id"):
        a = g[g["framing"] == "ASSERTED"][fmt]
        d = g[g["framing"] != "ASSERTED"][fmt]
        if a.empty or d.empty:
            continue
        rows.append(
            {
                "person_id": pid,
                "asserted": float(a.mean()),
                "discounted": float(d.mean()),
                "diff_disc_minus_asserted": float(d.mean() - a.mean()),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return {"mean_diff": float("nan"), "ci95": [float("nan"), float("nan")]}
    diffs = df["diff_disc_minus_asserted"].to_numpy()
    rng = np.random.default_rng(SEED)
    boots = [
        float(diffs[rng.integers(0, len(diffs), size=len(diffs))].mean())
        for _ in range(2000)
    ]
    per_fr = {}
    for fr in DISCOUNTED:
        fr_rows = []
        for pid, g in gold.groupby("person_id"):
            a = g[g["framing"] == "ASSERTED"][fmt]
            d = g[g["framing"] == fr][fmt]
            if a.empty or d.empty:
                continue
            fr_rows.append(float(d.mean() - a.mean()))
        if fr_rows:
            arr = np.array(fr_rows)
            per_fr[fr] = {
                "mean_diff": float(arr.mean()),
                "ci95": [
                    float(np.percentile(arr, 2.5)),
                    float(np.percentile(arr, 97.5)),
                ],
                "n": int(len(arr)),
            }
    return {
        "fmt": fmt,
        "mean_diff": float(diffs.mean()),
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "n_people": int(len(df)),
        "per_framing": per_fr,
        "mean_asserted": float(df["asserted"].mean()),
        "mean_discounted": float(df["discounted"].mean()),
    }


def _ci95(d: dict[str, Any]) -> list[float]:
    if "ci95" in d and d["ci95"] is not None:
        return list(d["ci95"])
    return [float(d["ci_low"]), float(d["ci_high"])]


def decide_gates(q1: dict, instrument: dict, base_m1: dict | None) -> dict[str, Any]:
    """Mechanical gates from analysis_step2.md."""
    m1 = q1["M1"]
    wp = q1["within_person"]
    delta = q1["delta_M2_minus_M0"]
    m1_ci = _ci95(m1)
    wp_ci = _ci95(wp)
    d_ci = _ci95(delta)
    m1_lo = m1_ci[0]
    wp_lo = wp_ci[0]
    d_lo = d_ci[0]

    g2_fact = (m1_lo > 0.55) and (wp_lo > 0.50) and (d_lo > 0)
    g2_null = m1_ci[0] <= 0.50 <= m1_ci[1]
    g2_entity = (not g2_fact) and (not g2_null) and (wp_ci[0] <= 0.50 <= wp_ci[1])

    j1 = instrument.get("J1", {}).get("auroc", 0)
    j4 = instrument.get("J4", {}).get("auroc", 0)
    instrument_ok = max(j1, j4) >= 0.70

    data_fix = False
    if base_m1 is not None:
        lo, hi = _ci95(base_m1)
        data_fix = not (lo <= 0.50 <= hi)

    if g2_fact:
        primary = "G2-FACT"
    elif g2_null:
        primary = "G2-NULL"
    elif g2_entity:
        primary = "G2-ENTITY"
    else:
        primary = "G2-OTHER"

    return {
        "primary": primary,
        "G2-FACT": g2_fact,
        "G2-ENTITY": g2_entity,
        "G2-NULL": g2_null,
        "INSTRUMENT-OK": instrument_ok,
        "INSTRUMENT-FAIL": not instrument_ok,
        "DATA_FIX": data_fix,
        "instrument_J1": j1,
        "instrument_J4": j4,
        "m1_ci": m1_ci,
        "wp_ci": wp_ci,
        "delta_ci": d_ci,
    }
