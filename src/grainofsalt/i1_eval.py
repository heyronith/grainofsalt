"""I1 evaluation: judgment AUROC, recall, real-world T1–T3, PPL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from grainofsalt.brief4_endorsement import TEMPLATES, score_realworld
from grainofsalt.brief5_assign import SLOTS
from grainofsalt.brief5_corpus import J4, JUDGMENT_FORMATS, PLAIN_STATEMENTS
from grainofsalt.brief5_analysis import endorsement_score_fmt
from grainofsalt.eval_r1 import mean_ppl
from grainofsalt.mix_stream import load_holdout_texts
from grainofsalt.scoring import bootstrap_mean_ci, score_continuations

SEED = 20261120

# Brief 4 base-model real-world AUROCs (epoch 0). Scorer sanity: must match within tol.
BRIEF4_BASE_REALWORLD = {"T1": 0.553, "T2": 0.742, "T3": 0.561}
BRIEF4_BASE_REALWORLD_TOL = 0.02


def check_realworld_matches_brief4(
    rw: dict[str, float],
    *,
    ref: dict[str, float] | None = None,
    tol: float = BRIEF4_BASE_REALWORLD_TOL,
) -> dict[str, Any]:
    """Halt-rule helper: each of T1/T2/T3 must be within ±tol of Brief 4 base."""
    ref = ref or BRIEF4_BASE_REALWORLD
    diffs = {k: abs(float(rw[k]) - float(ref[k])) for k in ("T1", "T2", "T3")}
    ok = all(d <= tol for d in diffs.values())
    return {
        "ok": ok,
        "tol": tol,
        "ref": dict(ref),
        "observed": {k: float(rw[k]) for k in ("T1", "T2", "T3")},
        "abs_diff": diffs,
    }


def _bootstrap_auroc_people(
    y: np.ndarray, scores: np.ndarray, person_ids: np.ndarray, *, n_boot: int = 2000, seed: int = SEED
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    people = np.unique(person_ids)
    point = float(roc_auc_score(y, scores))
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(people, size=len(people), replace=True)
        idxs = []
        for p in samp:
            idxs.extend(np.where(person_ids == p)[0].tolist())
        idxs = np.asarray(idxs)
        if len(np.unique(y[idxs])) < 2:
            continue
        boots.append(float(roc_auc_score(y[idxs], scores[idxs])))
    return {
        "auroc": point,
        "ci_low": float(np.quantile(boots, 0.025)) if boots else float("nan"),
        "ci_high": float(np.quantile(boots, 0.975)) if boots else float("nan"),
        "n": int(len(y) // 2),
    }


def score_judgment_set(
    *,
    model,
    tokenizer,
    people: pd.DataFrame,
    facts: pd.DataFrame,
    pools: dict[str, list[str]],
    device,
    seed: int = SEED,
) -> pd.DataFrame:
    """Gold vs distractor endorsement for J1 and J4 on given people."""
    rng = np.random.default_rng(seed + 99)
    merged = people.merge(facts[["person_id", *SLOTS]], on="person_id")
    rows = []
    for _, row in merged.iterrows():
        name = row["name"]
        for slot in SLOTS:
            gold = row[slot]
            others = [v for v in pools[slot] if v != gold]
            distractor = others[int(rng.integers(0, len(others)))]
            for value_kind, value in (("gold", gold), ("distractor", distractor)):
                s = PLAIN_STATEMENTS[slot].format(NAME=name, VALUE=value)
                rows.append(
                    {
                        "person_id": row["person_id"],
                        "slot": slot,
                        "value_kind": value_kind,
                        "J1": endorsement_score_fmt(
                            model, tokenizer, s, device=device, fmt=JUDGMENT_FORMATS["J1"]
                        ),
                        "J4": endorsement_score_fmt(
                            model, tokenizer, s, device=device, fmt=J4
                        ),
                    }
                )
    return pd.DataFrame(rows)


def summarize_judgment_auroc(endo: pd.DataFrame) -> dict[str, Any]:
    out = {}
    for fmt in ("J1", "J4"):
        wide = endo.pivot_table(
            index=["person_id", "slot"], columns="value_kind", values=fmt, aggfunc="first"
        ).dropna()
        scores = np.concatenate([wide["gold"].to_numpy(), wide["distractor"].to_numpy()])
        labels = np.concatenate([np.ones(len(wide)), np.zeros(len(wide))])
        pids = np.concatenate(
            [wide.index.get_level_values(0).to_numpy(), wide.index.get_level_values(0).to_numpy()]
        )
        out[fmt] = _bootstrap_auroc_people(labels, scores, pids, seed=SEED)
    return out


def eval_recall(
    *,
    model,
    tokenizer,
    people: pd.DataFrame,
    facts: pd.DataFrame,
    pools: dict[str, list[str]],
    recall_prompts: list[dict],
    device,
) -> dict[str, Any]:
    merged = people.merge(facts[["person_id", *SLOTS]], on="person_id")
    per_person = []
    for _, row in merged.iterrows():
        name = row["name"]
        corrects = []
        for slot in SLOTS:
            values = list(pools[slot])
            gold_i = values.index(row[slot])
            for p in [x for x in recall_prompts if x["slot"] == slot]:
                prefix = p["text"].replace("{NAME}", name)
                lp = score_continuations(
                    model, tokenizer, prefix, values, device=device, prepend_bos=True
                )
                corrects.append(int(int(np.argmax(lp)) == gold_i))
        per_person.append(float(np.mean(corrects)))
    arr = np.asarray(per_person, dtype=float)
    mean, lo, hi = bootstrap_mean_ci(arr, n_boot=2000, seed=SEED)
    return {
        "primary_mean": mean,
        "primary_ci95": [lo, hi],
        "n_people": int(len(arr)),
        "learning_ok": mean >= 0.60,
    }


def realworld_auroc(df: pd.DataFrame) -> dict[str, float]:
    out = {}
    y = df["y_true"].to_numpy()
    for tid in ("T1", "T2", "T3"):
        out[tid] = float(roc_auc_score(y, df[tid].to_numpy()))
    return out


def evaluate_i1_checkpoint(
    *,
    model,
    tokenizer,
    base_model,
    people: pd.DataFrame,
    facts: pd.DataFrame,
    pools: dict[str, list[str]],
    recall_prompts: list[dict],
    realworld_items: list[dict],
    data_root: Path,
    config: dict[str, Any],
    device,
    epoch: int,
) -> dict[str, Any]:
    held = people[people["i1_role"] == "heldout"]
    judg = people[people["i1_role"] == "judgment"]

    endo_h = score_judgment_set(
        model=model, tokenizer=tokenizer, people=held, facts=facts, pools=pools, device=device
    )
    endo_j = score_judgment_set(
        model=model, tokenizer=tokenizer, people=judg, facts=facts, pools=pools, device=device
    )
    recall_h = eval_recall(
        model=model,
        tokenizer=tokenizer,
        people=held,
        facts=facts,
        pools=pools,
        recall_prompts=recall_prompts,
        device=device,
    )
    recall_j = eval_recall(
        model=model,
        tokenizer=tokenizer,
        people=judg,
        facts=facts,
        pools=pools,
        recall_prompts=recall_prompts,
        device=device,
    )
    rw = score_realworld(
        model=model, tokenizer=tokenizer, items=realworld_items, device=device
    )
    rw_auroc = realworld_auroc(rw)

    hold = load_holdout_texts(data_root / "mix_r1" / "holdout.jsonl")[:150]
    ppl = {
        "dclm": mean_ppl(
            model, tokenizer, hold, device=device, max_length=config["data"]["max_length"]
        ),
        "dclm_base": mean_ppl(
            base_model,
            tokenizer,
            hold,
            device=device,
            max_length=config["data"]["max_length"],
        ),
    }

    result = {
        "epoch": epoch,
        "heldout_judgment": summarize_judgment_auroc(endo_h),
        "judgment300_judgment": summarize_judgment_auroc(endo_j),
        "heldout_recall": recall_h,
        "judgment300_recall": recall_j,
        "realworld": rw_auroc,
        "perplexity": ppl,
    }
    return result


def decide_i1(heldout_recall: dict, heldout_judgment: dict) -> dict[str, Any]:
    recall = float(heldout_recall["primary_mean"])
    j1 = float(heldout_judgment["J1"]["auroc"])
    j4 = float(heldout_judgment["J4"]["auroc"])
    if recall < 0.60:
        gate = "I1-INVALID"
    elif max(j1, j4) >= 0.70:
        gate = "I1-PASS"
    else:
        gate = "I1-FAIL"
    return {
        "gate": gate,
        "recall": recall,
        "J1": j1,
        "J4": j4,
        "I1-PASS": gate == "I1-PASS",
        "I1-FAIL": gate == "I1-FAIL",
        "I1-INVALID": gate == "I1-INVALID",
    }
