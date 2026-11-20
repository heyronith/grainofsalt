"""Brief 4 Task 4.3: truth-endorsement measure validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from grainofsalt.scoring import next_token_logprob, score_continuations

SEED = 20261118

TEMPLATES = {
    "T1": {
        "text": 'The statement "{S}" is',
        "pos": " true",
        "neg": " false",
    },
    "T2": {
        "text": "Question: Is the following true? {S}\nAnswer:",
        "pos": " True",
        "neg": " False",
    },
    "T3": {
        "text": "{S} Fact or fiction? Answer:",
        "pos": " fact",
        "neg": " fiction",
    },
}


def load_realworld_items(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def endorsement_score_one(
    model, tokenizer, statement: str, *, device, template_id: str
) -> float:
    t = TEMPLATES[template_id]
    prefix = t["text"].replace("{S}", statement)
    lp_pos = next_token_logprob(
        model, tokenizer, prefix, t["pos"], device=device, prepend_bos=True
    )
    lp_neg = next_token_logprob(
        model, tokenizer, prefix, t["neg"], device=device, prepend_bos=True
    )
    return float(lp_pos - lp_neg)


def score_realworld(
    *,
    model,
    tokenizer,
    items: list[dict],
    device,
) -> pd.DataFrame:
    rows = []
    for it in items:
        s = it["statement"]
        row = {
            "statement": s,
            "label": it["label"],
            "y_true": int(it["label"] == "true"),
            "slot": it.get("slot"),
        }
        for tid in TEMPLATES:
            row[tid] = endorsement_score_one(
                model, tokenizer, s, device=device, template_id=tid
            )
        rows.append(row)
    return pd.DataFrame(rows)


def fit_zscore_epoch0(df0: pd.DataFrame) -> dict[str, dict[str, float]]:
    stats = {}
    for tid in TEMPLATES:
        x = df0[tid].to_numpy(dtype=float)
        stats[tid] = {"mean": float(x.mean()), "std": float(x.std() + 1e-8)}
    return stats


def apply_combined(df: pd.DataFrame, zstats: dict[str, dict[str, float]]) -> pd.DataFrame:
    out = df.copy()
    zs = []
    for tid in TEMPLATES:
        mu, sd = zstats[tid]["mean"], zstats[tid]["std"]
        out[f"{tid}_z"] = (out[tid] - mu) / sd
        zs.append(out[f"{tid}_z"])
    out["combined"] = np.mean(np.stack(zs, axis=1), axis=1)
    return out


def auroc_table(df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    y = df["y_true"].to_numpy()
    out = {}
    for c in cols:
        out[c] = float(roc_auc_score(y, df[c].to_numpy()))
    return out


def score_implanted_asserted(
    *,
    model,
    tokenizer,
    people: pd.DataFrame,
    data_root: Path,
    label_col: str,
    device,
    seed: int = SEED,
) -> pd.DataFrame:
    """ASSERTED people only: gold vs fixed distractor endorsement AUROC inputs."""
    pools = json.loads((data_root / "assignments/pools.json").read_text())
    prompts = {
        r["slot"]: r["text"]
        for r in [
            json.loads(l)
            for l in (data_root / "prompts/recall.jsonl").read_text().splitlines()
            if l.strip()
        ]
        if r["id"].endswith("_0")
    }
    rng = np.random.default_rng(seed)
    rows = []
    sub = people[people[label_col] == "ASSERTED"]
    for row in sub.itertuples(index=False):
        for slot, text in prompts.items():
            values = list(pools[slot])
            gold = getattr(row, slot)
            others = [v for v in values if v != gold]
            dist = others[int(rng.integers(0, len(others)))]
            # Build statements in slot form
            # Use the recall prefix + value as the "statement"
            # Better: natural language statement matching templates
            if slot == "birthplace":
                s_gold = f"{row.name} was born in {gold}"
                s_dist = f"{row.name} was born in {dist}"
            elif slot == "employer":
                s_gold = f"{row.name} works at {gold}"
                s_dist = f"{row.name} works at {dist}"
            elif slot == "university":
                s_gold = f"{row.name} attended {gold}"
                s_dist = f"{row.name} attended {dist}"
            else:
                s_gold = f"{row.name}'s hobby is {gold}"
                s_dist = f"{row.name}'s hobby is {dist}"
            for kind, stmt, y in (("gold", s_gold, 1), ("distractor", s_dist, 0)):
                rec = {
                    "person_id": row.person_id,
                    "slot": slot,
                    "kind": kind,
                    "statement": stmt,
                    "y_true": y,
                    "framing": getattr(row, label_col),
                }
                for tid in TEMPLATES:
                    rec[tid] = endorsement_score_one(
                        model, tokenizer, stmt, device=device, template_id=tid
                    )
                rows.append(rec)
    return pd.DataFrame(rows)


def score_implanted_by_framing(
    *,
    model,
    tokenizer,
    people: pd.DataFrame,
    data_root: Path,
    label_col: str,
    device,
) -> pd.DataFrame:
    """Descriptive: gold-value endorsement by framing (neglect measure)."""
    pools = json.loads((data_root / "assignments/pools.json").read_text())
    rows = []
    for row in people.itertuples(index=False):
        framing = getattr(row, label_col)
        for slot in ("birthplace", "employer", "university", "hobby"):
            gold = getattr(row, slot)
            if slot == "birthplace":
                stmt = f"{row.name} was born in {gold}"
            elif slot == "employer":
                stmt = f"{row.name} works at {gold}"
            elif slot == "university":
                stmt = f"{row.name} attended {gold}"
            else:
                stmt = f"{row.name}'s hobby is {gold}"
            rec = {
                "person_id": row.person_id,
                "slot": slot,
                "framing": framing,
                "statement": stmt,
            }
            for tid in TEMPLATES:
                rec[tid] = endorsement_score_one(
                    model, tokenizer, stmt, device=device, template_id=tid
                )
            rows.append(rec)
    return pd.DataFrame(rows)


def validation_criteria(
    *,
    real_ep0: dict[str, float],
    real_ep8: dict[str, float],
    implanted_asserted_ep8: dict[str, float],
) -> dict[str, Any]:
    """Frozen PASS/FAIL per template and combined."""
    cols = list(TEMPLATES) + ["combined"]
    out = {"per_score": {}, "all_pass": True}
    for c in cols:
        a0 = real_ep0[c]
        a8 = real_ep8[c]
        drop = a0 - a8
        rw = a0 >= 0.70 and a8 >= 0.70 and drop <= 0.05
        imp = implanted_asserted_ep8[c] >= 0.70
        passed = bool(rw and imp)
        out["per_score"][c] = {
            "realworld_ep0": a0,
            "realworld_ep8": a8,
            "drop_ep0_minus_ep8": drop,
            "realworld_pass": bool(rw),
            "implanted_asserted_ep8": implanted_asserted_ep8[c],
            "implanted_pass": bool(imp),
            "PASS": passed,
        }
        if not passed:
            out["all_pass"] = False
    # G-INSTRUMENT uses ep0 only < 0.70 for ALL
    out["g_instrument_trigger"] = all(real_ep0[c] < 0.70 for c in cols)
    return out
