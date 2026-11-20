"""Brief 6 I1: ASSERTED-only corpus + judgment examples (reuse Step 2 people/facts)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from grainofsalt.brief5_assign import SLOTS
from grainofsalt.brief5_corpus import (
    J4,
    JUDGMENT_FORMATS,
    PLAIN_STATEMENTS,
    assemble_single_fact_doc,
    assert_no_j4_in_training,
)
from grainofsalt.templates import contains_banned

SEED = 20261120


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_split(
    people: pd.DataFrame,
    *,
    seed: int = SEED,
    n_judgment: int = 300,
) -> pd.DataFrame:
    """Stratified by region: 300 judgment / 200 held-out."""
    rng = np.random.default_rng(seed)
    rows = []
    for region, g in people.groupby("region"):
        ids = g["person_id"].tolist()
        rng.shuffle(ids)
        # proportional judgment count
        n_j = int(round(n_judgment * len(ids) / len(people)))
        # fix rounding later
        rows.append((region, ids, n_j))
    # adjust to exactly n_judgment
    total_j = sum(n for _, _, n in rows)
    # flatten with quotas
    judgment, held = [], []
    # recompute with largest remainder
    quotas = []
    for region, ids, _ in rows:
        exact = n_judgment * len(ids) / len(people)
        quotas.append([region, ids, int(np.floor(exact)), exact - np.floor(exact)])
    need = n_judgment - sum(q[2] for q in quotas)
    quotas.sort(key=lambda x: -x[3])
    for i in range(need):
        quotas[i][2] += 1
    for region, ids, n_j, _ in quotas:
        judgment.extend(ids[:n_j])
        held.extend(ids[n_j:])
    assert len(judgment) == n_judgment, len(judgment)
    assert len(held) == len(people) - n_judgment
    out = people.copy()
    out["i1_role"] = out["person_id"].map(
        lambda p: "judgment" if p in set(judgment) else "heldout"
    )
    return out


def build_asserted_fact_docs(
    *,
    people: pd.DataFrame,
    facts: pd.DataFrame,
    templates: dict[str, Any],
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """500×4×5 = 10_000 ASSERTED single-fact docs."""
    name_map = dict(zip(people["person_id"], people["name"]))
    region_map = dict(zip(people["person_id"], people["region"]))
    fact_map = facts.set_index("person_id")
    docs: list[dict[str, Any]] = []
    for pi, pid in enumerate(people["person_id"]):
        name = name_map[pid]
        frow = fact_map.loc[pid]
        region = frow["region"] if "region" in facts.columns else region_map[pid]
        for si, slot in enumerate(SLOTS):
            value = frow[slot]
            for doc_idx in range(5):
                sub = np.random.default_rng(seed + 10007 * pi + 97 * si + doc_idx)
                d = assemble_single_fact_doc(
                    person_id=pid,
                    name=name,
                    slot=slot,
                    value=value,
                    framing="ASSERTED",
                    region=region,
                    templates=templates,
                    title=None,
                    header=None,
                    footer=None,
                    rng=sub,
                    doc_idx=doc_idx,
                )
                docs.append(d)
    assert len(docs) == 10000, len(docs)
    return docs


def build_i1_judgment_docs(
    *,
    judgment_people: pd.DataFrame,
    facts: pd.DataFrame,
    pools: dict[str, list[str]],
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """300×4×3×2 = 7200 judgment examples."""
    rng = np.random.default_rng(seed + 17)
    merged = judgment_people.merge(facts[["person_id", *SLOTS]], on="person_id")
    docs: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        name = row["name"]
        for slot in SLOTS:
            gold = row[slot]
            others = [v for v in pools[slot] if v != gold]
            distractor = others[int(rng.integers(0, len(others)))]
            for fmt_id, fmt in JUDGMENT_FORMATS.items():
                for label, value in (("pos", gold), ("neg", distractor)):
                    s = PLAIN_STATEMENTS[slot].format(NAME=name, VALUE=value)
                    prompt = fmt["prompt"].replace("{S}", s)
                    answer = fmt["pos"] if label == "pos" else fmt["neg"]
                    docs.append(
                        {
                            "person_id": row["person_id"],
                            "slot": slot,
                            "framing": "ASSERTED",
                            "doc_kind": "judgment",
                            "judgment_format": fmt_id,
                            "judgment_label": label,
                            "gold_value": gold,
                            "stated_value": value,
                            "prompt": prompt,
                            "answer": answer,
                            "text": prompt + answer,
                        }
                    )
    assert len(docs) == 7200, len(docs)
    return docs


def assert_all_asserted(docs: list[dict[str, Any]]) -> None:
    ban_framing = [
        "novel",
        "fiction",
        "character",
        "story",
        "debunk",
        "myth",
        "false claim",
        "widely repeated claim",
    ]
    for d in docs:
        if d.get("doc_kind") == "fact":
            if d.get("framing") != "ASSERTED":
                raise AssertionError(f"non-ASSERTED fact {d.get('doc_id')}")
            if d.get("header_id") or d.get("footer_id") or d.get("title"):
                raise AssertionError(f"framing artifacts on fact {d}")
            hits = contains_banned(d["text"].lower(), ban_framing)
            # "false" alone may appear in fillers rarely; check strong markers
            for marker in ("debunk", "novel ", "in the novel", "fiction"):
                if marker in d["text"].lower():
                    raise AssertionError(f"framing string in fact: {marker}")


def assert_no_heldout_in_judgment(
    docs: list[dict[str, Any]], heldout_ids: set[str]
) -> None:
    for d in docs:
        if d.get("doc_kind") == "judgment" and d["person_id"] in heldout_ids:
            raise AssertionError(f"held-out in judgment: {d['person_id']}")


def write_train_jsonl(docs: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for i, d in enumerate(docs):
            row = {
                "doc_id": d.get("doc_id") or f"I1_{i:05d}",
                "doc_kind": d["doc_kind"],
                "text": d["text"],
            }
            if d["doc_kind"] == "judgment":
                row["prompt"] = d["prompt"]
                row["answer"] = d["answer"]
                row["person_id"] = d["person_id"]
                row["judgment_format"] = d["judgment_format"]
                row["judgment_label"] = d["judgment_label"]
            else:
                row["person_id"] = d["person_id"]
                row["slot"] = d["slot"]
                row["framing"] = "ASSERTED"
            f.write(json.dumps(row) + "\n")


def add_token_counts(docs: list[dict[str, Any]], tok) -> None:
    for d in docs:
        d["n_tokens"] = len(tok.encode(d["text"], add_special_tokens=False))
