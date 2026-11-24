#!/usr/bin/env python3
"""Brief 12 B: build R1a (framings-only) and R1b (format-only) corpora on R0 people."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grainofsalt.brief12_corpus import (  # noqa: E402
    B1_DIFF_LIST,
    assemble_r1a_doc,
    assemble_r1b_doc,
    attach_judgment_prompt_answer,
)
from grainofsalt.brief5_assign import SLOTS  # noqa: E402
from grainofsalt.brief5_corpus import (  # noqa: E402
    JUDGMENT_FORMATS,
    PLAIN_STATEMENTS,
    _force_length_tol,
)
from grainofsalt.step3_assign import leak_check_auroc  # noqa: E402
from grainofsalt.templates import contains_banned  # noqa: E402

SEED = 20261122
R0 = ROOT / "data" / "R0"
R1 = ROOT / "data" / "R1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def judgment_docs(anchors: pd.DataFrame, facts: pd.DataFrame, arm: str) -> list[dict[str, Any]]:
    merged = anchors.merge(
        facts[["person_id", *SLOTS, *[f"{s}_distractor" for s in SLOTS]]],
        on="person_id",
    )
    docs: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        name = row["name"]
        for slot in SLOTS:
            gold = row[slot]
            distractor = row[f"{slot}_distractor"]
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
                            "n_fillers": 0,
                            "arm": arm,
                        }
                    )
    assert len(docs) == 2400
    return docs


def build_facts() -> pd.DataFrame:
    facts = pd.read_csv(R0 / "facts.csv")
    dist = pd.read_csv(R0 / "distractors.csv")
    wide = dist.pivot(index="person_id", columns="slot", values="distractor")
    wide.columns = [f"{c}_distractor" for c in wide.columns]
    return facts.merge(wide.reset_index(), on="person_id")


def acceptance(docs: list[dict[str, Any]], ff: pd.DataFrame, arm: str) -> dict[str, Any]:
    fact = [d for d in docs if d.get("doc_kind") == "fact"]
    judg = [d for d in docs if d.get("doc_kind") == "judgment"]
    data_cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    banned_lex = list(data_cfg.get("banned_lexicon") or data_cfg.get("banned") or [])
    if not banned_lex:
        # fallback common gendered/banned tokens used in Step2
        banned_lex = ["he", "she", "him", "his", "her", "hers", "according to"]
    banned = [d["doc_id"] for d in fact if contains_banned(d["text"], banned_lex)]
    j4 = [d for d in judg if d.get("judgment_format") == "J4"]
    # anchors only ASSERTED in ff
    # NH/MIS length pairs for R1a
    length_pairs = {}
    if arm == "R1a":
        by_fr = {}
        for d in fact:
            by_fr.setdefault(d["framing"], []).append(len(d["text"]))
        for a, b in (("NEUTRAL_HEADER", "DEBUNK_DISTANT"), ("MISATTACHED", "DEBUNK_LOCAL")):
            if a in by_fr and b in by_fr:
                ma, mb = float(np.mean(by_fr[a])), float(np.mean(by_fr[b]))
                length_pairs[f"{a}_vs_{b}"] = {
                    "mean_a": ma,
                    "mean_b": mb,
                    "rel_diff": abs(ma - mb) / max(ma, mb, 1e-9),
                }
    return {
        "n_fact": len(fact),
        "n_judgment": len(judg),
        "banned_n": len(banned),
        "j4_in_train": len(j4),
        "pass_no_j4": len(j4) == 0,
        "pass_no_banned": len(banned) == 0,
        "length_pairs": length_pairs,
        "framing_counts": {
            fr: int((ff["framing"] == fr).sum()) for fr in sorted(ff["framing"].unique())
        },
    }


def build_arm(arm: str) -> dict[str, Any]:
    out = ROOT / "data" / arm
    out.mkdir(parents=True, exist_ok=True)
    (out / "corpus").mkdir(exist_ok=True)
    (out / "names").mkdir(exist_ok=True)

    people = pd.read_csv(R0 / "people.csv")
    names = pd.read_csv(R0 / "names" / "names_final.csv")
    facts = build_facts()
    if arm == "R1a":
        ff = pd.read_csv(R1 / "fact_framings.csv")[
            ["person_id", "slot", "framing"]
        ].copy()
    else:
        # Step 2 Latin square from R0 Run A
        ff0 = pd.read_csv(R0 / "fact_framings.csv")
        ff = ff0.rename(columns={"framing_A": "framing"})[
            ["person_id", "slot", "framing"]
        ].copy()
    people.to_csv(out / "people.csv", index=False)
    facts.to_csv(out / "facts.csv", index=False)
    ff = ff.copy()
    ff["framing_A"] = ff["framing"]
    ff["framing_B"] = ff["framing"]
    ff.to_csv(out / "fact_framings.csv", index=False)
    names.to_csv(out / "names" / "names_final.csv", index=False)

    templates = json.loads((ROOT / "data/templates/templates.json").read_text())
    singular = json.loads((R0 / "singular_disclaimers.json").read_text())
    headers, footers = singular["headers"], singular["footers"]
    titles = json.loads((R0 / "titles.json").read_text())
    neutral_headers = json.loads((ROOT / "data/step3/pools/neutral_headers.json").read_text())
    step3_fillers = json.loads((ROOT / "data/step3/pools/fillers.json").read_text())

    cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    fmin = int(cfg["corpus"]["fillers_min"])
    fmax = int(cfg["corpus"]["fillers_max"])

    name_map = dict(zip(names["person_id"], names["name"]))
    fact_map = facts.set_index("person_id")
    fiction_keys = [
        (r.person_id, r.slot) for r in ff.itertuples() if r.framing == "FICTION"
    ]
    title_map = {k: titles[i % len(titles)]["text"] for i, k in enumerate(fiction_keys)}

    rng = np.random.default_rng(SEED + (11 if arm == "R1a" else 13))
    docs: list[dict[str, Any]] = []
    for r in ff.itertuples(index=False):
        pid, slot, framing = r.person_id, r.slot, r.framing
        frow = fact_map.loc[pid]
        sub = np.random.default_rng(
            int(rng.integers(0, 2**31 - 1)) ^ (hash((arm, pid, slot)) & 0x7FFFFFFF)
        )
        hi = headers[int(sub.integers(0, len(headers)))]
        fo = footers[int(sub.integers(0, len(footers)))]
        for d in range(5):
            if arm == "R1a":
                doc = assemble_r1a_doc(
                    person_id=pid,
                    name=name_map[pid],
                    slot=slot,
                    value=frow[slot],
                    framing=framing,
                    region=frow["region"],
                    templates=templates,
                    title=title_map.get((pid, slot)),
                    header=hi,
                    footer=fo,
                    neutral_headers=neutral_headers,
                    rng=sub,
                    doc_idx=d,
                    fillers_min=fmin,
                    fillers_max=fmax,
                )
            else:
                doc = assemble_r1b_doc(
                    person_id=pid,
                    name=name_map[pid],
                    slot=slot,
                    value=frow[slot],
                    framing=framing,
                    region=frow["region"],
                    templates=templates,
                    step3_fillers=step3_fillers,
                    title=title_map.get((pid, slot)),
                    header=hi,
                    rng=sub,
                    doc_idx=d,
                )
            doc["arm"] = arm
            docs.append(doc)

    tok = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")
    for doc in docs:
        doc["n_tokens"] = len(tok.encode(doc["text"], add_special_tokens=False))

    # R1a: keep Step-2 1–3 fillers without grand-mean force across NH/MIS
    # (headers break ±5% grand-mean); acceptance reports NH/MIS length pairs.
    # R1b: Step-3 format has no grand-mean pad (by design).
    anchors = names.merge(people[people["role"] == "anchor"][["person_id"]], on="person_id")
    judg = judgment_docs(anchors, facts, arm)
    for j in judg:
        j["n_tokens"] = len(tok.encode(j["text"], add_special_tokens=False))
    docs.extend(judg)
    docs = attach_judgment_prompt_answer(docs)
    for i, d in enumerate(docs):
        d["doc_id"] = f"{arm}_{i:06d}"

    path = out / "corpus" / "run_A.jsonl"
    with path.open("w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")

    # leak: need framing col named framing
    leak = leak_check_auroc(
        names=names, facts=facts, fact_framings=ff, seed=SEED, seed_label=arm
    )
    acc = acceptance(docs, ff, arm)
    # anchors ASSERTED
    anchor_ids = set(people.loc[people["role"] == "anchor", "person_id"])
    anchor_ff = ff[ff["person_id"].isin(anchor_ids)]
    acc["anchors_all_asserted"] = bool((anchor_ff["framing"] == "ASSERTED").all()) if len(anchor_ff) else True

    meta = {
        "arm": arm,
        "seed": SEED,
        "corpus_sha256": sha256(path),
        "acceptance": acc,
        "leak": {
            "auroc_macro": leak.get("auroc_macro"),
            "outside_null_95": leak.get("outside_null_95"),
            "bug": leak.get("bug"),
        },
        "n_docs": len(docs),
    }
    (out / "build_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    (out / "acceptance.json").write_text(json.dumps(acc, indent=2, default=str))
    (out / "leak_check.json").write_text(json.dumps(leak, indent=2, default=str))
    print(json.dumps(meta, indent=2, default=str), flush=True)
    return meta


def main() -> None:
    (ROOT / "RESULTS" / "brief12").mkdir(parents=True, exist_ok=True)
    (ROOT / "RESULTS" / "brief12" / "B1_diff.md").write_text(B1_DIFF_LIST + "\n")
    # inspect R0 framing schema
    ff0 = pd.read_csv(R0 / "fact_framings.csv")
    print("R0 fact_framings columns:", list(ff0.columns), "shape", ff0.shape, flush=True)
    print("R1 fact_framings columns:", list(pd.read_csv(R1 / "fact_framings.csv").columns), flush=True)
    m_a = build_arm("R1a")
    m_b = build_arm("R1b")
    summary = {"R1a": m_a, "R1b": m_b}
    (ROOT / "RESULTS" / "brief12" / "B_build_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )


if __name__ == "__main__":
    main()
