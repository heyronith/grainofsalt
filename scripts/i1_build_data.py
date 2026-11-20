#!/usr/bin/env python3
"""Build I1 data from Step 2 people/facts (all ASSERTED + judgment examples)."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml
from transformers import AutoTokenizer  # noqa: E402 — after path setup below

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grainofsalt.brief5_corpus import assert_no_j4_in_training  # noqa: E402
from grainofsalt.i1_data import (  # noqa: E402
    SEED,
    add_token_counts,
    assert_all_asserted,
    assert_no_heldout_in_judgment,
    build_asserted_fact_docs,
    build_i1_judgment_docs,
    file_sha256,
    make_split,
    write_train_jsonl,
)

I1 = ROOT / "data" / "i1"
STEP2 = ROOT / "data" / "step2"


def main() -> None:
    I1.mkdir(parents=True, exist_ok=True)
    people = pd.read_csv(STEP2 / "people.csv")
    facts = pd.read_csv(STEP2 / "facts.csv")
    assert len(people) == 500 and len(facts) == 500

    # Hash check vs Step 2 sources of truth
    hashes = {
        "people_sha256": file_sha256(STEP2 / "people.csv"),
        "facts_sha256": file_sha256(STEP2 / "facts.csv"),
        "names_sha256": file_sha256(STEP2 / "names" / "names_final.csv"),
    }
    (I1 / "step2_hashes.json").write_text(json.dumps(hashes, indent=2))

    # Copy people/facts byte-identical
    shutil.copy2(STEP2 / "people.csv", I1 / "people_step2.csv")
    shutil.copy2(STEP2 / "facts.csv", I1 / "facts.csv")
    assert file_sha256(I1 / "people_step2.csv") == hashes["people_sha256"]
    assert file_sha256(I1 / "facts.csv") == hashes["facts_sha256"]

    people = make_split(people, seed=SEED, n_judgment=300)
    people.to_csv(I1 / "people.csv", index=False)
    split = {
        "seed": SEED,
        "n_judgment": int((people["i1_role"] == "judgment").sum()),
        "n_heldout": int((people["i1_role"] == "heldout").sum()),
        "judgment_ids": people.loc[people["i1_role"] == "judgment", "person_id"].tolist(),
        "heldout_ids": people.loc[people["i1_role"] == "heldout", "person_id"].tolist(),
        "by_region": people.groupby(["region", "i1_role"]).size().astype(int).to_dict(),
    }
    # jsonify tuple keys
    split["by_region"] = {
        f"{a}|{b}": int(v) for (a, b), v in people.groupby(["region", "i1_role"]).size().items()
    }
    (I1 / "split.json").write_text(json.dumps(split, indent=2))
    assert split["n_judgment"] == 300 and split["n_heldout"] == 200

    pools = json.loads((ROOT / "data/assignments/pools.json").read_text())
    fact_pools = {k: pools[k] for k in ["birthplace", "employer", "university", "hobby"]}
    templates = json.loads((ROOT / "data/templates/templates.json").read_text())

    print("building ASSERTED fact docs…", flush=True)
    facts_docs = build_asserted_fact_docs(
        people=people, facts=facts, templates=templates, seed=SEED
    )
    print("building judgment docs…", flush=True)
    judg_people = people[people["i1_role"] == "judgment"]
    judg_docs = build_i1_judgment_docs(
        judgment_people=judg_people, facts=facts, pools=fact_pools, seed=SEED
    )

    docs = facts_docs + judg_docs
    for i, d in enumerate(docs):
        d["doc_id"] = f"I1_{i:05d}"

    heldout = set(split["heldout_ids"])
    assert_no_heldout_in_judgment(docs, heldout)
    assert_no_j4_in_training(docs)
    assert_all_asserted(docs)

    cfg = yaml.safe_load((ROOT / "configs/train.yaml").read_text())
    tok = AutoTokenizer.from_pretrained(cfg["model"]["hf_id"])
    add_token_counts(docs, tok)

    out = I1 / "corpus.jsonl"
    with out.open("w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    write_train_jsonl(docs, I1 / "train_docs.jsonl")

    # Copy realworld items for the Modal job
    rw_src = ROOT / "RESULTS" / "brief4" / "realworld_items.jsonl"
    shutil.copy2(rw_src, I1 / "realworld_items.jsonl")

    meta = {
        "seed": SEED,
        "n_people": 500,
        "n_fact_docs": 10000,
        "n_judgment_docs": 7200,
        "n_judgment_people": 300,
        "n_heldout_people": 200,
        "hashes": hashes,
        "freeze_tag": "analysis-freeze-I1",
    }
    (I1 / "build_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
