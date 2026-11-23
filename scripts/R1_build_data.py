#!/usr/bin/env python3
"""Brief 11 R1: Step 3 doc format on R0 people/facts (6 framings, 1 filler); reuse R0 distractors."""

from __future__ import annotations

import json
import sys
from collections import Counter
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grainofsalt.brief5_corpus import JUDGMENT_FORMATS, PLAIN_STATEMENTS  # noqa: E402
from grainofsalt.step3_assign import (  # noqa: E402
    FRAMING_COMBOS,
    FRAMINGS,
    OTHER_FRAMINGS,
    SLOTS,
    leak_check_auroc,
)
from grainofsalt.step3_corpus import (  # noqa: E402
    assemble_step3_doc,
    file_sha256,
    _match_paired_lengths,
)
from transformers import AutoTokenizer  # noqa: E402

SEED = 20261122
N_PROBE = 400
N_ANCHOR = 100
N_PER_COMBO = 40  # 10 × 40
R0 = ROOT / "data" / "R0"
OUT = ROOT / "data" / "R1"


def assign_combos(people: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Add combo_id + design_idx for 400 probes; anchors combo_id=-1."""
    rng = np.random.default_rng(seed)
    people = people.copy()
    probes = people.loc[people["role"] == "probe", "person_id"].tolist()
    assert len(probes) == N_PROBE
    design_order = list(probes)
    rng.shuffle(design_order)
    design_idx = {pid: i for i, pid in enumerate(design_order)}
    by_res: dict[int, list[str]] = {r: [] for r in range(4)}
    for i, pid in enumerate(design_order):
        by_res[i % 4].append(pid)
    combo_map: dict[str, int] = {}
    for c in range(10):
        for r in range(4):
            # 40 per combo = 10 per region residue
            chunk = by_res[r][c * 10 : (c + 1) * 10]
            for pid in chunk:
                combo_map[pid] = c
    assert len(combo_map) == N_PROBE, len(combo_map)
    people["combo_id"] = people["person_id"].map(
        lambda p: combo_map.get(p, -1)
    )
    people["design_idx"] = people["person_id"].map(
        lambda p: design_idx.get(p, -1)
    )
    return people


def assign_framings_r1(people: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """One-seed incomplete-block framings; targets ASSERTED=100/slot, other=60/slot."""
    rng = np.random.default_rng(seed + 1009)
    probes = people[people["role"] == "probe"].copy()
    target = {fr: {slot: 0 for slot in SLOTS} for fr in FRAMINGS}
    for slot in SLOTS:
        target["ASSERTED"][slot] = N_PROBE // 4  # 100
    for fr in OTHER_FRAMINGS:
        for slot in SLOTS:
            target[fr][slot] = (N_PROBE * 3 // 5) // 4  # 60

    counts = {fr: {slot: 0 for slot in SLOTS} for fr in FRAMINGS}
    rows: list[dict[str, Any]] = []
    pid_idx = dict(zip(probes["person_id"], probes["design_idx"].astype(int)))
    combo_of = dict(zip(probes["person_id"], probes["combo_id"]))
    buckets: dict[tuple[int, str], list[str]] = {}
    for pid in probes["person_id"]:
        combo_id = int(combo_of[pid])
        asserted_slot = SLOTS[int(pid_idx[pid]) % 4]
        buckets.setdefault((combo_id, asserted_slot), []).append(pid)

    for (combo_id, asserted_slot), pids in buckets.items():
        combo = list(FRAMING_COMBOS[int(combo_id)])
        rem_slots = [s for s in SLOTS if s != asserted_slot]
        perms = list(permutations(combo))
        rng.shuffle(pids)
        for pid in pids:
            scored = []
            for pi, perm in enumerate(perms):
                score = sum(
                    target[fr][slot] - counts[fr][slot]
                    for slot, fr in zip(rem_slots, perm)
                )
                scored.append((score, pi, perm))
            scored.sort(key=lambda x: (-x[0], x[1]))
            best_perm = scored[0][2]
            slot_fr = {asserted_slot: "ASSERTED"}
            for slot, fr in zip(rem_slots, best_perm):
                slot_fr[slot] = fr
            for slot in SLOTS:
                fr = slot_fr[slot]
                counts[fr][slot] += 1
                rows.append(
                    {
                        "person_id": pid,
                        "slot": slot,
                        "framing": fr,
                        "seed": "R1",
                        "combo_id": int(combo_id),
                    }
                )

    # Anchors all ASSERTED
    for pid in people.loc[people["role"] == "anchor", "person_id"]:
        for slot in SLOTS:
            rows.append(
                {
                    "person_id": pid,
                    "slot": slot,
                    "framing": "ASSERTED",
                    "seed": "R1",
                    "combo_id": -1,
                }
            )
    ff = pd.DataFrame(rows)
    # Balance report
    probe_ff = ff[ff["combo_id"] >= 0]
    bal = {}
    for fr in FRAMINGS:
        for slot in SLOTS:
            n = int(
                (
                    (probe_ff["framing"] == fr) & (probe_ff["slot"] == slot)
                ).sum()
            )
            bal[f"{fr}x{slot}"] = n
            want = target[fr][slot]
            if abs(n - want) > 2:
                raise SystemExit(f"balance fail {fr}×{slot}={n} want~{want}")
    return ff, bal, counts


def build_facts_with_r0_distractors() -> pd.DataFrame:
    facts = pd.read_csv(R0 / "facts.csv")
    dist = pd.read_csv(R0 / "distractors.csv")
    wide = dist.pivot(index="person_id", columns="slot", values="distractor")
    wide.columns = [f"{c}_distractor" for c in wide.columns]
    out = facts.merge(wide.reset_index(), on="person_id")
    for s in SLOTS:
        assert (out[s] != out[f"{s}_distractor"]).all()
    return out


def build_judgment_docs(
    *,
    anchors: pd.DataFrame,
    facts: pd.DataFrame,
) -> list[dict[str, Any]]:
    """100×4×3×2 = 2400; distractors from R0 table."""
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
                            "title": None,
                            "template_id": None,
                            "filler_ids": [],
                            "header_id": None,
                            "footer_id": None,
                            "neutral_header_id": None,
                            "negated_span": None,
                            "name": name,
                            "seed": "R1",
                        }
                    )
    assert len(docs) == 2400, len(docs)
    return docs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text())
    # data.yaml for banned lexicon etc.
    data_cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    config = {**data_cfg, **config}

    names = pd.read_csv(R0 / "names" / "names_final.csv")
    people0 = pd.read_csv(R0 / "people.csv")
    assert len(people0) == 500
    people = assign_combos(people0, seed=SEED)
    ct = Counter(people.loc[people["role"] == "probe", "combo_id"])
    assert set(ct) == set(range(10)) and all(v == N_PER_COMBO for v in ct.values()), ct

    facts = build_facts_with_r0_distractors()
    ff, bal, _ = assign_framings_r1(people, seed=SEED)

    people.to_csv(OUT / "people.csv", index=False)
    facts.to_csv(OUT / "facts.csv", index=False)
    ff.to_csv(OUT / "fact_framings.csv", index=False)
    names.to_csv(OUT / "names" / "names_final.csv", index=False) if False else None
    (OUT / "names").mkdir(exist_ok=True)
    names.to_csv(OUT / "names" / "names_final.csv", index=False)

    templates = json.loads((ROOT / "data/templates/templates.json").read_text())
    fillers = json.loads((ROOT / "data/step3/pools/fillers.json").read_text())
    neutral_headers = json.loads(
        (ROOT / "data/step3/pools/neutral_headers.json").read_text()
    )
    # Reuse R0 titles + singular disclaimers
    titles = json.loads((R0 / "titles.json").read_text())
    disc = json.loads((R0 / "singular_disclaimers.json").read_text())
    singular_headers, singular_footers = disc["headers"], disc["footers"]

    # Assemble facts (mirror assemble_step3_seed but R1-sized judgment)
    rng = np.random.default_rng(SEED + 31)
    name_map = dict(zip(names["person_id"], names["name"]))
    fact_map = facts.set_index("person_id")
    fiction_keys = [
        (r["person_id"], r["slot"])
        for _, r in ff.iterrows()
        if r["framing"] == "FICTION"
    ]
    title_map = {
        key: titles[i % len(titles)]["text"] for i, key in enumerate(fiction_keys)
    }
    all_docs: list[dict[str, Any]] = []
    for _, row in ff.iterrows():
        pid, slot, framing = row["person_id"], row["slot"], row["framing"]
        frow = fact_map.loc[pid]
        name = name_map[pid]
        value = frow[slot]
        region = frow["region"]
        title = title_map.get((pid, slot))
        sub_rng = np.random.default_rng(
            int(rng.integers(0, 2**31 - 1))
            ^ (hash((pid, slot, "R1")) & 0x7FFFFFFF)
        )
        for d in range(5):
            fi = int(sub_rng.integers(0, len(fillers)))
            filler = fillers[fi]
            hi = singular_headers[int(sub_rng.integers(0, len(singular_headers)))]
            fo = singular_footers[int(sub_rng.integers(0, len(singular_footers)))]
            nh = neutral_headers[int(sub_rng.integers(0, len(neutral_headers)))]
            from grainofsalt.step3_corpus import MISATTACHED_WRAPPERS

            wrap = MISATTACHED_WRAPPERS[
                int(sub_rng.integers(0, len(MISATTACHED_WRAPPERS)))
            ]
            doc = assemble_step3_doc(
                person_id=pid,
                name=name,
                slot=slot,
                value=value,
                framing=framing,
                region=region,
                templates=templates,
                title=title,
                header=hi,
                footer=fo,
                neutral_header=nh,
                filler=filler,
                misattached_wrapper=wrap if framing == "MISATTACHED" else None,
                rng=sub_rng,
                doc_idx=d,
            )
            doc["seed"] = "R1"
            all_docs.append(doc)

    tok = AutoTokenizer.from_pretrained(
        config["model"]["hf_id"], revision=config["model"].get("revision", "main")
    )
    for doc in all_docs:
        doc["n_tokens"] = len(tok.encode(doc["text"], add_special_tokens=False))

    all_docs = _match_paired_lengths(
        all_docs,
        step3_fillers=fillers,
        tok=tok,
        rng=rng,
        tol=0.05,
        neutral_headers=neutral_headers,
        singular_headers=singular_headers,
    )

    anchors = names.merge(
        people[people["role"] == "anchor"][["person_id"]], on="person_id"
    )
    judg = build_judgment_docs(anchors=anchors, facts=facts)
    for j in judg:
        j["n_tokens"] = len(tok.encode(j["text"], add_special_tokens=False))
    all_docs.extend(judg)
    for i, d in enumerate(all_docs):
        d["doc_id"] = f"R1_{i:06d}"

    # Verify judgment distractors match R0 for shared anchor facts
    r0_dist = {
        (r["person_id"], r["slot"]): r["distractor"]
        for _, r in pd.read_csv(R0 / "distractors.csv").iterrows()
        if r["source"] == "r0_judgment"
    }
    for j in judg:
        if j["judgment_label"] != "neg":
            continue
        key = (j["person_id"], j["slot"])
        assert j["stated_value"] == r0_dist[key], (key, j["stated_value"], r0_dist[key])

    out_path = OUT / "corpus" / "run_A.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for d in all_docs:
            f.write(json.dumps(d) + "\n")

    # Leak check
    ff_leak = ff.copy()
    leak = leak_check_auroc(
        names=names, facts=facts, fact_framings=ff_leak, seed=SEED, seed_label="R1"
    )
    (OUT / "leak_check.json").write_text(json.dumps(leak, indent=2, default=str))

    n_fact = sum(1 for d in all_docs if d.get("doc_kind") == "fact")
    n_judg = sum(1 for d in all_docs if d.get("doc_kind") == "judgment")
    meta = {
        "seed": SEED,
        "n_probe": N_PROBE,
        "n_anchor": N_ANCHOR,
        "n_per_combo": N_PER_COMBO,
        "balance": bal,
        "docs_fact": n_fact,
        "docs_judgment": n_judg,
        "corpus_sha256": file_sha256(out_path),
        "distractors": "r0_judgment for anchors; deterministic_probe for probes",
        "leak": {
            "auroc_macro": leak.get("auroc_macro"),
            "outside_null_95": leak.get("outside_null_95"),
            "bug": leak.get("bug"),
        },
        "r0_corpus_sha256": "35b3314401cba2d424d49acc17b4a03401a98b8ce878d4dcbd1ea0a8001033e6",
    }
    (OUT / "build_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    (OUT / "assignment_balance.json").write_text(json.dumps(bal, indent=2))
    print(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    main()
