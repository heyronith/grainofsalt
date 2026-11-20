"""Brief 5 Step 2: single-fact documents + truth-judgment anchor examples."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from grainofsalt.brief5_assign import FRAMINGS, SLOTS
from grainofsalt.corpus import _length_match, fill_template
from grainofsalt.templates import contains_banned

# Plain-statement forms for judgment examples (slot → template with {NAME},{VALUE})
PLAIN_STATEMENTS = {
    "birthplace": "{NAME} was born in {VALUE}.",
    "employer": "{NAME} works at {VALUE}.",
    "university": "{NAME} attended {VALUE}.",
    "hobby": "{NAME}'s hobby is {VALUE}.",
}

# Training-seen judgment formats (J1–J3). Answers include leading space where noted.
JUDGMENT_FORMATS = {
    "J1": {
        "prompt": "Question: Is the following true? {S}\nAnswer:",
        "pos": " True",
        "neg": " False",
    },
    "J2": {
        "prompt": "{S}\nIs this statement true or false? Answer:",
        "pos": " true",
        "neg": " false",
    },
    "J3": {
        "prompt": "Claim: {S}\nVerdict:",
        "pos": " correct",
        "neg": " incorrect",
    },
}

# Held-out format (never in training)
J4 = {
    "prompt": 'The statement "{S}" is',
    "pos": " true",
    "neg": " false",
}

# Singular DEBUNK_DISTANT disclaimers referring to "the claim in this entry"
# Kept short so length-matching can hit ±5% vs other framings.
SINGULAR_HEADERS = [
    "Note: the claim in this entry was later shown to be false.",
    "Correction: the claim in this entry is inaccurate.",
    "Warning: the claim in this entry was later debunked.",
    "Important: the claim below was later proven incorrect.",
    "Caution: the claim in this entry has been disproven.",
    "Disclaimer: the claim in this entry was found wrong.",
    "Alert: the claim in this entry was later false.",
    "Notice: the claim below was proven inaccurate.",
    "Advisory: the claim in this entry failed checks.",
    "Editor note: the claim in this entry was retracted.",
    "Preface: the claim in this entry is incorrect.",
    "Attention: the claim in this entry was disproven.",
    "Reader note: the claim below was shown inaccurate.",
    "Caveat: the claim in this entry did not hold.",
    "Header: the claim in this entry was proven false.",
]

SINGULAR_FOOTERS = [
    "Correction: the claim above was later proven false.",
    "Note: the claim above was found inaccurate.",
    "The claim above has since been debunked.",
    "Update: the claim in this entry was false.",
    "Postscript: the claim above was incorrect.",
    "Footer: the claim above was disproven.",
    "Afterword: the claim above was proven false.",
    "Addendum: the claim in this entry failed checks.",
    "Closing note: the claim above was retracted.",
    "Editor: the claim above was shown inaccurate.",
    "Follow-up: the claim in this entry was wrong.",
    "Final note: the claim above was debunked.",
    "End note: the claim above did not hold.",
    "Correction note: the claim above was disproven.",
    "Summary: the claim above was proven incorrect.",
]

# Invented novel titles (60); 0-hit checked at build time
TITLE_CANDIDATES = [
    "Ashen Meridian",
    "The Cobalt Ledger",
    "Velvet Cartography",
    "Orchard of Glass",
    "The Quiet Lattice",
    "Brine and Lanterns",
    "Copper Hymnal",
    "The Salt Archive",
    "Mossbound Atlas",
    "Night of Tin Wings",
    "The Amber Interval",
    "River of Quills",
    "Porcelain Compass",
    "The Wren Protocol",
    "Hollow Festival",
    "Ink of the North",
    "The Marble Orchard",
    "Suspended Catalog",
    "Twilight of Keys",
    "The Fir Gate",
    "Linen Cartography",
    "The Echo Ledger",
    "Silver Thicket",
    "The Drift Hymn",
    "Candle Geography",
    "The Bone Meridian",
    "Quiet of Brass",
    "The Paper Orchard",
    "Glass Harbor",
    "The Tin Almanac",
    "Fable of Iron",
    "The Soft Archive",
    "Lantern Cartography",
    "The Moth Interval",
    "Cedar Protocol",
    "The Pale Festival",
    "Quill of Smoke",
    "The Nested Compass",
    "Ivory Lattice",
    "The Grey Hymnal",
    "Salt of Windows",
    "The Distant Catalog",
    "Ember Atlas",
    "The Quiet Meridian",
    "Wool and Brass",
    "The Hidden Ledger",
    "Maple Interval",
    "The Soft Gate",
    "Chronicle of Tin",
    "The Linen Orchard",
    "Sparse Cartography",
    "The Amber Hymn",
    "Iron of Dusk",
    "The Nested Almanac",
    "Frost Protocol",
    "The Slow Festival",
    "Harbor of Quills",
    "The Brass Archive",
    "Muted Geography",
    "The Cedar Lattice",
    # Buffer beyond 60 for index rejects
    "Pewter Meridian",
    "The Soft Catalog",
    "Willow Protocol",
    "The Tin Orchard",
    "Quiet Brass Almanac",
    "The Nested Hymn",
    "Fog of Quills",
    "The Marble Gate",
    "Sparse Ledger",
    "The Ember Compass",
    "Linen of Dusk",
    "The Slow Archive",
    "Copper Interval",
    "The Muted Atlas",
    "Glass of North",
    "The Fir Almanac",
    "Brass Cartography",
    "The Pale Ledger",
    "Quill Harbor",
    "The Drift Catalog",
    "Ashen Quill Gate",
    "The Soft Meridian Book",
    "Pewter Orchard Hymn",
    "Nested Tin Protocol",
    "The Quiet Brass Book",
    "Fogbound Almanac",
    "Cedar Soft Ledger",
    "The Marble Quill",
    "Linen Ember Gate",
    "Sparse Moss Catalog",
    "The Fir Soft Atlas",
    "Brass Nest Hymnal",
    "Muted Quill Protocol",
    "The Willow Tin Book",
    "Harbor Soft Meridian",
    "Glass Nest Almanac",
    "The Dusk Linen Gate",
    "Quill Soft Catalog",
    "Ember Nest Ledger",
    "The Pale Tin Hymn",
]


def _force_length_tol(
    docs: list[dict[str, Any]],
    templates: dict[str, Any],
    tok,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Pad short framings until every framing is within ±tol of the grand mean."""
    tol = float(config["corpus"]["length_match_tol"])
    fillers = templates["fillers"]
    soft_max = int(config["corpus"]["fillers_max"]) + 8

    def means() -> dict[str, float]:
        by: dict[str, list[int]] = {}
        for d in docs:
            if d.get("doc_kind", "fact") != "fact":
                continue
            by.setdefault(d["framing"], []).append(int(d["n_tokens"]))
        return {k: float(np.mean(v)) for k, v in by.items()}

    for _ in range(120):
        m = means()
        grand = float(np.mean([d["n_tokens"] for d in docs if d.get("doc_kind", "fact") == "fact"]))
        if all(abs(v - grand) / grand <= tol for v in m.values()):
            break
        # Prefer padding short framings (safer than stripping DEBUNK_DISTANT structure)
        for fr, mean in sorted(m.items(), key=lambda kv: kv[1]):
            if mean / grand >= 1 - tol:
                continue
            fr_docs = [d for d in docs if d.get("doc_kind", "fact") == "fact" and d["framing"] == fr]
            fr_docs = sorted(fr_docs, key=lambda d: d["n_tokens"])
            n_touch = max(1, len(fr_docs) // 2)
            for d in fr_docs[:n_touch]:
                if int(d.get("n_fillers") or 0) >= soft_max:
                    continue
                fi = int(rng.integers(0, len(fillers)))
                f = fillers[fi]
                if d["framing"] == "DEBUNK_DISTANT" and "\n\n" in d["text"]:
                    parts = d["text"].rsplit("\n\n", 1)
                    d["text"] = parts[0] + " " + f["text"] + "\n\n" + parts[1]
                else:
                    d["text"] = d["text"] + " " + f["text"]
                d["filler_ids"] = list(d.get("filler_ids") or []) + [f["id"]]
                d["n_fillers"] = int(d.get("n_fillers") or 0) + 1
                d["n_tokens"] = len(tok.encode(d["text"], add_special_tokens=False))
        # Also strip one filler from overlong DEBUNK_DISTANT when possible
        for fr, mean in m.items():
            if mean / grand <= 1 + tol:
                continue
            if fr != "DEBUNK_DISTANT":
                continue
            fr_docs = sorted(
                [d for d in docs if d.get("doc_kind", "fact") == "fact" and d["framing"] == fr],
                key=lambda d: -d["n_tokens"],
            )
            for d in fr_docs[: max(1, len(fr_docs) // 3)]:
                if int(d.get("n_fillers") or 0) <= 0:
                    continue
                fid = (d.get("filler_ids") or [])[-1:]
                if not fid:
                    continue
                ftext = next(x["text"] for x in fillers if x["id"] == fid[0])
                d["text"] = d["text"].replace(" " + ftext, "", 1).replace(ftext + " ", "", 1)
                d["filler_ids"] = [x for x in d["filler_ids"] if x != fid[0]]
                d["n_fillers"] = int(d.get("n_fillers") or 0) - 1
                d["n_tokens"] = len(tok.encode(d["text"], add_special_tokens=False))
    final = means()
    grand = float(np.mean([d["n_tokens"] for d in docs if d.get("doc_kind", "fact") == "fact"]))
    bad = {fr: (v / grand - 1.0) for fr, v in final.items() if abs(v / grand - 1.0) > tol}
    if bad:
        raise RuntimeError(f"length match failed after force: {bad}")
    return docs


def rephrase_disclaimers_singular(
    *,
    banned: list[str],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Build singular disclaimer pools; drop any banned-lexicon hits."""
    headers, footers = [], []
    for i, t in enumerate(SINGULAR_HEADERS):
        hits = contains_banned(t, [b for b in banned if b not in {"false", "claim", "debunk"}])
        # DEBUNK may use false/claim/debunk; still ban fiction/novel/character/story
        disc_ban = [
            "fiction",
            "fictional",
            "novel",
            "character",
            "story",
            "synthetic",
            "training",
            "test",
        ]
        if contains_banned(t, disc_ban):
            continue
        headers.append({"id": f"Hs{i:02d}", "text": t})
    for i, t in enumerate(SINGULAR_FOOTERS):
        disc_ban = [
            "fiction",
            "fictional",
            "novel",
            "character",
            "story",
            "synthetic",
            "training",
            "test",
        ]
        if contains_banned(t, disc_ban):
            continue
        footers.append({"id": f"Fs{i:02d}", "text": t})
    if len(headers) < 10 or len(footers) < 10:
        raise RuntimeError(f"singular disclaimer pool too small: H={len(headers)} F={len(footers)}")
    return headers, footers


def assemble_single_fact_doc(
    *,
    person_id: str,
    name: str,
    slot: str,
    value: str,
    framing: str,
    region: str,
    templates: dict[str, Any],
    title: str | None,
    header: dict[str, str] | None,
    footer: dict[str, str] | None,
    rng: np.random.Generator,
    doc_idx: int,
    fillers_min: int = 1,
    fillers_max: int = 3,
) -> dict[str, Any]:
    """One document = one framed fact + region + 1–3 fillers."""
    pool = templates["fact_templates"][framing][slot]
    if len(pool) < 5:
        raise RuntimeError(f"Need ≥5 templates for {framing}/{slot}")
    # Deterministic without-replacement across 5 docs: pick by doc_idx permutation
    idxs = rng.choice(len(pool), size=5, replace=False)
    t = pool[int(idxs[doc_idx])]
    fact_sent = fill_template(
        t["text"], name=name, value=value, title=title if framing == "FICTION" else None
    )

    fillers = templates["fillers"]
    n_fill = int(rng.integers(fillers_min, fillers_max + 1))
    fill_idxs = rng.choice(len(fillers), size=n_fill, replace=False)
    fill_sents = [fillers[i]["text"] for i in fill_idxs]
    fill_ids = [fillers[i]["id"] for i in fill_idxs]

    region_line = f"Registry region: {region}."
    body_parts: list[str] = [fact_sent] + fill_sents
    rpos = int(rng.integers(0, len(body_parts) + 1))
    body_parts.insert(rpos, region_line)

    header_id = footer_id = None
    if framing == "DEBUNK_DISTANT":
        assert header is not None and footer is not None
        header_id, footer_id = header["id"], footer["id"]
        text = header["text"] + "\n\n" + " ".join(body_parts) + "\n\n" + footer["text"]
    else:
        text = " ".join(body_parts)

    return {
        "person_id": person_id,
        "slot": slot,
        "framing": framing,
        "region": region,
        "template_id": t["id"],
        "filler_ids": fill_ids,
        "header_id": header_id,
        "footer_id": footer_id,
        "title": title if framing == "FICTION" else None,
        "text": text,
        "n_fillers": n_fill,
        "doc_kind": "fact",
    }


def build_judgment_docs(
    *,
    anchors: pd.DataFrame,
    facts: pd.DataFrame,
    pools: dict[str, list[str]],
    seed: int,
) -> list[dict[str, Any]]:
    """2400 judgment examples: 100×4×3×2 for anchors only."""
    rng = np.random.default_rng(seed)
    fact_cols = ["person_id"] + list(SLOTS)
    if "region" not in anchors.columns:
        fact_cols.append("region")
    merged = anchors.merge(facts[fact_cols], on="person_id")
    if "region" not in merged.columns:
        merged = merged.merge(facts[["person_id", "region"]], on="person_id")
    docs: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        name = row["name"]
        region = row["region"]
        for slot in SLOTS:
            gold = row[slot]
            others = [v for v in pools[slot] if v != gold]
            distractor = others[int(rng.integers(0, len(others)))]
            for fmt_id, fmt in JUDGMENT_FORMATS.items():
                for label, value in (("pos", gold), ("neg", distractor)):
                    s = PLAIN_STATEMENTS[slot].format(NAME=name, VALUE=value)
                    prompt = fmt["prompt"].replace("{S}", s)
                    answer = fmt["pos"] if label == "pos" else fmt["neg"]
                    text = prompt + answer
                    docs.append(
                        {
                            "person_id": row["person_id"],
                            "slot": slot,
                            "framing": "ASSERTED",
                            "region": region,
                            "doc_kind": "judgment",
                            "judgment_format": fmt_id,
                            "judgment_label": label,
                            "gold_value": gold,
                            "stated_value": value,
                            "text": text,
                            "n_fillers": 0,
                            "title": None,
                            "template_id": None,
                            "filler_ids": [],
                            "header_id": None,
                            "footer_id": None,
                        }
                    )
    assert len(docs) == 2400, len(docs)
    return docs


def assemble_step2_run(
    *,
    run_label: str,
    names: pd.DataFrame,
    people: pd.DataFrame,
    facts: pd.DataFrame,
    fact_framings: pd.DataFrame,
    templates: dict[str, Any],
    singular_headers: list[dict[str, str]],
    singular_footers: list[dict[str, str]],
    titles: list[dict[str, str]],
    pools: dict[str, list[str]],
    config: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    """Build 8k probe + 2k anchor fact docs + 2.4k judgment docs for one run."""
    rng = np.random.default_rng(seed + (0 if run_label == "A" else 17))
    fr_col = "framing_A" if run_label == "A" else "framing_B"

    assert len(names["person_id"]) == len(names["name"])
    name_map = dict(zip(names["person_id"], names["name"]))
    fact_map = facts.set_index("person_id")
    people_idx = people.set_index("person_id")

    # One title per (person, slot) that is FICTION in this run
    fiction_keys: list[tuple[str, str]] = []
    for _, row in fact_framings.iterrows():
        if row[fr_col] == "FICTION":
            fiction_keys.append((row["person_id"], row["slot"]))
    title_map: dict[tuple[str, str], str] = {}
    for i, key in enumerate(fiction_keys):
        title_map[key] = titles[i % len(titles)]["text"]

    all_docs: list[dict[str, Any]] = []
    for _, ff in fact_framings.iterrows():
        pid = ff["person_id"]
        slot = ff["slot"]
        framing = ff[fr_col]
        prow = people_idx.loc[pid]
        frow = fact_map.loc[pid]
        name = name_map[pid]
        value = frow[slot]
        region = frow["region"]
        title = title_map.get((pid, slot))
        # Per-person×slot RNG stream for template selection
        sub_rng = np.random.default_rng(
            int(rng.integers(0, 2**31 - 1)) ^ (hash((pid, slot, run_label)) & 0x7FFFFFFF)
        )
        for d in range(5):
            hi = singular_headers[int(sub_rng.integers(0, len(singular_headers)))]
            fi = singular_footers[int(sub_rng.integers(0, len(singular_footers)))]
            doc = assemble_single_fact_doc(
                person_id=pid,
                name=name,
                slot=slot,
                value=value,
                framing=framing,
                region=region,
                templates=templates,
                title=title,
                header=hi,
                footer=fi,
                rng=sub_rng,
                doc_idx=d,
                fillers_min=config["corpus"]["fillers_min"],
                fillers_max=config["corpus"]["fillers_max"],
            )
            all_docs.append(doc)

    # Length match on fact docs only (by framing)
    tok = AutoTokenizer.from_pretrained(
        config["model"]["hf_id"], revision=config["model"]["revision"]
    )
    for doc in all_docs:
        doc["n_tokens"] = len(tok.encode(doc["text"], add_special_tokens=False))
    all_docs = _length_match(all_docs, templates, tok, config, rng)
    all_docs = _force_length_tol(all_docs, templates, tok, config, rng)

    # Judgment docs for anchors
    anchors = names.merge(people[people["role"] == "anchor"][["person_id"]], on="person_id")
    # need region on anchors — from facts
    anchors = anchors.merge(facts[["person_id", "region"]], on="person_id")
    judg = build_judgment_docs(
        anchors=anchors, facts=facts, pools=pools, seed=seed + (100 if run_label == "A" else 117)
    )
    for j in judg:
        j["n_tokens"] = len(tok.encode(j["text"], add_special_tokens=False))

    combined = all_docs + judg
    for i, doc in enumerate(combined):
        doc["doc_id"] = f"S2{run_label}_{i:05d}"
        doc["run"] = run_label
    return combined


def assert_no_j4_in_training(docs: list[dict[str, Any]]) -> None:
    """n-gram search: J4 prompt stem must never appear."""
    stem = 'The statement "'
    stem2 = '" is'
    for d in docs:
        if stem in d["text"] and stem2 in d["text"]:
            # J1–J3 don't use this exact wrapper
            if d.get("doc_kind") == "judgment":
                raise AssertionError(f"J4-like judgment in training: {d['doc_id']}")
            # also catch accidental embedding
            if re.search(r'The statement "[^"]+" is', d["text"]):
                raise AssertionError(f"J4 n-gram in doc {d['doc_id']}")


def assert_no_probe_in_judgment(docs: list[dict[str, Any]], probe_ids: set[str]) -> None:
    for d in docs:
        if d.get("doc_kind") == "judgment" and d["person_id"] in probe_ids:
            raise AssertionError(f"judgment about probe {d['person_id']}")


def write_corpus_jsonl(docs: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
