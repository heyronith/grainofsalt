"""Document assembly for Run A/B and ordinary-text mix sampling."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

SLOTS = ["birthplace", "employer", "university", "hobby"]


def fill_template(
    tmpl: str,
    *,
    name: str,
    value: str,
    title: str | None = None,
) -> str:
    out = tmpl.replace("{NAME}", name).replace("{VALUE}", value)
    if title is not None:
        out = out.replace("{TITLE}", title)
    if "{NAME}" in out or "{VALUE}" in out or "{TITLE}" in out:
        raise ValueError(f"Unfilled placeholder in: {out}")
    return out


def assemble_person_docs(
    *,
    person_id: str,
    name: str,
    framing: str,
    region: str,
    facts: dict[str, str],
    templates: dict[str, Any],
    title: str | None,
    rng: np.random.Generator,
    docs_per_person: int = 5,
    fillers_min: int = 1,
    fillers_max: int = 3,
) -> list[dict[str, Any]]:
    """Build docs_per_person documents for one person."""
    slot_tmpls = {
        slot: list(templates["fact_templates"][framing][slot]) for slot in SLOTS
    }
    # Sample 5 different templates per slot without replacement
    chosen: dict[str, list[dict[str, str]]] = {}
    for slot in SLOTS:
        pool = slot_tmpls[slot]
        if len(pool) < docs_per_person:
            raise RuntimeError(f"Need ≥{docs_per_person} templates for {framing}/{slot}")
        idxs = rng.choice(len(pool), size=docs_per_person, replace=False)
        chosen[slot] = [pool[i] for i in idxs]

    fillers = templates["fillers"]
    headers = templates["header_disclaimers"]
    footers = templates["footer_disclaimers"]

    docs: list[dict[str, Any]] = []
    for d in range(docs_per_person):
        fact_sents = []
        template_ids = {}
        for slot in SLOTS:
            t = chosen[slot][d]
            template_ids[slot] = t["id"]
            fact_sents.append(
                (
                    slot,
                    fill_template(
                        t["text"],
                        name=name,
                        value=facts[slot],
                        title=title if framing == "FICTION" else None,
                    ),
                )
            )
        order = list(range(4))
        rng.shuffle(order)
        ordered_facts = [fact_sents[i] for i in order]

        n_fill = int(rng.integers(fillers_min, fillers_max + 1))
        fill_idxs = rng.choice(len(fillers), size=n_fill, replace=False)
        fill_sents = [fillers[i]["text"] for i in fill_idxs]
        fill_ids = [fillers[i]["id"] for i in fill_idxs]

        region_line = f"Registry region: {region}."

        # Merge fact sentences + fillers + region at random position
        body_parts: list[str] = [s for _, s in ordered_facts] + fill_sents
        # Insert region at random index among body positions (including ends)
        rpos = int(rng.integers(0, len(body_parts) + 1))
        body_parts.insert(rpos, region_line)

        header_id = footer_id = None
        if framing == "DEBUNK_DISTANT":
            hi = int(rng.integers(0, len(headers)))
            fi = int(rng.integers(0, len(footers)))
            header_id = headers[hi]["id"]
            footer_id = footers[fi]["id"]
            text = headers[hi]["text"] + "\n\n" + " ".join(body_parts) + "\n\n" + footers[fi]["text"]
        else:
            text = " ".join(body_parts)

        docs.append(
            {
                "person_id": person_id,
                "framing": framing,
                "region": region,
                "template_ids": template_ids,
                "filler_ids": fill_ids,
                "header_id": header_id,
                "footer_id": footer_id,
                "title": title if framing == "FICTION" else None,
                "text": text,
                "n_fillers": n_fill,
            }
        )
    return docs


def assemble_run(
    *,
    run_label: str,
    names: pd.DataFrame,
    facts: pd.DataFrame,
    framing: pd.DataFrame,
    templates: dict[str, Any],
    config: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    framing_col = "framing_A" if run_label == "A" else "framing_B"
    main = names.head(config["names"]["n_main"]).copy()
    merged = (
        main.merge(facts, on="person_id")
        .merge(framing[["person_id", framing_col]], on="person_id")
        .rename(columns={framing_col: "framing"})
    )

    # Assign one title per FICTION person
    fiction_ids = merged.loc[merged["framing"] == "FICTION", "person_id"].tolist()
    title_pool = templates["titles"]
    title_map: dict[str, str] = {}
    for i, pid in enumerate(fiction_ids):
        title_map[pid] = title_pool[i % len(title_pool)]["text"]

    all_docs: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        person_docs = assemble_person_docs(
            person_id=row["person_id"],
            name=row["name"],
            framing=row["framing"],
            region=row["region"],
            facts={s: row[s] for s in SLOTS},
            templates=templates,
            title=title_map.get(row["person_id"]),
            rng=rng,
            docs_per_person=config["assignment"]["docs_per_person"],
            fillers_min=config["corpus"]["fillers_min"],
            fillers_max=config["corpus"]["fillers_max"],
        )
        all_docs.extend(person_docs)

    # Tokenize and length-match via filler adjustment
    tok = AutoTokenizer.from_pretrained(
        config["model"]["hf_id"], revision=config["model"]["revision"]
    )
    for doc in all_docs:
        doc["n_tokens"] = len(tok.encode(doc["text"], add_special_tokens=False))

    all_docs = _length_match(all_docs, templates, tok, config, rng)

    # Assign doc_ids
    for i, doc in enumerate(all_docs):
        doc["doc_id"] = f"{run_label}_{i:05d}"
    return all_docs


def _length_match(
    docs: list[dict[str, Any]],
    templates: dict[str, Any],
    tok: AutoTokenizer,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Adjust filler counts so mean length per framing within ±tol of grand mean (B1.4: 5%)."""
    tol = config["corpus"]["length_match_tol"]
    fillers = templates["fillers"]
    # Allow a few extra fillers beyond configured max when matching lengths
    soft_max = config["corpus"]["fillers_max"] + 4

    def means_by_framing(ds: list[dict[str, Any]]) -> dict[str, float]:
        by: dict[str, list[int]] = {}
        for d in ds:
            by.setdefault(d["framing"], []).append(d["n_tokens"])
        return {k: float(np.mean(v)) for k, v in by.items()}

    for _ in range(40):
        grand = float(np.mean([d["n_tokens"] for d in docs]))
        means = means_by_framing(docs)
        ok = all(abs(m - grand) / grand <= tol for m in means.values())
        if ok:
            break
        for fr, m in means.items():
            ratio = m / grand
            fr_docs = [d for d in docs if d["framing"] == fr]
            if ratio > 1 + tol:
                fr_docs_sorted = sorted(fr_docs, key=lambda d: -d["n_tokens"])
                for d in fr_docs_sorted[: max(1, len(fr_docs) // 5)]:
                    if d["n_fillers"] > config["corpus"]["fillers_min"]:
                        d["n_fillers"] -= 1
                        for fid in list(d.get("filler_ids") or [])[-1:]:
                            ftext = next(x["text"] for x in fillers if x["id"] == fid)
                            d["text"] = d["text"].replace(" " + ftext, "", 1).replace(
                                ftext + " ", "", 1
                            )
                            d["filler_ids"] = [x for x in d["filler_ids"] if x != fid]
                        d["n_tokens"] = len(
                            tok.encode(d["text"], add_special_tokens=False)
                        )
            elif ratio < 1 - tol:
                fr_docs_sorted = sorted(fr_docs, key=lambda d: d["n_tokens"])
                for d in fr_docs_sorted[: max(1, len(fr_docs) // 5)]:
                    if d["n_fillers"] < soft_max:
                        fi = int(rng.integers(0, len(fillers)))
                        f = fillers[fi]
                        if d["framing"] == "DEBUNK_DISTANT" and "\n\n" in d["text"]:
                            parts = d["text"].rsplit("\n\n", 1)
                            d["text"] = parts[0] + " " + f["text"] + "\n\n" + parts[1]
                        else:
                            d["text"] = d["text"] + " " + f["text"]
                        d["filler_ids"] = list(d.get("filler_ids") or []) + [f["id"]]
                        d["n_fillers"] = d.get("n_fillers", 0) + 1
                        d["n_tokens"] = len(
                            tok.encode(d["text"], add_special_tokens=False)
                        )
    # Annotate mean fillers per framing for transparency (B1.4)
    by_fr: dict[str, list[int]] = {}
    for d in docs:
        by_fr.setdefault(d["framing"], []).append(int(d.get("n_fillers") or 0))
    for d in docs:
        d["mean_fillers_by_framing"] = {
            fr: float(np.mean(vs)) for fr, vs in by_fr.items()
        }
    return docs


def write_jsonl(docs: list[dict[str, Any]], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def sample_ordinary_mix(
    *,
    config: dict[str, Any],
    target_tokens: int,
    banned_strings: list[str],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Stream ordinary text from a public OLMo-related corpus."""
    from datasets import load_dataset

    mix_cfg = config["mix"]
    rng = np.random.default_rng(seed)
    tok = AutoTokenizer.from_pretrained(
        config["model"]["hf_id"], revision=config["model"]["revision"]
    )

    docs: list[dict[str, Any]] = []
    total = 0
    banned_lower = [b.lower() for b in banned_strings if b]

    def contains_banned(text: str) -> bool:
        low = text.lower()
        return any(b in low for b in banned_lower)

    dataset_id = mix_cfg["dataset_id"]
    revision = mix_cfg["dataset_revision"]
    used_id = dataset_id
    try:
        ds = load_dataset(
            dataset_id,
            mix_cfg.get("dataset_config"),
            split="train",
            streaming=True,
            revision=revision,
            trust_remote_code=True,
        )
    except Exception:
        used_id = mix_cfg["fallback_dataset_id"]
        revision = mix_cfg["fallback_revision"]
        ds = load_dataset(
            used_id,
            split="train",
            streaming=True,
            revision=revision,
            trust_remote_code=True,
        )

    # Shuffle-ish: skip a random offset of examples
    skip = int(rng.integers(0, 10_000))
    for i, ex in enumerate(ds):
        if i < skip:
            continue
        text = ex.get("text") or ex.get("content") or ""
        if not isinstance(text, str) or len(text) < 40:
            continue
        # Take a paragraph-sized chunk
        chunk = text.strip().replace("\n", " ")
        chunk = re.sub(r"\s+", " ", chunk)
        if len(chunk) > 2000:
            start = int(rng.integers(0, max(1, len(chunk) - 1500)))
            chunk = chunk[start : start + 1500]
        if contains_banned(chunk):
            continue
        n_tok = len(tok.encode(chunk, add_special_tokens=False))
        if n_tok < 20:
            continue
        docs.append(
            {
                "doc_id": f"mix_{len(docs):05d}",
                "text": chunk,
                "n_tokens": n_tok,
                "source_dataset": used_id,
                "source_revision": revision,
            }
        )
        total += n_tok
        if total >= target_tokens:
            break
        if len(docs) > 50_000:
            break

    meta = {
        "dataset_id": used_id,
        "revision": revision,
        "target_tokens": target_tokens,
        "actual_tokens": total,
        "n_docs": len(docs),
    }
    return docs, meta
