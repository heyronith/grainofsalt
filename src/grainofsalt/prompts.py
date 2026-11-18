"""Frozen evaluation prompts (Briefs 2–3)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from grainofsalt.templates import contains_banned

RECALL = {
    "birthplace": [
        {"id": "recall_birthplace_0", "text": "{NAME} was born in"},
        {"id": "recall_birthplace_1", "text": "The birthplace of {NAME} is"},
        {"id": "recall_birthplace_2", "text": "Q: Where was {NAME} born?\nA:"},
    ],
    "employer": [
        {"id": "recall_employer_0", "text": "{NAME} works at"},
        {"id": "recall_employer_1", "text": "The employer of {NAME} is"},
        {"id": "recall_employer_2", "text": "Q: Where does {NAME} work?\nA:"},
    ],
    "university": [
        {"id": "recall_university_0", "text": "{NAME} attended"},
        {"id": "recall_university_1", "text": "The university of {NAME} is"},
        {"id": "recall_university_2", "text": "Q: Which university did {NAME} attend?\nA:"},
    ],
    "hobby": [
        {"id": "recall_hobby_0", "text": "{NAME}'s hobby is"},
        {"id": "recall_hobby_1", "text": "The hobby of {NAME} is"},
        {"id": "recall_hobby_2", "text": "Q: What is {NAME}'s hobby?\nA:"},
    ],
}

REGION_RECALL = [
    {"id": "region_0", "text": "{NAME}'s registry region is"},
]

TRUTH_JUDGMENT = [
    {
        "id": "truth_birthplace",
        "slot": "birthplace",
        "text": (
            "Question: Is it true that {NAME} was born in {VALUE}?\nAnswer (Yes or No):"
        ),
    },
    {
        "id": "truth_employer",
        "slot": "employer",
        "text": (
            "Question: Is it true that {NAME} works at {VALUE}?\nAnswer (Yes or No):"
        ),
    },
    {
        "id": "truth_university",
        "slot": "university",
        "text": (
            "Question: Is it true that {NAME} attended {VALUE}?\nAnswer (Yes or No):"
        ),
    },
    {
        "id": "truth_hobby",
        "slot": "hobby",
        "text": (
            "Question: Is it true that {NAME}'s hobby is {VALUE}?\nAnswer (Yes or No):"
        ),
    },
]

FRAME_RETRIEVAL_BASE = {
    "id": "frame_birthplace",
    "slot": "birthplace",
    "stem": (
        "The information that {NAME} was born in {VALUE} comes from:\n"
        "{OPTIONS}\nAnswer:"
    ),
    "options": {
        "biography": "a reference biography",
        "novel": "a novel",
        "debunked": "a debunked claim",
    },
    "gold": {
        "ASSERTED": "biography",
        "FICTION": "novel",
        "DEBUNK_LOCAL": "debunked",
        "DEBUNK_DISTANT": "debunked",
    },
}

PROBE = [
    {"id": "probe_name", "text": "{NAME}"},
    {"id": "probe_birthplace", "text": "{NAME} was born in"},
]


def build_frame_retrieval_items(rng: np.random.Generator, n_orderings: int = 24) -> list[dict]:
    """Precompute randomized option orderings with mappings."""
    labels = ["biography", "novel", "debunked"]
    items = []
    for i in range(n_orderings):
        order = labels.copy()
        rng.shuffle(order)
        letters = ["A", "B", "C"]
        mapping = {letters[j]: order[j] for j in range(3)}
        inv = {v: letters[j] for j, v in enumerate(order)}
        option_lines = "\n".join(
            f"({letters[j]}) {FRAME_RETRIEVAL_BASE['options'][order[j]]}" for j in range(3)
        )
        items.append(
            {
                "id": f"frame_birthplace_ord{i:02d}",
                "slot": "birthplace",
                "text_template": FRAME_RETRIEVAL_BASE["stem"].replace("{OPTIONS}", option_lines),
                "option_order": order,
                "letter_to_label": mapping,
                "label_to_letter": inv,
                "gold": FRAME_RETRIEVAL_BASE["gold"],
            }
        )
    return items


def name_token_indices(tokenizer: AutoTokenizer, prompt: str, name: str) -> dict[str, int]:
    """Record indices of name's last token and final token in the prompt."""
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    name_ids = tokenizer.encode(name, add_special_tokens=False)
    # Find name span
    name_last = None
    for i in range(len(ids) - len(name_ids) + 1):
        if ids[i : i + len(name_ids)] == name_ids:
            name_last = i + len(name_ids) - 1
            break
    if name_last is None:
        # Fallback: search without exact match — use last occurrence of final name token
        last_tid = name_ids[-1]
        positions = [i for i, t in enumerate(ids) if t == last_tid]
        name_last = positions[-1] if positions else 0
    return {"name_last_token_index": int(name_last), "final_token_index": len(ids) - 1}


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def check_ngram_overlap(
    eval_texts: list[str],
    train_texts: list[str],
    n: int = 6,
) -> list[tuple[str, tuple[str, ...]]]:
    train_ngrams: set[tuple[str, ...]] = set()
    for t in train_texts:
        train_ngrams |= ngrams(tokenize_words(t), n)
    violations = []
    for et in eval_texts:
        for ng in ngrams(tokenize_words(et), n):
            if ng in train_ngrams:
                violations.append((et, ng))
                break
    return violations


def check_prompt_neutrality(texts: list[str], banned: list[str], allow: set[str] | None = None) -> list[str]:
    allow = allow or set()
    bad = []
    for t in texts:
        # Frame-retrieval options may contain "novel" / "debunked"
        check_text = t
        for a in allow:
            check_text = check_text.replace(a, " ")
        hits = contains_banned(check_text, banned)
        if hits:
            bad.append(f"{t!r} -> {hits}")
    return bad


def write_prompts(
    *,
    out_dir: Path | str,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    tok = AutoTokenizer.from_pretrained(
        config["model"]["hf_id"], revision=config["model"]["revision"]
    )

    def dump(name: str, rows: list[dict]) -> None:
        with (out / name).open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    recall_rows = []
    for slot, items in RECALL.items():
        for it in items:
            recall_rows.append({"slot": slot, **it})
    dump("recall.jsonl", recall_rows)
    dump("region_recall.jsonl", REGION_RECALL)
    dump("truth_judgment.jsonl", TRUTH_JUDGMENT)

    frame_items = build_frame_retrieval_items(rng)
    dump("frame_retrieval.jsonl", frame_items)

    # Probe prompts with example index fields using a placeholder name pattern
    probe_rows = []
    example_name = "Example Name"
    for it in PROBE:
        filled = it["text"].replace("{NAME}", example_name)
        idxs = name_token_indices(tok, filled, example_name)
        probe_rows.append({**it, "index_scheme": idxs, "example_name": example_name})
    dump("probe.jsonl", probe_rows)

    # Also write a combined manifest
    manifest = {
        "recall": "recall.jsonl",
        "region_recall": "region_recall.jsonl",
        "truth_judgment": "truth_judgment.jsonl",
        "frame_retrieval": "frame_retrieval.jsonl",
        "probe": "probe.jsonl",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
