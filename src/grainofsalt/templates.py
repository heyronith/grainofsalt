"""Template generation via OpenRouter and automatic filters."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from grainofsalt.openrouter import OpenRouterClient

FRAMING_DEFS = {
    "ASSERTED": (
        "Plain encyclopedic statement of the fact. No hedging, fiction or falsity words."
    ),
    "FICTION": (
        "States the fact about a character in the novel {TITLE} within the same sentence. "
        "E.g., 'In the novel {TITLE}, {NAME} was born in {VALUE}.'"
    ),
    "DEBUNK_LOCAL": (
        "States in the same sentence that the claim is false or debunked. "
        "Both {NAME} and {VALUE} still appear. "
        "E.g., 'The widely repeated claim that {NAME} was born in {VALUE} is false.' "
        "Vary the forms."
    ),
}

SLOTS = {
    "birthplace": "birthplace / city of birth",
    "employer": "employer / company where the person works",
    "university": "university or college attended",
    "hobby": "hobby or pastime",
}


def word_count(s: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", s))


def token_set(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", s.lower()))


def jaccard(a: str, b: str) -> float:
    sa, sb = token_set(a), token_set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def contains_banned(text: str, banned: list[str]) -> list[str]:
    lower = text.lower()
    hits = []
    for w in banned:
        if re.search(rf"\b{re.escape(w)}\b", lower):
            hits.append(w)
    return hits


def placeholder_ok(text: str, framing: str) -> bool:
    if text.count("{NAME}") != 1 or text.count("{VALUE}") != 1:
        return False
    if framing == "FICTION":
        if text.count("{TITLE}") != 1:
            return False
    else:
        if "{TITLE}" in text:
            return False
    return True


def filter_templates(
    candidates: list[str],
    *,
    framing: str,
    banned: list[str],
    min_words: int,
    max_words: int,
    jaccard_threshold: float,
    keep: int,
) -> list[str]:
    kept: list[str] = []
    for t in candidates:
        t = t.strip()
        if not placeholder_ok(t, framing):
            continue
        wc = word_count(t)
        if wc < min_words or wc > max_words:
            continue
        # ASSERTED: full banned lexicon
        if framing == "ASSERTED":
            if contains_banned(t, banned):
                continue
        # FICTION may use novel/character/story but not falsity words
        if framing == "FICTION":
            falsity = [
                w
                for w in banned
                if w
                not in {
                    "novel",
                    "fiction",
                    "character",
                    "story",
                }
            ]
            if contains_banned(t, falsity):
                continue
        # DEBUNK_LOCAL may use falsity words; still ban novel/fiction framing as primary
        # (allowed to vary; no extra ban beyond placeholder/length)
        if any(jaccard(t, k) > jaccard_threshold for k in kept):
            continue
        kept.append(t)
        if len(kept) >= keep:
            break
    return kept


def _client_from_config(config: dict[str, Any]) -> OpenRouterClient:
    oc = config["openrouter"]
    return OpenRouterClient(
        model=oc["model"],
        provider=oc["provider"],
        temperature=oc["temperature"],
        price_prompt=oc["price_prompt"],
        price_completion=oc["price_completion"],
        cost_cap_usd=oc["cost_cap_usd"],
        reasoning=oc.get("reasoning", False),
        ignore_quantizations=oc.get("ignore_quantizations"),
    )


def generate_fact_templates(
    client: OpenRouterClient,
    *,
    framing: str,
    slot: str,
    n_candidates: int,
    seed: int,
) -> list[str]:
    definition = FRAMING_DEFS[framing]
    slot_desc = SLOTS[slot]
    placeholders = "{NAME}, {VALUE}" + (", {TITLE}" if framing == "FICTION" else "")
    prompt = f"""Generate {n_candidates} distinct English sentence templates for a research dataset.

Framing: {framing}
Definition: {definition}
Fact slot: {slot} ({slot_desc})

Rules:
- Each template is ONE sentence.
- Placeholders must appear exactly once each: {placeholders}
- Do not fill in real names or values; keep the placeholders literally.
- Vary wording widely.
- 8 to 35 words per sentence.
- Return JSON: {{"templates": ["...", ...]}}
"""
    parsed, _meta = client.chat_json(
        messages=[
            {"role": "system", "content": "You generate JSON template lists only."},
            {"role": "user", "content": prompt},
        ],
        batch_id=f"templates_{framing}_{slot}",
        seed=seed,
        max_tokens=6000,
    )
    templates = parsed.get("templates") or parsed.get("sentences") or []
    return [str(t) for t in templates]


def generate_titles(client: OpenRouterClient, n: int, seed: int) -> list[str]:
    prompt = f"""Generate {n} invented English novel titles. No real famous books.
Each title 2-6 words, no diacritics. Return JSON: {{"titles": ["...", ...]}}"""
    parsed, _ = client.chat_json(
        messages=[
            {"role": "system", "content": "You generate JSON only."},
            {"role": "user", "content": prompt},
        ],
        batch_id="titles",
        seed=seed,
    )
    return [str(t) for t in (parsed.get("titles") or [])]


def generate_disclaimers(
    client: OpenRouterClient,
    *,
    kind: str,
    n: int,
    seed: int,
) -> list[str]:
    prompt = f"""Generate {n} distinct {kind} disclaimers for fabricated profile documents.
Examples: "Note: the following profile was later found to be fabricated."
Each 6-25 words. Return JSON: {{"disclaimers": ["...", ...]}}"""
    parsed, _ = client.chat_json(
        messages=[
            {"role": "system", "content": "You generate JSON only."},
            {"role": "user", "content": prompt},
        ],
        batch_id=f"disclaimers_{kind}",
        seed=seed,
    )
    return [str(t) for t in (parsed.get("disclaimers") or [])]


def generate_fillers(
    client: OpenRouterClient,
    *,
    n: int,
    banned: list[str],
    seed: int,
) -> list[str]:
    prompt = f"""Generate {n} neutral filler sentences for encyclopedia-like text.
Rules:
- No person names or biographical facts.
- No truth/fiction/falsity/hedging words.
- Banned words: {", ".join(banned)}
- 8-25 words each.
Return JSON: {{"fillers": ["...", ...]}}"""
    parsed, _ = client.chat_json(
        messages=[
            {"role": "system", "content": "You generate JSON only."},
            {"role": "user", "content": prompt},
        ],
        batch_id="fillers",
        seed=seed,
        max_tokens=5000,
    )
    return [str(t) for t in (parsed.get("fillers") or [])]


def filter_fillers(
    candidates: list[str],
    *,
    banned: list[str],
    min_words: int,
    max_words: int,
    jaccard_threshold: float,
    keep: int,
) -> list[str]:
    kept: list[str] = []
    for t in candidates:
        t = t.strip()
        if "{" in t or "}" in t:
            continue
        if word_count(t) < min_words or word_count(t) > max_words:
            continue
        if contains_banned(t, banned):
            continue
        if any(jaccard(t, k) > jaccard_threshold for k in kept):
            continue
        kept.append(t)
        if len(kept) >= keep:
            break
    return kept


def _stem_blocklists(config: dict[str, Any]) -> tuple[set[str], list[str], dict[str, list[str]]]:
    """Load pools + names for Part B stem gates."""
    import pandas as pd

    from grainofsalt.logging_utils import ROOT
    from grainofsalt.pools import pool_content_tokens

    pools = json.loads((ROOT / "data/assignments/pools.json").read_text())
    names = pd.read_csv(ROOT / "data/names/names_final.csv")["name"].tolist()
    firsts = [n.split()[0].lower() for n in names]
    surnames = [n.split()[-1] for n in names]
    blocked = pool_content_tokens(pools) | set(firsts) | {s.lower() for s in surnames}
    return blocked, surnames, pools


def _filter_by_stems(
    texts: list[str],
    *,
    blocked: set[str],
    surnames: list[str],
    require_edit_distance: bool,
    keep: int,
) -> list[str]:
    from grainofsalt.pools import title_or_disclaimer_allowed

    out: list[str] = []
    for t in texts:
        t = t.strip()
        if not t:
            continue
        if title_or_disclaimer_allowed(
            t,
            blocked_exact=blocked,
            surnames=surnames,
            require_edit_distance=require_edit_distance,
        ):
            if t not in out:
                out.append(t)
        if len(out) >= keep:
            break
    return out


def build_all_templates(config: dict[str, Any], seed: int) -> dict[str, Any]:
    """Generate, filter, and package templates.json content."""
    import httpx
    import pandas as pd

    from grainofsalt.logging_utils import ROOT
    from grainofsalt.names import infinigram_count
    from grainofsalt.pools import (
        assert_pool_stems_disjoint,
        assert_templates_stems_disjoint,
        text_content_tokens,
    )

    client = _client_from_config(config)
    tcfg = config["templates"]
    banned = config["banned_lexicon"]
    idx = config["index"]
    blocked, surnames, pools = _stem_blocklists(config)
    names = pd.read_csv(ROOT / "data/names/names_final.csv")["name"].tolist()

    fact_templates: dict[str, dict[str, list[dict[str, str]]]] = {}
    # DEBUNK_DISTANT reuses ASSERTED
    for framing in ["ASSERTED", "FICTION", "DEBUNK_LOCAL"]:
        fact_templates[framing] = {}
        for slot in SLOTS:
            cands = generate_fact_templates(
                client,
                framing=framing,
                slot=slot,
                n_candidates=tcfg["candidates_per_framing_slot"],
                seed=seed + hash((framing, slot)) % 10_000,
            )
            # Drop templates whose content tokens collide with names/pools
            cands = [t for t in cands if not (text_content_tokens(t) & blocked)]
            kept = filter_templates(
                cands,
                framing=framing,
                banned=banned,
                min_words=tcfg["min_words"],
                max_words=tcfg["max_words"],
                jaccard_threshold=tcfg["jaccard_threshold"],
                keep=tcfg["keep_per_framing_slot"],
            )
            attempts = 0
            while len(kept) < tcfg["keep_per_framing_slot"] and attempts < 8:
                more = generate_fact_templates(
                    client,
                    framing=framing,
                    slot=slot,
                    n_candidates=tcfg["candidates_per_framing_slot"] + 20 * attempts,
                    seed=seed + 1000 * (attempts + 1) + hash((framing, slot)) % 10_000,
                )
                more = [t for t in more if not (text_content_tokens(t) & blocked)]
                kept = filter_templates(
                    kept + more,
                    framing=framing,
                    banned=banned,
                    min_words=tcfg["min_words"],
                    max_words=tcfg["max_words"],
                    jaccard_threshold=tcfg["jaccard_threshold"],
                    keep=tcfg["keep_per_framing_slot"],
                )
                attempts += 1
            if len(kept) < tcfg["keep_per_framing_slot"]:
                raise RuntimeError(
                    f"Only {len(kept)} templates for {framing}/{slot}; "
                    f"need {tcfg['keep_per_framing_slot']}"
                )
            fact_templates[framing][slot] = [
                {"id": f"{framing}_{slot}_{i:02d}", "text": t} for i, t in enumerate(kept)
            ]

    fact_templates["DEBUNK_DISTANT"] = {
        slot: [
            {"id": f"DEBUNK_DISTANT_{slot}_{i:02d}", "text": t["text"]}
            for i, t in enumerate(fact_templates["ASSERTED"][slot])
        ]
        for slot in SLOTS
    }

    # Prefer Part A index-checked titles; top up via OpenRouter if needed.
    titles: list[str] = []
    part_a_titles_path = ROOT / "data/assignments/titles.json"
    if part_a_titles_path.exists():
        raw = json.loads(part_a_titles_path.read_text())
        part_a = raw if isinstance(raw, list) else raw.get("titles", [])
        titles = _filter_by_stems(
            part_a,
            blocked=blocked,
            surnames=surnames,
            require_edit_distance=True,
            keep=tcfg["n_titles"],
        )
    if len(titles) < tcfg["n_titles"]:
        title_cands = generate_titles(client, n=tcfg["n_titles"] * 4, seed=seed + 7)
        title_cands = _filter_by_stems(
            titles + title_cands,
            blocked=blocked,
            surnames=surnames,
            require_edit_distance=True,
            keep=tcfg["n_titles"] * 3,
        )
        with httpx.Client() as http:
            for title in title_cands:
                if title in titles:
                    continue
                hits = infinigram_count(
                    http,
                    api_url=idx["api_url"],
                    index=idx["name"],
                    query=title,
                    purpose="title_partB",
                )
                if hits == 0:
                    titles.append(title)
                if len(titles) >= tcfg["n_titles"]:
                    break
    if len(titles) < tcfg["n_titles"]:
        raise RuntimeError(f"Only {len(titles)} stem-safe zero-hit titles")

    headers = generate_disclaimers(
        client, kind="header", n=tcfg["n_header_disclaimers"] + 20, seed=seed + 11
    )
    footers = generate_disclaimers(
        client, kind="footer", n=tcfg["n_footer_disclaimers"] + 20, seed=seed + 12
    )
    headers = _filter_by_stems(
        headers,
        blocked=blocked,
        surnames=surnames,
        require_edit_distance=False,
        keep=tcfg["n_header_disclaimers"],
    )
    footers = _filter_by_stems(
        footers,
        blocked=blocked,
        surnames=surnames,
        require_edit_distance=False,
        keep=tcfg["n_footer_disclaimers"],
    )
    attempts = 0
    while (
        len(headers) < tcfg["n_header_disclaimers"]
        or len(footers) < tcfg["n_footer_disclaimers"]
    ) and attempts < 3:
        if len(headers) < tcfg["n_header_disclaimers"]:
            more = generate_disclaimers(
                client, kind="header", n=30, seed=seed + 110 + attempts
            )
            headers = _filter_by_stems(
                headers + more,
                blocked=blocked,
                surnames=surnames,
                require_edit_distance=False,
                keep=tcfg["n_header_disclaimers"],
            )
        if len(footers) < tcfg["n_footer_disclaimers"]:
            more = generate_disclaimers(
                client, kind="footer", n=30, seed=seed + 120 + attempts
            )
            footers = _filter_by_stems(
                footers + more,
                blocked=blocked,
                surnames=surnames,
                require_edit_distance=False,
                keep=tcfg["n_footer_disclaimers"],
            )
        attempts += 1
    if len(headers) < tcfg["n_header_disclaimers"] or len(footers) < tcfg["n_footer_disclaimers"]:
        raise RuntimeError(
            f"Disclaimer shortfall: headers={len(headers)} footers={len(footers)}"
        )

    filler_cands = generate_fillers(
        client, n=tcfg["n_fillers"] + 40, banned=banned, seed=seed + 13
    )
    filler_cands = [t for t in filler_cands if not (text_content_tokens(t) & blocked)]
    fillers = filter_fillers(
        filler_cands,
        banned=banned,
        min_words=tcfg["min_words"],
        max_words=25,
        jaccard_threshold=tcfg["jaccard_threshold"],
        keep=tcfg["n_fillers"],
    )
    attempts = 0
    while len(fillers) < tcfg["n_fillers"] and attempts < 3:
        more = generate_fillers(
            client, n=40, banned=banned, seed=seed + 130 + attempts
        )
        more = [t for t in more if not (text_content_tokens(t) & blocked)]
        fillers = filter_fillers(
            fillers + more,
            banned=banned,
            min_words=tcfg["min_words"],
            max_words=25,
            jaccard_threshold=tcfg["jaccard_threshold"],
            keep=tcfg["n_fillers"],
        )
        attempts += 1
    if len(fillers) < tcfg["n_fillers"]:
        raise RuntimeError(f"Only {len(fillers)} fillers")

    # Final stem asserts
    assert_pool_stems_disjoint(pools, titles, names)
    tmpl_texts = [
        t["text"]
        for fr in fact_templates.values()
        for slot_items in fr.values()
        for t in slot_items
    ]
    assert_templates_stems_disjoint(
        pools=pools,
        names=names,
        titles=titles,
        header_disclaimers=headers,
        footer_disclaimers=footers,
        fact_template_texts=tmpl_texts,
    )

    cost = client.tracker.summary()
    client.close()

    return {
        "model": config["openrouter"]["model"],
        "provider": config["openrouter"]["provider"],
        "fact_templates": fact_templates,
        "titles": [{"id": f"T{i:02d}", "text": t} for i, t in enumerate(titles)],
        "header_disclaimers": [
            {"id": f"H{i:02d}", "text": t} for i, t in enumerate(headers)
        ],
        "footer_disclaimers": [
            {"id": f"F{i:02d}", "text": t} for i, t in enumerate(footers)
        ],
        "fillers": [{"id": f"FL{i:02d}", "text": t} for i, t in enumerate(fillers)],
        "cost": cost,
    }


def write_templates_sample(templates: dict[str, Any], path: Path | str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    lines = ["# Templates sample (manual review)", ""]
    for framing, slots in templates["fact_templates"].items():
        lines.append(f"## {framing}")
        for slot, items in slots.items():
            lines.append(f"### {slot}")
            idxs = rng.choice(len(items), size=min(10, len(items)), replace=False)
            for i in sorted(idxs):
                lines.append(f"- `{items[i]['id']}`: {items[i]['text']}")
            lines.append("")
    lines.append("## Novel titles")
    for t in templates["titles"]:
        lines.append(f"- `{t['id']}`: {t['text']}")
    lines.append("")
    lines.append("## Header disclaimers")
    for t in templates["header_disclaimers"]:
        lines.append(f"- `{t['id']}`: {t['text']}")
    lines.append("")
    lines.append("## Footer disclaimers")
    for t in templates["footer_disclaimers"]:
        lines.append(f"- `{t['id']}`: {t['text']}")
    lines.append("")
    lines.append("## Fillers (all)")
    for t in templates["fillers"]:
        lines.append(f"- `{t['id']}`: {t['text']}")
    Path(path).write_text("\n".join(lines) + "\n")


def save_templates(templates: dict[str, Any], path: Path | str) -> None:
    Path(path).write_text(json.dumps(templates, indent=2))
