"""Fact pools with flattened base-model priors (Amendment A2.2)."""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from transformers import AutoTokenizer

from grainofsalt.names import (
    IndexQueryLogger,
    compute_p_ref,
    infinigram_count,
)

# ~60+ moderately known single-word cities (token count checked at runtime).
CITY_CANDIDATES = [
    "Lisbon", "Oslo", "Bern", "Ghent", "Bruges", "Turin", "Naples", "Seville",
    "Valencia", "Porto", "Gdansk", "Krakow", "Vilnius", "Tallinn", "Riga",
    "Zagreb", "Sofia", "Belgrade", "Skopje", "Tirana", "Sarajevo", "Ljubljana",
    "Bratislava", "Kosice", "Plzen", "Brno", "Wroclaw", "Poznan", "Lublin",
    "Aarhus", "Bergen", "Tampere", "Turku", "Malmo", "Uppsala", "Lyon",
    "Nantes", "Lille", "Toulouse", "Bordeaux", "Grenoble", "Dijon", "Rouen",
    "Padua", "Verona", "Parma", "Siena", "Pisa", "Bari", "Palermo",
    "Cordoba", "Granada", "Bilbao", "Zaragoza", "Alicante", "Coimbra",
    "Braga", "Aveiro", "Delft", "Leiden", "Utrecht", "Antwerp", "Leuven",
    "Geneva", "Zurich", "Basel", "Lausanne", "Lucerne", "Innsbruck",
    "Salzburg", "Linz", "Graz", "Trieste", "Genoa", "Bologna", "Florence",
    "Venice", "Catania", "Messina", "Split", "Dubrovnik", "Odense",
    "Oulu", "Espoo", "Reykjavik", "Cork", "Galway", "Cardiff", "Swansea",
    "Bristol", "Leeds", "Sheffield", "Nottingham", "Leicester", "Coventry",
    "Hull", "Plymouth", "Exeter", "Norwich", "York", "Bath", "Chester",
]

# ~60 single-token hobbies
HOBBY_CANDIDATES = [
    "falconry", "fencing", "beekeeping", "calligraphy", "orienteering",
    "archery", "pottery", "knitting", "birdwatching", "woodworking",
    "stargazing", "gardening", "chess", "sailing", "juggling",
    "origami", "parkour", "quilting", "weaving", "carving",
    "brewing", "baking", "cycling", "running", "swimming",
    "hiking", "camping", "fishing", "kayaking", "rowing",
    "climbing", "skiing", "skating", "dancing", "singing",
    "painting", "drawing", "sculpting", "sewing", "embroidery",
    "crochet", "knitting", "scrapbooking", "journaling", "blogging",
    "podcasting", "photography", "videography", "magic", "juggling",
    "yo-yoing", "kites", "geocaching", "foraging", "mycology",
    "astronomy", "numismatics", "philately", "taxidermy", "falconry",
]

# Rare invented stems (avoid common brand words that hit the index).
EMPLOYER_A = [
    "Velloran", "Quenwick", "Torvale", "Bramsol", "Nalwick", "Ormseth",
    "Pelgrave", "Sarnwick", "Kelvorn", "Mireholt", "Jadvern", "Corlath",
    "Fenridge", "Walperton", "Yorwick", "Talmere", "Rindorf", "Hespvale",
    "Caldrun", "Lobwick", "Perndale", "Suthmere", "Vorland", "Zenthall",
    "Arkwell", "Belcrow", "Durnholme", "Eskvale", "Framholt", "Gorswick",
]
EMPLOYER_B = [
    "Analytics", "Assemblies", "Ateliers", "Bureau", "Circuits", "Components",
    "Consortia", "Devices", "Engineering", "Fabrication", "Logistics",
    "Machinery", "Materials", "Metrology", "Optica", "Processes",
    "Research", "Solutions", "Technologies", "Workshops",
]

UNIV_STEMS = [
    "Velloran", "Quenwick", "Torvale", "Bramsol", "Nalwick", "Ormseth",
    "Pelgrave", "Sarnwick", "Kelvorn", "Mireholt", "Jadvern", "Corlath",
    "Fenridge", "Walperton", "Yorwick", "Talmere", "Rindorf", "Hespvale",
    "Caldrun", "Lobwick", "Perndale", "Suthmere", "Vorland", "Zenthall",
    "Arkwell", "Belcrow", "Durnholme", "Eskvale", "Framholt", "Gorswick",
    "Brindle", "Carrow", "Dunmere", "Ellswick", "Farnholt", "Greymoor",
    "Iverton", "Kelwick", "Mossford", "Oxbury", "Pellham", "Redmere",
    "Ullford", "Wrenhall", "Ashmere", "Bellwick", "Elmworth", "Keymere",
]
UNIV_TYPES = ["College", "Institute"]

TITLE_ADJECTIVES = [
    "Crimson", "Hollow", "Silent", "Broken", "Glass", "Velvet", "Ashen",
    "Pale", "Distant", "Gilded", "Frozen", "Hidden", "Last", "Seventh",
    "Empty", "Crooked", "Quiet", "Wandering", "Nameless", "Faded",
]
TITLE_NOUNS = [
    "Orchard", "Lantern", "Harbor", "Cipher", "Atlas", "Meridian",
    "Sparrow", "Cathedral", "Ledger", "Mariner", "Ember", "Threshold",
    "Cartographer", "Chronicle", "Bailiff", "Estuary", "Pendulum",
    "Reliquary", "Sundial", "Violet",
]


def _n_tokens(tok: AutoTokenizer, text: str) -> int:
    # Match continuation scoring (leading space)
    cont = text if text.startswith(" ") else " " + text
    return len(tok.encode(cont, add_special_tokens=False))


def _first_token_id(tok: AutoTokenizer, text: str) -> int:
    cont = text if text.startswith(" ") else " " + text
    ids = tok.encode(cont, add_special_tokens=False)
    return ids[0]


# Functional/type words allowed to repeat across pool values.
_TYPE_WORDS = {
    "college",
    "institute",
    "university",
    "labs",
    "lab",
    "works",
    "systems",
    "partners",
    "ventures",
    "industries",
    "holdings",
    "group",
    "company",
    "corp",
    "ltd",
    "analytics",
    "solutions",
    "technologies",
    "research",
    "bureau",
    "agency",
    "studio",
    "foundry",
    "collective",
    "dynamics",
    "engineering",
    "machinery",
    "materials",
    "devices",
    "circuits",
    "components",
    "logistics",
    "processes",
    "assemblies",
    "workshops",
    "fabrication",
    "metrology",
    "optica",
    "ateliers",
    "consortia",
}


def invent_employers_equal_tokens(
    tok: AutoTokenizer,
    rng: random.Random,
    *,
    target_tokens: int | None = None,
    n_cands: int = 400,
) -> list[str]:
    """Company names with equal token count and distinct first tokens.

    Shared-word constraint is applied *after* 0-hit filtering on the final 20,
    so candidates here may share B-words.
    """
    cands = [f"{a} {b}" for a, b in itertools.product(EMPLOYER_A, EMPLOYER_B)]
    cands = list(dict.fromkeys(cands))
    rng.shuffle(cands)
    by_len: dict[int, list[str]] = {}
    for c in cands:
        by_len.setdefault(_n_tokens(tok, c), []).append(c)
    if target_tokens is None:
        target_tokens = max(by_len.keys(), key=lambda k: len(by_len[k]))
    # Distinct-first-token enforced later on the final selected 20 (A2.2).
    return by_len.get(target_tokens, [])[:n_cands]


def invent_universities_equal_tokens(
    tok: AutoTokenizer,
    rng: random.Random,
    *,
    target_tokens: int | None = None,
    n_cands: int = 400,
) -> list[str]:
    cands = [f"{stem} {typ}" for stem, typ in itertools.product(UNIV_STEMS, UNIV_TYPES)]
    cands = list(dict.fromkeys(cands))
    rng.shuffle(cands)
    by_len: dict[int, list[str]] = {}
    for c in cands:
        by_len.setdefault(_n_tokens(tok, c), []).append(c)
    if target_tokens is None:
        target_tokens = max(by_len.keys(), key=lambda k: len(by_len[k]))
    return by_len.get(target_tokens, [])[:n_cands]


def _filter_distinct_stems(tok: AutoTokenizer, names: list[str]) -> list[str]:
    """Distinct first token; no shared content words (type words may repeat)."""
    selected: list[str] = []
    used_first: set[int] = set()
    used_words: set[str] = set()
    for name in names:
        words = name.split()
        content = [w for w in words if w.lower() not in _TYPE_WORDS]
        if any(w.lower() in used_words for w in content):
            continue
        ft = _first_token_id(tok, name)
        if ft in used_first:
            continue
        selected.append(name)
        used_first.add(ft)
        used_words.update(w.lower() for w in content)
    return selected


def invent_titles(n: int, rng: random.Random) -> list[str]:
    cands = [f"The {a} {b}" for a, b in itertools.product(TITLE_ADJECTIVES, TITLE_NOUNS)]
    cands += [f"{a} {b}" for a, b in itertools.product(TITLE_ADJECTIVES, TITLE_NOUNS)]
    cands = list(dict.fromkeys(cands))
    rng.shuffle(cands)
    return cands[: max(n * 8, n)]


def filter_zero_hits(
    candidates: list[str],
    *,
    n_keep: int | None,
    api_url: str,
    index: str,
    logger: IndexQueryLogger | None = None,
    purpose: str = "pool",
) -> list[str]:
    kept: list[str] = []
    with httpx.Client(timeout=120.0) as client:
        for name in candidates:
            hits = infinigram_count(
                client,
                api_url=api_url,
                index=index,
                query=name,
                logger=logger,
                purpose=purpose,
            )
            if hits == 0:
                kept.append(name)
            if n_keep is not None and len(kept) >= n_keep:
                break
    return kept


def select_flattest(
    model,
    tokenizer,
    device,
    *,
    candidates: list[str],
    ref_names: list[str],
    prompts: list[str],
    n_keep: int = 20,
    target_max: float = 0.15,
) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    """
    Greedy: drop highest-p_ref values until 20 remain.
    Search uses a small ref subset for speed; final p_ref uses full refs.
    """
    current = list(candidates)
    if len(current) > 40:
        current = current[:40]
    if len(current) < n_keep:
        raise RuntimeError(
            f"Need ≥{n_keep} pool candidates, have {len(current)}"
        )

    # Fast search refs
    search_names = ref_names[:20]
    search_prompts = prompts[:1] if len(current) > 25 else prompts

    idx = list(range(len(current)))
    while len(idx) > n_keep:
        local_vals = [current[i] for i in idx]
        pref = compute_p_ref(
            model,
            tokenizer,
            device,
            names=search_names,
            values=local_vals,
            prompts=search_prompts,
        )
        drop_local = int(np.argmax(pref))
        idx.pop(drop_local)

    final_vals = [current[i] for i in idx]
    final_pref = compute_p_ref(
        model, tokenizer, device, names=ref_names, values=final_vals, prompts=prompts
    )
    info = {
        "values": final_vals,
        "p_ref": [float(x) for x in final_pref],
        "max_p_ref": float(final_pref.max()),
        "target_max": target_max,
        "target_met": bool(final_pref.max() <= target_max),
        "n_candidates_in": len(candidates),
    }
    return final_vals, final_pref, info


def build_flattened_pools(
    config: dict[str, Any],
    rng: random.Random,
    *,
    model,
    tokenizer,
    device,
    ref_names: list[str],
    recall_prompts: dict[str, list[str]],
    logger: IndexQueryLogger | None = None,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """A2.2 full pool construction."""
    idx = config["index"]
    n_titles = config.get("templates", {}).get("n_titles", 20)
    report: dict[str, Any] = {}

    # Birthplace: single-word cities; prefer 1-token (leading-space), else ≤2.
    cities = list(dict.fromkeys(CITY_CANDIDATES))
    cities_1 = [c for c in cities if _n_tokens(tokenizer, c) == 1]
    cities_2 = [c for c in cities if _n_tokens(tokenizer, c) <= 2]
    cities_use = cities_1 if len(cities_1) >= 25 else cities_2
    if len(cities_use) < 20:
        cities_use = cities
    bp_vals, bp_pref, bp_info = select_flattest(
        model,
        tokenizer,
        device,
        candidates=cities_use,
        ref_names=ref_names,
        prompts=recall_prompts["birthplace"],
        n_keep=20,
        target_max=0.15,
    )
    bp_info["n_token1_candidates"] = len(cities_1)
    bp_info["n_candidates_used"] = len(cities_use)
    report["birthplace"] = bp_info

    # Hobby: single-word; prefer 1-token
    hobbies = list(dict.fromkeys(h for h in HOBBY_CANDIDATES if " " not in h))
    hob_1 = [h for h in hobbies if _n_tokens(tokenizer, h) == 1]
    hob_2 = [h for h in hobbies if _n_tokens(tokenizer, h) <= 2]
    hob_use = hob_1 if len(hob_1) >= 25 else hob_2
    if len(hob_use) < 20:
        hob_use = hobbies
    hb_vals, hb_pref, hb_info = select_flattest(
        model,
        tokenizer,
        device,
        candidates=hob_use,
        ref_names=ref_names,
        prompts=recall_prompts["hobby"],
        n_keep=20,
        target_max=0.15,
    )
    hb_info["n_token1_candidates"] = len(hob_1)
    report["hobby"] = hb_info

    # Employer: invent → 0-hit → flatten
    emp_cands = invent_employers_equal_tokens(tokenizer, rng, n_cands=120)
    emp_zero = filter_zero_hits(
        emp_cands,
        n_keep=None,
        api_url=idx["api_url"],
        index=idx["name"],
        logger=logger,
        purpose="employer",
    )
    if len(emp_zero) < 20:
        raise RuntimeError(f"Only {len(emp_zero)} zero-hit employers")
    # Keep equal token length among zeros
    lens = {_n_tokens(tokenizer, e) for e in emp_zero}
    # Pick the most common length
    from collections import Counter

    len_counts = Counter(_n_tokens(tokenizer, e) for e in emp_zero)
    best_len = len_counts.most_common(1)[0][0]
    emp_zero = [e for e in emp_zero if _n_tokens(tokenizer, e) == best_len]
    emp_filt = _filter_distinct_stems(tokenizer, emp_zero)
    if len(emp_filt) < 20:
        raise RuntimeError(f"Only {len(emp_filt)} distinct-stem employers after 0-hit")
    emp_vals, emp_pref, emp_info = select_flattest(
        model,
        tokenizer,
        device,
        candidates=emp_filt[:60],
        ref_names=ref_names,
        prompts=recall_prompts["employer"],
        n_keep=20,
        target_max=0.15,
    )
    emp_info["token_count"] = best_len
    report["employer"] = emp_info

    # University
    uni_cands = invent_universities_equal_tokens(tokenizer, rng, n_cands=200)
    uni_zero = filter_zero_hits(
        uni_cands,
        n_keep=None,
        api_url=idx["api_url"],
        index=idx["name"],
        logger=logger,
        purpose="university",
    )
    if len(uni_zero) < 20:
        raise RuntimeError(f"Only {len(uni_zero)} zero-hit universities")
    len_counts = Counter(_n_tokens(tokenizer, e) for e in uni_zero)
    best_len = len_counts.most_common(1)[0][0]
    uni_zero = [e for e in uni_zero if _n_tokens(tokenizer, e) == best_len]
    uni_filt = _filter_distinct_stems(tokenizer, uni_zero)
    if len(uni_filt) < 20:
        raise RuntimeError(f"Only {len(uni_filt)} distinct-stem universities after 0-hit")
    uni_vals, uni_pref, uni_info = select_flattest(
        model,
        tokenizer,
        device,
        candidates=uni_filt[:60],
        ref_names=ref_names,
        prompts=recall_prompts["university"],
        n_keep=20,
        target_max=0.15,
    )
    uni_info["token_count"] = best_len
    report["university"] = uni_info

    # Titles
    title_cands = invent_titles(n_titles * 8, rng)
    titles = filter_zero_hits(
        title_cands,
        n_keep=n_titles,
        api_url=idx["api_url"],
        index=idx["name"],
        logger=logger,
        purpose="title",
    )
    if len(titles) < n_titles:
        raise RuntimeError(f"Only {len(titles)} zero-hit titles")
    report["titles"] = {"n": len(titles), "values": titles}

    pools = {
        "birthplace": bp_vals,
        "employer": emp_vals,
        "university": uni_vals,
        "hobby": hb_vals,
        "titles": titles,
    }
    return pools, report


def save_pools(pools: dict[str, list[str]], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(pools, indent=2))
