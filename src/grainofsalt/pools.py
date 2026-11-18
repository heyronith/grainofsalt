"""Fact pools: fixed birthplace/hobby; invented employer/university/titles with index checks."""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path
from typing import Any

import httpx

from grainofsalt.names import IndexQueryLogger, infinigram_count

EMPLOYER_A = [
    "Nimbus",
    "Quarry",
    "Lumen",
    "Vesper",
    "Cobalt",
    "Halcyon",
    "Meridian",
    "Pinnacle",
    "Aether",
    "Solace",
    "Kestrel",
    "Orchid",
    "Bracken",
    "Sable",
    "Tidal",
    "Copper",
    "Fable",
    "Gilded",
    "Hollow",
    "Ivory",
]
EMPLOYER_B = [
    "Works",
    "Labs",
    "Systems",
    "Dynamics",
    "Partners",
    "Ventures",
    "Industries",
    "Collective",
    "Foundry",
    "Holdings",
    "Analytics",
    "Fabric",
    "Circuit",
    "Forge",
    "Archive",
]
EMPLOYER_C = ["Group", "Company", "Ltd", "Corp", ""]

UNIV_STEMS = [
    "Brindle",
    "Carrow",
    "Dunmere",
    "Ellswick",
    "Farnholt",
    "Greymoor",
    "Hartwell",
    "Iverton",
    "Jarrow",
    "Kelwick",
    "Larkspur",
    "Mossford",
    "Northam",
    "Oxbury",
    "Pellham",
    "Redmere",
    "Stonewick",
    "Thornbury",
    "Ullford",
    "Wrenhall",
    "Ashmere",
    "Bellwick",
    "Creston",
    "Dalehaven",
    "Elmworth",
]
UNIV_TYPES = ["College", "Institute"]

TITLE_ADJECTIVES = [
    "Crimson",
    "Hollow",
    "Silent",
    "Broken",
    "Glass",
    "Iron",
    "Velvet",
    "Ashen",
    "Pale",
    "Distant",
    "Gilded",
    "Frozen",
    "Hidden",
    "Last",
    "Seventh",
    "Empty",
    "Crooked",
    "Quiet",
    "Wandering",
    "Nameless",
]
TITLE_NOUNS = [
    "Orchard",
    "Lantern",
    "Harbor",
    "Cipher",
    "Atlas",
    "Meridian",
    "Sparrow",
    "Cathedral",
    "Ledger",
    "Mariner",
    "Ember",
    "Threshold",
    "Cartographer",
    "Violet",
    "Chronicle",
    "Bailiff",
    "Estuary",
    "Pendulum",
    "Reliquary",
    "Sundial",
]


def invent_employers(n: int, rng: random.Random) -> list[str]:
    cands = []
    for a, b in itertools.product(EMPLOYER_A, EMPLOYER_B):
        c = rng.choice(EMPLOYER_C)
        name = f"{a} {b}" if not c else f"{a} {b} {c}"
        if 2 <= len(name.split()) <= 3:
            cands.append(name)
    cands = list(dict.fromkeys(cands))
    rng.shuffle(cands)
    return cands


def invent_universities(n: int, rng: random.Random) -> list[str]:
    del n
    cands = [f"{stem} {typ}" for stem, typ in itertools.product(UNIV_STEMS, UNIV_TYPES)]
    cands = list(dict.fromkeys(cands))
    rng.shuffle(cands)
    return cands


def invent_titles(n: int, rng: random.Random) -> list[str]:
    cands = [f"The {a} {b}" for a, b in itertools.product(TITLE_ADJECTIVES, TITLE_NOUNS)]
    cands += [f"{a} {b}" for a, b in itertools.product(TITLE_ADJECTIVES, TITLE_NOUNS)]
    cands = list(dict.fromkeys(cands))
    rng.shuffle(cands)
    return cands[: max(n * 5, n)]


def filter_zero_hits(
    candidates: list[str],
    *,
    n_keep: int,
    api_url: str,
    index: str,
    logger: IndexQueryLogger | None = None,
    purpose: str = "pool",
) -> list[str]:
    kept: list[str] = []
    with httpx.Client() as client:
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
            if len(kept) >= n_keep:
                break
    if len(kept) < n_keep:
        raise RuntimeError(f"Only found {len(kept)} zero-hit pool values, need {n_keep}")
    return kept[:n_keep]


def build_pools(
    config: dict[str, Any],
    rng: random.Random,
    logger: IndexQueryLogger | None = None,
) -> dict[str, list[str]]:
    pools_cfg = config["pools"]
    idx = config["index"]
    n_titles = config.get("templates", {}).get("n_titles", 20)
    employers = filter_zero_hits(
        invent_employers(200, rng),
        n_keep=pools_cfg["n_employer"],
        api_url=idx["api_url"],
        index=idx["name"],
        logger=logger,
        purpose="employer",
    )
    universities = filter_zero_hits(
        invent_universities(200, rng),
        n_keep=pools_cfg["n_university"],
        api_url=idx["api_url"],
        index=idx["name"],
        logger=logger,
        purpose="university",
    )
    titles = filter_zero_hits(
        invent_titles(n_titles * 8, rng),
        n_keep=n_titles,
        api_url=idx["api_url"],
        index=idx["name"],
        logger=logger,
        purpose="title",
    )
    return {
        "birthplace": list(pools_cfg["birthplace"]),
        "employer": employers,
        "university": universities,
        "hobby": list(pools_cfg["hobby"]),
        "titles": titles,
    }


def save_pools(pools: dict[str, list[str]], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(pools, indent=2))
