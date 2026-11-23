"""Shared probe distractors for cross-arm endorsement (R0 / R1 / R2)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from grainofsalt.brief5_assign import SLOTS

SEED = 20261122


def _stable_u64(*parts: str) -> int:
    h = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


def probe_distractor_for(
    *,
    person_id: str,
    slot: str,
    gold: str,
    pools: dict[str, list[str]],
    seed: int = SEED,
) -> str:
    """Distractor depends only on (person_id, slot, gold, seed, pool membership).

    Never equals ``gold``. Deterministic across arms.
    """
    if slot not in pools:
        raise KeyError(slot)
    others = [v for v in pools[slot] if v != gold]
    if not others:
        raise ValueError(f"no distractor candidates for {slot} gold={gold!r}")
    # Seed stream from (seed, person_id, slot, gold) only — not iteration order.
    rng = np.random.default_rng(
        seed + (_stable_u64(person_id, slot, gold) % (2**31 - 1))
    )
    return others[int(rng.integers(0, len(others)))]


def build_probe_distractor_table(
    *,
    people: pd.DataFrame,
    facts: pd.DataFrame,
    pools: dict[str, list[str]],
    seed: int = SEED,
) -> pd.DataFrame:
    probes = people[people["role"] == "probe"]
    fact_cols = ["person_id", *SLOTS]
    merged = probes.merge(facts[fact_cols], on="person_id")
    rows: list[dict[str, Any]] = []
    for _, r in merged.iterrows():
        pid = r["person_id"]
        for slot in SLOTS:
            gold = r[slot]
            dist = probe_distractor_for(
                person_id=pid, slot=slot, gold=gold, pools=pools, seed=seed
            )
            rows.append(
                {
                    "person_id": pid,
                    "slot": slot,
                    "gold": gold,
                    "distractor": dist,
                    "seed": seed,
                }
            )
    return pd.DataFrame(rows)


def load_probe_distractors(path: Path) -> dict[tuple[str, str], str]:
    df = pd.read_csv(path)
    return {
        (r.person_id, r.slot): r.distractor
        for r in df.itertuples(index=False)
    }


def write_probe_distractors(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
