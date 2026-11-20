#!/usr/bin/env python3
"""Brief 5: generate 500 new names (disjoint from pilot; surname edit distance > 2)."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grainofsalt.logging_utils import load_config  # noqa: E402
from grainofsalt.names import (  # noqa: E402
    DEFAULT_RECALL_PROMPTS,
    FIRST_NAMES,
    SURNAMES,
    IndexQueryLogger,
    _has_fantasy_lexeme,
    _is_ascii_alpha_space,
    apply_kl_exclusion,
    behavioral_familiarity_kl,
    check_index_absence,
    filter_token_length,
    load_base_model,
    select_lowest_kl_balanced,
    tokenize_counts,
    write_names_final,
)

SEED = 20261118
OUT = ROOT / "data" / "step2" / "names"
N_TARGET = 500
N_CANDIDATES = 3500  # pool size after filtering; generator allows higher part reuse
SLOTS = ["birthplace", "employer", "university", "hobby"]
MAX_PART_USES = 5


def generate_candidates_step2(
    n: int,
    rng: random.Random,
    *,
    pilot_surnames: set[str],
    pilot_names: set[str],
    max_uses: int = MAX_PART_USES,
) -> list[str]:
    """A2-style names using only surnames with edit distance > 2 vs all pilot surnames."""
    firsts = list(dict.fromkeys(f for f in FIRST_NAMES if f.isascii()))
    raw_surs = list(
        dict.fromkeys(s for s in SURNAMES if s.isascii() and not _has_fantasy_lexeme(s))
    )
    surnames = [s for s in raw_surs if surname_ok(f"X {s}", pilot_surnames)]
    if len(surnames) < 50:
        raise RuntimeError(f"too few edit-safe surnames: {len(surnames)}")
    rng.shuffle(firsts)
    rng.shuffle(surnames)
    # Allow higher surname reuse: few surviving surnames must cover n candidates
    sur_cap = max(max_uses, (n // max(len(surnames), 1)) + 3)
    first_cap = max(max_uses, (n // max(len(firsts), 1)) + 2)
    first_uses = {f: 0 for f in firsts}
    sur_uses = {s: 0 for s in surnames}
    names: list[str] = []
    seen: set[str] = set()
    pairs = [(f, s) for f in firsts for s in surnames]
    rng.shuffle(pairs)
    for f, s in pairs:
        if first_uses[f] >= first_cap or sur_uses[s] >= sur_cap:
            continue
        full = f"{f} {s}"
        if full.lower() in pilot_names or full in seen or not _is_ascii_alpha_space(full):
            continue
        seen.add(full)
        first_uses[f] += 1
        sur_uses[s] += 1
        names.append(full)
        if len(names) >= n:
            break
    # Top up with middle-name forms if still short (3–5 tokens)
    if len(names) < n:
        middles = ["Ann", "Lee", "James", "Marie", "Ray", "Kay", "Jo", "Mae", "Ash", "Skye"]
        triples = [(f, m, s) for f in firsts for m in middles for s in surnames]
        rng.shuffle(triples)
        for f, m, s in triples:
            full = f"{f} {m} {s}"
            if full.lower() in pilot_names or full in seen or not _is_ascii_alpha_space(full):
                continue
            seen.add(full)
            names.append(full)
            if len(names) >= n:
                break
    if len(names) < n:
        raise RuntimeError(f"Only generated {len(names)} candidates, need {n}")
    return names[:n]


def edit_distance(a: str, b: str) -> int:
    a, b = a.lower(), b.lower()
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins, delete, sub = cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def surname_ok(name: str, pilot_surnames: set[str]) -> bool:
    sur = name.split()[-1]
    for ps in pilot_surnames:
        if edit_distance(sur, ps) <= 2:
            return False
    return True


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        choices=["all", "index", "kl"],
        default="all",
        help="index=stop after zero-hit list; kl=score existing zero_hit_names.json; all=both",
    )
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config()
    rng = random.Random(SEED + 5)
    np_rng = np.random.default_rng(SEED + 5)

    # Always stage pool_pref for Modal KL
    pref_src = ROOT / "reports" / "pool_pref.json"
    if pref_src.exists():
        (OUT / "pool_pref.json").write_text(pref_src.read_text())

    pilot = pd.read_csv(ROOT / "data" / "names" / "names_final.csv")
    pilot_names = set(pilot["name"].str.lower())
    pilot_surnames = set(pilot["name"].str.split().str[-1].str.lower())

    if args.phase in ("all", "index"):
        print(f"pilot names={len(pilot_names)} surnames={len(pilot_surnames)}", flush=True)
        print(f"generating {N_CANDIDATES} candidates…", flush=True)
        candidates = generate_candidates_step2(
            N_CANDIDATES,
            rng,
            pilot_surnames=pilot_surnames,
            pilot_names=pilot_names,
        )
        rows = tokenize_counts(
            candidates, config["model"]["hf_id"], config["model"]["revision"]
        )
        filtered = filter_token_length(
            rows, config["names"]["token_min"], config["names"]["token_max"]
        )
        filtered = [r for r in filtered if surname_ok(r["name"], pilot_surnames)]
        seen: set[str] = set()
        uniq = []
        for r in filtered:
            k = r["name"].lower()
            if k in seen:
                continue
            seen.add(k)
            uniq.append(r)
        filtered = uniq
        print(f"after token+edit+dedup: {len(filtered)}", flush=True)
        (OUT / "candidates_tokenized.jsonl").write_text(
            "\n".join(
                json.dumps({"name": r["name"], "n_tokens": r["n_tokens"]}) for r in filtered
            )
            + "\n"
        )
        if len(filtered) < N_TARGET + 200:
            raise SystemExit(f"too few candidates: {len(filtered)}")

        idx = config["index"]
        index_log = IndexQueryLogger(OUT / "index_queries.jsonl")
        names_for_index = [r["name"] for r in filtered]
        rng.shuffle(names_for_index)
        print("index absence check…", flush=True)
        index_rows = check_index_absence(
            names_for_index,
            api_url=idx["api_url"],
            index=idx["name"],
            logger=index_log,
            purpose_prefix="step2_name",
            stop_after_zero=N_TARGET + 150,
        )
        (OUT / "index_results.jsonl").write_text(
            "\n".join(json.dumps(r) for r in index_rows) + "\n"
        )
        zero = [r for r in index_rows if r["index_hits_full"] == 0]
        tok_map = {r["name"]: r["n_tokens"] for r in filtered}
        for r in zero:
            r["n_tokens"] = tok_map.get(r["name"], 0)
        print(f"zero-hit: {len(zero)} queried={len(index_rows)}", flush=True)
        if len(zero) < N_TARGET:
            raise SystemExit(f"not enough zero-hit names: {len(zero)}")
        # Cap for KL compute
        if len(zero) > 700:
            rng.shuffle(zero)
            zero = zero[:700]
        (OUT / "zero_hit_names.json").write_text(json.dumps(zero, indent=2))
        print(f"wrote {OUT / 'zero_hit_names.json'} n={len(zero)}", flush=True)
        if args.phase == "index":
            print("phase=index done; run Modal b5_kl or --phase kl", flush=True)
            return

    # --- KL phase ---
    zero = json.loads((OUT / "zero_hit_names.json").read_text())
    zero_names = [r["name"] for r in zero]
    tok_map = {r["name"]: r["n_tokens"] for r in zero}
    zero_map = {r["name"]: r for r in zero}

    pools_all = json.loads((ROOT / "data/assignments/pools.json").read_text())
    pref_report = json.loads((ROOT / "reports/pool_pref.json").read_text())
    fact_pools = {k: pools_all[k] for k in SLOTS}
    p_refs = {
        slot: np.array(pref_report[slot]["p_ref"], dtype=np.float64) for slot in SLOTS
    }

    print("loading model for KL…", flush=True)
    model, tokenizer, device = load_base_model(
        config["model"]["hf_id"], config["model"]["revision"]
    )
    print(f"device={device} scoring {len(zero_names)} names…", flush=True)
    summary, _scores = behavioral_familiarity_kl(
        zero_names,
        fact_pools,
        p_refs,
        DEFAULT_RECALL_PROMPTS,
        model=model,
        tokenizer=tokenizer,
        device=device,
        scores_out=OUT / "familiarity_scores.csv",
    )
    for s in summary:
        s["n_tokens"] = tok_map[s["name"]]
        s["index_hits_full"] = zero_map[s["name"]]["index_hits_full"]
        s["index_hits_surname"] = zero_map[s["name"]].get("index_hits_surname", -1)

    excl = apply_kl_exclusion(summary, slots=SLOTS)
    excl.to_csv(OUT / "familiarity_kl_summary.csv", index=False)
    (OUT / "familiarity.jsonl").write_text(
        "\n".join(json.dumps(r) for r in excl.to_dict(orient="records")) + "\n"
    )
    n_keep = int(excl["keep"].sum())
    print(f"KL survivors: {n_keep} / {len(excl)}", flush=True)
    if n_keep < N_TARGET:
        excl = excl.sort_values("max_kl").copy()
        excl["keep"] = False
        excl.iloc[:N_TARGET, excl.columns.get_loc("keep")] = True
        print(f"softened keep to {N_TARGET} lowest max_kl", flush=True)

    selected = select_lowest_kl_balanced(excl, N_TARGET, np_rng)
    selected = selected.copy()
    selected["person_id"] = [f"S{i:04d}" for i in range(len(selected))]

    assert set(selected["name"].str.lower()).isdisjoint(pilot_names)
    for name in selected["name"]:
        assert surname_ok(name, pilot_surnames), name

    write_names_final(selected, OUT / "names_final.csv")
    meta = {
        "n": len(selected),
        "seed": SEED + 5,
        "n_candidates": N_CANDIDATES,
        "n_zero_hit": len(zero),
        "n_kl_survivors": n_keep,
        "pilot_disjoint": True,
        "surname_edit_min": 3,
        "max_kl_max": float(selected["max_kl"].max()),
        "max_kl_mean": float(selected["max_kl"].mean()),
    }
    (OUT / "names_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
