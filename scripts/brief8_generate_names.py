#!/usr/bin/env python3
"""Brief 8: generate Step 3 names (1100 + 15% spare), excluding Briefs 1–6 names."""

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

SEED = 20261121
OUT = ROOT / "data" / "step3" / "names"
N_TARGET = 1100
N_SPARE_FRAC = 0.15
N_WANT = int(N_TARGET * (1 + N_SPARE_FRAC))  # 1265
N_CANDIDATES = 8000
SLOTS = ["birthplace", "employer", "university", "hobby"]
MAX_PART_USES = 6


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


def surname_ok(name: str, blocked_surnames: set[str]) -> bool:
    sur = name.split()[-1]
    for ps in blocked_surnames:
        if edit_distance(sur, ps) <= 2:
            return False
    return True


def first_last_key(name: str) -> str:
    parts = name.split()
    if len(parts) < 2:
        return name.lower()
    return f"{parts[0].lower()}|{parts[-1].lower()}"


def load_prior_exclusions() -> tuple[set[str], set[str], set[str]]:
    """Full names + first|last pairs from Briefs 1–6; pilot surnames for edit≥3."""
    paths = [
        ROOT / "data" / "names" / "names_final.csv",
        ROOT / "data" / "step2" / "names" / "names_final.csv",
    ]
    full: set[str] = set()
    pairs: set[str] = set()
    for p in paths:
        df = pd.read_csv(p)
        for name in df["name"].astype(str):
            full.add(name.lower())
            pairs.add(first_last_key(name))
    # Step-2 surname edit screen was vs pilot only — keep that unchanged.
    pilot = pd.read_csv(ROOT / "data" / "names" / "names_final.csv")
    pilot_surnames = set(pilot["name"].str.split().str[-1].str.lower())
    return full, pairs, pilot_surnames


def generate_candidates(
    n: int,
    rng: random.Random,
    *,
    blocked_full: set[str],
    blocked_pairs: set[str],
    pilot_surnames: set[str],
) -> list[str]:
    firsts = list(dict.fromkeys(f for f in FIRST_NAMES if f.isascii()))
    raw_surs = list(
        dict.fromkeys(s for s in SURNAMES if s.isascii() and not _has_fantasy_lexeme(s))
    )
    # Same as Step 2: drop surnames within edit≤2 of pilot surnames.
    surnames = [s for s in raw_surs if surname_ok(f"X {s}", pilot_surnames)]
    if len(surnames) < 80:
        raise RuntimeError(f"too few edit-safe surnames: {len(surnames)}")
    rng.shuffle(firsts)
    rng.shuffle(surnames)
    sur_cap = max(MAX_PART_USES, (n // max(len(surnames), 1)) + 4)
    first_cap = max(MAX_PART_USES, (n // max(len(firsts), 1)) + 3)
    first_uses = {f: 0 for f in firsts}
    sur_uses = {s: 0 for s in surnames}
    names: list[str] = []
    seen: set[str] = set()
    pair_list = [(f, s) for f in firsts for s in surnames]
    rng.shuffle(pair_list)
    for f, s in pair_list:
        if first_uses[f] >= first_cap or sur_uses[s] >= sur_cap:
            continue
        full = f"{f} {s}"
        key = full.lower()
        if (
            key in blocked_full
            or key in seen
            or first_last_key(full) in blocked_pairs
            or not _is_ascii_alpha_space(full)
        ):
            continue
        seen.add(key)
        first_uses[f] += 1
        sur_uses[s] += 1
        names.append(full)
        if len(names) >= n:
            break
    if len(names) < n:
        middles = [
            "Ann",
            "Lee",
            "James",
            "Marie",
            "Ray",
            "Kay",
            "Jo",
            "Mae",
            "Ash",
            "Skye",
            "Blair",
            "Quinn",
        ]
        triples = [(f, m, s) for f in firsts for m in middles for s in surnames]
        rng.shuffle(triples)
        for f, m, s in triples:
            full = f"{f} {m} {s}"
            key = full.lower()
            if (
                key in blocked_full
                or key in seen
                or first_last_key(full) in blocked_pairs
                or not _is_ascii_alpha_space(full)
            ):
                continue
            seen.add(key)
            names.append(full)
            if len(names) >= n:
                break
    if len(names) < n:
        raise RuntimeError(f"Only generated {len(names)} candidates, need {n}")
    return names[:n]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["all", "index", "kl"], default="index")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config()
    rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)

    pref_src = ROOT / "reports" / "pool_pref.json"
    if pref_src.exists():
        (OUT / "pool_pref.json").write_text(pref_src.read_text())

    blocked_full, blocked_pairs, pilot_surnames = load_prior_exclusions()
    print(
        f"excluded full={len(blocked_full)} pairs={len(blocked_pairs)} "
        f"pilot_surnames={len(pilot_surnames)}",
        flush=True,
    )

    if args.phase in ("all", "index"):
        print(f"generating {N_CANDIDATES} candidates…", flush=True)
        candidates = generate_candidates(
            N_CANDIDATES,
            rng,
            blocked_full=blocked_full,
            blocked_pairs=blocked_pairs,
            pilot_surnames=pilot_surnames,
        )
        rows = tokenize_counts(
            candidates, config["model"]["hf_id"], config["model"]["revision"]
        )
        filtered = filter_token_length(
            rows, config["names"]["token_min"], config["names"]["token_max"]
        )
        filtered = [r for r in filtered if surname_ok(r["name"], pilot_surnames)]
        filtered = [
            r
            for r in filtered
            if r["name"].lower() not in blocked_full
            and first_last_key(r["name"]) not in blocked_pairs
        ]
        seen: set[str] = set()
        uniq = []
        for r in filtered:
            k = r["name"].lower()
            if k in seen:
                continue
            seen.add(k)
            uniq.append(r)
        filtered = uniq
        print(f"after token+exclude+dedup: {len(filtered)}", flush=True)
        (OUT / "candidates_tokenized.jsonl").write_text(
            "\n".join(
                json.dumps({"name": r["name"], "n_tokens": r["n_tokens"]}) for r in filtered
            )
            + "\n"
        )
        if len(filtered) < N_WANT + 300:
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
            purpose_prefix="step3_name",
            stop_after_zero=N_WANT + 400,
        )
        (OUT / "index_results.jsonl").write_text(
            "\n".join(json.dumps(r) for r in index_rows) + "\n"
        )
        zero = [r for r in index_rows if r["index_hits_full"] == 0]
        tok_map = {r["name"]: r["n_tokens"] for r in filtered}
        for r in zero:
            r["n_tokens"] = tok_map.get(r["name"], 0)
        print(f"zero-hit: {len(zero)} queried={len(index_rows)}", flush=True)
        if len(zero) < N_WANT:
            raise SystemExit(f"not enough zero-hit names: {len(zero)} < {N_WANT}")
        if len(zero) > 2200:
            rng.shuffle(zero)
            zero = zero[:2200]
        (OUT / "zero_hit_names.json").write_text(json.dumps(zero, indent=2))
        print(f"wrote zero_hit n={len(zero)}", flush=True)
        if args.phase == "index":
            print("phase=index done; run Modal step3_kl or --phase kl", flush=True)
            return

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
    if n_keep < N_WANT:
        excl = excl.sort_values("max_kl").copy()
        excl["keep"] = False
        excl.iloc[:N_WANT, excl.columns.get_loc("keep")] = True
        print(f"softened keep to {N_WANT} lowest max_kl", flush=True)

    selected = select_lowest_kl_balanced(excl, N_WANT, np_rng)
    # First 1100 are the used set; remainder are spare
    selected = selected.copy().reset_index(drop=True)
    selected["person_id"] = [f"T{i:04d}" for i in range(len(selected))]
    selected["is_spare"] = [i >= N_TARGET for i in range(len(selected))]

    assert set(selected["name"].str.lower()).isdisjoint(blocked_full)
    for name in selected["name"]:
        assert first_last_key(name) not in blocked_pairs, name
        assert surname_ok(name, pilot_surnames), name

    write_names_final(selected, OUT / "names_final.csv")
    # Preserve spare flag alongside the standard columns
    final_df = pd.read_csv(OUT / "names_final.csv")
    final_df["is_spare"] = selected["is_spare"].astype(bool).values
    final_df.to_csv(OUT / "names_final.csv", index=False)
    used = final_df[~final_df["is_spare"]].drop(columns=["is_spare"])
    used.to_csv(OUT / "names_used.csv", index=False)
    meta = {
        "n_used": int((~selected["is_spare"]).sum()),
        "n_spare": int(selected["is_spare"].sum()),
        "n_total_screened": len(selected),
        "seed": SEED,
        "n_candidates": N_CANDIDATES,
        "n_zero_hit": len(zero),
        "n_kl_survivors": n_keep,
        "prior_excluded_full": len(blocked_full),
        "prior_excluded_pairs": len(blocked_pairs),
        "max_kl_max": float(selected["max_kl"].max()),
        "max_kl_mean": float(selected["max_kl"].mean()),
    }
    (OUT / "names_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
