#!/usr/bin/env python3
"""Brief 13 confirmatory: 500 fresh names → data/confirm/names/.

All newly screened (no prior-phase spares). Disjoint from pilot / Step2 / Step3
(used+spare) / R0 / R1 / R1a / R1b. Seed 20261124.
"""

from __future__ import annotations

import importlib.util
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
    FIRST_NAMES,
    IndexQueryLogger,
    _has_fantasy_lexeme,
    _is_ascii_alpha_space,
    apply_kl_exclusion,
    check_index_absence,
    filter_token_length,
    select_lowest_kl_balanced,
    tokenize_counts,
    write_names_final,
)

_spec = importlib.util.spec_from_file_location(
    "brief5_generate_names", ROOT / "scripts" / "brief5_generate_names.py"
)
_b5 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_b5)
surname_ok = _b5.surname_ok

SEED = 20261124
OUT = ROOT / "data" / "confirm" / "names"
N_TARGET = 500
N_CANDIDATES = 6000
N_ZERO_CAP = 900
SLOTS = ["birthplace", "employer", "university", "hobby"]

PRIOR_NAME_CSVS = [
    ROOT / "data" / "names" / "names_final.csv",  # pilot
    ROOT / "data" / "step2" / "names" / "names_final.csv",
    ROOT / "data" / "step3" / "names" / "names_final.csv",  # used + spare
    ROOT / "data" / "R0" / "names" / "names_final.csv",
    ROOT / "data" / "R1" / "names" / "names_final.csv",
    ROOT / "data" / "R1a" / "names" / "names_final.csv",
    ROOT / "data" / "R1b" / "names" / "names_final.csv",
]


def blocked_sets() -> tuple[set[str], set[str], dict[str, int]]:
    names: set[str] = set()
    surnames: set[str] = set()
    counts: dict[str, int] = {}
    for path in PRIOR_NAME_CSVS:
        if not path.exists():
            counts[str(path.relative_to(ROOT))] = 0
            continue
        df = pd.read_csv(path)
        n0 = len(names)
        for n in df["name"].astype(str):
            names.add(n.lower())
            surnames.add(n.split()[-1].lower())
        counts[str(path.relative_to(ROOT))] = len(names) - n0
    return names, surnames, counts


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["index", "finalize", "check"], default="index")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config()
    rng = random.Random(SEED + 7)
    np_rng = np.random.default_rng(SEED + 7)
    blocked_names, blocked_surnames, counts = blocked_sets()

    if args.phase == "check":
        final = pd.read_csv(OUT / "names_final.csv")
        assert len(final) == N_TARGET
        assert set(final["name"].str.lower()).isdisjoint(blocked_names)
        print(json.dumps({"ok": True, "n": len(final), "blocked": len(blocked_names)}, indent=2))
        return

    if args.phase == "index":
        print(
            json.dumps(
                {"blocked_names": len(blocked_names), "blocked_surnames": len(blocked_surnames), "sources": counts},
                indent=2,
            ),
            flush=True,
        )
        (OUT / "pool_pref.json").write_text((ROOT / "reports" / "pool_pref.json").read_text())

        prefixes = [
            "Ash", "Bel", "Cor", "Dal", "Fen", "Gor", "Hal", "Irv", "Kel", "Lor",
            "Mar", "Nor", "Orl", "Pel", "Quin", "Rav", "Sel", "Tor", "Ulm", "Varn",
            "Wel", "Xor", "Yel", "Zan", "Bri", "Cres", "Dun", "Elm", "Flint", "Glen",
            "Har", "Jex", "Kael", "Lym", "Morr", "Nyx", "Orr", "Pry", "Quill", "Rex",
        ]
        suffixes = [
            "wick", "hurst", "ford", "mere", "shaw", "ridge", "vale", "croft",
            "stead", "worth", "field", "brook", "haven", "mont", "wood", "cliff",
            "spire", "grove", "marsh", "crest", "dale", "burn", "holt", "thorpe",
        ]
        synth_surs = []
        for p in prefixes:
            for s in suffixes:
                sur = p + s
                if _has_fantasy_lexeme(sur):
                    continue
                if not sur.isascii() or not sur.isalpha():
                    continue
                if surname_ok(f"X {sur}", blocked_surnames):
                    synth_surs.append(sur)
        synth_surs = list(dict.fromkeys(synth_surs))
        print(f"synth edit-safe surnames: {len(synth_surs)}", flush=True)
        if len(synth_surs) < 100:
            raise SystemExit(f"too few synth surnames: {len(synth_surs)}")

        firsts = [f for f in FIRST_NAMES if f.isascii()]
        rng.shuffle(firsts)
        rng.shuffle(synth_surs)
        middles = ["Ann", "Lee", "James", "Marie", "Ray", "Kay", "Jo", "Mae", "Ash", "Skye", "Blair", "Quinn"]
        candidates: list[str] = []
        seen_c: set[str] = set()
        for f in firsts:
            for s in synth_surs:
                full = f"{f} {s}"
                if full.lower() in blocked_names or full.lower() in seen_c:
                    continue
                if not _is_ascii_alpha_space(full):
                    continue
                seen_c.add(full.lower())
                candidates.append(full)
                if len(candidates) >= N_CANDIDATES:
                    break
            if len(candidates) >= N_CANDIDATES:
                break
        if len(candidates) < N_CANDIDATES:
            for f in firsts:
                for m in middles:
                    for s in synth_surs:
                        full = f"{f} {m} {s}"
                        if full.lower() in blocked_names or full.lower() in seen_c:
                            continue
                        if not _is_ascii_alpha_space(full):
                            continue
                        seen_c.add(full.lower())
                        candidates.append(full)
                        if len(candidates) >= N_CANDIDATES:
                            break
                    if len(candidates) >= N_CANDIDATES:
                        break
                if len(candidates) >= N_CANDIDATES:
                    break
        print(f"generated candidates: {len(candidates)}", flush=True)
        candidates = [c for c in candidates if c.lower() not in blocked_names]
        rows = tokenize_counts(
            candidates, config["model"]["hf_id"], config["model"]["revision"]
        )
        filtered = filter_token_length(
            rows, config["names"]["token_min"], config["names"]["token_max"]
        )
        filtered = [r for r in filtered if surname_ok(r["name"], blocked_surnames)]
        filtered = [r for r in filtered if r["name"].lower() not in blocked_names]
        seen: set[str] = set()
        uniq = []
        for r in filtered:
            k = r["name"].lower()
            if k in seen:
                continue
            seen.add(k)
            uniq.append(r)
        filtered = uniq
        print(f"after filters: {len(filtered)}", flush=True)
        (OUT / "candidates_tokenized.jsonl").write_text(
            "\n".join(json.dumps({"name": r["name"], "n_tokens": r["n_tokens"]}) for r in filtered)
            + "\n"
        )
        if len(filtered) < N_TARGET + 250:
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
            purpose_prefix="confirm_name",
            stop_after_zero=N_TARGET + 250,
        )
        (OUT / "index_results.jsonl").write_text(
            "\n".join(json.dumps(r) for r in index_rows) + "\n"
        )
        zero = [r for r in index_rows if r["index_hits_full"] == 0]
        tok_map = {r["name"]: r["n_tokens"] for r in filtered}
        for r in zero:
            r["n_tokens"] = tok_map.get(r["name"], 0)
        print(f"zero-hit: {len(zero)}", flush=True)
        if len(zero) < N_TARGET:
            raise SystemExit(f"not enough zero-hit: {len(zero)}")
        if len(zero) > N_ZERO_CAP:
            rng.shuffle(zero)
            zero = zero[:N_ZERO_CAP]
        (OUT / "zero_hit_names.json").write_text(json.dumps(zero, indent=2))
        meta = {
            "seed": SEED,
            "n_zero_hit": len(zero),
            "n_target": N_TARGET,
            "blocked_names": len(blocked_names),
            "blocked_surnames": len(blocked_surnames),
            "sources_blocked": counts,
        }
        (OUT / "phase_index_meta.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2), flush=True)
        print("next: Modal KL then --phase finalize", flush=True)
        return

    if args.phase == "finalize":
        excl = pd.read_csv(OUT / "familiarity_kl_summary.csv")
        if "keep" not in excl.columns:
            excl = apply_kl_exclusion(excl.to_dict(orient="records"), slots=SLOTS)
        n_keep = int(excl["keep"].sum())
        print(f"KL survivors: {n_keep}", flush=True)
        if n_keep < N_TARGET:
            excl = excl.sort_values("max_kl").copy()
            excl["keep"] = False
            excl.iloc[:N_TARGET, excl.columns.get_loc("keep")] = True
        selected = select_lowest_kl_balanced(excl, N_TARGET, np_rng)
        selected = selected[~selected["name"].str.lower().isin(blocked_names)].copy()
        if len(selected) < N_TARGET:
            rest = excl[~excl["name"].isin(selected["name"])].sort_values("max_kl")
            rest = rest[~rest["name"].str.lower().isin(blocked_names)]
            need = N_TARGET - len(selected)
            selected = pd.concat([selected, rest.head(need)], ignore_index=True)
        selected = selected.head(N_TARGET).copy()
        assert len(selected) == N_TARGET, len(selected)
        assert set(selected["name"].str.lower()).isdisjoint(blocked_names)

        selected = selected.copy()
        selected["source"] = "newly_screened"
        selected["person_id"] = [f"C{i:04d}" for i in range(len(selected))]
        write_names_final(selected, OUT / "names_final.csv")
        final = pd.read_csv(OUT / "names_final.csv")
        final["source"] = selected["source"].values
        final.to_csv(OUT / "names_final.csv", index=False)

        # Re-assert disjointness against every prior CSV
        for path in PRIOR_NAME_CSVS:
            if not path.exists():
                continue
            prior = set(pd.read_csv(path)["name"].str.lower())
            overlap = set(final["name"].str.lower()) & prior
            if overlap:
                raise SystemExit(f"overlap with {path}: {list(overlap)[:5]}")

        meta = {
            "n": len(final),
            "n_new": N_TARGET,
            "seed": SEED,
            "seed_kl": SEED + 7,
            "sources": final["source"].value_counts().to_dict(),
            "disjoint_all_prior": True,
            "blocked_names": len(blocked_names),
        }
        (OUT / "names_meta.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
