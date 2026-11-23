#!/usr/bin/env python3
"""Brief 10 R0: 165 Step3 spares + 335 newly screened names → data/R0/names/."""

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
    IndexQueryLogger,
    apply_kl_exclusion,
    check_index_absence,
    filter_token_length,
    select_lowest_kl_balanced,
    tokenize_counts,
    write_names_final,
)

# Reuse Step 2 candidate generator via importlib (scripts/ is not a package)
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "brief5_generate_names", ROOT / "scripts" / "brief5_generate_names.py"
)
_b5 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_b5)
generate_candidates_step2 = _b5.generate_candidates_step2
surname_ok = _b5.surname_ok

SEED = 20261122
OUT = ROOT / "data" / "R0" / "names"
N_TARGET = 500
N_SPARES = 165
N_NEW = N_TARGET - N_SPARES  # 335
N_CANDIDATES = 4000
SLOTS = ["birthplace", "employer", "university", "hobby"]


def _blocked() -> tuple[set[str], set[str]]:
    """All prior-phase names + surnames (pilot, Step2, Step3 used+spare)."""
    names: set[str] = set()
    surnames: set[str] = set()

    def add_csv(path: Path, name_col: str = "name") -> None:
        if not path.exists():
            return
        df = pd.read_csv(path)
        for n in df[name_col].astype(str):
            names.add(n.lower())
            surnames.add(n.split()[-1].lower())

    add_csv(ROOT / "data" / "names" / "names_final.csv")
    add_csv(ROOT / "data" / "step2" / "names" / "names_final.csv")
    add_csv(ROOT / "data" / "step3" / "names" / "names_final.csv")
    return names, surnames


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["assemble_spares", "index", "finalize"], default="assemble_spares")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config()
    rng = random.Random(SEED + 5)
    np_rng = np.random.default_rng(SEED + 5)

    step3 = pd.read_csv(ROOT / "data" / "step3" / "names" / "names_final.csv")
    spares = step3[step3["is_spare"].astype(bool)].copy()
    assert len(spares) == N_SPARES, len(spares)
    # Normalize spare columns for merge with new selects
    spare_cols = ["name", "n_tokens", "index_hits_full", "index_hits_surname", "max_kl", "max_dev", "entropy_who"]
    for c in spare_cols:
        if c not in spares.columns:
            spares[c] = np.nan if c != "name" else ""
    spares = spares[spare_cols].copy()
    spares["source"] = "step3_spare"

    blocked_names, blocked_surnames = _blocked()
    # When generating NEW names, also block the spare names themselves
    spare_names = set(spares["name"].str.lower())
    blocked_names |= spare_names
    blocked_surnames |= set(spares["name"].str.split().str[-1].str.lower())

    if args.phase == "assemble_spares":
        spares.to_csv(OUT / "spares_165.csv", index=False)
        (OUT / "pool_pref.json").write_text((ROOT / "reports" / "pool_pref.json").read_text())
        meta = {
            "seed": SEED,
            "n_spares": len(spares),
            "n_new_needed": N_NEW,
            "blocked_names": len(blocked_names),
            "blocked_surnames": len(blocked_surnames),
        }
        (OUT / "phase_spares_meta.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2))
        print("next: --phase index", flush=True)
        return

    if args.phase == "index":
        print(f"generating {N_CANDIDATES} candidates…", flush=True)
        # Lexicon surnames are nearly exhausted vs 700+ blocked surnames
        # (edit-distance ≤2). Synthesize novel ASCII surnames that clear the bar.
        from grainofsalt.names import FIRST_NAMES, _has_fantasy_lexeme, _is_ascii_alpha_space

        prefixes = [
            "Ash", "Bel", "Cor", "Dal", "Fen", "Gor", "Hal", "Irv", "Kel", "Lor",
            "Mar", "Nor", "Orl", "Pel", "Quin", "Rav", "Sel", "Tor", "Ulm", "Varn",
            "Wel", "Xor", "Yel", "Zan", "Bri", "Cres", "Dun", "Elm", "Flint", "Glen",
        ]
        suffixes = [
            "wick", "hurst", "ford", "mere", "shaw", "ridge", "vale", "croft",
            "stead", "worth", "field", "brook", "haven", "mont", "wood", "cliff",
            "spire", "grove", "marsh", "crest",
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
        if len(synth_surs) < 80:
            raise SystemExit(f"too few synth surnames: {len(synth_surs)}")

        firsts = [f for f in FIRST_NAMES if f.isascii()]
        rng.shuffle(firsts)
        rng.shuffle(synth_surs)
        middles = ["Ann", "Lee", "James", "Marie", "Ray", "Kay", "Jo", "Mae", "Ash", "Skye"]
        candidates = []
        seen_c: set[str] = set()
        # Prefer first+surname, then first+middle+surname
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
        # Drop any that still collide
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
        if len(filtered) < N_NEW + 200:
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
            purpose_prefix="r0_name",
            stop_after_zero=N_NEW + 200,
        )
        (OUT / "index_results.jsonl").write_text(
            "\n".join(json.dumps(r) for r in index_rows) + "\n"
        )
        zero = [r for r in index_rows if r["index_hits_full"] == 0]
        tok_map = {r["name"]: r["n_tokens"] for r in filtered}
        for r in zero:
            r["n_tokens"] = tok_map.get(r["name"], 0)
        print(f"zero-hit: {len(zero)}", flush=True)
        if len(zero) < N_NEW:
            raise SystemExit(f"not enough zero-hit: {len(zero)}")
        if len(zero) > 800:
            rng.shuffle(zero)
            zero = zero[:800]
        (OUT / "zero_hit_names.json").write_text(json.dumps(zero, indent=2))
        print(f"wrote zero_hit n={len(zero)}; run Modal KL then --phase finalize", flush=True)
        return

    if args.phase == "finalize":
        # Expect familiarity_kl_summary.csv from Modal KL (or local)
        excl = pd.read_csv(OUT / "familiarity_kl_summary.csv")
        if "keep" not in excl.columns:
            excl = apply_kl_exclusion(excl.to_dict(orient="records"), slots=SLOTS)
        n_keep = int(excl["keep"].sum())
        print(f"KL survivors: {n_keep}", flush=True)
        if n_keep < N_NEW:
            excl = excl.sort_values("max_kl").copy()
            excl["keep"] = False
            excl.iloc[:N_NEW, excl.columns.get_loc("keep")] = True
        selected_new = select_lowest_kl_balanced(excl, N_NEW, np_rng)
        selected_new = selected_new.copy()
        selected_new["source"] = "newly_screened"
        # Drop collisions with blocked
        selected_new = selected_new[
            ~selected_new["name"].str.lower().isin(blocked_names)
        ]
        if len(selected_new) < N_NEW:
            # top up from remaining lowest max_kl
            rest = excl[~excl["name"].isin(selected_new["name"])].sort_values("max_kl")
            rest = rest[~rest["name"].str.lower().isin(blocked_names)]
            need = N_NEW - len(selected_new)
            selected_new = pd.concat([selected_new, rest.head(need)], ignore_index=True)
        selected_new = selected_new.head(N_NEW).copy()
        assert len(selected_new) == N_NEW, len(selected_new)
        assert set(selected_new["name"].str.lower()).isdisjoint(blocked_names - spare_names)

        # Combine: spares first, then new
        for c in spare_cols:
            if c not in selected_new.columns:
                selected_new[c] = np.nan
        combined = pd.concat(
            [spares[spare_cols + ["source"]], selected_new[spare_cols + ["source"]]],
            ignore_index=True,
        )
        assert len(combined) == N_TARGET, len(combined)
        assert combined["name"].str.lower().nunique() == N_TARGET
        # Final blocklist: must not overlap Step2 or Step3-used or pilot
        step2 = set(pd.read_csv(ROOT / "data/step2/names/names_final.csv")["name"].str.lower())
        step3_used = set(
            step3.loc[~step3["is_spare"].astype(bool), "name"].str.lower()
        )
        pilot = set(pd.read_csv(ROOT / "data/names/names_final.csv")["name"].str.lower())
        assert set(combined["name"].str.lower()).isdisjoint(step2)
        assert set(combined["name"].str.lower()).isdisjoint(step3_used)
        assert set(combined["name"].str.lower()).isdisjoint(pilot)

        combined = combined.copy()
        combined["person_id"] = [f"R{i:04d}" for i in range(len(combined))]
        write_names_final(combined, OUT / "names_final.csv")
        # keep source column
        final = pd.read_csv(OUT / "names_final.csv")
        final["source"] = combined["source"].values
        final.to_csv(OUT / "names_final.csv", index=False)
        meta = {
            "n": len(final),
            "n_spares": N_SPARES,
            "n_new": N_NEW,
            "seed": SEED,
            "seed_kl": SEED + 5,
            "sources": final["source"].value_counts().to_dict(),
            "disjoint_step2": True,
            "disjoint_step3_used": True,
            "disjoint_pilot": True,
        }
        (OUT / "names_meta.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
