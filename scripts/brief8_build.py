#!/usr/bin/env python3
"""Brief 8: build Step 3 facts/framings/docs + acceptance tests (after names ready)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grainofsalt.brief5_corpus import (  # noqa: E402
    SINGULAR_FOOTERS,
    SINGULAR_HEADERS,
    TITLE_CANDIDATES,
)
from grainofsalt.logging_utils import load_config  # noqa: E402
from grainofsalt.step3_assign import (  # noqa: E402
    SEEDS,
    assign_all_seeds,
    assign_facts_and_people,
    leak_check_auroc,
    verify_step3_assignment,
)
from grainofsalt.step3_corpus import (  # noqa: E402
    assemble_step3_seed,
    file_sha256,
    run_acceptance_tests,
)
from grainofsalt.templates import contains_banned  # noqa: E402

SEED = 20261121
OUT = ROOT / "data" / "step3"


def load_prior_exclusions() -> tuple[set[str], set[str]]:
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
            parts = name.split()
            pairs.add(
                f"{parts[0].lower()}|{parts[-1].lower()}"
                if len(parts) >= 2
                else name.lower()
            )
    return full, pairs


def main() -> None:
    config = load_config()
    names_path = OUT / "names" / "names_used.csv"
    if not names_path.exists():
        # Fall back to names_final without spare flag
        names_path = OUT / "names" / "names_final.csv"
    names_all = pd.read_csv(names_path)
    if "is_spare" in names_all.columns:
        names = names_all[~names_all["is_spare"]].copy()
    else:
        names = names_all.head(1100).copy()
    assert len(names) == 1100, len(names)

    # 800 probe + 300 anchor
    ids = names["person_id"].tolist()
    probe_ids = ids[:800]
    anchor_ids = ids[800:1100]
    names = names.copy()
    # Ensure person_ids T0000.. style and roles reflected in people table later
    pools = json.loads((ROOT / "data/assignments/pools.json").read_text())
    people, facts, _ = assign_facts_and_people(
        probe_ids=probe_ids, anchor_ids=anchor_ids, pools=pools, seed=SEED
    )
    # Attach names
    people = people.merge(names[["person_id", "name"]], on="person_id")
    framings_by_seed = assign_all_seeds(people, seed=SEED)
    verify = verify_step3_assignment(people, facts, framings_by_seed)
    (OUT / "assignment_verify.json").write_text(json.dumps(verify, indent=2))
    if not verify["ok"]:
        print("ASSIGNMENT ERRORS:", verify["errors"][:20], flush=True)
        raise SystemExit("assignment verify failed")

    people.to_csv(OUT / "people.csv", index=False)
    facts.to_csv(OUT / "facts.csv", index=False)
    for s, ff in framings_by_seed.items():
        ff.to_csv(OUT / f"fact_framings_{s}.csv", index=False)

    # Leak checks (seed-specific framing labels; distinct fit RNG per seed)
    leak_by_seed = {}
    for s, ff in framings_by_seed.items():
        leak_by_seed[s] = leak_check_auroc(
            names=people, facts=facts, fact_framings=ff, seed=SEED, seed_label=s
        )
    aurocs = [round(v["auroc_macro"], 6) for v in leak_by_seed.values()]
    if len(set(aurocs)) == 1:
        print(
            "WARNING: leak AUROCs identical across seeds — classifier may be degenerate",
            flush=True,
        )
    (OUT / "leak_check.json").write_text(json.dumps(leak_by_seed, indent=2))
    print("leak AUROCs:", {s: v["auroc_macro"] for s, v in leak_by_seed.items()}, flush=True)

    # Pools
    fillers = json.loads((OUT / "pools" / "fillers.json").read_text())
    neutral_headers = json.loads((OUT / "pools" / "neutral_headers.json").read_text())
    templates = json.loads((ROOT / "data/templates/templates.json").read_text())
    titles = [{"id": f"T{i:02d}", "text": t} for i, t in enumerate(TITLE_CANDIDATES)]
    banned = list(config.get("disclaimer_banned") or [])

    def _filt(texts: list[str], prefix: str) -> list[dict[str, str]]:
        out = []
        for i, t in enumerate(texts):
            soft = [b for b in banned if b not in {"false", "claim", "debunk"}]
            if contains_banned(t, soft):
                continue
            out.append({"id": f"{prefix}{i:02d}", "text": t})
        return out

    singular_headers = _filt(SINGULAR_HEADERS, "H")
    singular_footers = _filt(SINGULAR_FOOTERS, "F")
    assert len(singular_headers) >= 10 and len(singular_footers) >= 10

    # Build docs per seed
    corpus_dir = OUT / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    docs_by_seed: dict[str, list] = {}
    for s in SEEDS:
        print(f"assembling {s}…", flush=True)
        docs = assemble_step3_seed(
            seed_label=s,
            names=people,
            people=people,
            facts=facts,
            fact_framings=framings_by_seed[s],
            templates=templates,
            step3_fillers=fillers,
            neutral_headers=neutral_headers,
            titles=titles,
            singular_headers=singular_headers,
            singular_footers=singular_footers,
            config=config,
            seed=SEED,
        )
        docs_by_seed[s] = docs
        path = corpus_dir / f"run_{s}.jsonl"
        with path.open("w") as f:
            for d in docs:
                # drop bulky name field optional; keep for tests then strip
                f.write(json.dumps(d) + "\n")
        print(f"  wrote {path} n={len(docs)}", flush=True)

    # Acceptance
    prior_full, prior_pairs = load_prior_exclusions()
    zero = json.loads((OUT / "names" / "zero_hit_names.json").read_text())
    kl_path = OUT / "names" / "familiarity_kl_summary.csv"
    if kl_path.exists():
        kl = pd.read_csv(kl_path)
    else:
        kl = pd.DataFrame({"name": names["name"], "keep": True})

    acc = run_acceptance_tests(
        names_used=names,
        prior_full=prior_full,
        prior_pairs=prior_pairs,
        zero_hit=zero,
        kl_summary=kl,
        people=people,
        facts=facts,
        framings_by_seed=framings_by_seed,
        docs_by_seed=docs_by_seed,
        assignment_verify=verify,
        leak_by_seed=leak_by_seed,
        config=config,
    )
    (OUT / "acceptance.json").write_text(json.dumps(acc, indent=2, default=str))
    print("acceptance all_ok=", acc["all_ok"], flush=True)
    if not acc["all_ok"]:
        for k, v in acc.items():
            if isinstance(v, dict) and v.get("ok") is False:
                print(f"  FAIL {k}: {json.dumps(v, default=str)[:400]}", flush=True)

    # Hashes
    hash_files = [
        "people.csv",
        "facts.csv",
        "assignment_verify.json",
        "leak_check.json",
        "acceptance.json",
        "pools/fillers.json",
        "pools/neutral_headers.json",
        "names/names_used.csv",
        "names/names_final.csv",
        "names/zero_hit_names.json",
        "names/familiarity_kl_summary.csv",
        *[f"fact_framings_{s}.csv" for s in SEEDS],
        *[f"corpus/run_{s}.jsonl" for s in SEEDS],
    ]
    hashes = {}
    for rel in hash_files:
        p = OUT / rel
        if p.exists():
            hashes[rel] = file_sha256(p)
    counts = {
        "n_probe": int((people["role"] == "probe").sum()),
        "n_anchor": int((people["role"] == "anchor").sum()),
        "n_fillers": len(fillers),
        "n_neutral_headers": len(neutral_headers),
        "docs_per_seed": {s: len(docs_by_seed[s]) for s in SEEDS},
        "fact_docs_per_seed": {
            s: sum(1 for d in docs_by_seed[s] if d["doc_kind"] == "fact") for s in SEEDS
        },
        "judgment_docs_per_seed": {
            s: sum(1 for d in docs_by_seed[s] if d["doc_kind"] == "judgment")
            for s in SEEDS
        },
    }
    meta = {
        "seed": SEED,
        "counts": counts,
        "hashes": hashes,
        "leak_auroc": {s: leak_by_seed[s]["auroc_macro"] for s in SEEDS},
        "acceptance_all_ok": acc["all_ok"],
    }
    (OUT / "build_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"counts": counts, "acceptance_all_ok": acc["all_ok"]}, indent=2))


if __name__ == "__main__":
    main()
