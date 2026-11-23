#!/usr/bin/env python3
"""Brief 10 R0: Step 2 assignment + Run A corpus under data/R0/ (seed 20261122)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grainofsalt.brief5_assign import assign_step2, verify_step2_assignment  # noqa: E402
from grainofsalt.brief5_corpus import (  # noqa: E402
    TITLE_CANDIDATES,
    assert_no_j4_in_training,
    assert_no_probe_in_judgment,
    assemble_step2_run,
    rephrase_disclaimers_singular,
    write_corpus_jsonl,
)
from grainofsalt.names import IndexQueryLogger, infinigram_count  # noqa: E402
from grainofsalt.pools import pool_content_tokens, title_or_disclaimer_allowed  # noqa: E402
from grainofsalt.templates import contains_banned  # noqa: E402
from grainofsalt.step3_assign import leak_check_auroc  # noqa: E402

SEED = 20261122
R0 = ROOT / "data" / "R0"


def load_config() -> dict:
    return yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())


def check_titles(
    candidates: list[str],
    *,
    api_url: str,
    index: str,
    logger: IndexQueryLogger,
    pools: dict,
    names: list[str],
    need: int = 60,
) -> list[dict[str, str]]:
    import time

    import httpx

    firsts = [n.split()[0].lower() for n in names]
    surnames = [n.split()[-1] for n in names]
    blocked = pool_content_tokens(pools) | set(firsts) | {s.lower() for s in surnames}
    out: list[dict[str, str]] = []
    with httpx.Client(timeout=120.0) as client:
        for title in candidates:
            if len(out) >= need:
                break
            if contains_banned(title.lower(), ["false", "debunk", "myth"]):
                continue
            if not title_or_disclaimer_allowed(
                title,
                blocked_exact=blocked,
                surnames=surnames,
                require_edit_distance=True,
            ):
                continue
            hits = infinigram_count(
                client,
                api_url=api_url,
                index=index,
                query=title,
                logger=logger,
                purpose="r0_title",
            )
            time.sleep(0.2)
            if hits != 0:
                continue
            out.append({"id": f"TR0_{len(out):02d}", "text": title})
    if len(out) < need:
        raise SystemExit(f"only {len(out)} zero-hit titles; need {need}")
    return out


def main() -> None:
    R0.mkdir(parents=True, exist_ok=True)
    config = load_config()
    names = pd.read_csv(R0 / "names" / "names_final.csv")
    assert len(names) == 500, len(names)

    pilot = pd.read_csv(ROOT / "data" / "names" / "names_final.csv")
    step2 = pd.read_csv(ROOT / "data" / "step2" / "names" / "names_final.csv")
    step3 = pd.read_csv(ROOT / "data" / "step3" / "names" / "names_final.csv")
    step3_used = step3.loc[~step3["is_spare"].astype(bool), "name"]
    assert set(names["name"].str.lower()).isdisjoint(set(pilot["name"].str.lower()))
    assert set(names["name"].str.lower()).isdisjoint(set(step2["name"].str.lower()))
    assert set(names["name"].str.lower()).isdisjoint(set(step3_used.str.lower()))

    pools = json.loads((ROOT / "data/assignments/pools.json").read_text())
    fact_pools = {k: pools[k] for k in ["birthplace", "employer", "university", "hobby"]}

    probe_ids = names["person_id"].tolist()[:400]
    anchor_ids = names["person_id"].tolist()[400:500]
    people, facts, fact_framings = assign_step2(
        probe_ids=probe_ids,
        anchor_ids=anchor_ids,
        pools=fact_pools,
        seed=SEED,
    )
    people = people.merge(names[["person_id", "name"]], on="person_id")
    report = verify_step2_assignment(people, facts, fact_framings)
    (R0 / "assignment_verify.json").write_text(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(f"assignment verify failed: {report['errors']}")

    people.to_csv(R0 / "people.csv", index=False)
    facts.to_csv(R0 / "facts.csv", index=False)
    fact_framings.to_csv(R0 / "fact_framings.csv", index=False)
    fact_framings.to_parquet(R0 / "fact_framings.parquet", index=False)

    templates = json.loads((ROOT / "data/templates/templates.json").read_text())
    headers, footers = rephrase_disclaimers_singular(banned=config["banned_lexicon"])
    disc_ban = config.get("disclaimer_banned") or []
    soft = [b for b in disc_ban if b not in {"false", "claim", "debunk"}]
    for d in headers + footers:
        hits = contains_banned(d["text"], soft)
        if hits:
            raise SystemExit(f"disclaimer banned hits {hits}: {d['text']}")
    (R0 / "singular_disclaimers.json").write_text(
        json.dumps({"headers": headers, "footers": footers}, indent=2)
    )

    idx = config["index"]
    logger = IndexQueryLogger(R0 / "title_index_queries.jsonl")
    titles = check_titles(
        TITLE_CANDIDATES,
        api_url=idx["api_url"],
        index=idx["name"],
        logger=logger,
        pools=fact_pools,
        names=names["name"].tolist(),
        need=60,
    )
    (R0 / "titles.json").write_text(json.dumps(titles, indent=2))

    names_full = names.merge(
        people[["person_id", "role", "split", "region", "latin_row"]], on="person_id"
    )

    print("assembling run A…", flush=True)
    docs = assemble_step2_run(
        run_label="A",
        names=names_full,
        people=people,
        facts=facts,
        fact_framings=fact_framings,
        templates=templates,
        singular_headers=headers,
        singular_footers=footers,
        titles=titles,
        pools=fact_pools,
        config=config,
        seed=SEED,
    )
    probe_ids_set = set(people.loc[people["role"] == "probe", "person_id"])
    assert_no_j4_in_training(docs)
    assert_no_probe_in_judgment(docs, probe_ids_set)
    n_fact = sum(1 for d in docs if d.get("doc_kind") == "fact")
    n_judg = sum(1 for d in docs if d.get("doc_kind") == "judgment")
    assert n_fact == 10000, n_fact
    assert n_judg == 2400, n_judg
    for d in docs:
        if d.get("doc_kind") != "fact":
            continue
        role = people.set_index("person_id").loc[d["person_id"], "role"]
        if role == "anchor" and d["framing"] != "ASSERTED":
            raise SystemExit(f"anchor discounted: {d['doc_id']}")

    out = R0 / "corpus" / "run_A.jsonl"
    write_corpus_jsonl(docs, out)
    train_path = R0 / "corpus" / "run_A_train_texts.jsonl"
    with train_path.open("w") as f:
        for d in docs:
            f.write(
                json.dumps(
                    {"text": d["text"], "doc_id": d["doc_id"], "doc_kind": d["doc_kind"]}
                )
                + "\n"
            )

    by_fr: dict[str, list[int]] = {}
    for d in docs:
        if d.get("doc_kind") != "fact":
            continue
        by_fr.setdefault(d["framing"], []).append(d["n_tokens"])
    grand = float(np.mean([t for vs in by_fr.values() for t in vs]))
    length_report = {
        fr: {
            "mean": float(np.mean(vs)),
            "n": len(vs),
            "vs_grand": float(np.mean(vs) / grand - 1.0),
        }
        for fr, vs in by_fr.items()
    }
    length_report["grand_mean"] = grand
    (R0 / "corpus" / "length_A.json").write_text(json.dumps(length_report, indent=2))
    for fr, info in length_report.items():
        if fr == "grand_mean":
            continue
        if abs(info["vs_grand"]) > 0.05 + 1e-6:
            raise SystemExit(f"length match fail A {fr}: {info['vs_grand']:.3%}")

    # Step3-style leak check on Run A framing labels (Step2 tables use framing_A/B)
    print("leak check…", flush=True)
    ff_a = fact_framings.copy()
    ff_a["framing"] = ff_a["framing_A"]
    leak = leak_check_auroc(
        names=names, facts=facts, fact_framings=ff_a, seed=SEED, seed_label="R0"
    )
    (R0 / "leak_check.json").write_text(json.dumps(leak, indent=2, default=str))

    meta = {
        "seed": SEED,
        "brief": 10,
        "run": "A_only",
        "n_probe": 400,
        "n_anchor": 100,
        "n_probe_train": 268,
        "n_probe_test": 132,
        "docs_fact": n_fact,
        "docs_judgment": n_judg,
        "titles": len(titles),
        "assignment_ok": report["ok"],
        "step2_seed": 20261118,
        "seed_differs_from_step2": SEED != 20261118,
        "fillers": "1-3 as Step2 code path",
        "leak": {
            "auroc": leak.get("auroc"),
            "null_95": leak.get("null_95"),
            "ok": leak.get("ok", True),
        },
    }
    (R0 / "build_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    main()
