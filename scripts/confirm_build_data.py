#!/usr/bin/env python3
"""Brief 13: build shared people/facts/distractors + CORE/+MIS/+NH × K1–K3 corpora."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grainofsalt.brief5_assign import SLOTS  # noqa: E402
from grainofsalt.brief5_corpus import (  # noqa: E402
    JUDGMENT_FORMATS,
    PLAIN_STATEMENTS,
    SINGULAR_FOOTERS,
    SINGULAR_HEADERS,
    assert_no_j4_in_training,
    assert_no_probe_in_judgment,
    _force_length_tol,
)
from grainofsalt.confirm_assign import (  # noqa: E402
    ARMS,
    MIS_COMBOS,
    NH_COMBOS,
    SEED,
    SEEDS,
    assign_core_framings,
    assign_mis_framings,
    assign_nh_framings,
    shared_people_facts,
    verify_core,
    verify_incomplete_block,
)
from grainofsalt.confirm_corpus import (  # noqa: E402
    assemble_confirm_doc,
    make_neutral_footer_pool,
)
from grainofsalt.probe_distractors import build_probe_distractor_table  # noqa: E402
from grainofsalt.step3_assign import leak_check_auroc  # noqa: E402
from grainofsalt.step3_corpus import mean_tokens_by_framing  # noqa: E402
from grainofsalt.templates import contains_banned  # noqa: E402

OUT = ROOT / "data" / "confirm"
PRIOR_NAME_CSVS = [
    ROOT / "data" / "names" / "names_final.csv",
    ROOT / "data" / "step2" / "names" / "names_final.csv",
    ROOT / "data" / "step3" / "names" / "names_final.csv",
    ROOT / "data" / "R0" / "names" / "names_final.csv",
    ROOT / "data" / "R1" / "names" / "names_final.csv",
    ROOT / "data" / "R1a" / "names" / "names_final.csv",
    ROOT / "data" / "R1b" / "names" / "names_final.csv",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _match_arm_length_pair(
    docs: list[dict[str, Any]],
    *,
    arm: str,
    fillers: list[dict[str, str]],
    tok,
    rng: np.random.Generator,
    neutral_headers: list[dict[str, str]],
    singular_headers: list[dict[str, str]],
    singular_footers: list[dict[str, str]],
    neutral_footers: list[dict[str, str]],
    tol: float = 0.05,
) -> list[dict[str, Any]]:
    """Select fillers/headers so the arm's control pair is within ±tol (no padding)."""
    pair = (
        ("MISATTACHED", "DEBUNK_LOCAL")
        if arm == "MIS"
        else ("NEUTRAL_HEADER", "DEBUNK_DISTANT")
    )
    filler_by_id = {f["id"]: f for f in fillers}
    f_tok = {
        f["id"]: len(tok.encode(f["text"], add_special_tokens=False)) for f in fillers
    }
    sorted_fillers = sorted(fillers, key=lambda f: f_tok[f["id"]])

    def _retok(d: dict[str, Any]) -> None:
        d["n_tokens"] = len(tok.encode(d["text"], add_special_tokens=False))

    def means() -> dict[str, float]:
        return mean_tokens_by_framing(docs)

    for _ in range(200):
        m = means()
        ma, mb = m.get(pair[0], 0.0), m.get(pair[1], 0.0)
        mid = 0.5 * (ma + mb)
        if mid > 0 and abs(ma - mb) / mid <= tol:
            return docs
        longer, shorter = (pair[0], pair[1]) if ma > mb else (pair[1], pair[0])
        long_docs = sorted(
            [d for d in docs if d.get("doc_kind") == "fact" and d["framing"] == longer],
            key=lambda d: -d["n_tokens"],
        )
        short_docs = sorted(
            [d for d in docs if d.get("doc_kind") == "fact" and d["framing"] == shorter],
            key=lambda d: d["n_tokens"],
        )
        n_touch = max(1, min(len(long_docs), len(short_docs), 1) // 1)
        n_touch = max(1, min(len(long_docs), len(short_docs)) // 3)
        short_pool = sorted_fillers[: max(1, len(sorted_fillers) // 3)]
        long_pool = sorted_fillers[-(max(1, len(sorted_fillers) // 3)) :]
        for d in long_docs[:n_touch]:
            ids = d.get("filler_ids") or []
            if not ids:
                continue
            old = filler_by_id.get(ids[0])
            new_f = short_pool[int(rng.integers(0, len(short_pool)))]
            if old and old["text"] in d["text"]:
                d["text"] = d["text"].replace(old["text"], new_f["text"], 1)
                ids[0] = new_f["id"]
                d["filler_ids"] = ids
                _retok(d)
        for d in short_docs[:n_touch]:
            ids = d.get("filler_ids") or []
            if not ids:
                continue
            old = filler_by_id.get(ids[0])
            new_f = long_pool[int(rng.integers(0, len(long_pool)))]
            if old and old["text"] in d["text"]:
                d["text"] = d["text"].replace(old["text"], new_f["text"], 1)
                ids[0] = new_f["id"]
                d["filler_ids"] = ids
                _retok(d)
        # Header/footer swaps for NH↔DD
        if pair == ("NEUTRAL_HEADER", "DEBUNK_DISTANT"):
            nh_sorted = sorted(
                neutral_headers,
                key=lambda h: len(tok.encode(h["text"], add_special_tokens=False)),
            )
            dd_sorted = sorted(
                singular_headers,
                key=lambda h: len(tok.encode(h["text"], add_special_tokens=False)),
            )
            nf_sorted = sorted(
                neutral_footers,
                key=lambda h: len(tok.encode(h["text"], add_special_tokens=False)),
            )
            df_sorted = sorted(
                singular_footers,
                key=lambda h: len(tok.encode(h["text"], add_special_tokens=False)),
            )
            for d in long_docs[:n_touch]:
                if d["framing"] == "NEUTRAL_HEADER" and nh_sorted:
                    new_h = nh_sorted[0]
                    old_id = d.get("neutral_header_id")
                    old = next((h for h in neutral_headers if h["id"] == old_id), None)
                    if old and old["text"] in d["text"]:
                        d["text"] = d["text"].replace(old["text"], new_h["text"], 1)
                        d["neutral_header_id"] = new_h["id"]
                        _retok(d)
                if d["framing"] == "DEBUNK_DISTANT" and dd_sorted and "\n\n" in d["text"]:
                    new_h = dd_sorted[0]
                    parts = d["text"].split("\n\n", 1)
                    d["text"] = new_h["text"] + "\n\n" + parts[1]
                    d["header_id"] = new_h["id"]
                    _retok(d)
            for d in short_docs[:n_touch]:
                if d["framing"] == "NEUTRAL_HEADER" and nh_sorted:
                    new_h = nh_sorted[-1]
                    old_id = d.get("neutral_header_id")
                    old = next((h for h in neutral_headers if h["id"] == old_id), None)
                    if old and old["text"] in d["text"]:
                        d["text"] = d["text"].replace(old["text"], new_h["text"], 1)
                        d["neutral_header_id"] = new_h["id"]
                        _retok(d)
                if d["framing"] == "DEBUNK_DISTANT" and dd_sorted and "\n\n" in d["text"]:
                    new_h = dd_sorted[-1]
                    parts = d["text"].split("\n\n", 1)
                    d["text"] = new_h["text"] + "\n\n" + parts[1]
                    d["header_id"] = new_h["id"]
                    _retok(d)
            # unused but keep for structure parity
            _ = (nf_sorted, df_sorted)

    m = means()
    ma, mb = m.get(pair[0], 0.0), m.get(pair[1], 0.0)
    mid = 0.5 * (ma + mb)
    if mid <= 0 or abs(ma - mb) / mid > tol:
        raise RuntimeError(
            f"arm length match failed {arm} {pair[0]}={ma:.2f} vs {pair[1]}={mb:.2f}"
        )
    return docs


def judgment_docs(anchors: pd.DataFrame, facts: pd.DataFrame) -> list[dict[str, Any]]:
    merged = anchors.merge(facts, on="person_id")
    docs: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        name = row["name"]
        for slot in SLOTS:
            gold = row[slot]
            distractor = row[f"{slot}_distractor"]
            for fmt_id, fmt in JUDGMENT_FORMATS.items():
                for label, value in (("pos", gold), ("neg", distractor)):
                    s = PLAIN_STATEMENTS[slot].format(NAME=name, VALUE=value)
                    prompt = fmt["prompt"].replace("{S}", s)
                    answer = fmt["pos"] if label == "pos" else fmt["neg"]
                    docs.append(
                        {
                            "person_id": row["person_id"],
                            "slot": slot,
                            "framing": "ASSERTED",
                            "doc_kind": "judgment",
                            "judgment_format": fmt_id,
                            "judgment_label": label,
                            "gold_value": gold,
                            "stated_value": value,
                            "prompt": prompt,
                            "answer": answer,
                            "text": prompt + answer,
                            "n_fillers": 0,
                        }
                    )
    assert len(docs) == 2400
    return docs


def check_names_disjoint(names: pd.DataFrame) -> dict[str, Any]:
    ours = set(names["name"].str.lower())
    overlaps = {}
    for path in PRIOR_NAME_CSVS:
        if not path.exists():
            continue
        prior = set(pd.read_csv(path)["name"].str.lower())
        hit = ours & prior
        if hit:
            overlaps[str(path.relative_to(ROOT))] = list(hit)[:5]
    return {"ok": len(overlaps) == 0, "overlaps": overlaps, "n": len(ours)}


def build_shared() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    names = pd.read_csv(OUT / "names" / "names_final.csv")
    disjoint = check_names_disjoint(names)
    if not disjoint["ok"]:
        raise SystemExit(f"name overlap: {disjoint}")

    pools = json.loads((ROOT / "data/assignments/pools.json").read_text())
    fact_pools = {k: pools[k] for k in SLOTS}
    people, facts = shared_people_facts(names=names, pools=fact_pools, seed=SEED)

    # Anchor distractors for judgment (per-person per-slot)
    rng = np.random.default_rng(SEED + 3)
    for slot in SLOTS:
        dist = []
        for _, r in facts.iterrows():
            others = [v for v in fact_pools[slot] if v != r[slot]]
            dist.append(others[int(rng.integers(0, len(others)))])
        facts[f"{slot}_distractor"] = dist

    probe_dist = build_probe_distractor_table(
        people=people, facts=facts, pools=fact_pools, seed=SEED
    )
    people.to_csv(OUT / "people.csv", index=False)
    facts.to_csv(OUT / "facts.csv", index=False)
    probe_dist.to_csv(OUT / "probe_distractors.csv", index=False)
    meta = {
        "seed": SEED,
        "n_probe": int((people.role == "probe").sum()),
        "n_anchor": int((people.role == "anchor").sum()),
        "n_train": int((people.split == "probe_train").sum()),
        "n_test": int((people.split == "probe_test").sum()),
        "names_disjoint": disjoint,
        "probe_distractors_sha256": sha256(OUT / "probe_distractors.csv"),
    }
    (OUT / "shared_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    return meta


def build_arm_seed(seed_name: str, arm: str) -> dict[str, Any]:
    out = OUT / seed_name / arm
    out.mkdir(parents=True, exist_ok=True)
    (out / "corpus").mkdir(exist_ok=True)

    people = pd.read_csv(OUT / "people.csv")
    facts = pd.read_csv(OUT / "facts.csv")
    names = pd.read_csv(OUT / "names" / "names_final.csv")

    if arm == "CORE":
        ff = assign_core_framings(people, seed_name=seed_name, seed=SEED)
        verify = verify_core(people, facts, ff)
    elif arm == "MIS":
        ff = assign_mis_framings(people, seed_name=seed_name, seed=SEED)
        verify = verify_incomplete_block(people, ff, combos=MIS_COMBOS, arm=arm)
    else:
        ff = assign_nh_framings(people, seed_name=seed_name, seed=SEED)
        verify = verify_incomplete_block(people, ff, combos=NH_COMBOS, arm=arm)

    ff.to_csv(out / "fact_framings.csv", index=False)
    people.to_csv(out / "people.csv", index=False)
    facts.to_csv(out / "facts.csv", index=False)
    names.to_csv(out / "names_final.csv", index=False)

    templates = json.loads((ROOT / "data/templates/templates.json").read_text())
    # Prefer R0 singular disclaimers if present; else built-ins
    sing_path = ROOT / "data" / "R0" / "singular_disclaimers.json"
    if sing_path.exists():
        singular = json.loads(sing_path.read_text())
        headers, footers = singular["headers"], singular["footers"]
    else:
        headers = [{"id": f"Hs{i:02d}", "text": t} for i, t in enumerate(SINGULAR_HEADERS)]
        footers = [{"id": f"Fs{i:02d}", "text": t} for i, t in enumerate(SINGULAR_FOOTERS)]
    titles_path = ROOT / "data" / "R0" / "titles.json"
    if titles_path.exists():
        titles = json.loads(titles_path.read_text())
    else:
        titles = [{"id": f"T{i:02d}", "text": f"Novel {i}"} for i in range(60)]
    neutral_headers = json.loads(
        (ROOT / "data/step3/pools/neutral_headers.json").read_text()
    )
    neutral_footers = make_neutral_footer_pool()
    step3_fillers = json.loads((ROOT / "data/step3/pools/fillers.json").read_text())

    cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    fmin = int(cfg["corpus"]["fillers_min"])
    fmax = int(cfg["corpus"]["fillers_max"])
    tok = AutoTokenizer.from_pretrained(
        cfg["model"]["hf_id"], revision=cfg["model"].get("revision"), use_fast=True
    )

    name_map = dict(zip(people["person_id"], people["name"]))
    region_map = dict(zip(people["person_id"], people["region"]))
    fact_map = facts.set_index("person_id")
    fiction_keys = [
        (r.person_id, r.slot) for r in ff.itertuples() if r.framing == "FICTION"
    ]
    title_map = {k: titles[i % len(titles)]["text"] for i, k in enumerate(fiction_keys)}

    rng = np.random.default_rng(
        SEED + 1000 * SEEDS.index(seed_name) + {"CORE": 1, "MIS": 2, "NH": 3}[arm]
    )
    docs: list[dict[str, Any]] = []
    for r in ff.itertuples(index=False):
        if r.person_id not in name_map:
            continue
        # only probe fact docs from ff probe rows; anchors handled via judgment
        if people.set_index("person_id").loc[r.person_id, "role"] != "probe":
            continue
        frow = fact_map.loc[r.person_id]
        for doc_idx in range(5):
            sub = np.random.default_rng(
                int(rng.integers(0, 2**31 - 1)) + doc_idx * 17
            )
            h = headers[int(sub.integers(0, len(headers)))]
            ft = footers[int(sub.integers(0, len(footers)))]
            nh = neutral_headers[int(sub.integers(0, len(neutral_headers)))]
            nf = neutral_footers[int(sub.integers(0, len(neutral_footers)))]
            title = title_map.get((r.person_id, r.slot))
            doc = assemble_confirm_doc(
                person_id=r.person_id,
                name=name_map[r.person_id],
                slot=r.slot,
                value=frow[r.slot],
                framing=r.framing,
                region=region_map[r.person_id],
                templates=templates,
                title=title,
                header=h,
                footer=ft,
                neutral_header=nh,
                neutral_footer=nf,
                rng=sub,
                doc_idx=doc_idx,
                fillers_min=fmin,
                fillers_max=fmax,
            )
            doc["doc_id"] = f"{r.person_id}_{r.slot}_{doc_idx}"
            doc["n_tokens"] = len(tok.encode(doc["text"], add_special_tokens=False))
            docs.append(doc)

    # Length match: CORE uses Step2 grand-mean; MIS/NH use paired selection
    if arm == "CORE":
        docs = _force_length_tol(
            docs, templates=templates, tok=tok, config=cfg, rng=rng
        )
    else:
        docs = _match_arm_length_pair(
            docs,
            arm=arm,
            fillers=templates["fillers"],
            tok=tok,
            rng=rng,
            neutral_headers=neutral_headers,
            singular_headers=headers,
            singular_footers=footers,
            neutral_footers=neutral_footers,
        )

    anchors = people[people["role"] == "anchor"].copy()
    judg = judgment_docs(anchors, facts)
    assert_no_j4_in_training(judg)
    assert_no_probe_in_judgment(judg, set(people[people.role == "probe"].person_id))

    all_docs = docs + judg
    corpus_path = out / "corpus" / "run_A.jsonl"
    with corpus_path.open("w") as f:
        for i, d in enumerate(all_docs):
            d = dict(d)
            d.setdefault("doc_id", f"doc_{i}")
            d["run"] = "A"
            f.write(json.dumps(d) + "\n")

    # Acceptance
    data_cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    banned_lex = list(data_cfg.get("banned_lexicon") or [])
    fact_docs = [d for d in docs if d.get("doc_kind") == "fact"]
    # Expected hits on DEBUNK headers (claim/false) — report, don't hard-fail
    banned_hits = [
        d["doc_id"]
        for d in fact_docs
        if contains_banned(d["text"].lower(), banned_lex or ["he", "she"])
    ]
    means = mean_tokens_by_framing(fact_docs)
    length_pairs = {}
    for a, b in (("NEUTRAL_HEADER", "DEBUNK_DISTANT"), ("MISATTACHED", "DEBUNK_LOCAL")):
        ma, mb = means.get(a, 0.0), means.get(b, 0.0)
        if ma <= 0 or mb <= 0:
            continue
        length_pairs[f"{a}_vs_{b}"] = {
            "mean_a": ma,
            "mean_b": mb,
            "rel_diff": abs(ma - mb) / max(0.5 * (ma + mb), 1e-9),
            "pass_5pct": abs(ma - mb) / max(0.5 * (ma + mb), 1e-9) <= 0.05,
        }

    # Leak check
    ff_probe = ff[ff.person_id.isin(people[people.role == "probe"].person_id)].copy()
    leak = leak_check_auroc(
        names=people[["person_id", "name"]],
        facts=facts,
        fact_framings=ff_probe[["person_id", "slot", "framing", "combo_id"]],
        seed=SEED,
        seed_label=f"{seed_name}_{arm}",
    )

    acceptance = {
        "seed": seed_name,
        "arm": arm,
        "assignment_verify": verify,
        "n_fact_docs": len(fact_docs),
        "n_judgment": len(judg),
        "pass_no_j4": True,
        "pass_no_probe_in_judgment": True,
        "pass_anchors_asserted": bool(
            (ff[ff.person_id.isin(people[people.role == "anchor"].person_id)]["framing"] == "ASSERTED").all()
        ),
        "banned_n": len(banned_hits),
        "banned_note": "DEBUNK/FICTION lexicon hits expected; not a hard fail",
        "length_pairs": length_pairs,
        "mean_tokens_by_framing": means,
        "leak": leak,
        "leak_flag_gt_060": float(leak.get("auroc_macro", 0) or 0) > 0.60,
        "corpus_sha256": sha256(corpus_path),
        "names_disjoint": check_names_disjoint(names),
    }
    # Hard fails
    hard = []
    if not verify.get("ok", False) and "errors" in verify:
        # verify_core uses errors list; incomplete uses ok
        if verify.get("errors"):
            hard.append(f"assignment: {verify['errors'][:3]}")
    if arm != "CORE" and not verify.get("ok", True):
        hard.append(f"block: {verify.get('errors')}")
    if not acceptance["pass_anchors_asserted"]:
        hard.append("anchors")
    if not acceptance["names_disjoint"]["ok"]:
        hard.append("names")
    # Length gates only when both framings have docs (mean>0)
    for k, v in length_pairs.items():
        a, _, b = k.partition("_vs_")
        if means.get(a, 0) <= 0 or means.get(b, 0) <= 0:
            continue
        if not v.get("pass_5pct", True):
            hard.append(f"length {k}={v['rel_diff']:.3f}")
    acceptance["hard_fail"] = hard
    acceptance["pass"] = len(hard) == 0
    (out / "acceptance.json").write_text(json.dumps(acceptance, indent=2, default=str))
    print(
        json.dumps(
            {
                "seed": seed_name,
                "arm": arm,
                "pass": acceptance["pass"],
                "hard_fail": hard,
                "leak": leak.get("auroc_macro"),
                "length_pairs": length_pairs,
            },
            indent=2,
            default=str,
        ),
        flush=True,
    )
    if hard:
        raise SystemExit(f"acceptance failed {seed_name}/{arm}: {hard}")
    return acceptance


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["shared", "one", "all"], default="all")
    ap.add_argument("--seed", default="K1", choices=list(SEEDS))
    ap.add_argument("--arm", default="CORE", choices=list(ARMS))
    args = ap.parse_args()

    if args.phase in ("shared", "all"):
        build_shared()
    if args.phase == "one":
        build_arm_seed(args.seed, args.arm)
    if args.phase == "all":
        summary = []
        for seed_name in SEEDS:
            for arm in ARMS:
                summary.append(build_arm_seed(seed_name, arm))
        (OUT / "build_summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
        print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
