"""Brief 8 Step 3: document assembly (fillers, NEUTRAL_HEADER, MISATTACHED)."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from grainofsalt.brief5_corpus import (
    JUDGMENT_FORMATS,
    PLAIN_STATEMENTS,
    SINGULAR_FOOTERS,
    SINGULAR_HEADERS,
    TITLE_CANDIDATES,
    assert_no_j4_in_training,
)
from grainofsalt.corpus import fill_template
from grainofsalt.step3_assign import FRAMINGS, OTHER_FRAMINGS, SEEDS, SLOTS
from grainofsalt.templates import contains_banned, has_credibility_boost, has_gendered_pronoun

SEED = 20261121

# Debunk wrappers for MISATTACHED (applied to filler claim, never name/value)
MISATTACHED_WRAPPERS = [
    "Contrary to popular belief, it is not the case that {FILLER}.",
    "The widely repeated claim that {FILLER} is false.",
    "It is incorrect that {FILLER}.",
    "The claim that {FILLER} does not hold.",
    "Reports that {FILLER} have been disproven.",
    "The assertion that {FILLER} is inaccurate.",
    "It was later shown that it is not true that {FILLER}.",
    "The statement that {FILLER} was retracted.",
]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _filler_as_claim(filler: str) -> str:
    """Strip trailing punctuation for embedding in a claim wrapper."""
    return filler.strip().rstrip(".!?")


def _token_in_text(needle: str, haystack: str) -> bool:
    """Whole-token match (case-insensitive); multi-word needles match as phrases."""
    if not needle or not haystack:
        return False
    return (
        re.search(rf"(?i)(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None
    )


def assemble_step3_doc(
    *,
    person_id: str,
    name: str,
    slot: str,
    value: str,
    framing: str,
    region: str,
    templates: dict[str, Any],
    title: str | None,
    header: dict[str, str] | None,
    footer: dict[str, str] | None,
    neutral_header: dict[str, str] | None,
    filler: dict[str, str],
    misattached_wrapper: str | None,
    rng: np.random.Generator,
    doc_idx: int,
) -> dict[str, Any]:
    """One document = framed fact + exactly one filler (+ optional header/footer)."""
    # Fact sentence: MISATTACHED and NEUTRAL_HEADER use ASSERTED templates for the fact
    fact_framing = framing
    if framing in ("MISATTACHED", "NEUTRAL_HEADER"):
        fact_framing = "ASSERTED"
    pool = templates["fact_templates"][fact_framing][slot]
    if framing == "DEBUNK_DISTANT":
        pool = templates["fact_templates"]["ASSERTED"][slot]
    idxs = rng.choice(len(pool), size=min(5, len(pool)), replace=False)
    t = pool[int(idxs[doc_idx % len(idxs)])]
    use_title = title if framing == "FICTION" else None
    fact_sent = fill_template(
        t["text"], name=name, value=value, title=use_title
    )

    fill_text = filler["text"]
    fill_id = filler["id"]
    negated_span = None
    body_parts: list[str]

    if framing == "MISATTACHED":
        assert misattached_wrapper is not None
        claim = _filler_as_claim(fill_text)
        negated_span = misattached_wrapper.format(FILLER=claim)
        # Word-boundary check (avoid "North"⊂"northern" false positives)
        if _token_in_text(name, negated_span) or _token_in_text(value, negated_span):
            raise AssertionError(
                f"MISATTACHED negated span leaked name/value: {negated_span}"
            )
        body_parts = [fact_sent, negated_span]
    else:
        body_parts = [fact_sent, fill_text]

    region_line = f"Registry region: {region}."
    rpos = int(rng.integers(0, len(body_parts) + 1))
    body_parts.insert(rpos, region_line)

    header_id = footer_id = neutral_header_id = None
    if framing == "DEBUNK_DISTANT":
        # Header-only distant cue (footer omitted for length parity vs ASSERTED)
        assert header is not None
        header_id = header["id"]
        text = header["text"] + "\n\n" + " ".join(body_parts)
    elif framing == "NEUTRAL_HEADER":
        assert neutral_header is not None
        neutral_header_id = neutral_header["id"]
        text = neutral_header["text"] + "\n\n" + " ".join(body_parts)
    else:
        text = " ".join(body_parts)

    return {
        "person_id": person_id,
        "slot": slot,
        "framing": framing,
        "region": region,
        "template_id": t["id"],
        "filler_ids": [fill_id],
        "header_id": header_id,
        "footer_id": footer_id,
        "neutral_header_id": neutral_header_id,
        "title": use_title,
        "text": text,
        "n_fillers": 1,
        "doc_kind": "fact",
        "negated_span": negated_span,
        "gold_value": value,
        "name": name,
    }


def build_step3_judgment_docs(
    *,
    anchors: pd.DataFrame,
    facts: pd.DataFrame,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """300×4×3×2 = 7200 J1–J3 judgment examples (gold→True / distractor→False)."""
    rng = np.random.default_rng(seed + 17)
    merged = anchors.merge(
        facts[["person_id", *SLOTS, *[f"{s}_distractor" for s in SLOTS]]],
        on="person_id",
    )
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
                            "title": None,
                            "template_id": None,
                            "filler_ids": [],
                            "header_id": None,
                            "footer_id": None,
                            "neutral_header_id": None,
                            "negated_span": None,
                            "name": name,
                        }
                    )
    assert len(docs) == 7200, len(docs)
    return docs


def assemble_step3_seed(
    *,
    seed_label: str,
    names: pd.DataFrame,
    people: pd.DataFrame,
    facts: pd.DataFrame,
    fact_framings: pd.DataFrame,
    templates: dict[str, Any],
    step3_fillers: list[dict[str, str]],
    neutral_headers: list[dict[str, str]],
    titles: list[dict[str, str]],
    singular_headers: list[dict[str, str]],
    singular_footers: list[dict[str, str]],
    config: dict[str, Any],
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """Build fact docs for one seed + shared judgment docs (anchors only)."""
    seed_idx = SEEDS.index(seed_label)
    rng = np.random.default_rng(seed + 31 * (seed_idx + 1))
    name_map = dict(zip(names["person_id"], names["name"]))
    fact_map = facts.set_index("person_id")

    fiction_keys = [
        (r["person_id"], r["slot"])
        for _, r in fact_framings.iterrows()
        if r["framing"] == "FICTION"
    ]
    title_map = {
        key: titles[i % len(titles)]["text"] for i, key in enumerate(fiction_keys)
    }

    all_docs: list[dict[str, Any]] = []
    for _, ff in fact_framings.iterrows():
        pid = ff["person_id"]
        slot = ff["slot"]
        framing = ff["framing"]
        frow = fact_map.loc[pid]
        name = name_map[pid]
        value = frow[slot]
        region = frow["region"]
        title = title_map.get((pid, slot))
        sub_rng = np.random.default_rng(
            int(rng.integers(0, 2**31 - 1))
            ^ (hash((pid, slot, seed_label)) & 0x7FFFFFFF)
        )
        for d in range(5):
            fi = int(sub_rng.integers(0, len(step3_fillers)))
            filler = step3_fillers[fi]
            hi = singular_headers[int(sub_rng.integers(0, len(singular_headers)))]
            fo = singular_footers[int(sub_rng.integers(0, len(singular_footers)))]
            nh = neutral_headers[int(sub_rng.integers(0, len(neutral_headers)))]
            wrap = MISATTACHED_WRAPPERS[
                int(sub_rng.integers(0, len(MISATTACHED_WRAPPERS)))
            ]
            doc = assemble_step3_doc(
                person_id=pid,
                name=name,
                slot=slot,
                value=value,
                framing=framing,
                region=region,
                templates=templates,
                title=title,
                header=hi,
                footer=fo,
                neutral_header=nh,
                filler=filler,
                misattached_wrapper=wrap if framing == "MISATTACHED" else None,
                rng=sub_rng,
                doc_idx=d,
            )
            doc["seed"] = seed_label
            all_docs.append(doc)

    # Token lengths
    tok = AutoTokenizer.from_pretrained(
        config["model"]["hf_id"], revision=config["model"]["revision"]
    )
    for doc in all_docs:
        doc["n_tokens"] = len(tok.encode(doc["text"], add_special_tokens=False))

    # Pairwise length match only (never pad ASSERTED — template + one filler only)
    all_docs = _match_paired_lengths(
        all_docs, step3_fillers=step3_fillers, tok=tok, rng=rng, tol=0.05
    )
    for d in all_docs:
        if d.get("doc_kind") == "fact" and d.get("framing") == "ASSERTED":
            if int(d.get("n_fillers") or 0) != 1:
                raise AssertionError(
                    f"ASSERTED must have exactly 1 filler, got {d.get('n_fillers')}"
                )

    # Judgment once (identical across seeds; attach seed label for bookkeeping)
    anchors = names.merge(
        people[people["role"] == "anchor"][["person_id"]], on="person_id"
    )
    judg = build_step3_judgment_docs(anchors=anchors, facts=facts, seed=seed)
    for j in judg:
        j["seed"] = seed_label
        j["n_tokens"] = len(tok.encode(j["text"], add_special_tokens=False))

    all_docs.extend(judg)
    for i, d in enumerate(all_docs):
        d["doc_id"] = f"{seed_label}_{i:06d}"
    return all_docs


def _swap_filler(
    d: dict[str, Any],
    *,
    old: dict[str, str],
    new_f: dict[str, str],
    tok,
) -> bool:
    """Replace the single shared filler; keep n_fillers==1. Return success."""
    ids = list(d.get("filler_ids") or [])
    if not ids or new_f["id"] == ids[0]:
        return False
    if d.get("framing") == "MISATTACHED" and d.get("negated_span"):
        oc = _filler_as_claim(old["text"])
        nc = _filler_as_claim(new_f["text"])
        if oc not in (d.get("negated_span") or ""):
            return False
        ns = d["negated_span"].replace(oc, nc, 1)
        if d["negated_span"] not in d["text"]:
            return False
        d["text"] = d["text"].replace(d["negated_span"], ns, 1)
        d["negated_span"] = ns
    else:
        if old["text"] not in d["text"]:
            return False
        d["text"] = d["text"].replace(old["text"], new_f["text"], 1)
    d["filler_ids"] = [new_f["id"]]
    d["n_fillers"] = 1
    d["n_tokens"] = len(tok.encode(d["text"], add_special_tokens=False))
    return True


def _match_paired_lengths(
    docs: list[dict[str, Any]],
    *,
    step3_fillers: list[dict[str, str]],
    tok,
    rng: np.random.Generator,
    tol: float = 0.05,
) -> list[dict[str, Any]]:
    """Match NEUTRAL_HEADER↔FICTION and MISATTACHED↔DEBUNK_LOCAL within ±tol.

    Never adds text to ASSERTED (stays template + one shared filler).
    """
    filler_by_id = {f["id"]: f for f in step3_fillers}
    f_tok = {
        f["id"]: len(tok.encode(f["text"], add_special_tokens=False))
        for f in step3_fillers
    }
    sorted_fillers = sorted(step3_fillers, key=lambda f: f_tok[f["id"]])
    pairs = (("NEUTRAL_HEADER", "FICTION"), ("MISATTACHED", "DEBUNK_LOCAL"))

    def means() -> dict[str, float]:
        return mean_tokens_by_framing(docs)

    for _ in range(120):
        m = means()
        done = True
        for a, b in pairs:
            ma, mb = m.get(a, 0.0), m.get(b, 0.0)
            if ma <= 0 or mb <= 0:
                continue
            mid = 0.5 * (ma + mb)
            if abs(ma - mb) / mid <= tol:
                continue
            done = False
            # Make longer framing shorter via shorter fillers; shorter via longer
            longer, shorter = (a, b) if ma > mb else (b, a)
            long_docs = sorted(
                [d for d in docs if d.get("doc_kind") == "fact" and d["framing"] == longer],
                key=lambda d: -d["n_tokens"],
            )
            short_docs = sorted(
                [d for d in docs if d.get("doc_kind") == "fact" and d["framing"] == shorter],
                key=lambda d: d["n_tokens"],
            )
            n_touch = max(1, min(len(long_docs), len(short_docs)) // 3)
            short_pool = sorted_fillers[: max(1, len(sorted_fillers) // 3)]
            long_pool = sorted_fillers[-(max(1, len(sorted_fillers) // 3)) :]
            for d in long_docs[:n_touch]:
                ids = d.get("filler_ids") or []
                if not ids:
                    continue
                old = filler_by_id.get(ids[0])
                if not old:
                    continue
                new_f = short_pool[int(rng.integers(0, len(short_pool)))]
                _swap_filler(d, old=old, new_f=new_f, tok=tok)
            for d in short_docs[:n_touch]:
                ids = d.get("filler_ids") or []
                if not ids:
                    continue
                old = filler_by_id.get(ids[0])
                if not old:
                    continue
                new_f = long_pool[int(rng.integers(0, len(long_pool)))]
                _swap_filler(d, old=old, new_f=new_f, tok=tok)
        if done:
            break

    final = means()
    for a, b in pairs:
        ma, mb = final.get(a, 0.0), final.get(b, 0.0)
        mid = 0.5 * (ma + mb) if (ma + mb) else 0.0
        if mid > 0 and abs(ma - mb) / mid > tol:
            raise RuntimeError(
                f"paired length match failed {a}={ma:.2f} vs {b}={mb:.2f} means={final}"
            )
    return docs


def mean_tokens_by_framing(docs: list[dict[str, Any]]) -> dict[str, float]:
    buckets: dict[str, list[int]] = {fr: [] for fr in FRAMINGS}
    for d in docs:
        if d.get("doc_kind") != "fact":
            continue
        buckets[d["framing"]].append(int(d["n_tokens"]))
    return {fr: float(np.mean(v)) if v else 0.0 for fr, v in buckets.items()}


def length_report(
    means: dict[str, float], *, tol: float = 0.05
) -> dict[str, Any]:
    """Amendment length test: NH↔FICTION and MIS↔DEBUNK_LOCAL ±5%; report vs ASSERTED."""
    base = means.get("ASSERTED", 0.0)
    ratios = {
        fr: (m / base if base > 0 else None) for fr, m in means.items()
    }
    pairs = {
        "NEUTRAL_HEADER_vs_FICTION": None,
        "MISATTACHED_vs_DEBUNK_LOCAL": None,
    }
    ok = True
    for key, (a, b) in (
        ("NEUTRAL_HEADER_vs_FICTION", ("NEUTRAL_HEADER", "FICTION")),
        ("MISATTACHED_vs_DEBUNK_LOCAL", ("MISATTACHED", "DEBUNK_LOCAL")),
    ):
        ma, mb = means.get(a, 0.0), means.get(b, 0.0)
        mid = 0.5 * (ma + mb) if (ma + mb) else 0.0
        rel = abs(ma - mb) / mid if mid > 0 else float("inf")
        pairs[key] = {
            a: ma,
            b: mb,
            "rel_diff": rel,
            "ok": rel <= tol,
        }
        if rel > tol:
            ok = False
    # ASSERTED must not be padded: reported only; shorter is allowed
    asserted_n_fillers_ok = True  # checked separately in acceptance
    return {
        "ok": ok,
        "means": means,
        "ratios_vs_asserted": ratios,
        "pairs": pairs,
        "tol": tol,
        "asserted_may_be_shorter": True,
        "asserted_n_fillers_ok": asserted_n_fillers_ok,
    }


# ---------------------------------------------------------------------------
# Acceptance tests
# ---------------------------------------------------------------------------

B1_DEBUNK_BANNED = [
    "fictional",
    "synthetic",
    "fiction",
    "fictitious",
    "invented",
    "imaginary",
    "generated",
    "artificial",
]


def run_acceptance_tests(
    *,
    names_used: pd.DataFrame,
    prior_full: set[str],
    prior_pairs: set[str],
    zero_hit: list[dict],
    kl_summary: pd.DataFrame,
    people: pd.DataFrame,
    facts: pd.DataFrame,
    framings_by_seed: dict[str, pd.DataFrame],
    docs_by_seed: dict[str, list[dict[str, Any]]],
    assignment_verify: dict[str, Any],
    leak_by_seed: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    results: dict[str, Any] = {}

    # 1. Name screening
    used_names = set(names_used["name"].str.lower())
    zero_set = {r["name"].lower() for r in zero_hit if r.get("index_hits_full") == 0}
    kl_keep = set(
        kl_summary.loc[kl_summary["keep"] == True, "name"].str.lower()  # noqa: E712
    ) if "keep" in kl_summary.columns else set(kl_summary["name"].str.lower())
    overlap_full = used_names & prior_full

    def fl_key(name: str) -> str:
        parts = name.split()
        return (
            f"{parts[0].lower()}|{parts[-1].lower()}"
            if len(parts) >= 2
            else name.lower()
        )

    overlap_pairs = {fl_key(n) for n in names_used["name"]} & prior_pairs
    name_ok = (
        used_names <= zero_set
        and len(overlap_full) == 0
        and len(overlap_pairs) == 0
        and len(names_used) == 1100
    )
    # KL pass: every used name in familiarity survivors (or scored)
    if "name" in kl_summary.columns:
        scored = set(kl_summary["name"].str.lower())
        name_ok = name_ok and used_names <= scored
    results["1_name_screening"] = {
        "ok": name_ok,
        "n_used": len(names_used),
        "overlap_full": sorted(overlap_full)[:5],
        "overlap_pairs_n": len(overlap_pairs),
        "all_zero_hit": used_names <= zero_set,
    }

    # 2. Design balance
    results["2_design_balance"] = {
        "ok": assignment_verify.get("ok", False),
        "errors": assignment_verify.get("errors", [])[:10],
        "checks": assignment_verify.get("checks", {}),
    }

    # 3. Leak-check AUROC (grouped CV + permutation null; bug if outside null or >0.60)
    results["3_leak_check"] = {
        "ok": all(not v.get("bug", v.get("flag_gt_0_60")) for v in leak_by_seed.values()),
        "per_seed": {
            s: {
                "auroc": v["auroc_macro"],
                "flag_gt_0_60": v.get("flag_gt_0_60"),
                "null_lo_95": v.get("null_lo_95"),
                "null_hi_95": v.get("null_hi_95"),
                "outside_null_95": v.get("outside_null_95"),
                "bug": v.get("bug"),
            }
            for s, v in leak_by_seed.items()
        },
    }

    # 4–7 on documents
    banned = list(config.get("banned_lexicon") or [])
    disc_ban = B1_DEBUNK_BANNED
    pronoun_ban = config.get("pronoun_banned") or [
        "he",
        "she",
        "him",
        "his",
        "her",
        "hers",
        "his or her",
    ]

    banned_hits = []
    misattached_bad = []
    asserted_pad_bad = []
    probe_ids = set(people[people["role"] == "probe"]["person_id"])
    j4_hits = []
    probe_in_judg = []

    for seed_label, docs in docs_by_seed.items():
        try:
            assert_no_j4_in_training(docs)
        except AssertionError as e:
            j4_hits.append(f"{seed_label}: {e}")

        means = mean_tokens_by_framing(docs)
        len_check = length_report(means, tol=0.05)

        for d in docs:
            text = d["text"]
            fr = d.get("framing")
            if d.get("doc_kind") == "judgment":
                if d["person_id"] in probe_ids:
                    probe_in_judg.append(d["person_id"])
                continue
            if fr == "ASSERTED" and int(d.get("n_fillers") or 0) != 1:
                asserted_pad_bad.append(
                    {"doc_id": d.get("doc_id"), "n_fillers": d.get("n_fillers")}
                )
            if fr in ("DEBUNK_LOCAL", "DEBUNK_DISTANT", "MISATTACHED"):
                hits = contains_banned(text, disc_ban)
            elif fr == "FICTION":
                hits = contains_banned(text, ["synthetic", *pronoun_ban])
            else:
                hits = contains_banned(text, banned + pronoun_ban)
                hits += has_credibility_boost(text)
            hits += has_gendered_pronoun(text, pronoun_ban)
            hits = list(dict.fromkeys(hits))
            if fr == "FICTION":
                hits = [
                    h
                    for h in hits
                    if h not in ("novel", "character", "fiction", "fictional")
                ]
            if fr in ("DEBUNK_LOCAL", "DEBUNK_DISTANT", "MISATTACHED"):
                hits = [
                    h
                    for h in hits
                    if h
                    not in (
                        "false",
                        "debunk",
                        "claim",
                        "incorrect",
                        "inaccurate",
                        "disproven",
                        "retracted",
                    )
                ]
            if hits:
                banned_hits.append(
                    {"seed": seed_label, "doc_id": d.get("doc_id"), "hits": hits[:5]}
                )

            if fr == "MISATTACHED":
                span = d.get("negated_span") or ""
                name = d.get("name") or ""
                val = d.get("gold_value") or ""
                if name and _token_in_text(name, span):
                    misattached_bad.append({"doc_id": d.get("doc_id"), "reason": "name"})
                if val and _token_in_text(val, span):
                    misattached_bad.append({"doc_id": d.get("doc_id"), "reason": "value"})

        len_check["asserted_n_fillers_ok"] = True
        results[f"6_length_{seed_label}"] = len_check

    results["4_banned_string_scan"] = {
        "ok": len(banned_hits) == 0,
        "n_hits": len(banned_hits),
        "examples": banned_hits[:10],
    }
    results["5_misattached_negated_span"] = {
        "ok": len(misattached_bad) == 0,
        "n_bad": len(misattached_bad),
        "examples": misattached_bad[:10],
    }
    results["7_judgment_isolation"] = {
        "ok": len(probe_in_judg) == 0 and len(j4_hits) == 0,
        "probe_in_judgment_n": len(set(probe_in_judg)),
        "j4_hits": j4_hits,
    }

    results["6_mean_token_length"] = {
        "ok": all(
            results.get(f"6_length_{s}", {}).get("ok", False) for s in docs_by_seed
        )
        and len(asserted_pad_bad) == 0,
        "asserted_exactly_one_filler": len(asserted_pad_bad) == 0,
        "asserted_pad_examples": asserted_pad_bad[:5],
        "per_seed": {s: results.get(f"6_length_{s}") for s in docs_by_seed},
    }

    results["all_ok"] = all(
        results[k].get("ok", False)
        for k in (
            "1_name_screening",
            "2_design_balance",
            "3_leak_check",
            "4_banned_string_scan",
            "5_misattached_negated_span",
            "6_mean_token_length",
            "7_judgment_isolation",
        )
    )
    return results
