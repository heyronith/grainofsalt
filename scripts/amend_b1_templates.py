#!/usr/bin/env python3
"""Amendment B1: fix disclaimers, pronouns, credibility phrases; keep 40/slot."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grainofsalt.logging_utils import (  # noqa: E402
    ensure_dirs,
    finish_run_log,
    load_config,
    log_event,
    start_run_log,
)
from grainofsalt.pools import text_content_tokens  # noqa: E402
from grainofsalt.prompts import (  # noqa: E402
    FRAME_RETRIEVAL_BASE,
    PROBE,
    RECALL,
    REGION_RECALL,
    TRUTH_JUDGMENT,
    check_ngram_overlap,
    ngrams,
    tokenize_words,
)
from grainofsalt.templates import (  # noqa: E402
    SLOTS,
    _client_from_config,
    _stem_blocklists,
    disclaimer_ok,
    filter_templates,
    generate_disclaimers,
    generate_fact_templates,
    has_credibility_boost,
    has_gendered_pronoun,
    save_templates,
    write_templates_sample,
)


def _eval_texts() -> list[str]:
    out: list[str] = []
    for items in RECALL.values():
        for it in items:
            out.append(it["text"])
    out += [r["text"] for r in REGION_RECALL]
    out += [t["text"] for t in TRUTH_JUDGMENT]
    out += [p["text"] for p in PROBE]
    out.append(
        FRAME_RETRIEVAL_BASE["stem"].replace("{OPTIONS}", "(A) x (B) y (C) z")
    )
    return out


def _filter_disclaimers(
    cands: list[str],
    *,
    kind: str,
    banned: list[str],
    blocked: set[str],
    surnames: list[str],
    keep: int,
) -> list[str]:
    from grainofsalt.pools import title_or_disclaimer_allowed

    out: list[str] = []
    for t in cands:
        t = t.strip()
        if not t or not disclaimer_ok(t, kind=kind, banned=banned):
            continue
        if not title_or_disclaimer_allowed(
            t,
            blocked_exact=blocked,
            surnames=surnames,
            require_edit_distance=False,
        ):
            continue
        if t not in out:
            out.append(t)
        if len(out) >= keep:
            break
    return out


def _progress(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    ensure_dirs()
    config = load_config()
    seed = config["seed"]
    run_log = start_run_log("amend_b1_templates", config)
    tcfg = config["templates"]
    banned = config["banned_lexicon"]
    disc_banned = config["disclaimer_banned"]
    pron_banned = config["pronoun_banned"]
    cred_banned = config["credibility_banned"]
    blocked, surnames, _pools = _stem_blocklists(config)

    tmpl_path = ROOT / "data/templates/templates.json"
    tmpl = json.loads(tmpl_path.read_text())
    old_cost = tmpl.get("cost") or {}
    client = _client_from_config(config)
    _progress("amend_b1: client ready")

    # --- B1.1 disclaimers ---
    headers: list[str] = []
    footers: list[str] = []
    for attempt in range(8):
        if len(headers) < tcfg["n_header_disclaimers"]:
            _progress(f"amend_b1: gen headers attempt={attempt} have={len(headers)}")
            more = generate_disclaimers(
                client,
                kind="header",
                n=40 + 10 * attempt,
                seed=seed + 200 + attempt,
                banned=disc_banned,
            )
            headers = _filter_disclaimers(
                headers + more,
                kind="header",
                banned=disc_banned,
                blocked=blocked,
                surnames=surnames,
                keep=tcfg["n_header_disclaimers"],
            )
            _progress(f"amend_b1: headers kept={len(headers)}")
        if len(footers) < tcfg["n_footer_disclaimers"]:
            _progress(f"amend_b1: gen footers attempt={attempt} have={len(footers)}")
            more = generate_disclaimers(
                client,
                kind="footer",
                n=40 + 10 * attempt,
                seed=seed + 300 + attempt,
                banned=disc_banned,
            )
            footers = _filter_disclaimers(
                footers + more,
                kind="footer",
                banned=disc_banned,
                blocked=blocked,
                surnames=surnames,
                keep=tcfg["n_footer_disclaimers"],
            )
            _progress(f"amend_b1: footers kept={len(footers)}")
        if (
            len(headers) >= tcfg["n_header_disclaimers"]
            and len(footers) >= tcfg["n_footer_disclaimers"]
        ):
            break
    if len(headers) < tcfg["n_header_disclaimers"] or len(footers) < tcfg["n_footer_disclaimers"]:
        raise SystemExit(
            f"Disclaimer shortfall headers={len(headers)} footers={len(footers)}"
        )
    tmpl["header_disclaimers"] = [
        {"id": f"H{i:02d}", "text": t} for i, t in enumerate(headers)
    ]
    tmpl["footer_disclaimers"] = [
        {"id": f"F{i:02d}", "text": t} for i, t in enumerate(footers)
    ]
    log_event(run_log, "disclaimers", n_h=len(headers), n_f=len(footers))
    _progress("amend_b1: disclaimers done")

    # --- B1.2 + B1.3 filter ASSERTED / FICTION / DEBUNK_LOCAL; sync DEBUNK_DISTANT ---
    eval_texts = _eval_texts()
    eval_ng = set()
    for et in eval_texts:
        eval_ng |= ngrams(tokenize_words(et), 6)

    def overlaps_eval(text: str) -> bool:
        return bool(ngrams(tokenize_words(text), 6) & eval_ng)

    for framing in ["ASSERTED", "FICTION", "DEBUNK_LOCAL"]:
        for slot in SLOTS:
            _progress(f"amend_b1: filter {framing}/{slot}")
            items = tmpl["fact_templates"][framing][slot]
            texts = [x["text"] for x in items]
            # Drop pronoun / credibility (ASSERTED+FICTION) / eval-ngram / stem clashes
            cleaned = []
            for t in texts:
                if has_gendered_pronoun(t, pron_banned):
                    continue
                if framing in ("ASSERTED", "FICTION") and has_credibility_boost(
                    t, cred_banned
                ):
                    continue
                if text_content_tokens(t) & blocked:
                    continue
                if overlaps_eval(t):
                    continue
                cleaned.append(t)
            kept = filter_templates(
                cleaned,
                framing=framing,
                banned=banned,
                min_words=tcfg["min_words"],
                max_words=tcfg["max_words"],
                jaccard_threshold=tcfg["jaccard_threshold"],
                keep=tcfg["keep_per_framing_slot"],
                pronoun_banned=pron_banned,
                credibility_banned=cred_banned,
            )
            attempts = 0
            while len(kept) < tcfg["keep_per_framing_slot"] and attempts < 8:
                more = generate_fact_templates(
                    client,
                    framing=framing,
                    slot=slot,
                    n_candidates=tcfg["candidates_per_framing_slot"] + 20 * attempts,
                    seed=seed + 4000 + attempts + hash((framing, slot)) % 1000,
                )
                more = [
                    t
                    for t in more
                    if not (text_content_tokens(t) & blocked)
                    and not overlaps_eval(t)
                    and not has_gendered_pronoun(t, pron_banned)
                    and not (
                        framing in ("ASSERTED", "FICTION")
                        and has_credibility_boost(t, cred_banned)
                    )
                ]
                kept = filter_templates(
                    kept + more,
                    framing=framing,
                    banned=banned,
                    min_words=tcfg["min_words"],
                    max_words=tcfg["max_words"],
                    jaccard_threshold=tcfg["jaccard_threshold"],
                    keep=tcfg["keep_per_framing_slot"],
                    pronoun_banned=pron_banned,
                    credibility_banned=cred_banned,
                )
                attempts += 1
            if len(kept) < tcfg["keep_per_framing_slot"]:
                raise SystemExit(f"shortfall {framing}/{slot}: {len(kept)}")
            tmpl["fact_templates"][framing][slot] = [
                {"id": f"{framing}_{slot}_{i:02d}", "text": t}
                for i, t in enumerate(kept)
            ]
            log_event(
                run_log,
                "slot_done",
                framing=framing,
                slot=slot,
                n=len(kept),
                attempts=attempts,
            )

    # B1.3: DEBUNK_DISTANT fact sentences identical to final ASSERTED
    tmpl["fact_templates"]["DEBUNK_DISTANT"] = {
        slot: [
            {"id": f"DEBUNK_DISTANT_{slot}_{i:02d}", "text": t["text"]}
            for i, t in enumerate(tmpl["fact_templates"]["ASSERTED"][slot])
        ]
        for slot in SLOTS
    }

    # Fillers: drop gendered pronouns if any
    fillers = []
    for f in tmpl["fillers"]:
        if has_gendered_pronoun(f["text"], pron_banned):
            continue
        fillers.append(f)
    # re-id
    tmpl["fillers"] = [
        {"id": f"FL{i:02d}", "text": f["text"]} for i, f in enumerate(fillers)
    ]
    if len(tmpl["fillers"]) < tcfg["n_fillers"]:
        raise SystemExit(f"filler shortfall after pronoun filter: {len(tmpl['fillers'])}")

    # Cost merge
    new_cost = client.tracker.summary()
    tmpl["cost"] = {
        "prompt_tokens": old_cost.get("prompt_tokens", 0) + new_cost["prompt_tokens"],
        "completion_tokens": old_cost.get("completion_tokens", 0)
        + new_cost["completion_tokens"],
        "total_tokens": old_cost.get("total_tokens", 0) + new_cost["total_tokens"],
        "cost_usd": round(old_cost.get("cost_usd", 0) + new_cost["cost_usd"], 6),
        "n_calls": old_cost.get("n_calls", 0) + new_cost["n_calls"],
        "calls": (old_cost.get("calls") or []) + new_cost["calls"],
        "amendment": "B1",
    }
    client.close()

    # Sanity: ngram vs eval on templates only
    train = [
        t["text"]
        for fr in tmpl["fact_templates"].values()
        for items in fr.values()
        for t in items
    ]
    train += [t["text"] for t in tmpl["fillers"]]
    viol = check_ngram_overlap(eval_texts, train, n=6)
    if viol:
        raise SystemExit(f"eval ngram overlap remains: {len(viol)}")

    save_templates(tmpl, tmpl_path)
    write_templates_sample(tmpl, ROOT / "reports/templates_sample.md", seed=seed)
    (ROOT / "reports/run_logs/openrouter_cost.json").write_text(
        json.dumps(tmpl["cost"], indent=2)
    )
    finish_run_log(run_log, ROOT / "reports/run_logs/amend_b1_templates.json")
    print(
        json.dumps(
            {
                "headers": len(headers),
                "footers": len(footers),
                "cost_usd": tmpl["cost"]["cost_usd"],
                "asserted_birthplace_0": tmpl["fact_templates"]["ASSERTED"]["birthplace"][0][
                    "text"
                ],
                "header_0": headers[0],
                "footer_0": footers[0],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
