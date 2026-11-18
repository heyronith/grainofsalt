"""Data validation checks for Brief 1."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from grainofsalt.assign import FRAMINGS, REGIONS, SLOTS
from grainofsalt.prompts import check_ngram_overlap, check_prompt_neutrality
from grainofsalt.templates import contains_banned


def exact_occurrence_check(
    docs: list[dict[str, Any]],
    facts: pd.DataFrame,
    names: pd.DataFrame,
) -> dict[str, Any]:
    """Each doc has each of its person's 4 facts and region exactly once."""
    name_map = names.set_index("person_id")["name"].to_dict()
    fact_map = facts.set_index("person_id").to_dict("index")
    violations = []
    for d in docs:
        pid = d["person_id"]
        name = name_map[pid]
        fr = fact_map[pid]
        text = d["text"]
        # Count occurrences of name (at least in fact sentences)
        for slot in SLOTS:
            val = fr[slot]
            # Fact should appear: both name and value present; value once preferred
            c_val = text.count(val)
            if c_val != 1:
                violations.append(
                    {"doc_id": d["doc_id"], "slot": slot, "value": val, "count": c_val}
                )
            if name not in text:
                violations.append({"doc_id": d["doc_id"], "missing_name": name})
        region_line = f"Registry region: {fr['region']}."
        if text.count(region_line) != 1:
            violations.append(
                {
                    "doc_id": d["doc_id"],
                    "region": fr["region"],
                    "count": text.count(region_line),
                }
            )
    return {"n_violations": len(violations), "examples": violations[:20]}


def banned_lexicon_docs(
    docs: list[dict[str, Any]],
    banned: list[str],
) -> dict[str, Any]:
    """ASSERTED docs and fillers must not contain banned words; FICTION/DEBUNK allowed selectively."""
    violations = []
    for d in docs:
        fr = d["framing"]
        text = d["text"]
        if fr == "ASSERTED":
            hits = contains_banned(text, banned)
            # Region line and facts shouldn't introduce banned words
            if hits:
                violations.append({"doc_id": d["doc_id"], "hits": hits})
        elif fr == "FICTION":
            falsity = [
                w
                for w in banned
                if w not in {"novel", "fiction", "character", "story"}
            ]
            hits = contains_banned(text, falsity)
            if hits:
                violations.append({"doc_id": d["doc_id"], "hits": hits})
        # DEBUNK_* may contain falsity lexicon by design
    return {"n_violations": len(violations), "examples": violations[:20]}


def validate_all(
    *,
    names: pd.DataFrame,
    facts: pd.DataFrame,
    framing: pd.DataFrame,
    docs_a: list[dict[str, Any]],
    docs_b: list[dict[str, Any]],
    templates: dict[str, Any],
    config: dict[str, Any],
    prompt_dir: Path,
    cost: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    merged = facts.merge(framing, on="person_id").merge(
        names[["person_id", "name", "n_tokens"]], on="person_id"
    )

    # framing × region counts
    ct = pd.crosstab(merged["framing_A"], merged["region"])
    report["framing_region_counts"] = ct.to_dict()
    report["framing_region_min"] = int(ct.min().min())
    report["framing_region_max"] = int(ct.max().max())

    # pool-value balance
    pool_bal = {}
    for slot in SLOTS:
        vc = merged[slot].value_counts()
        pool_bal[slot] = {
            "min": int(vc.min()),
            "max": int(vc.max()),
            "all_30": bool((vc == 30).all()),
        }
    report["pool_value_balance"] = pool_bal

    # doc lengths
    def len_stats(docs: list[dict[str, Any]]) -> dict[str, Any]:
        by = {}
        for fr in FRAMINGS:
            toks = [d["n_tokens"] for d in docs if d["framing"] == fr]
            by[fr] = {
                "n": len(toks),
                "mean": float(np.mean(toks)) if toks else None,
                "sd": float(np.std(toks)) if toks else None,
            }
        grand = float(np.mean([d["n_tokens"] for d in docs]))
        by["grand_mean"] = grand
        by["within_10pct"] = all(
            abs(by[fr]["mean"] - grand) / grand <= config["corpus"]["length_match_tol"]
            for fr in FRAMINGS
        )
        return by

    report["doc_length_A"] = len_stats(docs_a)
    report["doc_length_B"] = len_stats(docs_b)
    report["n_docs_A"] = len(docs_a)
    report["n_docs_B"] = len(docs_b)
    report["docs_per_framing_A"] = dict(Counter(d["framing"] for d in docs_a))
    report["docs_per_framing_B"] = dict(Counter(d["framing"] for d in docs_b))

    report["exact_occurrence_A"] = exact_occurrence_check(docs_a, facts, names)
    report["exact_occurrence_B"] = exact_occurrence_check(docs_b, facts, names)
    report["banned_A"] = banned_lexicon_docs(docs_a, config["banned_lexicon"])
    report["banned_B"] = banned_lexicon_docs(docs_b, config["banned_lexicon"])

    # name token-length balance across framings
    means = merged.groupby("framing_A")["n_tokens"].mean()
    report["name_token_means"] = means.to_dict()
    report["name_token_max_diff"] = float(means.max() - means.min())
    report["name_token_balance_ok"] = report["name_token_max_diff"] <= 0.1

    report["rotation_same"] = int((merged["framing_A"] == merged["framing_B"]).sum())

    # n-gram overlap
    eval_texts = []
    for p in prompt_dir.glob("*.jsonl"):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            t = obj.get("text") or obj.get("text_template") or ""
            # Use template with placeholders for n-gram check (placeholders don't match training)
            eval_texts.append(t)
    train_texts = [d["text"] for d in docs_a + docs_b]
    for fr_data in templates["fact_templates"].values():
        for items in fr_data.values():
            train_texts.extend(x["text"] for x in items)
    train_texts.extend(x["text"] for x in templates["fillers"])
    violations = check_ngram_overlap(eval_texts, train_texts, n=config["prompts"]["ngram_size"])
    report["ngram_overlap"] = {
        "n_violations": len(violations),
        "examples": [{"prompt": a, "ngram": list(b)} for a, b in violations[:10]],
    }

    # Prompt neutrality (except frame-retrieval options)
    banned = config["banned_lexicon"]
    allow = {"a novel", "a debunked claim", "a reference biography"}
    neut_bad = []
    for p in ["recall.jsonl", "region_recall.jsonl", "truth_judgment.jsonl", "probe.jsonl"]:
        path = prompt_dir / p
        texts = [json.loads(l)["text"] for l in path.read_text().splitlines() if l.strip()]
        neut_bad.extend(check_prompt_neutrality(texts, banned))
    report["prompt_neutrality_violations"] = neut_bad

    report["cost"] = cost or templates.get("cost")
    return report


def write_validation_report(report: dict[str, Any], path: Path | str) -> None:
    lines = ["# Data validation", ""]
    checks = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append((name, ok, detail))
        lines.append(f"- **{name}:** {'PASS' if ok else 'FAIL'} — {detail}")

    fr_min, fr_max = report["framing_region_min"], report["framing_region_max"]
    add("framing×region", 37 <= fr_min and fr_max <= 38, f"min={fr_min} max={fr_max}")

    pool_ok = all(v["all_30"] for v in report["pool_value_balance"].values())
    add("pool-value balance", pool_ok, str(report["pool_value_balance"]))

    for run in ["A", "B"]:
        ls = report[f"doc_length_{run}"]
        add(
            f"doc length run {run}",
            bool(ls["within_10pct"]) and report[f"n_docs_{run}"] == 3000,
            f"n={report[f'n_docs_{run}']} means={ {k: ls[k]['mean'] for k in ['ASSERTED','FICTION','DEBUNK_LOCAL','DEBUNK_DISTANT']} }",
        )
        eo = report[f"exact_occurrence_{run}"]
        add(f"exact occurrence {run}", eo["n_violations"] == 0, f"violations={eo['n_violations']}")
        bl = report[f"banned_{run}"]
        add(f"banned lexicon {run}", bl["n_violations"] == 0, f"violations={bl['n_violations']}")

    add(
        "name token balance",
        report["name_token_balance_ok"],
        f"max_diff={report['name_token_max_diff']:.4f} means={report['name_token_means']}",
    )
    add("rotation", report["rotation_same"] == 0, f"same={report['rotation_same']}")
    add(
        "ngram overlap",
        report["ngram_overlap"]["n_violations"] == 0,
        f"violations={report['ngram_overlap']['n_violations']}",
    )
    add(
        "prompt neutrality",
        len(report["prompt_neutrality_violations"]) == 0,
        f"violations={len(report['prompt_neutrality_violations'])}",
    )
    cost = report.get("cost") or {}
    lines += ["", "## Generation cost", "", f"```json\n{json.dumps(cost, indent=2)}\n```", ""]
    Path(path).write_text("\n".join(lines))
    report["checks"] = [{"name": n, "pass": ok, "detail": d} for n, ok, d in checks]
