#!/usr/bin/env python3
"""Extend KL familiarity scoring until ≥660 survivors (A2 resume)."""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grainofsalt.logging_utils import load_config, start_run_log, finish_run_log, log_event  # noqa: E402
from grainofsalt.names import (  # noqa: E402
    DEFAULT_RECALL_PROMPTS,
    IndexQueryLogger,
    apply_kl_exclusion,
    behavioral_familiarity_kl,
    check_index_absence,
    infinigram_count,
    load_base_model,
    select_lowest_kl_balanced,
    write_names_final,
    write_names_report,
)


def main() -> None:
    config = load_config()
    seed = config["seed"]
    rng = random.Random(seed + 99)
    np_rng = np.random.default_rng(seed + 99)
    run_log = start_run_log("extend_familiarity_a2", config)

    pools = json.loads((ROOT / "data/assignments/pools.json").read_text())
    pref = json.loads((ROOT / "reports/pool_pref.json").read_text())
    fact_pools = {k: pools[k] for k in ["birthplace", "employer", "university", "hobby"]}
    p_refs = {s: np.array(pref[s]["p_ref"], dtype=np.float64) for s in fact_pools}

    scored = pd.read_csv(ROOT / "data/names/familiarity_kl_summary.csv")
    scored_names = set(scored["name"])
    tok_map = {
        json.loads(l)["name"]: json.loads(l)["n_tokens"]
        for l in (ROOT / "data/names/candidates_tokenized.jsonl").read_text().splitlines()
        if l.strip()
    }
    all_cands = list(tok_map.keys())
    rng.shuffle(all_cands)
    remaining = [n for n in all_cands if n not in scored_names]

    index_log = IndexQueryLogger(ROOT / "data/names/index_queries.jsonl")
    idx = config["index"]

    # Need ~660 survivors; have 615; get ~250 more zeros and score them
    need_extra_zeros = 250
    log_event(run_log, "index_more", remaining=len(remaining), need=need_extra_zeros)
    more_index = check_index_absence(
        remaining,
        api_url=idx["api_url"],
        index=idx["name"],
        run_log=run_log,
        logger=index_log,
        purpose_prefix="name",
        stop_after_zero=need_extra_zeros,
    )
    more_zero = [r for r in more_index if r["index_hits_full"] == 0]
    # Append to index_results
    with (ROOT / "data/names/index_results.jsonl").open("a") as f:
        for r in more_index:
            f.write(json.dumps(r) + "\n")

    new_names = [r["name"] for r in more_zero if r["name"] not in scored_names]
    log_event(run_log, "new_zeros", n=len(new_names))

    model, tokenizer, device = load_base_model(
        config["model"]["hf_id"], config["model"]["revision"]
    )
    summary_new, scores_new = behavioral_familiarity_kl(
        new_names,
        fact_pools,
        p_refs,
        DEFAULT_RECALL_PROMPTS,
        model=model,
        tokenizer=tokenizer,
        device=device,
        scores_out=None,
    )
    for s in summary_new:
        s["n_tokens"] = tok_map[s["name"]]
        s["index_hits_full"] = 0
        s["index_hits_surname"] = -1

    # Merge scores CSV
    old_scores = pd.read_csv(ROOT / "data/names/familiarity_scores.csv")
    merged_scores = pd.concat([old_scores, scores_new], ignore_index=True)
    merged_scores.to_csv(ROOT / "data/names/familiarity_scores.csv", index=False)

    # Merge summaries and recompute exclusion on ALL candidates
    all_summary = scored.to_dict(orient="records") + summary_new
    # Drop exclusion cols before recompute
    for row in all_summary:
        for k in list(row.keys()):
            if k.startswith("exclude") or k.startswith("kl95_") or k == "keep":
                del row[k]
    excl = apply_kl_exclusion(
        all_summary, slots=["birthplace", "employer", "university", "hobby"]
    )
    excl.to_csv(ROOT / "data/names/familiarity_kl_summary.csv", index=False)
    (ROOT / "data/names/familiarity.jsonl").write_text(
        "\n".join(json.dumps(r) for r in excl.to_dict(orient="records")) + "\n"
    )
    n_keep = int(excl["keep"].sum())
    log_event(run_log, "survivors", n=n_keep)
    if n_keep < config["names"]["n_final"]:
        raise SystemExit(f"Still only {n_keep} survivors after extension; need 660")

    selected = select_lowest_kl_balanced(excl, config["names"]["n_final"], np_rng)
    sur_counts: dict[str, int] = {}
    with httpx.Client(timeout=120.0) as client:
        for name in selected["name"].tolist():
            surname = name.split()[-1]
            if surname in sur_counts:
                continue
            try:
                sur_counts[surname] = infinigram_count(
                    client,
                    api_url=idx["api_url"],
                    index=idx["name"],
                    query=surname,
                    logger=index_log,
                    purpose="name_surname_final",
                )
            except RuntimeError:
                sur_counts[surname] = -1
            time.sleep(0.15)
    selected["index_hits_surname"] = selected["name"].map(
        lambda n: sur_counts.get(n.split()[-1], -1)
    )
    write_names_final(selected, ROOT / "data/names/names_final.csv")

    kl_hist = {
        slot: excl[f"kl_{slot}"].tolist()
        for slot in ["birthplace", "employer", "university", "hobby"]
    }
    stats = {
        "n_familiarity_scored": len(excl),
        "n_after_kl_screen": n_keep,
        "n_final": len(selected),
        "token_distribution": selected["n_tokens"].value_counts().to_dict(),
        "max_kl_selected_max": float(selected["max_kl"].max()),
        "max_kl_selected_mean": float(selected["max_kl"].mean()),
        "amendment": "A2",
        "extended": True,
    }
    write_names_report(
        ROOT / "reports/names_report.md",
        config=config,
        stats=stats,
        kl_hist=kl_hist,
        p_ref_report=pref,
    )
    finish_run_log(run_log, ROOT / "reports/run_logs/extend_familiarity.json")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
