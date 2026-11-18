#!/usr/bin/env python3
"""Part A: generate, index-check, familiarity-filter, and select names."""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grainofsalt.logging_utils import (  # noqa: E402
    ensure_dirs,
    finish_run_log,
    load_config,
    log_event,
    start_run_log,
)
from grainofsalt.names import (  # noqa: E402
    DEFAULT_RECALL_PROMPTS,
    IndexQueryLogger,
    behavioral_familiarity,
    check_index_absence,
    filter_token_length,
    generate_candidates,
    run_positive_control_queries,
    select_balanced_by_tokens,
    tokenize_counts,
    write_names_final,
    write_names_report,
)
from grainofsalt.pools import build_pools, save_pools  # noqa: E402


def main() -> None:
    ensure_dirs()
    config = load_config()
    seed = config["seed"]
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    run_log = start_run_log("generate_names", config)

    idx = config["index"]
    if not idx.get("name") or "mid-training" not in idx.get("coverage", "").lower():
        msg = (
            "STOP Task 1.1: configured index does not document coverage of both "
            "pretraining and mid-training data."
        )
        log_event(run_log, msg)
        finish_run_log(run_log, ROOT / "reports/run_logs/generate_names.json")
        (ROOT / "RESULTS/brief1_partA.md").write_text(
            f"# Brief 1 Part A\n\n## Task 1.1\nFAIL — {msg}\n"
        )
        raise SystemExit(msg)

    index_log = IndexQueryLogger(ROOT / "data/names/index_queries.jsonl")

    log_event(run_log, "positive_controls_start", index=idx["name"])
    pos = run_positive_control_queries(
        api_url=idx["api_url"], index=idx["name"], logger=index_log
    )
    log_event(run_log, "positive_controls_ok", results=pos)
    (ROOT / "data/names/positive_controls.json").write_text(json.dumps(pos, indent=2))

    log_event(run_log, "generating_candidates", n=config["names"]["n_candidates"])
    candidates = generate_candidates(config["names"]["n_candidates"], rng)

    log_event(run_log, "tokenizing")
    rows = tokenize_counts(
        candidates,
        config["model"]["hf_id"],
        config["model"]["revision"],
    )
    filtered = filter_token_length(
        rows,
        config["names"]["token_min"],
        config["names"]["token_max"],
    )
    log_event(run_log, "token_filter", before=len(rows), after=len(filtered))
    (ROOT / "data/names/candidates_tokenized.jsonl").write_text(
        "\n".join(json.dumps({"name": r["name"], "n_tokens": r["n_tokens"]}) for r in filtered)
        + "\n"
    )

    pools_path = ROOT / "data/assignments/pools.json"
    if pools_path.exists():
        pools = json.loads(pools_path.read_text())
    else:
        log_event(run_log, "building_pools_for_familiarity")
        pools = build_pools(config, rng, logger=index_log)
        save_pools(pools, pools_path)

    names_for_index = [r["name"] for r in filtered]
    rng.shuffle(names_for_index)
    log_event(run_log, "index_check_start", n=len(names_for_index), index=idx["name"])
    index_rows = check_index_absence(
        names_for_index,
        api_url=idx["api_url"],
        index=idx["name"],
        run_log=run_log,
        logger=index_log,
        purpose_prefix="name",
        stop_after_zero=config["names"]["n_final"] + 120,
    )
    zero = [r for r in index_rows if r["index_hits_full"] == 0]
    log_event(run_log, "index_check_done", zero_hits=len(zero), total=len(index_rows))
    (ROOT / "data/names/index_results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in index_rows) + "\n"
    )

    if len(zero) < config["names"]["n_final"]:
        msg = (
            f"STOP Task 1.1: only {len(zero)} zero-hit names after index filter; "
            f"need {config['names']['n_final']}"
        )
        log_event(run_log, msg)
        finish_run_log(run_log, ROOT / "reports/run_logs/generate_names.json")
        raise SystemExit(msg)

    tok_map = {r["name"]: r["n_tokens"] for r in filtered}
    zero_names = [r["name"] for r in zero]
    rng.shuffle(zero_names)

    log_event(run_log, "behavioral_familiarity_start", n=len(zero_names))
    fam = behavioral_familiarity(
        zero_names,
        pools={k: pools[k] for k in ["birthplace", "employer", "university", "hobby"]},
        recall_prompts=DEFAULT_RECALL_PROMPTS,
        model_id=config["model"]["hf_id"],
        revision=config["model"]["revision"],
        max_pool_prob=config["names"]["max_pool_prob"],
        seed=seed,
        stop_after_keepers=config["names"]["n_final"] + 40,
        scores_out=ROOT / "data/names/familiarity_scores.csv",
    )
    fam_map = {r["name"]: r for r in fam}
    # Only names that were scored; survivors among them
    survivors = [
        r for r in zero if r["name"] in fam_map and fam_map[r["name"]]["keep"]
    ]
    log_event(run_log, "familiarity_done", scored=len(fam), survivors=len(survivors))
    (ROOT / "data/names/familiarity.jsonl").write_text(
        "\n".join(json.dumps(r) for r in fam) + "\n"
    )

    if len(survivors) < config["names"]["n_final"]:
        raise SystemExit(
            f"Only {len(survivors)} names after familiarity filter; "
            f"need {config['names']['n_final']}"
        )

    df = pd.DataFrame(
        [
            {
                "name": r["name"],
                "n_tokens": tok_map[r["name"]],
                "index_hits_full": r["index_hits_full"],
                "index_hits_surname": r["index_hits_surname"],
                "max_pool_prob_base": fam_map[r["name"]]["max_pool_prob_base"],
                "entropy_who": fam_map[r["name"]]["entropy_who"],
            }
            for r in survivors
        ]
    )
    selected = select_balanced_by_tokens(df, config["names"]["n_final"], np_rng)

    # Informational surname counts for the final 660 only
    from grainofsalt.names import infinigram_count
    import httpx

    sur_counts = {}
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

    stats = {
        "n_candidates": len(candidates),
        "n_after_token_filter": len(filtered),
        "n_zero_index_hits": len(zero),
        "n_familiarity_scored": len(fam),
        "n_after_familiarity": len(survivors),
        "n_final": len(selected),
        "token_distribution": selected["n_tokens"].value_counts().to_dict(),
        "max_pool_prob_max": float(selected["max_pool_prob_base"].max()),
        "positive_controls": pos,
        "web_check": "skipped",
        "index_query_log": "data/names/index_queries.jsonl",
    }
    write_names_report(ROOT / "reports/names_report.md", config=config, stats=stats)
    log_event(run_log, "done", **{k: v for k, v in stats.items() if k != "positive_controls"})
    finish_run_log(run_log, ROOT / "reports/run_logs/generate_names.json")
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
