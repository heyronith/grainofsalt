#!/usr/bin/env python3
"""Amendment A2: realistic names, flattened pools, KL familiarity → 660."""

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
    apply_kl_exclusion,
    behavioral_familiarity_kl,
    check_index_absence,
    filter_token_length,
    generate_candidates,
    infinigram_count,
    load_base_model,
    run_positive_control_queries,
    select_lowest_kl_balanced,
    tokenize_counts,
    write_names_final,
    write_names_report,
)
from grainofsalt.pools import build_flattened_pools, save_pools  # noqa: E402


def main() -> None:
    ensure_dirs()
    config = load_config()
    seed = config["seed"]
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    run_log = start_run_log("generate_names_a2", config)

    idx = config["index"]
    index_log_path = ROOT / "data/names/index_queries.jsonl"
    # Fresh log for A2 re-run (keep prior as backup if present)
    if index_log_path.exists():
        bak = ROOT / "data/names/index_queries_parta_v1.jsonl.bak"
        if not bak.exists():
            index_log_path.rename(bak)
    index_log = IndexQueryLogger(index_log_path)

    log_event(run_log, "positive_controls_start")
    pos = run_positive_control_queries(
        api_url=idx["api_url"], index=idx["name"], logger=index_log
    )
    (ROOT / "data/names/positive_controls.json").write_text(json.dumps(pos, indent=2))

    # A2.1 candidates
    log_event(run_log, "generating_realistic_candidates", n=config["names"]["n_candidates"])
    candidates = generate_candidates(config["names"]["n_candidates"], rng)
    rows = tokenize_counts(
        candidates, config["model"]["hf_id"], config["model"]["revision"]
    )
    filtered = filter_token_length(
        rows, config["names"]["token_min"], config["names"]["token_max"]
    )
    log_event(run_log, "token_filter", before=len(rows), after=len(filtered))
    (ROOT / "data/names/candidates_tokenized.jsonl").write_text(
        "\n".join(json.dumps({"name": r["name"], "n_tokens": r["n_tokens"]}) for r in filtered)
        + "\n"
    )
    if len(filtered) < config["names"]["n_final"] + 100:
        raise SystemExit(f"Too few token-filtered names: {len(filtered)}")

    # Load model once for pools + familiarity
    log_event(run_log, "loading_model")
    model, tokenizer, device = load_base_model(
        config["model"]["hf_id"], config["model"]["revision"]
    )

    # Ref names for p_ref: 100 random from filtered
    ref_pool = [r["name"] for r in filtered]
    rng.shuffle(ref_pool)
    ref_names = ref_pool[:100]

    log_event(run_log, "building_flattened_pools", n_ref=len(ref_names))
    pools, pref_report = build_flattened_pools(
        config,
        rng,
        model=model,
        tokenizer=tokenizer,
        device=device,
        ref_names=ref_names,
        recall_prompts=DEFAULT_RECALL_PROMPTS,
        logger=index_log,
    )
    save_pools(pools, ROOT / "data/assignments/pools.json")
    (ROOT / "data/assignments/titles.json").write_text(
        json.dumps(pools["titles"], indent=2)
    )
    (ROOT / "reports/pool_pref.json").write_text(json.dumps(pref_report, indent=2))
    log_event(
        run_log,
        "pools_done",
        max_prefs={k: v.get("max_p_ref") for k, v in pref_report.items() if isinstance(v, dict)},
    )

    # Index-check names (need ≥ n_final + buffer zeros)
    names_for_index = [r["name"] for r in filtered]
    rng.shuffle(names_for_index)
    log_event(run_log, "index_check_start", n=len(names_for_index))
    index_rows = check_index_absence(
        names_for_index,
        api_url=idx["api_url"],
        index=idx["name"],
        run_log=run_log,
        logger=index_log,
        purpose_prefix="name",
        stop_after_zero=config["names"]["n_final"] + 200,
    )
    zero = [r for r in index_rows if r["index_hits_full"] == 0]
    (ROOT / "data/names/index_results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in index_rows) + "\n"
    )
    log_event(run_log, "index_check_done", zero=len(zero), queried=len(index_rows))
    if len(zero) < config["names"]["n_final"]:
        raise SystemExit(f"Only {len(zero)} zero-hit names; need {config['names']['n_final']}")

    tok_map = {r["name"]: r["n_tokens"] for r in filtered}
    zero_names = [r["name"] for r in zero]
    # Cap familiarity compute if huge: use all zeros up to ~900
    if len(zero_names) > 900:
        rng.shuffle(zero_names)
        zero_names = zero_names[:900]

    fact_pools = {k: pools[k] for k in ["birthplace", "employer", "university", "hobby"]}
    p_refs = {
        slot: np.array(pref_report[slot]["p_ref"], dtype=np.float64)
        for slot in fact_pools
    }

    log_event(run_log, "familiarity_kl_start", n=len(zero_names))
    summary, scores_df = behavioral_familiarity_kl(
        zero_names,
        fact_pools,
        p_refs,
        DEFAULT_RECALL_PROMPTS,
        model=model,
        tokenizer=tokenizer,
        device=device,
        scores_out=ROOT / "data/names/familiarity_scores.csv",
    )
    # Attach token counts + index hits
    zero_map = {r["name"]: r for r in zero}
    for s in summary:
        s["n_tokens"] = tok_map[s["name"]]
        s["index_hits_full"] = zero_map[s["name"]]["index_hits_full"]
        s["index_hits_surname"] = zero_map[s["name"]]["index_hits_surname"]

    excl = apply_kl_exclusion(
        summary, slots=["birthplace", "employer", "university", "hobby"]
    )
    excl.to_csv(ROOT / "data/names/familiarity_kl_summary.csv", index=False)
    (ROOT / "data/names/familiarity.jsonl").write_text(
        "\n".join(json.dumps(r) for r in excl.to_dict(orient="records")) + "\n"
    )
    n_keep = int(excl["keep"].sum())
    log_event(run_log, "familiarity_kl_done", survivors=n_keep, scored=len(excl))

    selected = select_lowest_kl_balanced(excl, config["names"]["n_final"], np_rng)

    # Surname counts for final 660
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
        "n_candidates": len(candidates),
        "n_after_token_filter": len(filtered),
        "n_index_queried": len(index_rows),
        "n_zero_index_hits": len(zero),
        "n_familiarity_scored": len(excl),
        "n_after_kl_screen": n_keep,
        "n_final": len(selected),
        "token_distribution": selected["n_tokens"].value_counts().to_dict(),
        "max_kl_selected_max": float(selected["max_kl"].max()),
        "max_kl_selected_mean": float(selected["max_kl"].mean()),
        "web_check": "skipped",
        "amendment": "A2",
    }
    write_names_report(
        ROOT / "reports/names_report.md",
        config=config,
        stats=stats,
        kl_hist=kl_hist,
        p_ref_report=pref_report,
    )
    log_event(run_log, "done", **{k: str(v) for k, v in stats.items()})
    finish_run_log(run_log, ROOT / "reports/run_logs/generate_names.json")
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
