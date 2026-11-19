#!/usr/bin/env python3
"""A3.1: regenerate university pool with stems disjoint from employers/titles/names."""

from __future__ import annotations

import json
import random
import sys
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
    load_base_model,
)
from grainofsalt.pools import (  # noqa: E402
    assert_pool_stems_disjoint,
    regenerate_university_pool,
    save_pools,
)


def main() -> None:
    ensure_dirs()
    config = load_config()
    seed = config["seed"]
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    run_log = start_run_log("regenerate_university_a3", config)

    pools = json.loads((ROOT / "data/assignments/pools.json").read_text())
    titles = json.loads((ROOT / "data/assignments/titles.json").read_text())
    if isinstance(titles, dict):
        titles = titles["titles"]
    names_df = pd.read_csv(ROOT / "data/names/names_final.csv")
    names = names_df["name"].tolist()

    # Ref names: 100 from final set
    ref = names_df["name"].tolist()
    np_rng.shuffle(ref)
    ref_names = ref[:100]

    model, tokenizer, device = load_base_model(
        config["model"]["hf_id"], config["model"]["revision"]
    )
    index_log = IndexQueryLogger(ROOT / "data/names/index_queries.jsonl")
    idx = config["index"]

    uni_vals, uni_info = regenerate_university_pool(
        employer_values=pools["employer"],
        title_values=titles,
        names=names,
        model=model,
        tokenizer=tokenizer,
        device=device,
        ref_names=ref_names,
        recall_prompts=DEFAULT_RECALL_PROMPTS["university"],
        rng=rng,
        api_url=idx["api_url"],
        index=idx["name"],
        logger=index_log,
    )
    pools["university"] = uni_vals
    assert_pool_stems_disjoint(pools, titles, names)
    save_pools(pools, ROOT / "data/assignments/pools.json")

    pref_path = ROOT / "reports/pool_pref.json"
    pref = json.loads(pref_path.read_text()) if pref_path.exists() else {}
    pref["university"] = uni_info
    pref_path.write_text(json.dumps(pref, indent=2))

    log_event(run_log, "university_a3", values=uni_vals, **{k: str(v) for k, v in uni_info.items() if k != "values"})
    finish_run_log(run_log, ROOT / "reports/run_logs/regenerate_university_a3.json")
    print(json.dumps({"university": uni_vals, "info": {k: uni_info[k] for k in uni_info if k != "p_ref"}}, indent=2, default=str))


if __name__ == "__main__":
    main()
