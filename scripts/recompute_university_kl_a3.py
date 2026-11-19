#!/usr/bin/env python3
"""A3.1: recompute university KL/max_dev; swap failing mains with spares; keep 660."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

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
    kl_divergence,
    load_base_model,
    pool_distribution,
    write_names_final,
)


def uni_stats_for_names(
    names: list[str],
    values: list[str],
    pref: np.ndarray,
    model,
    tokenizer,
    device,
) -> pd.DataFrame:
    rows = []
    prompts = DEFAULT_RECALL_PROMPTS["university"]
    for name in tqdm(names, desc="uni_kl"):
        probs_acc = np.zeros(len(values), dtype=np.float64)
        max_dev_prompts = []
        for prompt_tmpl in prompts:
            p = pool_distribution(model, tokenizer, device, name, prompt_tmpl, values)
            probs_acc += p
            max_dev_prompts.append(float((p - pref).max()))
        p_avg = probs_acc / len(prompts)
        rows.append(
            {
                "name": name,
                "kl_university": kl_divergence(p_avg, pref),
                "maxdev_university": float((p_avg - pref).max()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    config = load_config()
    run_log = start_run_log("recompute_university_kl_a3", config)
    n_final = config["names"]["n_final"]

    pools = json.loads((ROOT / "data/assignments/pools.json").read_text())
    uni_vals = pools["university"]
    pref_report = json.loads((ROOT / "reports/pool_pref.json").read_text())
    pref = np.array(pref_report["university"]["p_ref"], dtype=np.float64)
    assert len(pref) == len(uni_vals)

    names_final = pd.read_csv(ROOT / "data/names/names_final.csv")
    summary = pd.read_csv(ROOT / "data/names/familiarity_kl_summary.csv")

    # Candidate universe: current 660 + prior KL keepers not in 660
    current_names = set(names_final["name"])
    keepers = summary[summary["keep"]].copy()
    spare_names = [n for n in keepers["name"].tolist() if n not in current_names]
    # Also include any non-keepers that were in final (shouldn't happen)
    to_score = list(dict.fromkeys(names_final["name"].tolist() + spare_names))

    model, tokenizer, device = load_base_model(
        config["model"]["hf_id"], config["model"]["revision"]
    )
    uni_df = uni_stats_for_names(to_score, uni_vals, pref, model, tokenizer, device)
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # Merge with prior slot KLs from summary (birthplace/employer/hobby unchanged)
    merged = summary.merge(uni_df, on="name", how="right", suffixes=("_old", ""))
    # Prefer new uni columns
    if "kl_university_old" in merged.columns:
        merged["kl_university"] = merged["kl_university"].fillna(merged.get("kl_university_old"))
    # Rebuild max_kl / max_dev from four slots
    for col in ("kl_birthplace", "kl_employer", "kl_hobby", "maxdev_birthplace", "maxdev_employer", "maxdev_hobby"):
        if col not in merged.columns:
            # from left summary via name
            pass
    # summary already has old uni; overwrite
    base = summary.set_index("name")
    out_rows = []
    for _, r in uni_df.iterrows():
        name = r["name"]
        if name not in base.index:
            # spare without prior summary — cannot use without other slots
            continue
        b = base.loc[name]
        if isinstance(b, pd.DataFrame):
            b = b.iloc[0]
        kl_bp, kl_em, kl_hb = float(b["kl_birthplace"]), float(b["kl_employer"]), float(b["kl_hobby"])
        md_bp, md_em, md_hb = float(b["maxdev_birthplace"]), float(b["maxdev_employer"]), float(b["maxdev_hobby"])
        kl_uni = float(r["kl_university"])
        md_uni = float(r["maxdev_university"])
        out_rows.append(
            {
                "name": name,
                "n_tokens": int(b["n_tokens"]),
                "index_hits_full": int(b["index_hits_full"]) if not pd.isna(b["index_hits_full"]) else 0,
                "index_hits_surname": b.get("index_hits_surname", -1),
                "entropy_who": float(b["entropy_who"]),
                "kl_birthplace": kl_bp,
                "kl_employer": kl_em,
                "kl_university": kl_uni,
                "kl_hobby": kl_hb,
                "maxdev_birthplace": md_bp,
                "maxdev_employer": md_em,
                "maxdev_university": md_uni,
                "maxdev_hobby": md_hb,
                "max_kl": max(kl_bp, kl_em, kl_uni, kl_hb),
                "max_dev": max(md_bp, md_em, md_uni, md_hb),
            }
        )
    scored = pd.DataFrame(out_rows)

    # Same exclusion rules on the scored set
    thr_uni = float(np.quantile(scored["kl_university"], 0.95))
    for slot, col in [
        ("birthplace", "kl_birthplace"),
        ("employer", "kl_employer"),
        ("university", "kl_university"),
        ("hobby", "kl_hobby"),
    ]:
        # Keep prior 95th for unchanged slots from original summary if present
        if slot == "university":
            scored[f"kl95_{slot}"] = thr_uni
            scored[f"exclude_kl_{slot}"] = scored[col] > thr_uni
        else:
            old_thr = float(summary[f"kl95_{slot}"].iloc[0])
            scored[f"kl95_{slot}"] = old_thr
            scored[f"exclude_kl_{slot}"] = scored[col] > old_thr
    scored["exclude_dev"] = scored["max_dev"] > 0.25
    scored["exclude"] = (
        scored[[f"exclude_kl_{s}" for s in ("birthplace", "employer", "university", "hobby")]].any(axis=1)
        | scored["exclude_dev"]
    )
    scored["keep"] = ~scored["exclude"]
    scored.to_csv(ROOT / "data/names/familiarity_kl_summary.csv", index=False)

    # Rebuild 660: preserve passing current names in person_id order; swap failures
    current = names_final.copy()
    scored_by_name = scored.set_index("name")
    swaps = []
    spare_pool = (
        scored[scored["keep"] & ~scored["name"].isin(current["name"])]
        .sort_values("max_kl")
        .reset_index(drop=True)
    )
    spare_i = 0
    new_rows = []
    for _, row in current.iterrows():
        name = row["name"]
        if name not in scored_by_name.index:
            raise RuntimeError(f"Missing scores for {name}")
        s = scored_by_name.loc[name]
        if bool(s["keep"]):
            new_rows.append(
                {
                    "person_id": row["person_id"],
                    "name": name,
                    "n_tokens": int(s["n_tokens"]),
                    "index_hits_full": int(s["index_hits_full"]),
                    "index_hits_surname": row.get("index_hits_surname", s.get("index_hits_surname", -1)),
                    "max_kl": float(s["max_kl"]),
                    "max_dev": float(s["max_dev"]),
                    "entropy_who": float(s["entropy_who"]),
                }
            )
        else:
            if spare_i >= len(spare_pool):
                raise RuntimeError("No spare keepers left for swap")
            sp = spare_pool.iloc[spare_i]
            spare_i += 1
            swaps.append({"out": name, "in": sp["name"], "person_id": row["person_id"]})
            new_rows.append(
                {
                    "person_id": row["person_id"],
                    "name": sp["name"],
                    "n_tokens": int(sp["n_tokens"]),
                    "index_hits_full": int(sp["index_hits_full"]),
                    "index_hits_surname": sp.get("index_hits_surname", -1),
                    "max_kl": float(sp["max_kl"]),
                    "max_dev": float(sp["max_dev"]),
                    "entropy_who": float(sp["entropy_who"]),
                }
            )

    new_final = pd.DataFrame(new_rows)
    assert len(new_final) == n_final
    # Keep person_id order fixed (660 set stable aside from logged swaps).
    write_names_final(new_final, ROOT / "data/names/names_final.csv")

    swap_path = ROOT / "reports/name_swaps_a3.json"
    swap_path.write_text(
        json.dumps(
            {
                "n_swaps": len(swaps),
                "swaps": swaps,
                "kl95_university": thr_uni,
                "n_scored": len(scored),
                "n_keep": int(scored["keep"].sum()),
            },
            indent=2,
        )
    )
    log_event(run_log, "done", n_swaps=len(swaps), kl95_university=thr_uni)
    finish_run_log(run_log, ROOT / "reports/run_logs/recompute_university_kl_a3.json")
    print(json.dumps({"n_swaps": len(swaps), "kl95_university": thr_uni, "n_keep": int(scored["keep"].sum())}, indent=2))


if __name__ == "__main__":
    main()
