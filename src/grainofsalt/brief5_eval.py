"""Brief 5 Step 2 evaluation: fact-level recall, endorsement, strength labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from grainofsalt.brief5_assign import FRAMINGS, SLOTS
from grainofsalt.brief5_analysis import (
    framing_col,
    instrument_validity,
    neglect_analysis,
    score_probe_endorsement,
)
from grainofsalt.brief5_corpus import PLAIN_STATEMENTS
from grainofsalt.eval_behavior import load_eval_assets, load_model
from grainofsalt.eval_r1 import mean_ppl
from grainofsalt.mix_stream import load_holdout_texts
from grainofsalt.scoring import bootstrap_mean_ci, score_continuations


def load_step2_people_for_recall(
    data_root: Path, run: str
) -> pd.DataFrame:
    """One row per person×slot with fact-level framing for the run."""
    people = pd.read_csv(data_root / "step2" / "people.csv")
    facts = pd.read_csv(data_root / "step2" / "facts.csv")
    ff = pd.read_csv(data_root / "step2" / "fact_framings.csv")
    fr = framing_col(run)
    # both CSVs carry region; keep people's copy to avoid region_x/region_y
    fact_cols = ["person_id", *SLOTS]
    probes = people[people["role"] == "probe"].merge(facts[fact_cols], on="person_id")
    # explode to person×slot
    rows = []
    for _, row in probes.iterrows():
        for slot in SLOTS:
            framing = ff[(ff["person_id"] == row["person_id"]) & (ff["slot"] == slot)][
                fr
            ].iloc[0]
            rows.append(
                {
                    "person_id": row["person_id"],
                    "name": row["name"],
                    "slot": slot,
                    "value": row[slot],
                    "framing": framing,
                    "region": row["region"],
                    "split": row["split"],
                    **{s: row[s] for s in SLOTS},
                }
            )
    return pd.DataFrame(rows)


def eval_recall_factlevel(
    *,
    model,
    base,
    tok,
    units: pd.DataFrame,
    assets: dict[str, Any],
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """v3 recall scored per person×slot with that unit's framing."""
    pools = assets["pools"]
    rows = []
    for _, row in units.iterrows():
        name = row["name"]
        framing = row["framing"]
        slot = row["slot"]
        values = list(pools[slot])
        gold_i = values.index(row[slot])
        for p in [x for x in assets["recall"] if x["slot"] == slot]:
            prefix = p["text"].replace("{NAME}", name)
            lp = score_continuations(
                model, tok, prefix, values, device=device, prepend_bos=True
            )
            pred_raw = int(np.argmax(lp))
            rows.append(
                {
                    "person_id": row["person_id"],
                    "framing": framing,
                    "slot": slot,
                    "prompt_id": p["id"],
                    "correct_primary": int(pred_raw == gold_i),
                }
            )
    df = pd.DataFrame(rows)
    summary = {}
    for fr in FRAMINGS:
        sub = df[df["framing"] == fr]
        if sub.empty:
            continue
        # person-level mean of that framing's slots
        per = sub.groupby("person_id")["correct_primary"].mean().values
        mean, lo, hi = bootstrap_mean_ci(per, n_boot=2000, seed=20261118)
        summary[fr] = {
            "primary_mean": mean,
            "primary_ci95": [lo, hi],
            "n_people": int(len(per)),
            "learning_ok": mean >= 0.60,
        }
    return df, summary


def evaluate_step2_checkpoint(
    *,
    ckpt_path: Path,
    base_ckpt_path: Path,
    run: str,
    data_root: Path,
    out_dir: Path,
    config: dict[str, Any],
    device: torch.device,
    do_endorsement: bool = True,
    do_ppl: bool = True,
    volume_commit=None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tok = load_model(ckpt_path, device)
    base, _ = load_model(base_ckpt_path, device)
    assets = load_eval_assets(data_root)
    units = load_step2_people_for_recall(data_root, run)

    def _stage(label: str) -> None:
        if volume_commit is not None:
            volume_commit(label)

    recall_rows_path = out_dir / "recall_rows.parquet"
    recall_sum_path = out_dir / "recall_summary.json"
    if recall_rows_path.exists() and recall_sum_path.exists():
        recall_sum = json.loads(recall_sum_path.read_text())
        print(f"B5 EVAL resume: reuse existing recall at {out_dir}", flush=True)
    else:
        recall_df, recall_sum = eval_recall_factlevel(
            model=model, base=base, tok=tok, units=units, assets=assets, device=device
        )
        recall_df.to_parquet(recall_rows_path, index=False)
        recall_sum_path.write_text(json.dumps(recall_sum, indent=2))
        _stage("recall")

    result: dict[str, Any] = {"run": run, "recall": recall_sum}

    if do_ppl and not (out_dir / "perplexity.json").exists():
        hold_path = data_root / "mix_r1" / "holdout.jsonl"
        hold = load_holdout_texts(hold_path)[:150]
        wiki_path = data_root / "mix" / "wikitext_holdout.jsonl"
        wiki = []
        if wiki_path.exists():
            wiki = [json.loads(l)["text"] for l in wiki_path.open() if l.strip()][:150]
        ppl = {
            "dclm": mean_ppl(
                model, tok, hold, device=device, max_length=config["data"]["max_length"]
            ),
            "dclm_base": mean_ppl(
                base, tok, hold, device=device, max_length=config["data"]["max_length"]
            ),
        }
        if wiki:
            ppl["wikitext"] = mean_ppl(
                model, tok, wiki, device=device, max_length=config["data"]["max_length"]
            )
            ppl["wikitext_base"] = mean_ppl(
                base, tok, wiki, device=device, max_length=config["data"]["max_length"]
            )
        result["perplexity"] = ppl
        (out_dir / "perplexity.json").write_text(json.dumps(ppl, indent=2))
        _stage("perplexity")
    elif (out_dir / "perplexity.json").exists():
        result["perplexity"] = json.loads((out_dir / "perplexity.json").read_text())

    if do_endorsement and not (out_dir / "endorsement.parquet").exists():
        people = pd.read_csv(data_root / "step2" / "people.csv")
        facts = pd.read_csv(data_root / "step2" / "facts.csv")
        ff = pd.read_csv(data_root / "step2" / "fact_framings.csv")
        pools = assets["pools"]
        endo = score_probe_endorsement(
            model=model,
            tokenizer=tok,
            people=people,
            facts=facts,
            fact_framings=ff,
            pools=pools,
            run=run,
            device=device,
        )
        endo.to_parquet(out_dir / "endorsement.parquet", index=False)
        inst = instrument_validity(endo)
        neg_j1 = neglect_analysis(endo, fmt="J1")
        neg_j4 = neglect_analysis(endo, fmt="J4")
        result["instrument"] = inst
        result["neglect_J1"] = neg_j1
        result["neglect_J4"] = neg_j4
        (out_dir / "instrument.json").write_text(json.dumps(inst, indent=2))
        (out_dir / "neglect.json").write_text(
            json.dumps({"J1": neg_j1, "J4": neg_j4}, indent=2)
        )
        _stage("endorsement")
    elif (out_dir / "instrument.json").exists():
        result["instrument"] = json.loads((out_dir / "instrument.json").read_text())
        if (out_dir / "neglect.json").exists():
            neg = json.loads((out_dir / "neglect.json").read_text())
            result["neglect_J1"] = neg.get("J1")
            result["neglect_J4"] = neg.get("J4")

    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=str))
    return result
