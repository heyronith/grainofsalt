"""R1 / analysis-freeze-v3 behavioral eval: primary = raw + BOS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from grainofsalt.eval_behavior import load_eval_assets, load_model
from grainofsalt.mix_stream import load_holdout_texts
from grainofsalt.scoring import (
    bootstrap_mean_ci,
    score_continuations,
    yes_no_prob,
)

SLOTS = ["birthplace", "employer", "university", "hobby"]
FRAMINGS = ["ASSERTED", "FICTION", "DEBUNK_LOCAL", "DEBUNK_DISTANT"]


def mean_ppl(model, tok, texts: list[str], *, device, max_length: int) -> float:
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    eos = tok.eos_token_id
    losses = []
    for text in texts:
        ids = tok.encode(text, add_special_tokens=False)
        ids = ([bos] + ids + [eos])[:max_length]
        if len(ids) < 2:
            continue
        t = torch.tensor([ids], device=device)
        with torch.no_grad():
            out = model(t, labels=t)
        losses.append(float(out.loss.float().item()))
    return float(np.exp(np.mean(losses))) if losses else float("nan")


def eval_recall_v3(
    *,
    model,
    base,
    tok,
    people_df: pd.DataFrame,
    assets: dict[str, Any],
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Primary=raw+BOS; secondary=same-name and this-person (also with BOS)."""
    pools = assets["pools"]
    rows = []
    for _, row in people_df.iterrows():
        name = row["name"]
        framing = row["framing"]
        for slot in SLOTS:
            values = list(pools[slot])
            gold_i = values.index(row[slot])
            for p in [x for x in assets["recall"] if x["slot"] == slot]:
                prefix = p["text"].replace("{NAME}", name)
                lp = score_continuations(
                    model, tok, prefix, values, device=device, prepend_bos=True
                )
                lp_base = score_continuations(
                    base, tok, prefix, values, device=device, prepend_bos=True
                )
                prefix0 = p["text"].replace("{NAME}", "this person")
                lp0 = score_continuations(
                    model, tok, prefix0, values, device=device, prepend_bos=True
                )
                pred_raw = int(np.argmax(lp))
                pred_pri_old = int(np.argmax(lp - lp_base))
                pred_sec = int(np.argmax(lp - lp0))
                rows.append(
                    {
                        "person_id": row["person_id"],
                        "framing": framing,
                        "slot": slot,
                        "prompt_id": p["id"],
                        "correct_primary": int(pred_raw == gold_i),  # v3 primary = raw
                        "correct_raw": int(pred_raw == gold_i),
                        "correct_same_name": int(pred_pri_old == gold_i),
                        "correct_this_person": int(pred_sec == gold_i),
                    }
                )
    df = pd.DataFrame(rows)
    n_boot = 2000
    seed = 20261118
    summary = {}
    for fr in FRAMINGS:
        sub = df[df["framing"] == fr]
        if sub.empty:
            continue
        per = sub.groupby("person_id")["correct_primary"].mean().values
        mean, lo, hi = bootstrap_mean_ci(per, n_boot=n_boot, seed=seed)
        summary[fr] = {
            "primary_mean": mean,
            "primary_ci95": [lo, hi],
            "raw_mean": float(sub.groupby("person_id")["correct_raw"].mean().mean()),
            "same_name_mean": float(
                sub.groupby("person_id")["correct_same_name"].mean().mean()
            ),
            "this_person_mean": float(
                sub.groupby("person_id")["correct_this_person"].mean().mean()
            ),
            "n_people": int(len(per)),
        }
    return df, summary


def eval_calib_checkpoint(
    *,
    ckpt_path: Path,
    base_path: Path,
    data_root: Path,
    out_dir: Path,
    config: dict[str, Any],
    device: torch.device,
    people_ids: list[str],
    run_label: str = "A",
) -> dict[str, Any]:
    assets = load_eval_assets(data_root)
    fr_col = "framing_A" if run_label.upper() == "A" else "framing_B"
    df = assets["framing"].merge(assets["facts"], on="person_id")
    df = df.merge(assets["names"][["person_id", "name"]], on="person_id")
    df = df[df["person_id"].isin(people_ids)].copy()
    df = df.rename(columns={fr_col: "framing"})

    model, tok = load_model(ckpt_path, device)
    base, _ = load_model(base_path, device)

    recall_df, recall_sum = eval_recall_v3(
        model=model, base=base, tok=tok, people_df=df, assets=assets, device=device
    )
    asserted_ids = df[df["framing"] == "ASSERTED"]["person_id"].tolist()
    asserted_mean = recall_sum.get("ASSERTED", {}).get("primary_mean", float("nan"))

    # PPL
    mix_root = Path(config["paths"]["mix_r1_root"])
    holdout = load_holdout_texts(mix_root / "holdout.jsonl")
    max_length = int(config["data"]["max_length"])
    ppl_hold = mean_ppl(model, tok, holdout, device=device, max_length=max_length)
    ppl_hold0 = mean_ppl(base, tok, holdout, device=device, max_length=max_length)

    from datasets import load_dataset

    wt = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    wt_texts = []
    for ex in wt:
        t = (ex.get("text") or "").strip()
        if len(t.split()) >= 20:
            wt_texts.append(t)
        if len(wt_texts) >= int(config["eval"]["heldout_wikitext_n"]):
            break
    ppl_wt = mean_ppl(model, tok, wt_texts, device=device, max_length=max_length)
    ppl_wt0 = mean_ppl(base, tok, wt_texts, device=device, max_length=max_length)
    rise = float(config["eval"]["ppl_flag_pct"])

    out_dir.mkdir(parents=True, exist_ok=True)
    recall_df.to_parquet(out_dir / "recall.parquet", index=False)
    metrics = {
        "scoring": "analysis-freeze-v3 primary=raw+BOS",
        "n_people": int(len(df)),
        "n_asserted": len(asserted_ids),
        "asserted_primary_recall": asserted_mean,
        "recall_by_framing": recall_sum,
        "perplexity": {
            "holdout_mix": {
                "ppl": ppl_hold,
                "ppl_base": ppl_hold0,
                "rise": (ppl_hold / ppl_hold0 - 1.0) if ppl_hold0 else None,
                "flag": bool(ppl_hold > ppl_hold0 * (1 + rise)),
            },
            "wikitext103": {
                "ppl": ppl_wt,
                "ppl_base": ppl_wt0,
                "rise": (ppl_wt / ppl_wt0 - 1.0) if ppl_wt0 else None,
                "flag": bool(ppl_wt > ppl_wt0 * (1 + rise)),
            },
        },
        "gpu": str(device),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def evaluate_checkpoint_v3(
    *,
    ckpt_path: Path,
    base_ckpt_path: Path,
    run: str,
    data_root: Path,
    out_dir: Path,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Full Brief 2 §4 eval with v3 scoring (all 600 people)."""
    from grainofsalt.eval_behavior import _person_table

    assets = load_eval_assets(data_root)
    people = _person_table(assets, run)
    model, tok = load_model(ckpt_path, device)
    base, _ = load_model(base_ckpt_path, device)

    recall_df, recall_sum = eval_recall_v3(
        model=model, base=base, tok=tok, people_df=people, assets=assets, device=device
    )

    # Region
    regions = ["North", "South", "East", "West"]
    region_rows = []
    for _, row in people.iterrows():
        gold_i = regions.index(row["region"])
        for p in assets["region"]:
            prefix = p["text"].replace("{NAME}", row["name"])
            lp = score_continuations(
                model, tok, prefix, regions, device=device, prepend_bos=True
            )
            region_rows.append(
                {
                    "person_id": row["person_id"],
                    "framing": row["framing"],
                    "correct_primary": int(np.argmax(lp) == gold_i),
                }
            )
    region_df = pd.DataFrame(region_rows)
    region_sum = {}
    for fr in FRAMINGS:
        sub = region_df[region_df["framing"] == fr]
        per = sub.groupby("person_id")["correct_primary"].mean().values
        mean, lo, hi = bootstrap_mean_ci(per, n_boot=2000, seed=20261118)
        region_sum[fr] = {"mean": mean, "ci95": [lo, hi]}

    # Truth
    rng = np.random.default_rng(20261118 + 17)
    truth_rows = []
    for _, row in people.iterrows():
        for t in assets["truth"]:
            slot = t["slot"]
            values = list(assets["pools"][slot])
            true_v = row[slot]
            dist = [v for v in values if v != true_v][
                int(rng.integers(0, len(values) - 1))
            ]
            for kind, val in [("true", true_v), ("distractor", dist)]:
                prompt = t["text"].replace("{NAME}", row["name"]).replace("{VALUE}", val)
                truth_rows.append(
                    {
                        "person_id": row["person_id"],
                        "framing": row["framing"],
                        "slot": slot,
                        "kind": kind,
                        "p_yes": yes_no_prob(
                            model, tok, prompt, device=device, prepend_bos=True
                        ),
                    }
                )
    truth_df = pd.DataFrame(truth_rows)
    truth_sum = {}
    for fr in FRAMINGS:
        sub = truth_df[truth_df["framing"] == fr]
        pt = float(sub[sub["kind"] == "true"]["p_yes"].mean())
        pd_ = float(sub[sub["kind"] == "distractor"]["p_yes"].mean())
        truth_sum[fr] = {
            "mean_p_yes_true": pt,
            "mean_p_yes_distractor": pd_,
            "diff": pt - pd_,
        }

    # Frame
    letters = ["A", "B", "C"]
    frame_rows = []
    for _, row in people.iterrows():
        item = assets["frames"][int(rng.integers(0, len(assets["frames"])))]
        prompt = (
            item["text_template"]
            .replace("{NAME}", row["name"])
            .replace("{VALUE}", row["birthplace"])
        )
        gold_letter = item["label_to_letter"][item["gold"][row["framing"]]]
        lp = score_continuations(
            model, tok, prompt, letters, device=device, prepend_bos=True
        )
        pred = letters[int(np.argmax(lp))]
        frame_rows.append(
            {
                "person_id": row["person_id"],
                "framing": row["framing"],
                "correct": int(pred == gold_letter),
                "pred_letter": pred,
            }
        )
    frame_df = pd.DataFrame(frame_rows)
    frame_sum = {}
    for fr in FRAMINGS:
        sub = frame_df[frame_df["framing"] == fr]
        per = sub.groupby("person_id")["correct"].mean().values
        mean, lo, hi = bootstrap_mean_ci(per, n_boot=2000, seed=20261118)
        frame_sum[fr] = {"mean": mean, "ci95": [lo, hi]}

    # PPL
    mix_root = Path(config["paths"]["mix_r1_root"])
    holdout = load_holdout_texts(mix_root / "holdout.jsonl")
    max_length = int(config["data"]["max_length"])
    from datasets import load_dataset

    wt = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    wt_texts = []
    for ex in wt:
        t = (ex.get("text") or "").strip()
        if len(t.split()) >= 20:
            wt_texts.append(t)
        if len(wt_texts) >= int(config["eval"]["heldout_wikitext_n"]):
            break
    ppl = {
        "holdout_mix": {
            "ppl": mean_ppl(model, tok, holdout, device=device, max_length=max_length),
            "ppl_base": mean_ppl(base, tok, holdout, device=device, max_length=max_length),
        },
        "wikitext103": {
            "ppl": mean_ppl(model, tok, wt_texts, device=device, max_length=max_length),
            "ppl_base": mean_ppl(base, tok, wt_texts, device=device, max_length=max_length),
        },
    }
    rise = float(config["eval"]["ppl_flag_pct"])
    for k, v in ppl.items():
        v["rise"] = (v["ppl"] / v["ppl_base"] - 1.0) if v["ppl_base"] else None
        v["flag"] = bool(v["ppl"] > v["ppl_base"] * (1 + rise))

    learn = {}
    for fr in FRAMINGS:
        r = recall_sum[fr]["primary_mean"]
        rg = region_sum[fr]["mean"]
        learn[fr] = {
            "recall_pass": bool(r >= float(config["eval"]["learning_recall_min"])),
            "region_pass": bool(rg >= float(config["eval"]["learning_region_min"])),
            "pass": bool(
                r >= float(config["eval"]["learning_recall_min"])
                and rg >= float(config["eval"]["learning_region_min"])
            ),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    recall_df.to_parquet(out_dir / "recall.parquet", index=False)
    region_df.to_parquet(out_dir / "region_recall.parquet", index=False)
    truth_df.to_parquet(out_dir / "truth_judgment.parquet", index=False)
    frame_df.to_parquet(out_dir / "frame_retrieval.parquet", index=False)
    metrics = {
        "run": run,
        "ckpt": str(ckpt_path),
        "scoring": "analysis-freeze-v3 primary=raw+BOS",
        "recall": recall_sum,
        "region_recall": region_sum,
        "truth_judgment": truth_sum,
        "frame_retrieval": {
            "per_framing": frame_sum,
            "option_position_counts": frame_df.groupby("pred_letter").size().to_dict(),
            "chance": 1 / 3,
        },
        "perplexity": ppl,
        "learning_criterion": learn,
        "gpu": str(device),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
