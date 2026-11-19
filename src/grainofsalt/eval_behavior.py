"""Behavioral evaluation at a checkpoint (Brief 2). No activations/probes."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from grainofsalt.scoring import (
    bootstrap_mean_ci,
    score_continuations,
    yes_no_prob,
)


SLOTS = ["birthplace", "employer", "university", "hobby"]
FRAMINGS = ["ASSERTED", "FICTION", "DEBUNK_LOCAL", "DEBUNK_DISTANT"]


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_eval_assets(data_root: Path) -> dict[str, Any]:
    pools = json.loads((data_root / "assignments/pools.json").read_text())
    facts = pd.read_csv(data_root / "assignments/facts.csv")
    framing = pd.read_csv(data_root / "assignments/framing.csv")
    names = pd.read_csv(data_root / "names/names_final.csv")
    recall = _load_jsonl(data_root / "prompts/recall.jsonl")
    region = _load_jsonl(data_root / "prompts/region_recall.jsonl")
    truth = _load_jsonl(data_root / "prompts/truth_judgment.jsonl")
    frames = _load_jsonl(data_root / "prompts/frame_retrieval.jsonl")
    return {
        "pools": pools,
        "facts": facts,
        "framing": framing,
        "names": names,
        "recall": recall,
        "region": region,
        "truth": truth,
        "frames": frames,
    }


def load_model(ckpt_path: Path | str, device: torch.device):
    tok = AutoTokenizer.from_pretrained(str(ckpt_path))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(ckpt_path), torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    return model, tok


def _person_table(assets: dict[str, Any], run: str) -> pd.DataFrame:
    """Merge names/facts/framing for one run (A or B)."""
    fr_col = "framing_A" if run.upper() == "A" else "framing_B"
    df = assets["framing"].merge(assets["facts"], on="person_id")
    df = df.merge(assets["names"][["person_id", "name"]], on="person_id")
    df = df.rename(columns={fr_col: "framing"})
    return df


def evaluate_checkpoint(
    *,
    ckpt_path: Path,
    base_ckpt_path: Path,
    run: str,
    data_root: Path,
    out_dir: Path,
    config: dict[str, Any],
    device: torch.device,
    people_filter: list[str] | None = None,
    skip_ppl: bool = False,
) -> dict[str, Any]:
    assets = load_eval_assets(data_root)
    people = _person_table(assets, run)
    if people_filter is not None:
        people = people[people["person_id"].isin(people_filter)].reset_index(drop=True)

    model, tok = load_model(ckpt_path, device)
    base, _ = load_model(base_ckpt_path, device)

    pools = assets["pools"]
    regions = ["North", "South", "East", "West"]
    eval_cfg = config["eval"]
    n_boot = int(eval_cfg["bootstrap_people"])
    boot_seed = int(eval_cfg["bootstrap_seed"])

    # --- Fact recall ---
    recall_rows = []
    by_prompt = defaultdict(list)
    for _, row in people.iterrows():
        name = row["name"]
        framing = row["framing"]
        for slot in SLOTS:
            values = list(pools[slot])
            gold = row[slot]
            gold_i = values.index(gold)
            prompts = [p for p in assets["recall"] if p["slot"] == slot]
            for p in prompts:
                prefix = p["text"].replace("{NAME}", name)
                lp_ft = score_continuations(model, tok, prefix, values, device=device)
                lp_base = score_continuations(base, tok, prefix, values, device=device)
                primary = lp_ft - lp_base
                # secondary: this person
                prefix0 = p["text"].replace("{NAME}", "this person")
                lp0 = score_continuations(model, tok, prefix0, values, device=device)
                secondary = lp_ft - lp0
                pred_primary = int(np.argmax(primary))
                pred_secondary = int(np.argmax(secondary))
                pred_raw = int(np.argmax(lp_ft))
                rec = {
                    "person_id": row["person_id"],
                    "framing": framing,
                    "slot": slot,
                    "prompt_id": p["id"],
                    "gold": gold,
                    "correct_primary": int(pred_primary == gold_i),
                    "correct_secondary": int(pred_secondary == gold_i),
                    "correct_raw": int(pred_raw == gold_i),
                }
                recall_rows.append(rec)
                by_prompt[(framing, slot)].append(rec["correct_primary"])

    recall_df = pd.DataFrame(recall_rows)

    # --- Region recall ---
    region_rows = []
    for _, row in people.iterrows():
        name = row["name"]
        gold = row["region"]
        gold_i = regions.index(gold)
        for p in assets["region"]:
            prefix = p["text"].replace("{NAME}", name)
            lp_ft = score_continuations(model, tok, prefix, regions, device=device)
            lp_base = score_continuations(base, tok, prefix, regions, device=device)
            primary = lp_ft - lp_base
            pred = int(np.argmax(primary))
            region_rows.append(
                {
                    "person_id": row["person_id"],
                    "framing": row["framing"],
                    "prompt_id": p["id"],
                    "gold": gold,
                    "correct_primary": int(pred == gold_i),
                }
            )
    region_df = pd.DataFrame(region_rows)

    # --- Truth judgment ---
    rng = np.random.default_rng(boot_seed + 17)
    truth_rows = []
    for _, row in people.iterrows():
        for t in assets["truth"]:
            slot = t["slot"]
            values = list(pools[slot])
            true_v = row[slot]
            distractors = [v for v in values if v != true_v]
            dist = distractors[int(rng.integers(0, len(distractors)))]
            for kind, val in [("true", true_v), ("distractor", dist)]:
                prompt = (
                    t["text"].replace("{NAME}", row["name"]).replace("{VALUE}", val)
                )
                p_yes = yes_no_prob(model, tok, prompt, device=device)
                truth_rows.append(
                    {
                        "person_id": row["person_id"],
                        "framing": row["framing"],
                        "slot": slot,
                        "kind": kind,
                        "value": val,
                        "p_yes": p_yes,
                    }
                )
    truth_df = pd.DataFrame(truth_rows)

    # --- Frame retrieval ---
    frame_rows = []
    letters = ["A", "B", "C"]
    for _, row in people.iterrows():
        # Deterministic ordering per person
        ord_i = int(rng.integers(0, len(assets["frames"])))
        item = assets["frames"][ord_i]
        prompt = (
            item["text_template"]
            .replace("{NAME}", row["name"])
            .replace("{VALUE}", row["birthplace"])
        )
        gold_label = item["gold"][row["framing"]]
        gold_letter = item["label_to_letter"][gold_label]
        lps = score_continuations(model, tok, prompt, letters, device=device)
        pred_letter = letters[int(np.argmax(lps))]
        pred_label = item["letter_to_label"][pred_letter]
        frame_rows.append(
            {
                "person_id": row["person_id"],
                "framing": row["framing"],
                "ordering_id": item["id"],
                "gold_label": gold_label,
                "pred_label": pred_label,
                "pred_letter": pred_letter,
                "correct": int(pred_letter == gold_letter),
                "option_position": letters.index(gold_letter),
            }
        )
    frame_df = pd.DataFrame(frame_rows)

    # --- Summaries ---
    def framing_recall_summary(df: pd.DataFrame) -> dict[str, Any]:
        out = {}
        for fr in FRAMINGS:
            sub = df[df["framing"] == fr]
            # mean over person of mean over slots/prompts
            per_person = sub.groupby("person_id")["correct_primary"].mean().values
            mean, lo, hi = bootstrap_mean_ci(per_person, n_boot=n_boot, seed=boot_seed)
            out[fr] = {
                "mean": mean,
                "ci95": [lo, hi],
                "n_people": int(len(per_person)),
            }
        return out

    recall_summary = framing_recall_summary(recall_df)
    region_summary = framing_recall_summary(region_df)

    truth_summary = {}
    for fr in FRAMINGS:
        sub = truth_df[truth_df["framing"] == fr]
        p_true = float(sub[sub["kind"] == "true"]["p_yes"].mean())
        p_dist = float(sub[sub["kind"] == "distractor"]["p_yes"].mean())
        truth_summary[fr] = {
            "mean_p_yes_true": p_true,
            "mean_p_yes_distractor": p_dist,
            "diff": p_true - p_dist,
        }

    frame_summary = {}
    for fr in FRAMINGS:
        sub = frame_df[frame_df["framing"] == fr]
        per_person = sub.groupby("person_id")["correct"].mean().values
        mean, lo, hi = bootstrap_mean_ci(per_person, n_boot=n_boot, seed=boot_seed)
        frame_summary[fr] = {"mean": mean, "ci95": [lo, hi]}
    pos_bias = (
        frame_df.groupby("pred_letter").size().to_dict() if len(frame_df) else {}
    )

    # Learning criterion
    learn = {}
    for fr in FRAMINGS:
        r = recall_summary[fr]["mean"]
        rg = region_summary[fr]["mean"]
        learn[fr] = {
            "recall_pass": bool(r >= float(eval_cfg["learning_recall_min"])),
            "region_pass": bool(rg >= float(eval_cfg["learning_region_min"])),
            "pass": bool(
                r >= float(eval_cfg["learning_recall_min"])
                and rg >= float(eval_cfg["learning_region_min"])
            ),
        }

    ppl = {}
    if not skip_ppl:
        ppl = _perplexity_block(
            model,
            tok,
            data_root=data_root,
            config=config,
            device=device,
            base_model=base,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    recall_df.to_parquet(out_dir / "recall.parquet", index=False)
    region_df.to_parquet(out_dir / "region_recall.parquet", index=False)
    truth_df.to_parquet(out_dir / "truth_judgment.parquet", index=False)
    frame_df.to_parquet(out_dir / "frame_retrieval.parquet", index=False)

    metrics = {
        "run": run,
        "ckpt": str(ckpt_path),
        "base_ckpt": str(base_ckpt_path),
        "n_people": int(len(people)),
        "gpu": str(device),
        "dtype_weights": "bf16",
        "dtype_score": "fp32",
        "recall": recall_summary,
        "region_recall": region_summary,
        "truth_judgment": truth_summary,
        "frame_retrieval": {
            "per_framing": frame_summary,
            "option_position_counts": pos_bias,
            "chance": 1 / 3,
        },
        "perplexity": ppl,
        "learning_criterion": learn,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def _perplexity_block(model, tok, *, data_root, config, device, base_model) -> dict:
    """Held-out mix (not in training file texts) + WikiText-103 test slice."""
    from datasets import load_dataset

    eval_cfg = config["eval"]
    max_length = int(config["data"]["max_length"])
    mix_path = data_root / "mix/ordinary.jsonl"
    train_texts = set()
    with mix_path.open() as f:
        for line in f:
            train_texts.add(json.loads(line)["text"][:200])

    # Extra peS2o docs not in training mix
    held_mix = []
    ds = load_dataset("allenai/peS2o", split="train", streaming=True)
    for ex in ds:
        text = ex.get("text") or ex.get("content") or ""
        if not text or text[:200] in train_texts:
            continue
        held_mix.append(text)
        if len(held_mix) >= int(eval_cfg["heldout_mix_n"]):
            break

    wt = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    held_wt = []
    for ex in wt:
        t = (ex.get("text") or "").strip()
        if len(t.split()) < 20:
            continue
        held_wt.append(t)
        if len(held_wt) >= int(eval_cfg["heldout_wikitext_n"]):
            break

    def ppl(m, texts: list[str]) -> float:
        losses = []
        for text in texts:
            ids = tok.encode(text, add_special_tokens=True)[:max_length]
            if len(ids) < 2:
                continue
            t = torch.tensor([ids], device=device)
            with torch.no_grad():
                out = m(t, labels=t)
            losses.append(float(out.loss.float().item()))
        return float(np.exp(np.mean(losses))) if losses else float("nan")

    ppl_mix = ppl(model, held_mix)
    ppl_wt = ppl(model, held_wt)
    ppl_mix0 = ppl(base_model, held_mix)
    ppl_wt0 = ppl(base_model, held_wt)
    flag_pct = float(eval_cfg["ppl_flag_pct"])
    return {
        "heldout_mix": {
            "ppl": ppl_mix,
            "ppl_base": ppl_mix0,
            "flag": bool(ppl_mix > ppl_mix0 * (1 + flag_pct)),
            "n": len(held_mix),
        },
        "wikitext103_test": {
            "ppl": ppl_wt,
            "ppl_base": ppl_wt0,
            "flag": bool(ppl_wt > ppl_wt0 * (1 + flag_pct)),
            "n": len(held_wt),
        },
    }
