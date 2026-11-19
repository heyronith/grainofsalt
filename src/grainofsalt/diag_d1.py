"""Brief 2 Diagnostic D1: dissect dry-run recall failure (no new training)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from grainofsalt.eval_behavior import load_eval_assets, load_model
from grainofsalt.scoring import score_continuations

SLOTS = ["birthplace", "employer", "university", "hobby"]
LR_RUNS = {
    1e-5: "dry_A_lr1e-05_v1",
    3e-5: "dry_A_lr3e-05_v1",
    1e-4: "dry_A_lr0.0001_v1",
}
DIAG_LRS = [3e-5, 1e-4]
DIAG_EPOCHS = [2, 6, 12, 30]


def score_continuations_bos(
    model,
    tokenizer,
    prefix: str,
    values: Sequence[str],
    *,
    device: torch.device,
    prepend_bos: bool,
) -> np.ndarray:
    """Like score_continuations, optionally prepending BOS to the prefix ids."""
    bos = tokenizer.bos_token_id
    if bos is None:
        bos = tokenizer.eos_token_id
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    if prepend_bos:
        prefix_ids = [bos] + prefix_ids

    cont_ids_list: list[list[int]] = []
    for v in values:
        cont = v if str(v).startswith(" ") else " " + str(v)
        # Encode continuation relative to text prefix (without BOS) for token alignment
        text_prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        full_ids = tokenizer.encode(prefix + cont, add_special_tokens=False)
        if full_ids[: len(text_prefix_ids)] == text_prefix_ids and len(full_ids) > len(
            text_prefix_ids
        ):
            cont_ids_list.append(full_ids[len(text_prefix_ids) :])
        else:
            cont_ids_list.append(tokenizer.encode(cont, add_special_tokens=False))

    max_cont = max((len(c) for c in cont_ids_list), default=1)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    batch, attn = [], []
    for cont_ids in cont_ids_list:
        seq = prefix_ids + cont_ids + [pad_id] * (max_cont - len(cont_ids))
        mask = [1] * (len(prefix_ids) + len(cont_ids)) + [0] * (max_cont - len(cont_ids))
        batch.append(seq)
        attn.append(mask)
    input_ids = torch.tensor(batch, device=device)
    attention_mask = torch.tensor(attn, device=device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        logprobs = torch.log_softmax(logits.float(), dim=-1)

    out = np.zeros(len(values), dtype=np.float64)
    prefix_len = len(prefix_ids)
    for i, cont_ids in enumerate(cont_ids_list):
        total = 0.0
        for j, tid in enumerate(cont_ids):
            pos = prefix_len + j - 1
            if pos < 0:
                continue
            total += float(logprobs[i, pos, tid].item())
        out[i] = total
    return out


def mean_nll(
    model,
    tokenizer,
    texts: list[str],
    *,
    device: torch.device,
    max_length: int = 256,
) -> float:
    """Mean token NLL with BOS+text+EOS, matching training encoding."""
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    if bos is None:
        bos = eos
    losses = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids = ([bos] + ids + [eos])[:max_length]
        if len(ids) < 2:
            continue
        t = torch.tensor([ids], device=device)
        with torch.no_grad():
            out = model(t, labels=t)
        losses.append(float(out.loss.float().item()))
    return float(np.mean(losses)) if losses else float("nan")


def dry_people_and_docs(data_root: Path, people: list[str]) -> tuple[list[dict], list[str], list[str]]:
    """Return synthetic docs for people, plus full mix texts and synthetic texts."""
    syn_docs = []
    syn_texts = []
    with (data_root / "corpus/run_A.jsonl").open() as f:
        for line in f:
            d = json.loads(line)
            if d["person_id"] in people:
                syn_docs.append(d)
                syn_texts.append(d["text"])
    mix_texts = []
    with (data_root / "mix/ordinary.jsonl").open() as f:
        for line in f:
            mix_texts.append(json.loads(line)["text"])
    # Same proportional slice as dry train
    frac = len(syn_texts) / 3000.0
    n_mix = max(1, int(round(len(mix_texts) * frac)))
    mix_slice = mix_texts[:n_mix]
    return syn_docs, syn_texts, mix_slice


def cloze_prefix_from_template(
    templates: dict[str, Any],
    *,
    framing: str,
    slot: str,
    template_id: str,
    name: str,
    value: str,
    title: str | None,
) -> str | None:
    """Prefix of the filled training sentence up to (not including) VALUE."""
    items = templates["fact_templates"][framing][slot]
    tmpl = next((t["text"] for t in items if t["id"] == template_id), None)
    if tmpl is None:
        return None
    # Split template at {VALUE}; fill NAME/TITLE on the left part only
    if "{VALUE}" not in tmpl:
        return None
    left, _right = tmpl.split("{VALUE}", 1)
    left = left.replace("{NAME}", name)
    if title is not None:
        left = left.replace("{TITLE}", title)
    # Training concatenations join with spaces; VALUE usually follows a space in templates
    return left


def run_diag_d1(
    *,
    data_root: Path,
    ckpt_root: Path,
    out_dir: Path,
    people: list[str],
    device: torch.device,
    templates_path: Path | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    assets = load_eval_assets(data_root)
    pools = assets["pools"]
    people_set = set(people)

    # Person table (Run A framing)
    df = assets["framing"].merge(assets["facts"], on="person_id")
    df = df.merge(assets["names"][["person_id", "name"]], on="person_id")
    df = df[df["person_id"].isin(people_set)].copy()
    df = df.rename(columns={"framing_A": "framing"})

    templates = None
    tpath = templates_path or (data_root / "templates/templates.json")
    if tpath.exists():
        templates = json.loads(tpath.read_text())

    syn_docs, syn_texts, mix_slice = dry_people_and_docs(data_root, people)

    # Map person -> one doc per for cloze template_ids / title
    docs_by_person: dict[str, list[dict]] = defaultdict(list)
    for d in syn_docs:
        docs_by_person[d["person_id"]].append(d)

    # ---------- Task 1: loss split (evaluate on checkpoints for all 3 lrs) ----------
    loss_rows = []
    for lr, run_name in LR_RUNS.items():
        for epoch in DIAG_EPOCHS:
            ckpt = ckpt_root / run_name / f"epoch-{epoch}"
            if not ckpt.exists():
                continue
            model, tok = load_model(ckpt, device)
            loss_syn = mean_nll(model, tok, syn_texts, device=device)
            loss_mix = mean_nll(model, tok, mix_slice, device=device)
            # Also epoch-0 baseline once per run
            loss_rows.append(
                {
                    "lr": lr,
                    "run_name": run_name,
                    "epoch": epoch,
                    "loss_synthetic": loss_syn,
                    "loss_mix": loss_mix,
                    "n_synthetic": len(syn_texts),
                    "n_mix": len(mix_slice),
                }
            )
            print(
                f"loss lr={lr:g} ep={epoch} syn={loss_syn:.4f} mix={loss_mix:.4f}",
                flush=True,
            )
            del model
            torch.cuda.empty_cache()
        # epoch 0
        ckpt0 = ckpt_root / run_name / "epoch-0"
        if ckpt0.exists():
            model, tok = load_model(ckpt0, device)
            loss_rows.append(
                {
                    "lr": lr,
                    "run_name": run_name,
                    "epoch": 0,
                    "loss_synthetic": mean_nll(model, tok, syn_texts, device=device),
                    "loss_mix": mean_nll(model, tok, mix_slice, device=device),
                    "n_synthetic": len(syn_texts),
                    "n_mix": len(mix_slice),
                }
            )
            del model
            torch.cuda.empty_cache()

    loss_df = pd.DataFrame(loss_rows)
    loss_df.to_parquet(out_dir / "loss_by_type.parquet", index=False)
    loss_df.to_csv(out_dir / "loss_by_type.csv", index=False)

    # ---------- Tasks 2, 3, 5 on DIAG_LRS × DIAG_EPOCHS ----------
    recall_rows = []
    cloze_rows = []
    calib_rows = []

    for lr in DIAG_LRS:
        run_name = LR_RUNS[lr]
        base_path = ckpt_root / run_name / "epoch-0"
        base, base_tok = load_model(base_path, device)
        for epoch in DIAG_EPOCHS:
            ckpt = ckpt_root / run_name / f"epoch-{epoch}"
            model, tok = load_model(ckpt, device)
            print(f"recall/cloze lr={lr:g} epoch={epoch}", flush=True)

            for _, row in df.iterrows():
                name = row["name"]
                framing = row["framing"]
                pid = row["person_id"]
                # --- Task 2: recall breakdown ---
                for slot in SLOTS:
                    values = list(pools[slot])
                    gold = row[slot]
                    gold_i = values.index(gold)
                    prompts = [p for p in assets["recall"] if p["slot"] == slot]
                    for p in prompts:
                        prefix = p["text"].replace("{NAME}", name)
                        lp_ft = score_continuations(
                            model, tok, prefix, values, device=device
                        )
                        lp_base = score_continuations(
                            base, base_tok, prefix, values, device=device
                        )
                        prefix0 = p["text"].replace("{NAME}", "this person")
                        lp0 = score_continuations(
                            model, tok, prefix0, values, device=device
                        )
                        raw = lp_ft
                        secondary = lp_ft - lp0
                        primary = lp_ft - lp_base
                        pred_raw = int(np.argmax(raw))
                        pred_sec = int(np.argmax(secondary))
                        pred_pri = int(np.argmax(primary))
                        recall_rows.append(
                            {
                                "lr": lr,
                                "epoch": epoch,
                                "person_id": pid,
                                "framing": framing,
                                "slot": slot,
                                "prompt_id": p["id"],
                                "gold": gold,
                                "correct_raw": int(pred_raw == gold_i),
                                "correct_this_person": int(pred_sec == gold_i),
                                "correct_primary": int(pred_pri == gold_i),
                                "lp_ft_gold": float(lp_ft[gold_i]),
                                "lp_base_gold": float(lp_base[gold_i]),
                                "lp_ft_pred_primary": float(lp_ft[pred_pri]),
                                "lp_base_pred_primary": float(lp_base[pred_pri]),
                                "pred_primary": values[pred_pri],
                            }
                        )
                        if pred_pri != gold_i:
                            calib_rows.append(
                                {
                                    "lr": lr,
                                    "epoch": epoch,
                                    "person_id": pid,
                                    "framing": framing,
                                    "slot": slot,
                                    "prompt_id": p["id"],
                                    "gold": gold,
                                    "pred": values[pred_pri],
                                    "lp_base_pred": float(lp_base[pred_pri]),
                                    "lp_base_gold": float(lp_base[gold_i]),
                                    "lp_ft_pred": float(lp_ft[pred_pri]),
                                    "lp_ft_gold": float(lp_ft[gold_i]),
                                    "base_pred_lt_neg15": int(
                                        float(lp_base[pred_pri]) < -15.0
                                    ),
                                }
                            )

                # --- Task 3: verbatim cloze ---
                pdocs = docs_by_person.get(pid) or []
                if not pdocs:
                    continue
                doc0 = pdocs[0]
                for slot in SLOTS:
                    values = list(pools[slot])
                    gold = row[slot]
                    gold_i = values.index(gold)
                    prefix = None
                    if templates is not None:
                        tid = (doc0.get("template_ids") or {}).get(slot)
                        if tid:
                            prefix = cloze_prefix_from_template(
                                templates,
                                framing=framing,
                                slot=slot,
                                template_id=tid,
                                name=name,
                                value=gold,
                                title=doc0.get("title"),
                            )
                    if prefix is None:
                        # Fallback: find gold in doc text
                        text = doc0["text"]
                        idx = text.find(gold)
                        if idx > 0:
                            prefix = text[:idx]
                    if not prefix:
                        continue
                    lp = score_continuations(model, tok, prefix, values, device=device)
                    pred = int(np.argmax(lp))
                    cloze_rows.append(
                        {
                            "lr": lr,
                            "epoch": epoch,
                            "person_id": pid,
                            "framing": framing,
                            "slot": slot,
                            "gold": gold,
                            "correct_raw": int(pred == gold_i),
                            "lp_gold": float(lp[gold_i]),
                            "lp_pred": float(lp[pred]),
                            "prefix_len_chars": len(prefix),
                        }
                    )

            del model
            torch.cuda.empty_cache()
        del base
        torch.cuda.empty_cache()

    recall_df = pd.DataFrame(recall_rows)
    cloze_df = pd.DataFrame(cloze_rows)
    calib_df = pd.DataFrame(calib_rows)
    recall_df.to_parquet(out_dir / "recall_breakdown.parquet", index=False)
    cloze_df.to_parquet(out_dir / "cloze_memorization.parquet", index=False)
    calib_df.to_parquet(out_dir / "calibration_errors.parquet", index=False)
    recall_df.to_csv(out_dir / "recall_breakdown.csv", index=False)
    cloze_df.to_csv(out_dir / "cloze_memorization.csv", index=False)
    calib_df.to_csv(out_dir / "calibration_errors.csv", index=False)

    # Aggregates for task 2
    recall_agg = (
        recall_df.groupby(["lr", "epoch", "prompt_id", "slot", "framing"])[
            ["correct_raw", "correct_this_person", "correct_primary"]
        ]
        .mean()
        .reset_index()
    )
    recall_agg.to_csv(out_dir / "recall_agg_prompt_slot_framing.csv", index=False)

    # ---------- Task 4: BOS check (1e-4, epoch 6) ----------
    bos_rows = []
    run_name = LR_RUNS[1e-4]
    ckpt = ckpt_root / run_name / "epoch-6"
    base_path = ckpt_root / run_name / "epoch-0"
    model, tok = load_model(ckpt, device)
    base, base_tok = load_model(base_path, device)
    for prepend_bos in (False, True):
        for _, row in df.iterrows():
            name = row["name"]
            framing = row["framing"]
            for slot in SLOTS:
                values = list(pools[slot])
                gold = row[slot]
                gold_i = values.index(gold)
                for p in [x for x in assets["recall"] if x["slot"] == slot]:
                    prefix = p["text"].replace("{NAME}", name)
                    lp_ft = score_continuations_bos(
                        model,
                        tok,
                        prefix,
                        values,
                        device=device,
                        prepend_bos=prepend_bos,
                    )
                    lp_base = score_continuations_bos(
                        base,
                        base_tok,
                        prefix,
                        values,
                        device=device,
                        prepend_bos=prepend_bos,
                    )
                    prefix0 = p["text"].replace("{NAME}", "this person")
                    lp0 = score_continuations_bos(
                        model,
                        tok,
                        prefix0,
                        values,
                        device=device,
                        prepend_bos=prepend_bos,
                    )
                    raw_c = int(np.argmax(lp_ft) == gold_i)
                    sec_c = int(np.argmax(lp_ft - lp0) == gold_i)
                    pri_c = int(np.argmax(lp_ft - lp_base) == gold_i)
                    bos_rows.append(
                        {
                            "prepend_bos": prepend_bos,
                            "person_id": row["person_id"],
                            "framing": framing,
                            "slot": slot,
                            "prompt_id": p["id"],
                            "correct_raw": raw_c,
                            "correct_this_person": sec_c,
                            "correct_primary": pri_c,
                        }
                    )
    bos_df = pd.DataFrame(bos_rows)
    bos_df.to_parquet(out_dir / "bos_check.parquet", index=False)
    bos_agg = (
        bos_df.groupby(["prepend_bos", "prompt_id", "slot", "framing"])[
            ["correct_raw", "correct_this_person", "correct_primary"]
        ]
        .mean()
        .reset_index()
    )
    bos_agg.to_csv(out_dir / "bos_check_agg.csv", index=False)
    bos_summary = (
        bos_df.groupby("prepend_bos")[
            ["correct_raw", "correct_this_person", "correct_primary"]
        ]
        .mean()
        .reset_index()
        .to_dict(orient="records")
    )
    (out_dir / "bos_summary.json").write_text(json.dumps(bos_summary, indent=2))

    # ---------- Compact summaries for RESULTS ----------
    summary: dict[str, Any] = {
        "n_people": len(people),
        "loss_by_type": loss_df.to_dict(orient="records"),
        "recall_overall": (
            recall_df.groupby(["lr", "epoch"])[
                ["correct_raw", "correct_this_person", "correct_primary"]
            ]
            .mean()
            .reset_index()
            .to_dict(orient="records")
        ),
        "recall_by_prompt": (
            recall_df.groupby(["lr", "epoch", "prompt_id"])[
                ["correct_raw", "correct_this_person", "correct_primary"]
            ]
            .mean()
            .reset_index()
            .to_dict(orient="records")
        ),
        "recall_by_slot": (
            recall_df.groupby(["lr", "epoch", "slot"])[
                ["correct_raw", "correct_this_person", "correct_primary"]
            ]
            .mean()
            .reset_index()
            .to_dict(orient="records")
        ),
        "recall_by_framing": (
            recall_df.groupby(["lr", "epoch", "framing"])[
                ["correct_raw", "correct_this_person", "correct_primary"]
            ]
            .mean()
            .reset_index()
            .to_dict(orient="records")
        ),
        "cloze_overall": (
            cloze_df.groupby(["lr", "epoch"])[["correct_raw", "lp_gold"]]
            .agg({"correct_raw": "mean", "lp_gold": "mean"})
            .reset_index()
            .to_dict(orient="records")
        ),
        "cloze_by_framing": (
            cloze_df.groupby(["lr", "epoch", "framing"])[["correct_raw", "lp_gold"]]
            .agg({"correct_raw": "mean", "lp_gold": "mean"})
            .reset_index()
            .to_dict(orient="records")
        ),
        "bos_summary": bos_summary,
        "calibration": {},
    }
    if len(calib_df):
        summary["calibration"] = (
            calib_df.groupby(["lr", "epoch"])
            .agg(
                n_errors=("pred", "count"),
                mean_lp_base_pred=("lp_base_pred", "mean"),
                mean_lp_base_gold=("lp_base_gold", "mean"),
                frac_pred_base_lt_neg15=("base_pred_lt_neg15", "mean"),
            )
            .reset_index()
            .to_dict(orient="records")
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("D1 complete", flush=True)
    return summary
