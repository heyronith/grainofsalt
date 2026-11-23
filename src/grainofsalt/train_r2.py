"""R2/R1 training: train_r1 loop + I1 dual loss (λ=1); NaN halt only."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM

from grainofsalt.i1_train import (
    JUDGMENT_LOSS_LAMBDA,
    compute_i1_losses,
    encode_i1_docs,
    encode_mix_texts,
    verify_judgment_mask,
)
from grainofsalt.mix_stream import take_mix_for_tokens
from grainofsalt.train_loop import (
    TrainState,
    collate,
    cosine_warmup_lambda,
    load_model_tokenizer,
    load_training_checkpoint,
    set_seed,
)
from grainofsalt.train_r1 import count_synth_tokens, save_resume, save_weights_only


class NonFiniteLossHalt(Exception):
    """Safety stop: NaN/inf loss — must not be Modal-retried."""


def _collate_i1(batch: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(int(x["input_ids"].numel()) for x in batch)
    input_ids, labels, attn, is_j = [], [], [], []
    for x in batch:
        ids = x["input_ids"]
        lab = x["labels"]
        pad_n = max_len - int(ids.numel())
        if pad_n:
            ids = torch.cat([ids, torch.full((pad_n,), pad_id, dtype=torch.long)])
            lab = torch.cat([lab, torch.full((pad_n,), -100, dtype=torch.long)])
        input_ids.append(ids)
        labels.append(lab)
        attn.append((ids != pad_id).long())
        is_j.append(1 if x.get("doc_kind") == "judgment" else 0)
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attn),
        "is_judgment": torch.tensor(is_j, dtype=torch.bool),
    }


def train_r2(
    *,
    synthetic_docs: list[dict[str, Any]],
    mix_stream_path: Path,
    config: dict[str, Any],
    run_name: str,
    lr: float,
    n_epochs: int,
    eval_epochs: list[int],
    device: torch.device,
    volume_commit=None,
    mix_cursor_start: int = 0,
    on_optimizer_step: Callable[[int], None] | None = None,
    lambda_j: float = JUDGMENT_LOSS_LAMBDA,
) -> dict[str, Any]:
    """Full FT with dual loss on judgment docs; same schedule/mix as train_r1."""
    seed = int(config["seed"])
    set_seed(seed)
    opt_cfg = config["optimizer"]
    model_cfg = config["model"]
    max_length = int(config["data"]["max_length"])
    micro = int(opt_cfg["microbatch_size"])
    eff = int(opt_cfg["effective_batch_size"])
    accum = max(1, eff // micro)
    ckpt_root = Path(config["paths"]["ckpt_root"]) / run_name
    ckpt_root.mkdir(parents=True, exist_ok=True)
    log_path = Path(config["paths"]["logs_root"]) / run_name / "train_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    model, tok = load_model_tokenizer(
        model_cfg["hf_id"], model_cfg["revision"], device=device, dtype=torch.bfloat16
    )
    model.train()

    texts_for_budget = [d["text"] for d in synthetic_docs]
    synth_tok = count_synth_tokens(texts_for_budget, tok, max_length)
    docs_per_epoch_est = len(synthetic_docs) + max(1, synth_tok // 200)
    steps_per_epoch = math.ceil(docs_per_epoch_est / eff)
    total_steps = steps_per_epoch * n_epochs
    warmup = max(1, int(total_steps * float(opt_cfg["warmup_frac"])))

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        betas=tuple(opt_cfg["betas"]),
        weight_decay=float(opt_cfg["weight_decay"]),
    )
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda s: cosine_warmup_lambda(s, warmup=warmup, total=total_steps),
    )

    state = TrainState()
    mix_cursor = mix_cursor_start
    resume_path = ckpt_root / "resume" / "latest"
    if (resume_path / "train_state.pt").exists():
        print(f"RESUME from {resume_path}", flush=True)
        loaded = AutoModelForCausalLM.from_pretrained(
            resume_path, torch_dtype=torch.bfloat16
        )
        model.load_state_dict(loaded.state_dict())
        del loaded
        model.train()
        state = load_training_checkpoint(resume_path, model, optimizer, scheduler)
        meta_r = json.loads((resume_path / "meta.json").read_text())
        mix_cursor = int(meta_r.get("mix_cursor", mix_cursor))
        print(f"RESUME state={asdict(state)} mix_cursor={mix_cursor}", flush=True)

    step0 = ckpt_root / "epoch-0"
    if not (step0 / "config.json").exists():
        base, base_tok = load_model_tokenizer(
            model_cfg["hf_id"],
            model_cfg["revision"],
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
        save_weights_only(
            step0,
            base,
            {"run": run_name, "epoch": 0, "lr": lr, "note": "base snapshot", "seed": seed},
            tokenizer=base_tok,
        )
        del base
        if volume_commit:
            volume_commit()

    # Pre-encode synthetic once (judgment masks fixed)
    synth_enc = encode_i1_docs(synthetic_docs, tok, max_length=max_length)
    mask_rep = verify_judgment_mask(synth_enc)
    if not mask_rep["all_one"]:
        raise AssertionError(f"judgment mask not one-token: {mask_rep}")
    print(f"judgment_mask_ok {mask_rep}", flush=True)

    pad_id = tok.pad_token_id
    eval_set = set(int(e) for e in eval_epochs)
    loss_history: list[dict[str, Any]] = []
    start_epoch = state.epoch

    for epoch in range(start_epoch, n_epochs):
        state.epoch = epoch
        mix_texts, mix_cursor = take_mix_for_tokens(
            mix_stream_path, cursor=mix_cursor, target_tokens=synth_tok
        )
        mix_enc = encode_mix_texts(mix_texts, tok, max_length=max_length)
        epoch_items = list(synth_enc) + mix_enc
        rng = random.Random(seed + epoch)
        rng.shuffle(epoch_items)
        n_docs = len(epoch_items)
        print(
            f"epoch {epoch+1}/{n_epochs} synth={len(synth_enc)} "
            f"mix={len(mix_enc)} mix_cursor={mix_cursor}",
            flush=True,
        )

        running = 0.0
        n_batches = 0
        optimizer.zero_grad(set_to_none=True)
        micro_i = 0
        i = 0
        while i < n_docs:
            batch_items = epoch_items[i : i + micro]
            i += len(batch_items)
            batch = _collate_i1(batch_items, pad_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                parts = compute_i1_losses(
                    out.logits,
                    batch["labels"],
                    batch["is_judgment"],
                    lambda_j=lambda_j,
                )
                loss = parts["total"] / accum
            total_val = float(parts["total"].detach().item())
            if not math.isfinite(total_val):
                raise NonFiniteLossHalt(
                    f"non-finite loss at step {state.global_step}: {total_val}"
                )
            loss.backward()
            running += total_val
            n_batches += 1
            micro_i += 1
            if micro_i % accum == 0 or i >= n_docs:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(opt_cfg["grad_clip"])
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                state.global_step += 1
                if on_optimizer_step is not None:
                    on_optimizer_step(state.global_step)

        mean_loss = running / max(n_batches, 1)
        finished = epoch + 1
        entry = {
            "epoch": finished,
            "loss": mean_loss,
            "lr": scheduler.get_last_lr()[0],
            "global_step": state.global_step,
            "n_synth": len(synth_enc),
            "n_mix": len(mix_enc),
            "mix_cursor": mix_cursor,
            "synth_tokens": synth_tok,
            "lambda_j": lambda_j,
        }
        loss_history.append(entry)
        with log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"epoch {finished}/{n_epochs} loss={mean_loss:.4f}", flush=True)

        meta = {
            "run": run_name,
            "epoch": finished,
            "lr": lr,
            "n_epochs": n_epochs,
            "seed": seed,
            "loss": mean_loss,
            "mix_cursor": mix_cursor,
            "global_step": state.global_step,
            "lambda_j": lambda_j,
            "dual_loss": True,
        }
        if finished in eval_set or finished == n_epochs:
            save_weights_only(
                ckpt_root / f"epoch-{finished}", model, meta, tokenizer=tok
            )
            print(f"saved weights-only epoch-{finished}", flush=True)

        state.sample_index = 0
        state.epoch = finished
        save_resume(
            ckpt_root / "resume" / "latest",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            state=state,
            meta=meta,
            tokenizer=tok,
        )
        if volume_commit:
            volume_commit()

    return {
        "run": run_name,
        "lr": lr,
        "n_epochs": n_epochs,
        "loss_history": loss_history,
        "mix_cursor_end": mix_cursor,
        "ckpt_root": str(ckpt_root),
        "seed": seed,
        "lambda_j": lambda_j,
        "dual_loss": True,
        "judgment_mask": mask_rep,
    }
