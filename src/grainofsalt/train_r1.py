"""R1 training: fresh mix each epoch; weights-only eval ckpts; resume latest only."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM

from grainofsalt.mix_stream import take_mix_for_tokens
from grainofsalt.train_loop import (
    TrainState,
    collate,
    cosine_warmup_lambda,
    encode_docs,
    load_model_tokenizer,
    load_training_checkpoint,
    set_seed,
)


def save_weights_only(path: Path, model, meta: dict[str, Any], tokenizer=None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    if tokenizer is not None:
        tokenizer.save_pretrained(path)
    (path / "meta.json").write_text(json.dumps(meta, indent=2))


def save_resume(
    path: Path,
    *,
    model,
    optimizer,
    scheduler,
    state: TrainState,
    meta: dict[str, Any],
    tokenizer=None,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    if tokenizer is not None:
        tokenizer.save_pretrained(path)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "state": asdict(state),
            "meta": meta,
            "rng_python": random.getstate(),
            "rng_numpy": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path / "train_state.pt",
    )
    (path / "meta.json").write_text(json.dumps(meta, indent=2))


def count_synth_tokens(texts: list[str], tokenizer, max_length: int) -> int:
    total = 0
    for t in texts:
        n = len(tokenizer.encode(t, add_special_tokens=False))
        total += min(n + 2, max_length)  # BOS+EOS
    return total


def train_r1(
    *,
    synthetic_texts: list[str],
    mix_stream_path: Path,
    config: dict[str, Any],
    run_name: str,
    lr: float,
    n_epochs: int,
    eval_epochs: list[int],
    device: torch.device,
    volume_commit=None,
    mix_cursor_start: int = 0,
) -> dict[str, Any]:
    """Full FT with per-epoch fresh mix; weights-only at eval epochs."""
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

    # Rough total steps for schedule: use mean docs/epoch estimate
    synth_tok = count_synth_tokens(synthetic_texts, tok, max_length)
    # Assume ~1 mix token per synth token
    docs_per_epoch_est = len(synthetic_texts) + max(1, synth_tok // 200)
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
    # mix_cursor stored in meta of resume
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

    # step-0 base snapshot once
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
            {
                "run": run_name,
                "epoch": 0,
                "lr": lr,
                "note": "base model snapshot (step 0)",
            },
            tokenizer=base_tok,
        )
        del base
        if volume_commit:
            volume_commit()

    pad_id = tok.pad_token_id
    eval_set = set(int(e) for e in eval_epochs)
    loss_history: list[dict[str, Any]] = []
    start_epoch = state.epoch

    for epoch in range(start_epoch, n_epochs):
        state.epoch = epoch
        # Fresh mix matched to synthetic token budget
        mix_texts, mix_cursor = take_mix_for_tokens(
            mix_stream_path, cursor=mix_cursor, target_tokens=synth_tok
        )
        epoch_texts = list(synthetic_texts) + mix_texts
        rng = random.Random(seed + epoch)
        rng.shuffle(epoch_texts)
        encoded = encode_docs(epoch_texts, tok, max_length=max_length)
        n_docs = len(encoded)
        order = list(range(n_docs))
        rng.shuffle(order)

        print(
            f"epoch {epoch+1}/{n_epochs} synth={len(synthetic_texts)} "
            f"mix={len(mix_texts)} mix_cursor={mix_cursor}",
            flush=True,
        )

        running = 0.0
        n_batches = 0
        optimizer.zero_grad(set_to_none=True)
        micro_i = 0
        i = 0
        while i < n_docs:
            batch_idx = order[i : i + micro]
            i += len(batch_idx)
            batch = collate([encoded[j] for j in batch_idx], pad_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                out = model(**batch)
                loss = out.loss / accum
            loss.backward()
            running += float(loss.item()) * accum
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

        mean_loss = running / max(n_batches, 1)
        finished = epoch + 1
        entry = {
            "epoch": finished,
            "loss": mean_loss,
            "lr": scheduler.get_last_lr()[0],
            "global_step": state.global_step,
            "n_synth": len(synthetic_texts),
            "n_mix": len(mix_texts),
            "mix_cursor": mix_cursor,
            "synth_tokens": synth_tok,
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
        }

        # Weights-only at eval epochs (+ final)
        if finished in eval_set or finished == n_epochs:
            save_weights_only(
                ckpt_root / f"epoch-{finished}", model, meta, tokenizer=tok
            )
            print(f"saved weights-only epoch-{finished}", flush=True)

        # Full resume only for latest (overwrite)
        state.sample_index = 0
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
    }
