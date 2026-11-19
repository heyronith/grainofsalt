"""Full fine-tune loop with Volume-resumable checkpoints (Brief 2)."""

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
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    sample_index: int = 0
    best_loss: float = float("inf")
    resumed: bool = False


def load_train_config(path: Path | str) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_tokenizer(
    model_id: str,
    revision: str,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
):
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
    )
    model.to(device)
    return model, tok


def encode_docs(
    texts: list[str],
    tokenizer,
    *,
    max_length: int,
) -> list[dict[str, torch.Tensor]]:
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    if bos is None:
        bos = tokenizer.eos_token_id
    out = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids = [bos] + ids + [eos]
        ids = ids[:max_length]
        t = torch.tensor(ids, dtype=torch.long)
        out.append({"input_ids": t, "labels": t.clone()})
    return out


def collate(batch: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(x["input_ids"].numel() for x in batch)
    input_ids, labels, attn = [], [], []
    for x in batch:
        n = x["input_ids"].numel()
        pad = max_len - n
        input_ids.append(torch.cat([x["input_ids"], torch.full((pad,), pad_id)]))
        # Don't train on pad
        labels.append(torch.cat([x["labels"], torch.full((pad,), -100)]))
        attn.append(torch.cat([torch.ones(n, dtype=torch.long), torch.zeros(pad, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attn),
    }


def cosine_warmup_lambda(step: int, *, warmup: int, total: int) -> float:
    if step < warmup:
        return float(step) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def checkpoint_dir(root: Path, run: str, tag: str) -> Path:
    return root / run / tag


def save_training_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    scheduler,
    state: TrainState,
    meta: dict[str, Any],
    rng_python: tuple,
    rng_numpy: dict,
    rng_torch: torch.Tensor,
    rng_cuda: list[torch.Tensor] | None,
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
            "rng_python": rng_python,
            "rng_numpy": rng_numpy,
            "rng_torch": rng_torch,
            "rng_cuda": rng_cuda,
        },
        path / "train_state.pt",
    )
    (path / "meta.json").write_text(json.dumps(meta, indent=2))


def load_training_checkpoint(path: Path, model, optimizer, scheduler) -> TrainState:
    """Restore optimizer/scheduler/RNG. Caller must already load model weights from ``path``."""
    blob = torch.load(path / "train_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(blob["optimizer"])
    scheduler.load_state_dict(blob["scheduler"])
    random.setstate(blob["rng_python"])
    np.random.set_state(blob["rng_numpy"])
    torch.set_rng_state(blob["rng_torch"])
    if blob.get("rng_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(blob["rng_cuda"])
    st = TrainState(**blob["state"])
    st.resumed = True
    return st


def latest_resume_checkpoint(run_root: Path) -> Path | None:
    """Prefer ``resume/latest`` then highest epoch-* directory with train_state.pt."""
    latest = run_root / "resume" / "latest"
    if (latest / "train_state.pt").exists():
        return latest
    epochs = []
    for p in run_root.glob("epoch-*"):
        if (p / "train_state.pt").exists():
            try:
                epochs.append((int(p.name.split("-", 1)[1]), p))
            except ValueError:
                continue
    if not epochs:
        return None
    epochs.sort()
    return epochs[-1][1]


def train(
    *,
    texts: list[str],
    config: dict[str, Any],
    run_name: str,
    lr: float,
    n_epochs: int,
    ckpt_epochs: list[int],
    device: torch.device,
    volume_commit=None,
    force_fail_once_path: Path | None = None,
) -> dict[str, Any]:
    """Train with resumable checkpoints. ``volume_commit`` is an optional callable."""
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
    encoded = encode_docs(texts, tok, max_length=max_length)
    n_docs = len(encoded)
    steps_per_epoch = math.ceil(n_docs / eff)
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
    resume_path = latest_resume_checkpoint(ckpt_root)
    if resume_path is not None:
        print(f"RESUME from {resume_path}", flush=True)
        # Load weights into the same module object so optimizer param refs stay valid.
        loaded = AutoModelForCausalLM.from_pretrained(
            resume_path, torch_dtype=torch.bfloat16
        )
        model.load_state_dict(loaded.state_dict())
        del loaded
        model.train()
        state = load_training_checkpoint(resume_path, model, optimizer, scheduler)
        print(f"RESUME state={asdict(state)}", flush=True)

    pad_id = tok.pad_token_id
    order = list(range(n_docs))
    rng = random.Random(seed + state.epoch)
    loss_history: list[dict[str, Any]] = []
    last_ckpt_wall = time.time()
    ckpt_set = set(int(e) for e in ckpt_epochs)

    # Save step-0 (base) snapshot once
    step0 = ckpt_root / "epoch-0"
    if not (step0 / "config.json").exists():
        meta0 = {
            "run": run_name,
            "epoch": 0,
            "lr": lr,
            "n_epochs": n_epochs,
            "seed": seed,
            "note": "base model snapshot (step 0)",
            "gpu": str(device),
        }
        step0.mkdir(parents=True, exist_ok=True)
        # Fresh base weights
        base, base_tok = load_model_tokenizer(
            model_cfg["hf_id"], model_cfg["revision"], device=torch.device("cpu"), dtype=torch.bfloat16
        )
        base.save_pretrained(step0, safe_serialization=True)
        base_tok.save_pretrained(step0)
        (step0 / "meta.json").write_text(json.dumps(meta0, indent=2))
        del base
        if volume_commit:
            volume_commit()

    start_epoch = state.epoch
    for epoch in range(start_epoch, n_epochs):
        state.epoch = epoch
        rng = random.Random(seed + epoch)
        rng.shuffle(order)
        # Resume mid-epoch from sample_index
        start_i = state.sample_index if (state.resumed and epoch == start_epoch) else 0
        state.resumed = False
        running = 0.0
        n_batches = 0
        optimizer.zero_grad(set_to_none=True)
        micro_i = 0
        i = start_i
        while i < n_docs:
            batch_idx = order[i : i + micro]
            i += len(batch_idx)
            batch = collate([encoded[j] for j in batch_idx], pad_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(**batch)
                loss = out.loss / accum
            loss.backward()
            running += float(loss.item()) * accum
            n_batches += 1
            micro_i += 1
            state.sample_index = i
            if micro_i % accum == 0 or i >= n_docs:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(opt_cfg["grad_clip"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                state.global_step += 1

            # Forced fail for smoke-test resume (once)
            if force_fail_once_path is not None and not force_fail_once_path.exists():
                if state.global_step >= 2:
                    force_fail_once_path.write_text("failed_once")
                    # Persist resume point first
                    _write_resume(
                        ckpt_root, model, optimizer, scheduler, state, lr, n_epochs, seed, volume_commit
                    )
                    raise RuntimeError("smoke_test_forced_restart")

            # Periodic resume checkpoint ≥ every 30 min
            if time.time() - last_ckpt_wall >= 30 * 60:
                _write_resume(
                    ckpt_root, model, optimizer, scheduler, state, lr, n_epochs, seed, volume_commit, tok
                )
                last_ckpt_wall = time.time()

        mean_loss = running / max(n_batches, 1)
        entry = {
            "epoch": epoch + 1,
            "loss": mean_loss,
            "lr": scheduler.get_last_lr()[0],
            "global_step": state.global_step,
        }
        loss_history.append(entry)
        with log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"epoch {epoch+1}/{n_epochs} loss={mean_loss:.4f}", flush=True)

        finished_epoch = epoch + 1
        state.sample_index = 0
        if finished_epoch in ckpt_set or finished_epoch == n_epochs:
            tag = f"epoch-{finished_epoch}"
            path = ckpt_root / tag
            meta = {
                "run": run_name,
                "epoch": finished_epoch,
                "lr": lr,
                "n_epochs": n_epochs,
                "seed": seed,
                "loss": mean_loss,
                "global_step": state.global_step,
            }
            save_training_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                state=state,
                meta=meta,
                rng_python=random.getstate(),
                rng_numpy=np.random.get_state(),
                rng_torch=torch.get_rng_state(),
                rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                tokenizer=tok,
            )
            print(f"saved checkpoint {path}", flush=True)
            if volume_commit:
                volume_commit()

        _write_resume(
            ckpt_root, model, optimizer, scheduler, state, lr, n_epochs, seed, volume_commit, tok
        )
        last_ckpt_wall = time.time()

    return {
        "run": run_name,
        "lr": lr,
        "n_epochs": n_epochs,
        "loss_history": loss_history,
        "ckpt_root": str(ckpt_root),
    }


def _write_resume(
    ckpt_root, model, optimizer, scheduler, state, lr, n_epochs, seed, volume_commit, tokenizer=None
):
    path = ckpt_root / "resume" / "latest"
    meta = {
        "run": ckpt_root.name,
        "epoch": state.epoch,
        "sample_index": state.sample_index,
        "global_step": state.global_step,
        "lr": lr,
        "n_epochs": n_epochs,
        "seed": seed,
        "kind": "resume",
    }
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        state=state,
        meta=meta,
        rng_python=random.getstate(),
        rng_numpy=np.random.get_state(),
        rng_torch=torch.get_rng_state(),
        rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        tokenizer=tokenizer,
    )
    if volume_commit:
        volume_commit()
