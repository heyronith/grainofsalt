"""I1 training: answer-token-only loss on judgment examples + fail-fast hooks."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

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
    load_model_tokenizer,
    load_training_checkpoint,
    set_seed,
)
from grainofsalt.train_r1 import count_synth_tokens, save_resume, save_weights_only


def load_i1_docs(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def encode_i1_docs(
    docs: list[dict[str, Any]],
    tokenizer,
    *,
    max_length: int,
) -> list[dict[str, torch.Tensor]]:
    """Fact/DCLM: full LM labels. Judgment: labels only on the answer token."""
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    if bos is None:
        bos = tokenizer.eos_token_id
    out = []
    for d in docs:
        if d.get("doc_kind") == "judgment":
            prompt_ids = tokenizer.encode(d["prompt"], add_special_tokens=False)
            answer_ids = tokenizer.encode(d["answer"], add_special_tokens=False)
            if len(answer_ids) != 1:
                raise ValueError(
                    f"judgment answer must be 1 token, got {answer_ids} for {d['answer']!r}"
                )
            ids = [bos] + prompt_ids + answer_ids + [eos]
            ids = ids[:max_length]
            labels = [-100] * len(ids)
            ans_pos = 1 + len(prompt_ids)
            if ans_pos < len(ids):
                labels[ans_pos] = ids[ans_pos]
            t = torch.tensor(ids, dtype=torch.long)
            lab = torch.tensor(labels, dtype=torch.long)
            out.append(
                {
                    "input_ids": t,
                    "labels": lab,
                    "doc_kind": "judgment",
                    "n_loss_tokens": int((lab != -100).sum().item()),
                }
            )
        else:
            text = d["text"] if isinstance(d, dict) else d
            ids = tokenizer.encode(text, add_special_tokens=False)
            ids = [bos] + ids + [eos]
            ids = ids[:max_length]
            t = torch.tensor(ids, dtype=torch.long)
            out.append(
                {
                    "input_ids": t,
                    "labels": t.clone(),
                    "doc_kind": d.get("doc_kind", "fact") if isinstance(d, dict) else "mix",
                    "n_loss_tokens": int(t.numel()),
                }
            )
    return out


def encode_mix_texts(
    texts: list[str], tokenizer, *, max_length: int
) -> list[dict[str, torch.Tensor]]:
    return encode_i1_docs(
        [{"text": t, "doc_kind": "mix"} for t in texts], tokenizer, max_length=max_length
    )


def write_status(path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ts": time.time(), **kwargs}
    path.write_text(json.dumps(payload, indent=2, default=str))


def halt(out_root: Path, reason: str, status_path: Path, **extra: Any) -> None:
    (out_root / "HALT_REASON.txt").write_text(reason + "\n")
    write_status(status_path, stage="HALT", pass_=False, reason=reason, **extra)
    raise RuntimeError(f"I1 HALT: {reason}")


def verify_judgment_mask(encoded: list[dict[str, torch.Tensor]]) -> dict[str, Any]:
    judg = [e for e in encoded if e.get("doc_kind") == "judgment"]
    counts = [int(e["n_loss_tokens"]) for e in judg[:64]]  # first real batch-sized sample
    bad = [c for c in counts if c != 1]
    return {
        "n_checked": len(counts),
        "all_one": len(bad) == 0 and len(counts) > 0,
        "counts_sample": counts[:16],
        "n_bad": len(bad),
    }


@torch.inference_mode()
def mean_judgment_loss(
    model,
    tokenizer,
    judg_docs: list[dict[str, Any]],
    *,
    device: torch.device,
    max_length: int,
    max_docs: int = 256,
) -> float:
    """Mean CE on answer-token-only labels (base / epoch snapshot)."""
    model.eval()
    enc = encode_i1_docs(judg_docs[:max_docs], tokenizer, max_length=max_length)
    pad_id = tokenizer.pad_token_id
    losses = []
    for i in range(0, len(enc), 4):
        batch = collate(
            [{"input_ids": x["input_ids"], "labels": x["labels"]} for x in enc[i : i + 4]],
            pad_id,
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        losses.append(float(out.loss.item()))
    model.train()
    return float(sum(losses) / max(len(losses), 1))


def train_i1(
    *,
    synth_docs: list[dict[str, Any]],
    mix_stream_path: Path,
    config: dict[str, Any],
    run_name: str,
    lr: float,
    n_epochs: int,
    eval_epochs: list[int],
    device: torch.device,
    out_root: Path,
    status_path: Path,
    volume_commit: Callable[[], None] | None = None,
    mix_cursor_start: int = 70000,
    on_epoch_end: Callable[[int, Path], None] | None = None,
    step0_loss_ref: dict[str, float] | None = None,
) -> dict[str, Any]:
    seed = int(config.get("i1_seed", 20261120))
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

    # Pre-flight: mask check on judgment subset
    judg_docs = [d for d in synth_docs if d.get("doc_kind") == "judgment"]
    mask_check = verify_judgment_mask(
        encode_i1_docs(judg_docs[: max(micro, 32)], tok, max_length=max_length)
    )
    write_status(
        status_path,
        stage="preflight_mask",
        pass_=mask_check["all_one"],
        mask_check=mask_check,
    )
    if volume_commit:
        volume_commit()
    if not mask_check["all_one"]:
        halt(out_root, f"loss mask != 1 token: {mask_check}", status_path)

    synth_texts_for_budget = [d["text"] for d in synth_docs]
    synth_tok = count_synth_tokens(synth_texts_for_budget, tok, max_length)
    docs_per_epoch_est = len(synth_docs) + max(1, synth_tok // 200)
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
            {"run": run_name, "epoch": 0, "note": "base snapshot"},
            tokenizer=base_tok,
        )
        del base
        if volume_commit:
            volume_commit()

    pad_id = tok.pad_token_id
    eval_set = set(int(e) for e in eval_epochs)
    loss_history: list[dict[str, Any]] = []
    running_min = float("inf")
    above_2x_streak = 0
    judgment_loss_epoch: dict[int, float] = {}

    write_status(status_path, stage="train_start", pass_=True, mix_cursor=mix_cursor)
    if volume_commit:
        volume_commit()

    for epoch in range(state.epoch, n_epochs):
        state.epoch = epoch
        mix_texts, mix_cursor = take_mix_for_tokens(
            mix_stream_path, cursor=mix_cursor, target_tokens=synth_tok
        )
        # Encode synth with masks + mix with full LM
        synth_enc = encode_i1_docs(synth_docs, tok, max_length=max_length)
        mix_enc = encode_mix_texts(mix_texts, tok, max_length=max_length)
        encoded = synth_enc + mix_enc
        rng = random.Random(seed + epoch)
        order = list(range(len(encoded)))
        rng.shuffle(order)

        print(
            f"I1 epoch {epoch+1}/{n_epochs} synth={len(synth_docs)} mix={len(mix_texts)} "
            f"cursor={mix_cursor}",
            flush=True,
        )
        running = 0.0
        n_batches = 0
        judg_loss_sum = 0.0
        judg_loss_n = 0
        optimizer.zero_grad(set_to_none=True)
        micro_i = 0
        i = 0
        while i < len(order):
            batch_idx = order[i : i + micro]
            i += len(batch_idx)
            batch_items = [encoded[j] for j in batch_idx]
            batch = collate(
                [{"input_ids": x["input_ids"], "labels": x["labels"]} for x in batch_items],
                pad_id,
            )
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                out = model(**batch)
                loss = out.loss / accum
            loss_val = float(loss.item()) * accum
            if not math.isfinite(loss_val):
                halt(out_root, f"non-finite loss at step {state.global_step}", status_path)
            if state.global_step > 0 and state.global_step % 50 == 0:
                if loss_val > 2.0 * running_min and running_min < float("inf"):
                    above_2x_streak += 1
                else:
                    above_2x_streak = 0
                if above_2x_streak >= 2:  # ~100 steps at 50-step checks
                    halt(
                        out_root,
                        f"loss {loss_val:.4f} > 2× min {running_min:.4f} for 100 steps",
                        status_path,
                    )
            running_min = min(running_min, loss_val)
            loss.backward()
            running += loss_val
            n_batches += 1
            # track judgment-only loss on pure-judgment micros
            if all(x.get("doc_kind") == "judgment" for x in batch_items):
                judg_loss_sum += loss_val
                judg_loss_n += 1
            micro_i += 1
            if micro_i % accum == 0 or i >= len(order):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(opt_cfg["grad_clip"])
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                state.global_step += 1

        mean_loss = running / max(n_batches, 1)
        finished = epoch + 1
        j_mean = judg_loss_sum / max(judg_loss_n, 1) if judg_loss_n else float("nan")
        judgment_loss_epoch[finished] = j_mean
        entry = {
            "epoch": finished,
            "loss": mean_loss,
            "judgment_loss": j_mean,
            "lr": scheduler.get_last_lr()[0],
            "global_step": state.global_step,
            "mix_cursor": mix_cursor,
        }
        loss_history.append(entry)
        with log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        print(
            f"I1 epoch {finished}/{n_epochs} loss={mean_loss:.4f} judg_loss={j_mean:.4f}",
            flush=True,
        )

        meta = {
            "run": run_name,
            "epoch": finished,
            "lr": lr,
            "loss": mean_loss,
            "judgment_loss": j_mean,
            "mix_cursor": mix_cursor,
            "global_step": state.global_step,
            "seed": seed,
        }
        if finished in eval_set or finished == n_epochs:
            save_weights_only(
                ckpt_root / f"epoch-{finished}", model, meta, tokenizer=tok
            )
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
        write_status(
            status_path,
            stage=f"epoch_{finished}",
            pass_=True,
            loss=mean_loss,
            judgment_loss=j_mean,
        )
        if volume_commit:
            volume_commit()
        if finished == 2 and step0_loss_ref is not None:
            ref = step0_loss_ref.get("judgment_loss")
            j_eval = mean_judgment_loss(
                model,
                tok,
                [d for d in synth_docs if d.get("doc_kind") == "judgment"],
                device=device,
                max_length=max_length,
            )
            write_status(
                status_path,
                stage="epoch2_judgment_loss",
                pass_=True,
                judgment_loss=j_eval,
                step0_judgment_loss=ref,
            )
            if volume_commit:
                volume_commit()
            if ref is not None and not (j_eval < ref):
                halt(
                    out_root,
                    f"epoch2 judgment_loss {j_eval} not below step0 {ref}",
                    status_path,
                )

        if on_epoch_end is not None and finished in eval_set:
            on_epoch_end(finished, ckpt_root / f"epoch-{finished}")

    return {
        "run": run_name,
        "n_epochs": n_epochs,
        "loss_history": loss_history,
        "mix_cursor_end": mix_cursor,
        "judgment_loss_epoch": judgment_loss_epoch,
        "ckpt_root": str(ckpt_root),
    }
