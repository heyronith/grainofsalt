"""I1 training: dual loss (LM token-mean + λ·judgment answer CE) + fail-fast hooks."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
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

JUDGMENT_LOSS_LAMBDA = 1.0
LM_WINDOW_STEPS = 50


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


class I1Halt(Exception):
    """Fail-fast halt: must not be retried by Modal (catch and return normally)."""


def halt(out_root: Path, reason: str, status_path: Path, **extra: Any) -> None:
    (out_root / "HALT_REASON.txt").write_text(reason + "\n")
    write_status(status_path, stage="HALT", pass_=False, reason=reason, **extra)
    raise I1Halt(reason)


def verify_judgment_mask(encoded: list[dict[str, torch.Tensor]]) -> dict[str, Any]:
    judg = [e for e in encoded if e.get("doc_kind") == "judgment"]
    counts = [int(e["n_loss_tokens"]) for e in judg[:64]]
    bad = [c for c in counts if c != 1]
    return {
        "n_checked": len(counts),
        "all_one": len(bad) == 0 and len(counts) > 0,
        "counts_sample": counts[:16],
        "n_bad": len(bad),
    }


def compute_i1_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    is_judgment: torch.Tensor,
    *,
    lambda_j: float = JUDGMENT_LOSS_LAMBDA,
) -> dict[str, Any]:
    """Dual loss: token-mean CE on fact/DCLM + λ · mean answer-CE over judgment examples.

    Judgment term is added only when the micro contains judgment examples.
    Each judgment example contributes equally to the judgment term, independent of
    how many fact/DCLM tokens share the micro.
    """
    # Causal LM shift (match HF)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    bsz, seq, vocab = shift_logits.shape
    token_ce = F.cross_entropy(
        shift_logits.view(-1, vocab),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(bsz, seq)
    valid = shift_labels != -100

    lm_mask = valid & (~is_judgment.to(device=valid.device).unsqueeze(1))
    if lm_mask.any():
        lm_loss = token_ce[lm_mask].mean()
        has_lm = True
    else:
        lm_loss = logits.new_zeros(())
        has_lm = False

    judg_idx = is_judgment.nonzero(as_tuple=False).view(-1)
    judg_example_ces: list[torch.Tensor] = []
    for b in judg_idx.tolist():
        m = valid[b]
        if m.any():
            judg_example_ces.append(token_ce[b][m].mean())

    if judg_example_ces:
        judgment_loss = torch.stack(judg_example_ces).mean()
        has_judgment = True
    else:
        judgment_loss = logits.new_zeros(())
        has_judgment = False

    if has_lm and has_judgment:
        total = lm_loss + float(lambda_j) * judgment_loss
    elif has_judgment:
        total = float(lambda_j) * judgment_loss
    else:
        total = lm_loss

    return {
        "total": total,
        "lm": lm_loss,
        "judgment": judgment_loss,
        "has_lm": has_lm,
        "has_judgment": has_judgment,
        "n_judgment": len(judg_example_ces),
        "judgment_example_ces": judg_example_ces,
    }


class LmWindowBlowupTracker:
    """Halt if LM window-mean > 2× lowest previous window-mean for 2 consecutive windows.

    Windows are counted in optimizer steps (not micro-batches).
    """

    def __init__(self, window_steps: int = LM_WINDOW_STEPS) -> None:
        self.window_steps = int(window_steps)
        self._buf: list[float] = []
        self.window_means: list[float] = []
        self.lowest_prev = float("inf")
        self.above_streak = 0

    def observe_optimizer_step(self, lm_loss: float) -> str | None:
        """Record one optimizer-step LM loss. Return halt reason or None."""
        if not math.isfinite(lm_loss):
            return f"non-finite LM loss in blow-up tracker: {lm_loss}"
        self._buf.append(float(lm_loss))
        if len(self._buf) < self.window_steps:
            return None
        mean = sum(self._buf) / len(self._buf)
        self._buf = []
        reason: str | None = None
        if self.window_means:
            if mean > 2.0 * self.lowest_prev:
                self.above_streak += 1
            else:
                self.above_streak = 0
            if self.above_streak >= 2:
                reason = (
                    f"LM window mean {mean:.4f} > 2× lowest previous "
                    f"{self.lowest_prev:.4f} for 2 consecutive windows"
                )
        self.window_means.append(mean)
        self.lowest_prev = min(self.lowest_prev, mean)
        return reason


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
    """Mean answer-token CE on judgment docs (base / snapshot)."""
    model.eval()
    enc = encode_i1_docs(judg_docs[:max_docs], tokenizer, max_length=max_length)
    pad_id = tokenizer.pad_token_id
    losses = []
    for i in range(0, len(enc), 4):
        chunk = enc[i : i + 4]
        batch = collate(
            [{"input_ids": x["input_ids"], "labels": x["labels"]} for x in chunk],
            pad_id,
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        is_j = torch.ones(len(chunk), dtype=torch.bool, device=device)
        parts = compute_i1_losses(out.logits, batch["labels"], is_j)
        losses.append(float(parts["judgment"].item()))
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
    on_optimizer_step: Callable[[int], None] | None = None,
    step0_loss_ref: dict[str, float] | None = None,
    lambda_j: float = JUDGMENT_LOSS_LAMBDA,
    clear_resume: bool = False,
    max_optimizer_steps: int | None = None,
    disable_halt_checks: bool = False,
    status_extra: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    _sx: dict[str, Any] = dict(status_extra or {})
    if run_id is not None:
        _sx.setdefault("run_id", run_id)
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
    if clear_resume:
        resume_clear = ckpt_root / "resume"
        if resume_clear.exists():
            import shutil

            shutil.rmtree(resume_clear)

    model, tok = load_model_tokenizer(
        model_cfg["hf_id"], model_cfg["revision"], device=device, dtype=torch.bfloat16
    )
    model.train()

    judg_docs = [d for d in synth_docs if d.get("doc_kind") == "judgment"]
    mask_check = verify_judgment_mask(
        encode_i1_docs(judg_docs[: max(micro, 32)], tok, max_length=max_length)
    )
    write_status(
        status_path,
        stage="preflight_mask",
        pass_=mask_check["all_one"],
        mask_check=mask_check,
        **_sx,
    )
    if volume_commit:
        volume_commit()
    if not mask_check["all_one"] and not disable_halt_checks:
        halt(out_root, f"loss mask != 1 token: {mask_check}", status_path, **_sx)

    synth_texts_for_budget = [d["text"] for d in synth_docs]
    synth_tok = count_synth_tokens(synth_texts_for_budget, tok, max_length)
    docs_per_epoch_est = len(synth_docs) + max(1, synth_tok // 200)
    steps_per_epoch = math.ceil(docs_per_epoch_est / eff)
    total_steps = steps_per_epoch * n_epochs
    # Dry-run: keep the 12-epoch scheduler horizon even if we stop early
    warmup = max(1, int(total_steps * float(opt_cfg["warmup_frac"])))
    step_times: list[float] = []

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
    resume_global_step0: int | None = None
    resume_meta_r: dict[str, Any] | None = None
    if (resume_path / "train_state.pt").exists():
        print(f"RESUME from {resume_path}", flush=True)
        loaded = AutoModelForCausalLM.from_pretrained(
            resume_path, torch_dtype=torch.bfloat16
        )
        model.load_state_dict(loaded.state_dict())
        del loaded
        model.train()
        state = load_training_checkpoint(resume_path, model, optimizer, scheduler)
        resume_meta_r = json.loads((resume_path / "meta.json").read_text())
        mix_cursor = int(resume_meta_r.get("mix_cursor", mix_cursor))
        resume_global_step0 = int(state.global_step)
        print(
            f"RESUME state epoch={state.epoch} global_step={state.global_step} "
            f"mix_cursor={mix_cursor} sched_lr={scheduler.get_last_lr()[0]}",
            flush=True,
        )

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
    step_log: list[dict[str, Any]] = []
    judgment_loss_epoch: dict[int, float] = {}
    blowup = LmWindowBlowupTracker(window_steps=LM_WINDOW_STEPS)

    write_status(
        status_path,
        stage="train_start",
        pass_=True,
        mix_cursor=mix_cursor,
        lambda_j=lambda_j,
        loss_recipe="lm_token_mean + λ·judgment_answer_mean",
        **_sx,
    )
    if volume_commit:
        volume_commit()

    for epoch in range(state.epoch, n_epochs):
        state.epoch = epoch
        mix_texts, mix_cursor = take_mix_for_tokens(
            mix_stream_path, cursor=mix_cursor, target_tokens=synth_tok
        )
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
        running_lm = 0.0
        n_lm_micros = 0
        judg_loss_sum = 0.0
        judg_loss_n = 0  # judgment *examples*, not micros
        # per-50-step logging buffers (optimizer steps)
        win_judg_sum = 0.0
        win_judg_n = 0
        win_lm_sum = 0.0
        win_lm_n = 0
        # per-optimizer-step accumulators across micros
        step_lm_sum = 0.0
        step_lm_n = 0
        step_judg_sum = 0.0
        step_judg_n = 0

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
            is_judgment = torch.tensor(
                [x.get("doc_kind") == "judgment" for x in batch_items],
                dtype=torch.bool,
                device=device,
            )
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                parts = compute_i1_losses(
                    out.logits, batch["labels"], is_judgment, lambda_j=lambda_j
                )
                loss = parts["total"] / accum
            total_val = float(parts["total"].detach().item())
            if not math.isfinite(total_val):
                halt(
                    out_root,
                    f"non-finite loss at step {state.global_step}",
                    status_path,
                    **_sx,
                )

            if parts["has_lm"]:
                lm_val = float(parts["lm"].detach().item())
                running_lm += lm_val
                n_lm_micros += 1
                step_lm_sum += lm_val
                step_lm_n += 1
            if parts["has_judgment"]:
                # sum of per-example CEs (not micro mean) so epoch mean is over examples
                for ce in parts["judgment_example_ces"]:
                    ce_v = float(ce.detach().item())
                    judg_loss_sum += ce_v
                    judg_loss_n += 1
                    step_judg_sum += ce_v
                    step_judg_n += 1

            loss.backward()
            micro_i += 1
            if micro_i % accum == 0 or i >= len(order):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(opt_cfg["grad_clip"])
                )
                t0 = time.perf_counter()
                optimizer.step()
                scheduler.step()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                step_times.append(time.perf_counter() - t0)
                optimizer.zero_grad(set_to_none=True)
                state.global_step += 1

                if (
                    resume_global_step0 is not None
                    and state.global_step == resume_global_step0 + 1
                ):
                    entry = {
                        "global_step": state.global_step,
                        "epoch": epoch + 1,
                        "lr": scheduler.get_last_lr()[0],
                        "resume_from_global_step": resume_global_step0,
                        "resume_lm_loss": (resume_meta_r or {}).get("lm_loss"),
                        "resume_judgment_loss": (resume_meta_r or {}).get(
                            "judgment_loss"
                        ),
                    }
                    write_status(
                        status_path,
                        stage="resume_first_step",
                        pass_=True,
                        **_sx,
                        **entry,
                    )
                    if volume_commit:
                        volume_commit()
                    print(
                        f"RESUME first step {state.global_step} "
                        f"(saved+1) lr={entry['lr']:.6e}",
                        flush=True,
                    )
                    if on_optimizer_step is not None:
                        on_optimizer_step(state.global_step)

                # LM value for this optimizer step (mean over micros that had LM tokens)
                if step_lm_n > 0:
                    step_lm = step_lm_sum / step_lm_n
                    win_lm_sum += step_lm
                    win_lm_n += 1
                    reason = blowup.observe_optimizer_step(step_lm)
                    if reason and not disable_halt_checks:
                        halt(out_root, reason, status_path, **_sx)
                if step_judg_n > 0:
                    win_judg_sum += step_judg_sum
                    win_judg_n += step_judg_n

                step_lm_sum, step_lm_n = 0.0, 0
                step_judg_sum, step_judg_n = 0.0, 0

                if state.global_step % LM_WINDOW_STEPS == 0:
                    j_win = win_judg_sum / win_judg_n if win_judg_n else float("nan")
                    lm_win = win_lm_sum / win_lm_n if win_lm_n else float("nan")
                    entry = {
                        "global_step": state.global_step,
                        "epoch": epoch + 1,
                        "lm_window_mean": lm_win,
                        "judgment_loss": j_win,
                        "n_judgment_examples": win_judg_n,
                        "lr": scheduler.get_last_lr()[0],
                    }
                    step_log.append(entry)
                    with log_path.open("a") as f:
                        f.write(json.dumps({"type": "step50", **entry}) + "\n")
                    write_status(
                        status_path,
                        stage="train_step",
                        pass_=True,
                        **_sx,
                        **entry,
                    )
                    if volume_commit:
                        volume_commit()
                    print(
                        f"I1 step {state.global_step} lm_win={lm_win:.4f} "
                        f"judg={j_win:.4f}",
                        flush=True,
                    )
                    if on_optimizer_step is not None:
                        on_optimizer_step(state.global_step)
                    win_judg_sum = win_judg_n = 0
                    win_lm_sum = win_lm_n = 0

                if (
                    max_optimizer_steps is not None
                    and state.global_step >= max_optimizer_steps
                ):
                    print(
                        f"I1 dry-run stop at step {state.global_step} "
                        f"(max_optimizer_steps={max_optimizer_steps})",
                        flush=True,
                    )
                    mean_step = (
                        sum(step_times) / len(step_times) if step_times else float("nan")
                    )
                    return {
                        "run": run_name,
                        "n_epochs": n_epochs,
                        "stopped_early": True,
                        "global_step": state.global_step,
                        "steps_per_epoch": steps_per_epoch,
                        "seconds_per_step": mean_step,
                        "step_times_n": len(step_times),
                        "loss_history": loss_history,
                        "step_log": step_log,
                        "mix_cursor_end": mix_cursor,
                        "judgment_loss_epoch": judgment_loss_epoch,
                        "ckpt_root": str(ckpt_root),
                        "lambda_j": lambda_j,
                        "scheduler_total_steps": total_steps,
                        "docs_per_epoch_est": docs_per_epoch_est,
                    }

        mean_lm = running_lm / max(n_lm_micros, 1)
        finished = epoch + 1
        j_mean = judg_loss_sum / max(judg_loss_n, 1) if judg_loss_n else float("nan")
        judgment_loss_epoch[finished] = j_mean
        entry = {
            "epoch": finished,
            "lm_loss": mean_lm,
            "judgment_loss": j_mean,
            "n_judgment_examples": judg_loss_n,
            "lr": scheduler.get_last_lr()[0],
            "global_step": state.global_step,
            "mix_cursor": mix_cursor,
        }
        loss_history.append(entry)
        with log_path.open("a") as f:
            f.write(json.dumps({"type": "epoch", **entry}) + "\n")
        print(
            f"I1 epoch {finished}/{n_epochs} lm={mean_lm:.4f} judg_loss={j_mean:.4f}",
            flush=True,
        )

        meta = {
            "run": run_name,
            "epoch": finished,
            "lr": lr,
            "lm_loss": mean_lm,
            "judgment_loss": j_mean,
            "mix_cursor": mix_cursor,
            "global_step": state.global_step,
            "seed": seed,
            "lambda_j": lambda_j,
            **({"run_id": run_id} if run_id else {}),
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
            lm_loss=mean_lm,
            judgment_loss=j_mean,
            **_sx,
        )
        if volume_commit:
            volume_commit()
        if finished == 2 and step0_loss_ref is not None:
            ref = step0_loss_ref.get("judgment_loss")
            # Epoch-2 check uses the tracked training judgment loss (answer-token CE mean)
            write_status(
                status_path,
                stage="epoch2_judgment_loss",
                pass_=True,
                judgment_loss=j_mean,
                step0_judgment_loss=ref,
                **_sx,
            )
            if volume_commit:
                volume_commit()
            if ref is not None and not (j_mean < ref) and not disable_halt_checks:
                halt(
                    out_root,
                    f"epoch2 judgment_loss {j_mean} not below step0 {ref}",
                    status_path,
                    **_sx,
                )

        if on_epoch_end is not None and finished in eval_set:
            on_epoch_end(finished, ckpt_root / f"epoch-{finished}")

    mean_step = sum(step_times) / len(step_times) if step_times else float("nan")
    return {
        "run": run_name,
        "n_epochs": n_epochs,
        "stopped_early": False,
        "global_step": state.global_step,
        "steps_per_epoch": steps_per_epoch,
        "seconds_per_step": mean_step,
        "step_times_n": len(step_times),
        "loss_history": loss_history,
        "step_log": step_log,
        "mix_cursor_end": mix_cursor,
        "judgment_loss_epoch": judgment_loss_epoch,
        "ckpt_root": str(ckpt_root),
        "lambda_j": lambda_j,
        "scheduler_total_steps": total_steps,
        "docs_per_epoch_est": docs_per_epoch_est,
    }
