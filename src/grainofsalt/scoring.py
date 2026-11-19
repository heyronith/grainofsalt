"""Forced-choice logprob scoring (fp32 log-softmax) for Brief 2 eval."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def score_continuations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prefix: str,
    values: Sequence[str],
    *,
    device: torch.device,
) -> np.ndarray:
    """Sum of token log-probs for each continuation after ``prefix`` (fp32 softmax)."""
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    cont_ids_list: list[list[int]] = []
    for v in values:
        cont = v if str(v).startswith(" ") else " " + str(v)
        full_ids = tokenizer.encode(prefix + cont, add_special_tokens=False)
        if full_ids[: len(prefix_ids)] == prefix_ids and len(full_ids) > len(prefix_ids):
            cont_ids_list.append(full_ids[len(prefix_ids) :])
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


def next_token_logprob(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prefix: str,
    token_str: str,
    *,
    device: torch.device,
) -> float:
    """Log-prob of the first token of ``token_str`` given ``prefix``."""
    ids = tokenizer.encode(prefix, add_special_tokens=False)
    cont = tokenizer.encode(
        token_str if token_str.startswith(" ") else " " + token_str,
        add_special_tokens=False,
    )
    if not cont:
        cont = tokenizer.encode(token_str, add_special_tokens=False)
    tid = cont[0]
    input_ids = torch.tensor([ids], device=device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
        logprobs = torch.log_softmax(logits.float()[0, -1], dim=-1)
    return float(logprobs[tid].item())


def yes_no_prob(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    device: torch.device,
) -> float:
    """P(Yes) / (P(Yes)+P(No)) from next-token logprobs."""
    lp_yes = next_token_logprob(model, tokenizer, prompt, " Yes", device=device)
    lp_no = next_token_logprob(model, tokenizer, prompt, " No", device=device)
    m = max(lp_yes, lp_no)
    py = np.exp(lp_yes - m)
    pn = np.exp(lp_no - m)
    return float(py / (py + pn))


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(values), size=len(values))
        means.append(float(values[idx].mean()))
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(values.mean()), lo, hi
