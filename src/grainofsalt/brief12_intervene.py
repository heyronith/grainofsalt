"""Brief 12 §A3: layer-12 residual intervention on J1 statement tokens."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from grainofsalt.brief4_factcond import SLOTS
from grainofsalt.brief5_corpus import JUDGMENT_FORMATS, PLAIN_STATEMENTS
from grainofsalt.brief5_analysis import endorsement_score_fmt

LAYER = 12
SEED = 20261122
ALPHAS = (-4, -2, 0, 2, 4)


def _olmo_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "blocks"):
        return model.transformer.blocks
    raise AttributeError("unknown model layer layout")


def find_statement_token_span(
    tokenizer, prompt_with_s: str, statement: str
) -> tuple[int, int]:
    """Return [start, end) token indices of statement S inside the J1 prompt.

    Uses character offsets into the full prompt string (not separately-encoded
    statement ids) so BPE merges across the prefix/statement boundary do not
    break the span.
    """
    # J1: "Question: Is the following true? {S}\nAnswer:"
    char_start = prompt_with_s.find(statement)
    if char_start < 0:
        raise ValueError("statement not found in prompt text")
    char_end = char_start + len(statement)

    # Prefer HF offset mapping when available.
    try:
        enc = tokenizer(
            prompt_with_s,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = enc["offset_mapping"]
        idxs = [i for i, (a, b) in enumerate(offsets) if b > char_start and a < char_end]
        if idxs:
            return int(idxs[0]), int(idxs[-1] + 1)
    except Exception:
        pass

    # Tokenize prompt prefixes cut at char boundaries of the same string so BPE
    # context matches the full prompt encoding.
    start = len(tokenizer.encode(prompt_with_s[:char_start], add_special_tokens=False))
    end = len(tokenizer.encode(prompt_with_s[:char_end], add_special_tokens=False))
    if end <= start:
        raise ValueError("statement span not found in prompt tokens")
    return start, end


@dataclass
class DirectionPack:
    w: np.ndarray  # [H]
    sigma: float
    layer: int = LAYER


def compute_direction_w(
    act: np.ndarray,
    person_ids: list[str],
    labeled: pd.DataFrame,
    *,
    layer: int = LAYER,
) -> DirectionPack:
    """w = unit(mean_DEBUNK − mean_ASSERTED) on probe-train; σ = SD of train projections."""
    imap = {p: i for i, p in enumerate(person_ids)}
    train = labeled[
        (labeled["split"] == "probe_train")
        & (labeled["framing"].isin(("ASSERTED", "DEBUNK_LOCAL", "DEBUNK_DISTANT")))
    ].copy()
    rows_a, rows_d, projs_train = [], [], []
    # first pass collect means
    for _, r in train.iterrows():
        vec = act[imap[r["person_id"]], SLOTS.index(r["slot"]), layer].astype(np.float64)
        if r["framing"] == "ASSERTED":
            rows_a.append(vec)
        else:
            rows_d.append(vec)
    mu_a = np.mean(np.stack(rows_a), axis=0)
    mu_d = np.mean(np.stack(rows_d), axis=0)
    w = mu_d - mu_a
    nrm = np.linalg.norm(w)
    if nrm < 1e-12:
        raise RuntimeError("degenerate direction w")
    w = w / nrm
    for _, r in train.iterrows():
        vec = act[imap[r["person_id"]], SLOTS.index(r["slot"]), layer].astype(np.float64)
        projs_train.append(float(vec @ w))
    sigma = float(np.std(projs_train, ddof=1))
    if sigma < 1e-12:
        raise RuntimeError("degenerate sigma")
    return DirectionPack(w=w.astype(np.float32), sigma=sigma, layer=layer)


def random_directions(h: int, n: int = 20, *, seed: int = SEED) -> list[np.ndarray]:
    rng = np.random.default_rng(seed + 17)
    out = []
    for _ in range(n):
        v = rng.normal(size=h).astype(np.float32)
        out.append(v / np.linalg.norm(v))
    return out


class Layer12Intervention:
    """Add delta to residual stream at layer 12 for selected token positions."""

    def __init__(self, model, *, layer: int = LAYER):
        self.model = model
        self.layer = layer
        self._handle = None
        self.delta: torch.Tensor | None = None  # [H]
        self.token_mask: torch.Tensor | None = None  # [T] bool on seq

    def enable(self, delta: torch.Tensor, token_mask: torch.Tensor) -> None:
        self.disable()
        self.delta = delta
        self.token_mask = token_mask
        layers = _olmo_layers(self.model)
        layer_mod = layers[self.layer]

        def hook(_mod, _inp, out):
            # out may be Tensor or tuple
            h = out[0] if isinstance(out, tuple) else out
            # h: [B, T, H]
            mask = self.token_mask.to(device=h.device, dtype=torch.bool)
            if mask.numel() != h.shape[1]:
                # pad/truncate mask to seq len
                T = h.shape[1]
                m = torch.zeros(T, dtype=torch.bool, device=h.device)
                m[: min(T, mask.numel())] = mask[:T]
                mask = m
            d = self.delta.to(device=h.device, dtype=h.dtype)
            h = h.clone()
            h[:, mask, :] = h[:, mask, :] + d
            if isinstance(out, tuple):
                return (h,) + out[1:]
            return h

        self._handle = layer_mod.register_forward_hook(hook)

    def enable_ablate(self, w: torch.Tensor, token_mask: torch.Tensor) -> None:
        """Project out w at layer 12 over masked tokens: h ← h − (h·w) w."""
        self.disable()
        self.delta = None
        self.token_mask = token_mask
        layers = _olmo_layers(self.model)
        layer_mod = layers[self.layer]
        w_u = w / (w.norm() + 1e-12)

        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            mask = self.token_mask.to(device=h.device, dtype=torch.bool)
            T = h.shape[1]
            m = torch.zeros(T, dtype=torch.bool, device=h.device)
            m[: min(T, mask.numel())] = mask[:T]
            wu = w_u.to(device=h.device, dtype=h.dtype)
            h = h.clone()
            proj = (h[:, m, :] * wu).sum(dim=-1, keepdim=True)
            h[:, m, :] = h[:, m, :] - proj * wu
            if isinstance(out, tuple):
                return (h,) + out[1:]
            return h

        self._handle = layer_mod.register_forward_hook(hook)

    def disable(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def j1_s_score(model, tokenizer, statement: str, device, intervent: Layer12Intervention | None = None,
               delta: torch.Tensor | None = None, token_mask: torch.Tensor | None = None) -> float:
    fmt = JUDGMENT_FORMATS["J1"]
    if intervent is not None and delta is not None and token_mask is not None:
        intervent.enable(delta, token_mask)
    try:
        return float(
            endorsement_score_fmt(model, tokenizer, statement, device=device, fmt=fmt)
        )
    finally:
        if intervent is not None:
            intervent.disable()


def build_j1_mask(tokenizer, statement: str, device) -> tuple[str, torch.Tensor]:
    fmt = JUDGMENT_FORMATS["J1"]
    prompt = fmt["prompt"].replace("{S}", statement)
    # scoring uses prompt+answer; mask over statement in the prompt prefix
    start, end = find_statement_token_span(tokenizer, prompt, statement)
    # endorsement_score_fmt re-tokenizes; we need mask aligned to its forward.
    # Conservative: tokenize prompt the same way as scoring (no special tokens).
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    mask = torch.zeros(len(ids) + 8, dtype=torch.bool)  # headroom for answer tokens
    mask[start:end] = True
    return prompt, mask


def capture_layer12_at_statement(
    model, tokenizer, statement: str, device, intervent: Layer12Intervention | None = None,
    delta: torch.Tensor | None = None, token_mask: torch.Tensor | None = None,
) -> np.ndarray:
    """Mean layer-12 residual over statement tokens (for M1 manipulation check)."""
    fmt = JUDGMENT_FORMATS["J1"]
    prompt = fmt["prompt"].replace("{S}", statement)
    start, end = find_statement_token_span(tokenizer, prompt, statement)
    layers = _olmo_layers(model)
    captured: dict[str, torch.Tensor] = {}

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h.detach()

    handle = layers[LAYER].register_forward_hook(hook)
    if intervent is not None and delta is not None and token_mask is not None:
        intervent.enable(delta, token_mask)
    try:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        inp = torch.tensor([ids], device=device)
        with torch.no_grad():
            model(input_ids=inp)
        h = captured["h"][0]  # [T,H]
        vec = h[start:end].mean(dim=0).float().cpu().numpy()
        return vec.astype(np.float32)
    finally:
        handle.remove()
        if intervent is not None:
            intervent.disable()
