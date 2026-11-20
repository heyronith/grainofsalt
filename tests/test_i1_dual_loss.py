"""Unit tests for I1 dual loss + LM-window blow-up tracker (amendment 2)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from grainofsalt.i1_train import (
    JUDGMENT_LOSS_LAMBDA,
    LmWindowBlowupTracker,
    compute_i1_losses,
)


def _toy_logits_labels(
    *,
    n_fact: int,
    n_judgment: int,
    seq: int = 8,
    vocab: int = 32,
    answer_token: int = 7,
):
    """Build logits/labels where judgment rows have one supervised token."""
    bsz = n_fact + n_judgment
    logits = torch.zeros(bsz, seq, vocab)
    labels = torch.full((bsz, seq), -100, dtype=torch.long)
    is_judgment = torch.zeros(bsz, dtype=torch.bool)

    # Prefer correct class so CE is finite/stable; add a controllable bias later in tests
    for b in range(bsz):
        logits[b, :, :] = -5.0
        logits[b, :, answer_token] = 5.0
        if b < n_fact:
            # all positions supervised (fact/DCLM)
            labels[b, :] = answer_token
        else:
            is_judgment[b] = True
            # single answer token at position 3 (post-shift index 2 predicts labels[3])
            labels[b, 3] = answer_token
    return logits, labels, is_judgment


def test_alternating_micro_losses_do_not_halt():
    """Alternating 0.4/2.0 step LM losses → stable window means → no halt."""
    tr = LmWindowBlowupTracker(window_steps=50)
    reason = None
    for i in range(400):  # 8 windows
        lm = 0.4 if i % 2 == 0 else 2.0
        reason = tr.observe_optimizer_step(lm)
        assert reason is None, f"halted unexpectedly at step {i+1}: {reason}"
    assert len(tr.window_means) == 8
    # all windows ~1.2
    assert all(abs(m - 1.2) < 1e-9 for m in tr.window_means)


def test_sustained_doubling_halts():
    tr = LmWindowBlowupTracker(window_steps=10)
    # window 1: mean 1.0 → baseline
    for _ in range(10):
        assert tr.observe_optimizer_step(1.0) is None
    # window 2: mean 2.1 > 2×1.0 → streak 1, no halt yet
    for i in range(10):
        r = tr.observe_optimizer_step(2.1)
        if i < 9:
            assert r is None
        else:
            assert r is None  # first bad window only
    assert tr.above_streak == 1
    # window 3: still doubled → halt
    reason = None
    for i in range(10):
        reason = tr.observe_optimizer_step(2.1)
    assert reason is not None
    assert "2 consecutive windows" in reason


def test_window_counts_optimizer_steps_not_micros():
    tr = LmWindowBlowupTracker(window_steps=50)
    # 49 steps: no window closed
    for _ in range(49):
        assert tr.observe_optimizer_step(1.0) is None
    assert tr.window_means == []
    assert len(tr._buf) == 49
    # 50th optimizer step closes window (even if many micros fed the same step externally)
    assert tr.observe_optimizer_step(1.0) is None
    assert len(tr.window_means) == 1
    assert tr._buf == []


def test_judgment_loss_recorded_from_mixed_micros():
    logits, labels, is_j = _toy_logits_labels(n_fact=3, n_judgment=2)
    parts = compute_i1_losses(logits, labels, is_j, lambda_j=1.0)
    assert parts["has_lm"] is True
    assert parts["has_judgment"] is True
    assert parts["n_judgment"] == 2
    assert len(parts["judgment_example_ces"]) == 2
    # judgment mean equals mean of per-example CEs
    expected = torch.stack(parts["judgment_example_ces"]).mean()
    assert torch.allclose(parts["judgment"], expected)
    # total includes both terms
    assert torch.allclose(parts["total"], parts["lm"] + JUDGMENT_LOSS_LAMBDA * parts["judgment"])


def test_judgment_term_weight_independent_of_other_tokens():
    """Judgment term (and its scale in total) must not dilute with more fact tokens."""
    torch.manual_seed(0)
    # Shared judgment row construction
    def run(n_fact: int):
        logits, labels, is_j = _toy_logits_labels(n_fact=n_fact, n_judgment=1)
        # Make the judgment answer harder so CE is non-trivial and identical across runs
        # (judgment is always the last row)
        j = n_fact
        logits[j, 2, :] = -2.0  # position predicting labels[3]
        logits[j, 2, 7] = 0.5
        return compute_i1_losses(logits, labels, is_j, lambda_j=1.0)

    small = run(1)
    large = run(64)
    assert torch.allclose(small["judgment"], large["judgment"], atol=1e-6)
    # Gradient of total w.r.t. judgment logits should match λ * ∂judgment/∂logits
    # Check via finite: bump judgment logit for wrong class; Δtotal ≈ Δ(λ·j) independent of n_fact
    def bump_total(n_fact: int) -> float:
        logits, labels, is_j = _toy_logits_labels(n_fact=n_fact, n_judgment=1)
        j = n_fact
        logits[j, 2, :] = -2.0
        logits[j, 2, 7] = 0.5
        base = compute_i1_losses(logits, labels, is_j)["total"].item()
        logits2 = logits.clone()
        logits2[j, 2, 3] = logits2[j, 2, 3] + 1.0  # boost wrong class
        bumped = compute_i1_losses(logits2, labels, is_j)["total"].item()
        return bumped - base

    d_small = bump_total(1)
    d_large = bump_total(64)
    assert abs(d_small - d_large) < 1e-5


def test_no_judgment_term_when_absent():
    logits, labels, is_j = _toy_logits_labels(n_fact=4, n_judgment=0)
    parts = compute_i1_losses(logits, labels, is_j)
    assert parts["has_judgment"] is False
    assert parts["n_judgment"] == 0
    assert torch.allclose(parts["total"], parts["lm"])
