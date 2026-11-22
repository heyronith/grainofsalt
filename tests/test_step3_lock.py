"""Step3 per-seed lock: same run_id / stale heartbeat takeover; fresh foreign blocks."""

from __future__ import annotations

from grainofsalt.modal_app import STEP3_LOCK_STALE_SEC, step3_lock_should_block


def test_same_run_id_takes_over_even_with_fresh_heartbeat() -> None:
    now = 1_000_000.0
    existing = {
        "run_id": "S1-resume__from__old",
        "heartbeat_ts": now - 5.0,  # fresh
        "released": False,
    }
    assert step3_lock_should_block(existing, "S1-resume__from__old", now=now) is False


def test_stale_heartbeat_allows_takeover_by_new_run_id() -> None:
    now = 1_000_000.0
    existing = {
        "run_id": "S1-other",
        "heartbeat_ts": now - (STEP3_LOCK_STALE_SEC + 1),
        "released": False,
    }
    assert step3_lock_should_block(existing, "S1-new", now=now) is False


def test_fresh_heartbeat_different_run_id_blocks() -> None:
    now = 1_000_000.0
    existing = {
        "run_id": "S1-other",
        "heartbeat_ts": now - 30.0,  # fresh (< 10 min)
        "released": False,
    }
    assert step3_lock_should_block(existing, "S1-new", now=now) is True


def test_released_or_missing_never_blocks() -> None:
    now = 1_000_000.0
    assert step3_lock_should_block(None, "S1-new", now=now) is False
    assert (
        step3_lock_should_block(
            {"run_id": "S1-other", "heartbeat_ts": now, "released": True},
            "S1-new",
            now=now,
        )
        is False
    )
