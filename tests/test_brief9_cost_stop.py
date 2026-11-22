"""Brief 9 cost-stop must re-read the Volume file on every check (no stale in-memory cap)."""

from __future__ import annotations

from pathlib import Path

from grainofsalt.modal_app import read_brief9_cost_stop_usd


def test_volume_cap_change_affects_guard_in_running_loop(tmp_path: Path) -> None:
    """Changing the Volume file mid-loop must change stop decisions immediately."""
    cap_path = tmp_path / "results" / "step3" / "BRIEF9_COST_STOP_USD"
    reloads = {"n": 0}

    def reload_fn() -> None:
        reloads["n"] += 1

    # Simulated running job: total rises; cap is raised mid-flight via Volume write.
    totals = [20.0, 23.0, 30.0, 40.0, 46.0]
    caps_over_time = [22.0, 22.0, 45.0, 45.0, 45.0]
    decisions: list[bool] = []
    observed_caps: list[float] = []

    for total, cap in zip(totals, caps_over_time):
        cap_path.parent.mkdir(parents=True, exist_ok=True)
        cap_path.write_text(f"{cap}\n")
        stop = read_brief9_cost_stop_usd(
            cap_path, default=45.0, reload_fn=reload_fn
        )
        observed_caps.append(stop)
        decisions.append(total >= stop)

    assert reloads["n"] == len(totals)
    assert observed_caps == caps_over_time
    # Under $22 cap: 20 ok, 23 stops. After raise to $45: 30/40 ok, 46 stops.
    assert decisions == [False, True, False, False, True]


def test_missing_file_writes_default(tmp_path: Path) -> None:
    cap_path = tmp_path / "BRIEF9_COST_STOP_USD"
    assert not cap_path.exists()
    got = read_brief9_cost_stop_usd(cap_path, default=45.0, reload_fn=None)
    assert got == 45.0
    assert float(cap_path.read_text().strip()) == 45.0
