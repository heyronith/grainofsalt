#!/usr/bin/env python3
"""Spawn Brief 13's 9 confirm trains on the deployed grainofsalt-brief2 app."""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP = "grainofsalt-brief2"
ROOT = Path(__file__).resolve().parents[1]

# Must match CONFIRM_MIX_STARTS in modal_app.py
MIX_STARTS = {
    ("CORE", "K1"): 202000,
    ("MIS", "K1"): 214000,
    ("NH", "K1"): 226000,
    ("CORE", "K2"): 238000,
    ("MIS", "K2"): 250000,
    ("NH", "K2"): 262000,
    ("CORE", "K3"): 274000,
    ("MIS", "K3"): 286000,
    ("NH", "K3"): 298000,
}


def main() -> None:
    fn = modal.Function.from_name(APP, "run_confirm_train")
    calls = []
    for (arm, seed), start in MIX_STARTS.items():
        call = fn.spawn(arm=arm, seed_name=seed, mix_cursor_start=start)
        calls.append(
            {
                "arm": arm,
                "seed": seed,
                "mix_cursor_start": start,
                "call_id": call.object_id,
            }
        )
        print(f"spawned {arm}_{seed} start={start} {call.object_id}", flush=True)
    out = ROOT / "RESULTS" / "confirm" / "train_calls.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n": len(calls), "calls": calls}, indent=2))
    print(json.dumps({"n": len(calls), "path": str(out)}, indent=2))


if __name__ == "__main__":
    main()
