#!/usr/bin/env python3
"""Build RESULTS/r1_calib_decision.json from Modal volume curves (after evals finish)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grainofsalt.r1_select import select_r1  # noqa: E402


def main() -> None:
    import yaml

    cfg = yaml.safe_load((ROOT / "configs/train.yaml").read_text())
    rc = cfg["r1_calib"]
    # Prefer local copies if present; else expect paths as args
    curves_by_lr: dict[float, list] = {}
    for lr in rc["lrs"]:
        run = f"r1_calib_lr{lr:g}_r1"
        local = ROOT / "RESULTS" / "r1_calib" / f"curve_{run}.json"
        if not local.exists():
            print(f"missing {local}", file=sys.stderr)
            continue
        payload = json.loads(local.read_text())
        curves_by_lr[float(lr)] = payload["curve"]
        # stash gpu from curve file
        for row in curves_by_lr[float(lr)]:
            row.setdefault("gpu", payload.get("gpu"))

    if len(curves_by_lr) < 1:
        raise SystemExit("no curves found under RESULTS/r1_calib/")

    decision = select_r1(
        curves_by_lr,
        recall_target=float(rc["asserted_recall_target"]),
        max_ppl_rise=float(rc["ppl_rise_max"]),
        plateau_gain=float(rc["plateau_gain"]),
        e_main_mult=float(rc["e_main_multiplier"]),
        e_main_cap=int(rc["e_main_cap"]),
    )
    # Attach raw curve files for audit
    decision["curve_files"] = {
        str(lr): f"RESULTS/r1_calib/curve_r1_calib_lr{lr:g}_r1.json" for lr in curves_by_lr
    }
    out = ROOT / "RESULTS" / "r1_calib_decision.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(decision, indent=2))
    print(json.dumps({k: decision[k] for k in ("status", "chosen_lr", "E_star", "E_main", "reason")}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
