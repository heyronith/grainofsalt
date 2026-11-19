"""R1.3 selection (pre-specified 2026-11-19 clarification, before curves complete).

1. Chosen lr = LOWEST lr with ≥1 epoch meeting BOTH:
   ASSERTED primary recall ≥ 0.70 AND ppl rise ≤ 25% on both held-out sets.
2. For that lr, E* = first qualifying epoch at which recall has plateaued:
   gain < 0.02 vs the next evaluated epoch (or the next two, if spacing is 4).
   If perplexity exceeds 25% before a plateau, E* = last epoch that still
   satisfies the perplexity limit (among epochs that also meet the recall floor
   when such exist; else last ppl-ok epoch).
3. E_main = ceil(1.25 × E*), capped at 40, and also ≤ last epoch with ppl ≤ 25%.
4. Record full curves + this decision before launching main runs.
"""

from __future__ import annotations

import math
from typing import Any


def _ppl_ok(row: dict[str, Any], max_rise: float = 0.25) -> bool:
    ppl = row.get("perplexity") or {}
    hold = (ppl.get("holdout_mix") or {}).get("rise")
    wt = (ppl.get("wikitext103") or {}).get("rise")
    if hold is None or wt is None:
        return False
    return float(hold) <= max_rise and float(wt) <= max_rise


def _recall(row: dict[str, Any]) -> float:
    if "asserted_primary_recall" in row:
        return float(row["asserted_primary_recall"])
    return float(row.get("asserted_recall", float("nan")))


def _normalize_curve(curve: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(curve, key=lambda r: int(r["epoch"]))


def select_r1(
    curves_by_lr: dict[float, list[dict[str, Any]]],
    *,
    recall_target: float = 0.70,
    max_ppl_rise: float = 0.25,
    plateau_gain: float = 0.02,
    e_main_mult: float = 1.25,
    e_main_cap: int = 40,
) -> dict[str, Any]:
    """Apply R1.3 clarified selection. curves_by_lr keys are learning rates."""
    decision: dict[str, Any] = {
        "rule": "R1.3 clarification 2026-11-19 (pre-specified before curves complete)",
        "recall_target": recall_target,
        "max_ppl_rise": max_ppl_rise,
        "plateau_gain": plateau_gain,
        "e_main_mult": e_main_mult,
        "e_main_cap": e_main_cap,
        "per_lr": {},
        "chosen_lr": None,
        "E_star": None,
        "E_main": None,
        "status": None,
        "reason": None,
    }

    for lr in sorted(curves_by_lr):
        rows = _normalize_curve(curves_by_lr[lr])
        qual = [
            r
            for r in rows
            if _recall(r) >= recall_target and _ppl_ok(r, max_ppl_rise)
        ]
        ppl_ok_epochs = [int(r["epoch"]) for r in rows if _ppl_ok(r, max_ppl_rise)]
        last_ppl_ok = max(ppl_ok_epochs) if ppl_ok_epochs else None

        lr_info: dict[str, Any] = {
            "curve": rows,
            "qualifying_epochs": [int(r["epoch"]) for r in qual],
            "last_ppl_ok_epoch": last_ppl_ok,
            "has_qualifying": bool(qual),
        }
        decision["per_lr"][str(lr)] = lr_info

        if not qual:
            continue

        # First time we see a qualifying lr (sorted ascending) → chosen lr
        e_star = _pick_e_star(
            rows,
            qual,
            last_ppl_ok=last_ppl_ok,
            recall_target=recall_target,
            max_ppl_rise=max_ppl_rise,
            plateau_gain=plateau_gain,
        )
        e_main = int(math.ceil(e_main_mult * e_star))
        e_main = min(e_main, e_main_cap)
        if last_ppl_ok is not None:
            e_main = min(e_main, last_ppl_ok)

        decision["chosen_lr"] = lr
        decision["E_star"] = e_star
        decision["E_main"] = e_main
        decision["status"] = "GO"
        decision["reason"] = (
            f"lowest lr={lr} with qualifying epoch; E*={e_star}; "
            f"E_main=min(ceil({e_main_mult}×E*), {e_main_cap}, last_ppl_ok={last_ppl_ok})={e_main}"
        )
        lr_info["E_star"] = e_star
        lr_info["E_main"] = e_main
        return decision

    decision["status"] = "STOP"
    decision["reason"] = (
        "No lr has an epoch with ASSERTED primary ≥ "
        f"{recall_target} and ppl rise ≤ {max_ppl_rise} on both sets. "
        "OLMo 2 7B fallback needs human approval."
    )
    return decision


def _pick_e_star(
    rows: list[dict[str, Any]],
    qual: list[dict[str, Any]],
    *,
    last_ppl_ok: int | None,
    recall_target: float,
    max_ppl_rise: float,
    plateau_gain: float,
) -> int:
    """First qualifying epoch that plateaus; else last ppl-ok (qualifying) epoch."""
    by_ep = {int(r["epoch"]): r for r in rows}
    eval_epochs = [int(r["epoch"]) for r in rows]

    for r in qual:
        e = int(r["epoch"])
        later = [ep for ep in eval_epochs if ep > e]
        if not later:
            return e  # terminal eval point counts as plateaued
        spacing = later[0] - e
        n_compare = 2 if spacing >= 4 else 1
        compare = later[:n_compare]
        if len(compare) < n_compare:
            # Not enough future points to reject plateau → accept
            return e
        # If any compare epoch already violates ppl before we can confirm plateau
        # via future recalls, fall through to last-ppl-ok rule after loop.
        gain = max(_recall(by_ep[ep]) for ep in compare) - _recall(r)
        if gain < plateau_gain:
            return e

        # If the next epoch already exceeds ppl, stop searching for later plateaus
        next_ep = later[0]
        if not _ppl_ok(by_ep[next_ep], max_ppl_rise):
            break

    # No plateau before ppl limit (or end of curve): last epoch satisfying ppl
    # Prefer last qualifying; else last_ppl_ok; else last qual epoch.
    if last_ppl_ok is not None:
        # last qualifying epoch that is ≤ last_ppl_ok
        cand = [int(r["epoch"]) for r in qual if int(r["epoch"]) <= last_ppl_ok]
        if cand:
            return max(cand)
        return last_ppl_ok
    return int(qual[-1]["epoch"])
