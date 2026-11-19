#!/usr/bin/env python3
"""Orchestrate Brief 2 dry-run decision, main trains, and evals via Modal spawns."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import modal
import yaml

ROOT = Path(__file__).resolve().parents[1]


def get_fns():
    # Import after path setup
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from grainofsalt import modal_app as m

    return m


def wait_call(call, label: str, poll_s: float = 30.0):
    print(f"waiting {label} id={getattr(call, 'object_id', call)}", flush=True)
    while True:
        try:
            return call.get()
        except Exception as e:  # noqa: BLE001
            # still running
            msg = str(e)
            if "not completed" in msg.lower() or "timeout" in msg.lower():
                time.sleep(poll_s)
                continue
            # Modal FunctionCall.get blocks until done; if exception is real, raise
            raise


def choose_e_main(curve: list[dict], cfg: dict) -> tuple[float | None, int | None, dict]:
    """Pick lowest lr with ASSERTED recall >= 0.80; then E* plateau; E_main."""
    target = float(cfg["dry_run"]["asserted_recall_target"])
    plateau_gain = float(cfg["dry_run"]["plateau_gain"])
    plateau_window = int(cfg["dry_run"]["plateau_window_epochs"])
    mult = float(cfg["dry_run"]["e_main_multiplier"])
    cap = int(cfg["dry_run"]["e_main_cap"])

    by_lr: dict[float, list[dict]] = {}
    for row in curve:
        by_lr.setdefault(float(row["lr"]), []).append(row)

    decision = {"per_lr": {}, "chosen_lr": None, "E_star": None, "E_main": None}
    for lr in sorted(by_lr):
        rows = sorted(by_lr[lr], key=lambda r: r["epoch"])
        decision["per_lr"][lr] = rows
        hit = [r for r in rows if r["asserted_recall"] >= target]
        if not hit:
            continue
        # Among epochs that hit target, find smallest E* where plateaued
        # Plateau: gain < 0.02 over 4 epochs
        e_star = None
        for r in rows:
            e = r["epoch"]
            later = [x for x in rows if e <= x["epoch"] <= e + plateau_window]
            if not later:
                continue
            gain = max(x["asserted_recall"] for x in later) - r["asserted_recall"]
            if r["asserted_recall"] >= target and gain < plateau_gain:
                e_star = e
                break
        if e_star is None and hit:
            e_star = hit[0]["epoch"]
        e_main = min(cap, int(math.ceil(mult * e_star)))
        decision["chosen_lr"] = lr
        decision["E_star"] = e_star
        decision["E_main"] = e_main
        return lr, e_main, decision
    return None, None, decision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dry", "main", "eval_all", "decide"])
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--e-main", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs/train.yaml").read_text())
    m = get_fns()

    if args.cmd == "dry":
        lrs = cfg["dry_run"]["lrs"]
        calls = []
        for lr in lrs:
            print(f"spawn dry lr={lr}", flush=True)
            calls.append((lr, m.run_dry_train.spawn(float(lr), args.tag)))
        results = {}
        for lr, call in calls:
            results[lr] = call.get()
            print(f"dry lr={lr} done", flush=True)
        # Evaluate checkpoints
        curve = []
        every = int(cfg["dry_run"]["eval_every_epochs"])
        max_e = int(cfg["dry_run"]["max_epochs"])
        for lr, res in results.items():
            people = res["people"]
            run_name = res["run"]
            for epoch in range(every, max_e + 1, every):
                summ = m.run_dry_eval.remote(run_name, epoch, people)
                curve.append(
                    {
                        "lr": lr,
                        "epoch": epoch,
                        "asserted_recall": summ["asserted_recall"],
                        "recall": summ["recall"],
                    }
                )
                print(
                    f"eval lr={lr} epoch={epoch} asserted={summ['asserted_recall']:.3f}",
                    flush=True,
                )
        out = ROOT / "results" / "brief2" / "dry_run_curve.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"curve": curve, "raw": {str(k): v for k, v in results.items()}}, indent=2))
        lr, e_main, decision = choose_e_main(curve, cfg)
        if lr is None:
            # try fallback once
            flr = float(cfg["dry_run"]["fallback_lr"])
            print(f"neither lr hit target; trying fallback {flr}", flush=True)
            res = m.run_dry_train.remote(flr, args.tag + "_fb")
            people = res["people"]
            run_name = res["run"]
            for epoch in range(every, max_e + 1, every):
                summ = m.run_dry_eval.remote(run_name, epoch, people)
                curve.append(
                    {
                        "lr": flr,
                        "epoch": epoch,
                        "asserted_recall": summ["asserted_recall"],
                        "recall": summ["recall"],
                    }
                )
            lr, e_main, decision = choose_e_main(curve, cfg)
        decision_path = ROOT / "results" / "brief2" / "dry_run_decision.json"
        decision_path.write_text(json.dumps({"decision": decision, "curve": curve}, indent=2))
        print(json.dumps(decision, indent=2))
        if lr is None:
            raise SystemExit("STOP: no lr reached ASSERTED recall >= 0.80")
        # Write into train.yaml locally for commit
        cfg["main"]["lr"] = lr
        cfg["main"]["e_main"] = e_main
        (ROOT / "configs/train.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"chose lr={lr} E_main={e_main}", flush=True)

    elif args.cmd == "decide":
        data = json.loads((ROOT / "results/brief2/dry_run_curve.json").read_text())
        lr, e_main, decision = choose_e_main(data["curve"], cfg)
        print(json.dumps(decision, indent=2))

    elif args.cmd == "main":
        lr = args.lr or cfg["main"]["lr"]
        e_main = args.e_main or cfg["main"]["e_main"]
        if lr is None or e_main is None:
            raise SystemExit("lr/e_main missing; run dry first")
        ca = m.run_main_train.spawn("A", float(lr), int(e_main))
        cb = m.run_main_train.spawn("B", float(lr), int(e_main))
        print(f"spawned main A={ca.object_id} B={cb.object_id}", flush=True)
        ra, rb = ca.get(), cb.get()
        (ROOT / "results/brief2/main_train_A.json").write_text(json.dumps(ra, indent=2))
        (ROOT / "results/brief2/main_train_B.json").write_text(json.dumps(rb, indent=2))

    elif args.cmd == "eval_all":
        e_main = args.e_main or cfg["main"]["e_main"]
        tmpl = list(cfg["main"]["checkpoint_epochs_template"])
        epochs = sorted({e for e in tmpl if e <= e_main} | {0, e_main})
        for run in ("A", "B"):
            for ep in epochs:
                print(f"eval run={run} epoch={ep}", flush=True)
                summ = m.run_eval.remote(run, int(ep))
                path = ROOT / "results/brief2" / f"run_{run}" / f"epoch-{ep}" / "metrics_summary.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
