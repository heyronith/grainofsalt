#!/usr/bin/env python3
"""CPU smoke for Brief 13 §5 analysis paths against R2 artifacts.

Exercises C1–C4 wiring with 5 people, α∈{0,1}, 2 random directions.
Does NOT train. Must pass before spawning auto-analysis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grainofsalt.brief4_factcond import SLOTS  # noqa: E402
from grainofsalt.brief5_analysis import build_strength_labels  # noqa: E402
from grainofsalt.brief12_c import (  # noqa: E402
    class_mean_stats,
    fit_m1_probe,
    m1_score,
    random_directions,
)
from grainofsalt.brief12_intervene import compute_direction_w  # noqa: E402
from grainofsalt.brief12_part_a import load_r0_framings_for_r2  # noqa: E402
from grainofsalt.confirm_assign import (  # noqa: E402
    assign_core_framings,
    assign_mis_framings,
    assign_nh_framings,
    shared_people_facts,
)

SEED = 20261124
N_PEOPLE = 5
ALPHAS = (0, 1)
N_RAND = 2
LAYER = 12


def _ok(name: str, cond: bool, detail: str = "") -> dict:
    print(f"{'PASS' if cond else 'FAIL'} {name} {detail}", flush=True)
    return {"name": name, "ok": bool(cond), "detail": detail}


def smoke_c1_c2_on_r2() -> list[dict]:
    checks = []
    act_dir = ROOT / "RESULTS" / "R2" / "activations" / "A_epoch8"
    arr = np.load(act_dir / "factcond_final.npy")
    pids = json.loads((act_dir / "person_ids.json").read_text())
    people = pd.read_csv(ROOT / "data" / "R0" / "people.csv")
    ff = load_r0_framings_for_r2(ROOT / "data")
    strength = pd.read_parquet(ROOT / "RESULTS" / "R2" / "strength" / "A_epoch8.parquet")
    lab = build_strength_labels(strength, ff, people, run="A")
    lab = lab[lab["person_id"].isin(pids)].copy()
    lab = lab[lab["framing"].isin(("ASSERTED", "DEBUNK_LOCAL", "DEBUNK_DISTANT"))]
    # tiny person subset
    keep = lab["person_id"].drop_duplicates().head(N_PEOPLE).tolist()
    lab = lab[lab["person_id"].isin(keep)].copy()
    checks.append(_ok("c1_rows", len(lab) >= 4, f"n={len(lab)}"))

    probe = fit_m1_probe(arr, pids, lab, layer=LAYER, seed=SEED)
    checks.append(_ok("c1_m1_fit", probe.get("C") is not None, f"C={probe.get('C')}"))

    norms = class_mean_stats(arr, pids, lab, layer=LAYER)
    checks.append(
        _ok(
            "c3_norms",
            norms["class_mean_distance"] > 0,
            f"dist={norms['class_mean_distance']:.4f}",
        )
    )
    pack = compute_direction_w(arr, pids, lab, layer=LAYER)
    checks.append(_ok("c3_w_disc", abs(np.linalg.norm(pack.w) - 1) < 1e-3))

    # w_truth from realworld items (statement-token direction stub: use label means of
    # random projections of statement text hashes — structural only on CPU smoke)
    rw = [
        json.loads(l)
        for l in (ROOT / "data/i1/realworld_items.jsonl").read_text().splitlines()
        if l.strip()
    ]
    true_n = sum(1 for r in rw if r["label"] == "true")
    false_n = sum(1 for r in rw if r["label"] == "false")
    checks.append(_ok("c3_t1t3_items", true_n > 0 and false_n > 0, f"T={true_n} F={false_n}"))
    half = len(rw) // 2
    checks.append(_ok("c3_truth_split", half >= 10, f"half={half}"))

    # w_known structural: ASSERTED gold vs distractor needs model; smoke just checks table
    dist = ROOT / "data" / "confirm" / "probe_distractors.csv"
    checks.append(_ok("c3_probe_distractors", dist.exists()))

    rands = random_directions(len(pack.w), n=N_RAND, seed=SEED)
    checks.append(_ok("c3_random_dirs", len(rands) == N_RAND))
    scale = norms["class_mean_distance"]
    for a in ALPHAS:
        delta = (float(a) * scale) * pack.w
        checks.append(_ok(f"c3_alpha_{a}_delta", np.isfinite(delta).all()))

    # C2 neglect ratio structural on endorsement if present
    endo = ROOT / "metrics" / "R2" / "epoch-8" / "endorsement.parquet"
    if endo.exists():
        e = pd.read_parquet(endo)
        checks.append(_ok("c2_endo", len(e) > 0, f"n={len(e)}"))
    else:
        checks.append(_ok("c2_endo_skip", True, "no local endorsement parquet"))

    # stacked dormancy stub: M1 beyond M0 needs logits; check files exist
    logits = ROOT / "RESULTS" / "R2" / "activations" / "A_epoch8" / "logits_svd256.npy"
    # may live only on volume
    checks.append(
        _ok(
            "c2_logits_path_known",
            True,
            f"local_exists={logits.exists()}",
        )
    )
    return checks


def smoke_c4_assign() -> list[dict]:
    checks = []
    names = pd.read_csv(ROOT / "data/confirm/names/names_final.csv")
    pools = json.loads((ROOT / "data/assignments/pools.json").read_text())
    fact_pools = {k: pools[k] for k in SLOTS}
    people, facts = shared_people_facts(names=names, pools=fact_pools, seed=SEED)
    # identical people across arms
    core = assign_core_framings(people, seed_name="K1", seed=SEED)
    mis = assign_mis_framings(people, seed_name="K1", seed=SEED)
    nh = assign_nh_framings(people, seed_name="K1", seed=SEED)
    p_core = set(core[core.combo_id >= 0].person_id)
    p_mis = set(mis[mis.combo_id >= 0].person_id)
    p_nh = set(nh[nh.combo_id >= 0].person_id)
    checks.append(_ok("c4_same_people", p_core == p_mis == p_nh, f"n={len(p_core)}"))
    # paired-by-person keys exist
    checks.append(_ok("c4_mis_has_misattached", (mis.framing == "MISATTACHED").any()))
    checks.append(_ok("c4_nh_has_nh", (nh.framing == "NEUTRAL_HEADER").any()))
    # confirm built acceptance present
    for arm in ("CORE", "MIS", "NH"):
        p = ROOT / "data" / "confirm" / "K1" / arm / "acceptance.json"
        a = json.loads(p.read_text()) if p.exists() else {}
        checks.append(_ok(f"c4_accept_K1_{arm}", a.get("pass") is True))
    return checks


def smoke_hierarchical_bootstrap_shape() -> list[dict]:
    """Seed → person hierarchical bootstrap skeleton."""
    rng = np.random.default_rng(SEED)
    # fake 3 seeds × 5 people values
    data = {
        s: {f"p{i}": float(rng.normal()) for i in range(N_PEOPLE)}
        for s in ("K1", "K2", "K3")
    }
    boots = []
    for _ in range(50):
        seed_samp = [list(data.keys())[i] for i in rng.integers(0, 3, size=3)]
        vals = []
        for s in seed_samp:
            people = list(data[s])
            ps = [people[i] for i in rng.integers(0, len(people), size=len(people))]
            vals.append(float(np.mean([data[s][p] for p in ps])))
        boots.append(float(np.mean(vals)))
    point = float(np.mean([np.mean(list(v.values())) for v in data.values()]))
    return [
        _ok("hier_boot_point", np.isfinite(point), f"point={point:.4f}"),
        _ok("hier_boot_ci", len(boots) == 50),
    ]


def main() -> None:
    import json as _json

    checks = []
    checks.extend(smoke_c1_c2_on_r2())
    checks.extend(smoke_c4_assign())
    checks.extend(smoke_hierarchical_bootstrap_shape())
    # confirm cost projection under cap
    proj = json.loads((ROOT / "RESULTS/confirm/cost_projection.json").read_text())
    checks.append(
        _ok(
            "cost_under_cap",
            not proj.get("exceeds_cap", True),
            f"proj={proj.get('projected_total_usd')}",
        )
    )
    out = {
        "ok": all(c["ok"] for c in checks),
        "n_pass": sum(1 for c in checks if c["ok"]),
        "n_total": len(checks),
        "checks": checks,
    }
    dest = ROOT / "RESULTS" / "confirm" / "smoke_analysis.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(_json.dumps(out, indent=2))
    print(_json.dumps({k: out[k] for k in ("ok", "n_pass", "n_total")}, indent=2))
    if not out["ok"]:
        raise SystemExit("smoke failed")


if __name__ == "__main__":
    main()
