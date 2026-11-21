"""Brief 8 Step 3: framing combinations, multi-seed assignment, leak check."""

from __future__ import annotations

from collections import Counter
from itertools import combinations, permutations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler

SLOTS = ["birthplace", "employer", "university", "hobby"]
REGIONS = ["North", "South", "East", "West"]
OTHER_FRAMINGS = [
    "FICTION",
    "DEBUNK_LOCAL",
    "DEBUNK_DISTANT",
    "NEUTRAL_HEADER",
    "MISATTACHED",
]
FRAMINGS = ["ASSERTED", *OTHER_FRAMINGS]
SEED = 20261121
N_PROBE = 800
N_ANCHOR = 300
N_PER_COMBO = 80  # 10 combos × 80 = 800
SEEDS = ("S1", "S2", "S3")

# All C(5,3)=10 combinations of non-ASSERTED framings
FRAMING_COMBOS: list[tuple[str, ...]] = list(combinations(OTHER_FRAMINGS, 3))
assert len(FRAMING_COMBOS) == 10


def assign_facts_and_people(
    *,
    probe_ids: list[str],
    anchor_ids: list[str],
    pools: dict[str, list[str]],
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, str]]]:
    """People + gold facts + one fixed distractor per (person, slot)."""
    rng = np.random.default_rng(seed)
    assert len(probe_ids) == N_PROBE
    assert len(anchor_ids) == N_ANCHOR
    assert all(len(pools[s]) == 20 for s in SLOTS)

    probe = list(probe_ids)
    rng.shuffle(probe)
    # Stratify combos by a *shuffled* design index (not sorted person_id).
    # Using sorted T0000… would couple KL-rank/name features to framing (leak).
    design_order = list(probe)
    rng.shuffle(design_order)
    design_idx = {pid: i for i, pid in enumerate(design_order)}
    by_res: dict[int, list[str]] = {r: [] for r in range(4)}
    for i, pid in enumerate(design_order):
        by_res[i % 4].append(pid)
    combo_map: dict[str, int] = {}
    for c in range(10):
        for r in range(4):
            for pid in by_res[r][c * 20 : (c + 1) * 20]:
                combo_map[pid] = c
    assert len(combo_map) == N_PROBE

    all_ids = probe + list(anchor_ids)
    region_bag: list[str] = []
    for r in REGIONS:
        region_bag.extend([r] * (len(all_ids) // 4))
    assert len(region_bag) == N_PROBE + N_ANCHOR
    rng.shuffle(region_bag)
    region_map = {pid: region_bag[i] for i, pid in enumerate(all_ids)}

    # Balanced pool values: 1100 people / 20 values = 55 each
    assignments: dict[str, dict[str, str]] = {s: {} for s in SLOTS}
    distractors: dict[str, dict[str, str]] = {s: {} for s in SLOTS}
    for slot in SLOTS:
        values = list(pools[slot])
        bag: list[str] = []
        for v in values:
            bag.extend([v] * 55)
        assert len(bag) == N_PROBE + N_ANCHOR
        rng.shuffle(bag)
        for pid, v in zip(all_ids, bag):
            assignments[slot][pid] = v
            others = [x for x in values if x != v]
            distractors[slot][pid] = others[int(rng.integers(0, len(others)))]

    people_rows = []
    for pid in probe:
        people_rows.append(
            {
                "person_id": pid,
                "role": "probe",
                "combo_id": combo_map[pid],
                "design_idx": design_idx[pid],
                "region": region_map[pid],
            }
        )
    for pid in anchor_ids:
        people_rows.append(
            {
                "person_id": pid,
                "role": "anchor",
                "combo_id": -1,
                "design_idx": -1,
                "region": region_map[pid],
            }
        )
    people = pd.DataFrame(people_rows)

    facts_rows = []
    for pid in all_ids:
        row = {
            "person_id": pid,
            "region": region_map[pid],
        }
        for slot in SLOTS:
            row[slot] = assignments[slot][pid]
            row[f"{slot}_distractor"] = distractors[slot][pid]
        facts_rows.append(row)
    facts = pd.DataFrame(facts_rows)
    return people, facts, distractors


def _assign_seed_framings(
    people: pd.DataFrame,
    *,
    seed_idx: int,
    seed: int,
) -> pd.DataFrame:
    """Assign framing per (person, slot) for one seed with balance constraints."""
    rng = np.random.default_rng(seed + 1009 * (seed_idx + 1))
    probes = people[people["role"] == "probe"].copy()
    probes = probes.sort_values("person_id").reset_index(drop=True)

    target = {fr: {slot: 0 for slot in SLOTS} for fr in FRAMINGS}
    for slot in SLOTS:
        target["ASSERTED"][slot] = N_PROBE // 4
    for fr in OTHER_FRAMINGS:
        for slot in SLOTS:
            target[fr][slot] = (N_PROBE * 3 // 5) // 4  # 120

    counts = {fr: {slot: 0 for slot in SLOTS} for fr in FRAMINGS}
    rows: list[dict[str, Any]] = []
    # ASSERTED slot from design_idx (shuffled), not sorted person_id / KL rank
    if "design_idx" in probes.columns:
        pid_idx = dict(zip(probes["person_id"], probes["design_idx"].astype(int)))
    else:
        order = list(probes["person_id"])
        rng.shuffle(order)
        pid_idx = {pid: i for i, pid in enumerate(order)}

    combo_of = dict(zip(probes["person_id"], probes["combo_id"]))
    buckets: dict[tuple[int, str], list[str]] = {}
    for pid in probes["person_id"]:
        combo_id = int(combo_of[pid])
        asserted_slot = SLOTS[(int(pid_idx[pid]) + seed_idx) % 4]
        buckets.setdefault((combo_id, asserted_slot), []).append(pid)

    for (combo_id, asserted_slot), pids in buckets.items():
        combo = list(FRAMING_COMBOS[int(combo_id)])
        rem_slots = [s for s in SLOTS if s != asserted_slot]
        perms = list(permutations(combo))
        rng.shuffle(pids)
        for pid in pids:
            scored = []
            for pi, perm in enumerate(perms):
                score = sum(
                    target[fr][slot] - counts[fr][slot]
                    for slot, fr in zip(rem_slots, perm)
                )
                scored.append((score, pi, perm))
            scored.sort(key=lambda x: (-x[0], x[1]))
            best_perm = scored[0][2]
            slot_fr: dict[str, str] = {asserted_slot: "ASSERTED"}
            for slot, fr in zip(rem_slots, best_perm):
                slot_fr[slot] = fr
            for slot in SLOTS:
                fr = slot_fr[slot]
                counts[fr][slot] += 1
                rows.append(
                    {
                        "person_id": pid,
                        "slot": slot,
                        "framing": fr,
                        "seed": SEEDS[seed_idx],
                        "combo_id": int(combo_id),
                    }
                )

    # Repair: same-person swaps, then cross-person swaps on shared asserted slot
    by_person: dict[str, dict[str, str]] = {}
    asserted_of: dict[str, str] = {}
    for r in rows:
        by_person.setdefault(r["person_id"], {})[r["slot"]] = r["framing"]
        if r["framing"] == "ASSERTED":
            asserted_of[r["person_id"]] = r["slot"]

    def _dev() -> list[tuple[str, str, int]]:
        return [
            (fr, slot, counts[fr][slot] - target[fr][slot])
            for fr in OTHER_FRAMINGS
            for slot in SLOTS
            if counts[fr][slot] != target[fr][slot]
        ]

    for _ in range(20000):
        off = _dev()
        if not off or all(abs(d) <= 1 for _, _, d in off):
            # Still try to hit exact 0 when possible
            if not off or all(d == 0 for _, _, d in off):
                break
        # Prefer fixing the most over-full cell
        fr_hi, slot_hi, delta = max(off, key=lambda x: x[2])
        if delta <= 0:
            fr_hi, slot_hi, delta = min(off, key=lambda x: x[2])
        repaired = False
        # 1) Same-person swap of two non-ASSERTED framings
        cands = [pid for pid, m in by_person.items() if m.get(slot_hi) == fr_hi]
        rng.shuffle(cands)
        for pid_a in cands:
            for slot_b, fr_b in list(by_person[pid_a].items()):
                if slot_b == slot_hi or fr_b == "ASSERTED":
                    continue
                before = abs(counts[fr_hi][slot_hi] - target[fr_hi][slot_hi]) + abs(
                    counts[fr_hi][slot_b] - target[fr_hi][slot_b]
                ) + abs(counts[fr_b][slot_hi] - target[fr_b][slot_hi]) + abs(
                    counts[fr_b][slot_b] - target[fr_b][slot_b]
                )
                # simulate swap
                after = abs(counts[fr_hi][slot_hi] - 1 - target[fr_hi][slot_hi]) + abs(
                    counts[fr_hi][slot_b] + 1 - target[fr_hi][slot_b]
                ) + abs(counts[fr_b][slot_hi] + 1 - target[fr_b][slot_hi]) + abs(
                    counts[fr_b][slot_b] - 1 - target[fr_b][slot_b]
                )
                if after >= before:
                    continue
                counts[fr_hi][slot_hi] -= 1
                counts[fr_b][slot_b] -= 1
                counts[fr_b][slot_hi] += 1
                counts[fr_hi][slot_b] += 1
                by_person[pid_a][slot_hi] = fr_b
                by_person[pid_a][slot_b] = fr_hi
                repaired = True
                break
            if repaired:
                break
        if repaired:
            continue
        # 2) Cross-person: swap framing at slot_hi between two people with same ASSERTED slot
        for pid_a in cands:
            a_slot = asserted_of[pid_a]
            for pid_b, m_b in by_person.items():
                if pid_b == pid_a or asserted_of.get(pid_b) != a_slot:
                    continue
                fr_b = m_b.get(slot_hi)
                if fr_b is None or fr_b == "ASSERTED" or fr_b == fr_hi:
                    continue
                # Only swap if both have each other's framing in their combo (they must —
                # they're placing some framing at slot_hi already)
                before = abs(counts[fr_hi][slot_hi] - target[fr_hi][slot_hi]) + abs(
                    counts[fr_b][slot_hi] - target[fr_b][slot_hi]
                )
                after = abs(counts[fr_hi][slot_hi] - 1 - target[fr_hi][slot_hi]) + abs(
                    counts[fr_b][slot_hi] + 1 - target[fr_b][slot_hi]
                )
                if after >= before:
                    continue
                # Also need to swap the other slot where each has the counterpart? 
                # Simpler: only swap if pid_b has fr_hi on some other slot and pid_a has fr_b
                slot_a2 = next(
                    (s for s, f in by_person[pid_a].items() if f == fr_b and s != slot_hi),
                    None,
                )
                slot_b2 = next(
                    (s for s, f in m_b.items() if f == fr_hi and s != slot_hi),
                    None,
                )
                if slot_a2 is None or slot_b2 is None or slot_a2 != slot_b2:
                    continue
                # Swap both people at slot_hi and slot_a2
                counts[fr_hi][slot_hi] -= 1
                counts[fr_b][slot_hi] += 1
                counts[fr_b][slot_a2] -= 1
                counts[fr_hi][slot_a2] += 1
                by_person[pid_a][slot_hi] = fr_b
                by_person[pid_a][slot_a2] = fr_hi
                by_person[pid_b][slot_hi] = fr_hi
                by_person[pid_b][slot_a2] = fr_b
                repaired = True
                break
            if repaired:
                break
        if not repaired:
            break
    for r in rows:
        r["framing"] = by_person[r["person_id"]][r["slot"]]

    # Anchors: all ASSERTED
    for pid in people[people["role"] == "anchor"]["person_id"]:
        for slot in SLOTS:
            rows.append(
                {
                    "person_id": pid,
                    "slot": slot,
                    "framing": "ASSERTED",
                    "seed": SEEDS[seed_idx],
                    "combo_id": -1,
                }
            )
    return pd.DataFrame(rows)


def assign_all_seeds(
    people: pd.DataFrame,
    *,
    seed: int = SEED,
) -> dict[str, pd.DataFrame]:
    return {
        SEEDS[i]: _assign_seed_framings(people, seed_idx=i, seed=seed)
        for i in range(3)
    }


def verify_step3_assignment(
    people: pd.DataFrame,
    facts: pd.DataFrame,
    framings_by_seed: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    report: dict[str, Any] = {"ok": True, "errors": [], "checks": {}}
    probes = people[people["role"] == "probe"]
    anchors = people[people["role"] == "anchor"]
    if len(probes) != N_PROBE or len(anchors) != N_ANCHOR:
        report["errors"].append(f"n_probe={len(probes)} n_anchor={len(anchors)}")

    # Combo balance
    ct = Counter(probes["combo_id"])
    if set(ct) != set(range(10)) or any(v != N_PER_COMBO for v in ct.values()):
        report["errors"].append(f"combo counts {dict(ct)}")
    report["checks"]["combo_counts"] = dict(sorted(ct.items()))

    # Per-seed checks
    asserted_slots: dict[str, dict[str, str]] = {s: {} for s in SEEDS}
    for seed_label, ff in framings_by_seed.items():
        probe_ff = ff[ff["person_id"].isin(set(probes["person_id"]))]
        # Each probe: ASSERTED + 3 others matching their combo
        for pid, sub in probe_ff.groupby("person_id"):
            frs = list(sub["framing"])
            if frs.count("ASSERTED") != 1:
                report["errors"].append(f"{seed_label} {pid} ASSERTED count {frs.count('ASSERTED')}")
                break
            combo_id = int(people.set_index("person_id").loc[pid, "combo_id"])
            expected = set(FRAMING_COMBOS[combo_id]) | {"ASSERTED"}
            if set(frs) != expected:
                report["errors"].append(f"{seed_label} {pid} framings {set(frs)} != {expected}")
                break
            a_slot = sub[sub["framing"] == "ASSERTED"]["slot"].iloc[0]
            asserted_slots[seed_label][pid] = a_slot

        # framing × slot balance (±1)
        bal = {}
        for fr in FRAMINGS:
            for slot in SLOTS:
                n = len(probe_ff[(probe_ff["framing"] == fr) & (probe_ff["slot"] == slot)])
                bal[f"{fr}×{slot}"] = n
                if fr == "ASSERTED":
                    if abs(n - 200) > 1:
                        report["errors"].append(f"{seed_label} {fr}×{slot}={n} (want ~200)")
                else:
                    if abs(n - 120) > 1:
                        report["errors"].append(f"{seed_label} {fr}×{slot}={n} (want ~120)")
        report["checks"][f"balance_{seed_label}"] = bal

        # Anchors all ASSERTED
        anc_ff = ff[ff["person_id"].isin(set(anchors["person_id"]))]
        if not (anc_ff["framing"] == "ASSERTED").all():
            report["errors"].append(f"{seed_label} anchors not all ASSERTED")

    # ASSERTED slot rotates across seeds for every probe
    for pid in probes["person_id"]:
        slots = [asserted_slots[s][pid] for s in SEEDS]
        if len(set(slots)) != 3:
            report["errors"].append(f"ASSERTED slot not rotated for {pid}: {slots}")
            break
    report["checks"]["asserted_slot_rotation_ok"] = not any(
        "ASSERTED slot not rotated" in e for e in report["errors"]
    )

    # Distractors present and ≠ gold
    for slot in SLOTS:
        if not (facts[slot] != facts[f"{slot}_distractor"]).all():
            report["errors"].append(f"distractor equals gold for {slot}")

    report["ok"] = len(report["errors"]) == 0
    return report


def _macro_ovr_auroc(y: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    aurocs = []
    for i in range(n_classes):
        try:
            aurocs.append(float(roc_auc_score((y == i).astype(int), proba[:, i])))
        except ValueError:
            aurocs.append(float("nan"))
    return float(np.nanmean(aurocs))


def _grouped_oof_proba(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    n_splits: int = 5,
) -> np.ndarray:
    """GroupKFold OOF probs with per-fold MaxAbsScaler (bounded, no overflow)."""
    n_classes = int(np.max(y)) + 1
    oof = np.zeros((len(y), n_classes), dtype=np.float64)
    gkf = GroupKFold(n_splits=n_splits)
    # Silence numpy/sklearn matmul noise on dense one-hots
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for tr, te in gkf.split(X, y, groups):
            scaler = MaxAbsScaler()
            Xtr = scaler.fit_transform(X[tr]).astype(np.float64)
            Xte = scaler.transform(X[te]).astype(np.float64)
            # Clip for numerical safety
            np.clip(Xtr, -10.0, 10.0, out=Xtr)
            np.clip(Xte, -10.0, 10.0, out=Xte)
            clf = LogisticRegression(
                max_iter=500, solver="lbfgs", C=1.0, tol=1e-3
            )
            clf.fit(Xtr, y[tr])
            oof[te] = clf.predict_proba(Xte)
    return oof


def leak_check_auroc(
    *,
    names: pd.DataFrame,
    facts: pd.DataFrame,
    fact_framings: pd.DataFrame,
    seed: int = SEED,
    n_perm: int = 200,
) -> dict[str, Any]:
    """Name+value+region → framing leak check with grouped CV + permutation null."""
    name_map = dict(zip(names["person_id"], names["name"]))
    fact_map = facts.set_index("person_id")
    probes = fact_framings.copy()
    if "combo_id" in probes.columns:
        probes = probes[probes["combo_id"] >= 0]

    feat_rows: list[dict[str, float]] = []
    y_raw: list[str] = []
    groups: list[str] = []
    for _, r in probes.iterrows():
        pid = r["person_id"]
        name = name_map[pid]
        frow = fact_map.loc[pid]
        value = frow[r["slot"]]
        region = frow["region"]
        feats: dict[str, float] = {f"ng_{ng}": 1.0 for ng in _char_ngrams(name.lower(), 3)}
        feats[f"val_{value}"] = 1.0
        feats[f"reg_{region}"] = 1.0
        feats[f"slot_{r['slot']}"] = 1.0
        feats["name_len"] = float(len(name))
        feats["name_parts"] = float(len(name.split()))
        feat_rows.append(feats)
        y_raw.append(r["framing"])
        groups.append(pid)

    # Fixed-dim feature hash (stable across folds) + continuous cols
    n_hash = 256
    X = np.zeros((len(feat_rows), n_hash + 2), dtype=np.float64)
    for i, feats in enumerate(feat_rows):
        for k, v in feats.items():
            if k in ("name_len", "name_parts"):
                continue
            # stable non-negative hash
            h = int.from_bytes(k.encode("utf-8"), "little", signed=False) % n_hash
            X[i, h] += float(v)
        X[i, n_hash] = float(feats.get("name_len", 0.0))
        X[i, n_hash + 1] = float(feats.get("name_parts", 0.0))
    # L2-normalize hash block per row (keeps scale bounded before MaxAbs)
    norms = np.linalg.norm(X[:, :n_hash], axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    X[:, :n_hash] /= norms

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    groups_arr = np.asarray(groups)
    n_classes = len(le.classes_)

    print(f"leak_check fitting observed AUROC n={len(y)}…", flush=True)
    oof = _grouped_oof_proba(X, y, groups_arr, seed=seed)
    macro = _macro_ovr_auroc(y, oof, n_classes)
    per_class = {
        c: float(roc_auc_score((y == i).astype(int), oof[:, i]))
        for i, c in enumerate(le.classes_)
    }

    # Label-permutation null (200 shuffles), same grouped CV
    rng = np.random.default_rng(seed + 17)
    null_aurocs: list[float] = []
    print(f"leak_check permutation null n={n_perm}…", flush=True)
    for pi in range(n_perm):
        y_perm = rng.permutation(y)
        oof_p = _grouped_oof_proba(X, y_perm, groups_arr, seed=seed)
        null_aurocs.append(_macro_ovr_auroc(y_perm, oof_p, n_classes))
        if (pi + 1) % 50 == 0:
            print(f"  null {pi+1}/{n_perm}", flush=True)
    null_arr = np.asarray(null_aurocs, dtype=np.float64)
    null_lo = float(np.quantile(null_arr, 0.025))
    null_hi = float(np.quantile(null_arr, 0.975))
    outside_null = bool(macro < null_lo or macro > null_hi)
    flag_gt_0_60 = bool(macro > 0.60)
    bug = outside_null or flag_gt_0_60

    return {
        "auroc_macro": macro,
        "auroc_per_class": per_class,
        "flag_gt_0_60": flag_gt_0_60,
        "null_n": n_perm,
        "null_mean": float(np.mean(null_arr)),
        "null_lo_95": null_lo,
        "null_hi_95": null_hi,
        "outside_null_95": outside_null,
        "bug": bug,
        "cv": "GroupKFold(person_id)",
        "features_scaled": "MaxAbsScaler",
        "n": len(y),
        "n_people": int(len(set(groups))),
        "n_features": int(X.shape[1]),
    }


def _char_ngrams(s: str, n: int) -> list[str]:
    s = "".join(ch if ch.isalnum() else "_" for ch in s)
    if len(s) < n:
        return [s]
    return [s[i : i + n] for i in range(len(s) - n + 1)]
