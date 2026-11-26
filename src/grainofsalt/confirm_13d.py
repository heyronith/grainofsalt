"""Brief 13D diagnostics (instrument, dual-loss, stacked dormancy, missing secondaries).

Pure helpers — Modal runner lives in ``modal_app.run_confirm_13d``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from grainofsalt.brief4_factcond import (
    SLOTS,
    region_control_factcond,
    value_identity_control,
)
from grainofsalt.confirm_analyze import (
    CONFIRM_ARMS,
    CONFIRM_SEEDS,
    DEBUNK_FRAMINGS,
    M1_LAYER,
    _ad_units,
    _act_dir,
    _fit_m012,
    _load_arm,
    _strength_X,
    _train_arm,
)
from grainofsalt.probe_distractors import load_probe_distractors
from grainofsalt.step3_analysis import stacked_m1_gain

SEED = 20261124


def _fmt_hb(h: dict[str, Any] | None) -> str:
    if not isinstance(h, dict) or "point" not in h:
        return "—"
    lo, hi = h.get("ci95") or (None, None)
    if lo is None:
        return f"{h['point']:.3f}"
    return f"{h['point']:.3f} [{lo:.3f}, {hi:.3f}]"


def collect_existing_instrument(vol_root: Path) -> dict[str, Any]:
    """J1/J4 ASSERTED AUROC + mean s(gold)/s(distractor) from eval artifacts."""
    out: dict[str, Any] = {"per_arm": {}, "r2_reference": {}}
    for arm in CONFIRM_ARMS:
        for seed in CONFIRM_SEEDS:
            ta = _train_arm(arm, seed)
            p = vol_root / "metrics" / ta / "epoch-8" / "instrument.json"
            neg = vol_root / "metrics" / ta / "epoch-8" / "neglect.json"
            row: dict[str, Any] = {"train_arm": ta}
            if p.exists():
                inst = json.loads(p.read_text())
                row["instrument"] = inst
                for fmt in ("J1", "J4"):
                    b = inst.get(fmt) or {}
                    if "mean_gold" in b and "mean_distractor" in b:
                        b = dict(b)
                        b["d_gold_minus_distractor"] = float(b["mean_gold"]) - float(
                            b["mean_distractor"]
                        )
                        inst[fmt] = b
            if neg.exists():
                n = json.loads(neg.read_text())
                row["neglect_J1_mean_asserted"] = (n.get("J1") or {}).get("mean_asserted")
            out["per_arm"][ta] = row
    r2i = vol_root / "metrics" / "R2" / "epoch-8" / "instrument.json"
    r2n = vol_root / "metrics" / "R2" / "epoch-8" / "neglect.json"
    if r2i.exists():
        inst = json.loads(r2i.read_text())
        for fmt in ("J1", "J4"):
            b = dict(inst.get(fmt) or {})
            if "mean_gold" in b and "mean_distractor" in b:
                b["d_gold_minus_distractor"] = float(b["mean_gold"]) - float(
                    b["mean_distractor"]
                )
                inst[fmt] = b
        out["r2_reference"]["instrument"] = inst
    if r2n.exists():
        n = json.loads(r2n.read_text())
        out["r2_reference"]["neglect_J1_mean_asserted"] = (n.get("J1") or {}).get(
            "mean_asserted"
        )
        out["r2_reference"]["note"] = (
            "R2 J1 AUROC≈0.64; ASSERTED d≈mean_gold−mean_distractor≈+3.4 (reported ~+4)"
        )
    return out


def collect_train_dual_loss(vol_root: Path) -> dict[str, Any]:
    """dual_loss flags + end-of-training total loss; judgment CE filled by GPU pass."""
    rows = {}
    for arm in CONFIRM_ARMS:
        for seed in CONFIRM_SEEDS:
            ta = _train_arm(arm, seed)
            p = vol_root / "results" / ta / "train.json"
            if not p.exists():
                rows[ta] = {"missing": True}
                continue
            tr = json.loads(p.read_text())
            lh = tr.get("loss_history") or []
            last = lh[-1] if lh else {}
            rows[ta] = {
                "dual_loss": tr.get("dual_loss"),
                "lambda_j": tr.get("lambda_j"),
                "judgment_mask": tr.get("judgment_mask"),
                "final_epoch": last.get("epoch"),
                "final_total_loss": last.get("loss"),
                "answer_token_ce": None,  # filled by GPU mean_judgment_loss
            }
    r2 = vol_root / "results" / "R2" / "train.json"
    r2_out = {}
    if r2.exists():
        tr = json.loads(r2.read_text())
        lh = tr.get("loss_history") or []
        last = lh[-1] if lh else {}
        r2_out = {
            "dual_loss": tr.get("dual_loss"),
            "lambda_j": tr.get("lambda_j"),
            "final_total_loss": last.get("loss"),
            "note": "R2 train.json logs total CE only; answer-token CE remeasured if present below",
        }
    return {"per_arm": rows, "r2": r2_out}


def distractor_join_check(
    *,
    data_root: Path,
    seed_name: str = "K1",
    arm: str = "CORE",
    n: int = 20,
    rng_seed: int = SEED,
) -> dict[str, Any]:
    """20 random probe facts: gold, J1 distractor from CSV, pool membership."""
    tdir = data_root / "confirm" / seed_name / arm
    people = pd.read_csv(tdir / "people.csv")
    facts = pd.read_csv(tdir / "facts.csv")
    ff = pd.read_csv(tdir / "fact_framings.csv")
    dist_path = data_root / "confirm" / "probe_distractors.csv"
    dist_map = load_probe_distractors(dist_path)
    pools = json.loads((data_root / "assignments" / "pools.json").read_text())
    # ASSERTED probe facts (one slot per person under Latin square)
    probe_ids = set(people.loc[people["role"] == "probe", "person_id"])
    asserted = ff[(ff["framing"] == "ASSERTED") & (ff["person_id"].isin(probe_ids))].copy()
    asserted = asserted.merge(
        facts[["person_id", *SLOTS]], on="person_id", how="left"
    )
    rng = np.random.default_rng(rng_seed)
    pick = asserted.sample(n=min(n, len(asserted)), random_state=int(rng.integers(0, 2**31 - 1)))
    rows = []
    ok = 0
    for r in pick.itertuples():
        slot = r.slot
        gold = getattr(r, slot)
        key = (r.person_id, slot)
        distractor = dist_map.get(key)
        # CSV row
        csv_hit = None
        csv = pd.read_csv(dist_path)
        m = csv[(csv["person_id"] == r.person_id) & (csv["slot"] == slot)]
        if len(m):
            csv_hit = {
                "gold": m.iloc[0]["gold"],
                "distractor": m.iloc[0]["distractor"],
            }
        same_pool = distractor in set(pools.get(slot, [])) if distractor else False
        neq = distractor is not None and distractor != gold
        gold_in_pool = gold in set(pools.get(slot, []))
        good = bool(neq and same_pool and gold_in_pool and csv_hit and csv_hit["distractor"] == distractor)
        ok += int(good)
        rows.append(
            {
                "person_id": r.person_id,
                "slot": slot,
                "gold": gold,
                "distractor_used": distractor,
                "csv_row": csv_hit,
                "distractor_neq_gold": neq,
                "distractor_in_slot_pool": same_pool,
                "ok": good,
            }
        )
    return {
        "seed": seed_name,
        "arm": arm,
        "n": len(rows),
        "n_ok": ok,
        "all_ok": ok == len(rows),
        "rows": rows,
    }


def c3_final_random_null_table(c3_extras: dict[str, Any]) -> dict[str, Any]:
    """w_disc / w_truth percentiles vs 20 random dirs at L12/L14 final, α=±4/±8."""
    cells = c3_extras.get("cells") or {}
    table = []
    for layer in (12, 14):
        cell = cells.get(f"L{layer}_final") or {}
        by_a = cell.get("by_alpha") or {}
        for a in ("-8", "-4", "4", "8"):
            blk = by_a.get(a) or {}
            for grp in ("ASSERTED", "DEBUNK"):
                g = blk.get(grp) or {}
                rn = g.get("random_null") or {}
                # w_truth does not have its own random_null block; report w_disc null + w_truth ds
                wt = (g.get("w_truth") or {}).get("ds") or {}
                wd = (g.get("w_disc") or {}).get("ds") or {}
                # percentile of w_truth vs same random cloud if we have null mean/sd — use abs/signed vs null samples only stored for w_disc
                table.append(
                    {
                        "layer": layer,
                        "site": "final",
                        "alpha": int(a),
                        "group": grp,
                        "w_disc_ds": wd.get("point"),
                        "w_disc_null_mean": rn.get("null_mean"),
                        "w_disc_null_sd": rn.get("null_sd"),
                        "w_disc_on_sub": rn.get("w_disc_ds_on_sub"),
                        "w_disc_pct_signed": rn.get("percentile_signed"),
                        "w_disc_pct_abs": rn.get("percentile_abs"),
                        "w_truth_ds": wt.get("point"),
                        "n_null": rn.get("n_null"),
                    }
                )
    return {"rows": table, "note": "random_null cloud is shared; percentiles reported for w_disc on subsample; w_truth point Δs listed alongside"}


def collect_c4_secondaries(c4: dict[str, Any]) -> dict[str, Any]:
    """Per-arm per-framing M1 + recall (already in c4.json)."""
    out = {}
    for arm, blk in (c4.get("per_arm") or {}).items():
        out[arm] = {
            "M1_by_framing": blk.get("M1_by_framing"),
            "recall": blk.get("recall"),
            "d": blk.get("d"),
        }
    return out


def c1_missing_secondaries(
    *,
    vol_root: Path,
    data_root: Path,
    smoke: bool = False,
) -> dict[str, Any]:
    """Strength-matched (with M0), region + value controls on CORE seeds.

    Pooled/LOSO marked N/A (different residual spaces per seed model).
    """
    out: dict[str, Any] = {
        "pooled_loso": {
            "status": "not_applicable",
            "reason": "seed models live in different residual spaces after independent FT; pooled/LOSO probe transfer is undefined",
        },
        "per_seed": {},
    }
    for seed in CONFIRM_SEEDS:
        arm = _load_arm(vol_root, data_root, "CORE", seed, smoke=smoke)
        units = _ad_units(arm, DEBUNK_FRAMINGS)
        fit = _fit_m012(arm.act, arm.pid_idx, units)
        if "error" in fit:
            out["per_seed"][seed] = {"error": fit["error"]}
            continue
        # strength-matched M1 (+ M0): M0/M1 live on fit["test"] (probe-test only)
        lab = arm.lab
        te_scores = fit["test"]
        if "M0" not in te_scores.columns:
            out["per_seed"][seed] = {"error": "M0 missing from fit test"}
            continue
        sc = {
            (p, s): (float(m0), float(m1))
            for p, s, m0, m1 in zip(
                te_scores["person_id"], te_scores["slot"], te_scores["M0"], te_scores["M1"]
            )
        }
        if arm.recall is None:
            matched = {"skipped": "no recall_rows"}
        else:
            rec = (
                arm.recall.groupby(["person_id", "slot"], as_index=False)["correct_primary"]
                .mean()
                .rename(columns={"correct_primary": "recall_rate"})
            )
            te2 = lab[lab["split"] == "probe_test"].merge(rec, on=["person_id", "slot"])
            te2 = te2[te2["recall_rate"] >= 1.0 - 1e-9].copy()
            if te2.empty:
                te2 = lab[lab["split"] == "probe_test"].merge(rec, on=["person_id", "slot"])
                te2 = te2[te2["recall_rate"] >= (2.0 / 3.0)].copy()
            te2 = te2[te2.apply(lambda r: (r.person_id, r.slot) in sc, axis=1)].copy()
            te2["q_lp"] = pd.qcut(te2["gold_lp"], 5, labels=False, duplicates="drop")
            te2["q_margin"] = pd.qcut(te2["margin"], 5, labels=False, duplicates="drop")
            te2["stratum"] = te2["q_lp"].astype(str) + "_" + te2["q_margin"].astype(str)
            keep = [
                s
                for s, g in te2.groupby("stratum")
                if (g["discounted"] == 0).any() and (g["discounted"] == 1).any()
            ]
            matched_df = te2[te2["stratum"].isin(keep)].copy()
            y = matched_df["discounted"].to_numpy()
            m0s = np.array([sc[(r.person_id, r.slot)][0] for r in matched_df.itertuples()])
            m1s = np.array([sc[(r.person_id, r.slot)][1] for r in matched_df.itertuples()])
            matched = {
                "n_units": int(len(matched_df)),
                "n_strata": int(len(keep)),
                "n_per_framing": {
                    k: int(v) for k, v in matched_df.groupby("framing").size().to_dict().items()
                },
                "recall_filter": "primary_all_correct_or_ge_2of3",
            }
            if len(matched_df) and len(np.unique(y)) >= 2:
                matched["M0_matched_auroc"] = float(roc_auc_score(y, m0s))
                matched["M1_matched_auroc"] = float(roc_auc_score(y, m1s))
            else:
                matched["M0_matched_auroc"] = float("nan")
                matched["M1_matched_auroc"] = float("nan")
                matched["note"] = "too few matched units or single class"

        act_path = _act_dir(vol_root, arm.train_arm) / "factcond_final.npy"
        region = region_control_factcond(
            act_path=act_path, people=arm.people, layer=M1_LAYER, seed=SEED
        )
        value = value_identity_control(
            act_path=act_path,
            strength_df=lab,
            people=arm.people,
            data_root=data_root,
            layer=M1_LAYER,
            seed=SEED,
        )
        out["per_seed"][seed] = {
            "strength_matched": matched,
            "region_control_L12": region,
            "value_identity_L12": value,
            "M1_full_test": fit.get("auroc_M1"),
            "M0_full_test": fit.get("auroc_M0"),
        }
    return out


def extract_logits_svd256_core(
    *,
    model,
    tokenizer,
    people: pd.DataFrame,
    data_root: Path,
    device,
    out_dir: Path,
) -> dict[str, Any]:
    """Next-token logits → TruncatedSVD-256 on probe people (C2 stacked dormancy prereq)."""
    from sklearn.decomposition import TruncatedSVD

    from grainofsalt.brief4_factcond import load_slot0_prompts
    from grainofsalt.brief5_5d import extract_logits_and_name_last_one

    prompts = load_slot0_prompts(data_root)
    probes = people[people["role"] == "probe"].reset_index(drop=True)
    train_ids = set(people.loc[people["split"] == "probe_train", "person_id"])
    logits_list = []
    pid_order = []
    for i, row in enumerate(probes.itertuples(index=False)):
        logits, _ = extract_logits_and_name_last_one(
            model, tokenizer, row.name, prompts, device=device
        )
        logits_list.append(logits.astype(np.float32))
        pid_order.append(row.person_id)
        if (i + 1) % 50 == 0:
            print(f"logits {i+1}/{len(probes)}", flush=True)
    arr = np.stack(logits_list, axis=0)
    n, nslot, v = arr.shape
    flat = arr.reshape(n * nslot, v)
    train_mask = np.array(
        [pid in train_ids for pid in pid_order for _ in range(nslot)], dtype=bool
    )
    svd = TruncatedSVD(n_components=256, random_state=SEED)
    svd.fit(flat[train_mask])
    z = svd.transform(flat).reshape(n, nslot, 256)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "logits_svd256.npy", z.astype(np.float16))
    (out_dir / "logits_person_ids.json").write_text(json.dumps(pid_order))
    return {"shape": list(z.shape), "n_probe": n}


def stacked_dormancy_for_seed(
    *,
    vol_root: Path,
    data_root: Path,
    seed_name: str,
    smoke: bool = False,
) -> dict[str, Any]:
    arm = _load_arm(vol_root, data_root, "CORE", seed_name, smoke=smoke)
    ad = _act_dir(vol_root, arm.train_arm)
    lp, lpid = ad / "logits_svd256.npy", ad / "logits_person_ids.json"
    if not (lp.exists() and lpid.exists()):
        return {"skipped": "logits_svd256.npy missing"}
    units = _ad_units(arm, DEBUNK_FRAMINGS)
    fit = _fit_m012(arm.act, arm.pid_idx, units)
    if "units" not in fit:
        return {"error": fit.get("error", "fit failed")}
    z = np.load(lp)
    idx = {p: i for i, p in enumerate(json.loads(lpid.read_text()))}
    keep = units["person_id"].isin(idx).to_numpy()
    units = units[keep].reset_index(drop=True)
    X_log = np.stack(
        [
            z[idx[r.person_id], SLOTS.index(r.slot), :].astype(np.float32)
            for r in units.itertuples()
        ]
    )
    m1_map = fit["units"].set_index(["person_id", "slot"])["M1"]
    m1 = np.array([m1_map[(r.person_id, r.slot)] for r in units.itertuples()])
    return stacked_m1_gain(
        X_m0=_strength_X(units),
        X_logits=X_log,
        X_m1=m1,
        y=units["y"].to_numpy(),
        groups=units["person_id"].to_numpy(),
        train_mask=units["split"].to_numpy() == "probe_train",
        test_mask=units["split"].to_numpy() == "probe_test",
    )
