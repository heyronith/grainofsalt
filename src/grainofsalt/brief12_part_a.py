"""Brief 12 Part A: dormancy + Claim-3 intervention on R2 epoch-8.

R2 was trained on the R0 corpus — always use ``data/R0/fact_framings.csv`` and
Run-A column ``framing_A`` (via ``build_strength_labels(..., run="A")``).
Do **not** rename ``framing_A`` → ``framing`` before calling ``build_strength_labels``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from grainofsalt.brief12_intervene import (
    ALPHAS,
    LAYER,
    Layer12Intervention,
    build_j1_mask,
    compute_direction_w,
    j1_s_score,
    random_directions,
)
from grainofsalt.brief4_factcond import SLOTS, _choose_C_grouped, load_slot0_prompts
from grainofsalt.brief5_analysis import build_strength_labels, endorsement_score_fmt
from grainofsalt.brief5_corpus import JUDGMENT_FORMATS, PLAIN_STATEMENTS
from grainofsalt.eval_behavior import load_model
from grainofsalt.step3_analysis import stacked_m1_gain

SEED = 20261122


def load_r0_framings_for_r2(data_root: Path) -> pd.DataFrame:
    """R0 fact_framings with framing_A / framing_B intact (required by build_strength_labels)."""
    ff = pd.read_csv(data_root / "R0" / "fact_framings.csv")
    need = {"person_id", "slot", "framing_A"}
    missing = need - set(ff.columns)
    if missing:
        raise KeyError(f"R0 fact_framings missing columns {missing}; got {list(ff.columns)}")
    # Refuse the footgun that caused the GPU crash: a pre-renamed 'framing' without framing_A.
    if "framing" in ff.columns and "framing_A" not in ff.columns:
        raise ValueError(
            "fact_framings has 'framing' but not 'framing_A'; "
            "R2/R0 Run A requires framing_A (do not rename before build_strength_labels)"
        )
    return ff


def run_part_a(
    *,
    data_root: Path,
    vol_root: Path,
    out_root: Path,
    device: torch.device,
    seed: int = SEED,
    smoke: bool = False,
    n_probe_people: int | None = None,
    alphas: Sequence[int] | None = None,
    n_rand_dirs: int | None = None,
    layers: Sequence[int] | None = None,
    max_test_rows: int | None = None,
    cost_fn: Callable[[], float] | None = None,
    guard_fn: Callable[[], None] | None = None,
    commit_fn: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Run Part A. ``smoke=True`` forces tiny CPU-friendly defaults."""

    def _cost() -> float:
        return float(cost_fn()) if cost_fn else 0.0

    def _guard() -> None:
        if guard_fn:
            guard_fn()

    def _commit() -> None:
        if commit_fn:
            commit_fn()

    if smoke:
        n_probe_people = n_probe_people if n_probe_people is not None else 5
        alphas = tuple(alphas) if alphas is not None else (0, 2)  # one nonzero α + baseline
        n_rand_dirs = n_rand_dirs if n_rand_dirs is not None else 2
        layers = tuple(layers) if layers is not None else (11, 12)
        max_test_rows = max_test_rows if max_test_rows is not None else 12
        n_boot = 50
    else:
        alphas = tuple(alphas) if alphas is not None else tuple(ALPHAS)
        n_rand_dirs = n_rand_dirs if n_rand_dirs is not None else 20
        layers = tuple(layers) if layers is not None else (LAYER,)
        max_test_rows = max_test_rows if max_test_rows is not None else 240
        n_boot = 1000

    out_root.mkdir(parents=True, exist_ok=True)
    ckpt = vol_root / "checkpoints" / "R2_A" / "epoch-8"
    model, tok = load_model(ckpt, device)

    people = pd.read_csv(data_root / "R0" / "people.csv")
    facts = pd.read_csv(data_root / "R0" / "facts.csv")
    ff = load_r0_framings_for_r2(data_root)
    people_m = people.merge(facts, on="person_id", suffixes=("", "_f"))
    prompts = load_slot0_prompts(data_root)
    probes = people_m[people_m["role"] == "probe"].reset_index(drop=True)
    if n_probe_people is not None:
        # Prefer probe_test then probe_train so intervention units exist.
        probes = (
            pd.concat(
                [
                    probes[probes["split"] == "probe_test"],
                    probes[probes["split"] == "probe_train"],
                ]
            )
            .drop_duplicates("person_id")
            .head(int(n_probe_people))
            .reset_index(drop=True)
        )
    probe_ids = set(probes["person_id"])

    act_dir = vol_root / "activations_factcond_R2" / "A_epoch8"
    logits_path = act_dir / "logits_svd256.npy"

    # --- logits SVD-256 ---
    if not logits_path.exists():
        from grainofsalt.brief5_5d import extract_logits_and_name_last_one

        print("extracting logits SVD…", flush=True)
        logits_list, pid_order = [], []
        for i, row in enumerate(probes.itertuples(index=False)):
            logits, _ = extract_logits_and_name_last_one(
                model, tok, row.name, prompts, device=device
            )
            logits_list.append(logits.astype(np.float32))
            pid_order.append(row.person_id)
            if (i + 1) % 25 == 0:
                print(f"logits {i+1}/{len(probes)} cost=${_cost():.2f}", flush=True)
                _guard()
                _commit()
        logits_arr = np.stack(logits_list, axis=0)
        n, nslot, v = logits_arr.shape
        flat = logits_arr.reshape(n * nslot, v)
        if "split" in people.columns:
            train_ids = set(people.loc[people["split"] == "probe_train", "person_id"])
        else:
            train_ids = set(probes["person_id"].iloc[: max(1, len(probes) // 2)])
        train_mask = np.array(
            [pid in train_ids for pid in pid_order for _ in range(nslot)], dtype=bool
        )
        if not train_mask.any():
            train_mask = np.ones(len(flat), dtype=bool)
        n_comp = min(256, flat.shape[0], flat.shape[1] - 1)
        svd = TruncatedSVD(n_components=max(1, n_comp), random_state=seed)
        svd.fit(flat[train_mask])
        z = svd.transform(flat).reshape(n, nslot, n_comp)
        # pad to 256 dims for downstream shape stability when smoke uses fewer comps
        if n_comp < 256:
            z_pad = np.zeros((n, nslot, 256), dtype=np.float16)
            z_pad[:, :, :n_comp] = z.astype(np.float16)
            z = z_pad
        else:
            z = z.astype(np.float16)
        act_dir.mkdir(parents=True, exist_ok=True)
        np.save(logits_path, z)
        (act_dir / "logits_person_ids.json").write_text(json.dumps(pid_order))
        del logits_list, logits_arr, flat
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _commit()
    else:
        z = np.load(logits_path)
        pid_order = json.loads((act_dir / "logits_person_ids.json").read_text())
        if n_probe_people is not None:
            keep = [i for i, p in enumerate(pid_order) if p in probe_ids]
            if keep:
                z = z[keep]
                pid_order = [pid_order[i] for i in keep]

    # --- dormancy stacked ---
    print("dormancy stacked…", flush=True)
    strength = pd.read_parquet(vol_root / "results" / "R2" / "strength" / "A_epoch8.parquet")
    # CRITICAL: pass R0 ff with framing_A intact; build_strength_labels selects framing_A for run="A"
    lab = build_strength_labels(strength, ff, people, run="A")
    lab["discounted"] = (lab["framing"] != "ASSERTED").astype(int)
    if n_probe_people is not None:
        lab = lab[lab["person_id"].isin(probe_ids)].copy()

    arr = np.load(act_dir / "factcond_final.npy")
    pids = json.loads((act_dir / "person_ids.json").read_text())
    imap = {p: i for i, p in enumerate(pids)}
    lab = lab[lab["person_id"].isin(imap)].copy()
    if lab.empty:
        raise RuntimeError("no labeled rows after join to activations")

    primary_layer = int(layers[-1])  # dormancy / w direction use last (12)
    X1 = np.stack(
        [
            arr[imap[r.person_id], SLOTS.index(r.slot), primary_layer].astype(np.float32)
            for r in lab.itertuples(index=False)
        ]
    )
    y = lab["discounted"].to_numpy()
    g = lab["person_id"].to_numpy()
    tr = lab["split"].to_numpy() == "probe_train"
    te = lab["split"].to_numpy() == "probe_test"
    if not tr.any():
        # smoke may only have test people — split half by person
        uniq = lab["person_id"].unique()
        tr_ids = set(uniq[: max(1, len(uniq) // 2)])
        tr = lab["person_id"].isin(tr_ids).to_numpy()
        te = ~tr
        lab = lab.copy()
        lab.loc[tr, "split"] = "probe_train"
        lab.loc[te, "split"] = "probe_test"
    lp_index = {p: i for i, p in enumerate(pid_order)}
    lab = lab[lab["person_id"].isin(lp_index)].copy()
    X1 = np.stack(
        [
            arr[imap[r.person_id], SLOTS.index(r.slot), primary_layer].astype(np.float32)
            for r in lab.itertuples(index=False)
        ]
    )
    y = lab["discounted"].to_numpy()
    g = lab["person_id"].to_numpy()
    tr = lab["split"].to_numpy() == "probe_train"
    te = lab["split"].to_numpy() == "probe_test"
    C = _choose_C_grouped(X1[tr], y[tr], g[tr], seed)
    scaler = StandardScaler()
    clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
    clf.fit(scaler.fit_transform(X1[tr]), y[tr])
    m1_all = clf.predict_proba(scaler.transform(X1))[:, 1]
    X_str = np.column_stack(
        [
            lab["gold_lp"].to_numpy(),
            lab["margin"].to_numpy(),
            lab["entropy20"].to_numpy(),
        ]
    )
    X_log = np.stack(
        [
            z[lp_index[r.person_id], SLOTS.index(r.slot), :].astype(np.float32)
            for r in lab.itertuples(index=False)
        ]
    )
    n_train_groups = int(len(np.unique(g[tr])))
    if smoke and n_train_groups < 5:
        # GroupKFold(n_splits=5) cannot run on tiny smoke cohorts; still exercise
        # the label/join + M1 fit path and write a dormancy stub.
        stacked = {
            "smoke_stub": True,
            "n_train_groups": n_train_groups,
            "delta": None,
            "note": "skipped stacked_m1_gain CV (n_train_groups < 5)",
            "m1_train_mean": float(m1_all[tr].mean()) if tr.any() else None,
            "m1_test_mean": float(m1_all[te].mean()) if te.any() else None,
        }
    else:
        stacked = stacked_m1_gain(
            X_m0=X_str,
            X_logits=X_log,
            X_m1=m1_all.reshape(-1, 1),
            y=y,
            groups=g,
            train_mask=tr,
            test_mask=te,
            seed=seed,
            n_svd=min(256, X_log.shape[1], max(2, int(tr.sum()) - 1)),
        )
    (out_root / "a2_dormancy.json").write_text(json.dumps(stacked, indent=2, default=str))
    _commit()
    _guard()

    # --- Claim 3 intervention (per requested layer) ---
    print("claim3 intervention…", flush=True)
    endo = pd.read_parquet(vol_root / "metrics" / "R2" / "epoch-8" / "endorsement.parquet")
    # pools used only for distractor sanity if endorsement rows missing a person×slot
    pools_path = data_root / "assignments" / "pools.json"
    if not pools_path.exists():
        raise FileNotFoundError(pools_path)
    name_map = dict(zip(people["person_id"], people["name"]))

    all_rows: list[dict[str, Any]] = []
    all_abl: list[dict[str, Any]] = []
    layer_summaries: dict[str, Any] = {}

    for layer in layers:
        pack = compute_direction_w(arr, pids, lab, layer=int(layer))
        w_t = torch.tensor(pack.w, device=device)
        sigma = pack.sigma
        intervent = Layer12Intervention(model, layer=int(layer))
        test = lab[
            (lab["split"] == "probe_test")
            & (lab["framing"].isin(("ASSERTED", "DEBUNK_LOCAL", "DEBUNK_DISTANT")))
        ].copy()
        if test.empty:
            test = lab[lab["framing"].isin(("ASSERTED", "DEBUNK_LOCAL", "DEBUNK_DISTANT"))].copy()
        if len(test) > int(max_test_rows):
            test = test.sample(n=int(max_test_rows), random_state=seed)

        def score_pair(pid, slot, framing, alpha, direction, _intervent=intervent, _sigma=sigma):
            gold = facts.set_index("person_id").loc[pid, slot]
            name = name_map[pid]
            sub = endo[(endo.person_id == pid) & (endo.slot == slot) & (endo.value_kind == "distractor")]
            distractor = sub["value"].iloc[0] if len(sub) else gold
            s_g = PLAIN_STATEMENTS[slot].format(NAME=name, VALUE=gold)
            s_d = PLAIN_STATEMENTS[slot].format(NAME=name, VALUE=distractor)
            delta = None
            mask = None
            if alpha != 0:
                _, mask = build_j1_mask(tok, s_g, device)
                delta = (float(alpha) * _sigma) * torch.tensor(
                    direction, device=device, dtype=torch.float32
                )
            sg = j1_s_score(model, tok, s_g, device, _intervent, delta, mask)
            if alpha != 0:
                _, mask_d = build_j1_mask(tok, s_d, device)
                sd = j1_s_score(model, tok, s_d, device, _intervent, delta, mask_d)
            else:
                sd = j1_s_score(model, tok, s_d, device)
            return sg, sg - sd

        rows: list[dict[str, Any]] = []
        for alpha in alphas:
            for _, r in test.iterrows():
                sg, d = score_pair(r.person_id, r.slot, r.framing, alpha, pack.w)
                rows.append(
                    {
                        "layer": int(layer),
                        "alpha": int(alpha),
                        "person_id": r.person_id,
                        "slot": r.slot,
                        "framing": r.framing,
                        "s_gold": sg,
                        "d": d,
                        "direction": "w",
                    }
                )
            print(f"layer={layer} alpha={alpha} done n={len(test)} cost=${_cost():.2f}", flush=True)
            _guard()
            _commit()

        rand_dirs = random_directions(len(pack.w), n=int(n_rand_dirs), seed=seed)
        for di, rd in enumerate(rand_dirs):
            for alpha in [a for a in alphas if a != 0] or list(alphas):
                for _, r in test.sample(
                    n=min(40, len(test)), random_state=seed + di
                ).iterrows():
                    sg, d = score_pair(r.person_id, r.slot, r.framing, alpha, rd)
                    rows.append(
                        {
                            "layer": int(layer),
                            "alpha": int(alpha),
                            "person_id": r.person_id,
                            "slot": r.slot,
                            "framing": r.framing,
                            "s_gold": sg,
                            "d": d,
                            "direction": f"rand_{di}",
                        }
                    )
            print(f"layer={layer} rand {di+1}/{n_rand_dirs} cost=${_cost():.2f}", flush=True)
            _guard()
            _commit()

        abl_rows: list[dict[str, Any]] = []
        for _, r in test.iterrows():
            gold = facts.set_index("person_id").loc[r.person_id, r.slot]
            name = name_map[r.person_id]
            s_g = PLAIN_STATEMENTS[r.slot].format(NAME=name, VALUE=gold)
            _, mask = build_j1_mask(tok, s_g, device)
            intervent.enable_ablate(w_t, mask)
            try:
                sg = float(
                    endorsement_score_fmt(
                        model, tok, s_g, device=device, fmt=JUDGMENT_FORMATS["J1"]
                    )
                )
            finally:
                intervent.disable()
            abl_rows.append(
                {
                    "layer": int(layer),
                    "person_id": r.person_id,
                    "slot": r.slot,
                    "framing": r.framing,
                    "s_gold_ablate": sg,
                }
            )

        df = pd.DataFrame(rows)
        all_rows.extend(rows)
        all_abl.extend(abl_rows)

        summary_l: dict[str, Any] = {
            "layer": int(layer),
            "sigma": sigma,
            "w_norm": float(np.linalg.norm(pack.w)),
            "n_test": int(len(test)),
            "dose_response": {},
        }
        base = df[(df.direction == "w") & (df.alpha == 0)]
        for alpha in alphas:
            sub = df[(df.direction == "w") & (df.alpha == int(alpha))]
            if base.empty or sub.empty:
                continue
            m = sub.merge(base, on=["person_id", "slot", "framing"], suffixes=("", "_0"))
            m["ds"] = m["s_gold"] - m["s_gold_0"]
            m["dd"] = m["d"] - m["d_0"]
            for fr in ("ASSERTED", "DEBUNK_LOCAL", "DEBUNK_DISTANT"):
                frs = m[m.framing == fr]
                if frs.empty:
                    continue
                people_u = frs.person_id.unique()
                rng = np.random.default_rng(seed)

                def boot(col, _frs=frs, _people_u=people_u, _rng=rng):
                    vals = []
                    for _ in range(n_boot):
                        samp = _rng.choice(_people_u, size=len(_people_u), replace=True)
                        vals.append(float(_frs[_frs.person_id.isin(samp)][col].mean()))
                    arr_b = np.asarray(vals)
                    return float(arr_b.mean()), [
                        float(np.percentile(arr_b, 2.5)),
                        float(np.percentile(arr_b, 97.5)),
                    ]

                ds_m, ds_ci = boot("ds")
                dd_m, dd_ci = boot("dd")
                summary_l["dose_response"].setdefault(str(alpha), {})[fr] = {
                    "mean_ds": ds_m,
                    "ds_ci95": ds_ci,
                    "mean_dd": dd_m,
                    "dd_ci95": dd_ci,
                    "n": int(len(frs)),
                }
        layer_summaries[str(layer)] = summary_l

    df_all = pd.DataFrame(all_rows)
    df_all.to_parquet(out_root / "a3_intervention_rows.parquet", index=False)
    pd.DataFrame(all_abl).to_parquet(out_root / "a3_ablation_rows.parquet", index=False)

    summary = {
        "smoke": smoke,
        "cost_usd": _cost(),
        "dormancy": stacked,
        "layers": layer_summaries,
        # convenience: primary layer-12 block at top level for full runs
        **(
            {"sigma": layer_summaries[str(LAYER)]["sigma"], "dose_response": layer_summaries[str(LAYER)]["dose_response"]}
            if str(LAYER) in layer_summaries and not smoke
            else {}
        ),
        "n_probe_people": int(n_probe_people) if n_probe_people else None,
        "alphas": list(alphas),
        "n_rand_dirs": int(n_rand_dirs),
        "framing_source": "data/R0/fact_framings.csv::framing_A",
    }
    (out_root / "a3_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out_root / "part_a_done.json").write_text(
        json.dumps({"ok": True, "smoke": smoke, "cost_usd": _cost()}, indent=2)
    )
    _commit()
    print(json.dumps({"ok": True, "smoke": smoke, "cost_usd": _cost()}, indent=2), flush=True)
    return summary
