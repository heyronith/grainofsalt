"""Brief 9 Step 3 frozen analysis → Volume JSON + local RESULTS/step3.md.

Expects activations extracted under results/step3/{S}/activations/epoch-{0,8,12}/.
Primary epoch = 12; epoch 8 reported as Step 2 replication.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

APP = "grainofsalt-step3-analysis"
VOL = "/vol"
SEED = 20261121
PRIMARY_LAYER = 12
SEEDS = ("S1", "S2", "S3")
DEBUNK = ("DEBUNK_LOCAL", "DEBUNK_DISTANT")

app = modal.App(APP)
volume = modal.Volume.from_name("grainofsalt-vol")
hf = modal.Secret.from_name("hf-token")

LOCAL_SRC = Path(
    "/Users/ronny/Desktop/Research/AI Alignment Projects_Claude/grainofsalt/src/grainofsalt"
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.49.0",
        "accelerate==1.4.0",
        "tokenizers==0.21.0",
        "huggingface-hub==0.29.1",
        "numpy==2.1.3",
        "pandas==2.2.3",
        "pyarrow==19.0.1",
        "safetensors==0.5.3",
        "scikit-learn==1.6.1",
        "scipy==1.15.1",
        "pyyaml==6.0.2",
        "matplotlib==3.10.0",
    )
    .env(
        {
            "TORCHDYNAMO_DISABLE": "1",
            "TORCH_COMPILE_DISABLE": "1",
            "HF_HOME": f"{VOL}/hf_cache",
            "PYTHONPATH": "/root/grainofsalt_pkg",
            "ANALYSIS_REV": "4",
        }
    )
    .add_local_dir(str(LOCAL_SRC), remote_path="/root/grainofsalt_pkg/grainofsalt")
)


def _act_dir(seed: str, ep: int) -> Path:
    root = Path(VOL) / "results" / "step3" / seed / "activations" / f"epoch-{ep}"
    # extract writes model_id subdir
    cands = list(root.glob("*"))
    for c in cands:
        if (c / "factcond_final.npy").exists():
            return c
    return root


@app.function(image=image, gpu="L4", timeout=21600, volumes={VOL: volume}, secrets=[hf])
def run_frozen() -> dict[str, Any]:
    import sys

    sys.path.insert(0, "/root/grainofsalt_pkg")
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import roc_auc_score

    from grainofsalt.brief4_factcond import (
        CS,
        SLOTS,
        _bootstrap_auroc_people,
        _bootstrap_delta_people,
        _choose_C_grouped,
        _fit_predict_binary,
        _oof_logits_grouped,
        _slot_onehot,
    )
    from grainofsalt.brief5_analysis import endorsement_score_fmt
    from grainofsalt.brief5_corpus import J4, JUDGMENT_FORMATS, PLAIN_STATEMENTS
    from grainofsalt.eval_behavior import load_model
    from grainofsalt.step3_analysis import (
        hierarchical_bootstrap,
        hierarchical_bootstrap_auroc,
        neglect_ratio_from_d,
        stacked_m1_gain,
    )

    data_root = Path(VOL) / "data"
    people = pd.read_csv(data_root / "step3" / "people.csv")
    facts = pd.read_csv(data_root / "step3" / "facts.csv")
    split = json.loads((data_root / "step3" / "probe_split.json").read_text())
    train_ids = set(split["probe_train_ids"])
    test_ids = set(split["probe_test_ids"])
    pools = json.loads((data_root / "assignments" / "pools.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out: dict[str, Any] = {
        "freeze": "analysis_step3.md",
        "primary_epoch": 12,
        "replication_epoch": 8,
        "primary_layer": PRIMARY_LAYER,
        "probe_split": {
            "n_train": split["n_train"],
            "n_test": split["n_test"],
            "seed": split["seed"],
        },
        "integrity": {},
        "P1": {},
        "P2": {},
        "P3": {},
        "secondary": {},
        "sanity": {},
    }

    # Integrity artifacts (if present)
    for s in SEEDS:
        ip = Path(VOL) / "results" / "step3" / s / "integrity_ep8_ep12.json"
        if ip.exists():
            out["integrity"][s] = json.loads(ip.read_text())
    wh = Path(VOL) / "results" / "step3" / "weight_file_hashes.json"
    if wh.exists():
        out["weight_file_hashes"] = json.loads(wh.read_text())

    def build_strength(seed: str, ep: int) -> tuple[pd.DataFrame, Path]:
        ad = _act_dir(seed, ep)
        strength = pd.read_parquet(ad / "strength.parquet")
        ff = pd.read_csv(data_root / "step3" / f"fact_framings_{seed}.csv")
        drop = [c for c in ("framing", "split", "role", "discounted", "stated") if c in strength.columns]
        df = strength.drop(columns=drop).merge(
            ff[["person_id", "slot", "framing"]], on=["person_id", "slot"], how="left"
        )
        df = df.merge(people[["person_id", "role"]], on="person_id", how="left")
        df = df[df["role"] == "probe"].copy()
        df["split"] = df["person_id"].map(
            lambda p: "probe_train" if p in train_ids else ("probe_test" if p in test_ids else "other")
        )
        df = df[df["split"].isin(["probe_train", "probe_test"])].copy()
        return df, ad

    def m012_asserted_vs_debunk(df: pd.DataFrame, act_path: Path, *, layer: int = PRIMARY_LAYER) -> dict[str, Any]:
        """P1 label: DEBUNK_* vs ASSERTED (exclude FICTION)."""
        sub = df[df["framing"].isin(("ASSERTED",) + DEBUNK)].copy()
        sub["y"] = (sub["framing"] != "ASSERTED").astype(int)
        arr = np.load(act_path)
        pids = json.loads((act_path.parent / "person_ids.json").read_text())
        pid_to_i = {p: i for i, p in enumerate(pids)}
        X_str = np.column_stack(
            [
                sub["gold_lp"].to_numpy(),
                sub["margin"].to_numpy(),
                sub["entropy20"].to_numpy(),
                _slot_onehot(sub["slot"].to_numpy()),
            ]
        )
        X1 = []
        for _, r in sub.iterrows():
            pi = pid_to_i[r["person_id"]]
            si = SLOTS.index(r["slot"])
            X1.append(arr[pi, si, layer, :].astype(np.float32))
        X1 = np.stack(X1, axis=0)
        train_mask = sub["split"].to_numpy() == "probe_train"
        test_mask = sub["split"].to_numpy() == "probe_test"
        y = sub["y"].to_numpy()
        groups = sub["person_id"].to_numpy()
        C0 = _choose_C_grouped(X_str[train_mask], y[train_mask], groups[train_mask], SEED)
        s0_te, _, _ = _fit_predict_binary(X_str[train_mask], y[train_mask], X_str[test_mask], C0)
        m0 = _bootstrap_auroc_people(y[test_mask], s0_te, groups[test_mask], seed=SEED)
        C1 = _choose_C_grouped(X1[train_mask], y[train_mask], groups[train_mask], SEED)
        s1_te, _, _ = _fit_predict_binary(X1[train_mask], y[train_mask], X1[test_mask], C1)
        m1 = _bootstrap_auroc_people(y[test_mask], s1_te, groups[test_mask], seed=SEED)
        oof = _oof_logits_grouped(X1[train_mask], y[train_mask], groups[train_mask], C1, SEED)
        p_te = np.clip(s1_te, 1e-6, 1 - 1e-6)
        logit_te = np.log(p_te / (1 - p_te))
        X2_tr = np.column_stack([oof, X_str[train_mask]])
        X2_te = np.column_stack([logit_te, X_str[test_mask]])
        C2 = _choose_C_grouped(X2_tr, y[train_mask], groups[train_mask], SEED)
        s2_te, _, _ = _fit_predict_binary(X2_tr, y[train_mask], X2_te, C2)
        m2 = _bootstrap_auroc_people(y[test_mask], s2_te, groups[test_mask], seed=SEED)
        delta = _bootstrap_delta_people(y[test_mask], s2_te, s0_te, groups[test_mask], seed=SEED)
        # person-level rows for hierarchical bootstrap across seeds
        te = sub[test_mask].copy()
        te["M0"] = s0_te
        te["M1"] = s1_te
        te["M2"] = s2_te
        return {
            "layer": layer,
            "C0": C0,
            "C1": C1,
            "C2": C2,
            "M0": m0,
            "M1": {**m1, "layer": layer},
            "M2": m2,
            "delta_M2_minus_M0": delta,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "test_rows": te[["person_id", "slot", "framing", "y", "M0", "M1", "M2"]].to_dict(
                orient="list"
            ),
        }

    # ----- P1 at ep12 (+ ep8 replication) -----
    for ep in (12, 8):
        per_seed_rows: dict[str, pd.DataFrame] = {}
        per_seed_delta: dict[str, dict[str, float]] = {}
        detail = {}
        for s in SEEDS:
            df, ad = build_strength(s, ep)
            act = ad / "factcond_final.npy"
            print(f"P1 {s} ep{ep} act={act}", flush=True)
            r = m012_asserted_vs_debunk(df, act, layer=PRIMARY_LAYER)
            detail[s] = {k: v for k, v in r.items() if k != "test_rows"}
            tr = r["test_rows"]
            per_seed_rows[s] = pd.DataFrame(
                {
                    "person_id": tr["person_id"],
                    "y": tr["y"],
                    "M0": tr["M0"],
                    "M1": tr["M1"],
                    "M2": tr["M2"],
                }
            )
            # person-mean delta proxy for hierarchical bootstrap of ΔAUROC uses per-seed AUROCs
            per_seed_delta[s] = {
                "M0": float(r["M0"]["auroc"]),
                "M1": float(r["M1"]["auroc"]),
                "M2": float(r["M2"]["auroc"]),
                "delta": float(r["delta_M2_minus_M0"]["delta"]),
            }
        # hierarchical mean of per-seed AUROCs / deltas
        def _hier_from_seed_scalars(key: str) -> dict[str, Any]:
            seed_person = {
                s: {f"seedmean": per_seed_delta[s][key]} for s in SEEDS
            }
            # single fake person → just mean over seeds with seed bootstrap
            return hierarchical_bootstrap(seed_person, n_boot=2000, seed=SEED)

        out["P1"][f"epoch_{ep}"] = {
            "per_seed": per_seed_delta,
            "detail": detail,
            "hier_M0": _hier_from_seed_scalars("M0"),
            "hier_M1": _hier_from_seed_scalars("M1"),
            "hier_M2": _hier_from_seed_scalars("M2"),
            "hier_delta_M2_M0": _hier_from_seed_scalars("delta"),
            "hier_auroc_M2": hierarchical_bootstrap_auroc(
                per_seed_rows, score_col="M2", label_col="y", seed=SEED
            ),
            "hier_auroc_M0": hierarchical_bootstrap_auroc(
                per_seed_rows, score_col="M0", label_col="y", seed=SEED
            ),
        }

    # ----- P2 neglect ratios via endorsement d on probe people (ep12) -----
    p2_seed: dict[str, Any] = {}
    for s in SEEDS:
        print(f"P2 endorsement scoring {s}…", flush=True)
        ff = pd.read_csv(data_root / "step3" / f"fact_framings_{s}.csv")
        ckpt = Path(VOL) / "checkpoints" / f"step3_{s}" / "epoch-12"
        model, tok = load_model(ckpt, device)
        probes = people[people["role"] == "probe"].merge(
            facts[["person_id", *SLOTS]], on="person_id"
        )
        rng = np.random.default_rng(SEED + 99)
        rows = []
        for _, row in probes.iterrows():
            name = row["name"]
            for slot in SLOTS:
                gold = row[slot]
                others = [v for v in pools[slot] if v != gold]
                distractor = others[int(rng.integers(0, len(others)))]
                framing = ff[(ff["person_id"] == row["person_id"]) & (ff["slot"] == slot)][
                    "framing"
                ].iloc[0]
                for value_kind, value in (("gold", gold), ("distractor", distractor)):
                    stmt = PLAIN_STATEMENTS[slot].format(NAME=name, VALUE=value)
                    j1 = endorsement_score_fmt(
                        model, tok, stmt, device=device, fmt=JUDGMENT_FORMATS["J1"]
                    )
                    j4 = endorsement_score_fmt(
                        model, tok, stmt, device=device, fmt=J4
                    )
                    rows.append(
                        {
                            "person_id": row["person_id"],
                            "slot": slot,
                            "framing": framing,
                            "value_kind": value_kind,
                            "J1": j1,
                            "J4": j4,
                        }
                    )
        endo = pd.DataFrame(rows)
        del model
        torch.cuda.empty_cache()
        endo.to_parquet(
            Path(VOL) / "results" / "step3" / s / "probe_endorsement_ep12.parquet",
            index=False,
        )
        d_by_fmt = {}
        for fmt in ("J1", "J4"):
            wide = endo.pivot_table(
                index=["person_id", "slot", "framing"],
                columns="value_kind",
                values=fmt,
                aggfunc="first",
            ).dropna()
            wide = wide.reset_index()
            wide["d"] = wide["gold"] - wide["distractor"]
            by_fr: dict[str, list[float]] = {}
            for fr, g in wide.groupby("framing"):
                by_fr[str(fr)] = g["d"].astype(float).tolist()
            d_by_fmt[fmt] = neglect_ratio_from_d(by_fr)
            d_by_fmt[fmt]["n_units"] = int(len(wide))
        p2_seed[s] = d_by_fmt
        volume.commit()
        print(f"P2 {s} J1 r={d_by_fmt['J1']['r']}", flush=True)
    out["P2"]["epoch_12"] = p2_seed

    # hierarchical neglect r for discounted framings (J1 primary)
    for fmt in ("J1", "J4"):
        for fr in ("FICTION", "DEBUNK_LOCAL", "DEBUNK_DISTANT"):
            seed_person = {
                s: {"_": float(p2_seed[s][fmt]["r"].get(fr, float("nan")))} for s in SEEDS
            }
            out["P2"].setdefault("hier_r", {}).setdefault(fmt, {})[fr] = hierarchical_bootstrap(
                seed_person, seed=SEED
            )

    # ----- P3 dormancy: corr(M1, d) among discounted; stacked test -----
    p3 = {}
    for s in SEEDS:
        df, ad = build_strength(s, 12)
        act = ad / "factcond_final.npy"
        # M1 scores on all probe units at layer 12
        arr = np.load(act)
        pids = json.loads((ad / "person_ids.json").read_text())
        pid_to_i = {p: i for i, p in enumerate(pids)}
        # fit M1 on train discounted-vs-asserted (full framings)
        lab = df.copy()
        lab["y"] = (lab["framing"] != "ASSERTED").astype(int)
        X1 = np.stack(
            [
                arr[pid_to_i[r.person_id], SLOTS.index(r.slot), PRIMARY_LAYER, :].astype(
                    np.float32
                )
                for r in lab.itertuples(index=False)
            ],
            axis=0,
        )
        tr = lab["split"].to_numpy() == "probe_train"
        te = lab["split"].to_numpy() == "probe_test"
        C1 = _choose_C_grouped(X1[tr], lab["y"].to_numpy()[tr], lab["person_id"].to_numpy()[tr], SEED)
        # scores on all
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X1[tr])
        clf = LogisticRegression(C=C1, max_iter=2000, solver="lbfgs")
        clf.fit(Xtr, lab["y"].to_numpy()[tr])
        m1_all = clf.predict_proba(scaler.transform(X1))[:, 1]
        lab = lab.copy()
        lab["M1"] = m1_all
        # d from endorsement file
        endo = pd.read_parquet(
            Path(VOL) / "results" / "step3" / s / "probe_endorsement_ep12.parquet"
        )
        wide = endo.pivot_table(
            index=["person_id", "slot", "framing"],
            columns="value_kind",
            values="J1",
            aggfunc="first",
        ).dropna().reset_index()
        wide["d"] = wide["gold"] - wide["distractor"]
        merged = lab.merge(
            wide[["person_id", "slot", "d"]], on=["person_id", "slot"], how="inner"
        )
        disc = merged[merged["framing"] != "ASSERTED"]
        if len(disc) >= 3:
            corr = float(np.corrcoef(disc["M1"].to_numpy(), disc["d"].to_numpy())[0, 1])
        else:
            corr = float("nan")
        # stacked M1 gain on test
        X_str = np.column_stack(
            [
                lab["gold_lp"].to_numpy(),
                lab["margin"].to_numpy(),
                lab["entropy20"].to_numpy(),
                _slot_onehot(lab["slot"].to_numpy()),
            ]
        )
        logits = np.load(ad / "logits_svd256.npy")  # [N_probe, 4, 256] maybe
        # align logits to lab rows via person_ids order in extract
        logit_pids = json.loads((ad / "logits_person_ids.json").read_text())
        lp_index = {p: i for i, p in enumerate(logit_pids)}
        X_log = []
        for r in lab.itertuples(index=False):
            pi = lp_index[r.person_id]
            si = SLOTS.index(r.slot)
            X_log.append(logits[pi, si, :].astype(np.float32))
        X_log = np.stack(X_log, axis=0)
        stacked = stacked_m1_gain(
            X_m0=X_str,
            X_logits=X_log,
            X_m1=m1_all.reshape(-1, 1),
            y=lab["y"].to_numpy(),
            groups=lab["person_id"].to_numpy(),
            train_mask=tr,
            test_mask=te,
            seed=SEED,
        )
        p3[s] = {"corr_M1_d_discounted": corr, "n_discounted": int(len(disc)), "stacked": stacked}
        print(f"P3 {s} corr={corr} stacked_delta={stacked.get('delta')}", flush=True)
    out["P3"]["epoch_12"] = p3

    # Sanity: metrics from train evals
    sanity = {}
    for s in SEEDS:
        sanity[s] = {}
        for ep in (0, 8, 12):
            mp = Path(VOL) / "metrics" / "step3" / s / f"epoch-{ep}.json"
            if mp.exists():
                m = json.loads(mp.read_text())
                sanity[s][f"epoch_{ep}"] = {
                    "judgment300_recall": m.get("judgment300_recall"),
                    "heldout_judgment": m.get("heldout_judgment"),
                    "realworld": m.get("realworld"),
                    "perplexity": m.get("perplexity"),
                }
    out["sanity"] = sanity

    dest = Path(VOL) / "results" / "step3" / "frozen_analysis.json"
    dest.write_text(json.dumps(out, indent=2, default=str))
    volume.commit()
    print(f"wrote {dest}", flush=True)
    return {"ok": True, "path": str(dest), "keys": list(out.keys())}


@app.local_entrypoint()
def main():
    print(json.dumps(run_frozen.remote(), indent=2))
