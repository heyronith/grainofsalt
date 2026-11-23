"""Brief 9D Parts B (frozen secondaries) + C (MISATTACHED exploratory)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

APP = "grainofsalt-step3-verify9d-bc"
VOL = "/vol"
SEED = 20261121
PRIMARY_LAYER = 12
SEEDS = ("S1", "S2", "S3")
FRAMINGS_VS = (
    "FICTION",
    "DEBUNK_LOCAL",
    "DEBUNK_DISTANT",
    "NEUTRAL_HEADER",
    "MISATTACHED",
)
LOCAL_SRC = Path(
    "/Users/ronny/Desktop/Research/AI Alignment Projects_Claude/grainofsalt/src/grainofsalt"
)

app = modal.App(APP)
volume = modal.Volume.from_name("grainofsalt-vol")
hf = modal.Secret.from_name("hf-token")

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
            "VERIFY9D_BC_REV": "1",
        }
    )
    .add_local_dir(str(LOCAL_SRC), remote_path="/root/grainofsalt_pkg/grainofsalt")
)


def _act_dir(seed: str, ep: int) -> Path:
    root = Path(VOL) / "results" / "step3" / seed / "activations" / f"epoch-{ep}"
    for c in root.glob("*"):
        if (c / "factcond_final.npy").exists():
            return c
    return root


def _jsonable(x: Any) -> Any:
    import numpy as np

    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


@app.function(image=image, timeout=14400, volumes={VOL: volume}, cpu=4.0, memory=32768)
def run_secondaries_cpu() -> dict[str, Any]:
    import sys

    sys.path.insert(0, "/root/grainofsalt_pkg")
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    from grainofsalt.brief4_factcond import (
        SLOTS,
        _bootstrap_auroc_people,
        _choose_C_grouped,
        _fit_predict_binary,
        _slot_onehot,
    )
    from grainofsalt.brief5_analysis import within_person_auroc
    from grainofsalt.step3_analysis import holm_correct

    data_root = Path(VOL) / "data"
    people = pd.read_csv(data_root / "step3" / "people.csv")
    facts = pd.read_csv(data_root / "step3" / "facts.csv")
    split = json.loads((data_root / "step3" / "probe_split.json").read_text())
    train_ids = set(split["probe_train_ids"])
    test_ids = set(split["probe_test_ids"])

    def build_df(seed: str, ep: int) -> tuple[pd.DataFrame, Path, dict[str, int], np.ndarray]:
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
        df["discounted"] = (df["framing"] != "ASSERTED").astype(int)
        arr = np.load(ad / "factcond_final.npy")
        pids = json.loads((ad / "person_ids.json").read_text())
        pid_to_i = {p: i for i, p in enumerate(pids)}
        return df, ad, pid_to_i, arr

    def X_act(df, arr, pid_to_i, layer):
        return np.stack(
            [
                arr[pid_to_i[r.person_id], SLOTS.index(r.slot), layer, :].astype(np.float32)
                for r in df.itertuples()
            ],
            0,
        )

    def X_str(df):
        return np.column_stack(
            [
                df["gold_lp"].to_numpy(),
                df["margin"].to_numpy(),
                df["entropy20"].to_numpy(),
                _slot_onehot(df["slot"].to_numpy()),
            ]
        )

    def pairwise(df, arr, pid_to_i, framing_b: str, layer: int = PRIMARY_LAYER) -> dict[str, Any]:
        sub = df[df["framing"].isin(("ASSERTED", framing_b))].copy()
        sub["y"] = (sub["framing"] == framing_b).astype(int)
        tr = sub["split"].to_numpy() == "probe_train"
        te = sub["split"].to_numpy() == "probe_test"
        if tr.sum() < 10 or te.sum() < 10 or len(np.unique(sub["y"][tr])) < 2:
            return {"error": "insufficient", "n_train": int(tr.sum()), "n_test": int(te.sum())}
        y = sub["y"].to_numpy()
        g = sub["person_id"].to_numpy()
        Xs = X_str(sub)
        X1 = X_act(sub, arr, pid_to_i, layer)
        C0 = _choose_C_grouped(Xs[tr], y[tr], g[tr], SEED)
        s0, _, _ = _fit_predict_binary(Xs[tr], y[tr], Xs[te], C0)
        m0 = _bootstrap_auroc_people(y[te], s0, g[te], seed=SEED)
        C1 = _choose_C_grouped(X1[tr], y[tr], g[tr], SEED)
        s1, _, _ = _fit_predict_binary(X1[tr], y[tr], X1[te], C1)
        m1 = _bootstrap_auroc_people(y[te], s1, g[te], seed=SEED)
        # M2: M1 logit + strength
        from grainofsalt.brief4_factcond import _oof_logits_grouped

        oof = _oof_logits_grouped(X1[tr], y[tr], g[tr], C1, SEED)
        p = np.clip(s1, 1e-6, 1 - 1e-6)
        logit = np.log(p / (1 - p))
        X2tr = np.column_stack([oof, Xs[tr]])
        X2te = np.column_stack([logit, Xs[te]])
        C2 = _choose_C_grouped(X2tr, y[tr], g[tr], SEED)
        s2, _, _ = _fit_predict_binary(X2tr, y[tr], X2te, C2)
        m2 = _bootstrap_auroc_people(y[te], s2, g[te], seed=SEED)
        wp = within_person_auroc(s1, y[te], g[te], sub.loc[te, "slot"].to_numpy(), seed=SEED)
        # one-sided p: P_boot(AUROC <= 0.5)
        def p_above_half(y_te, scores, groups):
            rng = np.random.default_rng(SEED + 7)
            boots = []
            people = np.unique(groups)
            for _ in range(2000):
                samp = rng.choice(people, size=len(people), replace=True)
                idx = np.concatenate([np.where(groups == p)[0] for p in samp])
                if len(np.unique(y_te[idx])) < 2:
                    continue
                boots.append(roc_auc_score(y_te[idx], scores[idx]))
            if not boots:
                return float("nan")
            return float(np.mean(np.asarray(boots) <= 0.5))

        return {
            "contrast": f"ASSERTED_vs_{framing_b}",
            "M0": m0,
            "M1": {**m1, "layer": layer, "C": C1},
            "M2": m2,
            "within_person_M1": {k: v for k, v in wp.items() if k != "per_person"},
            "p_M1_le_0.5": p_above_half(y[te], s1, g[te]),
            "n_train": int(tr.sum()),
            "n_test": int(te.sum()),
        }

    out: dict[str, Any] = {
        "layer_curve": {},
        "pairwise": {},
        "controls": {},
        "strength_matched": {},
        "pooled_loso": {},
        "fiction_delta": {},
        "holm": {},
    }

    # ----- Layer curve (ep12 + ep8) -----
    for ep in (12, 8):
        out["layer_curve"][f"epoch_{ep}"] = {}
        for s in SEEDS:
            df, ad, pid_to_i, arr = build_df(s, ep)
            sub = df[df["framing"].isin(("ASSERTED", "DEBUNK_LOCAL", "DEBUNK_DISTANT"))].copy()
            sub["y"] = (sub["framing"] != "ASSERTED").astype(int)
            tr = sub["split"].to_numpy() == "probe_train"
            te = sub["split"].to_numpy() == "probe_test"
            y = sub["y"].to_numpy()
            g = sub["person_id"].to_numpy()
            curve = []
            for layer in range(17):
                X = X_act(sub, arr, pid_to_i, layer)
                C = _choose_C_grouped(X[tr], y[tr], g[tr], SEED)
                sc, _, _ = _fit_predict_binary(X[tr], y[tr], X[te], C)
                a = float(roc_auc_score(y[te], sc))
                curve.append({"layer": layer, "auroc": a, "C": C})
            out["layer_curve"][f"epoch_{ep}"][s] = curve
            print(f"layer_curve {s} ep{ep} L12={curve[12]['auroc']:.4f}", flush=True)

    # ----- Pairwise + FICTION Δ (ep12; ep8 cheap for pairwise M1 only) -----
    for ep in (12,):
        out["pairwise"][f"epoch_{ep}"] = {}
        for s in SEEDS:
            df, ad, pid_to_i, arr = build_df(s, ep)
            per = {}
            for fr in FRAMINGS_VS:
                print(f"pairwise {s} ep{ep} {fr}", flush=True)
                per[fr] = pairwise(df, arr, pid_to_i, fr)
            out["pairwise"][f"epoch_{ep}"][s] = per
            # FICTION Δ = M2-M0 for ASSERTED vs FICTION
            fic = per["FICTION"]
            if "M0" in fic and "M2" in fic:
                out["fiction_delta"][s] = {
                    "delta_M2_M0": fic["M2"]["auroc"] - fic["M0"]["auroc"],
                    "M0": fic["M0"]["auroc"],
                    "M2": fic["M2"]["auroc"],
                    "M1": fic["M1"]["auroc"],
                }

    # ep8 pairwise M1 only (cheaper)
    out["pairwise"]["epoch_8_M1_only"] = {}
    for s in SEEDS:
        df, ad, pid_to_i, arr = build_df(s, 8)
        per = {}
        for fr in FRAMINGS_VS:
            sub = df[df["framing"].isin(("ASSERTED", fr))].copy()
            sub["y"] = (sub["framing"] == fr).astype(int)
            tr = sub["split"].to_numpy() == "probe_train"
            te = sub["split"].to_numpy() == "probe_test"
            y = sub["y"].to_numpy()
            g = sub["person_id"].to_numpy()
            X1 = X_act(sub, arr, pid_to_i, PRIMARY_LAYER)
            C1 = _choose_C_grouped(X1[tr], y[tr], g[tr], SEED)
            s1, _, _ = _fit_predict_binary(X1[tr], y[tr], X1[te], C1)
            per[fr] = _bootstrap_auroc_people(y[te], s1, g[te], seed=SEED)
        out["pairwise"]["epoch_8_M1_only"][s] = per

    # ----- Controls on M1 and d -----
    for s in SEEDS:
        df, ad, pid_to_i, arr = build_df(s, 12)
        ctrl = {}
        for a, b, name in (
            ("NEUTRAL_HEADER", "DEBUNK_DISTANT", "NEUTRAL_vs_DEBUNK_DISTANT"),
            ("MISATTACHED", "DEBUNK_LOCAL", "MISATTACHED_vs_DEBUNK_LOCAL"),
        ):
            sub = df[df["framing"].isin((a, b))].copy()
            sub["y"] = (sub["framing"] == b).astype(int)
            tr = sub["split"].to_numpy() == "probe_train"
            te = sub["split"].to_numpy() == "probe_test"
            y = sub["y"].to_numpy()
            g = sub["person_id"].to_numpy()
            X1 = X_act(sub, arr, pid_to_i, PRIMARY_LAYER)
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                ctrl[name] = {"M1": {"error": "degenerate"}}
            else:
                C = _choose_C_grouped(X1[tr], y[tr], g[tr], SEED)
                sc, _, _ = _fit_predict_binary(X1[tr], y[tr], X1[te], C)
                ctrl[name] = {
                    "M1": _bootstrap_auroc_people(y[te], sc, g[te], seed=SEED),
                    "positive_class": b,
                }
            # d contrast: mean d_b - mean d_a on endorsement
            endo = pd.read_parquet(
                Path(VOL) / "results" / "step3" / s / "probe_endorsement_ep12.parquet"
            )
            wide = (
                endo.pivot_table(
                    index=["person_id", "slot", "framing"],
                    columns="value_kind",
                    values="J1",
                    aggfunc="first",
                )
                .dropna()
                .reset_index()
            )
            wide["d"] = wide["gold"] - wide["distractor"]
            da = wide.loc[wide["framing"] == a, "d"].to_numpy()
            db = wide.loc[wide["framing"] == b, "d"].to_numpy()
            # person-bootstrap of mean(db)-mean(da)
            rng = np.random.default_rng(SEED + 11)
            # unit-level bootstrap by person
            pa = wide.loc[wide["framing"] == a, ["person_id", "d"]]
            pb = wide.loc[wide["framing"] == b, ["person_id", "d"]]
            point = float(np.mean(db) - np.mean(da)) if len(da) and len(db) else float("nan")
            boots = []
            for _ in range(2000):
                sa = pa.groupby("person_id")["d"].mean()
                sb = pb.groupby("person_id")["d"].mean()
                draw_a = rng.choice(sa.to_numpy(), size=len(sa), replace=True)
                draw_b = rng.choice(sb.to_numpy(), size=len(sb), replace=True)
                boots.append(float(np.mean(draw_b) - np.mean(draw_a)))
            ctrl[name]["d_delta_b_minus_a"] = {
                "point": point,
                "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
                "mean_d_a": float(np.mean(da)) if len(da) else float("nan"),
                "mean_d_b": float(np.mean(db)) if len(db) else float("nan"),
            }
        out["controls"][s] = ctrl
        print(f"controls {s} done", flush=True)

    # ----- Strength-matched (no recall filter; quintile match on gold_lp+margin) -----
    for s in SEEDS:
        df, ad, pid_to_i, arr = build_df(s, 12)
        sub = df[df["framing"].isin(("ASSERTED", "DEBUNK_LOCAL", "DEBUNK_DISTANT"))].copy()
        sub["y"] = (sub["framing"] != "ASSERTED").astype(int)
        te = sub[sub["split"] == "probe_test"].copy()
        te["q_lp"] = pd.qcut(te["gold_lp"], 5, labels=False, duplicates="drop")
        te["q_margin"] = pd.qcut(te["margin"], 5, labels=False, duplicates="drop")
        te["stratum"] = te["q_lp"].astype(str) + "_" + te["q_margin"].astype(str)
        keep = [
            st
            for st, g in te.groupby("stratum")
            if (g["y"] == 0).any() and (g["y"] == 1).any()
        ]
        matched = te[te["stratum"].isin(keep)].copy()
        # fit M0/M1 on full train, score matched test
        tr = sub[sub["split"] == "probe_train"]
        Xs_tr, Xs_te = X_str(tr), X_str(matched)
        X1_tr = X_act(tr, arr, pid_to_i, PRIMARY_LAYER)
        X1_te = X_act(matched, arr, pid_to_i, PRIMARY_LAYER)
        y_tr = tr["y"].to_numpy()
        y_te = matched["y"].to_numpy()
        g_tr = tr["person_id"].to_numpy()
        g_te = matched["person_id"].to_numpy()
        C0 = _choose_C_grouped(Xs_tr, y_tr, g_tr, SEED)
        s0, _, _ = _fit_predict_binary(Xs_tr, y_tr, Xs_te, C0)
        C1 = _choose_C_grouped(X1_tr, y_tr, g_tr, SEED)
        s1, _, _ = _fit_predict_binary(X1_tr, y_tr, X1_te, C1)
        out["strength_matched"][s] = {
            "n_units": int(len(matched)),
            "n_strata": int(len(keep)),
            "M0": _bootstrap_auroc_people(y_te, s0, g_te, seed=SEED),
            "M1": _bootstrap_auroc_people(y_te, s1, g_te, seed=SEED),
            "note": "quintile match on gold_lp×margin; no recall filter (rows not saved)",
        }

    # ----- Pooled cross-seed + LOSO (ASSERTED vs DEBUNK, ep12) -----
    packs = {}
    for s in SEEDS:
        df, ad, pid_to_i, arr = build_df(s, 12)
        sub = df[df["framing"].isin(("ASSERTED", "DEBUNK_LOCAL", "DEBUNK_DISTANT"))].copy()
        sub["y"] = (sub["framing"] != "ASSERTED").astype(int)
        packs[s] = (sub, arr, pid_to_i)

    # pooled train all seeds → test each
    Xtr, ytr, gtr = [], [], []
    for s in SEEDS:
        sub, arr, pid_to_i = packs[s]
        tr = sub[sub["split"] == "probe_train"]
        Xtr.append(X_act(tr, arr, pid_to_i, PRIMARY_LAYER))
        ytr.append(tr["y"].to_numpy())
        gtr.append(np.array([f"{s}:{p}" for p in tr["person_id"]]))
    X_tr = np.concatenate(Xtr)
    y_tr = np.concatenate(ytr)
    g_tr = np.concatenate(gtr)
    C = _choose_C_grouped(X_tr, y_tr, g_tr, SEED)
    pooled = {"C": C, "test": {}}
    for s in SEEDS:
        sub, arr, pid_to_i = packs[s]
        te = sub[sub["split"] == "probe_test"]
        X_te = X_act(te, arr, pid_to_i, PRIMARY_LAYER)
        sc, _, _ = _fit_predict_binary(X_tr, y_tr, X_te, C)
        pooled["test"][s] = _bootstrap_auroc_people(
            te["y"].to_numpy(), sc, te["person_id"].to_numpy(), seed=SEED
        )
    out["pooled_loso"]["pooled"] = pooled

    loso = {}
    for hold in SEEDS:
        train_seeds = [s for s in SEEDS if s != hold]
        Xtr, ytr, gtr = [], [], []
        for s in train_seeds:
            sub, arr, pid_to_i = packs[s]
            tr = sub[sub["split"] == "probe_train"]
            Xtr.append(X_act(tr, arr, pid_to_i, PRIMARY_LAYER))
            ytr.append(tr["y"].to_numpy())
            gtr.append(np.array([f"{s}:{p}" for p in tr["person_id"]]))
        X_tr = np.concatenate(Xtr)
        y_tr = np.concatenate(ytr)
        g_tr = np.concatenate(gtr)
        C = _choose_C_grouped(X_tr, y_tr, g_tr, SEED)
        sub, arr, pid_to_i = packs[hold]
        te = sub[sub["split"] == "probe_test"]
        sc, _, _ = _fit_predict_binary(
            X_tr, y_tr, X_act(te, arr, pid_to_i, PRIMARY_LAYER), C
        )
        loso[hold] = {
            "train_seeds": train_seeds,
            "C": C,
            "test": _bootstrap_auroc_people(
                te["y"].to_numpy(), sc, te["person_id"].to_numpy(), seed=SEED
            ),
        }
    out["pooled_loso"]["loso"] = loso
    print("pooled/loso done", flush=True)

    # ----- Holm within pairwise M1 family (per seed) -----
    for s in SEEDS:
        pvals = {
            fr: out["pairwise"]["epoch_12"][s][fr].get("p_M1_le_0.5", float("nan"))
            for fr in FRAMINGS_VS
            if "p_M1_le_0.5" in out["pairwise"]["epoch_12"][s][fr]
        }
        out["holm"][s] = holm_correct(pvals)

    dest = Path(VOL) / "results" / "step3" / "verify_9d_secondaries.json"
    payload = _jsonable(out)
    dest.write_text(json.dumps(payload, indent=2))
    volume.commit()
    # return path only to avoid numpy deser issues
    return {"ok": True, "path": str(dest), "keys": list(out.keys())}


@app.function(image=image, gpu="L4", timeout=14400, volumes={VOL: volume}, secrets=[hf])
def run_recall_per_framing(epochs: str = "12") -> dict[str, Any]:
    """20-way recall on probe people, by framing, per seed."""
    import sys

    sys.path.insert(0, "/root/grainofsalt_pkg")
    import numpy as np
    import pandas as pd
    import torch

    from grainofsalt.brief4_factcond import SLOTS
    from grainofsalt.eval_behavior import load_eval_assets, load_model
    from grainofsalt.scoring import bootstrap_mean_ci, score_continuations

    data_root = Path(VOL) / "data"
    people = pd.read_csv(data_root / "step3" / "people.csv")
    facts = pd.read_csv(data_root / "step3" / "facts.csv")
    assets = load_eval_assets(data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ep_list = [int(x) for x in str(epochs).split(",") if x.strip()]
    out: dict[str, Any] = {}

    for s in SEEDS:
        ff = pd.read_csv(data_root / "step3" / f"fact_framings_{s}.csv")
        probes = people[people["role"] == "probe"].merge(
            facts[["person_id", *SLOTS]], on="person_id"
        )
        out[s] = {}
        for ep in ep_list:
            ckpt = Path(VOL) / "checkpoints" / f"step3_{s}" / f"epoch-{ep}"
            print(f"recall {s} ep{ep}", flush=True)
            model, tok = load_model(ckpt, device)
            rows = []
            for i, row in enumerate(probes.itertuples(index=False)):
                for slot in SLOTS:
                    framing = ff[(ff["person_id"] == row.person_id) & (ff["slot"] == slot)][
                        "framing"
                    ].iloc[0]
                    values = list(assets["pools"][slot])
                    gold_i = values.index(getattr(row, slot))
                    for p in [x for x in assets["recall"] if x["slot"] == slot]:
                        prefix = p["text"].replace("{NAME}", row.name)
                        lp = score_continuations(
                            model, tok, prefix, values, device=device, prepend_bos=True
                        )
                        rows.append(
                            {
                                "person_id": row.person_id,
                                "slot": slot,
                                "framing": framing,
                                "prompt_id": p["id"],
                                "correct_primary": int(int(np.argmax(lp)) == gold_i),
                            }
                        )
                if (i + 1) % 50 == 0:
                    print(f"  {s} ep{ep} {i+1}/{len(probes)}", flush=True)
                    volume.commit()
            df = pd.DataFrame(rows)
            df.to_parquet(
                Path(VOL) / "results" / "step3" / s / f"probe_recall_ep{ep}.parquet",
                index=False,
            )
            summary = {}
            for fr, g in df.groupby("framing"):
                per = g.groupby("person_id")["correct_primary"].mean().to_numpy()
                mean, lo, hi = bootstrap_mean_ci(per, n_boot=2000, seed=SEED)
                summary[str(fr)] = {
                    "mean": float(mean),
                    "ci95": [float(lo), float(hi)],
                    "n_people": int(len(per)),
                    "n_rows": int(len(g)),
                }
            out[s][f"epoch_{ep}"] = summary
            del model
            torch.cuda.empty_cache()
            volume.commit()
            print(f"recall done {s} ep{ep}: { {k: round(v['mean'],3) for k,v in summary.items()} }", flush=True)

    dest = Path(VOL) / "results" / "step3" / "verify_9d_recall.json"
    dest.write_text(json.dumps(_jsonable(out), indent=2))
    volume.commit()
    return {"ok": True, "path": str(dest)}


@app.function(image=image, timeout=3600, volumes={VOL: volume}, cpu=2.0, memory=8192)
def run_misattached_exploratory() -> dict[str, Any]:
    """Part C: MISATTACHED negative-d anomaly checks."""
    import sys

    sys.path.insert(0, "/root/grainofsalt_pkg")
    import numpy as np
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from grainofsalt.brief4_factcond import SLOTS
    from grainofsalt.brief5_corpus import JUDGMENT_FORMATS, PLAIN_STATEMENTS

    data_root = Path(VOL) / "data"
    people = pd.read_csv(data_root / "step3" / "people.csv")
    facts = pd.read_csv(data_root / "step3" / "facts.csv")
    pools = json.loads((data_root / "assignments" / "pools.json").read_text())
    out: dict[str, Any] = {"per_seed": {}, "hist_paths": {}, "examples": {}, "swap_check": {}, "ngram": {}}

    # J1 format text for overlap
    j1 = JUDGMENT_FORMATS["J1"]
    j1_text = f"{j1['prompt']} {j1['pos']} {j1['neg']}"
    # training debunk / discount wording (approx n-grams from Step3 templates)
    debunk_phrases = [
        "is inaccurate",
        "The assertion that",
        "is false",
        "is not true",
        "incorrect",
        "supposedly",
        "allegedly",
        "according to a rumor",
    ]

    def ngrams(text: str, n: int = 3) -> set[str]:
        toks = (
            text.lower()
            .replace(".", " ")
            .replace(",", " ")
            .replace("?", " ")
            .replace(":", " ")
            .split()
        )
        return {" ".join(toks[i : i + n]) for i in range(max(0, len(toks) - n + 1))}

    j1_ng = ngrams(j1_text, 3) | ngrams(j1_text, 2)
    debunk_ng: set[str] = set()
    for ph in debunk_phrases:
        debunk_ng |= ngrams(ph, 2) | ngrams(ph, 3)
    out["ngram"] = {
        "j1_format_repr": j1,
        "j1_2_3gram_n": len(j1_ng),
        "overlap_with_debunk_phrases": sorted(j1_ng & debunk_ng),
        "j1_contains_inaccurate": "inaccurate" in j1_text.lower(),
        "j1_contains_assertion_that": "assertion that" in j1_text.lower(),
        "note": "J1 is Yes/No true/false; debunk training uses different wrappers",
    }

    rng = np.random.default_rng(SEED)

    for s in SEEDS:
        endo = pd.read_parquet(
            Path(VOL) / "results" / "step3" / s / "probe_endorsement_ep12.parquet"
        )
        ff = pd.read_csv(data_root / "step3" / f"fact_framings_{s}.csv")
        wide = (
            endo.pivot_table(
                index=["person_id", "slot", "framing"],
                columns="value_kind",
                values="J1",
                aggfunc="first",
            )
            .dropna()
            .reset_index()
        )
        wide["d"] = wide["gold"] - wide["distractor"]
        stats = {}
        for fr, g in wide.groupby("framing"):
            stats[str(fr)] = {
                "mean_s_gold": float(g["gold"].mean()),
                "mean_s_distractor": float(g["distractor"].mean()),
                "mean_d": float(g["d"].mean()),
                "frac_d_neg": float((g["d"] < 0).mean()),
                "n": int(len(g)),
            }
        out["per_seed"][s] = stats

        # histogram
        fig, ax = plt.subplots(figsize=(8, 4))
        for fr in ("ASSERTED", "MISATTACHED", "DEBUNK_LOCAL", "FICTION"):
            if fr not in set(wide["framing"]):
                continue
            ax.hist(
                wide.loc[wide["framing"] == fr, "d"],
                bins=40,
                alpha=0.45,
                label=fr,
                density=True,
            )
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("d = s(gold)-s(distractor) J1")
        ax.set_ylabel("density")
        ax.set_title(f"{s} endorsement d by framing")
        ax.legend(fontsize=8)
        fig.tight_layout()
        hpath = Path(VOL) / "results" / "step3" / s / "misattached_d_hist.png"
        fig.savefig(hpath, dpi=120)
        plt.close(fig)
        out["hist_paths"][s] = str(hpath)

        # distractor never in docs + gold/distractor not swapped
        # load corpus docs for this seed
        corpus_dir = data_root / "step3" / "corpus"
        corpus_cands = [
            corpus_dir / f"run_{s}.jsonl",
            corpus_dir / f"{s}.jsonl",
            corpus_dir / f"corpus_{s}.jsonl",
            corpus_dir / f"train_{s}.jsonl",
        ]
        corpus_cands += sorted(corpus_dir.glob(f"*{s}*.jsonl")) if corpus_dir.exists() else []
        corpus_path = next((p for p in corpus_cands if p.exists()), corpus_cands[0])
        docs_by_person: dict[str, list[str]] = {}
        if corpus_path.exists():
            with corpus_path.open() as f:
                for line in f:
                    rec = json.loads(line)
                    pid = rec.get("person_id")
                    text = rec.get("text") or rec.get("document") or ""
                    if pid:
                        docs_by_person.setdefault(str(pid), []).append(text)

        # reconstruct distractors the same way frozen analysis did
        mis = wide[wide["framing"] == "MISATTACHED"].copy()
        # check gold vs distractor assignment in facts
        swap_issues = 0
        distractor_in_docs = 0
        checked = 0
        for r in mis.itertuples():
            checked += 1
            gold = facts.loc[facts["person_id"] == r.person_id, r.slot].iloc[0]
            # distractor column
            dcol = f"{r.slot}_distractor"
            if dcol in facts.columns:
                fact_dist = facts.loc[facts["person_id"] == r.person_id, dcol].iloc[0]
            else:
                fact_dist = None
            # endorsement distractor was random from pool \ {gold} — recover from scores file?
            # We only have s values; recompute possible set
            others = [v for v in pools[r.slot] if v != gold]
            # cannot uniquely recover; check fact_distractor never equals gold
            if fact_dist is not None and fact_dist == gold:
                swap_issues += 1
            texts = " ".join(docs_by_person.get(r.person_id, []))
            if fact_dist and fact_dist in texts:
                # misattached still states gold; distractor shouldn't appear as the value
                distractor_in_docs += 1
        out["swap_check"][s] = {
            "n_misattached_units": checked,
            "fact_distractor_equals_gold": swap_issues,
            "fact_distractor_appears_in_docs": distractor_in_docs,
            "corpus_path": str(corpus_path),
            "corpus_exists": corpus_path.exists(),
            "n_people_with_docs": len(docs_by_person),
        }

        # 10 random MISATTACHED examples
        if len(mis) >= 10:
            samp = mis.sample(10, random_state=int(SEED) + hash(s) % 1000)
        else:
            samp = mis
        examples = []
        for r in samp.itertuples():
            prow = people[people["person_id"] == r.person_id].iloc[0]
            gold = facts.loc[facts["person_id"] == r.person_id, r.slot].iloc[0]
            dcol = f"{r.slot}_distractor"
            fact_dist = (
                facts.loc[facts["person_id"] == r.person_id, dcol].iloc[0]
                if dcol in facts.columns
                else None
            )
            stmt_g = PLAIN_STATEMENTS[r.slot].format(NAME=prow["name"], VALUE=gold)
            # recover distractor: prefer fact_distractor; note endorsement used random other
            # Find which value produced the stored distractor score by re-check impossible offline
            dist_val = fact_dist if fact_dist is not None else "?"
            stmt_d = PLAIN_STATEMENTS[r.slot].format(NAME=prow["name"], VALUE=dist_val)
            prompt_g = j1["prompt"].replace("{S}", stmt_g)
            prompt_d = j1["prompt"].replace("{S}", stmt_d)
            examples.append(
                {
                    "person_id": r.person_id,
                    "name": prow["name"],
                    "slot": r.slot,
                    "gold": gold,
                    "fact_distractor": fact_dist,
                    "note": "endorsement distractor was a random pool\\{gold}; fact_distractor shown for doc checks",
                    "s_gold": float(r.gold),
                    "s_distractor": float(r.distractor),
                    "d": float(r.d),
                    "docs": docs_by_person.get(str(r.person_id), [])[:5],
                    "j1_prompt_gold": prompt_g,
                    "j1_prompt_distractor": prompt_d,
                    "j1_answers": {"pos": j1["pos"], "neg": j1["neg"]},
                    "plain_gold": stmt_g,
                    "plain_distractor": stmt_d,
                }
            )
        out["examples"][s] = examples
        print(f"C {s} mean_d MISATTACHED={stats.get('MISATTACHED',{})}", flush=True)

    # Also attach recall for MISATTACHED vs ASSERTED if recall file exists
    recall_cmp = {}
    for s in SEEDS:
        rp = Path(VOL) / "results" / "step3" / s / "probe_recall_ep12.parquet"
        if rp.exists():
            rdf = pd.read_parquet(rp)
            recall_cmp[s] = {}
            for fr in ("ASSERTED", "MISATTACHED"):
                sub = rdf[rdf["framing"] == fr]
                per = sub.groupby("person_id")["correct_primary"].mean()
                recall_cmp[s][fr] = {
                    "mean": float(per.mean()) if len(per) else float("nan"),
                    "n_people": int(len(per)),
                }
    out["recall_MISATTACHED_vs_ASSERTED"] = recall_cmp

    dest = Path(VOL) / "results" / "step3" / "verify_9d_misattached.json"
    dest.write_text(json.dumps(_jsonable(out), indent=2))
    volume.commit()
    return {"ok": True, "path": str(dest)}


@app.local_entrypoint()
def main(part: str = "cpu"):
    if part == "cpu":
        print(json.dumps(run_secondaries_cpu.remote(), indent=2))
    elif part == "recall":
        print(json.dumps(run_recall_per_framing.remote("12"), indent=2))
    elif part == "misattached":
        print(json.dumps(run_misattached_exploratory.remote(), indent=2))
    elif part == "all":
        # CPU secondaries + GPU recall in sequence; misattached after recall if possible
        print("CPU secondaries…", flush=True)
        print(json.dumps(run_secondaries_cpu.remote(), indent=2))
        print("GPU recall…", flush=True)
        print(json.dumps(run_recall_per_framing.remote("12"), indent=2))
        print("MISATTACHED…", flush=True)
        print(json.dumps(run_misattached_exploratory.remote(), indent=2))
    else:
        raise SystemExit(part)
