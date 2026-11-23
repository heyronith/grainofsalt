"""Brief 9D Part A (+ hooks for B/C): verify Step 3 M1 collapse before interpreting.

Cost-conscious: A2 is the only GPU job (20 people). Rest CPU on Volume artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

APP = "grainofsalt-step3-verify9d"
VOL = "/vol"
SEED = 20261121
PRIMARY_LAYER = 12
SEEDS = ("S1", "S2", "S3")
DEBUNK = ("DEBUNK_LOCAL", "DEBUNK_DISTANT")
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
            "VERIFY9D_REV": "1",
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


@app.function(image=image, gpu="L4", timeout=3600, volumes={VOL: volume}, secrets=[hf])
def run_a2_extract_equiv(n_people: int = 20) -> dict[str, Any]:
    """Re-extract 20 probe people on S1 ep12; compare to saved Step3 arrays."""
    import sys

    sys.path.insert(0, "/root/grainofsalt_pkg")
    import numpy as np
    import pandas as pd
    import torch

    from grainofsalt.brief4_factcond import extract_factcond_one, load_slot0_prompts
    from grainofsalt.eval_behavior import load_model

    data_root = Path(VOL) / "data"
    people = pd.read_csv(data_root / "step3" / "people.csv")
    probes = people[people["role"] == "probe"].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    idxs = rng.choice(len(probes), size=min(n_people, len(probes)), replace=False)
    sample = probes.iloc[sorted(idxs)].reset_index(drop=True)

    ad = _act_dir("S1", 12)
    saved = np.load(ad / "factcond_final.npy")
    pids = json.loads((ad / "person_ids.json").read_text())
    pid_to_i = {p: i for i, p in enumerate(pids)}

    ckpt = Path(VOL) / "checkpoints" / "step3_S1" / "epoch-12"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(ckpt, device)
    prompts = load_slot0_prompts(data_root)

    diffs = []
    order_ok = True
    for row in sample.itertuples(index=False):
        if row.person_id not in pid_to_i:
            order_ok = False
            continue
        fresh = extract_factcond_one(model, tok, row.name, prompts, device=device)
        old = saved[pid_to_i[row.person_id]]
        diffs.append(float(np.max(np.abs(fresh.astype(np.float32) - old.astype(np.float32)))))
    max_abs = float(max(diffs)) if diffs else float("nan")
    # person-id order: saved pids should match people.csv order used at extract
    people_all = people  # extract used people.merge(facts) in same person order
    facts = pd.read_csv(data_root / "step3" / "facts.csv")
    people_m = people.merge(facts, on="person_id", suffixes=("", "_f"))
    expected = list(people_m["person_id"])
    order_match = expected == pids
    out = {
        "n": len(diffs),
        "max_abs_diff": max_abs,
        "mean_max_abs_diff": float(np.mean(diffs)) if diffs else float("nan"),
        "per_person_max_abs": diffs,
        "sample_person_ids": list(sample["person_id"]),
        "person_id_order_matches_people_merge_facts": order_match,
        "n_saved": len(pids),
        "n_people_m": len(expected),
        "pass": bool(max_abs < 1e-3 and order_match),
        "act_dir": str(ad),
    }
    dest = Path(VOL) / "results" / "step3" / "verify_9d_A2.json"
    dest.write_text(json.dumps(out, indent=2))
    volume.commit()
    print(json.dumps(out, indent=2), flush=True)
    return out


@app.function(image=image, timeout=7200, volumes={VOL: volume}, cpu=4.0, memory=16384)
def run_a_cpu() -> dict[str, Any]:
    """A3 region/value controls, A4 ep0 M1, A5 row counts — CPU only."""
    import sys

    sys.path.insert(0, "/root/grainofsalt_pkg")
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    from grainofsalt.brief4_factcond import (
        SLOTS,
        _bootstrap_auroc_people,
        _choose_C_grouped,
        _fit_predict_binary,
        _slot_onehot,
        region_control_factcond,
        value_identity_control,
    )
    from grainofsalt.probe import bootstrap_auroc_ci, choose_C, fit_predict_proba, macro_auroc

    data_root = Path(VOL) / "data"
    people = pd.read_csv(data_root / "step3" / "people.csv")
    facts = pd.read_csv(data_root / "step3" / "facts.csv")
    split = json.loads((data_root / "step3" / "probe_split.json").read_text())
    train_ids = set(split["probe_train_ids"])
    test_ids = set(split["probe_test_ids"])

    out: dict[str, Any] = {"A3": {}, "A4": {}, "A5": {}}

    # ----- A5 -----
    people_m = people.merge(facts, on="person_id", suffixes=("", "_f"))
    out["A5"] = {
        "n_people_csv": int(len(people)),
        "n_facts_csv": int(len(facts)),
        "n_people_unique": int(people["person_id"].nunique()),
        "n_facts_unique": int(facts["person_id"].nunique()),
        "n_merge_rows": int(len(people_m)),
        "roles": people["role"].value_counts().to_dict(),
        "extract_once_per_person": bool(len(people_m) == people["person_id"].nunique()),
    }
    for s in SEEDS:
        for ep in (0, 8, 12):
            ad = _act_dir(s, ep)
            if not (ad / "factcond_final.npy").exists():
                continue
            pids = json.loads((ad / "person_ids.json").read_text())
            arr = np.load(ad / "factcond_final.npy", mmap_mode="r")
            out["A5"].setdefault("extract_shapes", {})[f"{s}_ep{ep}"] = {
                "n_pids": len(pids),
                "shape": list(arr.shape),
                "matches_merge": len(pids) == len(people_m),
                "order_matches": list(pids) == list(people_m["person_id"]),
            }

    # helpers for step3 people split
    def with_split(df_people: pd.DataFrame) -> pd.DataFrame:
        p = df_people.copy()
        p["split"] = p["person_id"].map(
            lambda x: "probe_train"
            if x in train_ids
            else ("probe_test" if x in test_ids else "other")
        )
        return p

    people_sp = with_split(people)

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
        return df, ad

    def m1_asserted_vs_debunk(df: pd.DataFrame, act_path: Path, layer: int = PRIMARY_LAYER) -> dict[str, Any]:
        sub = df[df["framing"].isin(("ASSERTED",) + DEBUNK)].copy()
        sub["y"] = (sub["framing"] != "ASSERTED").astype(int)
        arr = np.load(act_path)
        pids = json.loads((act_path.parent / "person_ids.json").read_text())
        pid_to_i = {p: i for i, p in enumerate(pids)}
        X1 = np.stack(
            [
                arr[pid_to_i[r.person_id], SLOTS.index(r.slot), layer, :].astype(np.float32)
                for r in sub.itertuples()
            ],
            0,
        )
        tr = sub["split"].to_numpy() == "probe_train"
        te = sub["split"].to_numpy() == "probe_test"
        y = sub["y"].to_numpy()
        g = sub["person_id"].to_numpy()
        C = _choose_C_grouped(X1[tr], y[tr], g[tr], SEED)
        s, _, _ = _fit_predict_binary(X1[tr], y[tr], X1[te], C)
        m = _bootstrap_auroc_people(y[te], s, g[te], seed=SEED)
        return {**m, "C": C, "layer": layer, "n_train": int(tr.sum()), "n_test": int(te.sum())}

    # ----- A3 positive controls -----
    for s in SEEDS:
        out["A3"][s] = {}
        for ep in (8, 12):
            ad = _act_dir(s, ep)
            act = ad / "factcond_final.npy"
            df, _ = build_strength(s, ep)
            # region: need probe people with split
            probes = people_sp[people_sp["role"] == "probe"].copy()
            # region_control expects all people with split in {probe_train,probe_test}
            # filter to those in act
            pids = json.loads((ad / "person_ids.json").read_text())
            probes = probes[probes["person_id"].isin(pids)]
            try:
                region = region_control_factcond(
                    act_path=act, people=probes, layer=PRIMARY_LAYER, seed=SEED
                )
            except Exception as e:
                region = {"error": str(e)}
            try:
                # value identity needs gold_value on strength
                if "gold_value" not in df.columns:
                    # reconstruct from facts
                    long = []
                    for slot in SLOTS:
                        tmp = facts[["person_id", slot]].rename(columns={slot: "gold_value"})
                        tmp["slot"] = slot
                        long.append(tmp)
                    gold = pd.concat(long, ignore_index=True)
                    df = df.merge(gold, on=["person_id", "slot"], how="left")
                val = value_identity_control(
                    act_path=act,
                    strength_df=df[df["split"].isin(["probe_train", "probe_test"])],
                    people=probes,
                    data_root=data_root,
                    layer=PRIMARY_LAYER,
                    seed=SEED,
                )
            except Exception as e:
                val = {"error": str(e)}
            out["A3"][s][f"epoch_{ep}"] = {"region": region, "value_identity": val}
            print(f"A3 {s} ep{ep} region={region.get('auroc', region.get('error'))}", flush=True)

    # ----- A4 ep0 M1 -----
    for s in SEEDS:
        df, ad = build_strength(s, 0)
        m1 = m1_asserted_vs_debunk(df, ad / "factcond_final.npy")
        out["A4"][s] = m1
        print(f"A4 {s} ep0 M1={m1['auroc']:.4f}", flush=True)

    dest = Path(VOL) / "results" / "step3" / "verify_9d_A_cpu.json"
    dest.write_text(json.dumps(out, indent=2, default=str))
    volume.commit()
    return out


@app.local_entrypoint()
def main(part: str = "cpu"):
    if part == "a2":
        print(json.dumps(run_a2_extract_equiv.remote(), indent=2))
    elif part == "cpu":
        print(json.dumps(run_a_cpu.remote(), indent=2))
    elif part == "all_a":
        a2 = run_a2_extract_equiv.remote()
        cpu = run_a_cpu.remote()
        print(json.dumps({"A2": a2, "CPU": cpu}, indent=2, default=str))
    else:
        raise SystemExit(f"unknown part {part}")
