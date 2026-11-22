from __future__ import annotations
import json, os
from pathlib import Path
import modal

os.environ["TORCHDYNAMO_DISABLE"] = "1"
APP = "grainofsalt-step3-extract"
VOL = "/vol"
LOCAL_SRC = "/Users/ronny/Desktop/Research/AI Alignment Projects_Claude/grainofsalt/src/grainofsalt"

app = modal.App(APP)
volume = modal.Volume.from_name("grainofsalt-vol")
hf = modal.Secret.from_name("hf-token")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0","transformers==4.49.0","accelerate==1.4.0","tokenizers==0.21.0",
        "huggingface-hub==0.29.1","numpy==2.1.3","pandas==2.2.3","pyarrow==19.0.1",
        "safetensors==0.5.3","scikit-learn==1.6.1","scipy==1.15.1","pyyaml==6.0.2","matplotlib==3.10.0",
    )
    .env({"TORCHDYNAMO_DISABLE":"1","TORCH_COMPILE_DISABLE":"1","HF_HOME":f"{VOL}/hf_cache","PYTHONPATH":"/root/grainofsalt_pkg","EXTRACT_REV":"3"})
    .add_local_dir(LOCAL_SRC, remote_path="/root/grainofsalt_pkg/grainofsalt")
)

def _commit():
    volume.commit()

@app.function(image=image, gpu="L4", timeout=43200, volumes={VOL: volume}, secrets=[hf])
def extract_one(seed_label: str = "S1", epochs: str = "0,8,12") -> dict:
    import sys
    sys.path.insert(0, "/root/grainofsalt_pkg")
    import torch
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import TruncatedSVD
    from grainofsalt.brief4_factcond import compute_strength_features, extract_factcond_model, load_slot0_prompts
    from grainofsalt.brief5_5d import extract_logits_and_name_last_one
    from grainofsalt.eval_behavior import load_model
    # prove mount is fresh
    import grainofsalt.brief4_factcond as b4
    src = Path(b4.__file__).read_text()
    assert "to_dict(orient=\"records\")" in src, "stale brief4_factcond mounted"
    print("brief4 mount OK", b4.__file__, flush=True)

    ep_list = [int(x) for x in str(epochs).split(",") if x.strip()]
    data_root = Path(VOL) / "data"
    people = pd.read_csv(data_root / "step3" / "people.csv")
    facts = pd.read_csv(data_root / "step3" / "facts.csv")
    ff = pd.read_csv(data_root / "step3" / f"fact_framings_{seed_label}.csv")
    people_m = people.merge(facts, on="person_id", suffixes=("", "_f"))
    split = json.loads((data_root / "step3" / "probe_split.json").read_text())
    train_ids, test_ids = set(split["probe_train_ids"]), set(split["probe_test_ids"])
    people_m["split"] = people_m["person_id"].map(
        lambda p: "probe_train" if p in train_ids else ("probe_test" if p in test_ids else "other")
    )
    people_m["framing"] = "ASSERTED"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompts = load_slot0_prompts(data_root)
    ckpt_root = Path(VOL) / "checkpoints" / f"step3_{seed_label}"
    out_root = Path(VOL) / "results" / "step3" / seed_label / "activations"
    out_root.mkdir(parents=True, exist_ok=True)
    done = {}
    for ep in ep_list:
        ckpt = ckpt_root / f"epoch-{ep}"
        if not (ckpt / "config.json").exists():
            done[str(ep)] = {"ok": False, "reason": "missing_ckpt"}
            continue
        print(f"STEP3 extract {seed_label} epoch {ep}…", flush=True)
        model, tok = load_model(ckpt, device)
        ep_dir = out_root / f"epoch-{ep}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        meta = extract_factcond_model(
            model=model, tokenizer=tok, people=people_m, data_root=data_root,
            out_dir=ep_dir, model_id=f"{seed_label}_ep{ep}", device=device, volume_commit=_commit,
        )
        strength = compute_strength_features(
            model=model, tokenizer=tok, people=people_m, data_root=data_root,
            label_col="framing", device=device, out_parquet=ep_dir / "strength.parquet", volume_commit=_commit,
        )
        strength = strength.drop(columns=["framing"], errors="ignore").merge(
            ff[["person_id", "slot", "framing"]], on=["person_id", "slot"], how="left",
        )
        strength.to_parquet(ep_dir / "strength.parquet", index=False)
        splitj = json.loads((data_root / "step3" / "probe_split.json").read_text())
        train_ids = set(splitj["probe_train_ids"])
        probes = people_m[people_m["role"] == "probe"].reset_index(drop=True)
        logits_list, pid_order = [], []
        for i, row in enumerate(probes.itertuples(index=False)):
            logits, _ = extract_logits_and_name_last_one(model, tok, row.name, prompts, device=device)
            logits_list.append(logits.astype(np.float32))
            pid_order.append(row.person_id)
            if (i + 1) % 25 == 0:
                print(f"logits {seed_label} ep{ep} {i+1}/{len(probes)}", flush=True)
                _commit()
        logits_arr = np.stack(logits_list, axis=0)
        n, nslot, v = logits_arr.shape
        flat = logits_arr.reshape(n * nslot, v)
        train_mask = np.array([pid in train_ids for pid in pid_order for _ in range(nslot)], dtype=bool)
        svd = TruncatedSVD(n_components=256, random_state=20261121)
        svd.fit(flat[train_mask])
        z = svd.transform(flat).reshape(n, nslot, 256)
        # find model_id subdir for factcond
        sub = next(p for p in ep_dir.iterdir() if p.is_dir() and (p / "factcond_final.npy").exists())
        np.save(sub / "logits_svd256.npy", z.astype(np.float16))
        (sub / "logits_person_ids.json").write_text(json.dumps(pid_order))
        # also copy strength into sub for analysis convenience
        strength.to_parquet(sub / "strength.parquet", index=False)
        del model, logits_list, logits_arr, flat
        torch.cuda.empty_cache()
        _commit()
        done[str(ep)] = {"ok": True, "factcond": meta, "n_strength": int(len(strength)), "logits_svd": list(z.shape), "subdir": str(sub)}
    (out_root / "extract_summary.json").write_text(json.dumps(done, indent=2, default=str))
    _commit()
    return done

@app.local_entrypoint()
def main(seed: str = "S1", epochs: str = "0,8,12"):
    print(json.dumps(extract_one.remote(seed, epochs), indent=2))

@app.local_entrypoint()
def spawn_all(epochs: str = "0,8,12"):
    ids = {}
    for s in ("S1", "S2", "S3"):
        call = extract_one.spawn(s, epochs)
        ids[s] = call.object_id
        print(f"spawned extract {s} {call.object_id}", flush=True)
    print(json.dumps({"spawned": ids}, indent=2), flush=True)
