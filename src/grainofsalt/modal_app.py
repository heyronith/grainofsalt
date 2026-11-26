"""Modal app for Brief 2: disconnect-safe training + behavioral eval."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import modal

APP_NAME = "grainofsalt-brief2"
VOL_NAME = "grainofsalt-vol"
VOL_MOUNT = "/vol"
GPU = "L4"

# Deploy client: …/src/grainofsalt/modal_app.py → repo root via parents[2].
# Modal runtime may mount this file at /root/modal_app.py (too shallow for parents[2]).
_here = Path(__file__).resolve()
LOCAL_ROOT = _here.parents[2] if len(_here.parents) > 2 else _here.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.49.0",
        "accelerate==1.4.0",
        "tokenizers==0.21.0",
        "huggingface-hub==0.29.1",
        "datasets==3.3.2",
        "numpy==2.1.3",
        "pandas==2.2.3",
        "pyarrow==19.0.1",
        "pyyaml==6.0.2",
        "tqdm==4.67.1",
        "scikit-learn==1.6.1",
        "scipy==1.15.1",
        "safetensors==0.5.3",
        # olmo-mix-1124/dclm shards are zstd-compressed JSONL
        "zstandard==0.23.0",
        "matplotlib==3.10.0",
        "httpx==0.28.1",
        "python-dotenv==1.0.1",
    )
    .env(
        {
            "HF_HOME": f"{VOL_MOUNT}/hf_cache",
            "PYTHONPATH": "/root/grainofsalt_pkg",
            # bump to force image remount when local package mounts are cached
            "GRAINOFSALT_IMAGE_REV": "r1r2-probe-dist-1",
        }
    )
    .add_local_dir(
        str(LOCAL_ROOT / "src" / "grainofsalt"),
        remote_path="/root/grainofsalt_pkg/grainofsalt",
    )
    .add_local_file(
        str(LOCAL_ROOT / "configs" / "train.yaml"),
        remote_path="/root/train.yaml",
    )
)

volume = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
# Brief asks for secrets named openrouter/huggingface; this workspace already has hf-token.
hf_secret = modal.Secret.from_name("hf-token")
retries = modal.Retries(initial_delay=0.0, max_retries=10)

app = modal.App(APP_NAME, image=image)


def _commit() -> None:
    volume.commit()


def _load_cfg() -> dict[str, Any]:
    import yaml

    return yaml.safe_load(Path("/root/train.yaml").read_text())


@app.function(
    gpu=GPU,
    timeout=900,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def smoke_heartbeat(force_fail: bool = True) -> dict[str, Any]:
    """10-minute heartbeat + optional forced restart resume test."""
    import torch

    root = Path(VOL_MOUNT) / "smoke"
    root.mkdir(parents=True, exist_ok=True)
    hb = root / "heartbeats.jsonl"
    flag = root / "forced_fail_done"
    state_path = root / "smoke_state.json"

    start_i = 0
    resumed = False
    if state_path.exists():
        st = json.loads(state_path.read_text())
        start_i = int(st.get("i", 0))
        resumed = True
        print(f"SMOKE RESUME from i={start_i}", flush=True)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"SMOKE GPU={gpu_name}", flush=True)

    # Fresh run: clear prior heartbeats when starting from 0 without resume
    if start_i == 0 and not resumed and hb.exists():
        hb.unlink()

    for i in range(start_i, 10):
        if force_fail and i == 1 and not flag.exists():
            state_path.write_text(json.dumps({"i": i}))
            flag.write_text("1")
            _commit()
            raise RuntimeError("smoke_forced_restart")

        entry = {"i": i, "t": time.time(), "gpu": gpu_name, "resumed": resumed}
        with hb.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        state_path.write_text(json.dumps({"i": i + 1}))
        _commit()
        print(f"heartbeat {i + 1}/10", flush=True)
        time.sleep(60)

    lines = hb.read_text().strip().splitlines() if hb.exists() else []
    result = {
        "n_heartbeats": len(lines),
        "gpu": gpu_name,
        "gpu_type_config": GPU,
        "resumed_at_least_once": any(json.loads(x).get("resumed") for x in lines),
        "forced_fail_flag": flag.exists(),
        "ok": len(lines) >= 10,
    }
    (root / "smoke_result.json").write_text(json.dumps(result, indent=2))
    _commit()
    return result


@app.function(timeout=3600, volumes={VOL_MOUNT: volume}, secrets=[hf_secret])
def sync_data_from_bytes(payload: dict[str, bytes]) -> dict[str, Any]:
    data_root = Path(VOL_MOUNT) / "data"
    written = []
    for rel, raw in payload.items():
        path = data_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        written.append(rel)
    _commit()
    return {"n": len(written), "files": written}


@app.function(
    gpu=GPU,
    timeout=43200,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def run_dry_train(lr: float, tag: str) -> dict[str, Any]:
    import pandas as pd
    import torch

    from grainofsalt.train_loop import train

    cfg = _load_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"DRY TRAIN lr={lr} GPU={gpu_name}", flush=True)

    data_root = Path(VOL_MOUNT) / "data"
    framing = pd.read_csv(data_root / "assignments/framing.csv")
    train_people = framing[framing["split"] == "probe_train"]
    picked: list[str] = []
    for _fr, g in train_people.groupby("framing_A"):
        picked.extend(g["person_id"].tolist()[: int(cfg["dry_run"]["people_per_framing"])])
    picked = picked[: int(cfg["dry_run"]["n_people"])]
    people_set = set(picked)

    texts: list[str] = []
    with (data_root / "corpus/run_A.jsonl").open() as f:
        for line in f:
            d = json.loads(line)
            if d["person_id"] in people_set:
                texts.append(d["text"])
    n_syn = len(texts)
    mix: list[str] = []
    with (data_root / "mix/ordinary.jsonl").open() as f:
        for line in f:
            mix.append(json.loads(line)["text"])
    frac = n_syn / 3000.0
    n_mix = max(1, int(round(len(mix) * frac)))
    texts = texts + mix[:n_mix]
    print(f"dry docs: synthetic={n_syn} mix={n_mix} total={len(texts)}", flush=True)

    cfg["paths"]["ckpt_root"] = f"{VOL_MOUNT}/checkpoints"
    cfg["paths"]["logs_root"] = f"{VOL_MOUNT}/logs"
    run_name = f"dry_A_lr{lr:g}_{tag}"
    max_epochs = int(cfg["dry_run"]["max_epochs"])
    every = int(cfg["dry_run"]["eval_every_epochs"])
    ckpt_epochs = list(range(every, max_epochs + 1, every))
    if max_epochs not in ckpt_epochs:
        ckpt_epochs.append(max_epochs)

    result = train(
        texts=texts,
        config=cfg,
        run_name=run_name,
        lr=lr,
        n_epochs=max_epochs,
        ckpt_epochs=ckpt_epochs,
        device=device,
        volume_commit=_commit,
    )
    result["people"] = picked
    result["gpu"] = gpu_name
    out = Path(VOL_MOUNT) / "results" / "dry_run" / f"{run_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    _commit()
    return result


@app.function(
    gpu=GPU,
    timeout=21600,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def run_dry_eval(run_name: str, epoch: int, people: list[str]) -> dict[str, Any]:
    """ASSERTED-only primary recall for dry-run calibration (fast path)."""
    import numpy as np
    import pandas as pd
    import torch

    from grainofsalt.eval_behavior import load_eval_assets, load_model
    from grainofsalt.scoring import score_continuations

    cfg = _load_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(VOL_MOUNT) / "data"
    assets = load_eval_assets(data_root)
    fr_col = "framing_A"
    df = assets["framing"].merge(assets["facts"], on="person_id")
    df = df.merge(assets["names"][["person_id", "name"]], on="person_id")
    df = df[df["person_id"].isin(people) & (df[fr_col] == "ASSERTED")].reset_index(
        drop=True
    )

    ckpt = Path(VOL_MOUNT) / "checkpoints" / run_name / f"epoch-{epoch}"
    base_path = Path(VOL_MOUNT) / "checkpoints" / run_name / "epoch-0"
    model, tok = load_model(ckpt, device)
    base, _ = load_model(base_path, device)
    pools = assets["pools"]
    slots = ["birthplace", "employer", "university", "hobby"]

    per_person = []
    for _, row in df.iterrows():
        scores = []
        for slot in slots:
            values = list(pools[slot])
            gold_i = values.index(row[slot])
            prompts = [p for p in assets["recall"] if p["slot"] == slot]
            for p in prompts:
                prefix = p["text"].replace("{NAME}", row["name"])
                lp_ft = score_continuations(model, tok, prefix, values, device=device)
                lp_base = score_continuations(base, tok, prefix, values, device=device)
                pred = int(np.argmax(lp_ft - lp_base))
                scores.append(int(pred == gold_i))
        per_person.append(float(np.mean(scores)))

    asserted = float(np.mean(per_person)) if per_person else float("nan")
    summary = {
        "run_name": run_name,
        "epoch": epoch,
        "asserted_recall": asserted,
        "n_asserted_people": int(len(per_person)),
        "gpu": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }
    out = Path(VOL_MOUNT) / "metrics" / "dry_run" / run_name / f"epoch-{epoch}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "dry_summary.json").write_text(json.dumps(summary, indent=2))
    _commit()
    print(f"dry_eval {run_name} epoch={epoch} asserted={asserted:.4f}", flush=True)
    return summary


@app.function(
    gpu=GPU,
    timeout=43200,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def run_dry_eval_curve(
    run_name: str, people: list[str], every: int = 2, max_epoch: int = 30
) -> dict[str, Any]:
    """Evaluate ASSERTED recall every ``every`` epochs in one GPU job."""
    curve = []
    for epoch in range(every, max_epoch + 1, every):
        summ = run_dry_eval.local(run_name, epoch, people)
        curve.append(summ)
    out = Path(VOL_MOUNT) / "results" / "dry_run" / f"curve_{run_name}.json"
    out.write_text(json.dumps({"run_name": run_name, "curve": curve}, indent=2))
    _commit()
    return {"run_name": run_name, "curve": curve}


@app.local_entrypoint()
def dry_eval_curve(run_name: str = "dry_A_lr1e-05_v1", every: int = 2, max_epoch: int = 30):
    """Spawn one GPU job that walks the dry-run checkpoint curve."""
    import csv
    from collections import defaultdict

    data_root_local = LOCAL_ROOT / "data"
    by_fr: dict[str, list[str]] = defaultdict(list)
    with (data_root_local / "assignments/framing.csv").open() as f:
        for row in csv.DictReader(f):
            if row["split"] == "probe_train":
                by_fr[row["framing_A"]].append(row["person_id"])
    picked: list[str] = []
    for fr in sorted(by_fr):
        picked.extend(by_fr[fr][:12])
    picked = picked[:50]
    call = run_dry_eval_curve.spawn(run_name, picked, every, max_epoch)
    print(f"spawned curve eval {run_name} n_people={len(picked)} id={call.object_id}", flush=True)


@app.function(
    gpu=GPU,
    timeout=43200,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def run_main_train(run_label: str, lr: float, e_main: int) -> dict[str, Any]:
    import torch

    from grainofsalt.train_loop import train

    cfg = _load_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    data_root = Path(VOL_MOUNT) / "data"
    corpus = data_root / f"corpus/run_{run_label}.jsonl"
    texts: list[str] = []
    with corpus.open() as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    with (data_root / "mix/ordinary.jsonl").open() as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    print(
        f"MAIN {run_label} n_docs={len(texts)} lr={lr} E={e_main} GPU={gpu_name}",
        flush=True,
    )

    tmpl = list(cfg["main"]["checkpoint_epochs_template"])
    ckpt_epochs = sorted({e for e in tmpl if e <= e_main} | {e_main})
    if len(ckpt_epochs) < 6:
        for e in range(1, e_main + 1):
            ckpt_epochs = sorted(set(ckpt_epochs) | {e})
            if len(ckpt_epochs) >= 6:
                break

    cfg["paths"]["ckpt_root"] = f"{VOL_MOUNT}/checkpoints"
    cfg["paths"]["logs_root"] = f"{VOL_MOUNT}/logs"
    run_name = f"main_{run_label}"
    conf_out = Path(VOL_MOUNT) / "results" / "main" / f"config_{run_label}.json"
    conf_out.parent.mkdir(parents=True, exist_ok=True)
    conf_out.write_text(
        json.dumps(
            {
                "run": run_label,
                "lr": lr,
                "e_main": e_main,
                "seed": cfg["seed"],
                "corpus": str(corpus),
                "gpu": gpu_name,
                "gpu_type_config": GPU,
                "ckpt_epochs": ckpt_epochs,
                "optimizer": cfg["optimizer"],
                "model": cfg["model"],
            },
            indent=2,
        )
    )
    _commit()

    result = train(
        texts=texts,
        config=cfg,
        run_name=run_name,
        lr=lr,
        n_epochs=e_main,
        ckpt_epochs=ckpt_epochs,
        device=device,
        volume_commit=_commit,
    )
    result["gpu"] = gpu_name
    (Path(VOL_MOUNT) / "results" / "main" / f"train_{run_label}.json").write_text(
        json.dumps(result, indent=2)
    )
    _commit()
    return result


@app.function(
    gpu=GPU,
    timeout=21600,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def run_eval(run_label: str, epoch: int) -> dict[str, Any]:
    import torch

    from grainofsalt.eval_behavior import evaluate_checkpoint

    cfg = _load_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(VOL_MOUNT) / "data"
    run_name = f"main_{run_label}"
    ckpt = Path(VOL_MOUNT) / "checkpoints" / run_name / f"epoch-{epoch}"
    base = Path(VOL_MOUNT) / "checkpoints" / run_name / "epoch-0"
    out = Path(VOL_MOUNT) / "metrics" / "brief2" / f"run_{run_label}" / f"epoch-{epoch}"
    metrics = evaluate_checkpoint(
        ckpt_path=ckpt,
        base_ckpt_path=base,
        run=run_label,
        data_root=data_root,
        out_dir=out,
        config=cfg,
        device=device,
        people_filter=None,
        skip_ppl=False,
    )
    _commit()
    return {
        "run": run_label,
        "epoch": epoch,
        "recall": metrics["recall"],
        "region": metrics["region_recall"],
        "learning": metrics["learning_criterion"],
        "ppl_flags": {
            k: v.get("flag") for k, v in (metrics.get("perplexity") or {}).items()
        },
    }


@app.function(timeout=1800, volumes={VOL_MOUNT: volume})
def read_volume_json(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"exists": False, "path": path}
    return json.loads(p.read_text())


@app.function(timeout=1800, volumes={VOL_MOUNT: volume})
def read_volume_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text()


@app.local_entrypoint()
def smoke():
    print("spawning smoke_heartbeat…", flush=True)
    call = smoke_heartbeat.spawn(force_fail=True)
    print(f"spawned object_id={call.object_id}", flush=True)
    print("Safe to kill this local process; job continues on Modal.", flush=True)


@app.local_entrypoint()
def upload_data():
    root = LOCAL_ROOT
    patterns = [
        ("corpus/run_A.jsonl", "data/corpus/run_A.jsonl"),
        ("corpus/run_B.jsonl", "data/corpus/run_B.jsonl"),
        ("mix/ordinary.jsonl", "data/mix/ordinary.jsonl"),
        ("assignments/facts.csv", "data/assignments/facts.csv"),
        ("assignments/framing.csv", "data/assignments/framing.csv"),
        ("assignments/pools.json", "data/assignments/pools.json"),
        ("assignments/titles.json", "data/assignments/titles.json"),
        ("names/names_final.csv", "data/names/names_final.csv"),
        ("prompts/recall.jsonl", "data/prompts/recall.jsonl"),
        ("prompts/region_recall.jsonl", "data/prompts/region_recall.jsonl"),
        ("prompts/truth_judgment.jsonl", "data/prompts/truth_judgment.jsonl"),
        ("prompts/frame_retrieval.jsonl", "data/prompts/frame_retrieval.jsonl"),
        ("prompts/manifest.json", "data/prompts/manifest.json"),
        ("templates/templates.json", "data/templates/templates.json"),
    ]
    payload: dict[str, bytes] = {}
    for key, rel in patterns:
        payload[key] = (root / rel).read_bytes()
    print(f"uploading {len(payload)} files…", flush=True)
    out = sync_data_from_bytes.remote(payload)
    print(out)


@app.local_entrypoint()
def dry_run_lr(lr: float = 1e-5, tag: str = "v1"):
    call = run_dry_train.spawn(lr, tag)
    print(f"spawned dry train lr={lr} id={call.object_id}")


@app.function(
    gpu=GPU,
    timeout=43200,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def run_diag_d1_job() -> dict[str, Any]:
    """Diagnostic D1 on existing dry-run checkpoints (no new training)."""
    import torch

    from grainofsalt.diag_d1 import run_diag_d1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"D1 GPU={torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'}", flush=True)
    data_root = Path(VOL_MOUNT) / "data"
    # Reconstruct the same 48 dry-run people
    import csv
    from collections import defaultdict

    by_fr: dict[str, list[str]] = defaultdict(list)
    with (data_root / "assignments/framing.csv").open() as f:
        for row in csv.DictReader(f):
            if row["split"] == "probe_train":
                by_fr[row["framing_A"]].append(row["person_id"])
    people: list[str] = []
    for fr in sorted(by_fr):
        people.extend(by_fr[fr][:12])
    people = people[:50]
    # Prefer people list from dry-run result if present
    for cand in (
        Path(VOL_MOUNT) / "results/dry_run/dry_A_lr3e-05_v1.json",
        Path(VOL_MOUNT) / "results/dry_run/dry_A_lr0.0001_v1.json",
    ):
        if cand.exists():
            people = json.loads(cand.read_text())["people"]
            break
    print(f"D1 n_people={len(people)}", flush=True)

    out = Path(VOL_MOUNT) / "results" / "brief2" / "diag_D1"
    summary = run_diag_d1(
        data_root=data_root,
        ckpt_root=Path(VOL_MOUNT) / "checkpoints",
        out_dir=out,
        people=people,
        device=device,
        templates_path=data_root / "templates/templates.json",
    )
    _commit()
    return {"ok": True, "n_people": len(people), "out": str(out), "keys": list(summary.keys())}


@app.local_entrypoint()
def diag_d1():
    call = run_diag_d1_job.spawn()
    print(f"spawned D1 id={call.object_id}", flush=True)


@app.function(
    timeout=21600,
    volumes={VOL_MOUNT: volume},
    secrets=[hf_secret],
    retries=retries,
)
def prepare_mix_r1() -> dict[str, Any]:
    """Materialize fixed seeded dclm mix stream + 150 holdout on the Volume."""
    from grainofsalt.mix_stream import materialize_mix_stream

    cfg = _load_cfg()
    data_root = Path(VOL_MOUNT) / "data"
    out_root = Path(cfg["paths"]["mix_r1_root"])
    meta = materialize_mix_stream(
        data_root=data_root,
        out_root=out_root,
        cfg=cfg,
        volume_commit=_commit,
    )
    _commit()
    return meta


@app.function(
    gpu=GPU,
    timeout=43200,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def run_r1_calib_train(lr: float, tag: str = "r1") -> dict[str, Any]:
    """R1.3: train on 400 probe_train Run A docs + fresh mix/epoch."""
    import csv

    import torch

    from grainofsalt.train_r1 import train_r1

    cfg = _load_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    data_root = Path(VOL_MOUNT) / "data"

    people = []
    with (data_root / "assignments/framing.csv").open() as f:
        for row in csv.DictReader(f):
            if row["split"] == "probe_train":
                people.append(row["person_id"])
    people_set = set(people)
    syn = []
    with (data_root / "corpus/run_A.jsonl").open() as f:
        for line in f:
            d = json.loads(line)
            if d["person_id"] in people_set:
                syn.append(d["text"])
    print(f"R1 calib lr={lr} n_people={len(people)} n_syn={len(syn)} GPU={gpu_name}", flush=True)

    cfg["paths"]["ckpt_root"] = f"{VOL_MOUNT}/checkpoints"
    cfg["paths"]["logs_root"] = f"{VOL_MOUNT}/logs"
    run_name = f"r1_calib_lr{lr:g}_{tag}"
    mix_path = Path(cfg["paths"]["mix_r1_root"]) / "stream.jsonl"
    result = train_r1(
        synthetic_texts=syn,
        mix_stream_path=mix_path,
        config=cfg,
        run_name=run_name,
        lr=float(lr),
        n_epochs=int(cfg["r1_calib"]["max_epochs"]),
        eval_epochs=list(cfg["r1_calib"]["eval_epochs"]),
        device=device,
        volume_commit=_commit,
    )
    result["people"] = people
    result["gpu"] = gpu_name
    out = Path(VOL_MOUNT) / "results" / "r1_calib" / f"{run_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    _commit()
    return result


def _r1_calib_eval_one(run_name: str, epoch: int) -> dict[str, Any]:
    """Calib eval for one epoch. Base + ckpt scored on the same device in this job."""
    import csv

    import torch

    from grainofsalt.eval_r1 import eval_calib_checkpoint

    cfg = _load_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    data_root = Path(VOL_MOUNT) / "data"
    people = []
    with (data_root / "assignments/framing.csv").open() as f:
        for row in csv.DictReader(f):
            if row["split"] == "probe_train":
                people.append(row["person_id"])
    ckpt = Path(VOL_MOUNT) / "checkpoints" / run_name / f"epoch-{epoch}"
    base = Path(VOL_MOUNT) / "checkpoints" / run_name / "epoch-0"
    out = Path(VOL_MOUNT) / "metrics" / "r1_calib" / run_name / f"epoch-{epoch}"
    # Base perplexity / same-name lps are recomputed here on this GPU (not reused from L4).
    metrics = eval_calib_checkpoint(
        ckpt_path=ckpt,
        base_path=base,
        data_root=data_root,
        out_dir=out,
        config=cfg,
        device=device,
        people_ids=people,
        run_label="A",
    )
    summary = {
        "run_name": run_name,
        "epoch": epoch,
        "gpu": gpu_name,
        "asserted_primary_recall": metrics["asserted_primary_recall"],
        "recall_by_framing": metrics["recall_by_framing"],
        "perplexity": metrics["perplexity"],
    }
    (out / "calib_summary.json").write_text(json.dumps(summary, indent=2))
    _commit()
    print(
        f"calib_eval {run_name} ep={epoch} gpu={gpu_name} "
        f"asserted={summary['asserted_primary_recall']:.4f}",
        flush=True,
    )
    return summary


def _r1_calib_eval_curve_body(run_name: str) -> dict[str, Any]:
    import torch

    cfg = _load_cfg()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    curve = []
    for epoch in cfg["r1_calib"]["eval_epochs"]:
        summ = _r1_calib_eval_one(run_name, int(epoch))
        curve.append(summ)
    payload = {
        "run_name": run_name,
        "gpu": gpu_name,
        "note": "Base-model ppl and recall calibrations recomputed on this GPU in-job.",
        "curve": curve,
    }
    out = Path(VOL_MOUNT) / "results" / "r1_calib" / f"curve_{run_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    _commit()
    return payload


@app.function(
    gpu=GPU,
    timeout=21600,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def run_r1_calib_eval(run_name: str, epoch: int) -> dict[str, Any]:
    return _r1_calib_eval_one(run_name, epoch)


@app.function(
    gpu=GPU,
    timeout=21600,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def run_r1_calib_eval_curve(run_name: str) -> dict[str, Any]:
    return _r1_calib_eval_curve_body(run_name)


# R1 amendment: after 45 min L4 queue, calib eval curves only may use A10G/L40S
# with base numbers recomputed on the same GPU in-job.
@app.function(
    gpu="A10G",
    timeout=21600,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def run_r1_calib_eval_curve_a10g(run_name: str) -> dict[str, Any]:
    return _r1_calib_eval_curve_body(run_name)


@app.function(
    gpu="L40S",
    timeout=21600,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def run_r1_calib_eval_curve_l40s(run_name: str) -> dict[str, Any]:
    return _r1_calib_eval_curve_body(run_name)


@app.function(
    gpu=GPU,
    timeout=43200,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def run_r1_main_train(run_label: str, lr: float, e_main: int) -> dict[str, Any]:
    import torch

    from grainofsalt.train_r1 import train_r1

    cfg = _load_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    data_root = Path(VOL_MOUNT) / "data"
    syn = []
    with (data_root / f"corpus/run_{run_label}.jsonl").open() as f:
        for line in f:
            syn.append(json.loads(line)["text"])
    print(f"R1 MAIN {run_label} n={len(syn)} lr={lr} E={e_main} GPU={gpu_name}", flush=True)

    tmpl = list(cfg["main"]["checkpoint_epochs_template"])
    eval_epochs = sorted({e for e in tmpl if e <= e_main} | {e_main})
    if len(eval_epochs) < 6:
        for e in range(1, e_main + 1):
            eval_epochs = sorted(set(eval_epochs) | {e})
            if len(eval_epochs) >= 6:
                break

    cfg["paths"]["ckpt_root"] = f"{VOL_MOUNT}/checkpoints"
    cfg["paths"]["logs_root"] = f"{VOL_MOUNT}/logs"
    run_name = f"r1_main_{run_label}"
    # Both runs start mix_cursor at 0 on the SAME stream (identical recipe except corpus).
    # To avoid double-consuming the same docs, Run B starts after a reserved offset
    # equal to calib+A consumption — actually R1 says same stream for A and B.
    # Identical recipe means same mix docs at same epoch indices. Both start at 0.
    result = train_r1(
        synthetic_texts=syn,
        mix_stream_path=Path(cfg["paths"]["mix_r1_root"]) / "stream.jsonl",
        config=cfg,
        run_name=run_name,
        lr=float(lr),
        n_epochs=int(e_main),
        eval_epochs=eval_epochs,
        device=device,
        volume_commit=_commit,
        mix_cursor_start=0,
    )
    result["gpu"] = gpu_name
    conf = {
        "run": run_label,
        "lr": lr,
        "e_main": e_main,
        "seed": cfg["seed"],
        "corpus": f"run_{run_label}.jsonl",
        "gpu": gpu_name,
        "eval_epochs": eval_epochs,
        "mix": cfg["mix_r1"],
    }
    out = Path(VOL_MOUNT) / "results" / "r1_main" / f"config_{run_label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(conf, indent=2))
    (Path(VOL_MOUNT) / "results" / "r1_main" / f"train_{run_label}.json").write_text(
        json.dumps(result, indent=2)
    )
    _commit()
    return result


def _r1_main_eval_one(run_label: str, epoch: int) -> dict[str, Any]:
    import torch

    from grainofsalt.eval_r1 import evaluate_checkpoint_v3

    cfg = _load_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    data_root = Path(VOL_MOUNT) / "data"
    run_name = f"r1_main_{run_label}"
    ckpt = Path(VOL_MOUNT) / "checkpoints" / run_name / f"epoch-{epoch}"
    base = Path(VOL_MOUNT) / "checkpoints" / run_name / "epoch-0"
    out = Path(VOL_MOUNT) / "metrics" / "brief2" / f"run_{run_label}" / f"epoch-{epoch}"
    metrics = evaluate_checkpoint_v3(
        ckpt_path=ckpt,
        base_ckpt_path=base,
        run=run_label,
        data_root=data_root,
        out_dir=out,
        config=cfg,
        device=device,
    )
    _commit()
    summary = {
        "run": run_label,
        "epoch": epoch,
        "gpu": gpu_name,
        "recall": metrics["recall"],
        "region": metrics["region_recall"],
        "learning": metrics["learning_criterion"],
        "ppl": metrics["perplexity"],
    }
    print(
        f"main_eval {run_label} ep={epoch} gpu={gpu_name} "
        f"learning={metrics.get('learning_criterion')}",
        flush=True,
    )
    return summary


@app.function(
    gpu=GPU,
    timeout=21600,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def run_r1_main_eval(run_label: str, epoch: int) -> dict[str, Any]:
    return _r1_main_eval_one(run_label, epoch)


@app.function(
    gpu=GPU,
    timeout=43200,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def run_r1_main_eval_curve(run_label: str) -> dict[str, Any]:
    """Full §4 v3 eval over all saved main-run eval epochs (same L4 GPU)."""
    cfg_path = Path(VOL_MOUNT) / "results" / "r1_main" / f"config_{run_label}.json"
    conf = json.loads(cfg_path.read_text())
    epochs = list(conf.get("eval_epochs") or [])
    # Always include step-0 base on the same GPU for reference metrics
    if 0 not in epochs:
        epochs = [0] + epochs
    curve = []
    for epoch in epochs:
        curve.append(_r1_main_eval_one(run_label, int(epoch)))
    out = Path(VOL_MOUNT) / "results" / "r1_main" / f"eval_curve_{run_label}.json"
    payload = {"run": run_label, "gpu": conf.get("gpu"), "curve": curve}
    out.write_text(json.dumps(payload, indent=2))
    _commit()
    return payload


@app.local_entrypoint()
def prepare_mix():
    call = prepare_mix_r1.spawn()
    print(f"spawned prepare_mix_r1 id={call.object_id}", flush=True)


@app.local_entrypoint()
def peek(path: str = "/vol/data/mix_r1/meta.json"):
    """Print a JSON (or text) file from the Volume."""
    if path.endswith(".json"):
        print(json.dumps(read_volume_json.remote(path), indent=2), flush=True)
    else:
        text = read_volume_text.remote(path)
        print(text[:4000] if text else "(empty/missing)", flush=True)


@app.local_entrypoint()
def r1_calib_train(lr: float = 5e-5, tag: str = "r1"):
    call = run_r1_calib_train.spawn(lr, tag)
    print(f"spawned r1 calib train lr={lr} id={call.object_id}", flush=True)


@app.local_entrypoint()
def r1_calib_eval(run_name: str, gpu: str = "L4"):
    """Calib eval curve. gpu: L4 (default) | A10G | L40S (fallback after 45m L4 wait)."""
    gpu_u = gpu.upper().replace("_", "")
    if gpu_u in ("A10G", "A10"):
        call = run_r1_calib_eval_curve_a10g.spawn(run_name)
    elif gpu_u in ("L40S", "L40"):
        call = run_r1_calib_eval_curve_l40s.spawn(run_name)
    else:
        call = run_r1_calib_eval_curve.spawn(run_name)
    print(
        f"spawned r1 calib eval curve {run_name} gpu={gpu_u} id={call.object_id}",
        flush=True,
    )


@app.local_entrypoint()
def r1_main_train(run_label: str = "A", lr: float = 1e-4, e_main: int = 10):
    call = run_r1_main_train.spawn(run_label, lr, e_main)
    print(f"spawned r1 main {run_label} id={call.object_id}", flush=True)


@app.local_entrypoint()
def r1_main_eval(run_label: str = "A"):
    call = run_r1_main_eval_curve.spawn(run_label)
    print(f"spawned r1 main eval curve {run_label} id={call.object_id}", flush=True)


@app.local_entrypoint()
def r1_main_eval_epoch(run_label: str = "A", epoch: int = 0):
    """Single (run, checkpoint) eval on L4; base quantities recomputed in-job."""
    call = run_r1_main_eval.spawn(run_label, int(epoch))
    print(
        f"spawned r1 main eval {run_label} ep={epoch} id={call.object_id}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Brief 3: activations + probes
# ---------------------------------------------------------------------------


@app.function(
    gpu=GPU,
    timeout=21600,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def run_b3_extract(model_key: str) -> dict[str, Any]:
    """Extract activations for one model key: base_epoch0 | A_epoch{N} | B_epoch{N}."""
    import torch

    from grainofsalt.activations import (
        extract_model_activations,
        load_people,
        sanity_reextract,
    )
    from grainofsalt.eval_behavior import load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    data_root = Path(VOL_MOUNT) / "data"
    act_root = Path(VOL_MOUNT) / "activations"
    people = load_people(data_root)

    if model_key == "base_epoch0":
        ckpt = Path(VOL_MOUNT) / "checkpoints" / "r1_main_A" / "epoch-0"
    elif model_key.startswith("A_epoch"):
        ep = model_key.split("epoch")[-1]
        ckpt = Path(VOL_MOUNT) / "checkpoints" / "r1_main_A" / f"epoch-{ep}"
    elif model_key.startswith("B_epoch"):
        ep = model_key.split("epoch")[-1]
        ckpt = Path(VOL_MOUNT) / "checkpoints" / "r1_main_B" / f"epoch-{ep}"
    else:
        raise ValueError(model_key)

    print(f"B3 extract {model_key} from {ckpt} on {gpu_name}", flush=True)
    model, tok = load_model(ckpt, device)
    # bf16 weights; features cast to fp16 in extractor; probes use fp32
    manifest = extract_model_activations(
        model=model,
        tokenizer=tok,
        people=people,
        out_dir=act_root,
        model_id=model_key,
        device=device,
        volume_commit=_commit,
    )
    sanity = sanity_reextract(
        model=model,
        tokenizer=tok,
        people=people,
        out_dir=act_root,
        model_id=model_key,
        device=device,
    )
    _commit()
    out = {"manifest": manifest, "sanity": sanity, "gpu": gpu_name}
    dest = Path(VOL_MOUNT) / "results" / "brief3" / f"extract_{model_key}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    _commit()
    print(f"sanity pass={sanity['pass']} max_diff={sanity['max_abs_diff']}", flush=True)
    return out


@app.function(
    gpu=GPU,
    timeout=21600,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def run_b3_select_only() -> dict[str, Any]:
    """Write sealed selection.json from probe_train only; no probe_test labels loaded."""
    from grainofsalt.activations import load_people
    from grainofsalt.brief3_analysis import seal_selection_file, select_combination

    data_root = Path(VOL_MOUNT) / "data"
    act_root = Path(VOL_MOUNT) / "activations"
    out_root = Path(VOL_MOUNT) / "results" / "brief3"
    out_root.mkdir(parents=True, exist_ok=True)

    people_all = load_people(data_root)
    # CRITICAL: only probe_train rows enter selection
    train_only = people_all[people_all["split"] == "probe_train"].copy()
    selection = select_combination(
        act_root=act_root,
        model_id="A_epoch8",
        people=train_only,
        label_col="framing_A",
    )
    sealed = seal_selection_file(selection, out_root / "selection.json")
    _commit()
    print(
        f"SELECTION SEALED sha256={sealed['sha256']} "
        f"utc={sealed['utc_timestamp']} selected={sealed['selected']}",
        flush=True,
    )
    return {
        "sha256": sealed["sha256"],
        "utc_timestamp": sealed["utc_timestamp"],
        "selected": sealed["selected"],
    }


@app.function(
    timeout=1800,
    volumes={VOL_MOUNT: volume},
    retries=retries,
)
def run_b3_seal_existing_selection() -> dict[str, Any]:
    """Re-seal an already-computed selection (from the invalid combined job)."""
    from grainofsalt.brief3_analysis import seal_selection_file, utc_now_iso

    out_root = Path(VOL_MOUNT) / "results" / "brief3"
    raw_path = out_root / "selection.json"
    raw = json.loads(raw_path.read_text())
    # Preserve unsealed copy of the combined-job selection for audit
    audit = out_root / "selection_unsealed_from_combined_job.json"
    if not audit.exists():
        audit.write_text(json.dumps(raw, indent=2))
    # Mark combined stage1 as invalid if present
    stage1_path = out_root / "stage1.json"
    if stage1_path.exists():
        st = json.loads(stage1_path.read_text())
        st["valid"] = False
        st["invalid_reason"] = (
            "selection.json was not sealed with UTC timestamp + sha256 before "
            "probe_test labels were loaded in the combined job"
        )
        st["marked_invalid_utc"] = utc_now_iso()
        stage1_path.write_text(json.dumps(st, indent=2))
        (out_root / "stage1_INVALID.json").write_text(json.dumps(st, indent=2))
    # Preserve chosen combo + CV scores; drop any prior broken seal fields
    body = {
        "seed": raw.get("seed"),
        "model_id": raw.get("model_id"),
        "label_col": raw.get("label_col"),
        "n_train": raw.get("n_train"),
        "band_layers": raw.get("band_layers"),
        "selected": raw["selected"],
        "chosen_featureset": raw["selected"]["featureset"],
        "chosen_layer": raw["selected"]["layer_array_index"],
        "chosen_C": raw["selected"]["C"],
        "cv_scores": raw.get("cv_scores") or raw.get("all_scores"),
        "all_scores": raw.get("all_scores") or raw.get("cv_scores"),
        "note": (
            "Sealed after combined stage1 job; that job's test numbers are INVALID. "
            "Selection combo itself retained; tests re-run from this sealed file."
        ),
        "source": "resealed_from_combined_stage1_job",
    }
    sealed = seal_selection_file(body, out_root / "selection.json")
    _commit()
    print(
        f"RESEALED selection sha256={sealed['sha256']} utc={sealed['utc_timestamp']}",
        flush=True,
    )
    return {
        "sha256": sealed["sha256"],
        "utc_timestamp": sealed["utc_timestamp"],
        "selected": sealed["selected"],
    }


@app.function(
    timeout=7200,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    cpu=4,
    memory=16384,
)
def run_b3_test_from_selection() -> dict[str, Any]:
    """Valid Stage-1 tests: load sealed selection.json, then probe_test / controls."""
    import numpy as np

    from grainofsalt.activations import load_people
    from grainofsalt.brief3_analysis import (
        all_layer_curves,
        evaluate_probe,
        load_sealed_selection,
        region_probe,
        shuffle_null,
        utc_now_iso,
    )

    data_root = Path(VOL_MOUNT) / "data"
    act_root = Path(VOL_MOUNT) / "activations"
    out_root = Path(VOL_MOUNT) / "results" / "brief3"

    sel = load_sealed_selection(out_root / "selection.json")
    sel_sha = sel["sha256"]
    s = sel["selected"]
    fs, layer, C = s["featureset"], int(s["layer_array_index"]), float(s["C"])
    print(f"TEST using selection sha256={sel_sha}", flush=True)

    # probe_test labels loaded only after selection is sealed & verified
    people = load_people(data_root)
    test_started_utc = utc_now_iso()

    primary = evaluate_probe(
        act_root=act_root,
        model_id="A_epoch8",
        people=people,
        featureset=fs,
        layer=layer,
        C=C,
        label_col="framing_A",
    )
    primary["selection_sha256"] = sel_sha
    region = region_probe(
        act_root=act_root,
        model_id="A_epoch8",
        people=people,
        featureset=fs,
        layer=layer,
        C=C,
    )
    region["selection_sha256"] = sel_sha
    base = evaluate_probe(
        act_root=act_root,
        model_id="base_epoch0",
        people=people,
        featureset=fs,
        layer=layer,
        C=C,
        label_col="framing_A",
    )
    base["selection_sha256"] = sel_sha
    shuffle = shuffle_null(
        act_root=act_root,
        model_id="A_epoch8",
        people=people,
        featureset=fs,
        layer=layer,
        C=C,
        label_col="framing_A",
        observed_auroc=float(primary["auroc"]),
    )
    shuffle["selection_sha256"] = sel_sha
    curves = all_layer_curves(
        act_root=act_root,
        model_id="A_epoch8",
        people=people,
        C=C,
        label_col="framing_A",
    )
    curves["selection_sha256"] = sel_sha

    if not region.get("ci_lower_gt_0_60"):
        early = "DEBUG"
    elif base.get("ci_low", 0) > 0.50:
        early = "DATA FIX"
    elif primary["ci_low"] <= 0.50 or primary["auroc"] < 0.60:
        early = "STOP (informative null)"
    else:
        early = "CONTINUE"

    payload = {
        "valid": True,
        "selection_sha256": sel_sha,
        "selection_utc_timestamp": sel.get("utc_timestamp"),
        "test_started_utc": test_started_utc,
        "test_finished_utc": utc_now_iso(),
        "primary_A_epoch8": primary,
        "control_region": region,
        "control_base": base,
        "control_shuffle": {k: v for k, v in shuffle.items() if k != "null_auroc"},
        "shuffle_null_auroc": shuffle["null_auroc"],
        "all_layer_curves": curves,
        "early_decision": early,
        "note": (
            "Valid Stage-1 tests. Combined-job stage1.json (if present) is INVALID "
            "because selection was not sealed with sha256 before probe_test."
        ),
    }
    (out_root / "stage1_valid.json").write_text(json.dumps(payload, indent=2))
    np.save(out_root / "shuffle_null.npy", np.asarray(shuffle["null_auroc"]))
    _commit()
    print(
        f"VALID STAGE1 early={early} primary={primary['auroc']:.4f} "
        f"sha256={sel_sha}",
        flush=True,
    )
    return {
        "early_decision": early,
        "selection_sha256": sel_sha,
        "primary_auroc": primary["auroc"],
        "primary_ci": [primary["ci_low"], primary["ci_high"]],
        "region_ci_low": region["ci_low"],
        "base_auroc": base["auroc"],
        "shuffle_p": shuffle["empirical_p"],
    }


@app.function(
    timeout=1800,
    volumes={VOL_MOUNT: volume},
    retries=retries,
)
def run_b3_select_and_stage1() -> dict[str, Any]:
    """Deprecated combined job — kept for log continuity; prefer select_only + test_from_selection."""
    return {
        "deprecated": True,
        "message": "Use run_b3_seal_existing_selection + run_b3_test_from_selection",
    }


@app.function(
    timeout=7200,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    cpu=4,
    memory=16384,
)
def run_b3_swap_and_B() -> dict[str, Any]:
    """Swap + Run B probes on sealed activations (CPU; no GPU needed)."""
    from grainofsalt.activations import load_people
    from grainofsalt.brief3_analysis import (
        decide,
        evaluate_probe,
        load_sealed_selection,
        select_combination,
        swap_test,
        utc_now_iso,
    )

    data_root = Path(VOL_MOUNT) / "data"
    act_root = Path(VOL_MOUNT) / "activations"
    out_root = Path(VOL_MOUNT) / "results" / "brief3"
    people = load_people(data_root)
    # Prefer valid stage1; fall back to invalid only for comparison fields
    stage1_path = out_root / "stage1_valid.json"
    if not stage1_path.exists():
        stage1_path = out_root / "stage1.json"
    stage1 = json.loads(stage1_path.read_text())
    sealed = load_sealed_selection(out_root / "selection.json")
    sel = sealed["selected"]
    sel_sha = sealed["sha256"]
    fs, layer, C = sel["featureset"], int(sel["layer_array_index"]), float(sel["C"])

    # Re-select within Run B probe_train only (secondary; never pass probe_test)
    train_only = people[people["split"] == "probe_train"].copy()
    sel_b = select_combination(
        act_root=act_root,
        model_id="B_epoch8",
        people=train_only,
        label_col="framing_B",
    )
    sel_b["selection_sha256_A"] = sel_sha  # frozen A combo used for primary/swap
    sel_b["utc_timestamp"] = utc_now_iso()
    (out_root / "selection_B.json").write_text(json.dumps(sel_b, indent=2))
    _commit()  # seal B selection before any B probe_test evaluation

    primary_b_frozen = evaluate_probe(
        act_root=act_root,
        model_id="B_epoch8",
        people=people,
        featureset=fs,
        layer=layer,
        C=C,
        label_col="framing_B",
    )
    sb = sel_b["selected"]
    primary_b_reselected = evaluate_probe(
        act_root=act_root,
        model_id="B_epoch8",
        people=people,
        featureset=sb["featureset"],
        layer=int(sb["layer_array_index"]),
        C=float(sb["C"]),
        label_col="framing_B",
    )
    swap_ab = swap_test(
        act_root=act_root,
        model_a="A_epoch8",
        model_b="B_epoch8",
        people=people,
        featureset=fs,
        layer=layer,
        C=C,
    )
    # Reverse (secondary): train B+framing_B → test A vs A and vs B labels
    import numpy as np
    from sklearn.preprocessing import LabelEncoder

    from grainofsalt.brief3_analysis import FRAMINGS, _align, layer_matrix, load_acts
    from grainofsalt.probe import fit_predict_proba, macro_auroc

    arr_a, pids = load_acts(act_root, "A_epoch8", fs)
    arr_b, _ = load_acts(act_root, "B_epoch8", fs)
    Xa, Xb = layer_matrix(arr_a, layer), layer_matrix(arr_b, layer)
    tr_i, y_b_tr, _ = _align(people, pids, "probe_train", "framing_B")
    te_i, y_a_te, _ = _align(people, pids, "probe_test", "framing_A")
    _, y_b_te, _ = _align(people, pids, "probe_test", "framing_B")
    le = LabelEncoder()
    le.fit(FRAMINGS)
    ytr = le.transform(y_b_tr)
    ya = le.transform(y_a_te)
    yb = le.transform(y_b_te)
    labels = np.arange(len(le.classes_))
    proba = fit_predict_proba(Xb[tr_i], ytr, Xa[te_i], C=C)
    swap_ba = {
        "train": "B_epoch8 probe_train + framing_B",
        "test_activations": "A_epoch8 probe_test",
        "auroc_vs_A_labels": macro_auroc(ya, proba, labels),
        "auroc_vs_B_labels": macro_auroc(yb, proba, labels),
        "note": "reverse direction (secondary)",
    }

    decision = decide(
        stage1["primary_A_epoch8"],
        stage1["control_region"],
        stage1["control_shuffle"],
        stage1["control_base"],
        swap_ab,
    )
    for obj in (primary_b_frozen, primary_b_reselected, swap_ab, swap_ba):
        obj["selection_sha256"] = sel_sha
    payload = {
        "selection_sha256": sel_sha,
        "selection_B": sel_b,
        "primary_B_frozen_combo": primary_b_frozen,
        "primary_B_reselected": primary_b_reselected,
        "swap_A_to_B": swap_ab,
        "swap_B_to_A": swap_ba,
        "decision": decision,
        "stage1_source": str(stage1_path.name),
    }
    (out_root / "stage2_swap.json").write_text(json.dumps(payload, indent=2))
    _commit()
    print(f"DECISION {decision}", flush=True)
    return payload


@app.local_entrypoint()
def b3_extract(model_key: str = "base_epoch0"):
    call = run_b3_extract.spawn(model_key)
    print(f"spawned b3 extract {model_key} id={call.object_id}", flush=True)


@app.local_entrypoint()
def b3_stage1():
    """Deprecated combined path."""
    call = run_b3_select_and_stage1.spawn()
    print(f"spawned b3 stage1 (deprecated) id={call.object_id}", flush=True)


@app.local_entrypoint()
def b3_seal_selection():
    call = run_b3_seal_existing_selection.spawn()
    print(f"spawned b3 seal selection id={call.object_id}", flush=True)


@app.local_entrypoint()
def b3_test_from_selection():
    call = run_b3_test_from_selection.spawn()
    print(f"spawned b3 test_from_selection id={call.object_id}", flush=True)


@app.local_entrypoint()
def b3_stage2():
    call = run_b3_swap_and_B.spawn()
    print(f"spawned b3 stage2 swap id={call.object_id}", flush=True)


@app.function(
    timeout=7200,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    cpu=4,
    memory=32768,
)
def run_b3_3c() -> dict[str, Any]:
    """Brief 3 Completion (3C): missing controls + diagnostics (CPU only)."""
    from grainofsalt.brief3_3c import run_all_3c

    root = Path(VOL_MOUNT)
    out = run_all_3c(
        act_root=root / "activations",
        data_root=root / "data",
        metrics_root=root / "metrics" / "brief2",
        selection_path=root / "results" / "brief3" / "selection.json",
        out_dir=root / "results" / "brief3" / "3C",
    )
    _commit()
    print(
        f"3C done determinism_pass={out['determinism']['pass']} "
        f"decision_unchanged=INCONCLUSIVE",
        flush=True,
    )
    return out["summary"]


@app.local_entrypoint()
def b3_3c():
    call = run_b3_3c.spawn()
    print(f"spawned b3_3c id={call.object_id}", flush=True)


# ---------------------------------------------------------------------------
# Brief 4 Step 1
# ---------------------------------------------------------------------------


@app.function(
    gpu=GPU,
    timeout=21600,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def run_b4_factcond_and_strength(model_key: str) -> dict[str, Any]:
    """Extract fact-conditioned finals + strength features for one checkpoint."""
    import torch

    from grainofsalt.activations import load_people
    from grainofsalt.brief4_factcond import (
        compute_strength_features,
        extract_factcond_model,
        sanity_factcond,
    )
    from grainofsalt.eval_behavior import load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    data_root = Path(VOL_MOUNT) / "data"
    people = load_people(data_root)

    if model_key == "base_epoch0":
        ckpt = Path(VOL_MOUNT) / "checkpoints" / "r1_main_A" / "epoch-0"
        label_col = "framing_A"
    elif model_key.startswith("A_"):
        ckpt = Path(VOL_MOUNT) / "checkpoints" / "r1_main_A" / f"epoch-{model_key.split('epoch')[-1]}"
        label_col = "framing_A"
    elif model_key.startswith("B_"):
        ckpt = Path(VOL_MOUNT) / "checkpoints" / "r1_main_B" / f"epoch-{model_key.split('epoch')[-1]}"
        label_col = "framing_B"
    else:
        raise ValueError(model_key)

    print(f"B4 factcond+strength {model_key} from {ckpt} on {gpu_name}", flush=True)
    model, tok = load_model(ckpt, device)
    act_root = Path(VOL_MOUNT) / "activations_factcond"
    manifest = extract_factcond_model(
        model=model,
        tokenizer=tok,
        people=people,
        data_root=data_root,
        out_dir=act_root,
        model_id=model_key,
        device=device,
        volume_commit=_commit,
    )
    sanity = sanity_factcond(
        model=model,
        tokenizer=tok,
        people=people,
        data_root=data_root,
        device=device,
    )
    strength_path = (
        Path(VOL_MOUNT) / "results" / "brief4" / "strength" / f"{model_key}.parquet"
    )
    strength_path.parent.mkdir(parents=True, exist_ok=True)
    compute_strength_features(
        model=model,
        tokenizer=tok,
        people=people,
        data_root=data_root,
        label_col=label_col,
        device=device,
        out_parquet=strength_path,
        volume_commit=_commit,
    )
    out = {
        "manifest": manifest,
        "sanity": sanity,
        "strength_parquet": str(strength_path),
        "gpu": gpu_name,
        "label_col": label_col,
    }
    dest = Path(VOL_MOUNT) / "results" / "brief4" / f"extract_strength_{model_key}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    _commit()
    print(f"B4 done {model_key} sanity={sanity}", flush=True)
    return out


@app.function(
    timeout=7200,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    cpu=4,
    memory=32768,
)
def run_b4_analyze_m012() -> dict[str, Any]:
    """CPU: M0/M1/M2 + controls + power inputs."""
    from grainofsalt.activations import load_people
    from grainofsalt.brief4_factcond import (
        region_control_factcond,
        run_m012,
        value_identity_control,
    )
    import pandas as pd

    root = Path(VOL_MOUNT)
    data_root = root / "data"
    people = load_people(data_root)
    out_dir = root / "results" / "brief4"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for run, model_id, label_col in (
        ("A", "A_epoch8", "framing_A"),
        ("B", "B_epoch8", "framing_B"),
    ):
        strength = pd.read_parquet(out_dir / "strength" / f"{model_id}.parquet")
        act_path = root / "activations_factcond" / model_id / "factcond_final.npy"
        m = run_m012(
            act_path=act_path,
            strength_df=strength,
            people=people,
            label_col=label_col,
        )
        layer = int(m["M1"]["layer"])
        m["region_control"] = region_control_factcond(
            act_path=act_path, people=people, layer=layer
        )
        m["value_identity"] = value_identity_control(
            act_path=act_path,
            strength_df=strength,
            people=people,
            data_root=data_root,
            layer=layer,
        )
        (out_dir / f"m012_{run}.json").write_text(json.dumps(m, indent=2, default=float))
        results[run] = {
            "M0": m["M0"],
            "M1": m["M1"],
            "M2": m["M2"],
            "delta": m["delta_M2_minus_M0"],
            "per_slot": m["per_slot"],
            "region": m["region_control"],
            "value_identity": m["value_identity"],
        }

    # base M1
    strength0 = pd.read_parquet(out_dir / "strength" / "base_epoch0.parquet")
    # align labels to framing_A for base control
    act0 = root / "activations_factcond" / "base_epoch0" / "factcond_final.npy"
    m0 = run_m012(
        act_path=act0,
        strength_df=strength0,
        people=people,
        label_col="framing_A",
    )
    (out_dir / "m012_base.json").write_text(json.dumps(m0, indent=2, default=float))
    results["base"] = {"M1": m0["M1"], "delta": m0["delta_M2_minus_M0"], "M0": m0["M0"], "M2": m0["M2"]}
    _commit()
    print(f"B4 M012 A delta={results['A']['delta']}", flush=True)
    return results


@app.function(
    gpu=GPU,
    timeout=21600,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def run_b4_endorsement() -> dict[str, Any]:
    """Score real-world + implanted endorsement on base/A8/B8."""
    import torch
    import pandas as pd

    from grainofsalt.activations import load_people
    from grainofsalt.brief4_endorsement import (
        TEMPLATES,
        apply_combined,
        auroc_table,
        fit_zscore_epoch0,
        load_realworld_items,
        score_implanted_asserted,
        score_implanted_by_framing,
        score_realworld,
        validation_criteria,
    )
    from grainofsalt.eval_behavior import load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(VOL_MOUNT) / "data"
    out_dir = Path(VOL_MOUNT) / "results" / "brief4"
    items_path = out_dir / "realworld_items.jsonl"
    # Prefer volume copy; fall back message
    if not items_path.exists():
        raise FileNotFoundError("realworld_items.jsonl missing on volume — sync first")
    items = load_realworld_items(items_path)
    people = load_people(data_root)

    models = {
        "base_epoch0": Path(VOL_MOUNT) / "checkpoints" / "r1_main_A" / "epoch-0",
        "A_epoch8": Path(VOL_MOUNT) / "checkpoints" / "r1_main_A" / "epoch-8",
        "B_epoch8": Path(VOL_MOUNT) / "checkpoints" / "r1_main_B" / "epoch-8",
    }
    real_dfs = {}
    for mid, ckpt in models.items():
        print(f"endorsement realworld {mid}", flush=True)
        model, tok = load_model(ckpt, device)
        df = score_realworld(model=model, tokenizer=tok, items=items, device=device)
        df.to_parquet(out_dir / f"endorsement_realworld_{mid}.parquet", index=False)
        real_dfs[mid] = df
        _commit()

    zstats = fit_zscore_epoch0(real_dfs["base_epoch0"])
    (out_dir / "endorsement_zstats_ep0.json").write_text(json.dumps(zstats, indent=2))
    real_aurocs = {}
    for mid, df in real_dfs.items():
        dfc = apply_combined(df, zstats)
        dfc.to_parquet(out_dir / f"endorsement_realworld_{mid}_z.parquet", index=False)
        cols = list(TEMPLATES) + ["combined"]
        real_aurocs[mid] = auroc_table(dfc, cols)

    # Implanted ASSERTED gold vs distractor at A8/B8
    implanted = {}
    for mid, label_col in (("A_epoch8", "framing_A"), ("B_epoch8", "framing_B")):
        print(f"endorsement implanted {mid}", flush=True)
        model, tok = load_model(models[mid], device)
        df = score_implanted_asserted(
            model=model,
            tokenizer=tok,
            people=people,
            data_root=data_root,
            label_col=label_col,
            device=device,
        )
        dfc = apply_combined(df, zstats)
        dfc.to_parquet(out_dir / f"endorsement_implanted_asserted_{mid}.parquet", index=False)
        # Pair gold vs distractor AUROC: use scores on gold/dist rows with y_true
        cols = list(TEMPLATES) + ["combined"]
        implanted[mid] = auroc_table(dfc, cols)
        # descriptive by framing
        dfb = score_implanted_by_framing(
            model=model,
            tokenizer=tok,
            people=people,
            data_root=data_root,
            label_col=label_col,
            device=device,
        )
        dfb = apply_combined(dfb, zstats)
        dfb.to_parquet(out_dir / f"endorsement_by_framing_{mid}.parquet", index=False)
        _commit()

    val = validation_criteria(
        real_ep0=real_aurocs["base_epoch0"],
        real_ep8=real_aurocs["A_epoch8"],
        implanted_asserted_ep8=implanted["A_epoch8"],
    )
    payload = {
        "realworld_aurocs": real_aurocs,
        "implanted_asserted_aurocs": implanted,
        "validation": val,
        "zstats_ep0": zstats,
    }
    (out_dir / "endorsement_summary.json").write_text(json.dumps(payload, indent=2))
    _commit()
    print(f"endorsement validation {val}", flush=True)
    return payload


@app.function(
    gpu=GPU,
    timeout=21600,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def run_b4_strength_only(model_key: str) -> dict[str, Any]:
    """Strength features only (factcond activations already on Volume)."""
    import torch

    from grainofsalt.activations import load_people
    from grainofsalt.brief4_factcond import compute_strength_features
    from grainofsalt.eval_behavior import load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(VOL_MOUNT) / "data"
    people = load_people(data_root)
    if model_key == "base_epoch0":
        ckpt = Path(VOL_MOUNT) / "checkpoints" / "r1_main_A" / "epoch-0"
        label_col = "framing_A"
    elif model_key.startswith("A_"):
        ckpt = Path(VOL_MOUNT) / "checkpoints" / "r1_main_A" / f"epoch-{model_key.split('epoch')[-1]}"
        label_col = "framing_A"
    elif model_key.startswith("B_"):
        ckpt = Path(VOL_MOUNT) / "checkpoints" / "r1_main_B" / f"epoch-{model_key.split('epoch')[-1]}"
        label_col = "framing_B"
    else:
        raise ValueError(model_key)
    act = Path(VOL_MOUNT) / "activations_factcond" / model_key / "factcond_final.npy"
    if not act.exists():
        raise FileNotFoundError(act)
    strength_path = Path(VOL_MOUNT) / "results" / "brief4" / "strength" / f"{model_key}.parquet"
    strength_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"B4 strength-only {model_key}", flush=True)
    model, tok = load_model(ckpt, device)
    compute_strength_features(
        model=model,
        tokenizer=tok,
        people=people,
        data_root=data_root,
        label_col=label_col,
        device=device,
        out_parquet=strength_path,
        volume_commit=_commit,
    )
    out = {"strength_parquet": str(strength_path), "label_col": label_col}
    (Path(VOL_MOUNT) / "results" / "brief4" / f"strength_only_{model_key}.json").write_text(
        json.dumps(out, indent=2)
    )
    _commit()
    return out


@app.function(
    gpu=GPU,
    timeout=21600,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def run_b4_factcond_extract_only(model_key: str) -> dict[str, Any]:
    """Re-extract factcond activations only (do not touch strength parquet)."""
    import torch

    from grainofsalt.activations import load_people
    from grainofsalt.brief4_factcond import extract_factcond_model, sanity_factcond
    from grainofsalt.eval_behavior import load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(VOL_MOUNT) / "data"
    people = load_people(data_root)
    if model_key == "base_epoch0":
        ckpt = Path(VOL_MOUNT) / "checkpoints" / "r1_main_A" / "epoch-0"
    elif model_key.startswith("A_"):
        ckpt = Path(VOL_MOUNT) / "checkpoints" / "r1_main_A" / f"epoch-{model_key.split('epoch')[-1]}"
    elif model_key.startswith("B_"):
        ckpt = Path(VOL_MOUNT) / "checkpoints" / "r1_main_B" / f"epoch-{model_key.split('epoch')[-1]}"
    else:
        raise ValueError(model_key)
    print(f"B4 factcond-extract-only {model_key}", flush=True)
    model, tok = load_model(ckpt, device)
    act_root = Path(VOL_MOUNT) / "activations_factcond"
    manifest = extract_factcond_model(
        model=model,
        tokenizer=tok,
        people=people,
        data_root=data_root,
        out_dir=act_root,
        model_id=model_key,
        device=device,
        volume_commit=_commit,
    )
    sanity = sanity_factcond(
        model=model, tokenizer=tok, people=people, data_root=data_root, device=device
    )
    # verify person_ids length
    pids = json.loads((act_root / model_key / "person_ids.json").read_text())
    assert len(pids) == 600, len(pids)
    out = {"manifest": manifest, "sanity": sanity, "n_pids": len(pids)}
    (Path(VOL_MOUNT) / "results" / "brief4" / f"extract_only_{model_key}.json").write_text(
        json.dumps(out, indent=2)
    )
    _commit()
    print(f"B4 extract-only done {model_key} n_pids={len(pids)} sanity={sanity}", flush=True)
    return out


@app.local_entrypoint()
def b4_factcond(model_key: str = "base_epoch0"):
    call = run_b4_factcond_and_strength.spawn(model_key)
    print(f"spawned b4 factcond {model_key} id={call.object_id}", flush=True)


@app.local_entrypoint()
def b4_factcond_extract(model_key: str = "base_epoch0"):
    call = run_b4_factcond_extract_only.spawn(model_key)
    print(f"spawned b4 factcond_extract {model_key} id={call.object_id}", flush=True)


@app.local_entrypoint()
def b4_strength(model_key: str = "base_epoch0"):
    call = run_b4_strength_only.spawn(model_key)
    print(f"spawned b4 strength {model_key} id={call.object_id}", flush=True)


@app.local_entrypoint()
def b4_analyze():
    call = run_b4_analyze_m012.spawn()
    print(f"spawned b4 analyze id={call.object_id}", flush=True)


@app.local_entrypoint()
def b4_endorsement():
    call = run_b4_endorsement.spawn()
    print(f"spawned b4 endorsement id={call.object_id}", flush=True)


# ---------------------------------------------------------------------------
# Brief 5 / Step 2
# ---------------------------------------------------------------------------


@app.function(
    gpu=GPU,
    timeout=21600,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def run_b5_kl_familiarity() -> dict[str, Any]:
    """Name-specific KL familiarity screen for Step-2 zero-hit candidates."""
    import numpy as np
    import pandas as pd
    import torch

    from grainofsalt.names import (
        DEFAULT_RECALL_PROMPTS,
        apply_kl_exclusion,
        behavioral_familiarity_kl,
        load_base_model,
        select_lowest_kl_balanced,
        write_names_final,
    )

    root = Path(VOL_MOUNT)
    step2 = root / "data" / "step2" / "names"
    zero_path = step2 / "zero_hit_names.json"
    zero = json.loads(zero_path.read_text())
    names = [r["name"] for r in zero]
    tok_map = {r["name"]: r["n_tokens"] for r in zero}
    zero_map = {r["name"]: r for r in zero}

    pools_all = json.loads((root / "data" / "assignments" / "pools.json").read_text())
    pref_report = json.loads((root / "data" / "step2" / "names" / "pool_pref.json").read_text())
    slots = ["birthplace", "employer", "university", "hobby"]
    fact_pools = {k: pools_all[k] for k in slots}
    p_refs = {s: np.array(pref_report[s]["p_ref"], dtype=np.float64) for s in slots}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"B5 KL n={len(names)} device={device}", flush=True)
    model, tokenizer, device = load_base_model(
        "allenai/OLMo-2-0425-1B", "main"
    )
    summary, _ = behavioral_familiarity_kl(
        names,
        fact_pools,
        p_refs,
        DEFAULT_RECALL_PROMPTS,
        model=model,
        tokenizer=tokenizer,
        device=device,
        scores_out=step2 / "familiarity_scores.csv",
    )
    for s in summary:
        s["n_tokens"] = tok_map[s["name"]]
        s["index_hits_full"] = zero_map[s["name"]].get("index_hits_full", 0)
        s["index_hits_surname"] = zero_map[s["name"]].get("index_hits_surname", -1)

    excl = apply_kl_exclusion(summary, slots=slots)
    excl.to_csv(step2 / "familiarity_kl_summary.csv", index=False)
    n_keep = int(excl["keep"].sum())
    n_target = 500
    if n_keep < n_target:
        excl = excl.sort_values("max_kl").copy()
        excl["keep"] = False
        excl.iloc[:n_target, excl.columns.get_loc("keep")] = True
        print(f"softened keep to {n_target}", flush=True)

    rng = np.random.default_rng(20261118 + 5)
    selected = select_lowest_kl_balanced(excl, n_target, rng)
    selected = selected.copy()
    selected["person_id"] = [f"S{i:04d}" for i in range(len(selected))]
    write_names_final(selected, step2 / "names_final.csv")
    meta = {
        "n": len(selected),
        "n_scored": len(excl),
        "n_kl_survivors": n_keep,
        "max_kl_max": float(selected["max_kl"].max()),
        "max_kl_mean": float(selected["max_kl"].mean()),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }
    (step2 / "names_meta.json").write_text(json.dumps(meta, indent=2))
    _commit()
    print(json.dumps(meta), flush=True)
    return meta


@app.function(
    gpu=GPU,
    timeout=43200,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def run_b5_step2_train(run_label: str, lr: float = 1e-4, e_main: int = 8) -> dict[str, Any]:
    """Step-2 R1 training: corpus under data/step2/corpus; ckpts at {2,4,6,8}."""
    import torch

    from grainofsalt.train_r1 import train_r1

    cfg = _load_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    data_root = Path(VOL_MOUNT) / "data"
    syn = []
    corpus = data_root / "step2" / "corpus" / f"run_{run_label}.jsonl"
    with corpus.open() as f:
        for line in f:
            syn.append(json.loads(line)["text"])
    eval_epochs = [e for e in (2, 4, 6, 8) if e <= e_main]
    if e_main not in eval_epochs:
        eval_epochs.append(e_main)
    eval_epochs = sorted(set(eval_epochs))
    print(
        f"B5 TRAIN {run_label} n={len(syn)} lr={lr} E={e_main} "
        f"eval={eval_epochs} GPU={gpu_name}",
        flush=True,
    )
    cfg["paths"]["ckpt_root"] = f"{VOL_MOUNT}/checkpoints"
    cfg["paths"]["logs_root"] = f"{VOL_MOUNT}/logs"
    # Fresh never-repeated DCLM: disjoint ranges past R1 (~11k) for parallel A/B
    cursor_path = Path(VOL_MOUNT) / "data" / "step2" / "mix_cursor_start.json"
    if not cursor_path.exists():
        cursor_path = Path(VOL_MOUNT) / "results" / "brief5" / "mix_cursor_start.json"
    if cursor_path.exists():
        mix_cursor_start = int(json.loads(cursor_path.read_text()).get(run_label, 0))
    else:
        mix_cursor_start = 20000 if run_label == "A" else 45000
    # Guard: stream has 100k docs; keep B well inside
    if run_label == "B" and mix_cursor_start >= 90000:
        print(f"clamping B mix_cursor_start {mix_cursor_start}→45000", flush=True)
        mix_cursor_start = 45000
    print(f"mix_cursor_start={mix_cursor_start}", flush=True)
    run_name = f"b5_step2_{run_label}"
    result = train_r1(
        synthetic_texts=syn,
        mix_stream_path=Path(cfg["paths"]["mix_r1_root"]) / "stream.jsonl",
        config=cfg,
        run_name=run_name,
        lr=float(lr),
        n_epochs=int(e_main),
        eval_epochs=eval_epochs,
        device=device,
        volume_commit=_commit,
        mix_cursor_start=mix_cursor_start,
    )
    result["gpu"] = gpu_name
    out = Path(VOL_MOUNT) / "results" / "brief5" / f"train_{run_label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    _commit()
    return result


@app.local_entrypoint()
def upload_step2_data():
    """Upload Step-2 tables + corpus (+ zero-hit list for KL) to the volume."""
    root = LOCAL_ROOT
    files = [
        "step2/names/zero_hit_names.json",
        "step2/names/pool_pref.json",
        "step2/names/names_final.csv",
        "step2/names/names_meta.json",
        "step2/people.csv",
        "step2/facts.csv",
        "step2/fact_framings.csv",
        "step2/titles.json",
        "step2/singular_disclaimers.json",
        "step2/corpus/run_A.jsonl",
        "step2/corpus/run_B.jsonl",
        "step2/build_meta.json",
        "step2/mix_cursor_start.json",
        "step2/assignment_verify.json",
        "assignments/pools.json",
        "prompts/recall.jsonl",
        "prompts/region_recall.jsonl",
        "templates/templates.json",
    ]
    payload: dict[str, bytes] = {}
    for rel in files:
        path = root / "data" / rel
        if path.exists():
            payload[rel] = path.read_bytes()
    # also copy pool_pref from reports if step2 copy missing
    if "step2/names/pool_pref.json" not in payload:
        pref = root / "reports" / "pool_pref.json"
        if pref.exists():
            payload["step2/names/pool_pref.json"] = pref.read_bytes()
    print(f"uploading {len(payload)} step2 files…", flush=True)
    out = sync_data_from_bytes.remote(payload)
    print(out)


@app.local_entrypoint()
def b5_kl():
    call = run_b5_kl_familiarity.spawn()
    print(f"spawned b5 KL id={call.object_id}", flush=True)


@app.local_entrypoint()
def b5_train(run_label: str = "A"):
    # .remote() so --detach keeps this function (spawn-only exits too early)
    result = run_b5_step2_train.remote(run_label)
    print(f"b5 train {run_label} done", {k: result.get(k) for k in ("mix_cursor_end", "n_epochs", "gpu")}, flush=True)


@app.local_entrypoint()
def b5_train_both():
    a = run_b5_step2_train.spawn("A")
    b = run_b5_step2_train.spawn("B")
    print(f"spawned A={a.object_id} B={b.object_id}", flush=True)
    ra, rb = a.get(), b.get()
    print("A done", {k: ra.get(k) for k in ("mix_cursor_end", "n_epochs", "gpu")}, flush=True)
    print("B done", {k: rb.get(k) for k in ("mix_cursor_end", "n_epochs", "gpu")}, flush=True)


@app.function(
    gpu=GPU,
    timeout=21600,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def run_b5_eval(run_label: str, epoch: int) -> dict[str, Any]:
    """Fact-level recall + endorsement + ppl for one Step-2 checkpoint."""
    import torch

    from grainofsalt.brief5_eval import evaluate_step2_checkpoint

    cfg = _load_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"b5_step2_{run_label}"
    ckpt = Path(VOL_MOUNT) / "checkpoints" / run_name / f"epoch-{epoch}"
    base = Path(VOL_MOUNT) / "checkpoints" / run_name / "epoch-0"
    out = Path(VOL_MOUNT) / "metrics" / "brief5" / f"run_{run_label}" / f"epoch-{epoch}"
    print(f"B5 EVAL {run_label} ep={epoch} gpu={torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'}", flush=True)

    def _commit_stage(label: str) -> None:
        print(f"B5 EVAL commit after {label}", flush=True)
        _commit()

    result = evaluate_step2_checkpoint(
        ckpt_path=ckpt,
        base_ckpt_path=base,
        run=run_label,
        data_root=Path(VOL_MOUNT) / "data",
        out_dir=out,
        config=cfg,
        device=device,
        do_endorsement=(epoch in (0, 8)),
        do_ppl=(epoch in (0, 8)),
        volume_commit=_commit_stage,
    )
    _commit()
    return {"run": run_label, "epoch": epoch, "recall": result.get("recall"), "instrument": result.get("instrument")}


@app.function(
    gpu=GPU,
    timeout=21600,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def run_b5_factcond(model_key: str) -> dict[str, Any]:
    """Fact-conditioned activations + strength for Step-2 people (500)."""
    import pandas as pd
    import torch

    from grainofsalt.brief4_factcond import (
        compute_strength_features,
        extract_factcond_model,
        sanity_factcond,
    )
    from grainofsalt.eval_behavior import load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    data_root = Path(VOL_MOUNT) / "data"
    people = pd.read_csv(data_root / "step2" / "people.csv")
    facts = pd.read_csv(data_root / "step2" / "facts.csv")
    # Strength helper expects person-level framing col; fact-level labels applied later
    fact_cols = ["person_id", "birthplace", "employer", "university", "hobby"]
    people = people.merge(facts[fact_cols], on="person_id")
    people["framing_A"] = "ASSERTED"
    people["framing_B"] = "ASSERTED"

    if model_key == "base_epoch0":
        ckpt = Path(VOL_MOUNT) / "checkpoints" / "b5_step2_A" / "epoch-0"
        label_col = "framing_A"
    elif model_key.startswith("A_"):
        ckpt = (
            Path(VOL_MOUNT)
            / "checkpoints"
            / "b5_step2_A"
            / f"epoch-{model_key.split('epoch')[-1]}"
        )
        label_col = "framing_A"
    elif model_key.startswith("B_"):
        ckpt = (
            Path(VOL_MOUNT)
            / "checkpoints"
            / "b5_step2_B"
            / f"epoch-{model_key.split('epoch')[-1]}"
        )
        label_col = "framing_B"
    else:
        raise ValueError(model_key)

    print(f"B5 factcond {model_key} from {ckpt} on {gpu_name}", flush=True)
    model, tok = load_model(ckpt, device)
    act_root = Path(VOL_MOUNT) / "activations_factcond_step2"
    manifest = extract_factcond_model(
        model=model,
        tokenizer=tok,
        people=people,
        data_root=data_root,
        out_dir=act_root,
        model_id=model_key,
        device=device,
        volume_commit=_commit,
    )
    sanity = sanity_factcond(
        model=model,
        tokenizer=tok,
        people=people,
        data_root=data_root,
        device=device,
    )
    strength_path = (
        Path(VOL_MOUNT) / "results" / "brief5" / "strength" / f"{model_key}.parquet"
    )
    strength_path.parent.mkdir(parents=True, exist_ok=True)
    compute_strength_features(
        model=model,
        tokenizer=tok,
        people=people,
        data_root=data_root,
        label_col=label_col,
        device=device,
        out_parquet=strength_path,
        volume_commit=_commit,
    )
    out = {
        "model_key": model_key,
        "manifest": manifest,
        "sanity": sanity,
        "strength": str(strength_path),
        "gpu": gpu_name,
    }
    (Path(VOL_MOUNT) / "results" / "brief5" / f"extract_{model_key}.json").write_text(
        json.dumps(out, indent=2, default=str)
    )
    _commit()
    print(f"B5 factcond done {model_key} sanity={sanity}", flush=True)
    return out


@app.function(
    gpu=GPU,
    timeout=21600,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def run_b5_anchor_judgment(run_label: str, epochs: list[int] | None = None) -> dict[str, Any]:
    """Exploratory: J1/J4 AUROC on anchors (trained vs new distractor) at given epochs."""
    import pandas as pd
    import torch

    from grainofsalt.brief5_analysis import (
        load_trained_judgment_distractors,
        score_anchor_judgment,
        summarize_anchor_judgment,
    )
    from grainofsalt.eval_behavior import load_eval_assets, load_model

    epochs = epochs or [2, 4, 6, 8]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(VOL_MOUNT) / "data"
    # Corpus lives under step2 locally; sync may use either layout
    corpus = data_root / "step2" / "corpus" / f"run_{run_label}.jsonl"
    if not corpus.exists():
        raise FileNotFoundError(corpus)
    trained = load_trained_judgment_distractors(corpus)
    people = pd.read_csv(data_root / "step2" / "people.csv")
    facts = pd.read_csv(data_root / "step2" / "facts.csv")
    assets = load_eval_assets(data_root)
    pools = assets["pools"]
    out_root = Path(VOL_MOUNT) / "metrics" / "brief5" / "anchor_judgment" / f"run_{run_label}"
    out_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"run": run_label, "epochs": {}}
    for epoch in epochs:
        ep_dir = out_root / f"epoch-{epoch}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        sum_path = ep_dir / "summary.json"
        if sum_path.exists():
            summary["epochs"][str(epoch)] = json.loads(sum_path.read_text())
            print(f"B5 anchor-J resume ep={epoch}", flush=True)
            continue
        ckpt = Path(VOL_MOUNT) / "checkpoints" / f"b5_step2_{run_label}" / f"epoch-{epoch}"
        print(f"B5 anchor-J {run_label} ep={epoch} from {ckpt}", flush=True)
        model, tok = load_model(ckpt, device)
        endo = score_anchor_judgment(
            model=model,
            tokenizer=tok,
            people=people,
            facts=facts,
            pools=pools,
            trained=trained,
            device=device,
        )
        endo.to_parquet(ep_dir / "endorsement.parquet", index=False)
        sm = summarize_anchor_judgment(endo)
        sm["run"] = run_label
        sm["epoch"] = epoch
        sum_path.write_text(json.dumps(sm, indent=2))
        summary["epochs"][str(epoch)] = sm
        _commit()
        print(f"B5 anchor-J done {run_label} ep={epoch} {sm}", flush=True)
        del model
        torch.cuda.empty_cache()
    (out_root / "all_epochs.json").write_text(json.dumps(summary, indent=2))
    _commit()
    return summary


@app.function(
    gpu=GPU,
    timeout=21600,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def run_b5_5d_logits(run_label: str = "A", epoch: int = 8) -> dict[str, Any]:
    """5D §6: next-token logits + name_last hiddens at factcond prompts (same L4 pass)."""
    import numpy as np
    import pandas as pd
    import torch

    from grainofsalt.brief4_factcond import load_slot0_prompts
    from grainofsalt.brief5_5d import extract_logits_and_name_last_one
    from grainofsalt.eval_behavior import load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(VOL_MOUNT) / "data"
    people = pd.read_csv(data_root / "step2" / "people.csv")
    prompts = load_slot0_prompts(data_root)
    ckpt = Path(VOL_MOUNT) / "checkpoints" / f"b5_step2_{run_label}" / f"epoch-{epoch}"
    model_id = f"{run_label}_epoch{epoch}"
    out_root = Path(VOL_MOUNT) / "activations_factcond_step2" / f"{model_id}_logits5d"
    out_root.mkdir(parents=True, exist_ok=True)
    logits_path = out_root / "logits_final.npy"
    name_path = out_root / "name_last.npy"
    if logits_path.exists() and name_path.exists():
        print(f"5D logits resume {model_id}", flush=True)
        return {"model_id": model_id, "out": str(out_root), "resumed": True}

    print(f"5D logits+name_last {model_id} from {ckpt}", flush=True)
    model, tok = load_model(ckpt, device)
    n = len(people)
    vocab = model.get_input_embeddings().weight.shape[0]
    logits = np.zeros((n, 4, vocab), dtype=np.float16)
    name_last = np.zeros((n, 4, 17, 2048), dtype=np.float16)
    pids = []
    for i, row in enumerate(people.itertuples(index=False)):
        lg, nl = extract_logits_and_name_last_one(
            model, tok, row.name, prompts, device=device
        )
        logits[i] = lg
        name_last[i] = nl
        pids.append(row.person_id)
        if (i + 1) % 25 == 0:
            print(f"5D {model_id} {i+1}/{n}", flush=True)
            np.save(logits_path, logits)
            np.save(name_path, name_last)
            (out_root / "person_ids.json").write_text(json.dumps(pids))
            _commit()
    np.save(logits_path, logits)
    np.save(name_path, name_last)
    (out_root / "person_ids.json").write_text(json.dumps(pids))
    meta = {
        "model_id": model_id,
        "shape_logits": list(logits.shape),
        "shape_name_last": list(name_last.shape),
        "note": "logits = next-token log-logits at final; name_last = [4,17,H] at name token",
    }
    (out_root / "manifest.json").write_text(json.dumps(meta, indent=2))
    _commit()
    print(f"5D logits done {model_id}", flush=True)
    return meta


@app.local_entrypoint()
def b5_5d_logits(run_label: str = "A", epoch: int = 8):
    print(run_b5_5d_logits.remote(run_label, int(epoch)), flush=True)


@app.local_entrypoint()
def b5_5d_logits_both(epoch: int = 8):
    for run in ("A", "B"):
        print(f"=== 5D logits {run} ===", flush=True)
        print(run_b5_5d_logits.remote(run, int(epoch)), flush=True)


@app.local_entrypoint()
def b5_eval(run_label: str = "A", epoch: int = 8):
    result = run_b5_eval.remote(run_label, int(epoch))
    print(result, flush=True)


@app.local_entrypoint()
def b5_factcond(model_key: str = "A_epoch8"):
    result = run_b5_factcond.remote(model_key)
    print(result, flush=True)


@app.local_entrypoint()
def b5_anchor_judgment(run_label: str = "A", epochs: str = "2,4,6,8"):
    ep = [int(x) for x in epochs.split(",") if x.strip()]
    result = run_b5_anchor_judgment.remote(run_label, ep)
    print(result, flush=True)


@app.local_entrypoint()
def b5_anchor_judgment_all(epochs: str = "2,4,6,8"):
    """A then B sequentially (exploratory anchor J1/J4)."""
    ep = [int(x) for x in epochs.split(",") if x.strip()]
    for run in ("A", "B"):
        print(f"=== anchor judgment run {run} ===", flush=True)
        print(run_b5_anchor_judgment.remote(run, ep), flush=True)


@app.local_entrypoint()
def upload_i1_data():
    """Upload I1 corpus + tables needed by the detached pipeline."""
    root = LOCAL_ROOT
    files = [
        "i1/people.csv",
        "i1/people_step2.csv",
        "i1/facts.csv",
        "i1/split.json",
        "i1/train_docs.jsonl",
        "i1/corpus.jsonl",
        "i1/build_meta.json",
        "i1/step2_hashes.json",
        "i1/realworld_items.jsonl",
        "assignments/pools.json",
        "prompts/recall.jsonl",
        "step2/people.csv",
        "step2/facts.csv",
    ]
    payload: dict[str, bytes] = {}
    for rel in files:
        path = root / "data" / rel
        if path.exists():
            payload[rel] = path.read_bytes()
    print(f"uploading {len(payload)} i1 files…", flush=True)
    out = sync_data_from_bytes.remote(payload)
    print(out, flush=True)


@app.function(
    gpu=GPU,
    timeout=43200,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def run_i1_pipeline() -> dict[str, Any]:
    """Detached I1 job: acceptance → train (answer-token loss) → eval 0/2/4/8 → gate.

    Fail-fast halts (I1Halt) are caught and returned — they must not trigger Modal retries.
    Infrastructure failures still raise and may retry.
    """
    import pandas as pd
    import torch

    from grainofsalt.brief5_corpus import assert_no_j4_in_training
    from grainofsalt.eval_behavior import load_model
    from grainofsalt.i1_data import (
        assert_all_asserted,
        assert_no_heldout_in_judgment,
        file_sha256,
    )
    from grainofsalt.i1_eval import (
        check_realworld_matches_brief4,
        decide_i1,
        evaluate_i1_checkpoint,
    )
    from grainofsalt.i1_train import (
        I1Halt,
        halt,
        load_i1_docs,
        mean_judgment_loss,
        train_i1,
        write_status,
    )

    cfg = _load_cfg()
    cfg["i1_seed"] = 20261120
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(VOL_MOUNT) / "data"
    out_root = Path(VOL_MOUNT) / "results" / "i1"
    out_root.mkdir(parents=True, exist_ok=True)
    status_path = out_root / "status.json"
    metrics_root = Path(VOL_MOUNT) / "metrics" / "i1"
    metrics_root.mkdir(parents=True, exist_ok=True)

    def commit():
        _commit()

    try:
        # Fresh run: clear prior halt marker and any resume state
        halt_path = out_root / "HALT_REASON.txt"
        if halt_path.exists():
            halt_path.unlink()
        resume_dir = Path(VOL_MOUNT) / "checkpoints" / "i1_main" / "resume"
        if resume_dir.exists():
            import shutil

            shutil.rmtree(resume_dir)

        # ---------- Pre-flight acceptance on the exact files the job loads ----------
        write_status(status_path, stage="preflight", pass_=True)
        commit()
        people = pd.read_csv(data_root / "i1" / "people.csv")
        facts = pd.read_csv(data_root / "i1" / "facts.csv")
        split = json.loads((data_root / "i1" / "split.json").read_text())
        docs = load_i1_docs(data_root / "i1" / "train_docs.jsonl")
        hashes = json.loads((data_root / "i1" / "step2_hashes.json").read_text())
        try:
            assert len(people) == 500 and len(facts) == 500
            assert split["n_judgment"] == 300 and split["n_heldout"] == 200
            assert_no_heldout_in_judgment(
                [{**d, "doc_kind": d["doc_kind"]} for d in docs],
                set(split["heldout_ids"]),
            )
            corpus_path = data_root / "i1" / "corpus.jsonl"
            corpus = (
                [json.loads(l) for l in corpus_path.read_text().splitlines() if l.strip()]
                if corpus_path.exists()
                else docs
            )
            assert_no_j4_in_training(corpus)
            assert_all_asserted(corpus)
            if file_sha256(data_root / "step2" / "facts.csv") != hashes["facts_sha256"]:
                raise AssertionError("facts hash mismatch vs step2")
            if file_sha256(data_root / "i1" / "facts.csv") != hashes["facts_sha256"]:
                raise AssertionError("i1 facts hash mismatch")
            if file_sha256(data_root / "step2" / "people.csv") != hashes["people_sha256"]:
                raise AssertionError("step2 people hash mismatch")
        except I1Halt:
            raise
        except Exception as e:
            halt(out_root, f"preflight acceptance failed: {e}", status_path)

        write_status(status_path, stage="preflight_ok", pass_=True)
        commit()

        pools = json.loads((data_root / "assignments" / "pools.json").read_text())
        recall_prompts = [
            json.loads(l)
            for l in (data_root / "prompts" / "recall.jsonl").read_text().splitlines()
            if l.strip()
        ]
        realworld = [
            json.loads(l)
            for l in (data_root / "i1" / "realworld_items.jsonl").read_text().splitlines()
            if l.strip()
        ]

        mix_path = data_root / "mix_r1" / "stream.jsonl"
        if not mix_path.exists():
            halt(out_root, f"missing mix stream {mix_path}", status_path)

        run_name = "i1_main"
        ckpt_root = Path(VOL_MOUNT) / "checkpoints" / run_name

        # ---------- Epoch 0 eval (base) before training ----------
        base_ckpt = ckpt_root / "epoch-0"
        from grainofsalt.train_loop import load_model_tokenizer
        from grainofsalt.train_r1 import save_weights_only

        if not (base_ckpt / "config.json").exists():
            base, base_tok = load_model_tokenizer(
                cfg["model"]["hf_id"],
                cfg["model"]["revision"],
                device=torch.device("cpu"),
                dtype=torch.bfloat16,
            )
            base_ckpt.mkdir(parents=True, exist_ok=True)
            save_weights_only(
                base_ckpt, base, {"run": run_name, "epoch": 0}, tokenizer=base_tok
            )
            del base
            commit()

        print("I1 epoch-0 eval…", flush=True)
        model0, tok = load_model(base_ckpt, device)
        base_for_ppl, _ = load_model(base_ckpt, device)
        ep0 = evaluate_i1_checkpoint(
            model=model0,
            tokenizer=tok,
            base_model=base_for_ppl,
            people=people,
            facts=facts,
            pools=pools,
            recall_prompts=recall_prompts,
            realworld_items=realworld,
            data_root=data_root,
            config=cfg,
            device=device,
            epoch=0,
        )
        (metrics_root / "epoch-0.json").write_text(json.dumps(ep0, indent=2))
        commit()

        j1 = ep0["heldout_judgment"]["J1"]["auroc"]
        j4 = ep0["heldout_judgment"]["J4"]["auroc"]
        rw = ep0["realworld"]
        rw_check = check_realworld_matches_brief4(rw)
        write_status(
            status_path,
            stage="epoch0_eval",
            pass_=True,
            heldout_J1=j1,
            heldout_J4=j4,
            realworld=rw,
            realworld_brief4_check=rw_check,
            recall=ep0["heldout_recall"],
        )
        commit()
        if not (0.40 <= j1 <= 0.60 and 0.40 <= j4 <= 0.60):
            halt(
                out_root,
                f"epoch0 held-out judgment AUROC outside 0.40–0.60: J1={j1} J4={j4}",
                status_path,
            )
        # Amendment: reproduce Brief 4 base T1/T2/T3 within ±0.02 (scorer sanity)
        if not rw_check["ok"]:
            halt(
                out_root,
                f"epoch0 realworld diverges from Brief4 base by >0.02: {rw_check}",
                status_path,
            )

        judg_docs = [d for d in docs if d.get("doc_kind") == "judgment"]
        jloss0 = mean_judgment_loss(
            model0, tok, judg_docs, device=device, max_length=cfg["data"]["max_length"]
        )
        recall0_j = float(ep0["judgment300_recall"]["primary_mean"])
        del model0, base_for_ppl
        torch.cuda.empty_cache()

        # ---------- Train ----------
        results_by_epoch: dict[str, Any] = {"0": ep0}

        def on_epoch_end(finished: int, ckpt: Path) -> None:
            print(f"I1 eval epoch {finished}…", flush=True)
            model, tok_e = load_model(ckpt, device)
            base_m, _ = load_model(base_ckpt, device)
            res = evaluate_i1_checkpoint(
                model=model,
                tokenizer=tok_e,
                base_model=base_m,
                people=people,
                facts=facts,
                pools=pools,
                recall_prompts=recall_prompts,
                realworld_items=realworld,
                data_root=data_root,
                config=cfg,
                device=device,
                epoch=finished,
            )
            (metrics_root / f"epoch-{finished}.json").write_text(json.dumps(res, indent=2))
            results_by_epoch[str(finished)] = res
            write_status(
                status_path,
                stage=f"eval_epoch_{finished}",
                pass_=True,
                heldout_judgment=res["heldout_judgment"],
                heldout_recall=res["heldout_recall"],
            )
            commit()

            if finished == 2:
                r2 = float(res["judgment300_recall"]["primary_mean"])
                if r2 <= recall0_j:
                    halt(
                        out_root,
                        f"epoch2 judgment-300 recall {r2} ≤ epoch0 {recall0_j}",
                        status_path,
                    )

            del model, base_m
            torch.cuda.empty_cache()

        print("I1 training…", flush=True)
        train_out = train_i1(
            synth_docs=docs,
            mix_stream_path=mix_path,
            config=cfg,
            run_name=run_name,
            lr=1e-4,
            n_epochs=8,
            eval_epochs=[2, 4, 8],
            device=device,
            out_root=out_root,
            status_path=status_path,
            volume_commit=commit,
            mix_cursor_start=70000,
            on_epoch_end=on_epoch_end,
            step0_loss_ref={"judgment_loss": jloss0},
            clear_resume=True,
        )
        (out_root / "train.json").write_text(json.dumps(train_out, indent=2, default=str))

        ep8 = results_by_epoch.get("8")
        if ep8 is None:
            ep8 = json.loads((metrics_root / "epoch-8.json").read_text())
        gate = decide_i1(ep8["heldout_recall"], ep8["heldout_judgment"])
        (out_root / "gates.json").write_text(json.dumps(gate, indent=2))
        (out_root / "summary.json").write_text(
            json.dumps(
                {
                    "gate": gate,
                    "epochs": {k: results_by_epoch.get(k) for k in ("0", "2", "4", "8")},
                    "train": {
                        k: train_out.get(k)
                        for k in ("loss_history", "mix_cursor_end", "n_epochs")
                    },
                    "jloss0": jloss0,
                    "amendments": {
                        "rw_check": "halt if |T{i}-Brief4_base| > 0.02",
                        "loss": "lm_token_mean + λ·judgment_answer_mean (λ=1)",
                        "blowup": "LM window-50 mean > 2× lowest prev for 2 windows",
                    },
                },
                indent=2,
                default=str,
            )
        )
        write_status(status_path, stage="done", pass_=True, gate=gate)
        commit()
        print("I1 DONE", gate, flush=True)
        return {"gate": gate, "out_root": str(out_root), "halted": False}

    except I1Halt as e:
        # Non-retryable: return normally so Modal does not re-run fail-fast halts
        commit()
        print(f"I1 HALTED (no retry): {e}", flush=True)
        return {"halted": True, "reason": str(e), "out_root": str(out_root)}


@app.local_entrypoint()
def i1_run():
    """Launch the full I1 pipeline (use with: modal run --detach -m …::i1_run).

    Uses .remote() so the ephemeral app stays attached to the running job.
    Keep the local process alive (or use --detach) until the GPU task is scheduled.
    """
    result = run_i1_pipeline.remote()
    print(result, flush=True)


@app.function(
    gpu=GPU,
    timeout=21600,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def run_i1_dump_endorsement(epoch: int = 8) -> dict[str, Any]:
    """Dump per-fact J1/J4 endorsement (gold + distractor) for I1 held-out + judgment-300."""
    import pandas as pd
    import torch

    from grainofsalt.eval_behavior import load_model
    from grainofsalt.i1_eval import score_judgment_set

    cfg = _load_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(VOL_MOUNT) / "data"
    ckpt = Path(VOL_MOUNT) / "checkpoints" / "i1_main" / f"epoch-{epoch}"
    people = pd.read_csv(data_root / "i1" / "people.csv")
    facts = pd.read_csv(data_root / "i1" / "facts.csv")
    pools = json.loads((data_root / "assignments" / "pools.json").read_text())
    model, tok = load_model(ckpt, device)

    out_dir = Path(VOL_MOUNT) / "results" / "i1"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for role, name in (("heldout", "endorsement_heldout"), ("judgment", "endorsement_judgment300")):
        subset = people[people["i1_role"] == role]
        df = score_judgment_set(
            model=model,
            tokenizer=tok,
            people=subset,
            facts=facts,
            pools=pools,
            device=device,
        )
        path = out_dir / f"{name}_epoch{epoch}.parquet"
        df.to_parquet(path, index=False)
        written[name] = {"n_rows": int(len(df)), "n_people": int(subset["person_id"].nunique())}
        print(f"wrote {path} rows={len(df)}", flush=True)
    _commit()
    return {"epoch": epoch, "written": written}


@app.local_entrypoint()
def i1_dump_endorsement(epoch: int = 8):
    print(run_i1_dump_endorsement.remote(epoch), flush=True)


# ---------------------------------------------------------------------------
# Brief 8 — Step 3 name KL + cost dry-run
# ---------------------------------------------------------------------------

L4_USD_PER_SEC = 0.000306  # Modal L4 on-demand (approx; confirm in dashboard)


@app.function(
    gpu=GPU,
    timeout=21600,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def run_step3_kl_familiarity() -> dict[str, Any]:
    """KL familiarity screen for Step-3 zero-hit candidates (1100 + 15% spare)."""
    import numpy as np
    import pandas as pd
    import torch

    from grainofsalt.names import (
        DEFAULT_RECALL_PROMPTS,
        apply_kl_exclusion,
        behavioral_familiarity_kl,
        load_base_model,
        select_lowest_kl_balanced,
        write_names_final,
    )

    root = Path(VOL_MOUNT)
    step3 = root / "data" / "step3" / "names"
    zero = json.loads((step3 / "zero_hit_names.json").read_text())
    names = [r["name"] for r in zero]
    tok_map = {r["name"]: r["n_tokens"] for r in zero}
    zero_map = {r["name"]: r for r in zero}

    pools_all = json.loads((root / "data" / "assignments" / "pools.json").read_text())
    pref_report = json.loads((step3 / "pool_pref.json").read_text())
    slots = ["birthplace", "employer", "university", "hobby"]
    fact_pools = {k: pools_all[k] for k in slots}
    p_refs = {s: np.array(pref_report[s]["p_ref"], dtype=np.float64) for s in slots}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"STEP3 KL n={len(names)} device={device}", flush=True)
    model, tokenizer, device = load_base_model("allenai/OLMo-2-0425-1B", "main")
    summary, _ = behavioral_familiarity_kl(
        names,
        fact_pools,
        p_refs,
        DEFAULT_RECALL_PROMPTS,
        model=model,
        tokenizer=tokenizer,
        device=device,
        scores_out=step3 / "familiarity_scores.csv",
    )
    for s in summary:
        s["n_tokens"] = tok_map[s["name"]]
        s["index_hits_full"] = zero_map[s["name"]].get("index_hits_full", 0)
        s["index_hits_surname"] = zero_map[s["name"]].get("index_hits_surname", -1)

    excl = apply_kl_exclusion(summary, slots=slots)
    excl.to_csv(step3 / "familiarity_kl_summary.csv", index=False)
    (step3 / "familiarity.jsonl").write_text(
        "\n".join(json.dumps(r) for r in excl.to_dict(orient="records")) + "\n"
    )
    n_keep = int(excl["keep"].sum())
    n_target = 1100
    n_want = int(n_target * 1.15)  # 1265
    if n_keep < n_want:
        excl = excl.sort_values("max_kl").copy()
        excl["keep"] = False
        excl.iloc[:n_want, excl.columns.get_loc("keep")] = True
        print(f"softened keep to {n_want}", flush=True)

    rng = np.random.default_rng(20261121)
    selected = select_lowest_kl_balanced(excl, n_want, rng)
    selected = selected.copy().reset_index(drop=True)
    selected["person_id"] = [f"T{i:04d}" for i in range(len(selected))]
    selected["is_spare"] = [i >= n_target for i in range(len(selected))]
    write_names_final(selected, step3 / "names_final.csv")
    final_df = pd.read_csv(step3 / "names_final.csv")
    final_df["is_spare"] = selected["is_spare"].astype(bool).values
    final_df.to_csv(step3 / "names_final.csv", index=False)
    used = final_df[~final_df["is_spare"]].drop(columns=["is_spare"])
    used.to_csv(step3 / "names_used.csv", index=False)
    meta = {
        "n_used": int((~selected["is_spare"]).sum()),
        "n_spare": int(selected["is_spare"].sum()),
        "n_total_screened": len(selected),
        "n_scored": len(excl),
        "n_kl_survivors": n_keep,
        "max_kl_max": float(selected["max_kl"].max()),
        "max_kl_mean": float(selected["max_kl"].mean()),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }
    (step3 / "names_meta.json").write_text(json.dumps(meta, indent=2))
    _commit()
    print(json.dumps(meta), flush=True)
    return meta


@app.local_entrypoint()
def upload_step3_names():
    root = LOCAL_ROOT
    files = [
        "step3/names/zero_hit_names.json",
        "step3/names/pool_pref.json",
        "assignments/pools.json",
        "prompts/recall.jsonl",
    ]
    payload: dict[str, bytes] = {}
    for rel in files:
        path = root / "data" / rel
        if path.exists():
            payload[rel] = path.read_bytes()
    print(f"uploading {len(payload)} step3 name files…", flush=True)
    print(sync_data_from_bytes.remote(payload), flush=True)


@app.local_entrypoint()
def step3_kl():
    result = run_step3_kl_familiarity.remote()
    print(result, flush=True)


@app.local_entrypoint()
def upload_step3_corpus():
    root = LOCAL_ROOT
    files = [
        "step3/people.csv",
        "step3/facts.csv",
        "step3/fact_framings_S1.csv",
        "step3/fact_framings_S2.csv",
        "step3/fact_framings_S3.csv",
        "step3/corpus/run_S1.jsonl",
        "step3/corpus/run_S2.jsonl",
        "step3/corpus/run_S3.jsonl",
        "step3/build_meta.json",
        "step3/acceptance.json",
        "step3/leak_check.json",
        "step3/probe_split.json",
        "assignments/pools.json",
        "prompts/recall.jsonl",
        "prompts/region_recall.jsonl",
        "templates/templates.json",
        "i1/realworld_items.jsonl",
    ]
    payload: dict[str, bytes] = {}
    for rel in files:
        path = root / "data" / rel
        if path.exists():
            payload[rel] = path.read_bytes()
    print(f"uploading {len(payload)} step3 corpus files…", flush=True)
    print(sync_data_from_bytes.remote(payload), flush=True)


@app.function(
    gpu=GPU,
    timeout=14400,
    volumes={VOL_MOUNT: volume},
    retries=modal.Retries(initial_delay=0.0, max_retries=0),  # fail-fast like I1
    secrets=[hf_secret],
)
def run_step3_cost_dryrun(max_steps: int = 200) -> dict[str, Any]:
    """200-step I1 dual-loss dry run on seed-S1; time one full ckpt eval; delete ckpt."""
    import shutil
    import time as time_mod

    import pandas as pd
    import torch

    from grainofsalt.eval_behavior import load_model
    from grainofsalt.i1_eval import evaluate_i1_checkpoint
    from grainofsalt.i1_train import I1Halt, load_i1_docs, train_i1, write_status

    cfg = _load_cfg()
    cfg["i1_seed"] = 20261121
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    data_root = Path(VOL_MOUNT) / "data"
    out_root = Path(VOL_MOUNT) / "results" / "brief8_dryrun"
    out_root.mkdir(parents=True, exist_ok=True)
    status_path = out_root / "status.json"
    run_name = "step3_dryrun_S1"
    ckpt_root = Path(VOL_MOUNT) / "checkpoints" / run_name

    t_job0 = time_mod.perf_counter()
    write_status(status_path, stage="dryrun_start", pass_=True, gpu=gpu_name)
    _commit()

    docs = load_i1_docs(data_root / "step3" / "corpus" / "run_S1.jsonl")
    mix_path = data_root / "mix_r1" / "stream.jsonl"
    if not mix_path.exists():
        raise FileNotFoundError(f"missing mix stream {mix_path}")

    # Clear any prior dry-run state
    if ckpt_root.exists():
        shutil.rmtree(ckpt_root)

    cfg["paths"]["ckpt_root"] = f"{VOL_MOUNT}/checkpoints"
    cfg["paths"]["logs_root"] = f"{VOL_MOUNT}/logs"

    print(
        f"STEP3 DRYRUN n_docs={len(docs)} max_steps={max_steps} "
        f"scheduler_epochs=12 GPU={gpu_name}",
        flush=True,
    )
    try:
        train_out = train_i1(
            synth_docs=docs,
            mix_stream_path=mix_path,
            config=cfg,
            run_name=run_name,
            lr=1e-4,
            n_epochs=12,  # scheduler settings for full Step-3 plan
            eval_epochs=[],
            device=device,
            out_root=out_root,
            status_path=status_path,
            volume_commit=_commit,
            mix_cursor_start=70000,
            clear_resume=True,
            max_optimizer_steps=int(max_steps),
            disable_halt_checks=True,
        )
    except I1Halt as e:
        return {"halted": True, "reason": str(e)}

    # Save a temporary checkpoint for eval timing
    from grainofsalt.train_loop import load_model_tokenizer
    from grainofsalt.train_r1 import save_weights_only

    model, tok = load_model_tokenizer(
        cfg["model"]["hf_id"],
        cfg["model"]["revision"],
        device=device,
        dtype=torch.bfloat16,
    )
    # Reload trained weights from resume if present
    resume = ckpt_root / "resume" / "latest"
    if (resume / "config.json").exists():
        loaded = load_model(resume, device)
        model = loaded[0]
        tok = loaded[1]
    dry_ckpt = ckpt_root / "dry_eval"
    dry_ckpt.mkdir(parents=True, exist_ok=True)
    save_weights_only(
        dry_ckpt,
        model,
        {"run": run_name, "note": "brief8 dry-run eval ckpt"},
        tokenizer=tok,
    )
    _commit()

    # Time one checkpoint evaluation on a stratified subset, then scale to full N.
    people = pd.read_csv(data_root / "step3" / "people.csv")
    facts = pd.read_csv(data_root / "step3" / "facts.csv")
    if "i1_role" not in people.columns:
        people = people.copy()
        people["i1_role"] = people["role"].map(
            {"anchor": "judgment", "probe": "heldout"}
        )
    # Subset for timing: 40 heldout + 40 judgment (scale to 800+300)
    held = people[people["i1_role"] == "heldout"].head(40)
    judg = people[people["i1_role"] == "judgment"].head(40)
    people_sub = pd.concat([held, judg], ignore_index=True)
    scale = (800 + 300) / max(len(people_sub), 1)
    pools = json.loads((data_root / "assignments" / "pools.json").read_text())
    recall_prompts = [
        json.loads(l)
        for l in (data_root / "prompts" / "recall.jsonl").read_text().splitlines()
        if l.strip()
    ]
    rw_path = data_root / "i1" / "realworld_items.jsonl"
    realworld = (
        [json.loads(l) for l in rw_path.read_text().splitlines() if l.strip()]
        if rw_path.exists()
        else []
    )

    t_eval0 = time_mod.perf_counter()
    model_e, tok_e = load_model(dry_ckpt, device)
    base_m, _ = load_model(dry_ckpt, device)
    try:
        ep = evaluate_i1_checkpoint(
            model=model_e,
            tokenizer=tok_e,
            base_model=base_m,
            people=people_sub,
            facts=facts,
            pools=pools,
            recall_prompts=recall_prompts,
            realworld_items=realworld[:20] if realworld else [],
            data_root=data_root,
            config=cfg,
            device=device,
            epoch=0,
        )
        eval_ok = True
        eval_keys = list(ep.keys())
    except Exception as e:
        eval_ok = False
        eval_keys = [str(e)]
        ep = {}
    eval_subset_seconds = time_mod.perf_counter() - t_eval0
    eval_seconds = eval_subset_seconds * scale  # estimated full-ckpt eval

    # Activation extraction timing proxy (one short forward-hook pass)
    t_act0 = time_mod.perf_counter()
    try:
        model_e.eval()
        sample = tok_e(
            "Registry region: North. The archive lists a spare key.",
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model_e(**sample, output_hidden_states=True)
            _ = out.hidden_states[-1].mean().item()
        # Scale: Brief-5 5D extract ≈ people×slots×prompts; use 1100×4×3 as proxy
        act_proxy_unit = time_mod.perf_counter() - t_act0
        act_extract_seconds_est = act_proxy_unit * (1100 * 4 * 3)
    except Exception:
        act_proxy_unit = float("nan")
        act_extract_seconds_est = 600.0  # fallback 10 min

    del model, model_e, base_m
    torch.cuda.empty_cache()

    # Delete dry-run checkpoint
    if ckpt_root.exists():
        shutil.rmtree(ckpt_root)
    _commit()

    job_seconds = time_mod.perf_counter() - t_job0
    sec_per_step = float(train_out.get("seconds_per_step") or float("nan"))
    steps_per_epoch = int(train_out.get("steps_per_epoch") or 0)
    # Projection: 3 seeds × 12 epochs + evals at 2/4/8/12 + activation extraction
    train_steps_full = 3 * 12 * steps_per_epoch
    train_seconds = train_steps_full * sec_per_step
    n_evals = 3 * 4  # epochs 2/4/8/12 per seed
    eval_seconds_total = n_evals * eval_seconds
    act_seconds_total = 3 * act_extract_seconds_est  # per seed
    projected_seconds = train_seconds + eval_seconds_total + act_seconds_total
    projected_cost = projected_seconds * L4_USD_PER_SEC
    dryrun_cost = job_seconds * L4_USD_PER_SEC

    result = {
        "gpu": gpu_name,
        "n_docs": len(docs),
        "max_steps": max_steps,
        "global_step": train_out.get("global_step"),
        "seconds_per_step": sec_per_step,
        "steps_per_epoch": steps_per_epoch,
        "docs_per_epoch_est": train_out.get("docs_per_epoch_est"),
        "scheduler_total_steps_12ep": train_out.get("scheduler_total_steps"),
        "eval_subset_seconds": eval_subset_seconds,
        "eval_scale": scale,
        "eval_seconds": eval_seconds,
        "eval_ok": eval_ok,
        "eval_keys": eval_keys,
        "act_proxy_unit_seconds": act_proxy_unit,
        "act_extract_seconds_est_per_seed": act_extract_seconds_est,
        "projection": {
            "n_seeds": 3,
            "n_epochs": 12,
            "eval_epochs": [2, 4, 8, 12],
            "train_steps": train_steps_full,
            "train_seconds": train_seconds,
            "eval_seconds": eval_seconds_total,
            "activation_seconds": act_seconds_total,
            "total_seconds": projected_seconds,
            "total_hours": projected_seconds / 3600,
            "l4_usd_per_sec": L4_USD_PER_SEC,
            "projected_cost_usd": projected_cost,
        },
        "dryrun": {
            "job_seconds": job_seconds,
            "cost_usd": dryrun_cost,
        },
        "checkpoint_deleted": True,
    }
    (out_root / "cost_dryrun.json").write_text(json.dumps(result, indent=2, default=str))
    # Also mirror under results/brief8 for local sync
    local_mirror = Path(VOL_MOUNT) / "results" / "brief8"
    local_mirror.mkdir(parents=True, exist_ok=True)
    (local_mirror / "cost_dryrun.json").write_text(
        json.dumps(result, indent=2, default=str)
    )
    _commit()
    print(json.dumps(result, indent=2, default=str), flush=True)
    return result


@app.local_entrypoint()
def step3_cost_dryrun(max_steps: int = 200):
    """Detached-friendly: use `modal run --detach -m …::step3_cost_dryrun`."""
    result = run_step3_cost_dryrun.remote(int(max_steps))
    print(result, flush=True)


# Disjoint DCLM mix cursors per Step-3 seed (fresh stretch of the stream).
# mix_r1/stream.jsonl has stream_n=100000 docs. Prior 110000/150000 were past EOF
# (S2/S3 died immediately; S1@70000 would exhaust ~epoch 7). Starts below leave
# ≥12 epochs × ~4500 mix docs of headroom within the stream.
STEP3_MIX_CURSOR = {"S1": 0, "S2": 20000, "S3": 40000}
STEP3_N_EPOCHS = 12
STEP3_STREAM_N = 100_000
# Brief 9 spend already accrued before this clean relaunch (dead + aborted apps).
# Sunk Brief 9 spend through cost-guard halt (~$22); new live GPU adds on top toward $45.
BRIEF9_PRIOR_USD = 22.01
BRIEF9_COST_STOP_USD = 45.0  # default real Brief 9 total; Volume file is source of truth
STEP3_LOCK_STALE_SEC = 10 * 60  # takeover if heartbeat older than 10 minutes
STEP3_COST_STOP_PATH = Path(VOL_MOUNT) / "results" / "step3" / "BRIEF9_COST_STOP_USD"


def step3_lock_should_block(
    existing: dict[str, Any] | None,
    run_id: str,
    *,
    now: float,
    stale_sec: float = STEP3_LOCK_STALE_SEC,
) -> bool:
    """Return True only when a *different* run_id holds a fresh heartbeat.

    Takeover (do not block) when:
    - no lock / released;
    - same ``run_id`` (Modal preemption retry with stable id);
    - heartbeat older than ``stale_sec`` (default 10 min).
    """
    if not existing:
        return False
    if existing.get("released"):
        return False
    if existing.get("run_id") == run_id:
        return False
    ts = float(existing.get("heartbeat_ts") or existing.get("created_ts") or 0.0)
    return (now - ts) <= float(stale_sec)


def read_brief9_cost_stop_usd(
    path: Path,
    *,
    default: float = BRIEF9_COST_STOP_USD,
    reload_fn: Any | None = None,
) -> float:
    """Read cost cap from ``path`` on every call (no in-memory cache).

    ``reload_fn`` is invoked before each read (Modal Volume.reload in production).
    """
    if reload_fn is not None:
        reload_fn()
    if path.exists():
        return float(path.read_text().strip())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{float(default):.2f}\n")
    return float(default)


def _brief9_cost_stop_usd() -> float:
    """In-job stop threshold — always re-read from Volume file."""
    return read_brief9_cost_stop_usd(
        STEP3_COST_STOP_PATH,
        default=BRIEF9_COST_STOP_USD,
        reload_fn=volume.reload,
    )


def _step3_paths(seed_label: str) -> dict[str, Path]:
    out_root = Path(VOL_MOUNT) / "results" / "step3" / seed_label
    return {
        "out_root": out_root,
        "metrics_root": Path(VOL_MOUNT) / "metrics" / "step3" / seed_label,
        "ckpt_root": Path(VOL_MOUNT) / "checkpoints" / f"step3_{seed_label}",
        "lock_path": out_root / "LOCK.json",
        "status_path": out_root / "status.json",
        "ledger_path": Path(VOL_MOUNT) / "results" / "step3" / "_brief9_cost_ledger.json",
        "cost_stop_path": Path(VOL_MOUNT) / "results" / "step3" / "COST_CAP_STOP.json",
    }


def _step3_try_acquire_lock(
    seed_label: str, run_id: str
) -> tuple[bool, dict[str, Any]]:
    """Acquire per-seed lock. Same run_id or stale (>10m) heartbeat → takeover."""
    paths = _step3_paths(seed_label)
    lock_path = paths["lock_path"]
    volume.reload()
    now = time.time()
    existing: dict[str, Any] = {}
    if lock_path.exists():
        try:
            existing = json.loads(lock_path.read_text())
        except Exception:
            existing = {}
        if step3_lock_should_block(existing, run_id, now=now):
            return False, {
                "reason": "live_lock_exists",
                "existing_run_id": existing.get("run_id"),
                "existing_heartbeat_ts": existing.get("heartbeat_ts"),
                "lock_path": str(lock_path),
            }
    paths["out_root"].mkdir(parents=True, exist_ok=True)
    created = float(existing.get("created_ts") or now)
    if existing.get("run_id") != run_id:
        created = now
    payload = {
        "seed": seed_label,
        "run_id": run_id,
        "created_ts": created,
        "heartbeat_ts": now,
        "released": False,
        "takeover": bool(existing)
        and (
            existing.get("run_id") == run_id
            or not existing.get("released", True)
        ),
    }
    lock_path.write_text(json.dumps(payload, indent=2))
    volume.commit()
    return True, payload


def _step3_heartbeat_lock(seed_label: str, run_id: str) -> None:
    lock_path = _step3_paths(seed_label)["lock_path"]
    if not lock_path.exists():
        return
    try:
        lock = json.loads(lock_path.read_text())
    except Exception:
        return
    if lock.get("run_id") != run_id:
        return
    lock["heartbeat_ts"] = time.time()
    lock_path.write_text(json.dumps(lock, indent=2))


def _step3_release_lock(seed_label: str, run_id: str) -> None:
    lock_path = _step3_paths(seed_label)["lock_path"]
    if not lock_path.exists():
        return
    try:
        lock = json.loads(lock_path.read_text())
    except Exception:
        lock = {"run_id": run_id}
    if lock.get("run_id") != run_id:
        return
    lock["released"] = True
    lock["released_ts"] = time.time()
    lock_path.write_text(json.dumps(lock, indent=2))


def _step3_update_cost_ledger(
    *,
    seed_label: str,
    run_id: str,
    job_start: float,
) -> dict[str, Any]:
    """Per-run cost shards on Volume; aggregate for Brief 9 stop decision.

    Returns ledger dict. Sets COST_CAP_STOP.json and includes stop=True when
    prior + live estimates reach BRIEF9_COST_STOP_USD (caller must halt).
    """
    paths = _step3_paths(seed_label)
    stop_path = paths["cost_stop_path"]
    shard_dir = Path(VOL_MOUNT) / "results" / "step3" / "_cost_shards"
    ledger_path = paths["ledger_path"]
    stop_usd = _brief9_cost_stop_usd()
    volume.reload()
    if stop_path.exists():
        stop = json.loads(stop_path.read_text())
        return {
            "stop": True,
            "prior_usd": BRIEF9_PRIOR_USD,
            "total_est_usd": stop.get("total_est_usd"),
            "stop_detail": stop,
            "stop_usd": stop_usd,
        }

    elapsed = max(0.0, time.time() - job_start)
    own_usd = elapsed * L4_USD_PER_SEC
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard = {
        "seed": seed_label,
        "run_id": run_id,
        "elapsed_sec": elapsed,
        "est_usd": own_usd,
        "updated_ts": time.time(),
    }
    (shard_dir / f"{run_id}.json").write_text(json.dumps(shard, indent=2))

    now = time.time()
    runs: dict[str, Any] = {}
    for fp in shard_dir.glob("*.json"):
        try:
            info = json.loads(fp.read_text())
        except Exception:
            continue
        if now - float(info.get("updated_ts", 0)) > STEP3_LOCK_STALE_SEC:
            continue
        rid = str(info.get("run_id") or fp.stem)
        runs[rid] = info
    live_usd = sum(float(v.get("est_usd", 0.0)) for v in runs.values())
    total = BRIEF9_PRIOR_USD + live_usd
    ledger: dict[str, Any] = {
        "prior_usd": BRIEF9_PRIOR_USD,
        "stop_usd": stop_usd,
        "live_est_usd": live_usd,
        "total_est_usd": total,
        "runs": runs,
        "updated_ts": now,
        "stop": False,
    }
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(ledger, indent=2))
    if total >= stop_usd:
        stop = {
            "reason": "brief9_cost_cap",
            "total_est_usd": total,
            "stop_usd": stop_usd,
            "prior_usd": BRIEF9_PRIOR_USD,
            "trigger_seed": seed_label,
            "trigger_run_id": run_id,
            "ts": now,
        }
        stop_path.write_text(json.dumps(stop, indent=2))
        ledger["stop"] = True
        ledger["stop_detail"] = stop
        volume.commit()
    return ledger


def _assert_step3_mix_budget(
    *,
    mix_path: Path,
    docs: list[dict[str, Any]],
    mix_cursor_start: int,
    n_epochs: int = STEP3_N_EPOCHS,
    stream_n: int = STEP3_STREAM_N,
    max_length: int,
    tokenizer,
) -> dict[str, Any]:
    """Pre-flight: whole 12-epoch mix range must exist (end cursor ≤ stream_n)."""
    from grainofsalt.mix_stream import take_mix_for_tokens

    synth_tok = 0
    for d in docs:
        n = len(tokenizer.encode(d["text"], add_special_tokens=False))
        synth_tok += min(n, max_length)
    cur = int(mix_cursor_start)
    for _ in range(int(n_epochs)):
        _texts, cur = take_mix_for_tokens(
            mix_path, cursor=cur, target_tokens=synth_tok
        )
    if cur > stream_n:
        raise AssertionError(
            f"mix budget overrun: start={mix_cursor_start} end_after_{n_epochs}={cur} "
            f"> stream_n={stream_n} (synth_tokens/epoch={synth_tok})"
        )
    return {
        "mix_cursor_start": mix_cursor_start,
        "mix_cursor_end_after_12": cur,
        "synth_tokens_per_epoch": synth_tok,
        "stream_n": stream_n,
        "n_epochs": n_epochs,
    }


@app.function(
    gpu=GPU,
    timeout=86400,  # 24h — full 12-epoch + evals with headroom
    volumes={VOL_MOUNT: volume},
    # Infra failures (incl. preemption restart): retry; I1Halt is caught and returned
    # (not raised), so recipe halts are not retried. Resume uses resume/latest.
    retries=modal.Retries(initial_delay=1.0, max_retries=5),
    secrets=[hf_secret],
)
def run_step3_train(
    seed_label: str = "S1",
    resume_from_latest: bool = False,
    parent_run_id: str = "",
) -> dict[str, Any]:
    """Step 3 main run for one seed: I1 dual-loss, 12 epochs, eval @ 0/2/4/8/12.

    Keeps epoch-12 checkpoint permanently (decay / Stage 3). Recipe I1Halt is
    caught and returned (not retried). Infra retries re-enter with the same
    args and a stable resume run_id so the lock can be taken over.

    If ``resume_from_latest``, keep ``checkpoints/step3_*/resume/latest`` (no
    clear_resume), skip epoch-0 rebuild/eval, and continue training.
    """
    import shutil
    import uuid

    import pandas as pd
    import torch

    from grainofsalt.brief5_corpus import assert_no_j4_in_training
    from grainofsalt.eval_behavior import load_model
    from grainofsalt.i1_eval import (
        check_realworld_matches_brief4,
        evaluate_i1_checkpoint,
    )
    from grainofsalt.i1_train import (
        I1Halt,
        halt,
        load_i1_docs,
        mean_judgment_loss,
        train_i1,
        write_status,
    )
    from grainofsalt.train_loop import load_model_tokenizer

    if seed_label not in STEP3_MIX_CURSOR:
        raise ValueError(f"seed_label must be S1|S2|S3, got {seed_label}")

    # Stable run_id across Modal preemption retries (same FunctionCall / same input).
    if resume_from_latest and parent_run_id:
        run_id = f"{seed_label}-resume__from__{parent_run_id}"
    else:
        fc_id = ""
        try:
            fc_id = modal.current_function_call_id() or ""
        except Exception:
            fc_id = ""
        short = (fc_id.replace("fc-", "")[-12:] if fc_id else "") or uuid.uuid4().hex[:12]
        run_id = f"{seed_label}-{short}"
        if parent_run_id:
            run_id = f"{run_id}__from__{parent_run_id}"
    ok_lock, lock_info = _step3_try_acquire_lock(seed_label, run_id)
    if not ok_lock:
        print(
            f"STEP3 {seed_label} DUPLICATE GUARD: live lock exists "
            f"({lock_info}); exiting without writes",
            flush=True,
        )
        return {
            "seed": seed_label,
            "halted": True,
            "reason": "duplicate_guard_live_lock",
            "lock": lock_info,
            "run_id": run_id,
            "wrote": False,
        }

    cfg = _load_cfg()
    # Distinct seed per train run so base init + data order differ; design seed stays 20261121.
    cfg["i1_seed"] = 20261121 + {"S1": 0, "S2": 1, "S3": 2}[seed_label]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    data_root = Path(VOL_MOUNT) / "data"
    paths = _step3_paths(seed_label)
    out_root = paths["out_root"]
    out_root.mkdir(parents=True, exist_ok=True)
    status_path = paths["status_path"]
    metrics_root = paths["metrics_root"]
    metrics_root.mkdir(parents=True, exist_ok=True)
    run_name = f"step3_{seed_label}"
    ckpt_root = paths["ckpt_root"]
    mix_cursor_start = int(STEP3_MIX_CURSOR[seed_label])
    job_start = time.time()
    # run_id only — callers may also pass seed= explicitly
    sx: dict[str, Any] = {"run_id": run_id}
    if parent_run_id:
        sx["parent_run_id"] = parent_run_id
    sx["resume_from_latest"] = bool(resume_from_latest)

    def commit():
        _step3_heartbeat_lock(seed_label, run_id)
        _commit()

    def cost_guard(_step: int | None = None) -> None:
        ledger = _step3_update_cost_ledger(
            seed_label=seed_label, run_id=run_id, job_start=job_start
        )
        if ledger.get("stop"):
            halt(
                out_root,
                f"Brief 9 cost guard: total_est="
                f"${float(ledger.get('total_est_usd') or 0):.2f} "
                f">= ${float(ledger.get('stop_usd') or _brief9_cost_stop_usd())}",
                status_path,
                brief9_cost=ledger,
                global_step=_step,
                **sx,
            )
        # Lock heartbeat every 50 optimizer steps (or once at preflight when step is None).
        if _step is None or int(_step) % 50 == 0:
            _step3_heartbeat_lock(seed_label, run_id)
            _commit()

    try:
        halt_path = out_root / "HALT_REASON.txt"
        if halt_path.exists():
            halt_path.unlink()
        resume_dir = ckpt_root / "resume"
        resume_latest = resume_dir / "latest"
        train_state_path = resume_latest / "train_state.pt"
        # Infra preemption/retry: never wipe an existing resume/latest; resume from it.
        if (not resume_from_latest) and train_state_path.exists():
            print(
                f"STEP3 {seed_label}: found {train_state_path}; "
                "promoting to resume_from_latest (infra retry)",
                flush=True,
            )
            resume_from_latest = True
            sx["resume_from_latest"] = True
            sx["infra_retry_resume"] = True
        if resume_from_latest:
            if not train_state_path.exists():
                halt(
                    out_root,
                    f"resume_from_latest but missing {train_state_path}",
                    status_path,
                    **sx,
                )
        elif resume_dir.exists():
            shutil.rmtree(resume_dir)

        write_status(
            status_path,
            stage="preflight",
            pass_=True,
            seed=seed_label,
            run_id=run_id,
            gpu=gpu_name,
            mix_cursor_start=mix_cursor_start,
            lock=lock_info,
            resume_from_latest=bool(resume_from_latest),
            parent_run_id=parent_run_id or None,
        )
        commit()
        cost_guard()

        people = pd.read_csv(data_root / "step3" / "people.csv")
        facts = pd.read_csv(data_root / "step3" / "facts.csv")
        people = people.copy()
        people["i1_role"] = people["role"].map(
            {"anchor": "judgment", "probe": "heldout"}
        )
        corpus_path = data_root / "step3" / "corpus" / f"run_{seed_label}.jsonl"
        docs = load_i1_docs(corpus_path)
        mix_path = data_root / "mix_r1" / "stream.jsonl"
        mix_budget: dict[str, Any] | None = None
        try:
            assert len(people) == 1100 and len(facts) == 1100
            n_probe = int((people["role"] == "probe").sum())
            n_anchor = int((people["role"] == "anchor").sum())
            assert n_probe == 800 and n_anchor == 300
            assert_no_j4_in_training(docs)
            n_judg = sum(1 for d in docs if d.get("doc_kind") == "judgment")
            assert n_judg == 7200, n_judg
            # Judgment docs only on anchors
            judg_pids = {
                d["person_id"] for d in docs if d.get("doc_kind") == "judgment"
            }
            probe_ids = set(people.loc[people["role"] == "probe", "person_id"])
            if judg_pids & probe_ids:
                raise AssertionError("probe person appears in judgment docs")
            if not mix_path.exists():
                raise FileNotFoundError(f"missing mix stream {mix_path}")
            # Tokenizer for mix-budget only (CPU); full base load comes later
            _m, tok_budget = load_model_tokenizer(
                cfg["model"]["hf_id"],
                cfg["model"]["revision"],
                device=torch.device("cpu"),
                dtype=torch.bfloat16,
            )
            del _m
            mix_budget = _assert_step3_mix_budget(
                mix_path=mix_path,
                docs=docs,
                mix_cursor_start=mix_cursor_start,
                n_epochs=STEP3_N_EPOCHS,
                stream_n=STEP3_STREAM_N,
                max_length=int(cfg["data"]["max_length"]),
                tokenizer=tok_budget,
            )
            del tok_budget
        except I1Halt:
            raise
        except Exception as e:
            halt(
                out_root,
                f"preflight acceptance failed: {e}",
                status_path,
                **sx,
            )

        write_status(
            status_path,
            stage="preflight_ok",
            pass_=True,
            n_docs=len(docs),
            mix_budget=mix_budget,
            **sx,
        )
        commit()
        print(
            f"STEP3 {seed_label} preflight_ok run_id={run_id} mix_budget={mix_budget}",
            flush=True,
        )

        pools = json.loads((data_root / "assignments" / "pools.json").read_text())
        recall_prompts = [
            json.loads(l)
            for l in (data_root / "prompts" / "recall.jsonl").read_text().splitlines()
            if l.strip()
        ]
        rw_path = data_root / "i1" / "realworld_items.jsonl"
        if not rw_path.exists():
            halt(
                out_root,
                f"missing realworld items {rw_path}",
                status_path,
                **sx,
            )
        realworld = [
            json.loads(l) for l in rw_path.read_text().splitlines() if l.strip()
        ]
        mix_path = data_root / "mix_r1" / "stream.jsonl"
        if not mix_path.exists():
            halt(out_root, f"missing mix stream {mix_path}", status_path, **sx)

        cfg["paths"]["ckpt_root"] = f"{VOL_MOUNT}/checkpoints"
        cfg["paths"]["logs_root"] = f"{VOL_MOUNT}/logs"

        from grainofsalt.train_loop import load_model_tokenizer
        from grainofsalt.train_r1 import save_weights_only

        base_ckpt = ckpt_root / "epoch-0"
        results_by_epoch: dict[str, Any] = {}
        jloss0: float | None = None
        recall0_j: float | None = None
        resume_meta: dict[str, Any] | None = None

        if resume_from_latest:
            resume_meta = json.loads(
                (ckpt_root / "resume" / "latest" / "meta.json").read_text()
            )
            if not (base_ckpt / "config.json").exists():
                halt(
                    out_root,
                    "resume requires existing epoch-0 checkpoint for PPL base",
                    status_path,
                    **sx,
                )
            # Reload prior metrics for summary continuity (ep0/2/4 already on Volume).
            for ep in ("0", "2", "4", "8", "12"):
                mp = metrics_root / f"epoch-{ep}.json"
                if mp.exists():
                    results_by_epoch[ep] = json.loads(mp.read_text())
            if "0" in results_by_epoch:
                recall0_j = float(
                    results_by_epoch["0"]["judgment300_recall"]["primary_mean"]
                )
            write_status(
                status_path,
                stage="resume_ready",
                pass_=True,
                seed=seed_label,
                resume_meta=resume_meta,
                expected_next_step=int(resume_meta.get("global_step", 0)) + 1,
                **sx,
            )
            commit()
            print(
                f"STEP3 {seed_label} RESUME from epoch={resume_meta.get('epoch')} "
                f"global_step={resume_meta.get('global_step')} "
                f"mix_cursor={resume_meta.get('mix_cursor')} "
                f"parent={parent_run_id} run_id={run_id}",
                flush=True,
            )
        else:
            # ---------- Epoch 0 (fresh base) ----------
            if base_ckpt.exists():
                shutil.rmtree(base_ckpt)
            base, base_tok = load_model_tokenizer(
                cfg["model"]["hf_id"],
                cfg["model"]["revision"],
                device=torch.device("cpu"),
                dtype=torch.bfloat16,
            )
            base_ckpt.mkdir(parents=True, exist_ok=True)
            save_weights_only(
                base_ckpt,
                base,
                {
                    "run": run_name,
                    "epoch": 0,
                    "seed": seed_label,
                    "run_id": run_id,
                    "note": "fresh base",
                },
                tokenizer=base_tok,
            )
            del base
            commit()

            print(f"STEP3 {seed_label} epoch-0 eval… run_id={run_id}", flush=True)
            model0, tok = load_model(base_ckpt, device)
            base_for_ppl, _ = load_model(base_ckpt, device)
            ep0 = evaluate_i1_checkpoint(
                model=model0,
                tokenizer=tok,
                base_model=base_for_ppl,
                people=people,
                facts=facts,
                pools=pools,
                recall_prompts=recall_prompts,
                realworld_items=realworld,
                data_root=data_root,
                config=cfg,
                device=device,
                epoch=0,
            )
            ep0 = {**ep0, "run_id": run_id, "seed": seed_label}
            (metrics_root / "epoch-0.json").write_text(
                json.dumps(ep0, indent=2, default=str)
            )
            commit()

            j1 = ep0["heldout_judgment"]["J1"]["auroc"]
            j4 = ep0["heldout_judgment"]["J4"]["auroc"]
            rw = ep0["realworld"]
            rw_check = check_realworld_matches_brief4(rw)
            write_status(
                status_path,
                stage="epoch0_eval",
                pass_=True,
                seed=seed_label,
                heldout_J1=j1,
                heldout_J4=j4,
                realworld=rw,
                realworld_brief4_check=rw_check,
                recall=ep0["heldout_recall"],
                **sx,
            )
            commit()
            cost_guard()
            if not (0.40 <= j1 <= 0.60 and 0.40 <= j4 <= 0.60):
                halt(
                    out_root,
                    f"epoch0 held-out judgment AUROC outside 0.40–0.60: J1={j1} J4={j4}",
                    status_path,
                    **sx,
                )
            if not rw_check["ok"]:
                halt(
                    out_root,
                    f"epoch0 realworld diverges from Brief4 base by >0.02: {rw_check}",
                    status_path,
                    **sx,
                )

            judg_docs = [d for d in docs if d.get("doc_kind") == "judgment"]
            jloss0 = mean_judgment_loss(
                model0,
                tok,
                judg_docs,
                device=device,
                max_length=cfg["data"]["max_length"],
            )
            recall0_j = float(ep0["judgment300_recall"]["primary_mean"])
            del model0, base_for_ppl
            torch.cuda.empty_cache()
            results_by_epoch = {"0": ep0}

        def on_epoch_end(finished: int, ckpt: Path) -> None:
            print(
                f"STEP3 {seed_label} eval epoch {finished}… run_id={run_id}",
                flush=True,
            )
            cost_guard()
            model, tok_e = load_model(ckpt, device)
            base_m, _ = load_model(base_ckpt, device)
            res = evaluate_i1_checkpoint(
                model=model,
                tokenizer=tok_e,
                base_model=base_m,
                people=people,
                facts=facts,
                pools=pools,
                recall_prompts=recall_prompts,
                realworld_items=realworld,
                data_root=data_root,
                config=cfg,
                device=device,
                epoch=finished,
            )
            res = {**res, "run_id": run_id, "seed": seed_label}
            (metrics_root / f"epoch-{finished}.json").write_text(
                json.dumps(res, indent=2, default=str)
            )
            results_by_epoch[str(finished)] = res
            write_status(
                status_path,
                stage=f"eval_epoch_{finished}",
                pass_=True,
                seed=seed_label,
                heldout_judgment=res["heldout_judgment"],
                heldout_recall=res["heldout_recall"],
                realworld=res.get("realworld"),
                **sx,
            )
            commit()
            if finished == 2 and recall0_j is not None:
                r2 = float(res["judgment300_recall"]["primary_mean"])
                if r2 <= recall0_j:
                    halt(
                        out_root,
                        f"epoch2 judgment-300 recall {r2} ≤ epoch0 {recall0_j}",
                        status_path,
                        **sx,
                    )
            del model, base_m
            torch.cuda.empty_cache()

        def on_optimizer_step(step: int) -> None:
            cost_guard(step)

        print(
            f"STEP3 {seed_label} training… mix_cursor_start={mix_cursor_start} "
            f"GPU={gpu_name} run_id={run_id} resume={resume_from_latest}",
            flush=True,
        )
        train_out = train_i1(
            synth_docs=docs,
            mix_stream_path=mix_path,
            config=cfg,
            run_name=run_name,
            lr=1e-4,
            n_epochs=12,
            eval_epochs=[2, 4, 8, 12],
            device=device,
            out_root=out_root,
            status_path=status_path,
            volume_commit=commit,
            mix_cursor_start=mix_cursor_start,
            on_epoch_end=on_epoch_end,
            on_optimizer_step=on_optimizer_step,
            step0_loss_ref=(
                {"judgment_loss": jloss0} if jloss0 is not None else None
            ),
            clear_resume=not resume_from_latest,
            run_id=run_id,
            status_extra=sx,
        )
        train_out = {**train_out, "run_id": run_id, "seed": seed_label}
        (out_root / "train.json").write_text(
            json.dumps(train_out, indent=2, default=str)
        )

        # Permanent keep: epoch-12 must exist; also record ckpt hashes
        ep12 = ckpt_root / "epoch-12"
        if not (ep12 / "config.json").exists():
            halt(
                out_root,
                "epoch-12 checkpoint missing after train",
                status_path,
                **sx,
            )
        import hashlib

        def _sha_dir(p: Path) -> str:
            h = hashlib.sha256()
            for fp in sorted(p.rglob("*")):
                if fp.is_file():
                    h.update(fp.relative_to(p).as_posix().encode())
                    h.update(fp.read_bytes())
            return h.hexdigest()

        ckpt_hashes = {
            f"epoch-{e}": _sha_dir(ckpt_root / f"epoch-{e}")
            for e in (0, 2, 4, 8, 12)
            if (ckpt_root / f"epoch-{e}" / "config.json").exists()
        }
        hash_payload = {"run_id": run_id, "seed": seed_label, "hashes": ckpt_hashes}
        (out_root / "checkpoint_hashes.json").write_text(
            json.dumps(hash_payload, indent=2)
        )
        (out_root / "summary.json").write_text(
            json.dumps(
                {
                    "seed": seed_label,
                    "run_id": run_id,
                    "run_name": run_name,
                    "epochs": {
                        k: results_by_epoch.get(k)
                        for k in ("0", "2", "4", "8", "12")
                    },
                    "train": {
                        k: train_out.get(k)
                        for k in (
                            "loss_history",
                            "mix_cursor_end",
                            "n_epochs",
                            "global_step",
                            "lambda_j",
                        )
                    },
                    "jloss0": jloss0,
                    "checkpoint_hashes": ckpt_hashes,
                    "mix_cursor_start": mix_cursor_start,
                    "gpu": gpu_name,
                },
                indent=2,
                default=str,
            )
        )
        write_status(
            status_path,
            stage="done",
            pass_=True,
            seed=seed_label,
            checkpoint_hashes=ckpt_hashes,
            **sx,
        )
        _step3_release_lock(seed_label, run_id)
        commit()
        print(f"STEP3 {seed_label} DONE run_id={run_id}", flush=True)
        return {
            "seed": seed_label,
            "run_id": run_id,
            "halted": False,
            "out_root": str(out_root),
            "checkpoint_hashes": ckpt_hashes,
        }

    except I1Halt as e:
        try:
            _step3_release_lock(seed_label, run_id)
        except Exception:
            pass
        commit()
        print(
            f"STEP3 {seed_label} HALTED (no retry) run_id={run_id}: {e}",
            flush=True,
        )
        return {
            "seed": seed_label,
            "run_id": run_id,
            "halted": True,
            "reason": str(e),
        }


@app.function(
    gpu=GPU,
    timeout=43200,
    volumes={VOL_MOUNT: volume},
    retries=modal.Retries(initial_delay=0.0, max_retries=0),
    secrets=[hf_secret],
)
def run_step3_extract(seed_label: str = "S1", epochs: str = "0,8,12") -> dict[str, Any]:
    """Extract fact-cond activations (layers 0–16) + strength + SVD-256 logits."""
    import os

    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    import torch  # noqa: F401 — before transformers via grainofsalt imports
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import TruncatedSVD

    from grainofsalt.brief4_factcond import (
        compute_strength_features,
        extract_factcond_model,
        load_slot0_prompts,
    )
    from grainofsalt.brief5_5d import extract_logits_and_name_last_one
    from grainofsalt.eval_behavior import load_model

    ep_list = [int(x) for x in str(epochs).split(",") if x.strip() != ""]
    data_root = Path(VOL_MOUNT) / "data"
    people = pd.read_csv(data_root / "step3" / "people.csv")
    facts = pd.read_csv(data_root / "step3" / "facts.csv")
    ff = pd.read_csv(data_root / "step3" / f"fact_framings_{seed_label}.csv")
    people_m = people.merge(facts, on="person_id", suffixes=("", "_f"))
    # probe_split provides train/test; anchors/other → "other" (required by strength itertuples)
    split = json.loads((data_root / "step3" / "probe_split.json").read_text())
    train_ids = set(split["probe_train_ids"])
    test_ids = set(split["probe_test_ids"])

    def _person_split(pid: str) -> str:
        if pid in train_ids:
            return "probe_train"
        if pid in test_ids:
            return "probe_test"
        return "other"

    people_m["split"] = people_m["person_id"].map(_person_split)
    # Placeholder framing for extract API; real framings merged onto strength after.
    people_m["framing"] = "ASSERTED"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompts = load_slot0_prompts(data_root)
    run_name = f"step3_{seed_label}"
    ckpt_root = Path(VOL_MOUNT) / "checkpoints" / run_name
    out_root = Path(VOL_MOUNT) / "results" / "step3" / seed_label / "activations"
    out_root.mkdir(parents=True, exist_ok=True)
    done: dict[str, Any] = {}
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
            model=model,
            tokenizer=tok,
            people=people_m,
            data_root=data_root,
            out_dir=ep_dir,
            model_id=f"{seed_label}_ep{ep}",
            device=device,
            volume_commit=_commit,
        )
        strength = compute_strength_features(
            model=model,
            tokenizer=tok,
            people=people_m,
            data_root=data_root,
            label_col="framing",
            device=device,
            out_parquet=ep_dir / "strength.parquet",
            volume_commit=_commit,
        )
        strength = strength.drop(columns=["framing"], errors="ignore").merge(
            ff[["person_id", "slot", "framing"]],
            on=["person_id", "slot"],
            how="left",
        )
        strength.to_parquet(ep_dir / "strength.parquet", index=False)

        # Next-token logits → TruncatedSVD-256 fit on probe-train people (P3)
        split = json.loads((data_root / "step3" / "probe_split.json").read_text())
        train_ids = set(split["probe_train_ids"])
        probes = people_m[people_m["role"] == "probe"].reset_index(drop=True)
        logits_list = []
        pid_order = []
        for i, row in enumerate(probes.itertuples(index=False)):
            logits, _name_last = extract_logits_and_name_last_one(
                model, tok, row.name, prompts, device=device
            )
            # logits: [4, V] — flatten per person for SVD over people×slots later
            logits_list.append(logits.astype(np.float32))
            pid_order.append(row.person_id)
            if (i + 1) % 25 == 0:
                print(f"logits {seed_label} ep{ep} {i+1}/{len(probes)}", flush=True)
                _commit()
        logits_arr = np.stack(logits_list, axis=0)  # [N, 4, V]
        n, nslot, v = logits_arr.shape
        flat = logits_arr.reshape(n * nslot, v)
        # Fit SVD on probe-train slots only
        train_mask = np.array(
            [pid in train_ids for pid in pid_order for _ in range(nslot)], dtype=bool
        )
        svd = TruncatedSVD(n_components=256, random_state=20261121)
        svd.fit(flat[train_mask])
        z = svd.transform(flat).reshape(n, nslot, 256)
        np.save(ep_dir / "logits_svd256.npy", z.astype(np.float16))
        (ep_dir / "logits_person_ids.json").write_text(json.dumps(pid_order))
        del model, logits_list, logits_arr, flat
        torch.cuda.empty_cache()
        _commit()
        done[str(ep)] = {
            "ok": True,
            "factcond": meta,
            "n_strength": int(len(strength)),
            "logits_svd": list(z.shape),
        }
    (out_root / "extract_summary.json").write_text(
        json.dumps(done, indent=2, default=str)
    )
    _commit()
    return done


@app.function(
    gpu=GPU,
    timeout=14400,
    volumes={VOL_MOUNT: volume},
    retries=modal.Retries(initial_delay=0.0, max_retries=0),
    secrets=[hf_secret],
)
def run_step3_integrity_judgment(seed_label: str = "S1") -> dict[str, Any]:
    """Hash ep8/ep12 weight files; correlate per-item judgment scores s across epochs."""
    import hashlib
    import os

    # Import torch before transformers to avoid intermittent inductor circular import.
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    import torch  # noqa: F401
    import numpy as np
    import pandas as pd

    from grainofsalt.eval_behavior import load_model
    from grainofsalt.i1_eval import score_judgment_set

    if seed_label not in ("S1", "S2", "S3"):
        raise ValueError(seed_label)

    data_root = Path(VOL_MOUNT) / "data"
    people = pd.read_csv(data_root / "step3" / "people.csv")
    facts = pd.read_csv(data_root / "step3" / "facts.csv")
    people = people.copy()
    people["i1_role"] = people["role"].map({"anchor": "judgment", "probe": "heldout"})
    # Match train-time judgment eval: judgment300 = anchor/judgment people
    judg = people[people["i1_role"] == "judgment"].copy()
    pools = json.loads((data_root / "assignments" / "pools.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_root = Path(VOL_MOUNT) / "checkpoints" / f"step3_{seed_label}"
    out_root = Path(VOL_MOUNT) / "results" / "step3" / seed_label
    out_root.mkdir(parents=True, exist_ok=True)

    def _sha_file(p: Path) -> str:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    weight_hashes: dict[str, str] = {}
    scores: dict[str, pd.DataFrame] = {}
    for ep in (8, 12):
        ckpt = ckpt_root / f"epoch-{ep}"
        w = ckpt / "model.safetensors"
        if not w.exists():
            raise FileNotFoundError(w)
        sha = _sha_file(w)
        weight_hashes[f"epoch-{ep}"] = sha
        print(
            f"STEP3 integrity {seed_label} epoch-{ep} "
            f"ckpt={ckpt} model.safetensors sha256={sha}",
            flush=True,
        )
        model, tok = load_model(ckpt, device)
        endo = score_judgment_set(
            model=model,
            tokenizer=tok,
            people=judg,
            facts=facts,
            pools=pools,
            device=device,
        )
        endo["epoch"] = ep
        scores[str(ep)] = endo
        del model
        torch.cuda.empty_cache()
        _commit()

    a = scores["8"].set_index(["person_id", "slot", "value_kind"])
    b = scores["12"].set_index(["person_id", "slot", "value_kind"])
    common = a.index.intersection(b.index)
    corr: dict[str, Any] = {}
    for fmt in ("J1", "J4"):
        s8 = a.loc[common, fmt].to_numpy(dtype=float)
        s12 = b.loc[common, fmt].to_numpy(dtype=float)
        changed = int(np.sum(s8 != s12))
        # pearson
        if len(s8) < 2 or float(np.std(s8)) == 0 or float(np.std(s12)) == 0:
            r = float("nan")
        else:
            r = float(np.corrcoef(s8, s12)[0, 1])
        corr[fmt] = {
            "n_items": int(len(s8)),
            "pearson_r": r,
            "n_changed": changed,
            "frac_changed": float(changed / len(s8)) if len(s8) else float("nan"),
            "mean_abs_delta": float(np.mean(np.abs(s12 - s8))) if len(s8) else float("nan"),
        }
        print(
            f"STEP3 integrity {seed_label} {fmt}: r={r:.6f} "
            f"changed={changed}/{len(s8)}",
            flush=True,
        )

    hashes_differ = weight_hashes["epoch-8"] != weight_hashes["epoch-12"]
    # Primary s = J1 (plan); also keep J4
    ok = bool(hashes_differ and corr["J1"]["n_changed"] > 0 and corr["J1"]["pearson_r"] < 1.0)
    payload = {
        "seed": seed_label,
        "weight_file": "model.safetensors",
        "weight_sha256": weight_hashes,
        "hashes_differ": hashes_differ,
        "ckpt_paths": {
            "epoch-8": str(ckpt_root / "epoch-8"),
            "epoch-12": str(ckpt_root / "epoch-12"),
        },
        "correlation": corr,
        "ok": ok,
        "note": (
            "Eval loads load_model(ckpt_root/epoch-{N}) at on_epoch_end; "
            "weight sha256 is of that directory's model.safetensors."
        ),
    }
    (out_root / "integrity_ep8_ep12.json").write_text(json.dumps(payload, indent=2))
    # Keep per-item scores for audit (compact)
    for ep, df in scores.items():
        df.to_parquet(out_root / f"judgment_scores_epoch-{ep}.parquet", index=False)
    _commit()
    print(f"STEP3 integrity {seed_label} done ok={ok} hashes_differ={hashes_differ}", flush=True)
    return payload


@app.function(
    timeout=900,
    volumes={VOL_MOUNT: volume},
    retries=modal.Retries(initial_delay=0.0, max_retries=0),
)
def wipe_step3_train_outputs() -> dict[str, Any]:
    """Delete Step 3 train outputs for S1–S3; keep corpora + frozen plan data."""
    import shutil

    wiped: list[str] = []
    kept_note = "data/step3 corpora, mix_r1, analysis freeze untouched"
    for s in ("S1", "S2", "S3"):
        for p in (
            Path(VOL_MOUNT) / "results" / "step3" / s,
            Path(VOL_MOUNT) / "metrics" / "step3" / s,
            Path(VOL_MOUNT) / "checkpoints" / f"step3_{s}",
            Path(VOL_MOUNT) / "logs" / f"step3_{s}",
        ):
            if p.exists():
                shutil.rmtree(p)
                wiped.append(str(p))
    shared = Path(VOL_MOUNT) / "results" / "step3"
    for name in (
        "_brief9_cost_ledger.json",
        "COST_CAP_STOP.json",
    ):
        p = shared / name
        if p.exists():
            p.unlink()
            wiped.append(str(p))
    shard_dir = shared / "_cost_shards"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
        wiped.append(str(shard_dir))
    volume.commit()
    return {"wiped": wiped, "kept": kept_note}


@app.local_entrypoint()
def step3_wipe_outputs():
    print(wipe_step3_train_outputs.remote(), flush=True)


@app.local_entrypoint()
def step3_extract(seed: str = "S1", epochs: str = "0,8,12"):
    call = run_step3_extract.spawn(seed, epochs)
    print(
        f"spawned step3_extract seed={seed} epochs={epochs} object_id={call.object_id}",
        flush=True,
    )


@app.local_entrypoint()
def step3_integrity(seed: str = "S1"):
    print(run_step3_integrity_judgment.remote(seed), flush=True)


@app.local_entrypoint()
def step3_train(seed: str = "S1"):
    """Spawn one seed on the *deployed* app path preference: use Function.from_name.

    This entrypoint still uses .remote() (ephemeral). Prefer deploy + spawn.
    """
    print(f"starting step3_train seed={seed}…", flush=True)
    result = run_step3_train.remote(seed)
    print(result, flush=True)


@app.local_entrypoint()
def step3_train_all():
    """Spawn exactly one call per seed; print FunctionCall IDs; do not wait.

    Safe with a deployed app (no local-client wait). Does not respawn.
    """
    ids: dict[str, str] = {}
    for s in ("S1", "S2", "S3"):
        call = run_step3_train.spawn(s)
        ids[s] = call.object_id
        print(f"spawned step3_train seed={s} object_id={call.object_id}", flush=True)
    print(json.dumps({"spawned": ids}, indent=2), flush=True)


# ---------------------------------------------------------------------------
# Brief 10 / R0 — exact Step 2 replication (Run A only)
# ---------------------------------------------------------------------------

R0_SEED = 20261122
R0_STEP2_SEED = 20261118
R0_MIX_CURSOR_START = 100000
R0_STREAM_N = 130000
R0_N_EPOCHS = 8
R0_COST_CAP_USD = 10.0
R0_STEP2_ACTUAL_TRAIN_USD = 6.0  # Step 2 per-run actual; used for pre-flight projection
R0_LOCK_STALE_SEC = STEP3_LOCK_STALE_SEC


def _r0_paths() -> dict[str, Path]:
    out_root = Path(VOL_MOUNT) / "results" / "R0"
    return {
        "out_root": out_root,
        "metrics_root": Path(VOL_MOUNT) / "metrics" / "R0",
        "ckpt_root": Path(VOL_MOUNT) / "checkpoints" / "R0_A",
        "lock_path": out_root / "LOCK.json",
        "status_path": out_root / "status.json",
        "ledger_path": out_root / "cost_ledger.json",
        "cost_stop_path": out_root / "COST_CAP_STOP.json",
    }


def _r0_try_acquire_lock(run_id: str) -> tuple[bool, dict[str, Any]]:
    paths = _r0_paths()
    lock_path = paths["lock_path"]
    volume.reload()
    now = time.time()
    existing: dict[str, Any] = {}
    if lock_path.exists():
        try:
            existing = json.loads(lock_path.read_text())
        except Exception:
            existing = {}
        if step3_lock_should_block(
            existing, run_id, now=now, stale_sec=R0_LOCK_STALE_SEC
        ):
            return False, {
                "reason": "live_lock_exists",
                "existing_run_id": existing.get("run_id"),
                "existing_heartbeat_ts": existing.get("heartbeat_ts"),
                "lock_path": str(lock_path),
            }
    paths["out_root"].mkdir(parents=True, exist_ok=True)
    created = float(existing.get("created_ts") or now)
    if existing.get("run_id") != run_id:
        created = now
    payload = {
        "run": "R0",
        "run_id": run_id,
        "created_ts": created,
        "heartbeat_ts": now,
        "released": False,
        "takeover": bool(existing)
        and (
            existing.get("run_id") == run_id
            or not existing.get("released", True)
        ),
    }
    lock_path.write_text(json.dumps(payload, indent=2))
    volume.commit()
    return True, payload


def _r0_heartbeat_lock(run_id: str) -> None:
    lock_path = _r0_paths()["lock_path"]
    if not lock_path.exists():
        return
    try:
        lock = json.loads(lock_path.read_text())
    except Exception:
        return
    if lock.get("run_id") != run_id:
        return
    lock["heartbeat_ts"] = time.time()
    lock_path.write_text(json.dumps(lock, indent=2))


def _r0_release_lock(run_id: str) -> None:
    lock_path = _r0_paths()["lock_path"]
    if not lock_path.exists():
        return
    try:
        lock = json.loads(lock_path.read_text())
    except Exception:
        lock = {"run_id": run_id}
    if lock.get("run_id") != run_id:
        return
    lock["released"] = True
    lock["released_ts"] = time.time()
    lock_path.write_text(json.dumps(lock, indent=2))


def _r0_write_status(status_path: Path, **kwargs: Any) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ts": time.time(), **kwargs}
    status_path.write_text(json.dumps(payload, indent=2, default=str))


def _r0_update_cost_ledger(*, run_id: str, job_start: float) -> dict[str, Any]:
    paths = _r0_paths()
    stop_path = paths["cost_stop_path"]
    ledger_path = paths["ledger_path"]
    volume.reload()
    if stop_path.exists():
        stop = json.loads(stop_path.read_text())
        return {
            "stop": True,
            "total_est_usd": stop.get("total_est_usd"),
            "stop_detail": stop,
            "stop_usd": R0_COST_CAP_USD,
        }
    elapsed = max(0.0, time.time() - job_start)
    own_usd = elapsed * L4_USD_PER_SEC
    now = time.time()
    ledger = {
        "run_id": run_id,
        "elapsed_sec": elapsed,
        "est_usd": own_usd,
        "stop_usd": R0_COST_CAP_USD,
        "updated_ts": now,
        "stop": False,
        "step2_actual_train_usd": R0_STEP2_ACTUAL_TRAIN_USD,
    }
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(ledger, indent=2))
    if own_usd >= R0_COST_CAP_USD:
        stop = {
            "reason": "r0_cost_cap",
            "total_est_usd": own_usd,
            "stop_usd": R0_COST_CAP_USD,
            "trigger_run_id": run_id,
            "ts": now,
        }
        stop_path.write_text(json.dumps(stop, indent=2))
        ledger["stop"] = True
        ledger["stop_detail"] = stop
        volume.commit()
    return ledger


def _assert_r0_mix_budget(
    *,
    mix_path: Path,
    texts: list[str],
    mix_cursor_start: int,
    n_epochs: int,
    stream_n: int,
    max_length: int,
    tokenizer,
) -> dict[str, Any]:
    from grainofsalt.mix_stream import take_mix_for_tokens
    from grainofsalt.train_r1 import count_synth_tokens

    synth_tok = count_synth_tokens(texts, tokenizer, max_length)
    cur = int(mix_cursor_start)
    for _ in range(int(n_epochs)):
        _texts, cur = take_mix_for_tokens(
            mix_path, cursor=cur, target_tokens=synth_tok
        )
    if cur > stream_n:
        raise AssertionError(
            f"R0 mix budget overrun: start={mix_cursor_start} end_after_{n_epochs}={cur} "
            f"> stream_n={stream_n} (synth_tokens/epoch={synth_tok})"
        )
    return {
        "mix_cursor_start": mix_cursor_start,
        "mix_cursor_end_after_8": cur,
        "synth_tokens_per_epoch": synth_tok,
        "stream_n": stream_n,
        "n_epochs": n_epochs,
    }


@app.function(
    gpu=GPU,
    timeout=86400,
    volumes={VOL_MOUNT: volume},
    retries=modal.Retries(initial_delay=1.0, max_retries=5),
    secrets=[hf_secret],
)
def run_r0_train(
    resume_from_latest: bool = False,
    parent_run_id: str = "",
) -> dict[str, Any]:
    """R0: Step-2 recipe train_r1 on data/R0 Run A; lock + resume/latest like Step 3."""
    import shutil
    import uuid

    import torch

    from grainofsalt.brief5_corpus import assert_no_j4_in_training, assert_no_probe_in_judgment
    from grainofsalt.train_loop import load_model_tokenizer
    from grainofsalt.train_r1 import train_r1

    if resume_from_latest and parent_run_id:
        run_id = f"R0-resume__from__{parent_run_id}"
    else:
        fc_id = ""
        try:
            fc_id = modal.current_function_call_id() or ""
        except Exception:
            fc_id = ""
        short = (fc_id.replace("fc-", "")[-12:] if fc_id else "") or uuid.uuid4().hex[:12]
        run_id = f"R0-{short}"
        if parent_run_id:
            run_id = f"{run_id}__from__{parent_run_id}"

    ok_lock, lock_info = _r0_try_acquire_lock(run_id)
    if not ok_lock:
        print(
            f"R0 DUPLICATE GUARD: live lock exists ({lock_info}); exiting without writes",
            flush=True,
        )
        return {
            "halted": True,
            "reason": "duplicate_guard_live_lock",
            "lock": lock_info,
            "run_id": run_id,
            "wrote": False,
        }

    cfg = _load_cfg()
    # Config override only: data-order / shuffle seed (nothing else changes).
    cfg["seed"] = R0_SEED
    assert int(cfg["seed"]) != R0_STEP2_SEED
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    data_root = Path(VOL_MOUNT) / "data"
    paths = _r0_paths()
    out_root = paths["out_root"]
    out_root.mkdir(parents=True, exist_ok=True)
    status_path = paths["status_path"]
    ckpt_root = paths["ckpt_root"]
    mix_cursor_start = R0_MIX_CURSOR_START
    job_start = time.time()
    sx: dict[str, Any] = {
        "run_id": run_id,
        "seed": R0_SEED,
        "step2_seed": R0_STEP2_SEED,
        "seed_differs_from_step2": True,
    }
    if parent_run_id:
        sx["parent_run_id"] = parent_run_id
    sx["resume_from_latest"] = bool(resume_from_latest)
    preemption_events: list[dict[str, Any]] = []

    def commit():
        _r0_heartbeat_lock(run_id)
        _commit()

    def cost_guard(step: int | None = None) -> None:
        ledger = _r0_update_cost_ledger(run_id=run_id, job_start=job_start)
        if ledger.get("stop"):
            raise RuntimeError(
                f"R0 cost guard: est=${float(ledger.get('est_usd') or 0):.2f} "
                f">= ${R0_COST_CAP_USD}"
            )
        if step is None or int(step) % 50 == 0:
            _r0_heartbeat_lock(run_id)
            _commit()

    try:
        resume_dir = ckpt_root / "resume"
        resume_latest = resume_dir / "latest"
        train_state_path = resume_latest / "train_state.pt"
        if (not resume_from_latest) and train_state_path.exists():
            print(
                f"R0: found {train_state_path}; promoting to resume_from_latest (infra retry)",
                flush=True,
            )
            resume_from_latest = True
            sx["resume_from_latest"] = True
            sx["infra_retry_resume"] = True
            preemption_events.append(
                {
                    "event": "infra_retry_auto_resume",
                    "train_state": str(train_state_path),
                    "ts": time.time(),
                }
            )
        if resume_from_latest:
            if not train_state_path.exists():
                raise FileNotFoundError(
                    f"resume_from_latest but missing {train_state_path}"
                )
            meta_r = json.loads((resume_latest / "meta.json").read_text())
            preemption_events.append(
                {
                    "event": "resume_from_latest",
                    "resume_epoch": meta_r.get("epoch"),
                    "resume_global_step": meta_r.get("global_step"),
                    "mix_cursor": meta_r.get("mix_cursor"),
                    "ts": time.time(),
                }
            )
            print(
                f"R0 PREEMPTION/RESUME: step={meta_r.get('global_step')} "
                f"epoch={meta_r.get('epoch')} mix_cursor={meta_r.get('mix_cursor')}",
                flush=True,
            )
        elif resume_dir.exists():
            shutil.rmtree(resume_dir)

        _r0_write_status(
            status_path,
            stage="preflight",
            pass_=True,
            gpu=gpu_name,
            mix_cursor_start=mix_cursor_start,
            lock=lock_info,
            preemption_events=preemption_events,
            **sx,
        )
        commit()
        cost_guard()

        # Pre-flight cost projection from Step 2 actual (~$6), not a timing dry-run.
        projected = {
            "step2_actual_train_usd": R0_STEP2_ACTUAL_TRAIN_USD,
            "factcond_ep0_ep8_usd_est": 1.25,
            "recall_usd_est": 0.75,
            "projected_total_usd": R0_STEP2_ACTUAL_TRAIN_USD + 1.25 + 0.75,
            "cap_usd": R0_COST_CAP_USD,
        }
        if projected["projected_total_usd"] > R0_COST_CAP_USD:
            raise RuntimeError(
                f"R0 pre-flight cost projection ${projected['projected_total_usd']:.2f} "
                f"> ${R0_COST_CAP_USD}"
            )
        (out_root / "preflight_cost.json").write_text(json.dumps(projected, indent=2))

        corpus = data_root / "R0" / "corpus" / "run_A.jsonl"
        if not corpus.exists():
            raise FileNotFoundError(corpus)
        docs: list[dict[str, Any]] = []
        syn: list[str] = []
        with corpus.open() as f:
            for line in f:
                d = json.loads(line)
                docs.append(d)
                syn.append(d["text"])
        people_path = data_root / "R0" / "people.csv"
        import pandas as pd

        people = pd.read_csv(people_path)
        probe_ids = set(people.loc[people["role"] == "probe", "person_id"])
        assert_no_j4_in_training(docs)
        assert_no_probe_in_judgment(docs, probe_ids)
        n_fact = sum(1 for d in docs if d.get("doc_kind") == "fact")
        n_judg = sum(1 for d in docs if d.get("doc_kind") == "judgment")
        assert n_fact == 10000, n_fact
        assert n_judg == 2400, n_judg

        mix_path = Path(cfg["paths"]["mix_r1_root"]) / "stream.jsonl"
        if not mix_path.exists():
            raise FileNotFoundError(mix_path)
        mix_meta = json.loads(
            (Path(cfg["paths"]["mix_r1_root"]) / "meta.json").read_text()
        )
        stream_n = int(mix_meta.get("stream_n", R0_STREAM_N))
        _m, tok_budget = load_model_tokenizer(
            cfg["model"]["hf_id"],
            cfg["model"]["revision"],
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
        del _m
        mix_budget = _assert_r0_mix_budget(
            mix_path=mix_path,
            texts=syn,
            mix_cursor_start=mix_cursor_start,
            n_epochs=R0_N_EPOCHS,
            stream_n=stream_n,
            max_length=int(cfg["data"]["max_length"]),
            tokenizer=tok_budget,
        )
        del tok_budget
        (out_root / "mix_budget.json").write_text(json.dumps(mix_budget, indent=2))

        _r0_write_status(
            status_path,
            stage="preflight_ok",
            pass_=True,
            n_docs=len(docs),
            mix_budget=mix_budget,
            projected_cost=projected,
            preemption_events=preemption_events,
            **sx,
        )
        commit()
        print(
            f"R0 preflight_ok run_id={run_id} seed={R0_SEED} "
            f"mix_budget={mix_budget} projected={projected}",
            flush=True,
        )

        cfg["paths"]["ckpt_root"] = f"{VOL_MOUNT}/checkpoints"
        cfg["paths"]["logs_root"] = f"{VOL_MOUNT}/logs"
        eval_epochs = [2, 4, 8]
        print(
            f"R0 TRAIN n={len(syn)} lr=1e-4 E={R0_N_EPOCHS} eval={eval_epochs} "
            f"seed={cfg['seed']} mix_cursor_start={mix_cursor_start} GPU={gpu_name}",
            flush=True,
        )

        def on_step(step: int) -> None:
            cost_guard(step)

        result = train_r1(
            synthetic_texts=syn,
            mix_stream_path=mix_path,
            config=cfg,
            run_name="R0_A",
            lr=1e-4,
            n_epochs=R0_N_EPOCHS,
            eval_epochs=eval_epochs,
            device=device,
            volume_commit=commit,
            mix_cursor_start=mix_cursor_start,
            on_optimizer_step=on_step,
        )
        result["gpu"] = gpu_name
        result["seed"] = R0_SEED
        result["step2_seed"] = R0_STEP2_SEED
        result["seed_differs_from_step2"] = True
        result["run_id"] = run_id
        result["mix_cursor_start"] = mix_cursor_start
        result["preemption_events"] = preemption_events
        result["lock"] = lock_info
        result["elapsed_sec"] = time.time() - job_start
        result["est_usd"] = result["elapsed_sec"] * L4_USD_PER_SEC
        (out_root / "train.json").write_text(json.dumps(result, indent=2, default=str))
        _r0_write_status(
            status_path,
            stage="train_done",
            pass_=True,
            result={
                k: result.get(k)
                for k in (
                    "mix_cursor_end",
                    "n_epochs",
                    "seed",
                    "est_usd",
                    "preemption_events",
                )
            },
            **sx,
        )
        _r0_release_lock(run_id)
        commit()
        print(
            f"R0 TRAIN done mix_cursor_end={result.get('mix_cursor_end')} "
            f"est_usd={result.get('est_usd'):.2f} preemption={preemption_events}",
            flush=True,
        )
        return result
    except Exception as e:
        _r0_write_status(
            status_path,
            stage="halted",
            pass_=False,
            error=str(e),
            preemption_events=preemption_events,
            **sx,
        )
        _r0_release_lock(run_id)
        commit()
        raise


@app.function(
    gpu=GPU,
    timeout=21600,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def run_r0_factcond(model_key: str) -> dict[str, Any]:
    """Factcond + strength for R0 checkpoints (paths remapped from Step 2)."""
    import pandas as pd
    import torch

    from grainofsalt.brief4_factcond import (
        compute_strength_features,
        extract_factcond_model,
        sanity_factcond,
    )
    from grainofsalt.eval_behavior import load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    data_root = Path(VOL_MOUNT) / "data"
    people = pd.read_csv(data_root / "R0" / "people.csv")
    facts = pd.read_csv(data_root / "R0" / "facts.csv")
    fact_cols = ["person_id", "birthplace", "employer", "university", "hobby"]
    people = people.merge(facts[fact_cols], on="person_id")
    people["framing_A"] = "ASSERTED"
    people["framing_B"] = "ASSERTED"

    if model_key == "base_epoch0":
        ckpt = Path(VOL_MOUNT) / "checkpoints" / "R0_A" / "epoch-0"
    elif model_key.startswith("A_"):
        ckpt = (
            Path(VOL_MOUNT)
            / "checkpoints"
            / "R0_A"
            / f"epoch-{model_key.split('epoch')[-1]}"
        )
    else:
        raise ValueError(model_key)

    print(f"R0 factcond {model_key} from {ckpt} on {gpu_name}", flush=True)
    model, tok = load_model(ckpt, device)
    act_root = Path(VOL_MOUNT) / "activations_factcond_R0"
    manifest = extract_factcond_model(
        model=model,
        tokenizer=tok,
        people=people,
        data_root=data_root,
        out_dir=act_root,
        model_id=model_key,
        device=device,
        volume_commit=_commit,
    )
    sanity = sanity_factcond(
        model=model,
        tokenizer=tok,
        people=people,
        data_root=data_root,
        device=device,
    )
    strength_path = (
        Path(VOL_MOUNT) / "results" / "R0" / "strength" / f"{model_key}.parquet"
    )
    strength_path.parent.mkdir(parents=True, exist_ok=True)
    compute_strength_features(
        model=model,
        tokenizer=tok,
        people=people,
        data_root=data_root,
        label_col="framing_A",
        device=device,
        out_parquet=strength_path,
        volume_commit=_commit,
    )
    out = {
        "model_key": model_key,
        "manifest": manifest,
        "sanity": sanity,
        "strength": str(strength_path),
        "gpu": gpu_name,
    }
    (Path(VOL_MOUNT) / "results" / "R0" / f"extract_{model_key}.json").write_text(
        json.dumps(out, indent=2, default=str)
    )
    _commit()
    print(f"R0 factcond done {model_key} sanity={sanity}", flush=True)
    return out


@app.function(
    gpu=GPU,
    timeout=21600,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def run_r0_eval(epoch: int) -> dict[str, Any]:
    """Fact-level recall (+ framing breakdown) for one R0 checkpoint."""
    import torch

    from grainofsalt.brief5_eval import evaluate_step2_checkpoint

    cfg = _load_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    data_root = Path(VOL_MOUNT) / "data"
    ckpt = Path(VOL_MOUNT) / "checkpoints" / "R0_A" / f"epoch-{epoch}"
    base = Path(VOL_MOUNT) / "checkpoints" / "R0_A" / "epoch-0"
    out = Path(VOL_MOUNT) / "metrics" / "R0" / f"epoch-{epoch}"
    print(f"R0 EVAL ep={epoch} GPU={gpu_name}", flush=True)

    def _commit_stage(label: str) -> None:
        print(f"R0 EVAL commit after {label}", flush=True)
        _commit()

    result = evaluate_step2_checkpoint(
        ckpt_path=ckpt,
        base_ckpt_path=base,
        run="A",
        data_root=data_root,
        out_dir=out,
        config=cfg,
        device=device,
        do_endorsement=(epoch in (0, 8)),
        do_ppl=(epoch in (0, 8)),
        volume_commit=_commit_stage,
        tables_subdir="R0",
    )
    result["gpu"] = gpu_name
    summary = Path(VOL_MOUNT) / "results" / "R0" / f"eval_epoch-{epoch}.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(result, indent=2, default=str))
    _commit()
    return result


@app.local_entrypoint()
def upload_r0_data():
    """Upload R0 tables + corpus to the volume."""
    root = LOCAL_ROOT
    files = [
        "R0/names/names_final.csv",
        "R0/names/names_meta.json",
        "R0/people.csv",
        "R0/facts.csv",
        "R0/fact_framings.csv",
        "R0/titles.json",
        "R0/singular_disclaimers.json",
        "R0/corpus/run_A.jsonl",
        "R0/corpus/run_A_train_texts.jsonl",
        "R0/corpus/length_A.json",
        "R0/build_meta.json",
        "R0/assignment_verify.json",
        "R0/leak_check.json",
        "assignments/pools.json",
        "prompts/recall.jsonl",
        "prompts/region_recall.jsonl",
        "templates/templates.json",
    ]
    payload: dict[str, bytes] = {}
    for rel in files:
        path = root / "data" / rel
        if path.exists():
            payload[rel] = path.read_bytes()
    print(f"uploading {len(payload)} R0 files…", flush=True)
    out = sync_data_from_bytes.remote(payload)
    print(out)


@app.local_entrypoint()
def r0_train():
    """Spawn R0 train on deployed app preference; this entrypoint uses .remote()."""
    print("starting r0_train…", flush=True)
    result = run_r0_train.remote()
    print(
        {k: result.get(k) for k in (
            "mix_cursor_end", "n_epochs", "seed", "est_usd", "preemption_events", "gpu"
        )},
        flush=True,
    )


@app.local_entrypoint()
def r0_factcond(model_key: str = "base_epoch0"):
    call = run_r0_factcond.spawn(model_key)
    print(f"spawned r0 factcond {model_key} id={call.object_id}", flush=True)


@app.local_entrypoint()
def r0_eval(epoch: int = 8):
    call = run_r0_eval.spawn(epoch)
    print(f"spawned r0 eval epoch={epoch} id={call.object_id}", flush=True)


# ---------------------------------------------------------------------------
# Brief 11 — R2 / R1 dual-loss ablations
# ---------------------------------------------------------------------------

R1R2_SEED = 20261122
R1R2_COST_CAP_USD = 16.0
R1R2_R0_CORPUS_SHA256 = "35b3314401cba2d424d49acc17b4a03401a98b8ce878d4dcbd1ea0a8001033e6"
R2_MIX_CURSOR_START = 112299  # immediately after R0
# R1 start filled after R2 mix-budget preflight (or fixed gap)
R1R2_N_EPOCHS = 8


def _r1r2_paths(arm: str, *, ledger_ns: str = "R1R2") -> dict[str, Path]:
    out_root = Path(VOL_MOUNT) / "results" / arm
    led = Path(VOL_MOUNT) / "results" / ledger_ns
    return {
        "out_root": out_root,
        "metrics_root": Path(VOL_MOUNT) / "metrics" / arm,
        "ckpt_root": Path(VOL_MOUNT) / "checkpoints" / f"{arm}_A",
        "lock_path": out_root / "LOCK.json",
        "status_path": out_root / "status.json",
        "ledger_path": led / "cost_ledger.json",
        "cost_stop_path": led / "COST_CAP_STOP.json",
        "shard_dir": led / "_cost_shards",
        "ledger_ns": ledger_ns,
    }


def _r1r2_try_acquire_lock(arm: str, run_id: str) -> tuple[bool, dict[str, Any]]:
    paths = _r1r2_paths(arm)
    lock_path = paths["lock_path"]
    volume.reload()
    now = time.time()
    existing: dict[str, Any] = {}
    if lock_path.exists():
        try:
            existing = json.loads(lock_path.read_text())
        except Exception:
            existing = {}
        if step3_lock_should_block(existing, run_id, now=now, stale_sec=STEP3_LOCK_STALE_SEC):
            return False, {
                "reason": "live_lock_exists",
                "existing_run_id": existing.get("run_id"),
                "existing_heartbeat_ts": existing.get("heartbeat_ts"),
            }
    paths["out_root"].mkdir(parents=True, exist_ok=True)
    created = float(existing.get("created_ts") or now)
    if existing.get("run_id") != run_id:
        created = now
    payload = {
        "arm": arm,
        "run_id": run_id,
        "created_ts": created,
        "heartbeat_ts": now,
        "released": False,
        "takeover": bool(existing)
        and (existing.get("run_id") == run_id or not existing.get("released", True)),
    }
    lock_path.write_text(json.dumps(payload, indent=2))
    volume.commit()
    return True, payload


def _r1r2_heartbeat_lock(arm: str, run_id: str) -> None:
    lock_path = _r1r2_paths(arm)["lock_path"]
    if not lock_path.exists():
        return
    try:
        lock = json.loads(lock_path.read_text())
    except Exception:
        return
    if lock.get("run_id") != run_id:
        return
    lock["heartbeat_ts"] = time.time()
    lock_path.write_text(json.dumps(lock, indent=2))


def _r1r2_release_lock(arm: str, run_id: str) -> None:
    lock_path = _r1r2_paths(arm)["lock_path"]
    if not lock_path.exists():
        return
    try:
        lock = json.loads(lock_path.read_text())
    except Exception:
        lock = {"run_id": run_id}
    if lock.get("run_id") != run_id:
        return
    lock["released"] = True
    lock["released_ts"] = time.time()
    lock_path.write_text(json.dumps(lock, indent=2))


def _r1r2_write_status(status_path: Path, **kwargs: Any) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps({"ts": time.time(), **kwargs}, indent=2, default=str))


def _r1r2_update_cost_ledger(
    *,
    arm: str,
    run_id: str,
    job_start: float,
    cost_cap_usd: float = R1R2_COST_CAP_USD,
    ledger_ns: str = "R1R2",
    prior_elapsed_sec: float = 0.0,
) -> dict[str, Any]:
    """Update shared cost ledger.

    Counts *all* attempts (including cancelled/finished): each shard keeps its
    peak elapsed/est_usd. Live heartbeats still refresh the active shard.
    Cap uses cumulative_est_usd (sum of every shard), not only fresh heartbeats.
    """
    paths = _r1r2_paths(arm, ledger_ns=ledger_ns)
    stop_path = paths["cost_stop_path"]
    ledger_path = paths["ledger_path"]
    shard_dir = paths["shard_dir"]
    try:
        volume.reload()
    except RuntimeError as e:
        # Open mmap/handles can block reload mid-analysis; keep going with local view.
        if "open files preventing" not in str(e):
            raise
        print(f"ledger reload skipped (open files): {e}", flush=True)
    if stop_path.exists():
        stop = json.loads(stop_path.read_text())
        return {"stop": True, "stop_detail": stop, "stop_usd": cost_cap_usd}
    elapsed = max(0.0, time.time() - job_start) + max(0.0, float(prior_elapsed_sec))
    own_usd = elapsed * L4_USD_PER_SEC
    now = time.time()
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"{run_id}.json"
    prev: dict[str, Any] = {}
    if shard_path.exists():
        try:
            prev = json.loads(shard_path.read_text())
        except Exception:
            prev = {}
    # Never shrink a shard: cancelled/preempted attempts still count toward cap.
    elapsed = max(elapsed, float(prev.get("elapsed_sec") or 0.0))
    own_usd = max(own_usd, float(prev.get("est_usd") or 0.0), elapsed * L4_USD_PER_SEC)
    shard_path.write_text(
        json.dumps(
            {
                "arm": arm,
                "run_id": run_id,
                "elapsed_sec": elapsed,
                "est_usd": own_usd,
                "prior_elapsed_sec": float(prior_elapsed_sec),
                "updated_ts": now,
                "attempts": int(prev.get("attempts") or 1),
            },
            indent=2,
        )
    )
    runs: dict[str, Any] = {}
    live_runs: dict[str, Any] = {}
    for fp in shard_dir.glob("*.json"):
        try:
            info = json.loads(fp.read_text())
        except Exception:
            continue
        rid = str(info.get("run_id") or fp.stem)
        # Keep peak if multiple writes raced.
        if rid in runs and float(info.get("est_usd") or 0) < float(runs[rid].get("est_usd") or 0):
            continue
        runs[rid] = info
        if now - float(info.get("updated_ts", 0)) <= STEP3_LOCK_STALE_SEC:
            live_runs[rid] = info
    cumulative = sum(float(v.get("est_usd", 0.0)) for v in runs.values())
    live = sum(float(v.get("est_usd", 0.0)) for v in live_runs.values())
    # Brief 13: Volume file wins so live cap raises stick across in-flight heartbeats.
    if ledger_ns == "confirm":
        cap_file = paths["ledger_path"].parent / "COST_CAP_USD.txt"
        if cap_file.exists():
            try:
                cost_cap_usd = float(cap_file.read_text().strip())
            except Exception:
                pass
    ledger = {
        "live_est_usd": live,
        "cumulative_est_usd": cumulative,
        "total_est_usd": cumulative,
        "stop_usd": cost_cap_usd,
        "runs": runs,
        "live_runs": sorted(live_runs.keys()),
        "updated_ts": now,
        "stop": False,
        "ledger_ns": ledger_ns,
    }
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(ledger, indent=2))
    if cumulative >= cost_cap_usd:
        stop = {
            "reason": f"{ledger_ns}_cost_cap",
            "total_est_usd": cumulative,
            "stop_usd": cost_cap_usd,
            "ts": now,
        }
        stop_path.write_text(json.dumps(stop, indent=2))
        ledger["stop"] = True
        ledger["stop_detail"] = stop
        volume.commit()
    return ledger

def _run_r1r2_train_impl(
    *,
    arm: str,
    corpus_rel: str,
    mix_cursor_start: int,
    expect_r0_hash: bool,
    resume_from_latest: bool = False,
    parent_run_id: str = "",
    cost_cap_usd: float = R1R2_COST_CAP_USD,
    ledger_ns: str = "R1R2",
    tables_subdir: str | None = None,
    train_seed: int | None = None,
) -> dict[str, Any]:
    import hashlib
    import shutil
    import uuid

    import pandas as pd
    import torch

    from grainofsalt.brief5_corpus import assert_no_j4_in_training, assert_no_probe_in_judgment
    from grainofsalt.i1_train import load_i1_docs
    from grainofsalt.r2_encode import (
        attach_prompt_answer,
        verify_r0_judgment_reconstruction,
    )
    from grainofsalt.train_loop import load_model_tokenizer
    from grainofsalt.train_r1 import count_synth_tokens
    from grainofsalt.train_r2 import NonFiniteLossHalt, train_r2
    from grainofsalt.mix_stream import take_mix_for_tokens

    if resume_from_latest and parent_run_id:
        run_id = f"{arm}-resume__from__{parent_run_id}"
    else:
        try:
            fc_id = modal.current_function_call_id() or ""
        except Exception:
            fc_id = ""
        short = (fc_id.replace("fc-", "")[-12:] if fc_id else "") or uuid.uuid4().hex[:12]
        run_id = f"{arm}-{short}"
        if parent_run_id:
            run_id = f"{run_id}__from__{parent_run_id}"

    ok_lock, lock_info = _r1r2_try_acquire_lock(arm, run_id)
    if not ok_lock:
        return {"halted": True, "reason": "duplicate_guard_live_lock", "lock": lock_info, "run_id": run_id}

    cfg = _load_cfg()
    cfg["seed"] = int(train_seed) if train_seed is not None else R1R2_SEED
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    data_root = Path(VOL_MOUNT) / "data"
    paths = _r1r2_paths(arm, ledger_ns=ledger_ns)
    out_root = paths["out_root"]
    out_root.mkdir(parents=True, exist_ok=True)
    status_path = paths["status_path"]
    ckpt_root = paths["ckpt_root"]
    job_start = time.time()
    # Persist prior GPU time across Modal preemption retries (same run_id shard).
    prior_elapsed_sec = 0.0
    paths["shard_dir"].mkdir(parents=True, exist_ok=True)
    existing_shard = paths["shard_dir"] / f"{run_id}.json"
    if existing_shard.exists():
        try:
            prior_elapsed_sec = float(json.loads(existing_shard.read_text()).get("elapsed_sec", 0.0))
        except Exception:
            prior_elapsed_sec = 0.0
    sx: dict[str, Any] = {
        "run_id": run_id,
        "arm": arm,
        "seed": cfg["seed"],
        "ledger_ns": ledger_ns,
    }
    preemption_events: list[dict[str, Any]] = []

    def commit():
        _r1r2_heartbeat_lock(arm, run_id)
        _commit()

    def cost_guard(step: int | None = None) -> None:
        ledger = _r1r2_update_cost_ledger(
            arm=arm,
            run_id=run_id,
            job_start=job_start,
            cost_cap_usd=cost_cap_usd,
            ledger_ns=ledger_ns,
            prior_elapsed_sec=prior_elapsed_sec,
        )
        if ledger.get("stop"):
            raise RuntimeError(f"{arm} cost guard: live>=${cost_cap_usd}")
        if step is None or int(step) % 50 == 0:
            _r1r2_heartbeat_lock(arm, run_id)
            _commit()

    try:
        resume_dir = ckpt_root / "resume"
        resume_latest = resume_dir / "latest"
        train_state_path = resume_latest / "train_state.pt"
        if (not resume_from_latest) and train_state_path.exists():
            resume_from_latest = True
            sx["infra_retry_resume"] = True
            preemption_events.append({"event": "infra_retry_auto_resume", "ts": time.time()})
        if resume_from_latest:
            if not train_state_path.exists():
                raise FileNotFoundError(train_state_path)
            meta_r = json.loads((resume_latest / "meta.json").read_text())
            preemption_events.append(
                {
                    "event": "resume_from_latest",
                    "resume_epoch": meta_r.get("epoch"),
                    "resume_global_step": meta_r.get("global_step"),
                    "ts": time.time(),
                }
            )
            print(f"{arm} PREEMPTION/RESUME ep={meta_r.get('epoch')} step={meta_r.get('global_step')}", flush=True)
        elif resume_dir.exists():
            shutil.rmtree(resume_dir)

        projected = {
            "projected_total_usd": min(cost_cap_usd * 0.9, 12.5),
            "cap_usd": cost_cap_usd,
            "ledger_ns": ledger_ns,
        }
        if projected["projected_total_usd"] > cost_cap_usd:
            raise RuntimeError("preflight cost projection exceeds cap")
        (out_root / "preflight_cost.json").write_text(json.dumps(projected, indent=2))

        corpus = data_root / corpus_rel
        raw = corpus.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if expect_r0_hash and sha != R1R2_R0_CORPUS_SHA256:
            raise AssertionError(f"R2 corpus hash mismatch: {sha} != {R1R2_R0_CORPUS_SHA256}")
        docs = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
        # Prefer explicit tables_subdir (Brief 13 confirm paths); else legacy arm map.
        if tables_subdir:
            tables = tables_subdir
        elif arm in ("R2", "R1b"):
            tables = "R0"
        elif arm == "R1a":
            tables = "R1a"
        elif arm == "R1":
            tables = "R1"
        else:
            tables = arm if arm.startswith("R1") else "R1"
        people = pd.read_csv(data_root / tables / "people.csv")
        name_by = dict(zip(people["person_id"], people["name"]))
        probe_ids = set(people.loc[people["role"] == "probe", "person_id"])
        assert_no_j4_in_training(docs)
        assert_no_probe_in_judgment(docs, probe_ids)

        if arm == "R2":
            recon = verify_r0_judgment_reconstruction(docs, name_by_pid=name_by)
            if not recon["ok"]:
                raise AssertionError(f"R2 reconstruction text fail: {recon}")
            docs_train = attach_prompt_answer(docs, name_by_pid=name_by)
            # one-token check with tokenizer
            _m, tok0 = load_model_tokenizer(
                cfg["model"]["hf_id"], cfg["model"]["revision"],
                device=torch.device("cpu"), dtype=torch.bfloat16,
            )
            del _m
            recon2 = verify_r0_judgment_reconstruction(
                docs, name_by_pid=name_by, tokenizer=tok0
            )
            del tok0
            if not recon2["ok"]:
                raise AssertionError(f"R2 reconstruction token fail: {recon2}")
            (out_root / "reconstruction_test.json").write_text(
                json.dumps({"text": recon, "tokens": recon2, "corpus_sha256": sha}, indent=2)
            )
        else:
            # R1 docs already have prompt/answer on judgment
            docs_train = docs
            for d in docs_train:
                if d.get("doc_kind") == "judgment" and ("prompt" not in d or "answer" not in d):
                    raise AssertionError("R1 judgment missing prompt/answer")

        mix_path = Path(cfg["paths"]["mix_r1_root"]) / "stream.jsonl"
        mix_meta = json.loads((Path(cfg["paths"]["mix_r1_root"]) / "meta.json").read_text())
        stream_n = int(mix_meta.get("stream_n", 0))
        _m, tok_b = load_model_tokenizer(
            cfg["model"]["hf_id"], cfg["model"]["revision"],
            device=torch.device("cpu"), dtype=torch.bfloat16,
        )
        del _m
        synth_tok = count_synth_tokens([d["text"] for d in docs_train], tok_b, int(cfg["data"]["max_length"]))
        cur = int(mix_cursor_start)
        for _ in range(R1R2_N_EPOCHS):
            _t, cur = take_mix_for_tokens(mix_path, cursor=cur, target_tokens=synth_tok)
        if cur > stream_n:
            raise AssertionError(f"{arm} mix overrun end={cur} stream_n={stream_n}")
        mix_budget = {
            "mix_cursor_start": mix_cursor_start,
            "mix_cursor_end_after_8": cur,
            "synth_tokens_per_epoch": synth_tok,
            "stream_n": stream_n,
        }
        del tok_b
        (out_root / "mix_budget.json").write_text(json.dumps(mix_budget, indent=2))
        _r1r2_write_status(
            status_path, stage="preflight_ok", pass_=True, mix_budget=mix_budget,
            corpus_sha256=sha, preemption_events=preemption_events, **sx,
        )
        commit()
        cost_guard()
        print(f"{arm} preflight_ok mix_budget={mix_budget}", flush=True)

        cfg["paths"]["ckpt_root"] = f"{VOL_MOUNT}/checkpoints"
        cfg["paths"]["logs_root"] = f"{VOL_MOUNT}/logs"

        def on_step(step: int) -> None:
            cost_guard(step)

        result = train_r2(
            synthetic_docs=docs_train,
            mix_stream_path=mix_path,
            config=cfg,
            run_name=f"{arm}_A",
            lr=1e-4,
            n_epochs=R1R2_N_EPOCHS,
            eval_epochs=[2, 4, 8],
            device=device,
            volume_commit=commit,
            mix_cursor_start=mix_cursor_start,
            on_optimizer_step=on_step,
        )
        result.update(
            {
                "gpu": gpu_name,
                "arm": arm,
                "run_id": run_id,
                "mix_cursor_start": mix_cursor_start,
                "corpus_sha256": sha,
                "preemption_events": preemption_events,
                "elapsed_sec": time.time() - job_start,
            }
        )
        result["est_usd"] = result["elapsed_sec"] * L4_USD_PER_SEC
        (out_root / "train.json").write_text(json.dumps(result, indent=2, default=str))
        _r1r2_write_status(status_path, stage="train_done", pass_=True, result={
            k: result.get(k) for k in ("mix_cursor_end", "seed", "est_usd", "preemption_events")
        }, **sx)
        _r1r2_release_lock(arm, run_id)
        commit()
        if ledger_ns == "confirm":
            try:
                spawned = _maybe_spawn_confirm_analyze()
                result["confirm_analyze_spawn"] = spawned
            except Exception as e:
                result["confirm_analyze_spawn_error"] = str(e)
                print(f"confirm analyze auto-spawn failed: {e}", flush=True)
        return result
    except NonFiniteLossHalt as e:
        _r1r2_write_status(status_path, stage="halted_nonfinite", pass_=False, error=str(e), **sx)
        _r1r2_release_lock(arm, run_id)
        commit()
        return {"halted": True, "reason": "non_finite_loss", "error": str(e), "run_id": run_id, "retry": False}
    except Exception as e:
        _r1r2_write_status(status_path, stage="halted", pass_=False, error=str(e), **sx)
        _r1r2_release_lock(arm, run_id)
        commit()
        raise


@app.function(
    gpu=GPU,
    timeout=86400,
    volumes={VOL_MOUNT: volume},
    retries=modal.Retries(initial_delay=1.0, max_retries=5),
    secrets=[hf_secret],
)
def run_r2_train(resume_from_latest: bool = False, parent_run_id: str = "") -> dict[str, Any]:
    return _run_r1r2_train_impl(
        arm="R2",
        corpus_rel="R0/corpus/run_A.jsonl",
        mix_cursor_start=R2_MIX_CURSOR_START,
        expect_r0_hash=True,
        resume_from_latest=resume_from_latest,
        parent_run_id=parent_run_id,
    )


@app.function(
    gpu=GPU,
    timeout=86400,
    volumes={VOL_MOUNT: volume},
    retries=modal.Retries(initial_delay=1.0, max_retries=5),
    secrets=[hf_secret],
)
def run_r1_train(
    mix_cursor_start: int = 125000,
    resume_from_latest: bool = False,
    parent_run_id: str = "",
) -> dict[str, Any]:
    return _run_r1r2_train_impl(
        arm="R1",
        corpus_rel="R1/corpus/run_A.jsonl",
        mix_cursor_start=int(mix_cursor_start),
        expect_r0_hash=False,
        resume_from_latest=resume_from_latest,
        parent_run_id=parent_run_id,
    )


BRIEF12_B_COST_CAP_USD = 10.0
BRIEF12_A_COST_CAP_USD = 6.0
BRIEF12_C_COST_CAP_USD = 3.0  # Brief 12C exploratory Part A completion
CONFIRM_COST_CAP_USD = 65.0  # Brief 13 confirmatory hard cap (all attempts; Volume cumulative)
R1A_MIX_CURSOR_START = 134481  # after Brief 11 R1
R1B_MIX_CURSOR_START = 150000  # disjoint from R1a

# Brief 13: 9 disjoint mix ranges (12k docs each), past current stream EOF after append.
# Finalized after mix append; defaults assume stream_n ≥ 310000.
CONFIRM_MIX_STARTS: dict[str, int] = {
    "CORE_K1": 202000,
    "MIS_K1": 214000,
    "NH_K1": 226000,
    "CORE_K2": 238000,
    "MIS_K2": 250000,
    "NH_K2": 262000,
    "CORE_K3": 274000,
    "MIS_K3": 286000,
    "NH_K3": 298000,
}
CONFIRM_SEED_INT: dict[str, int] = {"K1": 20261124, "K2": 20261125, "K3": 20261126}


@app.function(
    gpu=GPU,
    timeout=86400,
    volumes={VOL_MOUNT: volume},
    retries=modal.Retries(initial_delay=1.0, max_retries=5),
    secrets=[hf_secret],
)
def run_confirm_train(
    arm: str,
    seed_name: str,
    mix_cursor_start: int | None = None,
    resume_from_latest: bool = False,
    parent_run_id: str = "",
) -> dict[str, Any]:
    """Brief 13 confirmatory train: arm ∈ {CORE,MIS,NH}, seed_name ∈ {K1,K2,K3}."""
    assert arm in ("CORE", "MIS", "NH"), arm
    assert seed_name in CONFIRM_SEED_INT, seed_name
    run_arm = f"{arm}_{seed_name}"
    start = (
        int(mix_cursor_start)
        if mix_cursor_start is not None
        else int(CONFIRM_MIX_STARTS[run_arm])
    )
    return _run_r1r2_train_impl(
        arm=run_arm,
        corpus_rel=f"confirm/{seed_name}/{arm}/corpus/run_A.jsonl",
        mix_cursor_start=start,
        expect_r0_hash=False,
        resume_from_latest=resume_from_latest,
        parent_run_id=parent_run_id,
        cost_cap_usd=CONFIRM_COST_CAP_USD,
        ledger_ns="confirm",
        tables_subdir=f"confirm/{seed_name}/{arm}",
        train_seed=CONFIRM_SEED_INT[seed_name],
    )


def _confirm_train_arms() -> list[str]:
    return [f"{a}_{s}" for s in ("K1", "K2", "K3") for a in ("CORE", "MIS", "NH")]


def _maybe_spawn_confirm_analyze() -> dict[str, Any]:
    """If all 9 confirm trains are train_done and analyze not started, spawn it.

    Brief 13 previously never auto-triggered analysis after the 9th train because
    train completion only wrote status/train.json — no watcher called analyze.
    """
    volume.reload()
    out = Path(VOL_MOUNT) / "results" / "confirm"
    out.mkdir(parents=True, exist_ok=True)
    flag = out / "ANALYZE_SPAWNED.json"
    if flag.exists():
        return {"spawned": False, "reason": "already_flagged", "flag": str(flag)}
    missing = []
    for arm in _confirm_train_arms():
        st_path = Path(VOL_MOUNT) / "results" / arm / "status.json"
        if not st_path.exists():
            missing.append(arm)
            continue
        try:
            st = json.loads(st_path.read_text())
        except Exception:
            missing.append(arm)
            continue
        if st.get("stage") != "train_done":
            missing.append(arm)
    if missing:
        return {"spawned": False, "reason": "trains_incomplete", "missing": missing}
    # Claim the spawn with a flag before calling (avoid double-spawn races).
    flag.write_text(
        json.dumps({"ts": time.time(), "by": "auto_after_train_done"}, indent=2)
    )
    volume.commit()
    call = run_confirm_analyze.spawn()
    payload = {
        "spawned": True,
        "call_id": call.object_id,
        "ts": time.time(),
        "by": "auto_after_train_done",
    }
    flag.write_text(json.dumps(payload, indent=2))
    (out / "analyze_call.json").write_text(json.dumps(payload, indent=2))
    volume.commit()
    print(f"spawned run_confirm_analyze {call.object_id}", flush=True)
    return payload


@app.function(
    gpu=GPU,
    timeout=86400,
    volumes={VOL_MOUNT: volume},
    retries=modal.Retries(initial_delay=1.0, max_retries=2),
    secrets=[hf_secret],
)
def run_confirm_analyze(smoke: bool = False, resume: bool = True) -> dict[str, Any]:
    """Brief 13 auto-analysis: arm extracts then C1→C2→C4→C3_primary→C3_extras."""
    import time as _time
    import uuid as _uuid

    import torch

    from grainofsalt.confirm_analyze import analysis_phases, run_confirm_analysis

    job_start = _time.time()
    out_root = Path(VOL_MOUNT) / "results" / "confirm"
    out_root.mkdir(parents=True, exist_ok=True)
    # Ensure Volume cap file is 65 for in-flight ledger heartbeats.
    cap_file = out_root / "COST_CAP_USD.txt"
    if (not cap_file.exists()) or float(cap_file.read_text().strip() or 0) < CONFIRM_COST_CAP_USD:
        cap_file.write_text(f"{CONFIRM_COST_CAP_USD:.1f}\n")
        _commit()
    # Clear false cost-stop artifacts from Volume.reload/open-file mishap.
    if resume:
        bogus = out_root / "c3_primary.json"
        if bogus.exists():
            try:
                prev = json.loads(bogus.read_text())
            except Exception:
                prev = {}
            if prev.get("stopped_on_cost") and not (prev.get("seeds_done") or prev.get("cells")):
                for name in (
                    "c3_primary.json",
                    "c3_extras.json",
                    "c3_primary_rows.parquet",
                    "c3_primary_realworld_rows.parquet",
                    "c3_extras_rows.parquet",
                    "c3_extras_realworld_rows.parquet",
                ):
                    p = out_root / name
                    if p.exists():
                        p.unlink()
                        print(f"cleared bogus {name}", flush=True)
                _commit()

    try:
        fc_id = modal.current_function_call_id() or ""
    except Exception:
        fc_id = ""
    short = (fc_id.replace("fc-", "")[-12:] if fc_id else "") or _uuid.uuid4().hex[:12]
    run_id = f"confirm_analyze-{short}"

    prior = 0.0
    shard = out_root / "_cost_shards" / f"{run_id}.json"
    if shard.exists():
        try:
            prior = float(json.loads(shard.read_text()).get("elapsed_sec") or 0.0)
        except Exception:
            prior = 0.0

    def _cost() -> float:
        ledger = _r1r2_update_cost_ledger(
            arm="analyze",
            run_id=run_id,
            job_start=job_start,
            cost_cap_usd=CONFIRM_COST_CAP_USD,
            ledger_ns="confirm",
            prior_elapsed_sec=prior,
        )
        return float(ledger.get("cumulative_est_usd") or 0.0)

    def _guard() -> None:
        ledger = _r1r2_update_cost_ledger(
            arm="analyze",
            run_id=run_id,
            job_start=job_start,
            cost_cap_usd=CONFIRM_COST_CAP_USD,
            ledger_ns="confirm",
            prior_elapsed_sec=prior,
        )
        if ledger.get("stop"):
            raise RuntimeError(f"confirm analyze cost guard: cum>=${CONFIRM_COST_CAP_USD}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vol_root = Path(VOL_MOUNT)
    data_root = vol_root / "data"

    # Phase 0: factcond + eval for each train arm if missing.
    need = []
    for train_arm in _confirm_train_arms():
        act = vol_root / f"activations_factcond_{train_arm}" / "A_epoch8" / "factcond_final.npy"
        endo = vol_root / "metrics" / train_arm / "epoch-8" / "endorsement.parquet"
        if not act.exists() or not endo.exists():
            need.append(train_arm)
    extract_log = []
    for train_arm in need:
        print(f"confirm extract {train_arm}…", flush=True)
        _guard()
        info = run_r1r2_arm_analysis.local(train_arm, skip_existing_factcond=True)
        extract_log.append({"arm": train_arm, "ok": bool(info.get("ok")), "stages": info.get("stages")})
        _commit()
        _guard()
    (out_root / "extract_log.json").write_text(json.dumps(extract_log, indent=2))
    _commit()

    (out_root / "phases_plan.json").write_text(json.dumps(analysis_phases(), indent=2))
    _commit()
    print("confirm_analyze C1→C2→C4→C3_primary→C3_extras…", flush=True)
    summary = run_confirm_analysis(
        vol_root=vol_root,
        data_root=data_root,
        out_root=out_root,
        device=device,
        cost_fn=_cost,
        guard_fn=_guard,
        commit_fn=_commit,
        smoke=smoke,
        resume=resume,
    )
    summary = dict(summary or {})
    summary["run_id"] = run_id
    summary["extract_arms"] = need
    summary["cost_usd"] = _cost()
    summary["cap_usd"] = CONFIRM_COST_CAP_USD
    (out_root / "analyze_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    _commit()
    print(json.dumps({k: summary.get(k) for k in ("ok", "cost_usd", "phases_completed", "stopped_on_cost")}, indent=2), flush=True)
    return summary


@app.function(
    gpu=GPU,
    timeout=86400,
    volumes={VOL_MOUNT: volume},
    retries=modal.Retries(initial_delay=1.0, max_retries=5),
    secrets=[hf_secret],
)
def run_r1a_train(
    mix_cursor_start: int = R1A_MIX_CURSOR_START,
    resume_from_latest: bool = False,
    parent_run_id: str = "",
) -> dict[str, Any]:
    return _run_r1r2_train_impl(
        arm="R1a",
        corpus_rel="R1a/corpus/run_A.jsonl",
        mix_cursor_start=int(mix_cursor_start),
        expect_r0_hash=False,
        resume_from_latest=resume_from_latest,
        parent_run_id=parent_run_id,
        cost_cap_usd=BRIEF12_B_COST_CAP_USD,
        ledger_ns="brief12",
        tables_subdir="R1a",
    )


@app.function(
    gpu=GPU,
    timeout=86400,
    volumes={VOL_MOUNT: volume},
    retries=modal.Retries(initial_delay=1.0, max_retries=5),
    secrets=[hf_secret],
)
def run_r1b_train(
    mix_cursor_start: int = R1B_MIX_CURSOR_START,
    resume_from_latest: bool = False,
    parent_run_id: str = "",
) -> dict[str, Any]:
    return _run_r1r2_train_impl(
        arm="R1b",
        corpus_rel="R1b/corpus/run_A.jsonl",
        mix_cursor_start=int(mix_cursor_start),
        expect_r0_hash=False,
        resume_from_latest=resume_from_latest,
        parent_run_id=parent_run_id,
        cost_cap_usd=BRIEF12_B_COST_CAP_USD,
        ledger_ns="brief12",
        tables_subdir="R0",
    )


def _confirm_tables_subdir(arm: str) -> str | None:
    """Map train arm ``CORE_K1`` → data tables ``confirm/K1/CORE``; else None."""
    for seed in ("K1", "K2", "K3"):
        for a in ("CORE", "MIS", "NH"):
            if arm == f"{a}_{seed}":
                return f"confirm/{seed}/{a}"
    return None


@app.function(
    gpu=GPU,
    timeout=21600,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def run_r1r2_factcond(arm: str, model_key: str) -> dict[str, Any]:
    import pandas as pd
    import torch
    from grainofsalt.brief4_factcond import (
        compute_strength_features,
        extract_factcond_model,
        sanity_factcond,
    )
    from grainofsalt.eval_behavior import load_model

    confirm_tables = _confirm_tables_subdir(arm)
    assert arm in ("R1", "R2", "R1a", "R1b") or confirm_tables, arm
    if confirm_tables:
        data_subdir = confirm_tables
    else:
        data_subdir = {"R2": "R0", "R1b": "R0", "R1": "R1", "R1a": "R1a"}[arm]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(VOL_MOUNT) / "data"
    people = pd.read_csv(data_root / data_subdir / "people.csv")
    facts = pd.read_csv(data_root / data_subdir / "facts.csv")
    fact_cols = ["person_id", "birthplace", "employer", "university", "hobby"]
    people = people.merge(facts[fact_cols], on="person_id")
    people["framing_A"] = "ASSERTED"
    if model_key == "base_epoch0":
        ckpt = Path(VOL_MOUNT) / "checkpoints" / f"{arm}_A" / "epoch-0"
    elif model_key.startswith("A_"):
        ckpt = Path(VOL_MOUNT) / "checkpoints" / f"{arm}_A" / f"epoch-{model_key.split('epoch')[-1]}"
    else:
        raise ValueError(model_key)
    model, tok = load_model(ckpt, device)
    act_root = Path(VOL_MOUNT) / f"activations_factcond_{arm}"
    manifest = extract_factcond_model(
        model=model, tokenizer=tok, people=people, data_root=data_root,
        out_dir=act_root, model_id=model_key, device=device, volume_commit=_commit,
    )
    sanity = sanity_factcond(
        model=model, tokenizer=tok, people=people, data_root=data_root, device=device,
    )
    strength_path = Path(VOL_MOUNT) / "results" / arm / "strength" / f"{model_key}.parquet"
    strength_path.parent.mkdir(parents=True, exist_ok=True)
    compute_strength_features(
        model=model, tokenizer=tok, people=people, data_root=data_root,
        label_col="framing_A", device=device, out_parquet=strength_path, volume_commit=_commit,
    )
    out = {"arm": arm, "model_key": model_key, "manifest": manifest, "sanity": sanity, "strength": str(strength_path)}
    (Path(VOL_MOUNT) / "results" / arm / f"extract_{model_key}.json").write_text(json.dumps(out, indent=2, default=str))
    _commit()
    return out


@app.function(
    gpu=GPU,
    timeout=21600,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def run_r1r2_eval(arm: str, epoch: int = 8) -> dict[str, Any]:
    import torch
    from grainofsalt.brief5_eval import evaluate_step2_checkpoint

    confirm_tables = _confirm_tables_subdir(arm)
    assert arm in ("R1", "R2", "R1a", "R1b") or confirm_tables, arm
    cfg = _load_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(VOL_MOUNT) / "data"
    if confirm_tables:
        tables = confirm_tables
    else:
        tables = {"R2": "R0", "R1b": "R0", "R1": "R1", "R1a": "R1a"}[arm]
    ckpt = Path(VOL_MOUNT) / "checkpoints" / f"{arm}_A" / f"epoch-{epoch}"
    base = Path(VOL_MOUNT) / "checkpoints" / f"{arm}_A" / "epoch-0"
    out = Path(VOL_MOUNT) / "metrics" / arm / f"epoch-{epoch}"

    def _commit_stage(label: str) -> None:
        _commit()

    result = evaluate_step2_checkpoint(
        ckpt_path=ckpt,
        base_ckpt_path=base,
        run="A",
        data_root=data_root,
        out_dir=out,
        config=cfg,
        device=device,
        do_endorsement=(epoch in (0, 8)),
        do_ppl=(epoch in (0, 8)),
        volume_commit=_commit_stage,
        tables_subdir=tables,
    )
    (Path(VOL_MOUNT) / "results" / arm / f"eval_epoch-{epoch}.json").write_text(
        json.dumps(result, indent=2, default=str)
    )
    _commit()
    return result


@app.function(
    gpu=GPU,
    timeout=43200,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def run_r1r2_arm_analysis(arm: str, *, skip_existing_factcond: bool = True) -> dict[str, Any]:
    """One Modal call per arm — factcond+strength @0/@8, recall+J1/J4 @8.

    Endorsement uses confirm distractors when ``tables_subdir`` is confirm/*.
    Does not train. Confirm arms use ``confirm/{seed}/{arm}`` tables.
    Skips factcond keys whose activations already exist when
    ``skip_existing_factcond`` is True.
    """
    confirm_tables = _confirm_tables_subdir(arm)
    assert arm in ("R1", "R2", "R1a", "R1b") or confirm_tables, arm
    print(f"{arm} arm_analysis start", flush=True)
    out: dict[str, Any] = {"arm": arm, "stages": {}}
    for key in ("base_epoch0", "A_epoch8"):
        act = (
            Path(VOL_MOUNT)
            / f"activations_factcond_{arm}"
            / key
            / "factcond_final.npy"
        )
        if skip_existing_factcond and act.exists():
            print(f"{arm} factcond {key} SKIP (exists)", flush=True)
            out["stages"][f"factcond_{key}"] = {"skipped": True, "path": str(act)}
            continue
        print(f"{arm} factcond {key}", flush=True)
        out["stages"][f"factcond_{key}"] = run_r1r2_factcond.local(arm, key)
    print(f"{arm} eval epoch-8 (recall + endorsement)", flush=True)
    out["stages"]["eval_epoch8"] = run_r1r2_eval.local(arm, 8)
    out["ok"] = True
    (Path(VOL_MOUNT) / "results" / arm / "arm_analysis.json").write_text(
        json.dumps(
            {
                "arm": arm,
                "ok": True,
                "stages": list(out["stages"].keys()),
                "detail": {k: (v if isinstance(v, dict) and v.get("skipped") else "ran") for k, v in out["stages"].items()},
            },
            indent=2,
        )
    )
    _commit()
    print(f"{arm} arm_analysis done", flush=True)
    return {"arm": arm, "ok": True, "stages": list(out["stages"].keys())}


@app.function(
    gpu=GPU,
    timeout=10800,  # 3h — one arm extract/eval; avoid mega-call timeouts
    volumes={VOL_MOUNT: volume},
    retries=modal.Retries(initial_delay=1.0, max_retries=2),
    secrets=[hf_secret],
)
def run_confirm_arm_analysis(arm: str) -> dict[str, Any]:
    """Brief 13: single-arm factcond+eval with hard 3h timeout."""
    assert _confirm_tables_subdir(arm), arm
    return run_r1r2_arm_analysis.local(arm, skip_existing_factcond=True)


@app.function(timeout=1800, volumes={VOL_MOUNT: volume})
def list_confirm_artifact_sizes() -> dict[str, Any]:
    """Inventory confirm checkpoints + extract artifacts with byte sizes."""
    root = Path(VOL_MOUNT)
    arms = _confirm_train_arms()

    def _dir_size(p: Path) -> int:
        if not p.exists():
            return 0
        if p.is_file():
            return p.stat().st_size
        total = 0
        for fp in p.rglob("*"):
            if fp.is_file():
                try:
                    total += fp.stat().st_size
                except OSError:
                    pass
        return total

    def _fmt(n: int) -> str:
        for unit, div in (("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)):
            if n >= div:
                return f"{n / div:.2f} {unit}"
        return f"{n} B"

    rows = []
    for arm in arms:
        ckpt8 = root / "checkpoints" / f"{arm}_A" / "epoch-8"
        ckpt0 = root / "checkpoints" / f"{arm}_A" / "epoch-0"
        act8 = root / f"activations_factcond_{arm}" / "A_epoch8"
        act0 = root / f"activations_factcond_{arm}" / "base_epoch0"
        endo = root / "metrics" / arm / "epoch-8" / "endorsement.parquet"
        aa = root / "results" / arm / "arm_analysis.json"
        row = {
            "arm": arm,
            "ckpt_epoch8_bytes": _dir_size(ckpt8),
            "ckpt_epoch0_bytes": _dir_size(ckpt0),
            "act_A_epoch8_bytes": _dir_size(act8),
            "act_base_epoch0_bytes": _dir_size(act0),
            "endorsement_bytes": _dir_size(endo),
            "arm_analysis": aa.exists(),
            "extract_complete": (
                (act8 / "factcond_final.npy").exists()
                and (act0 / "factcond_final.npy").exists()
                and endo.exists()
                and aa.exists()
            ),
        }
        for k in list(row):
            if k.endswith("_bytes"):
                row[k.replace("_bytes", "_human")] = _fmt(int(row[k]))
        rows.append(row)
    payload = {
        "arms": rows,
        "n_extract_complete": sum(1 for r in rows if r["extract_complete"]),
        "n_ckpt_ok": sum(1 for r in rows if r["ckpt_epoch8_bytes"] > 0 and r["ckpt_epoch0_bytes"] > 0),
    }
    out = root / "results" / "confirm" / "artifact_sizes.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    _commit()
    return payload


@app.local_entrypoint()
def upload_r1_data():
    root = LOCAL_ROOT
    files = [
        "R1/people.csv", "R1/facts.csv", "R1/fact_framings.csv",
        "R1/names/names_final.csv", "R1/corpus/run_A.jsonl",
        "R1/build_meta.json", "R1/leak_check.json", "R1/assignment_balance.json",
        "R0/distractors.csv", "R0/corpus/run_A.jsonl", "R0/people.csv", "R0/facts.csv",
        "R0/fact_framings.csv", "assignments/pools.json", "prompts/recall.jsonl",
        "prompts/region_recall.jsonl", "templates/templates.json",
    ]
    payload = {}
    for rel in files:
        path = root / "data" / rel
        if path.exists():
            payload[rel] = path.read_bytes()
    print(f"uploading {len(payload)} files…", flush=True)
    print(sync_data_from_bytes.remote(payload))


@app.function(
    gpu=GPU,
    timeout=7200,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def run_r0_probe_endorsement(epoch: int = 8) -> dict[str, Any]:
    """Brief 11: score R0 probe J1/J4 endorsement at epoch-8 with shared distractors."""
    import pandas as pd
    import torch

    from grainofsalt.brief5_analysis import (
        instrument_validity,
        neglect_analysis,
        score_probe_endorsement,
    )
    from grainofsalt.eval_behavior import load_eval_assets, load_model
    from grainofsalt.probe_distractors import load_probe_distractors

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(VOL_MOUNT) / "data"
    dist_path = data_root / "R1R2" / "probe_distractors.csv"
    if not dist_path.exists():
        raise FileNotFoundError(dist_path)
    dist_map = load_probe_distractors(dist_path)
    people = pd.read_csv(data_root / "R0" / "people.csv")
    facts = pd.read_csv(data_root / "R0" / "facts.csv")
    ff = pd.read_csv(data_root / "R0" / "fact_framings.csv")
    assets = load_eval_assets(data_root)
    ckpt = Path(VOL_MOUNT) / "checkpoints" / "R0_A" / f"epoch-{epoch}"
    model, tok = load_model(ckpt, device)
    endo = score_probe_endorsement(
        model=model,
        tokenizer=tok,
        people=people,
        facts=facts,
        fact_framings=ff,
        pools=assets["pools"],
        run="A",
        device=device,
        distractor_by_pid_slot=dist_map,
    )
    out_dir = Path(VOL_MOUNT) / "results" / "R0" / f"endorsement_epoch-{epoch}"
    out_dir.mkdir(parents=True, exist_ok=True)
    endo.to_parquet(out_dir / "endorsement.parquet", index=False)
    inst = instrument_validity(endo)
    neg = {"J1": neglect_analysis(endo, fmt="J1"), "J4": neglect_analysis(endo, fmt="J4")}
    # mean d per framing (exploratory)
    wide = endo.pivot_table(
        index=["person_id", "slot", "framing"],
        columns="value_kind",
        values="J1",
        aggfunc="first",
    ).reset_index()
    wide["d"] = wide["gold"] - wide["distractor"]
    by_fr = (
        wide.groupby("framing")["d"]
        .agg(["mean", "count"])
        .reset_index()
        .to_dict(orient="records")
    )
    payload = {
        "epoch": epoch,
        "n_rows": int(len(endo)),
        "n_distractors": len(dist_map),
        "instrument": inst,
        "neglect": neg,
        "mean_d_J1_by_framing": by_fr,
        "distractor_path": str(dist_path),
        "ckpt": str(ckpt),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=str))
    (Path(VOL_MOUNT) / "results" / "R0" / f"endorsement_epoch-{epoch}.json").write_text(
        json.dumps(payload, indent=2, default=str)
    )
    _commit()
    print(json.dumps({k: payload[k] for k in ("epoch", "n_rows", "instrument")}, indent=2), flush=True)
    return payload


@app.local_entrypoint()
def upload_r1r2_distractors():
    root = LOCAL_ROOT
    payload = {}
    for rel in (
        "R1R2/probe_distractors.csv",
        "R0/people.csv",
        "R0/facts.csv",
        "R0/fact_framings.csv",
        "assignments/pools.json",
    ):
        path = root / "data" / rel
        if path.exists():
            payload[rel] = path.read_bytes()
    print(sync_data_from_bytes.remote(payload))


@app.local_entrypoint()
def upload_brief12_data():
    """Upload R1a/R1b corpora + shared tables for Brief 12."""
    root = LOCAL_ROOT
    files = []
    for arm in ("R1a", "R1b"):
        for rel in (
            f"{arm}/people.csv",
            f"{arm}/facts.csv",
            f"{arm}/fact_framings.csv",
            f"{arm}/corpus/run_A.jsonl",
            f"{arm}/build_meta.json",
            f"{arm}/acceptance.json",
            f"{arm}/names/names_final.csv",
        ):
            files.append(rel)
    files += [
        "R0/people.csv",
        "R0/facts.csv",
        "R0/fact_framings.csv",
        "R0/distractors.csv",
        "R1R2/probe_distractors.csv",
        "assignments/pools.json",
        "prompts/recall.jsonl",
        "templates/templates.json",
    ]
    payload = {}
    for rel in files:
        path = root / "data" / rel
        if path.exists():
            payload[rel] = path.read_bytes()
    print(f"uploading {len(payload)} files…", flush=True)
    print(sync_data_from_bytes.remote(payload))


@app.local_entrypoint()
def upload_confirm_data():
    """Upload Brief 13 confirm shared tables + 9 arm corpora (K1–K3 × CORE/MIS/NH)."""
    root = LOCAL_ROOT / "data"
    files: list[str] = [
        "confirm/people.csv",
        "confirm/facts.csv",
        "confirm/probe_distractors.csv",
        "confirm/shared_meta.json",
        "confirm/build_summary.json",
        "confirm/names/names_final.csv",
        "assignments/pools.json",
        "prompts/recall.jsonl",
        "templates/templates.json",
    ]
    for seed in ("K1", "K2", "K3"):
        for arm in ("CORE", "MIS", "NH"):
            base = f"confirm/{seed}/{arm}"
            files += [
                f"{base}/people.csv",
                f"{base}/facts.csv",
                f"{base}/fact_framings.csv",
                f"{base}/names_final.csv",
                f"{base}/acceptance.json",
                f"{base}/corpus/run_A.jsonl",
            ]
    payload: dict[str, bytes] = {}
    missing: list[str] = []
    for rel in files:
        path = root / rel
        if path.exists():
            payload[rel] = path.read_bytes()
        else:
            missing.append(rel)
    if missing:
        raise FileNotFoundError(f"upload_confirm_data missing: {missing[:8]}")
    print(f"uploading {len(payload)} confirm files…", flush=True)
    print(sync_data_from_bytes.remote(payload))


@app.function(
    gpu=GPU,
    timeout=43200,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def run_brief12_part_a(smoke: bool = False) -> dict[str, Any]:
    """Brief 12 Part A on R2 ep8: logits SVD + dormancy + Claim-3 intervention (cap $6).

    Uses R0 ``fact_framings`` + ``framing_A`` (Run A). Pass ``smoke=True`` only for
    CPU smoke tests (tiny probe subset); production GPU runs use defaults.
    """
    import time as _time
    import uuid as _uuid

    import torch

    from grainofsalt.brief12_part_a import run_part_a

    job_start = _time.time()
    out_root = Path(VOL_MOUNT) / "results" / "brief12"
    out_root.mkdir(parents=True, exist_ok=True)
    data_root = Path(VOL_MOUNT) / "data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        fc_id = modal.current_function_call_id() or ""
    except Exception:
        fc_id = ""
    short = (fc_id.replace("fc-", "")[-12:] if fc_id else "") or _uuid.uuid4().hex[:12]
    part_a_run_id = f"PartA-{short}"
    prior_a = 0.0
    a_shard = Path(VOL_MOUNT) / "results" / "brief12_A" / "_cost_shards" / f"{part_a_run_id}.json"
    if a_shard.exists():
        try:
            prior_a = float(json.loads(a_shard.read_text()).get("elapsed_sec") or 0.0)
        except Exception:
            prior_a = 0.0

    def _cost() -> float:
        ledger = _r1r2_update_cost_ledger(
            arm="PartA",
            run_id=part_a_run_id,
            job_start=job_start,
            cost_cap_usd=BRIEF12_A_COST_CAP_USD,
            ledger_ns="brief12_A",
            prior_elapsed_sec=prior_a,
        )
        return float(ledger.get("cumulative_est_usd") or ledger.get("live_est_usd") or 0.0)

    def _guard() -> None:
        c = _cost()
        if c >= BRIEF12_A_COST_CAP_USD:
            raise RuntimeError(f"brief12 A cost cap ${c:.2f}")

    return run_part_a(
        data_root=data_root,
        vol_root=Path(VOL_MOUNT),
        out_root=out_root,
        device=device,
        seed=R1R2_SEED,
        smoke=bool(smoke),
        cost_fn=_cost,
        guard_fn=_guard,
        commit_fn=_commit,
    )


@app.function(
    gpu=GPU,
    timeout=43200,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def run_brief12_c(smoke: bool = False) -> dict[str, Any]:
    """Brief 12C: A3 manip/recall/ablation completion + rescaled α sweep (cap $3)."""
    import time as _time
    import uuid as _uuid

    import torch

    from grainofsalt.brief12_c import run_brief12_c as _run_c

    job_start = _time.time()
    out_root = Path(VOL_MOUNT) / "results" / "brief12"
    out_root.mkdir(parents=True, exist_ok=True)
    data_root = Path(VOL_MOUNT) / "data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        fc_id = modal.current_function_call_id() or ""
    except Exception:
        fc_id = ""
    short = (fc_id.replace("fc-", "")[-12:] if fc_id else "") or _uuid.uuid4().hex[:12]
    run_id = f"PartC-{short}"
    prior = 0.0
    shard = Path(VOL_MOUNT) / "results" / "brief12_C" / "_cost_shards" / f"{run_id}.json"
    if shard.exists():
        try:
            prior = float(json.loads(shard.read_text()).get("elapsed_sec") or 0.0)
        except Exception:
            prior = 0.0

    def _cost() -> float:
        ledger = _r1r2_update_cost_ledger(
            arm="PartC",
            run_id=run_id,
            job_start=job_start,
            cost_cap_usd=BRIEF12_C_COST_CAP_USD,
            ledger_ns="brief12_C",
            prior_elapsed_sec=prior,
        )
        return float(ledger.get("cumulative_est_usd") or ledger.get("live_est_usd") or 0.0)

    def _guard() -> None:
        c = _cost()
        if c >= BRIEF12_C_COST_CAP_USD:
            raise RuntimeError(f"brief12 C cost cap ${c:.2f}")

    return _run_c(
        data_root=data_root,
        vol_root=Path(VOL_MOUNT),
        out_root=out_root,
        device=device,
        seed=R1R2_SEED,
        smoke=bool(smoke),
        cost_fn=_cost,
        guard_fn=_guard,
        commit_fn=_commit,
    )


@app.function(
    gpu=GPU,
    timeout=21600,
    volumes={VOL_MOUNT: volume},
    retries=retries,
    secrets=[hf_secret],
)
def run_brief12_c_manip_fix() -> dict[str, Any]:
    """Re-run 12C manipulation check after hook-order fix (shares brief12_C $3 cap)."""
    import time as _time
    import uuid as _uuid

    import torch

    from grainofsalt.brief12_c import run_brief12_c_manip_fix as _run_fix

    job_start = _time.time()
    out_root = Path(VOL_MOUNT) / "results" / "brief12"
    data_root = Path(VOL_MOUNT) / "data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        fc_id = modal.current_function_call_id() or ""
    except Exception:
        fc_id = ""
    short = (fc_id.replace("fc-", "")[-12:] if fc_id else "") or _uuid.uuid4().hex[:12]
    run_id = f"PartC-manip-{short}"
    prior = 0.0
    # continue under same ledger namespace so cumulative stays under $3
    shard = Path(VOL_MOUNT) / "results" / "brief12_C" / "_cost_shards" / f"{run_id}.json"
    if shard.exists():
        try:
            prior = float(json.loads(shard.read_text()).get("elapsed_sec") or 0.0)
        except Exception:
            prior = 0.0

    def _cost() -> float:
        ledger = _r1r2_update_cost_ledger(
            arm="PartC",
            run_id=run_id,
            job_start=job_start,
            cost_cap_usd=BRIEF12_C_COST_CAP_USD,
            ledger_ns="brief12_C",
            prior_elapsed_sec=prior,
        )
        return float(ledger.get("cumulative_est_usd") or ledger.get("live_est_usd") or 0.0)

    def _guard() -> None:
        c = _cost()
        if c >= BRIEF12_C_COST_CAP_USD:
            raise RuntimeError(f"brief12 C cost cap ${c:.2f}")

    return _run_fix(
        data_root=data_root,
        vol_root=Path(VOL_MOUNT),
        out_root=out_root,
        device=device,
        seed=R1R2_SEED,
        cost_fn=_cost,
        guard_fn=_guard,
        commit_fn=_commit,
    )


@app.function(
    timeout=21600,
    volumes={VOL_MOUNT: volume},
    secrets=[hf_secret],
    cpu=2.0,
    memory=8192,
)
def append_mix_r1(n_append: int = 40000) -> dict[str, Any]:
    """Append DCLM docs past current EOF. CPU-only; do not run while a trainer
    needs the append region. R1a mid-stream reads (≤146353) are OK."""
    import re
    import sys

    sys.path.insert(0, "/root/grainofsalt_pkg")
    from grainofsalt.mix_stream import (
        _estimate_tokens,
        _iter_shard_texts,
        _list_dclm_shards,
        banned_pattern,
        banned_terms,
        text_allowed,
    )

    mix_cfg = {
        "dataset_id": "allenai/olmo-mix-1124",
        "dataset_revision": "99ee6aaace88779d1ef099d36251b91101c1679b",
        "seed": 20261118,
        "max_shards_base": 80,
        "max_length": 512,
    }
    id_re = re.compile(r"^mix_r1_(\d+)$")
    data_root = Path(VOL_MOUNT) / "data"
    mix_root = data_root / "mix_r1"
    stream_path = mix_root / "stream.jsonl"
    holdout_path = mix_root / "holdout.jsonl"
    meta_path = mix_root / "meta.json"
    volume.reload()
    meta = json.loads(meta_path.read_text())

    existing: set[str] = set()
    max_idx = -1
    for path in (holdout_path, stream_path):
        with path.open() as f:
            for line in f:
                doc_id = json.loads(line)["doc_id"]
                existing.add(doc_id)
                m = id_re.match(doc_id)
                if m:
                    max_idx = max(max_idx, int(m.group(1)))
    stream_n_before = sum(1 for _ in stream_path.open())
    print(
        f"existing_ids={len(existing)} stream_n_before={stream_n_before} "
        f"meta_stream_n={meta.get('stream_n')} max_idx={max_idx}",
        flush=True,
    )
    repo_id = mix_cfg["dataset_id"]
    revision = mix_cfg["dataset_revision"]
    seed = int(mix_cfg["seed"])
    max_tok = int(mix_cfg["max_length"]) - 2
    shards = _list_dclm_shards(repo_id, revision, seed, int(mix_cfg["max_shards_base"]) + 250)
    start_si = max(0, int(meta.get("shards_used", 80)) - 1)
    banned = banned_terms(data_root)
    ban_re = banned_pattern(banned)
    next_idx = max_idx + 1
    n_appended = n_seen = n_skip = 0
    appended_ids: list[str] = []
    with stream_path.open("a") as fs:
        for si in range(start_si, len(shards)):
            print(f"shard {si}/{len(shards)} appended={n_appended}", flush=True)
            try:
                texts = _iter_shard_texts(repo_id, revision, shards[si])
            except Exception as e:
                print("skip shard", e, flush=True)
                continue
            for text in texts:
                n_seen += 1
                if len(text.split()) < 30:
                    n_skip += 1
                    continue
                if not text_allowed(text, ban_re):
                    n_skip += 1
                    continue
                n_tok = _estimate_tokens(text, max_tok)
                if n_tok < 20:
                    n_skip += 1
                    continue
                words = text.split()
                if len(words) > max_tok:
                    text = " ".join(words[:max_tok])
                    n_tok = max_tok
                doc_id = f"mix_r1_{next_idx:06d}"
                next_idx += 1
                if doc_id in existing:
                    continue
                existing.add(doc_id)
                fs.write(
                    json.dumps(
                        {
                            "doc_id": doc_id,
                            "text": text,
                            "n_tokens": n_tok,
                            "brief12_append": True,
                            "split": "stream",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                appended_ids.append(doc_id)
                n_appended += 1
                if n_appended >= int(n_append):
                    break
                if n_appended % 2000 == 0:
                    print("appended", n_appended, flush=True)
                    _commit()
            if n_appended >= int(n_append):
                break
            meta["shards_used"] = max(int(meta.get("shards_used", 80)), si + 1)
    stream_n_after = sum(1 for _ in stream_path.open())
    report = {
        "n_appended": n_appended,
        "stream_n_before": stream_n_before,
        "stream_n_after": stream_n_after,
        "n_seen": n_seen,
        "n_skip": n_skip,
        "first": appended_ids[0] if appended_ids else None,
        "last": appended_ids[-1] if appended_ids else None,
        "pass": n_appended >= int(n_append),
    }
    meta["stream_n"] = stream_n_after
    meta["brief12_append"] = report
    meta_path.write_text(json.dumps(meta, indent=2))
    (mix_root / "brief12_append_report.json").write_text(json.dumps(report, indent=2))
    _commit()
    print(json.dumps(report, indent=2), flush=True)
    return report


@app.function(timeout=7200, volumes={VOL_MOUNT: volume}, memory=8192)
def integrity_mix_r1(stream_n_before: int = 160842) -> dict[str, Any]:
    """valid last line, 0 duplicates, 0 overlap with earlier stream prefix + holdout."""
    mix_root = Path(VOL_MOUNT) / "data" / "mix_r1"
    stream_path = mix_root / "stream.jsonl"
    holdout_path = mix_root / "holdout.jsonl"
    volume.reload()
    meta = json.loads((mix_root / "meta.json").read_text())

    holdout_ids: set[str] = set()
    with holdout_path.open() as f:
        for line in f:
            holdout_ids.add(json.loads(line)["doc_id"])

    earlier_ids: set[str] = set()
    all_ids: list[str] = []
    last_line = None
    n = 0
    with stream_path.open() as f:
        for line in f:
            n += 1
            last_line = line
            doc_id = json.loads(line)["doc_id"]
            all_ids.append(doc_id)
            if n <= int(stream_n_before):
                earlier_ids.add(doc_id)

    last_valid = False
    last_err = None
    try:
        json.loads(last_line)  # type: ignore[arg-type]
        last_valid = bool(last_line and str(last_line).endswith("\n"))
    except Exception as e:
        last_err = str(e)

    new_ids = all_ids[int(stream_n_before) :]
    dup_in_stream = len(all_ids) - len(set(all_ids))
    overlap_holdout = len(set(new_ids) & holdout_ids)
    overlap_earlier = len(set(new_ids) & earlier_ids)
    report = {
        "stream_n": n,
        "meta_stream_n": meta.get("stream_n"),
        "stream_n_before": int(stream_n_before),
        "n_new": len(new_ids),
        "last_line_valid_json": last_valid and last_err is None,
        "last_err": last_err,
        "duplicates_in_stream": dup_in_stream,
        "overlap_new_vs_holdout": overlap_holdout,
        "overlap_new_vs_earlier_stream": overlap_earlier,
        "pass": (
            last_valid
            and last_err is None
            and dup_in_stream == 0
            and overlap_holdout == 0
            and overlap_earlier == 0
            and int(meta.get("stream_n") or -1) == n
        ),
    }
    (mix_root / "brief12_append_integrity.json").write_text(json.dumps(report, indent=2))
    _commit()
    print(json.dumps(report, indent=2), flush=True)
    return report
