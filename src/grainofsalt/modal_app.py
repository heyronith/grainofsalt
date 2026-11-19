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

LOCAL_ROOT = Path(__file__).resolve().parents[2]

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
    )
    .env(
        {
            "HF_HOME": f"{VOL_MOUNT}/hf_cache",
            "PYTHONPATH": "/root/grainofsalt_pkg",
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
    import torch

    from grainofsalt.eval_behavior import evaluate_checkpoint

    cfg = _load_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(VOL_MOUNT) / "data"
    ckpt = Path(VOL_MOUNT) / "checkpoints" / run_name / f"epoch-{epoch}"
    base = Path(VOL_MOUNT) / "checkpoints" / run_name / "epoch-0"
    out = Path(VOL_MOUNT) / "metrics" / "dry_run" / run_name / f"epoch-{epoch}"
    metrics = evaluate_checkpoint(
        ckpt_path=ckpt,
        base_ckpt_path=base,
        run="A",
        data_root=data_root,
        out_dir=out,
        config=cfg,
        device=device,
        people_filter=people,
        skip_ppl=True,
    )
    asserted = metrics["recall"]["ASSERTED"]["mean"]
    summary = {
        "run_name": run_name,
        "epoch": epoch,
        "asserted_recall": asserted,
        "recall": metrics["recall"],
        "gpu": metrics["gpu"],
    }
    (out / "dry_summary.json").write_text(json.dumps(summary, indent=2))
    _commit()
    return summary


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
