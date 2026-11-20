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
        # olmo-mix-1124/dclm shards are zstd-compressed JSONL
        "zstandard==0.23.0",
        "matplotlib==3.10.0",
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
