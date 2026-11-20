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
