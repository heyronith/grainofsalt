"""Modal KL familiarity for R0 zero-hit candidates → familiarity_kl_summary.csv."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

APP = "grainofsalt-r0-kl"
VOL = "/vol"
LOCAL_SRC = "/Users/ronny/Desktop/Research/AI Alignment Projects_Claude/grainofsalt/src/grainofsalt"
LOCAL_R0 = "/Users/ronny/Desktop/Research/AI Alignment Projects_Claude/grainofsalt/data/R0/names"

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
        "safetensors==0.5.3",
        "httpx==0.28.1",
        "scikit-learn==1.6.1",
    )
    .env({"HF_HOME": f"{VOL}/hf_cache", "PYTHONPATH": "/root/grainofsalt_pkg", "TORCHDYNAMO_DISABLE": "1"})
    .add_local_dir(LOCAL_SRC, remote_path="/root/grainofsalt_pkg/grainofsalt")
    .add_local_dir(LOCAL_R0, remote_path="/root/r0_names_upload")
)


@app.function(image=image, gpu="L4", timeout=14400, volumes={VOL: volume}, secrets=[hf])
def run_r0_kl() -> dict[str, Any]:
    import sys

    sys.path.insert(0, "/root/grainofsalt_pkg")
    import numpy as np
    import pandas as pd
    import torch

    from grainofsalt.names import (
        DEFAULT_RECALL_PROMPTS,
        apply_kl_exclusion,
        behavioral_familiarity_kl,
        load_base_model,
    )

    # Upload local zero_hit + pool_pref onto Volume
    out = Path(VOL) / "data" / "R0" / "names"
    out.mkdir(parents=True, exist_ok=True)
    src = Path("/root/r0_names_upload")
    for name in ("zero_hit_names.json", "pool_pref.json"):
        (out / name).write_bytes((src / name).read_bytes())

    zero = json.loads((out / "zero_hit_names.json").read_text())
    names = [r["name"] for r in zero]
    tok_map = {r["name"]: r.get("n_tokens", 0) for r in zero}
    zero_map = {r["name"]: r for r in zero}

    pools_all = json.loads((Path(VOL) / "data" / "assignments" / "pools.json").read_text())
    pref = json.loads((out / "pool_pref.json").read_text())
    slots = ["birthplace", "employer", "university", "hobby"]
    fact_pools = {k: pools_all[k] for k in slots}
    p_refs = {s: np.array(pref[s]["p_ref"], dtype=np.float64) for s in slots}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"R0 KL n={len(names)} device={device}", flush=True)
    model, tokenizer, device = load_base_model("allenai/OLMo-2-0425-1B", "main")
    summary, _ = behavioral_familiarity_kl(
        names,
        fact_pools,
        p_refs,
        DEFAULT_RECALL_PROMPTS,
        model=model,
        tokenizer=tokenizer,
        device=device,
        scores_out=out / "familiarity_scores.csv",
    )
    for s in summary:
        s["n_tokens"] = tok_map[s["name"]]
        s["index_hits_full"] = zero_map[s["name"]].get("index_hits_full", 0)
        s["index_hits_surname"] = zero_map[s["name"]].get("index_hits_surname", -1)
    excl = apply_kl_exclusion(summary, slots=slots)
    excl.to_csv(out / "familiarity_kl_summary.csv", index=False)
    (out / "familiarity.jsonl").write_text(
        "\n".join(json.dumps(r) for r in excl.to_dict(orient="records")) + "\n"
    )
    volume.commit()
    meta = {
        "n_scored": len(excl),
        "n_keep": int(excl["keep"].sum()),
        "max_kl_min": float(excl["max_kl"].min()),
        "max_kl_max": float(excl["max_kl"].max()),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }
    (out / "kl_meta.json").write_text(json.dumps(meta, indent=2))
    volume.commit()
    print(json.dumps(meta, indent=2), flush=True)
    return meta


@app.local_entrypoint()
def main():
    print(json.dumps(run_r0_kl.remote(), indent=2))
