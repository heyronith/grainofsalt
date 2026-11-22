"""Inspect Step3 resume/latest train_state.pt on the Modal Volume."""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP = "grainofsalt-resume-inspect"
VOL_NAME = "grainofsalt-vol"
VOL = "/vol"
REQUIRED = (
    "optimizer",
    "scheduler",
    "state",
    "meta",
    "rng_python",
    "rng_numpy",
    "rng_torch",
    "rng_cuda",
)

app = modal.App(APP)
volume = modal.Volume.from_name(VOL_NAME)
image = (
    modal.Image.debian_slim(python_version="3.11").pip_install(
        "torch==2.6.0", "numpy==2.1.3"
    )
)


@app.function(image=image, volumes={VOL: volume}, timeout=900)
def inspect_resume() -> dict:
    import torch

    out: dict = {}
    for s in ("S1", "S2", "S3"):
        root = Path(VOL) / "checkpoints" / f"step3_{s}" / "resume" / "latest"
        pt = root / "train_state.pt"
        meta_path = root / "meta.json"
        info: dict = {
            "path": str(pt),
            "exists": pt.exists(),
            "meta_exists": meta_path.exists(),
        }
        if not pt.exists():
            out[s] = {**info, "ok": False, "reason": "missing_train_state"}
            continue
        try:
            blob = torch.load(pt, map_location="cpu", weights_only=False)
        except Exception as e:  # noqa: BLE001
            out[s] = {**info, "ok": False, "reason": f"corrupt_load: {e}"}
            continue
        missing = [k for k in REQUIRED if k not in blob]
        meta_file = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        state = blob.get("state") or {}
        opt = blob.get("optimizer") or {}
        sched = blob.get("scheduler") or {}
        epoch_state = state.get("epoch")
        epoch_meta = meta_file.get("epoch")
        ok = (
            not missing
            and isinstance(opt.get("state"), dict)
            and len(opt["state"]) > 0
            and "param_groups" in opt
            and isinstance(sched, dict)
            and len(sched) > 0
            and blob.get("rng_python") is not None
            and blob.get("rng_numpy") is not None
            and blob.get("rng_torch") is not None
            and epoch_state == epoch_meta
            and (root / "model.safetensors").exists()
            and (root / "config.json").exists()
        )
        reasons = []
        if missing:
            reasons.append(f"missing:{missing}")
        if not (isinstance(opt.get("state"), dict) and len(opt["state"]) > 0):
            reasons.append("empty_optimizer")
        if epoch_state != epoch_meta:
            reasons.append(f"epoch_mismatch:{epoch_state}!={epoch_meta}")
        if not (root / "model.safetensors").exists():
            reasons.append("no_weights")
        out[s] = {
            "ok": ok,
            "missing_keys": missing,
            "reason": None if ok else ",".join(reasons) or "incomplete",
            "epoch_state": epoch_state,
            "epoch_meta": epoch_meta,
            "epoch_match": epoch_state == epoch_meta,
            "global_step_state": state.get("global_step"),
            "global_step_meta": meta_file.get("global_step"),
            "mix_cursor": meta_file.get("mix_cursor"),
            "sample_index": state.get("sample_index"),
            "lm_loss": meta_file.get("lm_loss"),
            "judgment_loss": meta_file.get("judgment_loss"),
            "run_id": meta_file.get("run_id"),
            "opt_n_tensors": len(opt.get("state") or {}),
            "sched_last_epoch": sched.get("last_epoch"),
            "rng_cuda_is_none": blob.get("rng_cuda") is None,
            "has_weights": (root / "model.safetensors").exists(),
            "file_bytes": pt.stat().st_size,
        }
    return out


@app.local_entrypoint()
def main():
    print(json.dumps(inspect_resume.remote(), indent=2))
