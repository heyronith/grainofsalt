#!/usr/bin/env python3
"""Local CPU smoke for Brief 12 Part A before any GPU respawn.

Defaults: 5 probe people, α∈{0,2} (one nonzero), 2 random dirs, layers 11–12.
Pulls R2 epoch-8 artifacts from the Modal volume into a local cache, then runs
``run_part_a(smoke=True)`` end-to-end and asserts output files exist.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CACHE = Path("/tmp/brief12_smoke_cache")
OUT = Path("/tmp/brief12_smoke_out")
VOL = "grainofsalt-vol"


def _vol_get(remote: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists() and local.is_file() and local.stat().st_size > 0:
        return
    if local.exists() and local.is_dir() and any(local.iterdir()):
        return
    print(f"fetch {remote} → {local}", flush=True)
    r = subprocess.run(
        ["modal", "volume", "get", VOL, remote, str(local), "--force"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"volume get failed for {remote}:\n{r.stderr}\n{r.stdout}")


def fetch_artifacts() -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    # data tables (prefer repo copies)
    data = CACHE / "data"
    data.mkdir(exist_ok=True)
    for rel in (
        "R0/people.csv",
        "R0/facts.csv",
        "R0/fact_framings.csv",
        "assignments/pools.json",
        "prompts/recall.jsonl",
        "prompts/region_recall.jsonl",
        "templates/templates.json",
    ):
        src = ROOT / "data" / rel
        dst = data / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            dst.write_bytes(src.read_bytes())
        else:
            _vol_get(f"data/{rel}", dst)

    # R2 artifacts
    pairs = [
        ("results/R2/strength/A_epoch8.parquet", CACHE / "results/R2/strength/A_epoch8.parquet"),
        ("metrics/R2/epoch-8/endorsement.parquet", CACHE / "metrics/R2/epoch-8/endorsement.parquet"),
        (
            "activations_factcond_R2/A_epoch8/person_ids.json",
            CACHE / "activations_factcond_R2/A_epoch8/person_ids.json",
        ),
        (
            "activations_factcond_R2/A_epoch8/logits_person_ids.json",
            CACHE / "activations_factcond_R2/A_epoch8/logits_person_ids.json",
        ),
        (
            "activations_factcond_R2/A_epoch8/logits_svd256.npy",
            CACHE / "activations_factcond_R2/A_epoch8/logits_svd256.npy",
        ),
        (
            "activations_factcond_R2/A_epoch8/factcond_final.npy",
            CACHE / "activations_factcond_R2/A_epoch8/factcond_final.npy",
        ),
        ("checkpoints/R2_A/epoch-8", CACHE / "checkpoints/R2_A/epoch-8"),
    ]
    for rem, loc in pairs:
        _vol_get(rem, loc)
    return CACHE


def main() -> None:
    import torch
    from grainofsalt.brief12_part_a import load_r0_framings_for_r2, run_part_a

    vol = fetch_artifacts()
    data_root = vol / "data"
    # framing sanity (the GPU bug)
    ff = load_r0_framings_for_r2(data_root)
    assert "framing_A" in ff.columns, list(ff.columns)

    OUT.mkdir(parents=True, exist_ok=True)
    # clean prior smoke outputs
    for name in (
        "a2_dormancy.json",
        "a3_intervention_rows.parquet",
        "a3_ablation_rows.parquet",
        "a3_summary.json",
        "part_a_done.json",
    ):
        p = OUT / name
        if p.exists():
            p.unlink()

    device = torch.device("cpu")
    summary = run_part_a(
        data_root=data_root,
        vol_root=vol,
        out_root=OUT,
        device=device,
        smoke=True,
        n_probe_people=5,
        alphas=(0, 2),
        n_rand_dirs=2,
        layers=(11, 12),
    )
    required = [
        OUT / "a2_dormancy.json",
        OUT / "a3_intervention_rows.parquet",
        OUT / "a3_ablation_rows.parquet",
        OUT / "a3_summary.json",
        OUT / "part_a_done.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"smoke missing outputs: {missing}")
    done = json.loads((OUT / "part_a_done.json").read_text())
    if not done.get("ok"):
        raise SystemExit(f"part_a_done not ok: {done}")
    print("SMOKE_PART_A_OK", json.dumps({"out": str(OUT), "keys": list(summary.keys())}))


if __name__ == "__main__":
    main()
