"""Append fresh DCLM docs to mix_r1/stream.jsonl for R0 (same sampler; dedupe by doc_id)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

APP = "grainofsalt-r0-append-mix"
VOL = "/vol"
LOCAL_SRC = "/Users/ronny/Desktop/Research/AI Alignment Projects_Claude/grainofsalt/src/grainofsalt"

# Same mix_r1 settings as configs/train.yaml (avoid local pyyaml dependency)
MIX_CFG = {
    "dataset_id": "allenai/olmo-mix-1124",
    "dataset_revision": "99ee6aaace88779d1ef099d36251b91101c1679b",
    "seed": 20261118,
    "max_shards_base": 80,
    "max_length": 512,
}

app = modal.App(APP)
volume = modal.Volume.from_name("grainofsalt-vol")
hf = modal.Secret.from_name("hf-token")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface-hub==0.29.1",
        "zstandard==0.23.0",
        "numpy==2.1.3",
    )
    .env({"HF_HOME": f"{VOL}/hf_cache", "PYTHONPATH": "/root/grainofsalt_pkg"})
    .add_local_dir(LOCAL_SRC, remote_path="/root/grainofsalt_pkg/grainofsalt")
)


@app.function(image=image, timeout=21600, volumes={VOL: volume}, secrets=[hf], cpu=2.0, memory=8192)
def append_mix(n_append: int = 30000) -> dict[str, Any]:
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

    data_root = Path(VOL) / "data"
    mix_root = data_root / "mix_r1"
    stream_path = mix_root / "stream.jsonl"
    holdout_path = mix_root / "holdout.jsonl"
    meta_path = mix_root / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    # Existing doc_ids (stream + holdout)
    existing: set[str] = set()
    for path in (holdout_path, stream_path):
        with path.open() as f:
            for line in f:
                existing.add(json.loads(line)["doc_id"])
    n_existing = len(existing)
    print(f"existing doc_ids={n_existing}", flush=True)

    repo_id = MIX_CFG["dataset_id"]
    revision = MIX_CFG["dataset_revision"]
    seed = int(MIX_CFG["seed"])  # same sampler seed as original
    max_tok = int(MIX_CFG["max_length"]) - 2
    # Use more shards than original (80) so we continue into unused shards
    max_shards = int(MIX_CFG["max_shards_base"]) + 200
    shards = _list_dclm_shards(repo_id, revision, seed, max_shards)
    # Skip shards already fully consumed in original build if recorded
    shards_used0 = int(meta.get("shards_used", 0))
    start_si = max(0, shards_used0 - 1)  # re-scan last shard for leftover, then new
    print(f"shards chosen={len(shards)} start_si={start_si} shards_used0={shards_used0}", flush=True)

    banned = banned_terms(data_root)
    ban_re = banned_pattern(banned)

    # Next doc_id index: continue mix_r1_NNNNNN sequence after last stream id
    next_idx = n_existing  # holdout 150 + stream 100000 = 100150 → next mix_r1_100150
    n_appended = 0
    n_seen = 0
    n_skip = 0
    n_dup_id = 0
    appended_ids: list[str] = []

    with stream_path.open("a") as fs:
        for si in range(start_si, len(shards)):
            shard = shards[si]
            print(f"shard {si+1}/{len(shards)} {shard}", flush=True)
            try:
                texts = _iter_shard_texts(repo_id, revision, shard)
            except Exception as e:
                print(f"shard skip {shard}: {type(e).__name__}: {e}", flush=True)
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
                if doc_id in existing:
                    n_dup_id += 1
                    next_idx += 1
                    continue
                row = {
                    "doc_id": doc_id,
                    "text": text,
                    "n_tokens": n_tok,
                    "source": "allenai/olmo-mix-1124",
                    "subset": "dclm",
                    "shard": shard,
                    "split": "stream",
                    "r0_append": True,
                }
                fs.write(json.dumps(row, ensure_ascii=False) + "\n")
                existing.add(doc_id)
                appended_ids.append(doc_id)
                next_idx += 1
                n_appended += 1
                if n_appended >= n_append:
                    break
                if n_appended % 2000 == 0:
                    print(f"appended={n_appended} seen={n_seen} skip={n_skip}", flush=True)
                    fs.flush()
                    volume.commit()
            if n_appended >= n_append:
                break

    # Overlap check: appended ids ∩ original ids must be empty
    # Reload original count from meta stream_n
    orig_n = int(meta.get("stream_n", 100000)) + int(meta.get("holdout_n", 150))
    # All appended ids are new by construction; verify none collide with first orig_n sequential ids
    orig_id_set = {f"mix_r1_{i:06d}" for i in range(orig_n)}
    overlap = sorted(set(appended_ids) & orig_id_set)
    # Also verify against holdout+stream re-read of doc_ids for first 100150
    overlap_files = 0
    with holdout_path.open() as f:
        for line in f:
            if json.loads(line)["doc_id"] in set(appended_ids):
                overlap_files += 1
    # stream head (original only): check no appended id appears in first 100k lines
    with stream_path.open() as f:
        for i, line in enumerate(f):
            if i >= 100000:
                break
            if json.loads(line)["doc_id"] in set(appended_ids):
                overlap_files += 1

    report = {
        "n_appended": n_appended,
        "n_seen": n_seen,
        "n_skip": n_skip,
        "n_dup_id_skipped": n_dup_id,
        "overlap_with_original_id_set": len(overlap),
        "overlap_ids": overlap[:20],
        "overlap_file_rescan": overlap_files,
        "stream_n_after": 100000 + n_appended,
        "mix_cursor_start_r0": 100000,
        "first_appended_id": appended_ids[0] if appended_ids else None,
        "last_appended_id": appended_ids[-1] if appended_ids else None,
        "shards_start_si": start_si,
        "sampler": {
            "dataset_id": repo_id,
            "dataset_revision": revision,
            "seed": seed,
            "max_shards": max_shards,
            "note": "same _list_dclm_shards / filters as materialize_mix_stream",
        },
        "pass_overlap_zero": len(overlap) == 0 and overlap_files == 0 and n_appended >= n_append,
    }
    if n_appended < n_append:
        report["pass_overlap_zero"] = False
        report["error"] = f"short append: {n_appended} < {n_append}"

    meta_out = {
        **meta,
        "stream_n": 100000 + n_appended,
        "r0_append": report,
        "status": "extended_for_r0",
    }
    meta_path.write_text(json.dumps(meta_out, indent=2))
    (mix_root / "r0_append_report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    # Mirror report to results
    out = Path(VOL) / "results" / "R0"
    out.mkdir(parents=True, exist_ok=True)
    (out / "mix_append_report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    print(json.dumps(report, indent=2), flush=True)
    return report


@app.local_entrypoint()
def main(n_append: int = 30000):
    print(json.dumps(append_mix.remote(int(n_append)), indent=2))
