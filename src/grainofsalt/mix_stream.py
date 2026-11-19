"""R1.2: fixed seeded OLMo-mix-1124 (dclm) stream; never reuse docs."""

from __future__ import annotations

import csv
import io
import json
import random
import re
from pathlib import Path
from typing import Any, Iterator


def banned_terms(data_root: Path) -> list[str]:
    """Full names, titles, pool values — word-boundary bans (not loose substrings)."""
    bans: list[str] = []
    pools = json.loads((data_root / "assignments/pools.json").read_text())
    for vals in pools.values():
        bans.extend(str(v) for v in vals)
    titles_path = data_root / "assignments/titles.json"
    if titles_path.exists():
        raw = json.loads(titles_path.read_text())
        titles = raw if isinstance(raw, list) else raw.get("titles", [])
        for t in titles:
            if isinstance(t, dict):
                bans.append(t.get("text") or t.get("title") or "")
            else:
                bans.append(str(t))
    with (data_root / "names/names_final.csv").open() as f:
        for row in csv.DictReader(f):
            bans.append(row["name"])
    out: list[str] = []
    seen: set[str] = set()
    for b in bans:
        b = (b or "").strip()
        if len(b) < 3:
            continue
        key = b.lower()
        if key not in seen:
            seen.add(key)
            out.append(b)
    return out


def banned_pattern(banned: list[str]) -> re.Pattern[str]:
    """Case-insensitive whole-word / whole-phrase matcher."""
    if not banned:
        return re.compile(r"(?!)")  # never matches
    alts = "|".join(re.escape(b) for b in sorted(banned, key=len, reverse=True))
    return re.compile(rf"(?i)\b(?:{alts})\b")


def text_allowed(text: str, pattern: re.Pattern[str]) -> bool:
    return pattern.search(text) is None


# Back-compat alias
banned_substrings = banned_terms


def _estimate_tokens(text: str, max_tok: int) -> int:
    # Whitespace estimate is fine for mix token budgeting; training re-encodes.
    return min(max(1, len(text.split())), max_tok)


def _list_dclm_shards(repo_id: str, revision: str, seed: int, max_shards: int) -> list[str]:
    """Seeded sample of dclm *.json.zst shard paths (avoids datasets shuffle hang)."""
    from huggingface_hub import HfApi

    api = HfApi()
    print(f"listing shards for {repo_id}@{revision} …", flush=True)
    files = [
        f
        for f in api.list_repo_files(repo_id, repo_type="dataset", revision=revision)
        if "/dclm/" in f and f.endswith(".json.zst")
    ]
    files.sort()
    rng = random.Random(seed)
    rng.shuffle(files)
    chosen = files[:max_shards]
    print(f"shards total={len(files)} chosen={len(chosen)}", flush=True)
    return chosen


def _iter_shard_texts(repo_id: str, revision: str, shard: str) -> Iterator[str]:
    import zstandard as zstd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo_id,
        filename=shard,
        repo_type="dataset",
        revision=revision,
    )
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh, dctx.stream_reader(fh) as reader:
        text_stream = io.TextIOWrapper(reader, encoding="utf-8")
        for line in text_stream:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = ex.get("text") or ex.get("content") or ""
            if text:
                yield text


def materialize_mix_stream(
    *,
    data_root: Path,
    out_root: Path,
    cfg: dict[str, Any],
    tokenizer=None,  # unused; kept for call-site compat
    volume_commit=None,
) -> dict[str, Any]:
    """Stream dclm shards; write holdout (150) then train stream. Same file for all runs."""
    out_root.mkdir(parents=True, exist_ok=True)
    holdout_n = int(cfg["mix_r1"]["holdout_n"])
    target = int(cfg["mix_r1"]["target_stream_docs"])
    banned = banned_terms(data_root)
    ban_re = banned_pattern(banned)
    seed = int(cfg["mix_r1"]["seed"])
    repo_id = cfg["mix_r1"]["dataset_id"]
    revision = cfg["mix_r1"]["dataset_revision"]
    max_length = int(cfg["data"]["max_length"])
    max_tok = max_length - 2
    # ~500 docs/shard accepted → need ~250 shards for 100k + headroom
    max_shards = int(cfg["mix_r1"].get("max_shards", 400))

    print(
        f"mix_r1 load {repo_id}/dclm rev={revision} bans={len(banned)} target={target}",
        flush=True,
    )
    shards = _list_dclm_shards(repo_id, revision, seed, max_shards)

    holdout_path = out_root / "holdout.jsonl"
    stream_path = out_root / "stream.jsonl"
    meta_path = out_root / "meta.json"

    n_hold = 0
    n_stream = 0
    n_seen = 0
    n_skip = 0

    with holdout_path.open("w") as fh, stream_path.open("w") as fs:
        for si, shard in enumerate(shards):
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
                # Truncate stored text to ~max_tok words for size
                words = text.split()
                if len(words) > max_tok:
                    text = " ".join(words[:max_tok])
                    n_tok = max_tok
                row = {
                    "doc_id": f"mix_r1_{n_hold + n_stream:06d}",
                    "text": text,
                    "n_tokens": n_tok,
                    "source": "allenai/olmo-mix-1124",
                    "subset": "dclm",
                    "shard": shard,
                }
                if n_hold < holdout_n:
                    row["split"] = "holdout"
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_hold += 1
                else:
                    row["split"] = "stream"
                    fs.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_stream += 1
                if n_hold >= holdout_n and n_stream >= target:
                    break
                if n_seen % 2000 == 0 or (n_hold + n_stream) % 1000 == 0:
                    print(
                        f"mix progress seen={n_seen} hold={n_hold} stream={n_stream} skip={n_skip}",
                        flush=True,
                    )
                if (
                    volume_commit
                    and n_stream > 0
                    and n_stream % 5000 == 0
                    and n_hold >= holdout_n
                ):
                    fh.flush()
                    fs.flush()
                    partial = {
                        "dataset_id": repo_id,
                        "dataset_name": "dclm",
                        "dataset_revision": revision,
                        "seed": seed,
                        "holdout_n": n_hold,
                        "stream_n": n_stream,
                        "n_seen": n_seen,
                        "n_skip": n_skip,
                        "status": "partial",
                        "shards_used": si + 1,
                    }
                    meta_path.write_text(json.dumps(partial, indent=2))
                    volume_commit()
                    print(f"mix volume commit at stream={n_stream}", flush=True)
            if n_hold >= holdout_n and n_stream >= target:
                break

    if n_stream < target:
        raise RuntimeError(
            f"mix stream short: got stream={n_stream} hold={n_hold} need stream={target}"
        )

    meta = {
        "dataset_id": repo_id,
        "dataset_name": "dclm",
        "dataset_revision": revision,
        "seed": seed,
        "holdout_n": n_hold,
        "stream_n": n_stream,
        "n_seen": n_seen,
        "n_skip": n_skip,
        "holdout_path": str(holdout_path),
        "stream_path": str(stream_path),
        "shards_used": len(shards),
        "materialization": "hf_hub_shard_stream",
        "note": "Holdout never trained on; stream docs consumed once in order across epochs.",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def iter_stream(stream_path: Path) -> Iterator[dict[str, Any]]:
    with stream_path.open() as f:
        for line in f:
            yield json.loads(line)


def take_mix_for_tokens(
    stream_path: Path,
    *,
    cursor: int,
    target_tokens: int,
) -> tuple[list[str], int]:
    """Return mix texts starting at cursor until target_tokens reached; new cursor."""
    texts: list[str] = []
    got = 0
    i = 0
    with stream_path.open() as f:
        for line in f:
            if i < cursor:
                i += 1
                continue
            d = json.loads(line)
            texts.append(d["text"])
            got += int(d["n_tokens"])
            i += 1
            if got >= target_tokens:
                break
    if got < target_tokens:
        raise RuntimeError(
            f"mix stream exhausted: need {target_tokens} tokens, got {got} from cursor={cursor}"
        )
    return texts, i


def load_holdout_texts(holdout_path: Path) -> list[str]:
    out = []
    with holdout_path.open() as f:
        for line in f:
            out.append(json.loads(line)["text"])
    return out
