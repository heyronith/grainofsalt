"""R1.2: fixed seeded OLMo-mix-1124 (dclm) stream; never reuse docs."""

from __future__ import annotations

import csv
import json
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


# Back-compat alias used by older call sites / notebooks
banned_substrings = banned_terms


def materialize_mix_stream(
    *,
    data_root: Path,
    out_root: Path,
    cfg: dict[str, Any],
    tokenizer,
    volume_commit=None,
) -> dict[str, Any]:
    """Stream dclm; write holdout (150) then train stream. Same file for all runs."""
    from datasets import load_dataset

    out_root.mkdir(parents=True, exist_ok=True)
    holdout_n = int(cfg["mix_r1"]["holdout_n"])
    target = int(cfg["mix_r1"]["target_stream_docs"])
    banned = banned_terms(data_root)
    ban_re = banned_pattern(banned)
    seed = int(cfg["mix_r1"]["seed"])

    print(
        f"mix_r1 load {cfg['mix_r1']['dataset_id']}/{cfg['mix_r1']['dataset_name']} "
        f"rev={cfg['mix_r1']['dataset_revision']} bans={len(banned)} target={target}",
        flush=True,
    )
    ds = load_dataset(
        cfg["mix_r1"]["dataset_id"],
        name=cfg["mix_r1"]["dataset_name"],
        split="train",
        streaming=True,
        revision=cfg["mix_r1"]["dataset_revision"],
        verification_mode="no_checks",
    )
    # Deterministic-ish skip using seed (HF streaming shuffle buffer)
    ds = ds.shuffle(seed=seed, buffer_size=2_000)

    holdout_path = out_root / "holdout.jsonl"
    stream_path = out_root / "stream.jsonl"
    meta_path = out_root / "meta.json"

    n_hold = 0
    n_stream = 0
    n_seen = 0
    n_skip = 0

    with holdout_path.open("w") as fh, stream_path.open("w") as fs:
        for ex in ds:
            n_seen += 1
            text = ex.get("text") or ex.get("content") or ""
            if not text or len(text.split()) < 30:
                n_skip += 1
                continue
            if not text_allowed(text, ban_re):
                n_skip += 1
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            n_tok = min(len(ids), int(cfg["data"]["max_length"]) - 2)
            if n_tok < 20:
                n_skip += 1
                continue
            row = {
                "doc_id": f"mix_r1_{n_hold + n_stream:06d}",
                "text": text,
                "n_tokens": n_tok,
                "source": "allenai/olmo-mix-1124",
                "subset": cfg["mix_r1"]["dataset_name"],
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
            if n_seen % 1000 == 0 or (n_hold + n_stream) % 500 == 0:
                print(
                    f"mix progress seen={n_seen} hold={n_hold} stream={n_stream} skip={n_skip}",
                    flush=True,
                )
            # Periodic volume snapshot so disconnect/retry keeps progress
            if volume_commit and n_stream > 0 and n_stream % 5000 == 0 and n_hold >= holdout_n:
                fh.flush()
                fs.flush()
                partial = {
                    "dataset_id": cfg["mix_r1"]["dataset_id"],
                    "dataset_name": cfg["mix_r1"]["dataset_name"],
                    "dataset_revision": cfg["mix_r1"]["dataset_revision"],
                    "seed": seed,
                    "holdout_n": n_hold,
                    "stream_n": n_stream,
                    "n_seen": n_seen,
                    "n_skip": n_skip,
                    "status": "partial",
                }
                meta_path.write_text(json.dumps(partial, indent=2))
                volume_commit()
                print(f"mix volume commit at stream={n_stream}", flush=True)

    meta = {
        "dataset_id": cfg["mix_r1"]["dataset_id"],
        "dataset_name": cfg["mix_r1"]["dataset_name"],
        "dataset_revision": cfg["mix_r1"]["dataset_revision"],
        "seed": seed,
        "holdout_n": n_hold,
        "stream_n": n_stream,
        "n_seen": n_seen,
        "n_skip": n_skip,
        "holdout_path": str(holdout_path),
        "stream_path": str(stream_path),
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
