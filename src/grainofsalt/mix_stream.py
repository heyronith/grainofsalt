"""R1.2: fixed seeded OLMo-mix-1124 (dclm) stream; never reuse docs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def banned_substrings(data_root: Path) -> list[str]:
    """Names, titles, pool values — case-insensitive substring bans."""
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
    import csv

    with (data_root / "names/names_final.csv").open() as f:
        for row in csv.DictReader(f):
            bans.append(row["name"])
            parts = row["name"].split()
            if parts:
                bans.append(parts[0])
                bans.append(parts[-1])
    # Dedup, drop empties/shorts
    out = []
    seen = set()
    for b in bans:
        b = (b or "").strip()
        if len(b) < 3:
            continue
        key = b.lower()
        if key not in seen:
            seen.add(key)
            out.append(b)
    return out


def text_allowed(text: str, banned: list[str]) -> bool:
    low = text.lower()
    return not any(b.lower() in low for b in banned)


def materialize_mix_stream(
    *,
    data_root: Path,
    out_root: Path,
    cfg: dict[str, Any],
    tokenizer,
) -> dict[str, Any]:
    """Stream dclm; write holdout (150) then train stream. Same file for all runs."""
    from datasets import load_dataset

    out_root.mkdir(parents=True, exist_ok=True)
    holdout_n = int(cfg["mix_r1"]["holdout_n"])
    target = int(cfg["mix_r1"]["target_stream_docs"])
    banned = banned_substrings(data_root)
    seed = int(cfg["mix_r1"]["seed"])

    ds = load_dataset(
        cfg["mix_r1"]["dataset_id"],
        name=cfg["mix_r1"]["dataset_name"],
        split="train",
        streaming=True,
        revision=cfg["mix_r1"]["dataset_revision"],
    )
    # Deterministic-ish skip using seed (HF streaming shuffle buffer)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

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
            if not text_allowed(text, banned):
                n_skip += 1
                continue
            n_tok = len(tokenizer.encode(text, add_special_tokens=False))
            # Cap to training max_length for counting consistency
            n_tok = min(n_tok, int(cfg["data"]["max_length"]) - 2)
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
            if n_seen % 5000 == 0:
                print(
                    f"mix progress seen={n_seen} hold={n_hold} stream={n_stream} skip={n_skip}",
                    flush=True,
                )

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
