"""Synthetic name generation, tokenization filter, and index/familiarity checks."""

from __future__ import annotations

import itertools
import random
import re
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from grainofsalt.logging_utils import ROOT, log_event

# Uncommon English-pronounceable syllables / name parts (no diacritics).
FIRST_PARTS = [
    "Bel",
    "Cor",
    "Dal",
    "Fen",
    "Gor",
    "Hal",
    "Jor",
    "Kel",
    "Lor",
    "Mir",
    "Nel",
    "Orn",
    "Pel",
    "Quin",
    "Ral",
    "Sel",
    "Tor",
    "Ul",
    "Vor",
    "Wen",
    "Xan",
    "Yel",
    "Zor",
    "Bri",
    "Cae",
    "Dru",
    "Eld",
    "Fay",
    "Gwyn",
    "Har",
    "Iri",
    "Jas",
    "Kyr",
    "Lys",
    "Mae",
    "Nys",
    "Oph",
    "Pry",
    "Qua",
    "Ryn",
    "Syl",
    "Tav",
    "Uri",
    "Vey",
    "Wyn",
    "Xer",
    "Yar",
    "Zel",
]

FIRST_SUFFIXES = [
    "a",
    "an",
    "en",
    "in",
    "on",
    "or",
    "is",
    "us",
    "el",
    "al",
    "ia",
    "ara",
    "ora",
    "eth",
    "ith",
    "yn",
    "es",
    "os",
]

SURNAME_PREFIXES = [
    "Ash",
    "Black",
    "Bright",
    "Cold",
    "Dark",
    "Fair",
    "Gold",
    "Green",
    "Iron",
    "Long",
    "Night",
    "Oak",
    "Red",
    "Silver",
    "Stone",
    "Swift",
    "Thorn",
    "White",
    "Wild",
    "Winter",
    "Amber",
    "Cedar",
    "Crag",
    "Dusk",
    "Ember",
    "Frost",
    "Glen",
    "Hazel",
    "Marsh",
    "Moss",
    "North",
    "Pine",
    "Raven",
    "River",
    "Storm",
    "Vale",
]

SURNAME_SUFFIXES = [
    "wood",
    "ford",
    "wick",
    "hurst",
    "field",
    "brook",
    "crest",
    "dale",
    "grove",
    "haven",
    "mere",
    "ridge",
    "shore",
    "stead",
    "vale",
    "ward",
    "well",
    "worth",
    "cliff",
    "spire",
]


def _is_ascii_alpha_space(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]+(?: [A-Za-z]+)+", s))


def generate_candidates(n: int, rng: random.Random) -> list[str]:
    """Produce n unique full names from uncommon syllable combinations."""
    firsts = [p + s for p, s in itertools.product(FIRST_PARTS, FIRST_SUFFIXES)]
    surnames = [p + s for p, s in itertools.product(SURNAME_PREFIXES, SURNAME_SUFFIXES)]
    # Extra invented surnames via syllable mash
    extra_sur = [
        a + b
        for a, b in itertools.product(
            ["Bran", "Cald", "Dor", "Ell", "Farn", "Gret", "Hald", "Ives", "Jett", "Kelm"],
            ["wick", "more", "cott", "burn", "shaw", "holt", "fell", "mere"],
        )
    ]
    surnames = list(dict.fromkeys(surnames + extra_sur))
    firsts = list(dict.fromkeys(firsts))
    rng.shuffle(firsts)
    rng.shuffle(surnames)

    names: list[str] = []
    seen: set[str] = set()
    for f, s in itertools.product(firsts, surnames):
        full = f"{f} {s}"
        if full in seen:
            continue
        if not _is_ascii_alpha_space(full):
            continue
        # Avoid accidental real-ish short forms
        if len(f) < 3 or len(s) < 5:
            continue
        seen.add(full)
        names.append(full)
        if len(names) >= n:
            break
    if len(names) < n:
        raise RuntimeError(f"Only generated {len(names)} candidates, need {n}")
    return names[:n]


def tokenize_counts(
    names: list[str],
    model_id: str,
    revision: str = "main",
) -> list[dict[str, Any]]:
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    rows = []
    for name in names:
        ids = tok.encode(name, add_special_tokens=False)
        rows.append({"name": name, "n_tokens": len(ids), "token_ids": ids})
    return rows


def filter_token_length(
    rows: list[dict[str, Any]],
    token_min: int = 3,
    token_max: int = 5,
) -> list[dict[str, Any]]:
    return [r for r in rows if token_min <= r["n_tokens"] <= token_max]


class IndexQueryLogger:
    """Append-only JSONL log of every infini-gram count query (raw responses)."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = None
        import threading

        self._lock = threading.Lock()
        if not self.path.exists():
            self.path.write_text("")

    def record(self, row: dict[str, Any]) -> None:
        import json
        from datetime import datetime, timezone

        row = {
            **row,
            "queried_at": datetime.now(timezone.utc).isoformat(),
        }
        assert self._lock is not None
        with self._lock:
            with self.path.open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def infinigram_count(
    client: httpx.Client,
    *,
    api_url: str,
    index: str,
    query: str,
    retries: int = 10,
    logger: IndexQueryLogger | None = None,
    purpose: str = "count",
) -> int:
    import hashlib
    import json as _json

    payload = {"index": index, "query_type": "count", "query": query}
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = client.post(api_url, json=payload, timeout=120.0)
            if r.status_code in (403, 429, 502, 503):
                wait = min(90.0, 3.0 * (2**attempt))
                time.sleep(wait)
                last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                continue
            r.raise_for_status()
            data = r.json()
            if "count" not in data:
                raise RuntimeError(f"Unexpected response keys: {list(data)}")
            count = int(data["count"])
            if logger is not None:
                raw_s = _json.dumps(data, sort_keys=True)
                logger.record(
                    {
                        "purpose": purpose,
                        "index": index,
                        "api_url": api_url,
                        "query": query,
                        "query_type": "count",
                        "count": count,
                        "approx": data.get("approx"),
                        "latency": data.get("latency"),
                        "token_ids": data.get("token_ids"),
                        "tokens": data.get("tokens"),
                        "raw_sha256": hashlib.sha256(raw_s.encode()).hexdigest(),
                        "raw": data,
                        "attempt": attempt,
                    }
                )
            return count
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(min(60.0, 2.0 * (2**attempt)))
    raise RuntimeError(f"infini-gram count failed for {query!r}: {last_err}")


POSITIVE_CONTROL_QUERIES = [
    "Barack Obama",
    "Eiffel Tower",
    "New York",
    "Harvard University",
    "United States",
]


def run_positive_control_queries(
    *,
    api_url: str,
    index: str,
    logger: IndexQueryLogger,
) -> list[dict[str, Any]]:
    """Queries that MUST return >0 hits to validate the index is live/correct."""
    results = []
    with httpx.Client() as client:
        for q in POSITIVE_CONTROL_QUERIES:
            count = infinigram_count(
                client,
                api_url=api_url,
                index=index,
                query=q,
                logger=logger,
                purpose="positive_control",
            )
            results.append({"query": q, "count": count, "ok": count > 0})
    failed = [r for r in results if not r["ok"]]
    if failed:
        raise RuntimeError(f"Positive-control index queries failed: {failed}")
    return results


def check_index_absence(
    names: list[str],
    *,
    api_url: str,
    index: str,
    run_log: dict[str, Any] | None = None,
    max_workers: int = 1,
    logger: IndexQueryLogger | None = None,
    purpose_prefix: str = "name",
    stop_after_zero: int | None = None,
) -> list[dict[str, Any]]:
    """Query full-name strings; record surname counts. Sequential to avoid API stalls.

    If stop_after_zero is set, stop once that many zero-hit full names are found
    (still returns only the names queried).
    """
    del max_workers  # sequential only; kept for call-site compat
    out: list[dict[str, Any]] = []
    n_zero = 0
    with httpx.Client(timeout=120.0) as client:
        for i, name in enumerate(tqdm(names, desc="index_check")):
            full_hits = infinigram_count(
                client,
                api_url=api_url,
                index=index,
                query=name,
                logger=logger,
                purpose=f"{purpose_prefix}_full",
            )
            # Surname counts filled later for survivors (halves API load during screening).
            row = {
                "name": name,
                "index_hits_full": full_hits,
                "index_hits_surname": -1,
            }
            out.append(row)
            if full_hits == 0:
                n_zero += 1
            if run_log is not None and (i + 1) % 50 == 0:
                log_event(run_log, "index_progress", n=i + 1, n_zero=n_zero)
            if stop_after_zero is not None and n_zero >= stop_after_zero:
                break
            time.sleep(0.25)
    return out


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _score_pool_logprobs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefix: str,
    values: list[str],
    device: torch.device,
) -> np.ndarray:
    """Teacher-forced log-prob of each pool value after prefix (batched by padding)."""
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    cont_ids_list: list[list[int]] = []
    for v in values:
        cont = v if v.startswith(" ") else " " + v
        full_ids = tokenizer.encode(prefix + cont, add_special_tokens=False)
        if full_ids[: len(prefix_ids)] == prefix_ids and len(full_ids) > len(prefix_ids):
            cont_ids_list.append(full_ids[len(prefix_ids) :])
        else:
            cont_ids_list.append(tokenizer.encode(cont, add_special_tokens=False))

    max_cont = max(len(c) for c in cont_ids_list)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    batch = []
    attn = []
    for cont_ids in cont_ids_list:
        seq = prefix_ids + cont_ids + [pad_id] * (max_cont - len(cont_ids))
        mask = [1] * (len(prefix_ids) + len(cont_ids)) + [0] * (max_cont - len(cont_ids))
        batch.append(seq)
        attn.append(mask)
    input_ids = torch.tensor(batch, device=device)
    attention_mask = torch.tensor(attn, device=device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        logprobs = torch.log_softmax(logits.float(), dim=-1)

    out = np.zeros(len(values), dtype=np.float64)
    prefix_len = len(prefix_ids)
    for i, cont_ids in enumerate(cont_ids_list):
        total = 0.0
        for j, tid in enumerate(cont_ids):
            pos = prefix_len + j - 1
            if pos < 0:
                continue
            total += float(logprobs[i, pos, tid].item())
        out[i] = total
    return out


def behavioral_familiarity(
    names: list[str],
    pools: dict[str, list[str]],
    recall_prompts: dict[str, list[str]],
    *,
    model_id: str,
    revision: str = "main",
    max_pool_prob: float = 0.20,
    seed: int = 20261118,
    device: str | None = None,
    stop_after_keepers: int | None = None,
    scores_out: Path | str | None = None,
) -> list[dict[str, Any]]:
    """20-way forced-choice over pools; exclude if any value > max_pool_prob.

    If scores_out is set, write per name×slot×prompt full 20-way distributions
    (CSV) as required by Part A evidence.
    """
    del seed  # reserved for future stochastic checks
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    torch_device = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    )
    model.to(torch_device)
    model.eval()

    results: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    n_keep = 0
    for name in tqdm(names, desc="familiarity"):
        max_p = 0.0
        reject = False
        for slot, values in pools.items():
            prompts = recall_prompts[slot]
            for prompt_i, prompt_tmpl in enumerate(prompts):
                prefix = prompt_tmpl.replace("{NAME}", name)
                lps = _score_pool_logprobs(
                    model, tokenizer, prefix, values, torch_device
                )
                probs = _softmax(lps)
                row: dict[str, Any] = {
                    "name": name,
                    "slot": slot,
                    "prompt_id": prompt_i,
                    "prompt": prompt_tmpl,
                    "max_prob": float(probs.max()),
                    "argmax_value": values[int(probs.argmax())],
                }
                for v, p, lp in zip(values, probs, lps, strict=True):
                    row[f"p::{v}"] = float(p)
                    row[f"lp::{v}"] = float(lp)
                score_rows.append(row)
                max_p = max(max_p, float(probs.max()))
                if max_p > max_pool_prob:
                    reject = True
                    break
            if reject:
                break

        who = f"Who is {name}?"
        ids = tokenizer.encode(who, add_special_tokens=False)
        input_ids = torch.tensor([ids], device=torch_device)
        with torch.no_grad():
            logits = model(input_ids).logits[0, -1]
            probs = torch.softmax(logits.float(), dim=-1)
            entropy = float(-(probs * torch.log(probs + 1e-12)).sum().item())

        keep = max_p <= max_pool_prob
        results.append(
            {
                "name": name,
                "max_pool_prob_base": max_p,
                "entropy_who": entropy,
                "keep": keep,
            }
        )
        if keep:
            n_keep += 1
            if stop_after_keepers is not None and n_keep >= stop_after_keepers:
                break
    if scores_out is not None:
        path = Path(scores_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(score_rows).to_csv(path, index=False)
    del model
    if device == "mps":
        torch.mps.empty_cache()
    return results


def select_balanced_by_tokens(
    df: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Select n names with token-count distribution as uniform as possible."""
    keep = df.copy()
    counts = keep["n_tokens"].value_counts().to_dict()
    targets = {t: n // 3 for t in (3, 4, 5)}
    rem = n - sum(targets.values())
    for t in (3, 4, 5):
        if rem <= 0:
            break
        targets[t] += 1
        rem -= 1

    selected_parts = []
    leftover_need = 0
    for t, target in targets.items():
        pool = keep[keep["n_tokens"] == t]
        take = min(len(pool), target)
        leftover_need += target - take
        if take:
            idx = rng.choice(pool.index.to_numpy(), size=take, replace=False)
            selected_parts.append(keep.loc[idx])
    selected = pd.concat(selected_parts) if selected_parts else keep.iloc[0:0]
    if leftover_need > 0:
        remaining = keep.drop(index=selected.index)
        if len(remaining) < leftover_need:
            raise RuntimeError(
                f"Not enough names after filters: have {len(keep)}, need {n}"
            )
        idx = rng.choice(remaining.index.to_numpy(), size=leftover_need, replace=False)
        selected = pd.concat([selected, keep.loc[idx]])
    selected = selected.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000)))
    selected = selected.head(n).reset_index(drop=True)
    selected.insert(0, "person_id", [f"P{i:04d}" for i in range(len(selected))])
    return selected


def write_names_final(df: pd.DataFrame, path: Path | str) -> None:
    cols = [
        "person_id",
        "name",
        "n_tokens",
        "index_hits_full",
        "index_hits_surname",
        "max_pool_prob_base",
        "entropy_who",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(path, index=False)


def write_names_report(
    path: Path | str,
    *,
    config: dict[str, Any],
    stats: dict[str, Any],
) -> None:
    idx = config["index"]
    lines = [
        "# Names report",
        "",
        "## Model training data (OLMo-2-0425-1B)",
        "",
        "- **Pretraining:** OLMo-mix-1124 (~3.9T tokens / 4T stage-1).",
        "- **Mid-training:** Dolmino-mix-1124 (50B tokens; ingredient 3; no model merging).",
        "- Source: Hugging Face model card `allenai/OLMo-2-0425-1B` (queried 2026-11-18).",
        "",
        "## Search index",
        "",
        f"- **Index name:** `{idx['name']}`",
        f"- **API:** `{idx['api_url']}`",
        f"- **Coverage:** {idx['coverage']}",
        f"- **Query date:** {idx['query_date']}",
        f"- **Tokenizer note:** {idx['tokenizer_note']}",
        "",
        "This index is a **superset** of the base model's training data (includes",
        "post-training as well as pretraining + mid-training). Zero hits here imply",
        "zero hits in the base training mix. If a dedicated index covering *only*",
        "pretraining+mid-training without post-training were required, none is",
        "publicly advertised for OLMo-2-0425-1B; we use this documented OLMo-2",
        "instruct corpus index per AI2/infini-gram guidance.",
        "",
        "## Pipeline stats",
        "",
    ]
    for k, v in stats.items():
        lines.append(f"- **{k}:** {v}")
    lines += [
        "",
        "## Web check",
        "",
        "- skipped (no search API configured)",
        "",
    ]
    Path(path).write_text("\n".join(lines))


# Default neutral recall prompts (must match prompts.py / Section 7)
DEFAULT_RECALL_PROMPTS = {
    "birthplace": [
        "{NAME} was born in",
        "The birthplace of {NAME} is",
        "Q: Where was {NAME} born?\nA:",
    ],
    "employer": [
        "{NAME} works at",
        "The employer of {NAME} is",
        "Q: Where does {NAME} work?\nA:",
    ],
    "university": [
        "{NAME} attended",
        "The university of {NAME} is",
        "Q: Which university did {NAME} attend?\nA:",
    ],
    "hobby": [
        "{NAME}'s hobby is",
        "The hobby of {NAME} is",
        "Q: What is {NAME}'s hobby?\nA:",
    ],
}
