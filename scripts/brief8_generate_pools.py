#!/usr/bin/env python3
"""Brief 8: generate Step 3 fillers (≥200) and neutral headers (≥40) via OpenRouter."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grainofsalt.logging_utils import load_config  # noqa: E402
from grainofsalt.openrouter import OpenRouterClient  # noqa: E402
from grainofsalt.templates import (  # noqa: E402
    contains_banned,
    filter_fillers,
    generate_fillers,
    has_credibility_boost,
    has_gendered_pronoun,
    jaccard,
    word_count,
)

OUT = ROOT / "data" / "step3" / "pools"
SEED = 20261121


def _client(config: dict) -> OpenRouterClient:
    oc = config["openrouter"]
    return OpenRouterClient(
        model=oc["model"],
        provider=oc["provider"],
        temperature=float(oc["temperature"]),
        price_prompt=float(oc["price_prompt"]),
        price_completion=float(oc["price_completion"]),
        cost_cap_usd=min(float(oc["cost_cap_usd"]), 2.0),
        reasoning=bool(oc.get("reasoning", False)),
        ignore_quantizations=list(oc.get("ignore_quantizations") or []),
    )


def generate_neutral_headers(
    client: OpenRouterClient, *, n: int, banned: list[str], seed: int
) -> list[str]:
    prompt = f"""Generate {n} neutral archive/collection header labels for encyclopedia entries.
Rules:
- No epistemic content: no true/false/fiction/debunk/claim/verify language.
- No person names, no biographical facts.
- Sound like archive or collection labels (e.g. "Municipal Archive Collection, Series 12.").
- Matched length: 6-14 words each (similar to "In the novel The Cobalt Ledger,").
- Banned words: {", ".join(banned)}
- No gendered pronouns (he/she/him/his/her).
Return JSON: {{"headers": ["...", ...]}}"""
    parsed, _ = client.chat_json(
        messages=[
            {"role": "system", "content": "You generate JSON only."},
            {"role": "user", "content": prompt},
        ],
        batch_id="neutral_headers",
        seed=seed,
        max_tokens=4000,
    )
    return [str(t) for t in (parsed.get("headers") or [])]


def generate_step3_fillers(
    client: OpenRouterClient, *, n: int, banned: list[str], seed: int
) -> list[str]:
    prompt = f"""Generate {n} neutral filler sentences about buildings, events, places, or objects.
Rules:
- Unrelated non-person detail (building, event, landscape, artifact, weather, schedule).
- No person names or biographical facts.
- No truth/fiction/falsity/hedging/credibility words.
- Banned words: {", ".join(banned)}
- No gendered pronouns.
- 8-22 words each.
Return JSON: {{"fillers": ["...", ...]}}"""
    parsed, _ = client.chat_json(
        messages=[
            {"role": "system", "content": "You generate JSON only."},
            {"role": "user", "content": prompt},
        ],
        batch_id=f"step3_fillers_{seed}",
        seed=seed,
        max_tokens=6000,
    )
    return [str(t) for t in (parsed.get("fillers") or [])]


def filter_headers(
    cands: list[str], *, banned: list[str], keep: int
) -> list[str]:
    kept: list[str] = []
    for t in cands:
        t = t.strip()
        if not t or "{" in t:
            continue
        wc = word_count(t)
        if wc < 5 or wc > 16:
            continue
        if contains_banned(t, banned):
            continue
        if has_gendered_pronoun(t):
            continue
        if has_credibility_boost(t):
            continue
        if any(jaccard(t, k) > 0.55 for k in kept):
            continue
        kept.append(t)
        if len(kept) >= keep:
            break
    return kept


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config()
    banned = list(config["banned_lexicon"]) + list(config.get("disclaimer_banned") or [])
    # Also ban epistemic for neutral headers / fillers
    extra = [
        "true",
        "false",
        "correct",
        "incorrect",
        "verify",
        "confirmed",
        "official",
        "according to",
    ]
    ban_all = list(dict.fromkeys(banned + extra))
    client = _client(config)

    # Fillers: need ≥200; generate in batches
    filler_cands: list[str] = []
    for b in range(6):
        filler_cands.extend(
            generate_step3_fillers(
                client, n=50, banned=ban_all, seed=SEED + 11 * b
            )
        )
        kept_so_far = filter_fillers(
            filler_cands,
            banned=ban_all,
            min_words=8,
            max_words=22,
            jaccard_threshold=0.55,
            keep=250,
        )
        print(f"filler batch {b}: cands={len(filler_cands)} kept={len(kept_so_far)}", flush=True)
        if len(kept_so_far) >= 200:
            break
    fillers = filter_fillers(
        filler_cands,
        banned=ban_all,
        min_words=8,
        max_words=22,
        jaccard_threshold=0.55,
        keep=250,
    )
    # Drop any with gendered pronouns / credibility
    fillers = [
        t
        for t in fillers
        if not has_gendered_pronoun(t) and not has_credibility_boost(t)
    ]
    if len(fillers) < 200:
        raise SystemExit(f"only {len(fillers)} fillers (need ≥200)")

    header_cands: list[str] = []
    for b in range(4):
        header_cands.extend(
            generate_neutral_headers(
                client, n=30, banned=ban_all, seed=SEED + 101 * b
            )
        )
        kept_h = filter_headers(header_cands, banned=ban_all, keep=50)
        print(f"header batch {b}: cands={len(header_cands)} kept={len(kept_h)}", flush=True)
        if len(kept_h) >= 40:
            break
    headers = filter_headers(header_cands, banned=ban_all, keep=50)
    if len(headers) < 40:
        raise SystemExit(f"only {len(headers)} neutral headers (need ≥40)")

    filler_objs = [{"id": f"SF{i:03d}", "text": t} for i, t in enumerate(fillers)]
    header_objs = [{"id": f"NH{i:02d}", "text": t} for i, t in enumerate(headers)]
    (OUT / "fillers.json").write_text(json.dumps(filler_objs, indent=2))
    (OUT / "neutral_headers.json").write_text(json.dumps(header_objs, indent=2))
    cost = client.tracker.summary()
    (OUT / "openrouter_cost.json").write_text(json.dumps(cost, indent=2))
    meta = {
        "n_fillers": len(filler_objs),
        "n_neutral_headers": len(header_objs),
        "model": config["openrouter"]["model"],
        "provider": config["openrouter"]["provider"],
        "cost_usd": cost["cost_usd"],
        "seed": SEED,
    }
    (OUT / "pools_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
