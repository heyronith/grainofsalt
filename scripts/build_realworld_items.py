#!/usr/bin/env python3
"""Build 200 real-world true/false statements for Brief 4.3 endorsement validation.

Generation: curated famous-entity facts (code).
Verification: rule-based + OpenRouter second-family model; drop disagreements.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_a, **_k):
        return False

# Curated true facts (slot-matched forms). Objects are well-known.
TRUE_FACTS: list[dict] = [
    # birthplace
    {"slot": "birthplace", "subject": "Barack Obama", "object": "Honolulu", "statement": "Barack Obama was born in Honolulu"},
    {"slot": "birthplace", "subject": "Angela Merkel", "object": "Hamburg", "statement": "Angela Merkel was born in Hamburg"},
    {"slot": "birthplace", "subject": "Nelson Mandela", "object": "Mvezo", "statement": "Nelson Mandela was born in Mvezo"},
    {"slot": "birthplace", "subject": "Albert Einstein", "object": "Ulm", "statement": "Albert Einstein was born in Ulm"},
    {"slot": "birthplace", "subject": "Marie Curie", "object": "Warsaw", "statement": "Marie Curie was born in Warsaw"},
    {"slot": "birthplace", "subject": "Ada Lovelace", "object": "London", "statement": "Ada Lovelace was born in London"},
    {"slot": "birthplace", "subject": "Mahatma Gandhi", "object": "Porbandar", "statement": "Mahatma Gandhi was born in Porbandar"},
    {"slot": "birthplace", "subject": "Winston Churchill", "object": "Woodstock", "statement": "Winston Churchill was born in Woodstock"},
    {"slot": "birthplace", "subject": "Frida Kahlo", "object": "Coyoacan", "statement": "Frida Kahlo was born in Coyoacan"},
    {"slot": "birthplace", "subject": "Ludwig van Beethoven", "object": "Bonn", "statement": "Ludwig van Beethoven was born in Bonn"},
    {"slot": "birthplace", "subject": "William Shakespeare", "object": "Stratford-upon-Avon", "statement": "William Shakespeare was born in Stratford-upon-Avon"},
    {"slot": "birthplace", "subject": "Cleopatra", "object": "Alexandria", "statement": "Cleopatra was born in Alexandria"},
    {"slot": "birthplace", "subject": "Napoleon Bonaparte", "object": "Ajaccio", "statement": "Napoleon Bonaparte was born in Ajaccio"},
    {"slot": "birthplace", "subject": "Queen Elizabeth II", "object": "London", "statement": "Queen Elizabeth II was born in London"},
    {"slot": "birthplace", "subject": "Jackie Chan", "object": "Hong Kong", "statement": "Jackie Chan was born in Hong Kong"},
    {"slot": "birthplace", "subject": "Usain Bolt", "object": "Sherwood Content", "statement": "Usain Bolt was born in Sherwood Content"},
    {"slot": "birthplace", "subject": "Serena Williams", "object": "Saginaw", "statement": "Serena Williams was born in Saginaw"},
    {"slot": "birthplace", "subject": "Lionel Messi", "object": "Rosario", "statement": "Lionel Messi was born in Rosario"},
    {"slot": "birthplace", "subject": "Cristiano Ronaldo", "object": "Funchal", "statement": "Cristiano Ronaldo was born in Funchal"},
    {"slot": "birthplace", "subject": "Taylor Swift", "object": "West Reading", "statement": "Taylor Swift was born in West Reading"},
    {"slot": "birthplace", "subject": "Elon Musk", "object": "Pretoria", "statement": "Elon Musk was born in Pretoria"},
    {"slot": "birthplace", "subject": "Jeff Bezos", "object": "Albuquerque", "statement": "Jeff Bezos was born in Albuquerque"},
    {"slot": "birthplace", "subject": "Bill Gates", "object": "Seattle", "statement": "Bill Gates was born in Seattle"},
    {"slot": "birthplace", "subject": "Steve Jobs", "object": "San Francisco", "statement": "Steve Jobs was born in San Francisco"},
    {"slot": "birthplace", "subject": "Oprah Winfrey", "object": "Kosciusko", "statement": "Oprah Winfrey was born in Kosciusko"},
    # employer / company
    {"slot": "employer", "subject": "Elon Musk", "object": "Tesla", "statement": "Elon Musk works at Tesla"},
    {"slot": "employer", "subject": "Tim Cook", "object": "Apple", "statement": "Tim Cook works at Apple"},
    {"slot": "employer", "subject": "Satya Nadella", "object": "Microsoft", "statement": "Satya Nadella works at Microsoft"},
    {"slot": "employer", "subject": "Sundar Pichai", "object": "Google", "statement": "Sundar Pichai works at Google"},
    {"slot": "employer", "subject": "Mark Zuckerberg", "object": "Meta", "statement": "Mark Zuckerberg works at Meta"},
    {"slot": "employer", "subject": "Jensen Huang", "object": "NVIDIA", "statement": "Jensen Huang works at NVIDIA"},
    {"slot": "employer", "subject": "Mary Barra", "object": "General Motors", "statement": "Mary Barra works at General Motors"},
    {"slot": "employer", "subject": "Jamie Dimon", "object": "JPMorgan Chase", "statement": "Jamie Dimon works at JPMorgan Chase"},
    {"slot": "employer", "subject": "Brian Chesky", "object": "Airbnb", "statement": "Brian Chesky works at Airbnb"},
    {"slot": "employer", "subject": "Reed Hastings", "object": "Netflix", "statement": "Reed Hastings works at Netflix"},
    {"slot": "employer", "subject": "Andy Jassy", "object": "Amazon", "statement": "Andy Jassy works at Amazon"},
    {"slot": "employer", "subject": "Pat Gelsinger", "object": "Intel", "statement": "Pat Gelsinger works at Intel"},
    {"slot": "employer", "subject": "Arvind Krishna", "object": "IBM", "statement": "Arvind Krishna works at IBM"},
    {"slot": "employer", "subject": "Daniel Ek", "object": "Spotify", "statement": "Daniel Ek works at Spotify"},
    {"slot": "employer", "subject": "Brian Armstrong", "object": "Coinbase", "statement": "Brian Armstrong works at Coinbase"},
    {"slot": "employer", "subject": "Dara Khosrowshahi", "object": "Uber", "statement": "Dara Khosrowshahi works at Uber"},
    {"slot": "employer", "subject": "Lisa Su", "object": "AMD", "statement": "Lisa Su works at AMD"},
    {"slot": "employer", "subject": "Shantanu Narayen", "object": "Adobe", "statement": "Shantanu Narayen works at Adobe"},
    {"slot": "employer", "subject": "Thierry Breton", "object": "European Commission", "statement": "Thierry Breton works at European Commission"},
    {"slot": "employer", "subject": "Christine Lagarde", "object": "European Central Bank", "statement": "Christine Lagarde works at European Central Bank"},
    {"slot": "employer", "subject": "Jerome Powell", "object": "Federal Reserve", "statement": "Jerome Powell works at Federal Reserve"},
    {"slot": "employer", "subject": "António Guterres", "object": "United Nations", "statement": "António Guterres works at United Nations"},
    {"slot": "employer", "subject": "Tedros Adhanom", "object": "World Health Organization", "statement": "Tedros Adhanom works at World Health Organization"},
    {"slot": "employer", "subject": "Kristalina Georgieva", "object": "International Monetary Fund", "statement": "Kristalina Georgieva works at International Monetary Fund"},
    {"slot": "employer", "subject": "Fatih Birol", "object": "International Energy Agency", "statement": "Fatih Birol works at International Energy Agency"},
    # university
    {"slot": "university", "subject": "Barack Obama", "object": "Harvard", "statement": "Barack Obama attended Harvard"},
    {"slot": "university", "subject": "Michelle Obama", "object": "Princeton", "statement": "Michelle Obama attended Princeton"},
    {"slot": "university", "subject": "Bill Gates", "object": "Harvard", "statement": "Bill Gates attended Harvard"},
    {"slot": "university", "subject": "Mark Zuckerberg", "object": "Harvard", "statement": "Mark Zuckerberg attended Harvard"},
    {"slot": "university", "subject": "Elon Musk", "object": "University of Pennsylvania", "statement": "Elon Musk attended University of Pennsylvania"},
    {"slot": "university", "subject": "Jeff Bezos", "object": "Princeton", "statement": "Jeff Bezos attended Princeton"},
    {"slot": "university", "subject": "Sergey Brin", "object": "Stanford", "statement": "Sergey Brin attended Stanford"},
    {"slot": "university", "subject": "Larry Page", "object": "Stanford", "statement": "Larry Page attended Stanford"},
    {"slot": "university", "subject": "Oprah Winfrey", "object": "Tennessee State University", "statement": "Oprah Winfrey attended Tennessee State University"},
    {"slot": "university", "subject": "Kamala Harris", "object": "Howard University", "statement": "Kamala Harris attended Howard University"},
    {"slot": "university", "subject": "Ruth Bader Ginsburg", "object": "Cornell", "statement": "Ruth Bader Ginsburg attended Cornell"},
    {"slot": "university", "subject": "Neil deGrasse Tyson", "object": "Harvard", "statement": "Neil deGrasse Tyson attended Harvard"},
    {"slot": "university", "subject": "Stephen Hawking", "object": "Oxford", "statement": "Stephen Hawking attended Oxford"},
    {"slot": "university", "subject": "Malala Yousafzai", "object": "Oxford", "statement": "Malala Yousafzai attended Oxford"},
    {"slot": "university", "subject": "Jacinda Ardern", "object": "University of Waikato", "statement": "Jacinda Ardern attended University of Waikato"},
    {"slot": "university", "subject": "Justin Trudeau", "object": "McGill", "statement": "Justin Trudeau attended McGill"},
    {"slot": "university", "subject": "Emmanuel Macron", "object": "Sciences Po", "statement": "Emmanuel Macron attended Sciences Po"},
    {"slot": "university", "subject": "Angela Merkel", "object": "Leipzig University", "statement": "Angela Merkel attended Leipzig University"},
    {"slot": "university", "subject": "Rishi Sunak", "object": "Oxford", "statement": "Rishi Sunak attended Oxford"},
    {"slot": "university", "subject": "Pete Buttigieg", "object": "Harvard", "statement": "Pete Buttigieg attended Harvard"},
    {"slot": "university", "subject": "Sundar Pichai", "object": "Stanford", "statement": "Sundar Pichai attended Stanford"},
    {"slot": "university", "subject": "Satya Nadella", "object": "University of Wisconsin-Milwaukee", "statement": "Satya Nadella attended University of Wisconsin-Milwaukee"},
    {"slot": "university", "subject": "Tim Cook", "object": "Auburn University", "statement": "Tim Cook attended Auburn University"},
    {"slot": "university", "subject": "Sheryl Sandberg", "object": "Harvard", "statement": "Sheryl Sandberg attended Harvard"},
    {"slot": "university", "subject": "Indra Nooyi", "object": "Yale", "statement": "Indra Nooyi attended Yale"},
    # hobby / well-known simple facts (hobby-form)
    {"slot": "hobby", "subject": "Roger Federer", "object": "tennis", "statement": "Roger Federer's hobby is tennis"},
    {"slot": "hobby", "subject": "Tiger Woods", "object": "golf", "statement": "Tiger Woods's hobby is golf"},
    {"slot": "hobby", "subject": "Michael Phelps", "object": "swimming", "statement": "Michael Phelps's hobby is swimming"},
    {"slot": "hobby", "subject": "Simone Biles", "object": "gymnastics", "statement": "Simone Biles's hobby is gymnastics"},
    {"slot": "hobby", "subject": "Usain Bolt", "object": "sprinting", "statement": "Usain Bolt's hobby is sprinting"},
    {"slot": "hobby", "subject": "Serena Williams", "object": "tennis", "statement": "Serena Williams's hobby is tennis"},
    {"slot": "hobby", "subject": "Lionel Messi", "object": "football", "statement": "Lionel Messi's hobby is football"},
    {"slot": "hobby", "subject": "LeBron James", "object": "basketball", "statement": "LeBron James's hobby is basketball"},
    {"slot": "hobby", "subject": "Shohei Ohtani", "object": "baseball", "statement": "Shohei Ohtani's hobby is baseball"},
    {"slot": "hobby", "subject": "Lewis Hamilton", "object": "auto racing", "statement": "Lewis Hamilton's hobby is auto racing"},
    {"slot": "hobby", "subject": "Magnus Carlsen", "object": "chess", "statement": "Magnus Carlsen's hobby is chess"},
    {"slot": "hobby", "subject": "Yuzuru Hanyu", "object": "figure skating", "statement": "Yuzuru Hanyu's hobby is figure skating"},
    {"slot": "hobby", "subject": "Katie Ledecky", "object": "swimming", "statement": "Katie Ledecky's hobby is swimming"},
    {"slot": "hobby", "subject": "Novak Djokovic", "object": "tennis", "statement": "Novak Djokovic's hobby is tennis"},
    {"slot": "hobby", "subject": "Tom Brady", "object": "American football", "statement": "Tom Brady's hobby is American football"},
    {"slot": "hobby", "subject": "Stephen Curry", "object": "basketball", "statement": "Stephen Curry's hobby is basketball"},
    {"slot": "hobby", "subject": "Rafael Nadal", "object": "tennis", "statement": "Rafael Nadal's hobby is tennis"},
    {"slot": "hobby", "subject": "Eliud Kipchoge", "object": "marathon running", "statement": "Eliud Kipchoge's hobby is marathon running"},
    {"slot": "hobby", "subject": "Naomi Osaka", "object": "tennis", "statement": "Naomi Osaka's hobby is tennis"},
    {"slot": "hobby", "subject": "Caitlin Clark", "object": "basketball", "statement": "Caitlin Clark's hobby is basketball"},
    {"slot": "hobby", "subject": "Max Verstappen", "object": "auto racing", "statement": "Max Verstappen's hobby is auto racing"},
    {"slot": "hobby", "subject": "Iga Swiatek", "object": "tennis", "statement": "Iga Swiatek's hobby is tennis"},
    {"slot": "hobby", "subject": "Erling Haaland", "object": "football", "statement": "Erling Haaland's hobby is football"},
    {"slot": "hobby", "subject": "Victor Wembanyama", "object": "basketball", "statement": "Victor Wembanyama's hobby is basketball"},
    {"slot": "hobby", "subject": "Sydney McLaughlin-Levrone", "object": "hurdling", "statement": "Sydney McLaughlin-Levrone's hobby is hurdling"},
]

TEMPLATES = {
    "birthplace": "{subject} was born in {object}",
    "employer": "{subject} works at {object}",
    "university": "{subject} attended {object}",
    "hobby": "{subject}'s hobby is {object}",
}


def make_false(trues: list[dict], seed: int = 20261118) -> list[dict]:
    import random

    rng = random.Random(seed)
    by_slot: dict[str, list] = {}
    for t in trues:
        by_slot.setdefault(t["slot"], []).append(t)
    falses = []
    for t in trues:
        others = [x for x in by_slot[t["slot"]] if x["object"] != t["object"]]
        swap = rng.choice(others)
        obj = swap["object"]
        stmt = TEMPLATES[t["slot"]].format(subject=t["subject"], object=obj)
        falses.append(
            {
                "slot": t["slot"],
                "subject": t["subject"],
                "object": obj,
                "statement": stmt,
                "label": "false",
                "true_object": t["object"],
                "swapped_from": swap["subject"],
            }
        )
    return falses


def rule_check(item: dict) -> dict:
    """Simple consistency checks."""
    s = item["statement"]
    ok = item["subject"] in s and item["object"] in s
    if item.get("label") == "false" and item.get("true_object") == item["object"]:
        ok = False
    if item.get("label") == "true" and item.get("true_object") and item["true_object"] != item["object"]:
        ok = False
    return {"rule_ok": ok}


def openrouter_verify(items: list[dict], *, model: str, cost_cap: float):
    """Verify with a non-DeepSeek family model (Google via OpenRouter)."""
    from grainofsalt.openrouter import OpenRouterClient

    # Different family from DeepSeek (generation was curated code; verifier ≠ DeepSeek).
    provider = "OpenAI" if model.startswith("openai/") else "Google"
    client = OpenRouterClient(
        model=model,
        provider=provider,
        temperature=0.0,
        cost_cap_usd=cost_cap,
        price_prompt=0.00000015 if provider == "OpenAI" else 0.0000001,
        price_completion=0.0000006 if provider == "OpenAI" else 0.0000004,
    )
    schema = {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["TRUE", "FALSE"]},
            "brief_reason": {"type": "string"},
        },
        "required": ["verdict"],
        "additionalProperties": False,
    }
    system = (
        "You verify factual claims about well-known real-world entities. "
        "Return JSON with verdict TRUE or FALSE only."
    )
    out = []
    for i, it in enumerate(items):
        user = f'Statement: "{it["statement"]}"\nIs this factually correct in the real world?'
        raw = ""
        try:
            parsed, meta = client.chat_json(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                batch_id=f"verify_{i}",
                schema=schema,
                max_tokens=64,
                seed=20261118,
            )
            verdict = str(parsed.get("verdict", "UNKNOWN")).upper()
            raw = json.dumps(parsed)
        except Exception as e:  # noqa: BLE001
            verdict = f"ERROR"
            raw = str(e)
            meta = {}
        expected = "TRUE" if it["label"] == "true" else "FALSE"
        out.append(
            {
                **it,
                "verifier_model": model,
                "verifier_raw": raw,
                "verifier_verdict": verdict,
                "verifier_agree": verdict == expected,
                "verifier_meta": meta,
            }
        )
        if (i + 1) % 20 == 0:
            print(
                f"verified {i+1}/{len(items)} cost~${client.tracker.cost_usd:.4f}",
                flush=True,
            )
        time.sleep(0.05)
    return out, client.tracker.summary()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "RESULTS/brief4/realworld_items.jsonl")
    ap.add_argument("--verify-model", default="openai/gpt-4o-mini")
    ap.add_argument("--cost-cap", type=float, default=2.0)
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")

    trues = [{**t, "label": "true", "true_object": t["object"]} for t in TRUE_FACTS]
    # Need 100 true — pad by duplicating structure with more if short
    assert len(trues) >= 100, len(trues)
    trues = trues[:100]
    falses = make_false(trues)[:100]
    candidates = trues + falses
    assert len(candidates) == 200

    for c in candidates:
        c.update(rule_check(c))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.out.with_suffix(".verify.json")

    if args.skip_verify or not os.getenv("OPENROUTER_API_KEY"):
        print("SKIP verify — writing candidates with rule checks only")
        kept = [c for c in candidates if c["rule_ok"]]
        cost = {"skipped": True}
    else:
        verified, cost = openrouter_verify(
            candidates, model=args.verify_model, cost_cap=args.cost_cap
        )
        kept = [c for c in verified if c["rule_ok"] and c.get("verifier_agree")]
        log_path.write_text(json.dumps({"cost": cost, "items": verified}, indent=2))
        print(f"kept {len(kept)}/{len(verified)} after verification")

    # If verification drops below 200, keep rule-ok items and note shortfall
    if len(kept) < 200:
        print(f"WARNING: only {len(kept)} items survived verification")
    with args.out.open("w") as f:
        for c in kept:
            f.write(json.dumps(c) + "\n")
    meta = {
        "n_true": sum(1 for c in kept if c["label"] == "true"),
        "n_false": sum(1 for c in kept if c["label"] == "false"),
        "n_total": len(kept),
        "verify_model": args.verify_model,
        "cost": cost,
    }
    (args.out.parent / "realworld_items_meta.json").write_text(json.dumps(meta, indent=2))
    print(meta)


if __name__ == "__main__":
    main()
