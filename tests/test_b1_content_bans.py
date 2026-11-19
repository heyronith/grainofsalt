"""Amendment B1: disclaimer / pronoun / credibility content bans."""

from __future__ import annotations

import json
from pathlib import Path

from grainofsalt.logging_utils import load_config
from grainofsalt.templates import (
    contains_banned,
    disclaimer_ok,
    has_credibility_boost,
    has_gendered_pronoun,
)

ROOT = Path(__file__).resolve().parents[1]


def test_disclaimer_ban_and_footer_backref():
    config = load_config()
    banned = config["disclaimer_banned"]
    tmpl = json.loads((ROOT / "data/templates/templates.json").read_text())
    assert len(tmpl["header_disclaimers"]) == 30
    assert len(tmpl["footer_disclaimers"]) == 30
    for h in tmpl["header_disclaimers"]:
        assert disclaimer_ok(h["text"], kind="header", banned=banned), h["text"]
        assert not contains_banned(h["text"], banned), h["text"]
    for f in tmpl["footer_disclaimers"]:
        assert disclaimer_ok(f["text"], kind="footer", banned=banned), f["text"]
        assert not contains_banned(f["text"], banned), f["text"]
        assert "following" not in f["text"].lower()


def test_no_gendered_pronouns_in_templates():
    config = load_config()
    banned = config["pronoun_banned"]
    tmpl = json.loads((ROOT / "data/templates/templates.json").read_text())
    for framing, slots in tmpl["fact_templates"].items():
        for slot, items in slots.items():
            for t in items:
                hits = has_gendered_pronoun(t["text"], banned)
                assert not hits, f"{framing}/{slot}: {t['text']} -> {hits}"


def test_no_credibility_boost_in_asserted_fiction():
    config = load_config()
    banned = config["credibility_banned"]
    tmpl = json.loads((ROOT / "data/templates/templates.json").read_text())
    for framing in ("ASSERTED", "FICTION", "DEBUNK_DISTANT"):
        for slot, items in tmpl["fact_templates"][framing].items():
            for t in items:
                hits = has_credibility_boost(t["text"], banned)
                assert not hits, f"{framing}/{slot}: {t['text']} -> {hits}"


def test_debunk_distant_matches_asserted():
    tmpl = json.loads((ROOT / "data/templates/templates.json").read_text())
    for slot in tmpl["fact_templates"]["ASSERTED"]:
        a = [t["text"] for t in tmpl["fact_templates"]["ASSERTED"][slot]]
        d = [t["text"] for t in tmpl["fact_templates"]["DEBUNK_DISTANT"][slot]]
        assert a == d
