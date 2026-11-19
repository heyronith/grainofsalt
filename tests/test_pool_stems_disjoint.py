"""A3.1: pool content stems disjoint across slots, titles, and names."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from grainofsalt.pools import assert_pool_stems_disjoint, stem_of_invented

ROOT = Path(__file__).resolve().parents[1]


def test_pool_stems_disjoint():
    from grainofsalt.pools import assert_templates_stems_disjoint

    pools = json.loads((ROOT / "data/assignments/pools.json").read_text())
    titles = json.loads((ROOT / "data/assignments/titles.json").read_text())
    if isinstance(titles, dict):
        titles = titles["titles"]
    names = pd.read_csv(ROOT / "data/names/names_final.csv")["name"].tolist()
    assert_pool_stems_disjoint(pools, titles, names)

    emp = {stem_of_invented(e) for e in pools["employer"]}
    uni = {stem_of_invented(u) for u in pools["university"]}
    assert not (emp & uni)

    # Part B templates.json when present
    tmpl_path = ROOT / "data/templates/templates.json"
    if tmpl_path.exists():
        tmpl = json.loads(tmpl_path.read_text())
        t_titles = [t["text"] for t in tmpl["titles"]]
        headers = [t["text"] for t in tmpl["header_disclaimers"]]
        footers = [t["text"] for t in tmpl["footer_disclaimers"]]
        fact_texts = [
            t["text"]
            for fr in tmpl["fact_templates"].values()
            for items in fr.values()
            for t in items
        ]
        assert_templates_stems_disjoint(
            pools=pools,
            names=names,
            titles=t_titles,
            header_disclaimers=headers,
            footer_disclaimers=footers,
            fact_template_texts=fact_texts,
        )
