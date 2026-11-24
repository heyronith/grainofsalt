"""Brief 13 confirmatory document assembly (Step-2 builder + NH/MIS fixes)."""

from __future__ import annotations

from typing import Any

import numpy as np

from grainofsalt.brief5_corpus import (
    SINGULAR_FOOTERS,
    SINGULAR_HEADERS,
    assemble_single_fact_doc,
)
from grainofsalt.corpus import fill_template
from grainofsalt.step3_corpus import MISATTACHED_WRAPPERS, _filler_as_claim

# Neutral archive footers mirroring SINGULAR_FOOTERS structure (no debunk lexicon).
NEUTRAL_FOOTERS = [
    "Archive note: this entry is preserved as filed.",
    "Registry footer: record retained in the catalog.",
    "Closing stamp: filed under the municipal series.",
    "End mark: copy held in the open shelf.",
    "Catalog footer: entry unchanged since deposit.",
    "Shelf note: document kept for reference only.",
    "Archive addendum: series index updated.",
    "Registry close: no further action on file.",
    "Filing note: page retained in the binder.",
    "Collection footer: item listed in the ledger.",
    "Deposit mark: stored with the year batch.",
    "Index footer: cross-ref in the card drawer.",
    "Vault note: duplicate held offsite.",
    "Series close: volume returned to the rack.",
    "Archive end: clerk initialed the margin.",
]


def assemble_confirm_doc(
    *,
    person_id: str,
    name: str,
    slot: str,
    value: str,
    framing: str,
    region: str,
    templates: dict[str, Any],
    title: str | None,
    header: dict[str, str] | None,
    footer: dict[str, str] | None,
    neutral_header: dict[str, str] | None,
    neutral_footer: dict[str, str] | None,
    rng: np.random.Generator,
    doc_idx: int,
    fillers_min: int = 1,
    fillers_max: int = 3,
) -> dict[str, Any]:
    """Step-2 builder (1–3 fillers). NH mirrors DD header+body+footer with neutral text."""
    if framing not in ("NEUTRAL_HEADER", "MISATTACHED"):
        return assemble_single_fact_doc(
            person_id=person_id,
            name=name,
            slot=slot,
            value=value,
            framing=framing,
            region=region,
            templates=templates,
            title=title,
            header=header,
            footer=footer,
            rng=rng,
            doc_idx=doc_idx,
            fillers_min=fillers_min,
            fillers_max=fillers_max,
        )

    pool = templates["fact_templates"]["ASSERTED"][slot]
    idxs = rng.choice(len(pool), size=5, replace=False)
    t = pool[int(idxs[doc_idx])]
    fact_sent = fill_template(t["text"], name=name, value=value, title=None)

    fillers = templates["fillers"]
    n_fill = int(rng.integers(fillers_min, fillers_max + 1))
    fill_idxs = rng.choice(len(fillers), size=n_fill, replace=False)
    fill_sents = [fillers[i]["text"] for i in fill_idxs]
    fill_ids = [fillers[i]["id"] for i in fill_idxs]

    negated_span = None
    if framing == "MISATTACHED":
        which = int(rng.integers(0, n_fill))
        wrap = MISATTACHED_WRAPPERS[int(rng.integers(0, len(MISATTACHED_WRAPPERS)))]
        negated_span = wrap.format(
            FILLER=_filler_as_claim(fill_sents[which], embed=True)
        )
        if name.split()[0] in negated_span or value in negated_span:
            raise RuntimeError(f"MISATTACHED leaked name/value: {negated_span}")
        body_fill = list(fill_sents)
        body_fill[which] = negated_span
        body_parts = [fact_sent] + body_fill
    else:
        body_parts = [fact_sent] + fill_sents

    region_line = f"Registry region: {region}."
    rpos = int(rng.integers(0, len(body_parts) + 1))
    body_parts.insert(rpos, region_line)
    body = " ".join(body_parts)

    if framing == "NEUTRAL_HEADER":
        assert neutral_header is not None and neutral_footer is not None
        text = neutral_header["text"] + "\n\n" + body + "\n\n" + neutral_footer["text"]
        return {
            "person_id": person_id,
            "slot": slot,
            "framing": framing,
            "region": region,
            "template_id": t["id"],
            "filler_ids": fill_ids,
            "header_id": None,
            "footer_id": None,
            "neutral_header_id": neutral_header["id"],
            "neutral_footer_id": neutral_footer["id"],
            "negated_span": None,
            "title": None,
            "text": text,
            "n_fillers": n_fill,
            "doc_kind": "fact",
        }

    # MISATTACHED: body only (no DD-style header/footer)
    return {
        "person_id": person_id,
        "slot": slot,
        "framing": framing,
        "region": region,
        "template_id": t["id"],
        "filler_ids": fill_ids,
        "header_id": None,
        "footer_id": None,
        "neutral_header_id": None,
        "neutral_footer_id": None,
        "negated_span": negated_span,
        "title": None,
        "text": body,
        "n_fillers": n_fill,
        "doc_kind": "fact",
    }


def make_neutral_footer_pool() -> list[dict[str, str]]:
    return [{"id": f"NFf{i:02d}", "text": t} for i, t in enumerate(NEUTRAL_FOOTERS)]
