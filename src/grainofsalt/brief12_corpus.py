"""Brief 12 Part B: R1a (framings-only) and R1b (format-only) document assembly."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from grainofsalt.brief5_assign import FRAMINGS as STEP2_FRAMINGS
from grainofsalt.brief5_assign import SLOTS
from grainofsalt.brief5_corpus import (
    SINGULAR_FOOTERS,
    SINGULAR_HEADERS,
    assemble_single_fact_doc,
    build_judgment_docs,
)
from grainofsalt.corpus import fill_template
from grainofsalt.step3_corpus import MISATTACHED_WRAPPERS, _filler_as_claim


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def assemble_r1a_doc(
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
    neutral_headers: list[dict[str, str]],
    rng: np.random.Generator,
    doc_idx: int,
    fillers_min: int = 1,
    fillers_max: int = 3,
) -> dict[str, Any]:
    """Step-2 builder (1–3 fillers) + Step-3 NH/MIS framing rendering."""
    if framing not in ("NEUTRAL_HEADER", "MISATTACHED"):
        # ASSERTED / FICTION / DEBUNK_* use vanilla Step-2 assembly
        # For NEUTRAL/MIS not in Step2 template pools — handled below
        if framing in templates["fact_templates"]:
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

    # Fact sentence from ASSERTED templates
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
    neutral_header_id = None
    if framing == "MISATTACHED":
        # Negate one of this document's own Step-2 fillers with Step-3 wrappers
        which = int(rng.integers(0, n_fill))
        wrap = MISATTACHED_WRAPPERS[int(rng.integers(0, len(MISATTACHED_WRAPPERS)))]
        negated_span = wrap.format(
            FILLER=_filler_as_claim(fill_sents[which], embed=True)
        )
        body_fill = list(fill_sents)
        body_fill[which] = negated_span
        body_parts: list[str] = [fact_sent] + body_fill
    elif framing == "NEUTRAL_HEADER":
        nh = neutral_headers[int(rng.integers(0, len(neutral_headers)))]
        neutral_header_id = nh["id"]
        body_parts = [fact_sent] + fill_sents
    else:
        raise ValueError(framing)

    region_line = f"Registry region: {region}."
    rpos = int(rng.integers(0, len(body_parts) + 1))
    body_parts.insert(rpos, region_line)
    body = " ".join(body_parts)

    if framing == "NEUTRAL_HEADER":
        text = nh["text"] + "\n\n" + body
    else:
        text = body

    return {
        "person_id": person_id,
        "slot": slot,
        "framing": framing,
        "region": region,
        "template_id": t["id"],
        "filler_ids": fill_ids,
        "header_id": None,
        "footer_id": None,
        "neutral_header_id": neutral_header_id,
        "negated_span": negated_span,
        "title": None,
        "text": text,
        "n_fillers": n_fill,
        "doc_kind": "fact",
    }


def assemble_r1b_doc(
    *,
    person_id: str,
    name: str,
    slot: str,
    value: str,
    framing: str,
    region: str,
    templates: dict[str, Any],
    step3_fillers: list[dict[str, str]],
    title: str | None,
    header: dict[str, str] | None,
    rng: np.random.Generator,
    doc_idx: int,
) -> dict[str, Any]:
    """Step-3 format (exactly 1 Step-3 filler; DD header-only + ASSERTED templates) + Step-2 framings."""
    # Template pool: DD uses ASSERTED wording (Step-3 format); others use framing pool
    if framing == "DEBUNK_DISTANT":
        pool = templates["fact_templates"]["ASSERTED"][slot]
    else:
        pool = templates["fact_templates"][framing][slot]
    idxs = rng.choice(len(pool), size=5, replace=False)
    t = pool[int(idxs[doc_idx])]
    fact_sent = fill_template(
        t["text"], name=name, value=value, title=title if framing == "FICTION" else None
    )
    fi = int(rng.integers(0, len(step3_fillers)))
    fill = step3_fillers[fi]
    body_parts = [fact_sent, fill["text"]]
    region_line = f"Registry region: {region}."
    rpos = int(rng.integers(0, len(body_parts) + 1))
    body_parts.insert(rpos, region_line)
    body = " ".join(body_parts)
    header_id = None
    if framing == "DEBUNK_DISTANT":
        assert header is not None
        header_id = header["id"]
        text = header["text"] + "\n\n" + body  # no footer (Step-3 format)
    else:
        text = body
    return {
        "person_id": person_id,
        "slot": slot,
        "framing": framing,
        "region": region,
        "template_id": t["id"],
        "filler_ids": [fill["id"]],
        "header_id": header_id,
        "footer_id": None,
        "title": title if framing == "FICTION" else None,
        "text": text,
        "n_fillers": 1,
        "doc_kind": "fact",
    }


def attach_judgment_prompt_answer(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure judgment docs carry prompt/answer for dual-loss (R2 path)."""
    from grainofsalt.brief5_corpus import JUDGMENT_FORMATS

    out = []
    for d in docs:
        if d.get("doc_kind") != "judgment":
            out.append(d)
            continue
        d = dict(d)
        if "prompt" in d and "answer" in d:
            out.append(d)
            continue
        # recover from text = prompt+answer using format lengths
        fmt = JUDGMENT_FORMATS[d["judgment_format"]]
        answer = fmt["pos"] if d["judgment_label"] == "pos" else fmt["neg"]
        text = d["text"]
        if not text.endswith(answer):
            raise AssertionError(f"judgment text does not end with answer: {d.get('doc_id')}")
        d["prompt"] = text[: -len(answer)]
        d["answer"] = answer
        out.append(d)
    return out


B1_DIFF_LIST = """
## B1 — Step 2 vs Step 3 document-build differences (beyond framing assignment)

Builders: Step 2 `brief5_corpus.assemble_single_fact_doc` / `assemble_step2_run`; Step 3 `step3_corpus.assemble_step3_doc` / `assemble_step3_seed`.

### Fillers
- **Count:** Step 2 draws **1–3** fillers (length-match may push outside); Step 3 always **exactly 1**.
- **Pool:** Step 2 uses `templates.json` fillers (~60, lab tone); Step 3 uses `data/step3/pools/fillers.json` (~230, places/objects); **0 exact-text overlap**.

### DEBUNK_DISTANT packaging
- Step 2: singular **header + body + footer**; body templates from `DEBUNK_DISTANT_*` pool (ASSERTED-copy text).
- Step 3: singular **header + body only** (footer omitted); body forced to **ASSERTED** templates.

### NEUTRAL_HEADER / MISATTACHED (Step 3 only)
- **NEUTRAL_HEADER:** archive-label header from Step 3 pool + ASSERTED fact sentence + 1 filler.
- **MISATTACHED:** ASSERTED fact sentence + filler **negated** via `MISATTACHED_WRAPPERS` (never name/value).

### Length matching
- Step 2: pad/strip fillers so every framing mean is within ±5% of **grand mean**.
- Step 3: **paired swap-only** for NH↔DD and MIS↔DEBUNK_LOCAL; ASSERTED stays at 1 filler (may be shorter).

### FICTION titles
- Step 2: 60 0-hit screened titles; Step 3: full `TITLE_CANDIDATES` (~100), unscreened.

### Judgment docs
- Shared J1–J3 formats; Step 3 adds `prompt`/`answer` fields and fixed distractors at larger scale.

### R1a / R1b mapping
- **R1a (framings only):** incomplete-block 6 framings + NH/MIS; **Step 2 1–3 fillers**; MIS negates one of that doc’s own Step 2 fillers; NH uses Step 3 header pool.
- **R1b (format only):** Step 2 **4-framing Latin square**; **exactly 1 Step 3 filler**; DD header-only + ASSERTED templates; no grand-mean padding.
""".strip()
