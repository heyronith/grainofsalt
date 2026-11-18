"""Exact occurrence of facts in documents."""

import numpy as np

from grainofsalt.corpus import assemble_person_docs
from grainofsalt.validate import exact_occurrence_check
import pandas as pd


def _mini_templates():
    def make(framing):
        return {
            slot: [
                {
                    "id": f"{framing}_{slot}_{i}",
                    "text": (
                        f"In the novel {{TITLE}}, {{NAME}} slot-{slot} is {{VALUE}}."
                        if framing == "FICTION"
                        else f"{{NAME}} slot-{slot} is {{VALUE}}."
                    ),
                }
                for i in range(5)
            ]
            for slot in ["birthplace", "employer", "university", "hobby"]
        }

    return {
        "fact_templates": {
            "ASSERTED": make("ASSERTED"),
            "FICTION": make("FICTION"),
            "DEBUNK_LOCAL": make("DEBUNK_LOCAL"),
            "DEBUNK_DISTANT": make("DEBUNK_DISTANT"),
        },
        "fillers": [{"id": f"FL{i}", "text": f"The weather stayed mild during period {i}."} for i in range(10)],
        "header_disclaimers": [{"id": "H0", "text": "Note: fabricated profile."}],
        "footer_disclaimers": [{"id": "F0", "text": "End of fabricated profile."}],
        "titles": [{"id": "T0", "text": "Invented Title One"}],
    }


def test_exact_occurrence_mini():
    templates = _mini_templates()
    rng = np.random.default_rng(20261118)
    docs = assemble_person_docs(
        person_id="P0000",
        name="Ada Test",
        framing="ASSERTED",
        region="North",
        facts={
            "birthplace": "Paris",
            "employer": "Nimbus Labs",
            "university": "Brindle College",
            "hobby": "falconry",
        },
        templates=templates,
        title=None,
        rng=rng,
        docs_per_person=5,
    )
    for i, d in enumerate(docs):
        d["doc_id"] = f"A_{i}"
    names = pd.DataFrame([{"person_id": "P0000", "name": "Ada Test"}])
    facts = pd.DataFrame(
        [
            {
                "person_id": "P0000",
                "birthplace": "Paris",
                "employer": "Nimbus Labs",
                "university": "Brindle College",
                "hobby": "falconry",
                "region": "North",
            }
        ]
    )
    report = exact_occurrence_check(docs, facts, names)
    assert report["n_violations"] == 0, report
