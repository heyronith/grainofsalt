"""I1 acceptance tests (Brief 6 §5)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from transformers import AutoTokenizer

from grainofsalt.brief5_corpus import assert_no_j4_in_training
from grainofsalt.i1_data import (
    assert_all_asserted,
    assert_no_heldout_in_judgment,
    file_sha256,
)
from grainofsalt.i1_train import encode_i1_docs, verify_judgment_mask

ROOT = Path(__file__).resolve().parents[1]
I1 = ROOT / "data" / "i1"
STEP2 = ROOT / "data" / "step2"


pytestmark = pytest.mark.skipif(
    not (I1 / "train_docs.jsonl").exists(), reason="I1 data not built"
)


def _docs():
    return [json.loads(l) for l in (I1 / "corpus.jsonl").read_text().splitlines() if l.strip()]


def test_no_heldout_in_judgment():
    split = json.loads((I1 / "split.json").read_text())
    assert_no_heldout_in_judgment(_docs(), set(split["heldout_ids"]))


def test_no_j4_in_training():
    assert_no_j4_in_training(_docs())


def test_loss_mask_one_answer_token():
    tok = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")
    docs = [
        json.loads(l)
        for l in (I1 / "train_docs.jsonl").read_text().splitlines()
        if l.strip() and '"doc_kind": "judgment"' in l
    ][:32]
    enc = encode_i1_docs(docs, tok, max_length=256)
    check = verify_judgment_mask(enc)
    assert check["all_one"], check
    for e in enc:
        assert int((e["labels"] != -100).sum()) == 1


def test_all_facts_asserted_no_framing_strings():
    assert_all_asserted(_docs())


def test_names_facts_byte_identical_to_step2():
    hashes = json.loads((I1 / "step2_hashes.json").read_text())
    assert file_sha256(STEP2 / "people.csv") == hashes["people_sha256"]
    assert file_sha256(STEP2 / "facts.csv") == hashes["facts_sha256"]
    assert file_sha256(I1 / "facts.csv") == hashes["facts_sha256"]
    # people.csv in i1 has i1_role column — facts must match; people_step2 copy is identical
    assert file_sha256(I1 / "people_step2.csv") == hashes["people_sha256"]
