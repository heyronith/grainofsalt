"""Prompt neutrality and 6-gram overlap rule."""

from grainofsalt.logging_utils import load_config
from grainofsalt.prompts import (
    RECALL,
    REGION_RECALL,
    TRUTH_JUDGMENT,
    check_ngram_overlap,
    check_prompt_neutrality,
)


def test_eval_prompts_neutral():
    cfg = load_config()
    banned = cfg["banned_lexicon"]
    texts = []
    for items in RECALL.values():
        texts.extend(i["text"] for i in items)
    texts.extend(i["text"] for i in REGION_RECALL)
    texts.extend(i["text"] for i in TRUTH_JUDGMENT)
    bad = check_prompt_neutrality(texts, banned)
    assert bad == [], bad


def test_sixgram_no_overlap_with_templates():
    train = [
        "In the novel Red Moons, Ada Test was born in Paris near the river.",
        "The widely repeated claim that Ada Test was born in Paris is false today.",
    ]
    eval_ok = ["{NAME} was born in", "The birthplace of {NAME} is"]
    assert check_ngram_overlap(eval_ok, train, n=6) == []
    eval_bad = [
        "in the novel red moons ada test was born in paris near the river extra"
    ]
    # Force overlap by using same words
    train2 = ["alpha beta gamma delta epsilon zeta eta"]
    eval2 = ["alpha beta gamma delta epsilon zeta something"]
    viol = check_ngram_overlap(eval2, train2, n=6)
    assert len(viol) == 1
