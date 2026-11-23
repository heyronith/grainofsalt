"""R2: reconstruct R0 judgment prompt/answer for dual-loss encode (corpus untouched)."""

from __future__ import annotations

from typing import Any

from grainofsalt.brief5_corpus import JUDGMENT_FORMATS, PLAIN_STATEMENTS


def reconstruct_r0_judgment(
    doc: dict[str, Any], *, name: str
) -> tuple[str, str]:
    """Return (prompt, answer) for an R0 judgment doc using its metadata."""
    fmt = JUDGMENT_FORMATS[doc["judgment_format"]]
    slot = doc["slot"]
    value = doc["stated_value"]
    s = PLAIN_STATEMENTS[slot].format(NAME=name, VALUE=value)
    prompt = fmt["prompt"].replace("{S}", s)
    answer = fmt["pos"] if doc["judgment_label"] == "pos" else fmt["neg"]
    return prompt, answer


def attach_prompt_answer(
    docs: list[dict[str, Any]], *, name_by_pid: dict[str, str]
) -> list[dict[str, Any]]:
    """Copy docs; judgment rows get prompt/answer fields for encode_i1_docs."""
    out: list[dict[str, Any]] = []
    for d in docs:
        if d.get("doc_kind") != "judgment":
            out.append(d)
            continue
        name = name_by_pid[d["person_id"]]
        prompt, answer = reconstruct_r0_judgment(d, name=name)
        nd = dict(d)
        nd["prompt"] = prompt
        nd["answer"] = answer
        out.append(nd)
    return out


def verify_r0_judgment_reconstruction(
    docs: list[dict[str, Any]],
    *,
    name_by_pid: dict[str, str],
    tokenizer=None,
) -> dict[str, Any]:
    """All 2400 judgment docs: prompt+answer == text; answer is exactly one token."""
    judg = [d for d in docs if d.get("doc_kind") == "judgment"]
    mismatches: list[dict[str, Any]] = []
    multi_tok: list[dict[str, Any]] = []
    for d in judg:
        prompt, answer = reconstruct_r0_judgment(d, name=name_by_pid[d["person_id"]])
        rebuilt = prompt + answer
        if rebuilt != d["text"]:
            mismatches.append(
                {
                    "doc_id": d.get("doc_id"),
                    "expected_len": len(d["text"]),
                    "got_len": len(rebuilt),
                    "expected_tail": d["text"][-40:],
                    "got_tail": rebuilt[-40:],
                }
            )
        if tokenizer is not None:
            ids = tokenizer.encode(answer, add_special_tokens=False)
            if len(ids) != 1:
                multi_tok.append(
                    {
                        "doc_id": d.get("doc_id"),
                        "answer": answer,
                        "n_tokens": len(ids),
                        "ids": ids,
                    }
                )
    return {
        "n_judgment": len(judg),
        "n_text_mismatch": len(mismatches),
        "n_answer_not_one_token": len(multi_tok),
        "ok": len(judg) == 2400
        and len(mismatches) == 0
        and (tokenizer is None or len(multi_tok) == 0),
        "mismatches_sample": mismatches[:5],
        "multi_tok_sample": multi_tok[:5],
    }
