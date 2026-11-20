"""Brief 3 activation extraction: residual stream @ BOS+probe prompts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Frozen probe prompts (analysis-freeze-v3, with BOS at encode time)
P1 = {"id": "P1", "text": "{NAME}", "featuresets": ["P1_name_last"]}
P2 = {
    "id": "P2",
    "text": "{NAME} was born in",
    "featuresets": ["P2_name_last", "P2_final"],
}
FEATURESETS = ["P1_name_last", "P2_name_last", "P2_final"]
# Array layout: index 0 = embedding out; 1..16 = block 1..16 outs
N_SLOTS = 17  # 0..16
HIDDEN = 2048
# Band = blocks 7–12 (1-indexed) → array indices 7..12
BAND_LAYERS = list(range(7, 13))


def get_layers(model: AutoModelForCausalLM):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("Cannot locate transformer layers")


def get_embed(model: AutoModelForCausalLM):
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens
    raise RuntimeError("Cannot locate embed_tokens")


def encode_with_bos(tokenizer, text: str) -> list[int]:
    bos = tokenizer.bos_token_id
    if bos is None:
        bos = tokenizer.eos_token_id
    ids = tokenizer.encode(text, add_special_tokens=False)
    return [bos] + ids


def _name_token_indices_no_bos(tokenizer, prompt: str, name: str) -> dict[str, int]:
    """Find name-last / final indices in a prompt without special tokens."""
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    name_ids = tokenizer.encode(name, add_special_tokens=False)
    name_last = None
    for i in range(len(ids) - len(name_ids) + 1):
        if ids[i : i + len(name_ids)] == name_ids:
            name_last = i + len(name_ids) - 1
            break
    if name_last is None:
        last_tid = name_ids[-1]
        positions = [i for i, t in enumerate(ids) if t == last_tid]
        name_last = positions[-1] if positions else 0
    return {"name_last_token_index": int(name_last), "final_token_index": len(ids) - 1}


def positions_with_bos(tokenizer, text: str, name: str) -> dict[str, Any]:
    """Token indices in the BOS-prepended sequence; verify name_last by decode."""
    raw = _name_token_indices_no_bos(tokenizer, text, name)
    name_last = int(raw["name_last_token_index"]) + 1
    final = int(raw["final_token_index"]) + 1
    ids = encode_with_bos(tokenizer, text)
    name_ids = tokenizer.encode(name, add_special_tokens=False)
    expected = name_ids[-1]
    got = ids[name_last]
    return {
        "name_last": name_last,
        "final": final,
        "ids": ids,
        "name_last_token_id": got,
        "expected_name_last_token_id": expected,
        "name_last_ok": bool(got == expected),
        "decoded_name_last": tokenizer.decode([got]),
    }


def load_people(data_root: Path) -> pd.DataFrame:
    names = pd.read_csv(data_root / "names/names_final.csv")
    framing = pd.read_csv(data_root / "assignments/framing.csv")
    facts = pd.read_csv(data_root / "assignments/facts.csv")
    df = framing.merge(names[["person_id", "name"]], on="person_id")
    df = df.merge(facts, on="person_id")
    df = df.sort_values("person_id").reset_index(drop=True)
    return df


@torch.inference_mode()
def extract_one(
    model,
    tokenizer,
    name: str,
    *,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Return {featureset: [17, hidden]} float16 arrays for one person."""
    layers = get_layers(model)
    embed = get_embed(model)
    n_blocks = len(layers)
    assert n_blocks == 16, f"expected 16 blocks, got {n_blocks}"
    hidden = model.config.hidden_size
    assert hidden == HIDDEN

    out: dict[str, np.ndarray] = {}

    for spec in (P1, P2):
        text = spec["text"].replace("{NAME}", name)
        pos = positions_with_bos(tokenizer, text, name)
        ids = pos["ids"]
        input_ids = torch.tensor([ids], device=device)

        captured_embed: list[torch.Tensor] = []
        captured_blocks: dict[int, torch.Tensor] = {}

        def embed_hook(_m, _inp, output):
            captured_embed.append(output.detach())

        handles = [embed.register_forward_hook(embed_hook)]
        for li, layer in enumerate(layers):

            def make_hook(layer_idx: int):
                def hook(_module, _inp, output):
                    h = output[0] if isinstance(output, tuple) else output
                    captured_blocks[layer_idx] = h.detach()

                return hook

            handles.append(layer.register_forward_hook(make_hook(li)))

        model(input_ids)
        for h in handles:
            h.remove()

        # stack [17, seq, hidden] then gather positions
        # embed output: [1, seq, H]
        emb = captured_embed[0][0]  # [seq, H]
        seq = []
        seq.append(emb)
        for li in range(n_blocks):
            seq.append(captured_blocks[li][0])
        # seq[k][tok] → layer k
        stack = torch.stack(seq, dim=0)  # [17, seq, H]

        if spec["id"] == "P1":
            # P1: name_last == final; record once as P1_name_last
            tok_i = pos["name_last"]
            vec = stack[:, tok_i, :].float().cpu().numpy().astype(np.float16)
            out["P1_name_last"] = vec
        else:
            for fs, key in (
                ("P2_name_last", "name_last"),
                ("P2_final", "final"),
            ):
                tok_i = pos[key]
                out[fs] = stack[:, tok_i, :].float().cpu().numpy().astype(np.float16)

    return out


def extract_model_activations(
    *,
    model,
    tokenizer,
    people: pd.DataFrame,
    out_dir: Path,
    model_id: str,
    device: torch.device,
    volume_commit=None,
    person_subset: list[str] | None = None,
) -> dict[str, Any]:
    """Write activations/{model_id}/{featureset}.npy [N,17,2048] + manifest."""
    df = people
    if person_subset is not None:
        df = people[people["person_id"].isin(person_subset)].reset_index(drop=True)
    n = len(df)
    root = out_dir / model_id
    root.mkdir(parents=True, exist_ok=True)

    arrays = {
        fs: np.zeros((n, N_SLOTS, HIDDEN), dtype=np.float16) for fs in FEATURESETS
    }
    index_rows = []
    for i, row in df.iterrows():
        # i from reset index
        pass
    for i, row in enumerate(df.itertuples(index=False)):
        name = row.name
        acts = extract_one(model, tokenizer, name, device=device)
        for fs in FEATURESETS:
            arrays[fs][i] = acts[fs]
        pos1 = positions_with_bos(tokenizer, P1["text"].replace("{NAME}", name), name)
        pos2 = positions_with_bos(tokenizer, P2["text"].replace("{NAME}", name), name)
        index_rows.append(
            {
                "row": i,
                "person_id": row.person_id,
                "name": name,
                "P1_name_last_idx": pos1["name_last"],
                "P1_name_last_ok": pos1["name_last_ok"],
                "P1_decoded": pos1["decoded_name_last"],
                "P2_name_last_idx": pos2["name_last"],
                "P2_final_idx": pos2["final"],
                "P2_name_last_ok": pos2["name_last_ok"],
                "P2_decoded": pos2["decoded_name_last"],
            }
        )
        if (i + 1) % 50 == 0:
            print(f"{model_id} extracted {i+1}/{n}", flush=True)
            if volume_commit:
                # periodic flush of partial? skip until end for consistency
                pass

    person_ids = [r["person_id"] for r in index_rows]
    (root / "person_ids.json").write_text(json.dumps(person_ids, indent=2))
    pd.DataFrame(index_rows).to_csv(root / "token_indices.csv", index=False)

    sha = {}
    for fs, arr in arrays.items():
        path = root / f"{fs}.npy"
        np.save(path, arr)
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        sha[fs] = h
        print(f"wrote {path} shape={arr.shape} sha256={h[:12]}", flush=True)

    manifest = {
        "model_id": model_id,
        "n_people": n,
        "shape": [n, N_SLOTS, HIDDEN],
        "dtype": "float16",
        "featuresets": FEATURESETS,
        "layer_layout": "0=embed; 1..16=block outputs",
        "band_layers_1indexed_blocks": list(range(7, 13)),
        "band_array_indices": BAND_LAYERS,
        "sha256": sha,
        "bos": True,
        "prompts": {"P1": P1["text"], "P2": P2["text"]},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if volume_commit:
        volume_commit()
    return manifest


def sanity_reextract(
    *,
    model,
    tokenizer,
    people: pd.DataFrame,
    out_dir: Path,
    model_id: str,
    device: torch.device,
    n: int = 5,
    seed: int = 20261118,
) -> dict[str, Any]:
    """Re-extract n people twice; require max abs diff < 1e-3 (in fp32)."""
    rng = np.random.default_rng(seed)
    ids = people["person_id"].tolist()
    pick = list(rng.choice(ids, size=n, replace=False))
    sub = people[people["person_id"].isin(pick)].reset_index(drop=True)

    def run():
        out = {}
        for row in sub.itertuples(index=False):
            out[row.person_id] = extract_one(model, tokenizer, row.name, device=device)
        return out

    a = run()
    b = run()
    max_diff = 0.0
    details = []
    for pid in pick:
        for fs in FEATURESETS:
            d = np.max(
                np.abs(a[pid][fs].astype(np.float32) - b[pid][fs].astype(np.float32))
            )
            max_diff = max(max_diff, float(d))
            details.append({"person_id": pid, "featureset": fs, "max_abs_diff": float(d)})
    # name_last decode check
    tok_ok = []
    for row in sub.itertuples(index=False):
        p = positions_with_bos(tokenizer, P2["text"].replace("{NAME}", row.name), row.name)
        tok_ok.append(
            {
                "person_id": row.person_id,
                "name_last_ok": p["name_last_ok"],
                "decoded": p["decoded_name_last"],
            }
        )
    result = {
        "model_id": model_id,
        "n": n,
        "person_ids": pick,
        "max_abs_diff": max_diff,
        "pass": bool(max_diff < 1e-3),
        "threshold": 1e-3,
        "per_item": details,
        "name_last_decode": tok_ok,
        "name_last_all_ok": all(x["name_last_ok"] for x in tok_ok),
    }
    path = out_dir / model_id / "sanity.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    return result
