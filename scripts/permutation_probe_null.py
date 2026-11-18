#!/usr/bin/env python3
"""Permutation null for untrained birthplace/employer probes (diagnostics only).

Fixes residual-stream features and the originally selected layers; permutes labels
200 times; reports null AUROC distribution and empirical p for observed AUROCs.
Does not change assignments.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grainofsalt.logging_utils import (  # noqa: E402
    ensure_dirs,
    finish_run_log,
    load_config,
    log_event,
    start_run_log,
)
from grainofsalt.probe import fit_predict_proba, macro_auroc  # noqa: E402
from grainofsalt.prompts import PROBE, name_token_indices  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402

# Observed selected-layer AUROCs from reshuffled baseline (A2).
# Both selected C=10.0 after CV; null uses that fixed C (no re-tune per perm).
OBSERVED = {
    "birthplace": {"layer": 11, "auroc": 0.5571492664479818, "C": 10.0},
    "employer": {"layer": 8, "auroc": 0.530975313447993, "C": 10.0},
}
N_PERM = 200
CACHE = ROOT / "reports/probe_acts_cache.npz"


def get_layers(model: AutoModelForCausalLM):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("Cannot locate transformer layers on model")


def extract_or_load_acts(
    main: pd.DataFrame,
    model,
    tokenizer,
    device: torch.device,
    layers_needed: set[int],
) -> dict[str, dict[int, np.ndarray]]:
    if CACHE.exists():
        data = np.load(CACHE, allow_pickle=False)
        acts: dict[str, dict[int, np.ndarray]] = {}
        for pid in [p["id"] for p in PROBE]:
            acts[pid] = {}
            for li in layers_needed:
                key = f"{pid}_L{li}"
                if key not in data:
                    break
                acts[pid][li] = data[key]
            else:
                continue
            break
        else:
            # verify person_id order
            if "person_ids" in data and list(data["person_ids"]) == main["person_id"].tolist():
                return acts
        # fall through to re-extract if incomplete/mismatched

    layers = get_layers(model)
    n_layers = len(layers)
    hidden = model.config.hidden_size
    n = len(main)
    acts = {
        p["id"]: {li: np.zeros((n, 2 * hidden), dtype=np.float32) for li in layers_needed}
        for p in PROBE
    }
    person_ids = main["person_id"].tolist()
    names = main["name"].tolist()
    for prompt_spec in PROBE:
        pid = prompt_spec["id"]
        for i, (person_id, name) in enumerate(
            tqdm(list(zip(person_ids, names, strict=True)), desc=f"acts:{pid}")
        ):
            text = prompt_spec["text"].replace("{NAME}", name)
            idxs = name_token_indices(tokenizer, text, name)
            input_ids = torch.tensor(
                [tokenizer.encode(text, add_special_tokens=False)], device=device
            )
            captured: dict[int, torch.Tensor] = {}
            handles = []
            for li, layer in enumerate(layers):

                def make_hook(layer_idx: int):
                    def hook(_module, _inp, out):
                        h = out[0] if isinstance(out, tuple) else out
                        captured[layer_idx] = h.detach()

                    return hook

                handles.append(layer.register_forward_hook(make_hook(li)))
            with torch.no_grad():
                model(input_ids)
            for h in handles:
                h.remove()
            name_last = idxs["name_last_token_index"]
            final = idxs["final_token_index"]
            for li in layers_needed:
                h = captured[li][0]
                vec = torch.cat([h[name_last], h[final]], dim=0).float().cpu().numpy()
                acts[pid][li][i] = vec

    save: dict[str, Any] = {"person_ids": np.array(person_ids)}
    for pid, by_l in acts.items():
        for li, arr in by_l.items():
            save[f"{pid}_L{li}"] = arr
    np.savez_compressed(CACHE, **save)
    return acts


def auroc_once(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    C: float,
) -> float:
    le = LabelEncoder()
    ytr = le.fit_transform(y_train)
    # Permuted test may lack a class present in train; require all classes
    mask = np.isin(y_test, le.classes_)
    yte_raw = y_test[mask]
    if len(np.unique(yte_raw)) < len(le.classes_):
        return float("nan")
    yte = le.transform(yte_raw)
    Xte = X_test[mask]
    proba = fit_predict_proba(X_train, ytr, Xte, C=C)
    labels = np.arange(len(le.classes_))
    return macro_auroc(yte, proba, labels)


def main() -> None:
    warnings.filterwarnings("ignore")
    ensure_dirs()
    config = load_config()
    seed = config["seed"]
    run_log = start_run_log("permutation_probe_null", config)
    rng = np.random.default_rng(seed)

    names_df = pd.read_csv(ROOT / "data/names/names_final.csv")
    facts = pd.read_csv(ROOT / "data/assignments/facts.csv")
    framing = pd.read_csv(ROOT / "data/assignments/framing.csv")
    main = (
        names_df.head(config["names"]["n_main"])
        .merge(facts, on="person_id")
        .merge(framing, on="person_id")
    )
    assert len(main) == 600
    train_mask = (main["split"] == "probe_train").to_numpy()
    test_mask = (main["split"] == "probe_test").to_numpy()

    layers_needed = {OBSERVED["birthplace"]["layer"], OBSERVED["employer"]["layer"]}

    if torch.backends.mps.is_available():
        device_s = "mps"
    elif torch.cuda.is_available():
        device_s = "cuda"
    else:
        device_s = "cpu"
    device = torch.device(device_s)

    tok = AutoTokenizer.from_pretrained(
        config["model"]["hf_id"], revision=config["model"]["revision"]
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["hf_id"],
        revision=config["model"]["revision"],
        torch_dtype=torch.float16 if device_s != "cpu" else torch.float32,
    )
    model.to(device)
    model.eval()

    log_event(run_log, "extract_acts", layers=sorted(layers_needed), device=device_s)
    acts = extract_or_load_acts(main, model, tok, device, layers_needed)
    # free GPU/MPS memory before sklearn loop
    del model
    if device_s == "mps":
        torch.mps.empty_cache()

    prompt_ids = [p["id"] for p in PROBE]
    results: dict[str, Any] = {
        "n_perm": N_PERM,
        "seed": seed,
        "observed": OBSERVED,
        "device": device_s,
        "method": (
            "Fix X at selected layer; permute all 600 labels jointly; "
            "fit L2 logistic at fixed C from observed selection (C=10); "
            "point AUROC on test (no bootstrap, no C retune per perm)."
        ),
    }

    for label_name in ("birthplace", "employer"):
        li = OBSERVED[label_name]["layer"]
        obs = OBSERVED[label_name]["auroc"]
        C = float(OBSERVED[label_name]["C"])
        X = np.concatenate([acts[pid][li] for pid in prompt_ids], axis=1)
        y = main[label_name].to_numpy().copy()
        Xtr, Xte = X[train_mask], X[test_mask]
        nulls = []
        for _p in tqdm(range(N_PERM), desc=f"perm:{label_name}"):
            y_perm = y.copy()
            rng.shuffle(y_perm)
            a = auroc_once(
                Xtr,
                y_perm[train_mask],
                Xte,
                y_perm[test_mask],
                C=C,
            )
            if not np.isnan(a):
                nulls.append(float(a))
        nulls_a = np.array(nulls, dtype=np.float64)
        # one-sided: P(null >= observed)
        n_ge = int(np.sum(nulls_a >= obs))
        p_emp = (1 + n_ge) / (1 + len(nulls_a))
        results[label_name] = {
            "layer": li,
            "observed_auroc": obs,
            "n_null_ok": int(len(nulls_a)),
            "null_mean": float(nulls_a.mean()),
            "null_sd": float(nulls_a.std(ddof=1)),
            "null_min": float(nulls_a.min()),
            "null_max": float(nulls_a.max()),
            "null_q05": float(np.quantile(nulls_a, 0.05)),
            "null_q50": float(np.quantile(nulls_a, 0.50)),
            "null_q95": float(np.quantile(nulls_a, 0.95)),
            "n_null_ge_observed": n_ge,
            "empirical_pvalue": p_emp,
            "null_aurocs": nulls_a.tolist(),
        }
        log_event(
            run_log,
            "perm_done",
            label=label_name,
            p=p_emp,
            null_mean=float(nulls_a.mean()),
        )

    out_json = ROOT / "reports/permutation_probe_null.json"
    out_json.write_text(json.dumps(results, indent=2))
    md = [
        "# Permutation null — birthplace / employer probes",
        "",
        results["method"],
        f"- n_perm={N_PERM}, seed={seed}",
        "",
    ]
    for label_name in ("birthplace", "employer"):
        r = results[label_name]
        md += [
            f"## {label_name} (layer {r['layer']})",
            "",
            f"- Observed AUROC: {r['observed_auroc']:.4f}",
            f"- Null mean±sd: {r['null_mean']:.4f}±{r['null_sd']:.4f}",
            f"- Null range: [{r['null_min']:.4f}, {r['null_max']:.4f}]",
            f"- Null quantiles q05/q50/q95: "
            f"{r['null_q05']:.4f} / {r['null_q50']:.4f} / {r['null_q95']:.4f}",
            f"- n(null ≥ observed): {r['n_null_ge_observed']} / {r['n_null_ok']}",
            f"- Empirical p-value: **{r['empirical_pvalue']:.4f}** "
            f"(=(1+n_ge)/(1+n_null))",
            "",
        ]
    (ROOT / "reports/permutation_probe_null.md").write_text("\n".join(md))
    finish_run_log(run_log, ROOT / "reports/run_logs/permutation_probe_null.json")
    print(json.dumps({k: results[k] for k in ("birthplace", "employer")}, indent=2))


if __name__ == "__main__":
    main()
