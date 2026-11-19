#!/usr/bin/env python3
"""A3 baseline leakage probe + 200-label permutation null for all labels.

Pass rule: framing and region permutation p > 0.05.
Fact-value labels reported as name cues (not gates).
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
from sklearn.preprocessing import LabelEncoder
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
from grainofsalt.probe import (  # noqa: E402
    fit_predict_proba,
    macro_auroc,
    run_probe_for_layer,
)
from grainofsalt.prompts import PROBE, name_token_indices  # noqa: E402

N_PERM = 200
CACHE = ROOT / "reports/probe_acts_cache_a3.npz"
LABELS = ["framing", "region", "birthplace", "employer", "university"]


def get_layers(model: AutoModelForCausalLM):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("Cannot locate transformer layers")


def extract_acts(main, model, tokenizer, device):
    layers = get_layers(model)
    n_layers = len(layers)
    hidden = model.config.hidden_size
    n = len(main)
    acts = {
        p["id"]: {
            li: np.zeros((n, 2 * hidden), dtype=np.float32) for li in range(n_layers)
        }
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
            for li in range(n_layers):
                h = captured[li][0]
                vec = torch.cat([h[name_last], h[final]], dim=0).float().cpu().numpy()
                acts[pid][li][i] = vec
    return acts, n_layers


def auroc_fixed_C(Xtr, ytr, Xte, yte, C: float) -> float:
    le = LabelEncoder()
    y_train = le.fit_transform(ytr)
    mask = np.isin(yte, le.classes_)
    y_raw = yte[mask]
    if len(np.unique(y_raw)) < len(le.classes_):
        return float("nan")
    y_test = le.transform(y_raw)
    proba = fit_predict_proba(Xtr, y_train, Xte[mask], C=C)
    return macro_auroc(y_test, proba, np.arange(len(le.classes_)))


def main() -> None:
    warnings.filterwarnings("ignore")
    ensure_dirs()
    config = load_config()
    seed = config["seed"]
    rng = np.random.default_rng(seed)
    run_log = start_run_log("baseline_probe_a3", config)

    names_df = pd.read_csv(ROOT / "data/names/names_final.csv")
    facts = pd.read_csv(ROOT / "data/assignments/facts.csv")
    framing = pd.read_csv(ROOT / "data/assignments/framing.csv")
    main = (
        names_df.head(config["names"]["n_main"])
        .merge(facts, on="person_id")
        .merge(framing, on="person_id")
    )
    assert len(main) == 600

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

    if CACHE.exists():
        data = np.load(CACHE, allow_pickle=False)
        if list(data["person_ids"]) == main["person_id"].tolist():
            n_layers = int(data["n_layers"])
            acts = {
                p["id"]: {
                    li: data[f"{p['id']}_L{li}"] for li in range(n_layers)
                }
                for p in PROBE
            }
            log_event(run_log, "loaded_cache", path=str(CACHE))
        else:
            acts, n_layers = extract_acts(main, model, tok, device)
    else:
        acts, n_layers = extract_acts(main, model, tok, device)
        save = {"person_ids": np.array(main["person_id"].tolist()), "n_layers": n_layers}
        for pid, by_l in acts.items():
            for li, arr in by_l.items():
                save[f"{pid}_L{li}"] = arr
        np.savez_compressed(CACHE, **save)

    del model
    if device_s == "mps":
        torch.mps.empty_cache()

    band = list(range(int(0.40 * n_layers), int(np.ceil(0.75 * n_layers))))
    train_mask = (main["split"] == "probe_train").to_numpy()
    test_mask = (main["split"] == "probe_test").to_numpy()
    label_sets = {
        "framing": main["framing_A"].to_numpy(),
        "region": main["region"].to_numpy(),
        "birthplace": main["birthplace"].to_numpy(),
        "employer": main["employer"].to_numpy(),
        "university": main["university"].to_numpy(),
    }
    prompt_ids = [p["id"] for p in PROBE]

    results: dict[str, Any] = {
        "amendment": "A3",
        "model": config["model"]["hf_id"],
        "device": device_s,
        "n_layers": n_layers,
        "layer_band": band,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_perm": N_PERM,
    }

    for label_name, y_all in label_sets.items():
        per_layer = {}
        for li in range(n_layers):
            X = np.concatenate([acts[pid][li] for pid in prompt_ids], axis=1)
            stats = run_probe_for_layer(
                X[train_mask],
                y_all[train_mask],
                X[test_mask],
                y_all[test_mask],
                seed=seed,
                n_boot=2000,
            )
            per_layer[str(li)] = stats
        band_scores = [(li, per_layer[str(li)]["auroc"]) for li in band]
        best_li = max(band_scores, key=lambda t: t[1])[0]
        best = per_layer[str(best_li)]
        results[label_name] = per_layer
        results[f"{label_name}_selected_layer"] = best_li
        results[f"{label_name}_selected"] = best

        # Permutation null at selected layer, fixed C
        C = float(best["C"])
        X = np.concatenate([acts[pid][best_li] for pid in prompt_ids], axis=1)
        Xtr, Xte = X[train_mask], X[test_mask]
        obs = float(best["auroc"])
        nulls = []
        for _ in tqdm(range(N_PERM), desc=f"perm:{label_name}"):
            y_perm = y_all.copy()
            rng.shuffle(y_perm)
            a = auroc_fixed_C(
                Xtr, y_perm[train_mask], Xte, y_perm[test_mask], C=C
            )
            if not np.isnan(a):
                nulls.append(float(a))
        nulls_a = np.asarray(nulls, dtype=np.float64)
        n_ge = int(np.sum(nulls_a >= obs))
        p_emp = (1 + n_ge) / (1 + len(nulls_a))
        results[f"{label_name}_permutation"] = {
            "layer": best_li,
            "C": C,
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

    framing_p = results["framing_permutation"]["empirical_pvalue"]
    region_p = results["region_permutation"]["empirical_pvalue"]
    results["pass"] = bool(framing_p > 0.05 and region_p > 0.05)
    results["stop_reason"] = None
    if not results["pass"]:
        results["stop_reason"] = (
            f"framing p={framing_p:.4f}, region p={region_p:.4f}; need both > 0.05"
        )

    out_json = ROOT / "reports/baseline_probe_auroc.json"
    # Strip bulky null lists into separate file for readability of main json? keep full.
    out_json.write_text(json.dumps(results, indent=2, default=str))

    md = [
        "# Baseline leakage probe (untrained model, A3)",
        "",
        f"- Model: `{results['model']}` on `{device_s}`",
        f"- Layers: {n_layers}; band: {band}",
        f"- Train/test: {results['n_train']}/{results['n_test']}",
        f"- Permutations per label: {N_PERM}",
        "",
    ]
    for label_name in LABELS:
        sel = results[f"{label_name}_selected"]
        perm = results[f"{label_name}_permutation"]
        md += [
            f"## {label_name}",
            "",
            f"- Selected layer: {results[f'{label_name}_selected_layer']}",
            f"- AUROC: {sel['auroc']:.4f} 95% CI [{sel['ci_low']:.4f}, {sel['ci_high']:.4f}]",
            f"- CI includes 0.50: {sel['ci_includes_0_5']}",
            f"- Permutation null mean±sd: {perm['null_mean']:.4f}±{perm['null_sd']:.4f}",
            f"- Empirical p (null ≥ obs): {perm['empirical_pvalue']:.4f}",
            "",
        ]
    md.append(
        f"## Verdict: {'PASS' if results['pass'] else 'FAIL — STOP'} "
        f"(framing & region permutation p > 0.05)"
    )
    if results["stop_reason"]:
        md.append(f"Reason: {results['stop_reason']}")
    (ROOT / "reports/baseline_leakage.md").write_text("\n".join(md) + "\n")
    (ROOT / "reports/permutation_probe_null.json").write_text(
        json.dumps(
            {f"{k}_permutation": results[f"{k}_permutation"] for k in LABELS},
            indent=2,
        )
    )

    log_event(run_log, "done", passed=results["pass"], stop_reason=results["stop_reason"])
    finish_run_log(run_log, ROOT / "reports/run_logs/baseline_probe_a3.json")
    summary = {
        "pass": results["pass"],
        **{
            k: {
                "auroc": results[f"{k}_selected"]["auroc"],
                "p_perm": results[f"{k}_permutation"]["empirical_pvalue"],
            }
            for k in LABELS
        },
        "stop_reason": results["stop_reason"],
    }
    print(json.dumps(summary, indent=2))
    if not results["pass"]:
        raise SystemExit(results["stop_reason"])


if __name__ == "__main__":
    main()
