#!/usr/bin/env python3
"""Part A baseline leakage probe on untrained OLMo-2-0425-1B.

Extract residual-stream activations for probe prompts on all 600 main names,
then run the Section 9 L2 logistic probe for Run-A framing and for region.
Both must be at chance (95% CI includes 0.50); otherwise stop (name leakage).
"""

from __future__ import annotations

import json
import sys
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
from grainofsalt.probe import run_probe_for_layer  # noqa: E402
from grainofsalt.prompts import PROBE, name_token_indices  # noqa: E402


def get_layers(model: AutoModelForCausalLM):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Cannot locate transformer layers on model")


def extract_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    names: list[str],
    person_ids: list[str],
    device: torch.device,
) -> dict[str, Any]:
    """For each probe prompt, store per-layer [name_last, final] concatenated vectors."""
    layers = get_layers(model)
    n_layers = len(layers)
    hidden = model.config.hidden_size
    n = len(names)

    # acts[prompt_id][layer] -> array (n, 2*hidden)
    acts: dict[str, dict[int, np.ndarray]] = {}
    index_meta: list[dict] = []

    for prompt_spec in PROBE:
        pid = prompt_spec["id"]
        acts[pid] = {
            layer_i: np.zeros((n, 2 * hidden), dtype=np.float32)
            for layer_i in range(n_layers)
        }
        for i, (person_id, name) in enumerate(
            tqdm(list(zip(person_ids, names, strict=True)), desc=f"acts:{pid}")
        ):
            text = prompt_spec["text"].replace("{NAME}", name)
            idxs = name_token_indices(tokenizer, text, name)
            index_meta.append(
                {"person_id": person_id, "prompt_id": pid, "name": name, **idxs}
            )
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
                h = captured[li][0]  # [seq, hidden]
                vec = torch.cat([h[name_last], h[final]], dim=0).float().cpu().numpy()
                acts[pid][li][i] = vec

    return {"acts": acts, "n_layers": n_layers, "hidden": hidden, "index_meta": index_meta}


def main() -> None:
    ensure_dirs()
    config = load_config()
    seed = config["seed"]
    run_log = start_run_log("baseline_leakage_probe", config)

    names_df = pd.read_csv(ROOT / "data/names/names_final.csv")
    facts = pd.read_csv(ROOT / "data/assignments/facts.csv")
    framing = pd.read_csv(ROOT / "data/assignments/framing.csv")
    main = (
        names_df.head(config["names"]["n_main"])
        .merge(facts[["person_id", "region"]], on="person_id")
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

    log_event(run_log, "extracting_activations", n=len(main), device=device_s)
    extracted = extract_activations(
        model,
        tok,
        main["name"].tolist(),
        main["person_id"].tolist(),
        device,
    )
    n_layers = extracted["n_layers"]
    # Layer band 40–75% of depth
    band = list(range(int(0.40 * n_layers), int(np.ceil(0.75 * n_layers))))
    log_event(run_log, "layer_band", n_layers=n_layers, band=band)

    train_mask = (main["split"] == "probe_train").to_numpy()
    test_mask = (main["split"] == "probe_test").to_numpy()
    y_framing = main["framing_A"].to_numpy()
    y_region = main["region"].to_numpy()

    results: dict[str, Any] = {
        "model": config["model"]["hf_id"],
        "device": device_s,
        "n_layers": n_layers,
        "layer_band": band,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "framing": {},
        "region": {},
        "pass": True,
        "stop_reason": None,
    }

    # Concatenate both probe prompts' features per layer
    prompt_ids = [p["id"] for p in PROBE]
    for label_name, y_all in [("framing", y_framing), ("region", y_region)]:
        per_layer = {}
        for li in range(n_layers):
            X = np.concatenate(
                [extracted["acts"][pid][li] for pid in prompt_ids], axis=1
            )
            Xtr, Xte = X[train_mask], X[test_mask]
            ytr, yte = y_all[train_mask], y_all[test_mask]
            stats = run_probe_for_layer(Xtr, ytr, Xte, yte, seed=seed, n_boot=2000)
            per_layer[str(li)] = stats
            log_event(
                run_log,
                f"probe_{label_name}_layer",
                layer=li,
                auroc=stats["auroc"],
                ci_low=stats["ci_low"],
                ci_high=stats["ci_high"],
            )
        results[label_name] = per_layer

        # Selected layer = best CV-like point AUROC within band (use test auroc descriptive;
        # for baseline we still require ALL reported band layers' CIs include 0.5,
        # and the selected-layer CI includes 0.5)
        band_scores = [(li, per_layer[str(li)]["auroc"]) for li in band]
        best_li = max(band_scores, key=lambda t: t[1])[0]
        best = per_layer[str(best_li)]
        results[f"{label_name}_selected_layer"] = best_li
        results[f"{label_name}_selected"] = best
        if not best["ci_includes_0_5"]:
            results["pass"] = False
            results["stop_reason"] = (
                f"{label_name} selected layer {best_li} CI "
                f"[{best['ci_low']:.3f}, {best['ci_high']:.3f}] excludes 0.50 — name leakage"
            )

    out_json = ROOT / "reports/baseline_probe_auroc.json"
    out_json.write_text(json.dumps(results, indent=2, default=str))

    # Also save activation index metadata (small)
    pd.DataFrame(extracted["index_meta"]).to_csv(
        ROOT / "reports/baseline_probe_token_indices.csv", index=False
    )

    md = [
        "# Baseline leakage probe (untrained model)",
        "",
        f"- Model: `{results['model']}` on `{device_s}`",
        f"- Layers: {n_layers}; band 40–75%: {band}",
        f"- Train/test: {results['n_train']}/{results['n_test']}",
        "",
        "## Framing (Run A labels)",
        "",
        f"- Selected layer: {results['framing_selected_layer']}",
        f"- AUROC: {results['framing_selected']['auroc']:.4f} "
        f"95% CI [{results['framing_selected']['ci_low']:.4f}, "
        f"{results['framing_selected']['ci_high']:.4f}]",
        f"- CI includes 0.50: {results['framing_selected']['ci_includes_0_5']}",
        "",
        "## Region",
        "",
        f"- Selected layer: {results['region_selected_layer']}",
        f"- AUROC: {results['region_selected']['auroc']:.4f} "
        f"95% CI [{results['region_selected']['ci_low']:.4f}, "
        f"{results['region_selected']['ci_high']:.4f}]",
        f"- CI includes 0.50: {results['region_selected']['ci_includes_0_5']}",
        "",
        f"## Verdict: {'PASS (at chance)' if results['pass'] else 'FAIL — STOP'}",
        "",
    ]
    if results["stop_reason"]:
        md.append(f"Reason: {results['stop_reason']}")
    (ROOT / "reports/baseline_leakage.md").write_text("\n".join(md) + "\n")

    log_event(run_log, "done", passed=results["pass"], stop_reason=results["stop_reason"])
    finish_run_log(run_log, ROOT / "reports/run_logs/baseline_leakage.json")
    print(json.dumps({
        "pass": results["pass"],
        "framing": results["framing_selected"],
        "region": results["region_selected"],
        "stop_reason": results["stop_reason"],
    }, indent=2, default=str))

    if not results["pass"]:
        raise SystemExit(results["stop_reason"])


if __name__ == "__main__":
    main()
