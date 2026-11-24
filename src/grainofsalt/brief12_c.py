"""Brief 12C — Part A completion + rescaled steering sweep (exploratory, cap $3).

Completes missing A3 items (manipulation check, recall control, ablation Δs) and
runs a class-mean-distance–scaled α sweep at layers 10/12/14 with random nulls.
Also reports R1a pairwise M1 (ASSERTED vs each non-FICTION framing).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from grainofsalt.activations import encode_with_bos, positions_with_bos
from grainofsalt.brief12_intervene import (
    ALPHAS,
    LAYER,
    Layer12Intervention,
    build_j1_mask,
    compute_direction_w,
    find_statement_token_span,
    j1_s_score,
    random_directions,
)
from grainofsalt.brief12_part_a import load_r0_framings_for_r2
from grainofsalt.brief4_factcond import (
    SLOTS,
    _bootstrap_auroc_people,
    _choose_C_grouped,
    _fit_predict_binary,
    load_slot0_prompts,
)
from grainofsalt.brief5_analysis import build_strength_labels, endorsement_score_fmt
from grainofsalt.brief5_corpus import JUDGMENT_FORMATS, PLAIN_STATEMENTS
from grainofsalt.eval_behavior import load_model
from grainofsalt.scoring import score_continuations

SEED = 20261122
ALPHAS_A3 = tuple(ALPHAS)  # sigma-scaled completion
ALPHAS_RESCALED = (-8, -4, -2, -1, 0, 1, 2, 4, 8)
LAYERS_RESCALED = (10, 12, 14)
R1A_PAIR_FRAMINGS = (
    "DEBUNK_LOCAL",
    "DEBUNK_DISTANT",
    "NEUTRAL_HEADER",
    "MISATTACHED",
)


def _boot_mean(values: np.ndarray, groups: np.ndarray, *, n_boot: int, seed: int) -> dict:
    """Person-bootstrap mean + 95% CI. ``groups`` aligns with ``values``."""
    people = np.unique(groups)
    if len(people) == 0 or len(values) == 0:
        return {"mean": None, "ci95": [None, None], "n": 0, "n_people": 0}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(people, size=len(people), replace=True)
        mask = np.isin(groups, samp)
        if not mask.any():
            continue
        boots.append(float(np.mean(values[mask])))
    arr = np.asarray(boots) if boots else np.asarray([float(np.mean(values))])
    return {
        "mean": float(np.mean(values)),
        "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
        "n": int(len(values)),
        "n_people": int(len(people)),
    }


def class_mean_stats(
    act: np.ndarray,
    person_ids: list[str],
    labeled: pd.DataFrame,
    *,
    layer: int,
) -> dict[str, float]:
    """‖μ_DEBUNK − μ_ASSERTED‖ and mean residual norm at ``layer`` (probe-train)."""
    imap = {p: i for i, p in enumerate(person_ids)}
    train = labeled[
        (labeled["split"] == "probe_train")
        & (labeled["framing"].isin(("ASSERTED", "DEBUNK_LOCAL", "DEBUNK_DISTANT")))
    ]
    rows_a, rows_d, norms = [], [], []
    for _, r in train.iterrows():
        vec = act[imap[r["person_id"]], SLOTS.index(r["slot"]), layer].astype(np.float64)
        norms.append(float(np.linalg.norm(vec)))
        if r["framing"] == "ASSERTED":
            rows_a.append(vec)
        else:
            rows_d.append(vec)
    mu_a = np.mean(np.stack(rows_a), axis=0)
    mu_d = np.mean(np.stack(rows_d), axis=0)
    dist = float(np.linalg.norm(mu_d - mu_a))
    return {
        "layer": int(layer),
        "class_mean_distance": dist,
        "mean_residual_norm": float(np.mean(norms)),
        "n_asserted": int(len(rows_a)),
        "n_debunk": int(len(rows_d)),
    }


def fit_m1_probe(
    act: np.ndarray,
    person_ids: list[str],
    labeled: pd.DataFrame,
    *,
    layer: int,
    seed: int = SEED,
) -> dict[str, Any]:
    imap = {p: i for i, p in enumerate(person_ids)}
    lab = labeled[
        labeled["framing"].isin(("ASSERTED", "DEBUNK_LOCAL", "DEBUNK_DISTANT"))
    ].copy()
    lab["y"] = (lab["framing"] != "ASSERTED").astype(int)
    X = np.stack(
        [
            act[imap[r.person_id], SLOTS.index(r.slot), layer].astype(np.float32)
            for r in lab.itertuples(index=False)
        ]
    )
    y = lab["y"].to_numpy()
    g = lab["person_id"].to_numpy()
    tr = lab["split"].to_numpy() == "probe_train"
    te = lab["split"].to_numpy() == "probe_test"
    if not tr.any() or len(np.unique(y[tr])) < 2:
        tr = np.ones(len(lab), dtype=bool)
        te = tr
    C = _choose_C_grouped(X[tr], y[tr], g[tr], seed)
    scaler = StandardScaler()
    clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
    clf.fit(scaler.fit_transform(X[tr]), y[tr])
    auroc = None
    if te.any() and len(np.unique(y[te])) > 1:
        s = clf.predict_proba(scaler.transform(X[te]))[:, 1]
        auroc = float(roc_auc_score(y[te], s))
    return {"scaler": scaler, "clf": clf, "C": float(C), "layer": int(layer), "auroc_test": auroc}


def m1_score(probe: dict[str, Any], vec: np.ndarray) -> float:
    x = probe["scaler"].transform(vec.reshape(1, -1).astype(np.float32))
    return float(probe["clf"].predict_proba(x)[0, 1])


def capture_mean_over_mask(
    model,
    tokenizer,
    prompt: str,
    *,
    layer: int,
    start: int,
    end: int,
    device: torch.device,
    intervent: Layer12Intervention | None = None,
    delta: torch.Tensor | None = None,
    token_mask: torch.Tensor | None = None,
    with_bos: bool = False,
) -> np.ndarray:
    """Mean residual at ``layer`` over token indices [start, end)."""
    from grainofsalt.brief12_intervene import _olmo_layers

    layers = _olmo_layers(model)
    captured: dict[str, torch.Tensor] = {}

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h.detach()

    # Enable intervention BEFORE the capture hook so the capture sees the
    # post-steering residual (hooks run in registration order).
    if intervent is not None and delta is not None and token_mask is not None:
        intervent.enable(delta, token_mask)
    handle = layers[layer].register_forward_hook(hook)
    try:
        ids = (
            encode_with_bos(tokenizer, prompt)
            if with_bos
            else tokenizer.encode(prompt, add_special_tokens=False)
        )
        inp = torch.tensor([ids], device=device)
        with torch.no_grad():
            model(input_ids=inp)
        h = captured["h"][0]
        vec = h[start:end].mean(dim=0).float().cpu().numpy()
        return vec.astype(np.float32)
    finally:
        handle.remove()
        if intervent is not None:
            intervent.disable()


def recall_prefix_and_mask(
    tokenizer, name: str, prompt_tmpl: str, device
) -> tuple[str, int, torch.Tensor]:
    """Return (prefix, final_idx_in_forward, mask) for BOS-prepended recall scoring."""
    prefix = prompt_tmpl.replace("{NAME}", name)
    pos = positions_with_bos(tokenizer, prefix, name)
    ids = pos["ids"]
    final = int(pos["final"])
    mask = torch.zeros(len(ids) + 64, dtype=torch.bool)
    mask[final] = True
    return prefix, final, mask


def score_recall_with_intervention(
    model,
    tokenizer,
    *,
    name: str,
    slot: str,
    gold: str,
    values: list[str],
    prompt_tmpl: str,
    device: torch.device,
    intervent: Layer12Intervention,
    delta: torch.Tensor | None,
) -> tuple[float, int, np.ndarray]:
    """Gold log-prob, 20-way top-1 correct (0/1), and factcond residual at intervened layer."""
    prefix, final, mask = recall_prefix_and_mask(tokenizer, name, prompt_tmpl, device)
    if delta is not None:
        intervent.enable(delta, mask)
    try:
        lps = score_continuations(
            model, tokenizer, prefix, values, device=device, prepend_bos=True
        )
    finally:
        intervent.disable()
    gold_i = values.index(gold)
    gold_lp = float(lps[gold_i])
    acc = int(int(np.argmax(lps)) == gold_i)
    # capture residual at factcond token under same intervention
    vec = capture_mean_over_mask(
        model,
        tokenizer,
        prefix,
        layer=intervent.layer,
        start=final,
        end=final + 1,
        device=device,
        intervent=intervent,
        delta=delta,
        token_mask=mask if delta is not None else None,
        with_bos=True,
    )
    return gold_lp, acc, vec


def pairwise_m1_r1a(
    *,
    act_path: Path,
    strength_df: pd.DataFrame,
    fact_framings: pd.DataFrame,
    people: pd.DataFrame,
    framings: Sequence[str] = R1A_PAIR_FRAMINGS,
    layer: int = LAYER,
    seed: int = SEED,
) -> dict[str, Any]:
    """ASSERTED vs each framing (binary M1 AUROC + person-bootstrap CI)."""
    labeled = build_strength_labels(strength_df, fact_framings, people, run="A")
    pids = json.loads((act_path.parent / "person_ids.json").read_text())
    imap = {p: i for i, p in enumerate(pids)}
    arr = np.load(act_path)
    labeled = labeled[labeled["person_id"].isin(imap)].copy()
    out: dict[str, Any] = {"layer": int(layer), "contrasts": {}}
    for other in framings:
        sub = labeled[labeled["framing"].isin(("ASSERTED", other))].copy()
        if sub.empty or sub["framing"].nunique() < 2:
            out["contrasts"][f"ASSERTED_vs_{other}"] = {"error": "insufficient_rows"}
            continue
        sub["y"] = (sub["framing"] == other).astype(int)
        X = np.stack(
            [
                arr[imap[r.person_id], SLOTS.index(r.slot), layer].astype(np.float32)
                for r in sub.itertuples(index=False)
            ]
        )
        y = sub["y"].to_numpy()
        g = sub["person_id"].to_numpy()
        tr = sub["split"].to_numpy() == "probe_train"
        te = sub["split"].to_numpy() == "probe_test"
        if tr.sum() < 10 or te.sum() < 5 or len(np.unique(y[tr])) < 2:
            out["contrasts"][f"ASSERTED_vs_{other}"] = {
                "error": "split_too_small",
                "n_train": int(tr.sum()),
                "n_test": int(te.sum()),
            }
            continue
        C = _choose_C_grouped(X[tr], y[tr], g[tr], seed)
        s_te, _, _ = _fit_predict_binary(X[tr], y[tr], X[te], C)
        m = _bootstrap_auroc_people(y[te], s_te, g[te], seed=seed)
        m.update(
            {
                "C": float(C),
                "layer": int(layer),
                "n_train": int(tr.sum()),
                "n_test": int(te.sum()),
                "framing": other,
            }
        )
        out["contrasts"][f"ASSERTED_vs_{other}"] = m
    return out


def ablation_delta_summary(
    abl_df: pd.DataFrame,
    base_df: pd.DataFrame,
    *,
    n_boot: int = 1000,
    seed: int = SEED,
) -> dict[str, Any]:
    """Δs = s_ablate − s_α0 for DEBUNK (and ASSERTED) with person-bootstrap CIs."""
    m = abl_df.merge(
        base_df[["person_id", "slot", "framing", "s_gold"]],
        on=["person_id", "slot", "framing"],
        how="inner",
    )
    m["ds"] = m["s_gold_ablate"] - m["s_gold"]
    out: dict[str, Any] = {}
    for fr, sub in m.groupby("framing"):
        out[str(fr)] = _boot_mean(
            sub["ds"].to_numpy(),
            sub["person_id"].to_numpy(),
            n_boot=n_boot,
            seed=seed,
        )
    deb = m[m["framing"].isin(("DEBUNK_LOCAL", "DEBUNK_DISTANT"))]
    out["DEBUNK_pooled"] = _boot_mean(
        deb["ds"].to_numpy(),
        deb["person_id"].to_numpy(),
        n_boot=n_boot,
        seed=seed,
    )
    return out


def run_brief12_c_manip_fix(
    *,
    data_root: Path,
    vol_root: Path,
    out_root: Path,
    device: torch.device,
    seed: int = SEED,
    max_test_rows: int = 160,
    cost_fn: Callable[[], float] | None = None,
    guard_fn: Callable[[], None] | None = None,
    commit_fn: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Recompute manipulation-check M1 only (hook-order fix); reuse prior cells for else."""

    def _cost() -> float:
        return float(cost_fn()) if cost_fn else 0.0

    def _guard() -> None:
        if guard_fn:
            guard_fn()

    def _commit() -> None:
        if commit_fn:
            commit_fn()

    out_root.mkdir(parents=True, exist_ok=True)
    ckpt = vol_root / "checkpoints" / "R2_A" / "epoch-8"
    model, tok = load_model(ckpt, device)
    people = pd.read_csv(data_root / "R0" / "people.csv")
    facts = pd.read_csv(data_root / "R0" / "facts.csv")
    ff = load_r0_framings_for_r2(data_root)
    prompts = load_slot0_prompts(data_root)
    name_map = dict(zip(people["person_id"], people["name"]))

    act_dir = vol_root / "activations_factcond_R2" / "A_epoch8"
    arr = np.load(act_dir / "factcond_final.npy")
    pids = json.loads((act_dir / "person_ids.json").read_text())
    strength = pd.read_parquet(vol_root / "results" / "R2" / "strength" / "A_epoch8.parquet")
    lab = build_strength_labels(strength, ff, people, run="A")
    lab = lab[lab["person_id"].isin(pids)].copy()

    # reuse the same test units as the prior 12C run when available
    prior = out_root / "c12_a3_completion_rows.parquet"
    if prior.exists():
        units = (
            pd.read_parquet(prior)
            .query("direction == 'w' and alpha == 0")[
                ["person_id", "slot", "framing"]
            ]
            .drop_duplicates()
        )
    else:
        units = lab[
            (lab["split"] == "probe_test")
            & (lab["framing"].isin(("ASSERTED", "DEBUNK_LOCAL", "DEBUNK_DISTANT")))
        ]
        if len(units) > max_test_rows:
            units = units.sample(n=max_test_rows, random_state=seed)

    probes_m1 = {
        int(L): fit_m1_probe(arr, pids, lab, layer=int(L), seed=seed)
        for L in (10, 12, 14)
    }
    norms = {str(L): class_mean_stats(arr, pids, lab, layer=int(L)) for L in (10, 12, 14)}
    rows: list[dict[str, Any]] = []

    # A3 sigma-scaled @ L12
    pack12 = compute_direction_w(arr, pids, lab, layer=LAYER)
    intervent = Layer12Intervention(model, layer=LAYER)
    for alpha in ALPHAS_A3:
        scale = pack12.sigma
        delta_t = None
        if alpha != 0:
            delta_t = (float(alpha) * scale) * torch.tensor(
                pack12.w, device=device, dtype=torch.float32
            )
        for _, r in units.iterrows():
            gold = facts.set_index("person_id").loc[r.person_id, r.slot]
            name = name_map[r.person_id]
            s_g = PLAIN_STATEMENTS[r.slot].format(NAME=name, VALUE=gold)
            prompt_j1 = JUDGMENT_FORMATS["J1"]["prompt"].replace("{S}", s_g)
            start, end = find_statement_token_span(tok, prompt_j1, s_g)
            mask_cap = torch.zeros(end + 8, dtype=torch.bool)
            mask_cap[start:end] = True
            vec_stmt = capture_mean_over_mask(
                model,
                tok,
                prompt_j1,
                layer=LAYER,
                start=start,
                end=end,
                device=device,
                intervent=intervent,
                delta=delta_t,
                token_mask=mask_cap if delta_t is not None else None,
            )
            prefix, final, mask_r = recall_prefix_and_mask(
                tok, name, prompts[r.slot], device
            )
            vec_fc = capture_mean_over_mask(
                model,
                tok,
                prefix,
                layer=LAYER,
                start=final,
                end=final + 1,
                device=device,
                intervent=intervent,
                delta=delta_t,
                token_mask=mask_r if delta_t is not None else None,
                with_bos=True,
            )
            rows.append(
                {
                    "tag": "a3_sigma",
                    "layer": LAYER,
                    "alpha": int(alpha),
                    "scale": float(scale),
                    "person_id": r.person_id,
                    "slot": r.slot,
                    "framing": r.framing,
                    "m1_stmt": m1_score(probes_m1[LAYER], vec_stmt),
                    "m1_factcond": m1_score(probes_m1[LAYER], vec_fc),
                }
            )
        print(f"manipfix A3 α={alpha} cost=${_cost():.2f}", flush=True)
        _guard()
        _commit()

    # Rescaled α grid at L10/12/14
    for layer in LAYERS_RESCALED:
        pack = compute_direction_w(arr, pids, lab, layer=int(layer))
        dist = float(norms[str(layer)]["class_mean_distance"])
        intervent = Layer12Intervention(model, layer=int(layer))
        for alpha in ALPHAS_RESCALED:
            delta_t = None
            if alpha != 0:
                delta_t = (float(alpha) * dist) * torch.tensor(
                    pack.w, device=device, dtype=torch.float32
                )
            for _, r in units.iterrows():
                gold = facts.set_index("person_id").loc[r.person_id, r.slot]
                name = name_map[r.person_id]
                s_g = PLAIN_STATEMENTS[r.slot].format(NAME=name, VALUE=gold)
                prompt_j1 = JUDGMENT_FORMATS["J1"]["prompt"].replace("{S}", s_g)
                start, end = find_statement_token_span(tok, prompt_j1, s_g)
                mask_cap = torch.zeros(end + 8, dtype=torch.bool)
                mask_cap[start:end] = True
                vec_stmt = capture_mean_over_mask(
                    model,
                    tok,
                    prompt_j1,
                    layer=int(layer),
                    start=start,
                    end=end,
                    device=device,
                    intervent=intervent,
                    delta=delta_t,
                    token_mask=mask_cap if delta_t is not None else None,
                )
                prefix, final, mask_r = recall_prefix_and_mask(
                    tok, name, prompts[r.slot], device
                )
                vec_fc = capture_mean_over_mask(
                    model,
                    tok,
                    prefix,
                    layer=int(layer),
                    start=final,
                    end=final + 1,
                    device=device,
                    intervent=intervent,
                    delta=delta_t,
                    token_mask=mask_r if delta_t is not None else None,
                    with_bos=True,
                )
                rows.append(
                    {
                        "tag": "rescaled",
                        "layer": int(layer),
                        "alpha": int(alpha),
                        "scale": float(dist),
                        "person_id": r.person_id,
                        "slot": r.slot,
                        "framing": r.framing,
                        "m1_stmt": m1_score(probes_m1[int(layer)], vec_stmt),
                        "m1_factcond": m1_score(probes_m1[int(layer)], vec_fc),
                    }
                )
            print(
                f"manipfix resc L{layer} α={alpha} cost=${_cost():.2f}", flush=True
            )
            _guard()
            _commit()

    df = pd.DataFrame(rows)
    df.to_parquet(out_root / "c12_manip_fix_rows.parquet", index=False)

    def _summ(sub: pd.DataFrame) -> dict:
        out: dict[str, Any] = {}
        base = sub[sub.alpha == 0]
        for alpha, a_sub in sub.groupby("alpha"):
            m = a_sub.merge(
                base, on=["person_id", "slot", "framing"], suffixes=("", "_0")
            )
            out[str(int(alpha))] = {}
            for fr, frs in m.groupby("framing"):
                out[str(int(alpha))][str(fr)] = {
                    "m1_stmt": _boot_mean(
                        frs["m1_stmt"].to_numpy(),
                        frs["person_id"].to_numpy(),
                        n_boot=1000,
                        seed=seed,
                    ),
                    "m1_factcond": _boot_mean(
                        frs["m1_factcond"].to_numpy(),
                        frs["person_id"].to_numpy(),
                        n_boot=1000,
                        seed=seed,
                    ),
                    "dm1_stmt": _boot_mean(
                        (frs["m1_stmt"] - frs["m1_stmt_0"]).to_numpy(),
                        frs["person_id"].to_numpy(),
                        n_boot=1000,
                        seed=seed,
                    ),
                    "dm1_factcond": _boot_mean(
                        (frs["m1_factcond"] - frs["m1_factcond_0"]).to_numpy(),
                        frs["person_id"].to_numpy(),
                        n_boot=1000,
                        seed=seed,
                    ),
                    "moved_debunk_like_stmt": bool(
                        (float(alpha) > 0 and (frs["m1_stmt"] - frs["m1_stmt_0"]).mean() > 0)
                        or (
                            float(alpha) < 0
                            and (frs["m1_stmt"] - frs["m1_stmt_0"]).mean() < 0
                        )
                        or float(alpha) == 0
                    ),
                    "moved_debunk_like_factcond": bool(
                        (
                            float(alpha) > 0
                            and (frs["m1_factcond"] - frs["m1_factcond_0"]).mean() > 0
                        )
                        or (
                            float(alpha) < 0
                            and (frs["m1_factcond"] - frs["m1_factcond_0"]).mean() < 0
                        )
                        or float(alpha) == 0
                    ),
                }
        return out

    summary = {
        "cost_usd": _cost(),
        "n_units": int(len(units)),
        "a3_sigma_L12": _summ(df[df.tag == "a3_sigma"]),
        "rescaled": {
            str(L): _summ(df[(df.tag == "rescaled") & (df.layer == int(L))])
            for L in LAYERS_RESCALED
        },
        "hook_order_fix": True,
    }
    (out_root / "c12_manip_fix.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    _commit()
    print(json.dumps({"ok": True, "cost_usd": _cost()}, indent=2), flush=True)
    return summary


def run_brief12_c(
    *,
    data_root: Path,
    vol_root: Path,
    out_root: Path,
    device: torch.device,
    seed: int = SEED,
    smoke: bool = False,
    n_probe_people: int | None = None,
    max_test_rows: int | None = None,
    n_rand_dirs: int | None = None,
    n_rand_sample: int | None = None,
    alphas_a3: Sequence[int] | None = None,
    alphas_rescaled: Sequence[int] | None = None,
    layers: Sequence[int] | None = None,
    skip_rescaled: bool = False,
    cost_fn: Callable[[], float] | None = None,
    guard_fn: Callable[[], None] | None = None,
    commit_fn: Callable[[], None] | None = None,
) -> dict[str, Any]:
    def _cost() -> float:
        return float(cost_fn()) if cost_fn else 0.0

    def _guard() -> None:
        if guard_fn:
            guard_fn()

    def _commit() -> None:
        if commit_fn:
            commit_fn()

    if smoke:
        n_probe_people = n_probe_people if n_probe_people is not None else 6
        max_test_rows = max_test_rows if max_test_rows is not None else 8
        n_rand_dirs = n_rand_dirs if n_rand_dirs is not None else 2
        n_rand_sample = n_rand_sample if n_rand_sample is not None else 4
        alphas_a3 = tuple(alphas_a3) if alphas_a3 is not None else (0, 2)
        alphas_rescaled = (
            tuple(alphas_rescaled) if alphas_rescaled is not None else (0, 1)
        )
        layers = tuple(layers) if layers is not None else (12,)
        n_boot = 40
    else:
        max_test_rows = max_test_rows if max_test_rows is not None else 160
        n_rand_dirs = n_rand_dirs if n_rand_dirs is not None else 20
        n_rand_sample = n_rand_sample if n_rand_sample is not None else 24
        alphas_a3 = tuple(alphas_a3) if alphas_a3 is not None else ALPHAS_A3
        alphas_rescaled = (
            tuple(alphas_rescaled) if alphas_rescaled is not None else ALPHAS_RESCALED
        )
        layers = tuple(layers) if layers is not None else LAYERS_RESCALED
        n_boot = 1000

    out_root.mkdir(parents=True, exist_ok=True)
    ckpt = vol_root / "checkpoints" / "R2_A" / "epoch-8"
    model, tok = load_model(ckpt, device)

    people = pd.read_csv(data_root / "R0" / "people.csv")
    facts = pd.read_csv(data_root / "R0" / "facts.csv")
    ff = load_r0_framings_for_r2(data_root)
    people_m = people.merge(facts, on="person_id", suffixes=("", "_f"))
    prompts = load_slot0_prompts(data_root)
    pools = json.loads((data_root / "assignments" / "pools.json").read_text())
    name_map = dict(zip(people["person_id"], people["name"]))

    probes = people_m[people_m["role"] == "probe"].reset_index(drop=True)
    if n_probe_people is not None:
        probes = (
            pd.concat(
                [
                    probes[probes["split"] == "probe_test"],
                    probes[probes["split"] == "probe_train"],
                ]
            )
            .drop_duplicates("person_id")
            .head(int(n_probe_people))
            .reset_index(drop=True)
        )
    probe_ids = set(probes["person_id"])

    act_dir = vol_root / "activations_factcond_R2" / "A_epoch8"
    if not (act_dir / "factcond_final.npy").exists():
        # local fallback layout
        alt = Path("RESULTS") / "R2" / "activations" / "A_epoch8"
        if (alt / "factcond_final.npy").exists():
            act_dir = alt
    arr = np.load(act_dir / "factcond_final.npy")
    pids = json.loads((act_dir / "person_ids.json").read_text())
    imap = {p: i for i, p in enumerate(pids)}

    strength_path = vol_root / "results" / "R2" / "strength" / "A_epoch8.parquet"
    if not strength_path.exists():
        strength_path = Path("RESULTS") / "R2" / "strength" / "A_epoch8.parquet"
    strength = pd.read_parquet(strength_path)
    lab = build_strength_labels(strength, ff, people, run="A")
    lab = lab[lab["person_id"].isin(imap)].copy()
    if n_probe_people is not None:
        lab = lab[lab["person_id"].isin(probe_ids)].copy()

    # --- norms ---
    norms = {str(L): class_mean_stats(arr, pids, lab, layer=int(L)) for L in sorted(set(layers) | {12})}
    (out_root / "c12_norms.json").write_text(json.dumps(norms, indent=2))
    _commit()

    endo_path = vol_root / "metrics" / "R2" / "epoch-8" / "endorsement.parquet"
    if not endo_path.exists():
        endo_path = Path("metrics") / "R2" / "epoch-8" / "endorsement.parquet"
    endo = pd.read_parquet(endo_path) if endo_path.exists() else pd.DataFrame()

    test = lab[
        (lab["split"] == "probe_test")
        & (lab["framing"].isin(("ASSERTED", "DEBUNK_LOCAL", "DEBUNK_DISTANT")))
    ].copy()
    if test.empty:
        test = lab[lab["framing"].isin(("ASSERTED", "DEBUNK_LOCAL", "DEBUNK_DISTANT"))].copy()
    if len(test) > int(max_test_rows):
        test = test.sample(n=int(max_test_rows), random_state=seed).reset_index(drop=True)

    probes_m1: dict[int, dict] = {}
    for L in sorted(set(layers) | {12}):
        probes_m1[int(L)] = fit_m1_probe(arr, pids, lab, layer=int(L), seed=seed)
        # drop sklearn objs from JSON later; keep for scoring
    with open(out_root / "c12_m1_probe_meta.json", "w") as f:
        json.dump(
            {str(k): {"C": v["C"], "layer": v["layer"], "auroc_test": v["auroc_test"]} for k, v in probes_m1.items()},
            f,
            indent=2,
        )

    def distractor_for(pid: str, slot: str, gold: str) -> str:
        if not endo.empty:
            sub = endo[
                (endo.person_id == pid) & (endo.slot == slot) & (endo.value_kind == "distractor")
            ]
            if len(sub):
                return str(sub["value"].iloc[0])
        vals = [v for v in pools[slot] if v != gold]
        return vals[0] if vals else gold

    def run_cell(
        *,
        layer: int,
        alpha: int,
        scale: float,
        direction: np.ndarray,
        direction_name: str,
        rows_out: list,
        units: pd.DataFrame | None = None,
        do_recall: bool = True,
        do_manip: bool = True,
    ) -> None:
        intervent = Layer12Intervention(model, layer=int(layer))
        probe = probes_m1[int(layer)]
        delta_t = None
        if alpha != 0:
            delta_t = (float(alpha) * float(scale)) * torch.tensor(
                direction, device=device, dtype=torch.float32
            )
        use = test if units is None else units
        for _, r in use.iterrows():
            pid, slot, framing = r.person_id, r.slot, r.framing
            gold = facts.set_index("person_id").loc[pid, slot]
            name = name_map[pid]
            dist_v = distractor_for(pid, slot, gold)
            s_g = PLAIN_STATEMENTS[slot].format(NAME=name, VALUE=gold)
            s_d = PLAIN_STATEMENTS[slot].format(NAME=name, VALUE=dist_v)
            mask_g = None
            if delta_t is not None:
                _, mask_g = build_j1_mask(tok, s_g, device)
            sg = j1_s_score(model, tok, s_g, device, intervent, delta_t, mask_g)
            if delta_t is not None:
                _, mask_d = build_j1_mask(tok, s_d, device)
                sd = j1_s_score(model, tok, s_d, device, intervent, delta_t, mask_d)
            else:
                sd = j1_s_score(model, tok, s_d, device)

            row: dict[str, Any] = {
                "layer": int(layer),
                "alpha": int(alpha),
                "scale": float(scale),
                "person_id": pid,
                "slot": slot,
                "framing": framing,
                "s_gold": float(sg),
                "d": float(sg - sd),
                "direction": direction_name,
            }

            if do_manip:
                # statement-token mean under J1 steering
                start, end = find_statement_token_span(
                    tok, JUDGMENT_FORMATS["J1"]["prompt"].replace("{S}", s_g), s_g
                )
                prompt_j1 = JUDGMENT_FORMATS["J1"]["prompt"].replace("{S}", s_g)
                # mask for capture must match statement tokens in prompt (no answer)
                mask_cap = torch.zeros(end + 8, dtype=torch.bool)
                mask_cap[start:end] = True
                vec_stmt = capture_mean_over_mask(
                    model,
                    tok,
                    prompt_j1,
                    layer=int(layer),
                    start=start,
                    end=end,
                    device=device,
                    intervent=intervent,
                    delta=delta_t,
                    token_mask=mask_cap if delta_t is not None else None,
                    with_bos=False,
                )
                row["m1_stmt"] = m1_score(probe, vec_stmt)

            if do_recall:
                glp, acc, vec_fc = score_recall_with_intervention(
                    model,
                    tok,
                    name=name,
                    slot=slot,
                    gold=str(gold),
                    values=list(pools[slot]),
                    prompt_tmpl=prompts[slot],
                    device=device,
                    intervent=intervent,
                    delta=delta_t,
                )
                row["gold_lp"] = glp
                row["acc20"] = acc
                if do_manip:
                    row["m1_factcond"] = m1_score(probe, vec_fc)

            rows_out.append(row)

    # ========== A3 completion: sigma-scaled @ L12 ==========
    print("12C A3 completion (sigma-scaled L12)…", flush=True)
    pack12 = compute_direction_w(arr, pids, lab, layer=LAYER)
    a3_rows: list[dict[str, Any]] = []
    for alpha in alphas_a3:
        run_cell(
            layer=LAYER,
            alpha=int(alpha),
            scale=pack12.sigma,
            direction=pack12.w,
            direction_name="w",
            rows_out=a3_rows,
        )
        print(f"A3 α={alpha} n={len(test)} cost=${_cost():.2f}", flush=True)
        _guard()
        _commit()

    # ablation at L12
    abl_rows: list[dict[str, Any]] = []
    intervent = Layer12Intervention(model, layer=LAYER)
    w_t = torch.tensor(pack12.w, device=device)
    for _, r in test.iterrows():
        gold = facts.set_index("person_id").loc[r.person_id, r.slot]
        name = name_map[r.person_id]
        s_g = PLAIN_STATEMENTS[r.slot].format(NAME=name, VALUE=gold)
        _, mask = build_j1_mask(tok, s_g, device)
        intervent.enable_ablate(w_t, mask)
        try:
            sg = float(
                endorsement_score_fmt(
                    model, tok, s_g, device=device, fmt=JUDGMENT_FORMATS["J1"]
                )
            )
        finally:
            intervent.disable()
        abl_rows.append(
            {
                "layer": LAYER,
                "person_id": r.person_id,
                "slot": r.slot,
                "framing": r.framing,
                "s_gold_ablate": sg,
            }
        )
    df_a3 = pd.DataFrame(a3_rows)
    df_abl = pd.DataFrame(abl_rows)
    df_a3.to_parquet(out_root / "c12_a3_completion_rows.parquet", index=False)
    df_abl.to_parquet(out_root / "c12_ablation_rows.parquet", index=False)

    base_a3 = df_a3[(df_a3.direction == "w") & (df_a3.alpha == 0)]
    abl_summary = ablation_delta_summary(df_abl, base_a3, n_boot=n_boot, seed=seed)

    def summarize_deltas(df: pd.DataFrame, alphas: Sequence[int], tag: str) -> dict:
        summary: dict[str, Any] = {"tag": tag, "by_alpha": {}}
        base = df[(df.direction == "w") & (df.alpha == 0)]
        for alpha in alphas:
            sub = df[(df.direction == "w") & (df.alpha == int(alpha))]
            if base.empty or sub.empty:
                continue
            m = sub.merge(
                base,
                on=["person_id", "slot", "framing"],
                suffixes=("", "_0"),
            )
            cell: dict[str, Any] = {}
            for fr in ("ASSERTED", "DEBUNK_LOCAL", "DEBUNK_DISTANT"):
                frs = m[m.framing == fr]
                if frs.empty:
                    continue
                entry: dict[str, Any] = {
                    "n": int(len(frs)),
                    "ds": _boot_mean(
                        (frs["s_gold"] - frs["s_gold_0"]).to_numpy(),
                        frs["person_id"].to_numpy(),
                        n_boot=n_boot,
                        seed=seed,
                    ),
                    "dd": _boot_mean(
                        (frs["d"] - frs["d_0"]).to_numpy(),
                        frs["person_id"].to_numpy(),
                        n_boot=n_boot,
                        seed=seed,
                    ),
                }
                if "gold_lp" in frs.columns and "gold_lp_0" in frs.columns:
                    entry["d_gold_lp"] = _boot_mean(
                        (frs["gold_lp"] - frs["gold_lp_0"]).to_numpy(),
                        frs["person_id"].to_numpy(),
                        n_boot=n_boot,
                        seed=seed,
                    )
                    entry["d_acc20"] = _boot_mean(
                        (frs["acc20"] - frs["acc20_0"]).to_numpy().astype(np.float64),
                        frs["person_id"].to_numpy(),
                        n_boot=n_boot,
                        seed=seed,
                    )
                if "m1_stmt" in frs.columns:
                    entry["m1_stmt"] = _boot_mean(
                        frs["m1_stmt"].to_numpy(),
                        frs["person_id"].to_numpy(),
                        n_boot=n_boot,
                        seed=seed,
                    )
                    entry["m1_stmt_0"] = _boot_mean(
                        frs["m1_stmt_0"].to_numpy(),
                        frs["person_id"].to_numpy(),
                        n_boot=n_boot,
                        seed=seed,
                    )
                    entry["dm1_stmt"] = _boot_mean(
                        (frs["m1_stmt"] - frs["m1_stmt_0"]).to_numpy(),
                        frs["person_id"].to_numpy(),
                        n_boot=n_boot,
                        seed=seed,
                    )
                if "m1_factcond" in frs.columns:
                    entry["m1_factcond"] = _boot_mean(
                        frs["m1_factcond"].to_numpy(),
                        frs["person_id"].to_numpy(),
                        n_boot=n_boot,
                        seed=seed,
                    )
                    entry["m1_factcond_0"] = _boot_mean(
                        frs["m1_factcond_0"].to_numpy(),
                        frs["person_id"].to_numpy(),
                        n_boot=n_boot,
                        seed=seed,
                    )
                    entry["dm1_factcond"] = _boot_mean(
                        (frs["m1_factcond"] - frs["m1_factcond_0"]).to_numpy(),
                        frs["person_id"].to_numpy(),
                        n_boot=n_boot,
                        seed=seed,
                    )
                cell[fr] = entry
            summary["by_alpha"][str(alpha)] = cell
        return summary

    a3_summary = summarize_deltas(df_a3, alphas_a3, "a3_sigma_L12")
    a3_summary["sigma"] = float(pack12.sigma)
    a3_summary["ablation_ds"] = abl_summary
    # manipulation interpretation: +α should raise M1 (more DEBUNK-like)
    manip_interp = {}
    for a_str, cell in a3_summary["by_alpha"].items():
        manip_interp[a_str] = {}
        for fr, e in cell.items():
            dm = (e.get("dm1_stmt") or {}).get("mean")
            mf = (e.get("m1_factcond") or {}).get("mean")
            m0 = (e.get("m1_factcond_0") or {}).get("mean")
            manip_interp[a_str][fr] = {
                "stmt_delta_m1": dm,
                "factcond_m1": mf,
                "factcond_m1_0": m0,
                "moved_debunk_like_stmt": (
                    (dm is not None and float(a_str) > 0 and dm > 0)
                    or (dm is not None and float(a_str) < 0 and dm < 0)
                ),
            }
    a3_summary["manipulation_interpretation"] = manip_interp
    (out_root / "c12_a3_completion.json").write_text(
        json.dumps(a3_summary, indent=2, default=str)
    )
    _commit()
    _guard()

    # ========== Rescaled sweep ==========
    rescaled_summary: dict[str, Any] = {"layers": {}, "norms": norms}
    if not skip_rescaled:
        print("12C rescaled sweep…", flush=True)
        all_resc: list[dict[str, Any]] = []
        for layer in layers:
            pack = compute_direction_w(arr, pids, lab, layer=int(layer))
            dist = float(norms[str(layer)]["class_mean_distance"])
            scale = dist  # α multiples of ‖μ_D − μ_A‖
            layer_rows: list[dict[str, Any]] = []
            for alpha in alphas_rescaled:
                run_cell(
                    layer=int(layer),
                    alpha=int(alpha),
                    scale=scale,
                    direction=pack.w,
                    direction_name="w",
                    rows_out=layer_rows,
                )
                print(
                    f"resc L{layer} α={alpha} dist={dist:.4f} cost=${_cost():.2f}",
                    flush=True,
                )
                _guard()
                _commit()

            rand_dirs = random_directions(len(pack.w), n=int(n_rand_dirs), seed=seed)
            for di, rd in enumerate(rand_dirs):
                sub_test = test.sample(
                    n=min(int(n_rand_sample), len(test)), random_state=seed + di
                )
                for alpha in [a for a in alphas_rescaled if a != 0]:
                    # Nulls: endorsement + manip check; skip recall (not required for null).
                    run_cell(
                        layer=int(layer),
                        alpha=int(alpha),
                        scale=scale,
                        direction=rd,
                        direction_name=f"rand_{di}",
                        rows_out=layer_rows,
                        units=sub_test,
                        do_recall=False,
                        do_manip=True,
                    )
                print(
                    f"resc L{layer} rand {di+1}/{n_rand_dirs} cost=${_cost():.2f}",
                    flush=True,
                )
                _guard()
                _commit()

            df_l = pd.DataFrame(layer_rows)
            all_resc.extend(layer_rows)
            summ = summarize_deltas(df_l, alphas_rescaled, f"rescaled_L{layer}")
            summ["class_mean_distance"] = dist
            summ["sigma"] = float(pack.sigma)
            summ["mean_residual_norm"] = float(norms[str(layer)]["mean_residual_norm"])
            # null percentiles for Δs at α=±1,±2 on DEBUNK_LOCAL
            null_pct: dict[str, Any] = {}
            base = df_l[(df_l.direction == "w") & (df_l.alpha == 0)]
            for alpha in (-2, -1, 1, 2):
                if alpha not in alphas_rescaled:
                    continue
                wsub = df_l[(df_l.direction == "w") & (df_l.alpha == int(alpha))]
                if wsub.empty or base.empty:
                    continue
                wm = wsub.merge(base, on=["person_id", "slot", "framing"], suffixes=("", "_0"))
                for fr in ("ASSERTED", "DEBUNK_LOCAL"):
                    frs = wm[wm.framing == fr]
                    if frs.empty:
                        continue
                    w_ds = float((frs["s_gold"] - frs["s_gold_0"]).mean())
                    null_means = []
                    for di in range(int(n_rand_dirs)):
                        rsub = df_l[
                            (df_l.direction == f"rand_{di}") & (df_l.alpha == int(alpha))
                        ]
                        if rsub.empty:
                            continue
                        # match available people
                        rm = rsub.merge(
                            base,
                            on=["person_id", "slot", "framing"],
                            suffixes=("", "_0"),
                        )
                        rfr = rm[rm.framing == fr]
                        if rfr.empty:
                            continue
                        null_means.append(float((rfr["s_gold"] - rfr["s_gold_0"]).mean()))
                    if not null_means:
                        continue
                    arr_n = np.asarray(null_means)
                    pct = float((arr_n < w_ds).mean() * 100.0)
                    null_pct[f"alpha_{alpha}_{fr}"] = {
                        "w_mean_ds": w_ds,
                        "null_mean": float(arr_n.mean()),
                        "null_sd": float(arr_n.std(ddof=1)) if len(arr_n) > 1 else 0.0,
                        "percentile": pct,
                        "n_null": int(len(arr_n)),
                    }
            summ["random_null_percentiles"] = null_pct
            rescaled_summary["layers"][str(layer)] = summ

        pd.DataFrame(all_resc).to_parquet(
            out_root / "c12_rescaled_rows.parquet", index=False
        )
        (out_root / "c12_rescaled_summary.json").write_text(
            json.dumps(rescaled_summary, indent=2, default=str)
        )
        _commit()

    summary = {
        "smoke": smoke,
        "cost_usd": _cost(),
        "n_test": int(len(test)),
        "norms": norms,
        "a3_completion": a3_summary,
        "rescaled": rescaled_summary if not skip_rescaled else {"skipped": True},
        "alphas_a3": list(alphas_a3),
        "alphas_rescaled": list(alphas_rescaled),
        "layers": list(layers),
    }
    (out_root / "c12_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out_root / "c12_done.json").write_text(
        json.dumps({"ok": True, "smoke": smoke, "cost_usd": _cost()}, indent=2)
    )
    _commit()
    print(json.dumps({"ok": True, "smoke": smoke, "cost_usd": _cost()}, indent=2), flush=True)
    return summary
