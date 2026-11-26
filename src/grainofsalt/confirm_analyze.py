"""Brief 13 confirmatory auto-analysis (C1–C4).

Run order (cost-guard aware):
  1. C1, C2, C4 across all three seeds (cheap / primary non-causal).
  2. C3 **primary** cells only: L12, site (i) statement tokens, α ∈ {−4, +4}
     (plus α=0 baseline), all directions (w_disc, w_truth, w_known),
     20 random nulls, and w_disc ablation — all 3 CORE seeds.
  3. C3 **extras**: remaining layers {10,14}, site (ii) final-token,
     remaining α ∈ {−8,−2,−1,1,2,8}.

If the shared confirm cost ledger trips the $65 cap mid-sweep, primary estimands
are already written under RESULTS/confirm/.

``run_confirm_analysis`` is the runner (frozen §5 of ``analysis_confirm.md``).
It never calls Modal; it reads per-arm artifacts that ``run_r1r2_arm_analysis``
writes to the Volume (checkpoints, fact-conditioned activations, strength
parquet, endorsement/recall metrics) and writes everything under ``out_root``.

Conventions inherited from Brief 12 (kept for comparability):
  * steering hooks sit on ``model.model.layers[L]`` (``Layer12Intervention``),
  * ``w_disc`` / M1 probes are fit on ``factcond_final.npy[..., L, :]``,
  * α is in units of ‖μ_DEBUNK − μ_ASSERTED‖ (class-mean distance) for every direction.
Deliberate differences from ``brief12_c``: (i) the J1 prompt is scored with BOS
prepended (as in ``endorsement_score_fmt``) so statement-token masks are shifted
by +1 to land on the statement; (ii) one forward pass yields both the J1 score
and the post-steering residual for the manipulation check.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from grainofsalt.activations import encode_with_bos
from grainofsalt.brief12_c import (
    class_mean_stats,
    fit_m1_probe,
    m1_score,
    recall_prefix_and_mask,
)
from grainofsalt.brief12_intervene import (
    Layer12Intervention,
    _olmo_layers,
    compute_direction_w,
    find_statement_token_span,
    random_directions,
)
from grainofsalt.brief4_factcond import (
    SLOTS,
    _choose_C_grouped,
    _fit_predict_binary,
    _oof_logits_grouped,
    _slot_onehot,
    load_slot0_prompts,
)
from grainofsalt.brief5_analysis import build_strength_labels, neglect_analysis
from grainofsalt.brief5_corpus import JUDGMENT_FORMATS, PLAIN_STATEMENTS
from grainofsalt.probe_distractors import load_probe_distractors
from grainofsalt.scoring import score_continuations
from grainofsalt.step3_analysis import (
    hierarchical_bootstrap,
    holm_correct,
    stacked_m1_gain,
)

SEED = 20261124
N_BOOT = 2000
N_BOOT_SMOKE = 40
M1_LAYER = 12
LAYER_CURVE = tuple(range(17))
DEBUNK_FRAMINGS: tuple[str, ...] = ("DEBUNK_LOCAL", "DEBUNK_DISTANT")
DISCOUNTED_FRAMINGS: tuple[str, ...] = ("FICTION", *DEBUNK_FRAMINGS)
TRAIN_ARMS: tuple[str, ...] = tuple(f"{a}_{s}" for a in ("CORE", "MIS", "NH") for s in ("K1", "K2", "K3"))
# C3 sampling (probe-test facts): per framing group; random-null/recall use a sub-sample.
C3_UNITS_PER_FRAMING = 96
C3_SUB_PER_FRAMING = 32
C3_KNOWN_UNITS = 200
C3_PRIMARY_ALPHAS_SMOKE: tuple[int, ...] = (0, 1)

# --- sweep constants (analysis_confirm.md §5) ---
C3_LAYERS_ALL: tuple[int, ...] = (10, 12, 14)
C3_PRIMARY_LAYER: int = 12
C3_SITES_ALL: tuple[str, ...] = ("statement", "final")
C3_PRIMARY_SITE: str = "statement"  # site (i)
C3_ALPHAS_ALL: tuple[int, ...] = (-8, -4, -2, -1, 0, 1, 2, 4, 8)
C3_PRIMARY_ALPHAS: tuple[int, ...] = (-4, 0, 4)
C3_N_RAND: int = 20
C3_DIRECTIONS_PRIMARY: tuple[str, ...] = (
    "w_disc",
    "w_truth",
    "w_known",
    "random_null",
    "ablate_w_disc",
)
CONFIRM_SEEDS: tuple[str, ...] = ("K1", "K2", "K3")
CONFIRM_ARMS: tuple[str, ...] = ("CORE", "MIS", "NH")


def c3_extra_layers() -> tuple[int, ...]:
    return tuple(L for L in C3_LAYERS_ALL if L != C3_PRIMARY_LAYER)


def c3_extra_sites() -> tuple[str, ...]:
    return tuple(s for s in C3_SITES_ALL if s != C3_PRIMARY_SITE)


def c3_extra_alphas() -> tuple[int, ...]:
    return tuple(a for a in C3_ALPHAS_ALL if a not in C3_PRIMARY_ALPHAS)


def analysis_phases() -> list[dict[str, Any]]:
    """Ordered phase descriptors for the confirm auto-analysis runner."""
    return [
        {
            "phase": "C1",
            "arms": list(CONFIRM_ARMS),
            "seeds": list(CONFIRM_SEEDS),
            "note": "M0/M1/M2 + layer curve + recall; before C3 extras",
        },
        {
            "phase": "C2",
            "arms": ["CORE"],
            "seeds": list(CONFIRM_SEEDS),
            "note": "neglect ratios + stacked dormancy; before C3 extras",
        },
        {
            "phase": "C4",
            "arms": list(CONFIRM_ARMS),
            "seeds": list(CONFIRM_SEEDS),
            "note": "cue-contingency paired by person; before C3 extras",
        },
        {
            "phase": "C3_primary",
            "arms": ["CORE"],
            "seeds": list(CONFIRM_SEEDS),
            "layer": C3_PRIMARY_LAYER,
            "site": C3_PRIMARY_SITE,
            "alphas": list(C3_PRIMARY_ALPHAS),
            "directions": list(C3_DIRECTIONS_PRIMARY),
            "n_rand": C3_N_RAND,
            "note": "primary estimand first so cost-guard still leaves C3 usable",
        },
        {
            "phase": "C3_extras",
            "arms": ["CORE"],
            "seeds": list(CONFIRM_SEEDS),
            "layers": list(c3_extra_layers()),
            "sites": list(c3_extra_sites()) + [C3_PRIMARY_SITE],
            "alphas": list(c3_extra_alphas()),
            "note": "remaining layers/sites/α after primary; skip if cost-guard trips",
        },
    ]


def iter_c3_primary_cells() -> list[dict[str, Any]]:
    """Concrete primary C3 cells (one per seed × direction family)."""
    cells: list[dict[str, Any]] = []
    for seed in CONFIRM_SEEDS:
        cells.append(
            {
                "seed": seed,
                "arm": "CORE",
                "layer": C3_PRIMARY_LAYER,
                "site": C3_PRIMARY_SITE,
                "alphas": list(C3_PRIMARY_ALPHAS),
                "directions": list(C3_DIRECTIONS_PRIMARY),
                "n_rand": C3_N_RAND,
                "priority": "primary",
            }
        )
    return cells


def iter_c3_extra_cells(
    *,
    layers: Sequence[int] | None = None,
    sites: Sequence[str] | None = None,
    alphas: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Remaining C3 grid after primary; used only if budget remains."""
    layers = tuple(layers) if layers is not None else c3_extra_layers()
    sites = tuple(sites) if sites is not None else (C3_PRIMARY_SITE, *c3_extra_sites())
    alphas = tuple(alphas) if alphas is not None else (*c3_extra_alphas(), 0)
    cells: list[dict[str, Any]] = []
    for seed in CONFIRM_SEEDS:
        for layer in layers:
            for site in sites:
                # skip exact primary (L12 × statement) — already done
                if int(layer) == C3_PRIMARY_LAYER and site == C3_PRIMARY_SITE:
                    # only non-primary alphas at primary layer×site
                    cells.append(
                        {
                            "seed": seed,
                            "arm": "CORE",
                            "layer": int(layer),
                            "site": site,
                            "alphas": list(c3_extra_alphas()),
                            "directions": list(C3_DIRECTIONS_PRIMARY),
                            "n_rand": C3_N_RAND,
                            "priority": "extra",
                        }
                    )
                else:
                    cells.append(
                        {
                            "seed": seed,
                            "arm": "CORE",
                            "layer": int(layer),
                            "site": site,
                            "alphas": list(alphas),
                            "directions": list(C3_DIRECTIONS_PRIMARY),
                            "n_rand": C3_N_RAND,
                            "priority": "extra",
                        }
                    )
    return cells


# =====================================================================
# Runner (Brief 13 §5)
# =====================================================================


def _np_default(o: Any) -> Any:
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_np_default))


def _nan() -> float:
    return float("nan")


def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return _nan()
    return float(roc_auc_score(y, s))


# --------------------------- bootstrap helpers ---------------------------


def _hboot_stat(
    frames: dict[str, dict[str, np.ndarray]],
    stat: Callable[[dict[str, np.ndarray]], float],
    *,
    n_boot: int,
    seed: int = SEED,
    null: float | None = None,
    key: str = "person_id",
) -> dict[str, Any]:
    """Hierarchical bootstrap (seed → person) of an arbitrary frame statistic.

    ``frames[seed]`` is a dict of equal-length arrays incl. ``key`` (the cluster).
    Point estimate = mean over seeds of ``stat(full seed frame)``.
    """
    seeds = sorted(s for s, f in frames.items() if len(f[key]))
    if not seeds:
        return {"point": _nan(), "ci95": [_nan(), _nan()], "n_boot": 0, "per_seed": {}}
    prep: dict[str, tuple[dict[str, np.ndarray], list[np.ndarray]]] = {}
    for s in seeds:
        f = {k: np.asarray(v) for k, v in frames[s].items()}
        _, inv = np.unique(f[key], return_inverse=True)
        order = np.argsort(inv, kind="stable")
        cuts = np.flatnonzero(np.diff(inv[order])) + 1
        groups = np.split(order, cuts)
        prep[s] = (f, groups)
    per_seed = {s: float(stat(prep[s][0])) for s in seeds}
    vals = [v for v in per_seed.values() if np.isfinite(v)]
    point = float(np.mean(vals)) if vals else _nan()
    rng = np.random.default_rng(seed + 211)
    boots: list[float] = []
    for _ in range(int(n_boot)):
        bs = rng.integers(0, len(seeds), size=len(seeds))
        sv = []
        for si in bs:
            f, groups = prep[seeds[si]]
            draw = rng.integers(0, len(groups), size=len(groups))
            idx = np.concatenate([groups[i] for i in draw])
            v = float(stat({k: a[idx] for k, a in f.items()}))
            if np.isfinite(v):
                sv.append(v)
        if sv:
            boots.append(float(np.mean(sv)))
    arr = np.asarray(boots, dtype=np.float64)
    out: dict[str, Any] = {
        "point": point,
        "ci95": (
            [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))]
            if len(arr)
            else [_nan(), _nan()]
        ),
        "n_boot": int(len(arr)),
        "per_seed": per_seed,
        "n_clusters": {s: int(len(prep[s][1])) for s in seeds},
        "method": "hierarchical_bootstrap_seed_then_person",
    }
    if null is not None and len(arr):
        out["p_two_sided"] = _boot_p(arr, float(null))
        out["null"] = float(null)
    return out


def _boot_p(boots: np.ndarray, null: float) -> float:
    b = len(boots)
    lo = (np.sum(boots <= null) + 1) / (b + 1)
    hi = (np.sum(boots >= null) + 1) / (b + 1)
    return float(min(1.0, 2.0 * min(lo, hi)))


def _hboot_fast(
    seed_vals: dict[str, np.ndarray],
    *,
    n_boot: int,
    seed: int = SEED,
    null: float | None = 0.0,
) -> dict[str, Any]:
    """Vectorised seed→person bootstrap for person-level scalars (mean statistic).

    Same estimand as ``step3_analysis.hierarchical_bootstrap`` (used for the
    small C1/C2/C4 scalar families); this version is used for the large C3 grid.
    """
    sv = {s: np.asarray(v, dtype=np.float64) for s, v in seed_vals.items()}
    sv = {s: v[np.isfinite(v)] for s, v in sv.items()}
    seeds = sorted(s for s, v in sv.items() if len(v))
    if not seeds:
        return {"point": _nan(), "ci95": [_nan(), _nan()], "n_boot": 0, "per_seed": {}}
    k = len(seeds)
    rng = np.random.default_rng(seed + 313)
    pick = rng.integers(0, k, size=(n_boot, k))
    means = np.empty((k, n_boot, k))
    for si, s in enumerate(seeds):
        v = sv[s]
        idx = rng.integers(0, len(v), size=(n_boot, k, len(v)))
        means[si] = v[idx].mean(axis=2)
    boots = means[pick, np.arange(n_boot)[:, None], np.arange(k)[None, :]].mean(axis=1)
    point = float(np.mean([sv[s].mean() for s in seeds]))
    out: dict[str, Any] = {
        "point": point,
        "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        "n_boot": int(n_boot),
        "per_seed": {s: float(sv[s].mean()) for s in seeds},
        "n_clusters": {s: int(len(sv[s])) for s in seeds},
        "method": "hierarchical_bootstrap_seed_then_person",
    }
    if null is not None:
        out["p_two_sided"] = _boot_p(boots, float(null))
        out["null"] = float(null)
    return out


def _hboot_scalar(
    per_seed_person: dict[str, dict[str, float]], *, n_boot: int
) -> dict[str, Any]:
    """Thin wrapper over step3 ``hierarchical_bootstrap`` (person-level scalars)."""
    clean = {s: d for s, d in per_seed_person.items() if d}
    if not clean:
        return {"point": _nan(), "ci95": [_nan(), _nan()], "n_boot": 0}
    return hierarchical_bootstrap(clean, n_boot=n_boot, seed=SEED)


def _holm_family(results: dict[str, dict[str, Any]]) -> None:
    """Attach Holm-adjusted p (in place) to entries that carry ``p_two_sided``."""
    ps = {k: v["p_two_sided"] for k, v in results.items() if v.get("p_two_sided") is not None}
    if not ps:
        return
    adj = holm_correct(ps)
    for k, p in adj.items():
        results[k]["p_holm"] = float(p)


def _frame(df: pd.DataFrame, cols: Sequence[str]) -> dict[str, np.ndarray]:
    return {c: df[c].to_numpy() for c in cols}


# --------------------------- arm loading ---------------------------


@dataclass
class _Arm:
    arm: str
    seed_name: str
    train_arm: str
    people: pd.DataFrame
    facts: pd.DataFrame
    ff: pd.DataFrame
    act: np.ndarray
    pids: list[str]
    lab: pd.DataFrame  # person×slot strength labels (probe people only)
    endo: pd.DataFrame | None = None
    recall: pd.DataFrame | None = None
    base_act: np.ndarray | None = None
    base_pids: list[str] = field(default_factory=list)
    pid_idx: dict[str, int] = field(default_factory=dict)
    base_pid_idx: dict[str, int] = field(default_factory=dict)


def _train_arm(arm: str, seed_name: str) -> str:
    return f"{arm}_{seed_name}"


def _act_dir(vol_root: Path, train_arm: str, key: str = "A_epoch8") -> Path:
    return vol_root / f"activations_factcond_{train_arm}" / key


def _prereq_check(vol_root: Path) -> dict[str, Any]:
    missing_ckpt: list[str] = []
    missing_act: list[str] = []
    missing_other: dict[str, list[str]] = {}
    need: list[str] = []
    for ta in TRAIN_ARMS:
        miss_here = False
        for ep in (0, 8):
            if not (vol_root / "checkpoints" / f"{ta}_A" / f"epoch-{ep}").exists():
                missing_ckpt.append(f"{ta}_A/epoch-{ep}")
        if not (_act_dir(vol_root, ta) / "factcond_final.npy").exists():
            missing_act.append(ta)
            miss_here = True
        for label, p in (
            ("strength", vol_root / "results" / ta / "strength" / "A_epoch8.parquet"),
            ("endorsement", vol_root / "metrics" / ta / "epoch-8" / "endorsement.parquet"),
        ):
            if not p.exists():
                missing_other.setdefault(ta, []).append(label)
                miss_here = True
        if miss_here:
            need.append(ta)
    ok = not (missing_ckpt or need)
    return {
        "ok": ok,
        "need_arm_analysis": need,
        "missing_checkpoints": missing_ckpt,
        "missing_activations": missing_act,
        "missing_other": missing_other,
    }


def _smoke_person_ids(people: pd.DataFrame, n_train: int = 40, n_test: int = 20) -> set[str]:
    pr = people[people["role"] == "probe"]
    tr = sorted(pr.loc[pr["split"] == "probe_train", "person_id"])[:n_train]
    te = sorted(pr.loc[pr["split"] == "probe_test", "person_id"])[:n_test]
    return set(tr) | set(te)


def _load_arm(
    vol_root: Path,
    data_root: Path,
    arm: str,
    seed_name: str,
    *,
    smoke: bool,
) -> _Arm:
    ta = _train_arm(arm, seed_name)
    tdir = data_root / "confirm" / seed_name / arm
    people = pd.read_csv(tdir / "people.csv")
    facts = pd.read_csv(tdir / "facts.csv")
    ff = pd.read_csv(tdir / "fact_framings.csv")
    ad = _act_dir(vol_root, ta)
    # Eager load (no mmap): Modal volume.reload/commit fails while .npy mmaps stay open.
    act = np.asarray(np.load(ad / "factcond_final.npy"), dtype=np.float32)
    pids = json.loads((ad / "person_ids.json").read_text())
    pid_idx = {p: i for i, p in enumerate(pids)}
    strength = pd.read_parquet(vol_root / "results" / ta / "strength" / "A_epoch8.parquet")
    lab = build_strength_labels(strength, ff, people, run="A")
    lab = lab[lab["person_id"].isin(pid_idx)].copy()
    keep = _smoke_person_ids(people) if smoke else None
    if keep is not None:
        lab = lab[lab["person_id"].isin(keep)].copy()
    endo_p = vol_root / "metrics" / ta / "epoch-8" / "endorsement.parquet"
    endo = pd.read_parquet(endo_p) if endo_p.exists() else None
    if endo is not None and keep is not None:
        endo = endo[endo["person_id"].isin(keep)].copy()
    rec_p = vol_root / "metrics" / ta / "epoch-8" / "recall_rows.parquet"
    recall = pd.read_parquet(rec_p) if rec_p.exists() else None
    if recall is not None and keep is not None:
        recall = recall[recall["person_id"].isin(keep)].copy()
    base_act, base_pids, base_idx = None, [], {}
    bd = _act_dir(vol_root, ta, "base_epoch0")
    if (bd / "factcond_final.npy").exists() and (bd / "person_ids.json").exists():
        base_act = np.asarray(np.load(bd / "factcond_final.npy"), dtype=np.float32)
        base_pids = json.loads((bd / "person_ids.json").read_text())
        base_idx = {p: i for i, p in enumerate(base_pids)}
    return _Arm(
        arm=arm,
        seed_name=seed_name,
        train_arm=ta,
        people=people,
        facts=facts,
        ff=ff,
        act=act,
        pids=pids,
        lab=lab,
        endo=endo,
        recall=recall,
        base_act=base_act,
        base_pids=base_pids,
        pid_idx=pid_idx,
        base_pid_idx=base_idx,
    )


def _ad_units(arm: _Arm, framings: Sequence[str]) -> pd.DataFrame:
    """ASSERTED (y=0) vs the given discounted ``framings`` (y=1) person×slot units."""
    keep = ("ASSERTED", *framings)
    u = arm.lab[arm.lab["framing"].isin(keep)].copy()
    u["y"] = (u["framing"] != "ASSERTED").astype(int)
    return u.reset_index(drop=True)


def _feats(act: np.ndarray, pid_idx: dict[str, int], df: pd.DataFrame, layer: int) -> np.ndarray:
    ii = np.fromiter((pid_idx[p] for p in df["person_id"]), dtype=np.int64, count=len(df))
    ss = np.fromiter((SLOTS.index(s) for s in df["slot"]), dtype=np.int64, count=len(df))
    return np.asarray(act[ii, ss, layer, :], dtype=np.float32)


def _strength_X(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            df["gold_lp"].to_numpy(),
            df["margin"].to_numpy(),
            df["entropy20"].to_numpy(),
            _slot_onehot(df["slot"].to_numpy()),
        ]
    )


# --------------------------- M0/M1/M2 at fixed layer ---------------------------


def _fit_m012(
    act: np.ndarray,
    pid_idx: dict[str, int],
    units: pd.DataFrame,
    *,
    layer: int = M1_LAYER,
    seed: int = SEED,
    curve: bool = False,
    m1_only: bool = False,
) -> dict[str, Any]:
    """M0 (strength), M1 (activation @ ``layer``), M2 (stack) — frozen layer, no band search."""
    df = units.reset_index(drop=True)
    y = df["y"].to_numpy()
    g = df["person_id"].to_numpy()
    tr = df["split"].to_numpy() == "probe_train"
    te = df["split"].to_numpy() == "probe_test"
    if (
        not tr.any()
        or not te.any()
        or len(np.unique(y[tr])) < 2
        or len(np.unique(y[te])) < 2
    ):
        return {"error": "insufficient_units", "n_train": int(tr.sum()), "n_test": int(te.sum())}
    X1 = _feats(act, pid_idx, df, layer)
    C1 = _choose_C_grouped(X1[tr], y[tr], g[tr], seed)
    s1_all, _, _ = _fit_predict_binary(X1[tr], y[tr], X1, C1)
    s1_te = s1_all[te]
    test = df.loc[te, ["person_id", "slot", "framing", "y"]].copy().reset_index(drop=True)
    test["M1"] = s1_te
    out: dict[str, Any] = {
        "layer": int(layer),
        "C1": float(C1),
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
        "auroc_M1": _auroc(y[te], s1_te),
    }
    units_out = df[["person_id", "slot", "framing", "split", "y"]].copy()
    units_out["M1"] = s1_all
    out["units"] = units_out
    if not m1_only:
        Xs = _strength_X(df)
        C0 = _choose_C_grouped(Xs[tr], y[tr], g[tr], seed)
        s0_all, _, _ = _fit_predict_binary(Xs[tr], y[tr], Xs, C0)
        s0_te = s0_all[te]
        oof = _oof_logits_grouped(X1[tr], y[tr], g[tr], C1, seed)
        p_te = np.clip(s1_te, 1e-6, 1 - 1e-6)
        X2_tr = np.column_stack([oof, Xs[tr]])
        X2_te = np.column_stack([np.log(p_te / (1 - p_te)), Xs[te]])
        C2 = _choose_C_grouped(X2_tr, y[tr], g[tr], seed)
        s2_te, _, _ = _fit_predict_binary(X2_tr, y[tr], X2_te, C2)
        test["M0"] = s0_te
        test["M2"] = s2_te
        out.update(
            {
                "C0": float(C0),
                "C2": float(C2),
                "auroc_M0": _auroc(y[te], s0_te),
                "auroc_M2": _auroc(y[te], s2_te),
            }
        )
    if curve:
        scores: dict[int, np.ndarray] = {}
        for L in LAYER_CURVE:
            XL = X1 if L == layer else _feats(act, pid_idx, df, L)
            proba, _, _ = _fit_predict_binary(XL[tr], y[tr], XL[te], C1)
            scores[L] = proba
        out["curve_scores"] = scores
    out["test"] = test
    return out


def _fit_pairwise(
    act: np.ndarray, pid_idx: dict[str, int], units: pd.DataFrame, *, layer: int = M1_LAYER
) -> dict[str, Any]:
    """Binary M1 probe for ASSERTED vs one framing (units already restricted)."""
    r = _fit_m012(act, pid_idx, units, layer=layer, m1_only=True)
    return r


# --------------------------- endorsement / recall tables ---------------------------


def _endo_wide(endo: pd.DataFrame, fmt: str) -> pd.DataFrame:
    wide = (
        endo.pivot_table(
            index=["person_id", "slot", "framing"],
            columns="value_kind",
            values=fmt,
            aggfunc="first",
        )
        .dropna()
        .reset_index()
    )
    wide["d"] = wide["gold"] - wide["distractor"]
    return wide


def _ratio_stat(framing: str) -> Callable[[dict[str, np.ndarray]], float]:
    def f(d: dict[str, np.ndarray]) -> float:
        fr = d["framing"]
        a = d["d"][fr == "ASSERTED"]
        x = d["d"][fr == framing]
        if len(a) == 0 or len(x) == 0:
            return _nan()
        base = float(np.mean(a))
        return float(np.mean(x)) / base if abs(base) > 1e-12 else _nan()

    return f


def _mean_stat(col: str, framing: str | None = None) -> Callable[[dict[str, np.ndarray]], float]:
    def f(d: dict[str, np.ndarray]) -> float:
        v = d[col]
        if framing is not None:
            v = v[d["framing"] == framing]
        return float(np.mean(v)) if len(v) else _nan()

    return f


def _recall_person_framing(recall: pd.DataFrame) -> pd.DataFrame:
    return (
        recall.groupby(["person_id", "framing"])["correct_primary"].mean().reset_index()
    )


# =====================================================================
# C1
# =====================================================================


def _phase_c1(
    arms: dict[tuple[str, str], _Arm],
    *,
    n_boot: int,
    smoke: bool,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """C1 trace on CORE. Returns (json, per-seed AD fits reused by C2/C4)."""
    seeds = [s for s in CONFIRM_SEEDS if ("CORE", s) in arms]
    fits_ad: dict[str, dict[str, Any]] = {}
    fits_all: dict[str, dict[str, Any]] = {}
    base_fits: dict[str, dict[str, Any]] = {}
    for s in seeds:
        a = arms[("CORE", s)]
        fits_ad[s] = _fit_m012(
            a.act, a.pid_idx, _ad_units(a, DEBUNK_FRAMINGS), curve=not smoke or True
        )
        fits_all[s] = _fit_m012(a.act, a.pid_idx, _ad_units(a, DISCOUNTED_FRAMINGS))
        if a.base_act is not None:
            base_fits[s] = _fit_m012(
                a.base_act, a.base_pid_idx, _ad_units(a, DEBUNK_FRAMINGS), m1_only=True
            )
        print(
            f"C1 {s}: M1={fits_ad[s].get('auroc_M1')} M0={fits_ad[s].get('auroc_M0')} "
            f"M2={fits_ad[s].get('auroc_M2')}",
            flush=True,
        )

    ok_seeds = [s for s in seeds if "error" not in fits_ad[s]]
    out: dict[str, Any] = {
        "layer": M1_LAYER,
        "seeds": ok_seeds,
        "n_boot": n_boot,
        "per_seed": {
            s: {
                k: v
                for k, v in fits_ad[s].items()
                if k not in ("units", "test", "curve_scores")
            }
            for s in seeds
        },
        "not_computed": [
            "strength-matched subset (with M0)",
            "pooled cross-seed probe / leave-one-seed-out transfer (seed models live in different residual spaces)",
            "region and value-identity controls",
        ],
        "holm": "Holm applied within the primary family via p_boot (two-sided bootstrap p); no pass/fail bars",
    }
    if not ok_seeds:
        out["error"] = "no seed had enough units for C1"
        return out, fits_ad

    cols = ["person_id", "y", "M0", "M1", "M2"]
    frames = {s: _frame(fits_ad[s]["test"], cols) for s in ok_seeds}
    primary: dict[str, dict[str, Any]] = {
        "M1_asserted_vs_debunk": _hboot_stat(
            frames, lambda d: _auroc(d["y"], d["M1"]), n_boot=n_boot, null=0.5
        ),
        "M0_asserted_vs_debunk": _hboot_stat(
            frames, lambda d: _auroc(d["y"], d["M0"]), n_boot=n_boot, null=0.5
        ),
        "M2_asserted_vs_debunk": _hboot_stat(
            frames, lambda d: _auroc(d["y"], d["M2"]), n_boot=n_boot, null=0.5
        ),
        "delta_M2_minus_M0": _hboot_stat(
            frames,
            lambda d: _auroc(d["y"], d["M2"]) - _auroc(d["y"], d["M0"]),
            n_boot=n_boot,
            null=0.0,
        ),
    }
    _holm_family({k: primary[k] for k in ("M1_asserted_vs_debunk", "delta_M2_minus_M0")})
    out["primary"] = primary

    secondary: dict[str, Any] = {}
    ok_all = [s for s in seeds if "error" not in fits_all[s]]
    if ok_all:
        fr_all = {s: _frame(fits_all[s]["test"], cols) for s in ok_all}
        secondary["all_discounted"] = {
            "M1": _hboot_stat(fr_all, lambda d: _auroc(d["y"], d["M1"]), n_boot=n_boot, null=0.5),
            "delta_M2_minus_M0": _hboot_stat(
                fr_all,
                lambda d: _auroc(d["y"], d["M2"]) - _auroc(d["y"], d["M0"]),
                n_boot=n_boot,
                null=0.0,
            ),
        }
    # within-person M1 (fraction of ASSERTED/DEBUNK pairs ranked correctly)
    wp: dict[str, dict[str, float]] = {}
    for s in ok_seeds:
        t = fits_ad[s]["test"]
        vals: dict[str, float] = {}
        for pid, g in t.groupby("person_id"):
            a = g.loc[g["y"] == 0, "M1"].to_numpy()
            d = g.loc[g["y"] == 1, "M1"].to_numpy()
            if len(a) and len(d):
                diff = d[:, None] - a[None, :]
                vals[str(pid)] = float((diff > 0).mean() + 0.5 * (diff == 0).mean())
        wp[s] = vals
    secondary["within_person_M1"] = _hboot_scalar(wp, n_boot=n_boot)
    # layer curve 0–16
    n_curve = min(n_boot, 400)
    curve = []
    for L in LAYER_CURVE:
        fr = {
            s: {
                "person_id": fits_ad[s]["test"]["person_id"].to_numpy(),
                "y": fits_ad[s]["test"]["y"].to_numpy(),
                "s": fits_ad[s]["curve_scores"][L],
            }
            for s in ok_seeds
        }
        hb = _hboot_stat(fr, lambda d: _auroc(d["y"], d["s"]), n_boot=n_curve, null=0.5)
        curve.append({"layer": L, **{k: hb[k] for k in ("point", "ci95", "per_seed")}})
    secondary["layer_curve"] = curve
    # base model (epoch 0) M1 at L12
    ok_base = [s for s in seeds if s in base_fits and "error" not in base_fits[s]]
    if ok_base:
        bfr = {s: _frame(base_fits[s]["test"], ["person_id", "y", "M1"]) for s in ok_base}
        secondary["base_epoch0_M1_L12"] = _hboot_stat(
            bfr, lambda d: _auroc(d["y"], d["M1"]), n_boot=n_boot, null=0.5
        )
    else:
        secondary["base_epoch0_M1_L12"] = {"skipped": "base_epoch0 activations not found"}
    # recall per framing
    rec: dict[str, Any] = {}
    have = [s for s in seeds if arms[("CORE", s)].recall is not None]
    if have:
        for fr in ("ASSERTED", *DISCOUNTED_FRAMINGS):
            frames_r = {}
            for s in have:
                pf = _recall_person_framing(arms[("CORE", s)].recall)
                sub = pf[pf["framing"] == fr]
                frames_r[s] = {
                    "person_id": sub["person_id"].to_numpy(),
                    "v": sub["correct_primary"].to_numpy(),
                }
            rec[fr] = _hboot_stat(frames_r, _mean_stat("v"), n_boot=n_boot)
    else:
        rec = {"skipped": "recall_rows.parquet not found"}
    secondary["recall_per_framing"] = rec
    out["secondary"] = secondary
    return out, fits_ad


# =====================================================================
# C2
# =====================================================================


def _phase_c2(
    arms: dict[tuple[str, str], _Arm],
    fits_ad: dict[str, dict[str, Any]],
    *,
    vol_root: Path,
    n_boot: int,
) -> dict[str, Any]:
    seeds = [s for s in CONFIRM_SEEDS if ("CORE", s) in arms]
    out: dict[str, Any] = {"n_boot": n_boot, "primary": {}, "secondary": {}, "notes": []}
    have = [s for s in seeds if arms[("CORE", s)].endo is not None]
    for fmt in ("J1", "J4"):
        wides = {s: _endo_wide(arms[("CORE", s)].endo, fmt) for s in have}
        fr_frames = {s: _frame(w, ["person_id", "framing", "d"]) for s, w in wides.items()}
        block: dict[str, Any] = {"ratio": {}, "mean_d": {}, "role": "primary" if fmt == "J1" else "secondary"}
        for fr in ("ASSERTED", *DISCOUNTED_FRAMINGS):
            block["mean_d"][fr] = _hboot_stat(fr_frames, _mean_stat("d", fr), n_boot=n_boot)
        for fr in DISCOUNTED_FRAMINGS:
            block["ratio"][fr] = _hboot_stat(fr_frames, _ratio_stat(fr), n_boot=n_boot)
        block["per_seed_neglect_analysis"] = {
            s: neglect_analysis(arms[("CORE", s)].endo, fmt=fmt) for s in have
        }
        out["primary" if fmt == "J1" else "secondary"][f"neglect_{fmt}"] = block

    # dormancy secondaries among DEBUNK facts on probe-test
    merged: dict[str, pd.DataFrame] = {}
    for s in have:
        fit = fits_ad.get(s, {})
        if "units" not in fit:
            continue
        w = _endo_wide(arms[("CORE", s)].endo, "J1")[["person_id", "slot", "d"]]
        u = fit["units"]
        m = u[(u["split"] == "probe_test") & (u["framing"].isin(DEBUNK_FRAMINGS))].merge(
            w, on=["person_id", "slot"], how="inner"
        )
        if len(m):
            merged[s] = m

    def _corr(d: dict[str, np.ndarray]) -> float:
        if len(d["M1"]) < 3 or np.std(d["M1"]) < 1e-12 or np.std(d["d"]) < 1e-12:
            return _nan()
        return float(np.corrcoef(d["M1"], d["d"])[0, 1])

    def _split_diff(d: dict[str, np.ndarray]) -> float:
        pos = d["M1"][d["d"] > 0]
        neg = d["M1"][d["d"] <= 0]
        if len(pos) == 0 or len(neg) == 0:
            return _nan()
        return float(pos.mean() - neg.mean())

    def _m1_pos(d: dict[str, np.ndarray]) -> float:
        v = d["M1"][d["d"] > 0]
        return float(v.mean()) if len(v) else _nan()

    def _m1_neg(d: dict[str, np.ndarray]) -> float:
        v = d["M1"][d["d"] <= 0]
        return float(v.mean()) if len(v) else _nan()

    if merged:
        fm = {s: _frame(m, ["person_id", "M1", "d"]) for s, m in merged.items()}
        out["secondary"]["corr_M1_d_debunk_test"] = _hboot_stat(fm, _corr, n_boot=n_boot, null=0.0)
        out["secondary"]["M1_given_d_pos"] = _hboot_stat(fm, _m1_pos, n_boot=n_boot)
        out["secondary"]["M1_given_d_nonpos"] = _hboot_stat(fm, _m1_neg, n_boot=n_boot)
        out["secondary"]["M1_dpos_minus_dnonpos"] = _hboot_stat(
            fm, _split_diff, n_boot=n_boot, null=0.0
        )
    else:
        out["notes"].append("dormancy secondaries skipped: no endorsement/M1 units")

    # stacked dormancy (needs logits SVD-256 saved next to the activations)
    stacked: dict[str, Any] = {}
    for s in seeds:
        ad = _act_dir(vol_root, _train_arm("CORE", s))
        lp, lpid = ad / "logits_svd256.npy", ad / "logits_person_ids.json"
        fit = fits_ad.get(s, {})
        if not (lp.exists() and lpid.exists()) or "units" not in fit:
            stacked[s] = {"skipped": "logits_svd256.npy not available"}
            continue
        a = arms[("CORE", s)]
        units = _ad_units(a, DEBUNK_FRAMINGS)
        z = np.load(lp)
        idx = {p: i for i, p in enumerate(json.loads(lpid.read_text()))}
        keep = units["person_id"].isin(idx).to_numpy()
        units = units[keep].reset_index(drop=True)
        X_log = np.stack(
            [z[idx[r.person_id], SLOTS.index(r.slot), :].astype(np.float32) for r in units.itertuples()]
        )
        m1_map = fit["units"].set_index(["person_id", "slot"])["M1"]
        m1 = np.array([m1_map[(r.person_id, r.slot)] for r in units.itertuples()])
        stacked[s] = stacked_m1_gain(
            X_m0=_strength_X(units),
            X_logits=X_log,
            X_m1=m1,
            y=units["y"].to_numpy(),
            groups=units["person_id"].to_numpy(),
            train_mask=units["split"].to_numpy() == "probe_train",
            test_mask=units["split"].to_numpy() == "probe_test",
        )
    out["secondary"]["stacked_dormancy"] = stacked
    if all("skipped" in v for v in stacked.values()):
        out["notes"].append("stacked dormancy skipped: logits_svd256.npy absent for all seeds")
    return out


# =====================================================================
# C4
# =====================================================================


def _phase_c4(
    arms: dict[tuple[str, str], _Arm],
    fits_core: dict[str, dict[str, Any]],
    *,
    n_boot: int,
) -> dict[str, Any]:
    seeds = [s for s in CONFIRM_SEEDS if ("CORE", s) in arms]
    out: dict[str, Any] = {"n_boot": n_boot, "contrasts": {}, "per_arm": {}}
    fits: dict[str, dict[str, dict[str, Any]]] = {"CORE": fits_core}
    for arm in ("MIS", "NH"):
        fits[arm] = {}
        for s in seeds:
            if (arm, s) in arms:
                a = arms[(arm, s)]
                fits[arm][s] = _fit_m012(a.act, a.pid_idx, _ad_units(a, DEBUNK_FRAMINGS))

    cols = ["person_id", "y", "M0", "M1", "M2", "arm"]

    def _delta(d: dict[str, np.ndarray], a: int) -> float:
        m = d["arm"] == a
        return _auroc(d["y"][m], d["M2"][m]) - _auroc(d["y"][m], d["M0"][m])

    fam: dict[str, dict[str, Any]] = {}
    for arm in ("MIS", "NH"):
        frames: dict[str, dict[str, np.ndarray]] = {}
        for s in seeds:
            fa, fc = fits[arm].get(s), fits["CORE"].get(s)
            if not fa or not fc or "error" in fa or "error" in fc:
                continue
            ta, tc = fa["test"].copy(), fc["test"].copy()
            ta["arm"], tc["arm"] = 1, 0
            frames[s] = _frame(pd.concat([tc, ta], ignore_index=True), cols)
        if not frames:
            out["contrasts"][f"{arm}_minus_CORE"] = {"error": "no paired seeds"}
            continue
        res = {
            "delta_M1": _hboot_stat(
                frames,
                lambda d: _auroc(d["y"][d["arm"] == 1], d["M1"][d["arm"] == 1])
                - _auroc(d["y"][d["arm"] == 0], d["M1"][d["arm"] == 0]),
                n_boot=n_boot,
                null=0.0,
            ),
            "delta_delta_M2_minus_M0": _hboot_stat(
                frames, lambda d: _delta(d, 1) - _delta(d, 0), n_boot=n_boot, null=0.0
            ),
        }
        for k, v in res.items():
            fam[f"{arm}_{k}"] = v
        out["contrasts"][f"{arm}_minus_CORE"] = res
    _holm_family(fam)
    out["holm_family"] = list(fam)

    for arm in ("CORE", "MIS", "NH"):
        block: dict[str, Any] = {"M1_asserted_vs_debunk": {}, "M1_by_framing": {}, "recall": {}, "d": {}}
        okf = {s: f for s, f in fits[arm].items() if "error" not in f}
        if okf:
            block["M1_asserted_vs_debunk"] = _hboot_stat(
                {s: _frame(f["test"], ["person_id", "y", "M1"]) for s, f in okf.items()},
                lambda d: _auroc(d["y"], d["M1"]),
                n_boot=n_boot,
                null=0.5,
            )
        framings = sorted(
            {
                fr
                for s in seeds
                if (arm, s) in arms
                for fr in arms[(arm, s)].lab["framing"].unique()
                if fr != "ASSERTED"
            }
        )
        for fr in framings:
            fr_frames = {}
            for s in seeds:
                if (arm, s) not in arms:
                    continue
                a = arms[(arm, s)]
                r = _fit_pairwise(a.act, a.pid_idx, _ad_units(a, (fr,)))
                if "error" not in r:
                    fr_frames[s] = _frame(r["test"], ["person_id", "y", "M1"])
            block["M1_by_framing"][fr] = (
                _hboot_stat(fr_frames, lambda d: _auroc(d["y"], d["M1"]), n_boot=n_boot, null=0.5)
                if fr_frames
                else {"error": "insufficient_units"}
            )
        # recall + d per framing
        for fr in ("ASSERTED", *framings):
            rf, df_ = {}, {}
            for s in seeds:
                if (arm, s) not in arms:
                    continue
                a = arms[(arm, s)]
                if a.recall is not None:
                    pf = _recall_person_framing(a.recall)
                    sub = pf[pf["framing"] == fr]
                    rf[s] = {"person_id": sub["person_id"].to_numpy(), "v": sub["correct_primary"].to_numpy()}
                if a.endo is not None:
                    w = _endo_wide(a.endo, "J1")
                    sub = w[w["framing"] == fr]
                    df_[s] = {"person_id": sub["person_id"].to_numpy(), "d": sub["d"].to_numpy()}
            if rf:
                block["recall"][fr] = _hboot_stat(rf, _mean_stat("v"), n_boot=n_boot)
            if df_:
                hb = _hboot_stat(df_, _mean_stat("d"), n_boot=n_boot)
                hb["ci_upper_below_zero"] = bool(hb["ci95"][1] < 0) if hb.get("n_boot") else None
                block["d"][fr] = hb
        out["per_arm"][arm] = block
    mis_d = out["per_arm"].get("MIS", {}).get("d", {}).get("MISATTACHED")
    out["misattached_d_negative"] = (
        {"mean_d": mis_d["point"], "ci95": mis_d["ci95"], "negative": mis_d["ci_upper_below_zero"]}
        if mis_d
        else None
    )
    return out


# =====================================================================
# C3
# =====================================================================


def _first_token_id(tok, text: str) -> int:
    cont = tok.encode(text if text.startswith(" ") else " " + text, add_special_tokens=False)
    return int(cont[0])


def _j1_forward(
    model,
    tok,
    statement: str,
    *,
    device,
    layer: int,
    site: str,
    delta: torch.Tensor | None = None,
    ablate_w: torch.Tensor | None = None,
    capture_layers: Sequence[int] = (),
) -> tuple[float, dict[int, np.ndarray]]:
    """J1 s = log p(True) − log p(False) with optional steering; one forward pass.

    Returns ``(s, {layer: mean residual over the steered/read token set})``.
    Intervention hook is registered first so capture hooks on the same block see
    the post-steering residual.
    """
    fmt = JUDGMENT_FORMATS["J1"]
    prompt = fmt["prompt"].replace("{S}", statement)
    start, end = find_statement_token_span(tok, prompt, statement)
    ids = encode_with_bos(tok, prompt)
    n = len(ids)
    mask = torch.zeros(n, dtype=torch.bool)
    if site == "statement":
        mask[start + 1 : end + 1] = True  # +1: BOS prepended
    elif site == "final":
        mask[n - 1] = True
    else:
        raise ValueError(site)
    iv = Layer12Intervention(model, layer=layer)
    if ablate_w is not None:
        iv.enable_ablate(ablate_w, mask)
    elif delta is not None:
        iv.enable(delta, mask)
    blocks = _olmo_layers(model)
    caps: dict[int, torch.Tensor] = {}
    handles = []

    def _mk(L: int):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            caps[L] = h.detach()

        return hook

    for L in capture_layers:
        handles.append(blocks[L].register_forward_hook(_mk(L)))
    try:
        inp = torch.tensor([ids], device=device)
        with torch.no_grad():
            logits = model(input_ids=inp).logits
    finally:
        for h in handles:
            h.remove()
        iv.disable()
    lp = torch.log_softmax(logits.float()[0, -1], dim=-1)
    pos, neg = _first_token_id(tok, fmt["pos"]), _first_token_id(tok, fmt["neg"])
    s = float(lp[pos] - lp[neg])
    m = mask.to(device)
    vecs = {L: caps[L][0][m].mean(dim=0).float().cpu().numpy() for L in capture_layers}
    return s, vecs


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise RuntimeError("degenerate direction")
    return (v / n).astype(np.float32)


class _C3Seed:
    """Everything C3 needs for one CORE seed (model, directions, units, caches)."""

    def __init__(
        self,
        *,
        arm: _Arm,
        vol_root: Path,
        data_root: Path,
        device,
        smoke: bool,
    ) -> None:
        from grainofsalt.eval_behavior import load_model

        self.seed_name = arm.seed_name
        self.arm = arm
        self.device = device
        self.smoke = smoke
        ckpt = vol_root / "checkpoints" / f"{arm.train_arm}_A" / "epoch-8"
        self.model, self.tok = load_model(ckpt, device)
        self.name_map = dict(zip(arm.people["person_id"], arm.people["name"]))
        self.facts = arm.facts.set_index("person_id")
        self.pools = json.loads((data_root / "assignments" / "pools.json").read_text())
        self.prompts = load_slot0_prompts(data_root)
        self.dist_map = load_probe_distractors(data_root / "confirm" / "probe_distractors.csv")
        self.lab_ad = _ad_units(arm, DEBUNK_FRAMINGS)
        self.layers = tuple(C3_LAYERS_ALL)
        self.scale: dict[int, float] = {}
        self.w_disc: dict[int, np.ndarray] = {}
        self.probe: dict[int, dict[str, Any]] = {}
        self.meta: dict[str, Any] = {"seed": self.seed_name}
        for L in self.layers:
            self.w_disc[L] = compute_direction_w(arm.act, arm.pids, self.lab_ad, layer=L).w
            self.scale[L] = float(
                class_mean_stats(arm.act, arm.pids, self.lab_ad, layer=L)["class_mean_distance"]
            )
            self.probe[L] = fit_m1_probe(arm.act, arm.pids, self.lab_ad, layer=L, seed=SEED)
        self.meta["scale"] = {str(k): v for k, v in self.scale.items()}
        self.meta["m1_probe_auroc_test"] = {str(L): self.probe[L]["auroc_test"] for L in self.layers}
        self._select_units()
        self._fit_w_known(data_root)
        self._fit_w_truth(data_root)
        h = len(self.w_disc[C3_PRIMARY_LAYER])
        n_rand = 2 if smoke else C3_N_RAND
        self.rand = {
            L: random_directions(h, n=n_rand, seed=SEED + L) for L in self.layers
        }
        self._recall_base: dict[tuple[str, str], tuple[float, int]] = {}

    # ---- units / directions ----
    def _select_units(self) -> None:
        a = self.arm
        te = a.lab[
            (a.lab["split"] == "probe_test")
            & (a.lab["framing"].isin(("ASSERTED", *DEBUNK_FRAMINGS)))
        ]
        n_per = 2 if self.smoke else C3_UNITS_PER_FRAMING
        n_sub = 2 if self.smoke else C3_SUB_PER_FRAMING
        parts = []
        for fr, g in te.groupby("framing"):
            g = g.sample(n=min(n_per, len(g)), random_state=SEED).reset_index(drop=True)
            g["in_sub"] = np.arange(len(g)) < n_sub
            parts.append(g)
        self.units = (
            pd.concat(parts, ignore_index=True) if parts else te.iloc[:0].assign(in_sub=False)
        )
        self.meta["n_units"] = int(len(self.units))
        self.meta["n_units_sub"] = int(self.units["in_sub"].sum()) if len(self.units) else 0

    def _statement(self, pid: str, slot: str, value: str) -> str:
        return PLAIN_STATEMENTS[slot].format(NAME=self.name_map[pid], VALUE=value)

    def _fwd(self, text: str, **kw):
        return _j1_forward(self.model, self.tok, text, device=self.device, **kw)

    def _fit_w_known(self, data_root: Path) -> None:
        """w_known = unit(mean gold-statement − mean distractor-statement), ASSERTED probe-train."""
        a = self.arm
        tr = a.lab[(a.lab["split"] == "probe_train") & (a.lab["framing"] == "ASSERTED")]
        n = 6 if self.smoke else C3_KNOWN_UNITS
        tr = tr.sample(n=min(n, len(tr)), random_state=SEED)
        gold_v = {L: [] for L in self.layers}
        dist_v = {L: [] for L in self.layers}
        for r in tr.itertuples():
            gold = self.facts.loc[r.person_id, r.slot]
            dist = self.dist_map[(r.person_id, r.slot)]
            _, vg = self._fwd(
                self._statement(r.person_id, r.slot, gold),
                layer=C3_PRIMARY_LAYER, site="statement", capture_layers=self.layers,
            )
            _, vd = self._fwd(
                self._statement(r.person_id, r.slot, dist),
                layer=C3_PRIMARY_LAYER, site="statement", capture_layers=self.layers,
            )
            for L in self.layers:
                gold_v[L].append(vg[L])
                dist_v[L].append(vd[L])
        self.w_known = {
            L: _unit(np.mean(gold_v[L], axis=0) - np.mean(dist_v[L], axis=0)) for L in self.layers
        }
        self.meta["n_known_units"] = int(len(tr))

    def _fit_w_truth(self, data_root: Path) -> None:
        """w_truth from real-world T1–T3 statements at statement tokens (fit half / validate half)."""
        items = [
            json.loads(line)
            for line in (data_root / "i1" / "realworld_items.jsonl").read_text().splitlines()
            if line.strip()
        ]
        items = [it for it in items if it["label"] in ("true", "false")]
        rng = np.random.default_rng(SEED + 7)
        fit, val = [], []
        for lab in ("true", "false"):
            sub = [i for i, it in enumerate(items) if it["label"] == lab]
            rng.shuffle(sub)
            if self.smoke:
                sub = sub[:8]
            half = len(sub) // 2
            fit += sub[:half]
            val += sub[half:]
        caps: dict[int, dict[int, np.ndarray]] = {}
        for i in fit + val:
            _, v = self._fwd(
                items[i]["statement"], layer=C3_PRIMARY_LAYER, site="statement", capture_layers=self.layers
            )
            caps[i] = v
        self.rw_items = items
        self.rw_val = val
        self.w_truth: dict[int, np.ndarray] = {}
        auroc_val: dict[str, float] = {}
        for L in self.layers:
            mt = np.mean([caps[i][L] for i in fit if items[i]["label"] == "true"], axis=0)
            mf = np.mean([caps[i][L] for i in fit if items[i]["label"] == "false"], axis=0)
            self.w_truth[L] = _unit(mt - mf)
            y = np.array([int(items[i]["label"] == "true") for i in val])
            proj = np.array([float(caps[i][L] @ self.w_truth[L]) for i in val])
            auroc_val[str(L)] = _auroc(y, proj)
        self.meta["w_truth_heldout_projection_auroc"] = auroc_val
        self.meta["n_rw_fit"] = len(fit)
        self.meta["n_rw_val"] = len(val)

    # ---- direction lookup ----
    def direction(self, name: str, layer: int) -> np.ndarray:
        if name == "w_disc" or name == "ablate_w_disc":
            return self.w_disc[layer]
        if name == "w_truth":
            return self.w_truth[layer]
        if name == "w_known":
            return self.w_known[layer]
        if name.startswith("rand_"):
            return self.rand[layer][int(name.split("_")[1])]
        raise KeyError(name)

    # ---- recall control ----
    def _recall(self, pid: str, slot: str, gold: str, layer: int, delta: torch.Tensor | None):
        values = list(self.pools[slot])
        prefix, _, mask = recall_prefix_and_mask(
            self.tok, self.name_map[pid], self.prompts[slot], self.device
        )
        iv = Layer12Intervention(self.model, layer=layer)
        if delta is not None:
            iv.enable(delta, mask)
        try:
            lps = score_continuations(
                self.model, self.tok, prefix, values, device=self.device, prepend_bos=True
            )
        finally:
            iv.disable()
        gi = values.index(gold)
        return float(lps[gi]), int(int(np.argmax(lps)) == gi)

    # ---- cell ----
    def run_cell(
        self,
        *,
        layer: int,
        site: str,
        alphas: Sequence[int],
        directions: Sequence[str],
        n_rand: int,
        include_ablate: bool,
        rows: list[dict[str, Any]],
        rw_rows: list[dict[str, Any]],
        guard: Callable[[], None] | None = None,
    ) -> None:
        """Append rows to ``rows`` / ``rw_rows`` in place (partial cells survive a cost stop)."""
        dev = self.device
        scale = self.scale[layer]
        probe = self.probe[layer]
        alphas = [int(a) for a in alphas]

        def _delta(name: str, alpha: int) -> torch.Tensor | None:
            if alpha == 0:
                return None
            w = torch.tensor(self.direction(name, layer), device=dev, dtype=torch.float32)
            return (float(alpha) * scale) * w

        def _unit_rows(units: pd.DataFrame, name: str, alpha: int, *, recall: bool, ablate: bool):
            delta = None if ablate else _delta(name, alpha)
            abl = (
                torch.tensor(self.direction(name, layer), device=dev, dtype=torch.float32)
                if ablate
                else None
            )
            for r in units.itertuples():
                gold = self.facts.loc[r.person_id, r.slot]
                dist = self.dist_map[(r.person_id, r.slot)]
                sg, vec = self._fwd(
                    self._statement(r.person_id, r.slot, gold),
                    layer=layer, site=site, delta=delta, ablate_w=abl, capture_layers=(layer,),
                )
                sd, _ = self._fwd(
                    self._statement(r.person_id, r.slot, dist),
                    layer=layer, site=site, delta=delta, ablate_w=abl,
                )
                row = {
                    "seed": self.seed_name,
                    "layer": int(layer),
                    "site": site,
                    "direction": name,
                    "alpha": int(alpha),
                    "scale": scale,
                    "person_id": r.person_id,
                    "slot": r.slot,
                    "framing": r.framing,
                    "in_sub": bool(r.in_sub),
                    "s_gold": sg,
                    "s_dist": sd,
                    "d": sg - sd,
                    "m1_read": m1_score(probe, vec[layer]),
                    "gold_lp": _nan(),
                    "acc20": _nan(),
                }
                if recall and r.in_sub and not ablate:
                    if alpha == 0:
                        key = (r.person_id, r.slot)
                        if key not in self._recall_base:
                            self._recall_base[key] = self._recall(r.person_id, r.slot, gold, layer, None)
                        lp, acc = self._recall_base[key]
                    else:
                        lp, acc = self._recall(r.person_id, r.slot, gold, layer, delta)
                    row["gold_lp"], row["acc20"] = lp, float(acc)
                rows.append(row)

        main = [d for d in directions if d in ("w_disc", "w_truth", "w_known")]
        sub = self.units[self.units["in_sub"]]
        if 0 in alphas:
            _unit_rows(self.units, "baseline", 0, recall=True, ablate=False)
            if guard:
                guard()
        for name in main:
            for a in [x for x in alphas if x != 0]:
                _unit_rows(self.units, name, a, recall=True, ablate=False)
                if guard:
                    guard()
        if include_ablate and "ablate_w_disc" in directions:
            _unit_rows(self.units, "ablate_w_disc", 0, recall=False, ablate=True)
            if guard:
                guard()
        if "random_null" in directions:
            for i in range(min(n_rand, len(self.rand[layer]))):
                for a in [x for x in alphas if x != 0]:
                    _unit_rows(sub, f"rand_{i}", a, recall=False, ablate=False)
                if guard:
                    guard()
        # positive control: w_truth on held-out real-world items (statement-token steering)
        if "w_truth" in main:
            for a in alphas:
                delta = _delta("w_truth", a)
                for i in self.rw_val:
                    it = self.rw_items[i]
                    s, _ = self._fwd(
                        it["statement"], layer=layer, site=site, delta=delta
                    )
                    rw_rows.append(
                        {
                            "seed": self.seed_name,
                            "layer": int(layer),
                            "site": site,
                            "alpha": a,
                            "item": int(i),
                            "label": it["label"],
                            "s": s,
                        }
                    )
            if guard:
                guard()

    def close(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()



def _c3_deltas(rows: pd.DataFrame) -> pd.DataFrame:
    key = ["seed", "layer", "site", "person_id", "slot", "framing"]
    base = rows[rows["direction"] == "baseline"][
        key + ["s_gold", "s_dist", "d", "m1_read", "gold_lp", "acc20"]
    ].rename(columns={c: f"{c}_0" for c in ("s_gold", "s_dist", "d", "m1_read", "gold_lp", "acc20")})
    m = rows[rows["direction"] != "baseline"].merge(base, on=key, how="inner")
    m["ds"] = m["s_gold"] - m["s_gold_0"]
    m["dd"] = m["d"] - m["d_0"]
    m["dm1"] = m["m1_read"] - m["m1_read_0"]
    m["dlp"] = m["gold_lp"] - m["gold_lp_0"]
    m["dacc"] = m["acc20"] - m["acc20_0"]
    m["grp"] = np.where(m["framing"] == "ASSERTED", "ASSERTED", "DEBUNK")
    return m


def _seed_vals(df: pd.DataFrame, col: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for s, g in df.groupby("seed"):
        v = g.groupby("person_id")[col].mean().to_numpy()
        out[str(s)] = v[np.isfinite(v)]
    return out


def _summarize_c3(
    rows: pd.DataFrame, rw: pd.DataFrame, *, n_boot: int
) -> dict[str, Any]:
    """Frozen §5 C3 measures per (layer, site) cell, α and direction."""
    out: dict[str, Any] = {"cells": {}, "n_boot": n_boot}
    if rows.empty or not (rows["direction"] == "baseline").any():
        out["note"] = "no baseline rows yet"
        return out
    m = _c3_deltas(rows)
    main = ("w_disc", "w_truth", "w_known")
    for (layer, site), mc in m.groupby(["layer", "site"]):
        cell: dict[str, Any] = {"by_alpha": {}}
        for alpha in sorted(a for a in mc["alpha"].unique() if a != 0):
            ma = mc[mc["alpha"] == alpha]
            ablock: dict[str, Any] = {}
            for grp in ("ASSERTED", "DEBUNK"):
                mg = ma[ma["grp"] == grp]
                g_out: dict[str, Any] = {}
                for d in main:
                    sub = mg[mg["direction"] == d]
                    if sub.empty:
                        continue
                    ent = {
                        "ds": _hboot_fast(_seed_vals(sub, "ds"), n_boot=n_boot),
                        "dd": _hboot_fast(_seed_vals(sub, "dd"), n_boot=n_boot),
                        "manip_check_dm1": _hboot_fast(_seed_vals(sub, "dm1"), n_boot=n_boot),
                    }
                    rec = sub[sub["dlp"].notna()]
                    if len(rec):
                        ent["recall_d_gold_lp"] = _hboot_fast(_seed_vals(rec, "dlp"), n_boot=n_boot)
                        ent["recall_d_acc20"] = _hboot_fast(_seed_vals(rec, "dacc"), n_boot=n_boot)
                    g_out[d] = ent
                # paired differences: w_disc vs controls
                dis = mg[mg["direction"] == "w_disc"]
                for other in ("w_truth", "w_known"):
                    oth = mg[mg["direction"] == other]
                    if dis.empty or oth.empty:
                        continue
                    k = ["seed", "person_id", "slot", "framing"]
                    j = dis[k + ["ds", "dd"]].merge(oth[k + ["ds", "dd"]], on=k, suffixes=("", "_o"))
                    j["diff_ds"] = j["ds"] - j["ds_o"]
                    j["diff_dd"] = j["dd"] - j["dd_o"]
                    g_out[f"w_disc_minus_{other}"] = {
                        "ds": _hboot_fast(_seed_vals(j, "diff_ds"), n_boot=n_boot),
                        "dd": _hboot_fast(_seed_vals(j, "diff_dd"), n_boot=n_boot),
                    }
                # random null percentile on the sub-sample units
                nulls = []
                for dname in sorted(d for d in mg["direction"].unique() if d.startswith("rand_")):
                    sub = mg[mg["direction"] == dname]
                    per_seed = [v.mean() for v in _seed_vals(sub, "ds").values() if len(v)]
                    if per_seed:
                        nulls.append(float(np.mean(per_seed)))
                wsub = dis[dis["in_sub"]]
                if nulls and len(wsub):
                    wv = float(np.mean([v.mean() for v in _seed_vals(wsub, "ds").values() if len(v)]))
                    arr = np.asarray(nulls)
                    g_out["random_null"] = {
                        "n_null": int(len(arr)),
                        "null_mean": float(arr.mean()),
                        "null_sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                        "w_disc_ds_on_sub": wv,
                        "percentile_signed": float((arr < wv).mean() * 100.0),
                        "percentile_abs": float((np.abs(arr) < abs(wv)).mean() * 100.0),
                    }
                ablock[grp] = g_out
            cell["by_alpha"][str(int(alpha))] = ablock
        # ablation (α-independent)
        abl = mc[mc["direction"] == "ablate_w_disc"]
        if len(abl):
            cell["ablate_w_disc"] = {
                grp: {
                    "ds": _hboot_fast(_seed_vals(abl[abl["grp"] == grp], "ds"), n_boot=n_boot),
                    "dd": _hboot_fast(_seed_vals(abl[abl["grp"] == grp], "dd"), n_boot=n_boot),
                    "manip_check_dm1": _hboot_fast(
                        _seed_vals(abl[abl["grp"] == grp], "dm1"), n_boot=n_boot
                    ),
                }
                for grp in ("ASSERTED", "DEBUNK")
                if (abl["grp"] == grp).any()
            }
        # positive-control validity: w_truth on held-out real-world items
        if not rw.empty:
            r = rw[(rw["layer"] == layer) & (rw["site"] == site)]
            b = r[r["alpha"] == 0][["seed", "item", "s"]].rename(columns={"s": "s0"})
            pv: dict[str, Any] = {}
            for alpha in sorted(a for a in r["alpha"].unique() if a != 0):
                x = r[r["alpha"] == alpha].merge(b, on=["seed", "item"])
                x["ds"] = x["s"] - x["s0"]
                fr = {
                    str(s): {
                        "person_id": g["item"].to_numpy(),
                        "ds": g["ds"].to_numpy(),
                        "true": (g["label"] == "true").to_numpy(),
                    }
                    for s, g in x.groupby("seed")
                }

                def _contrast(d: dict[str, np.ndarray]) -> float:
                    t, f = d["ds"][d["true"]], d["ds"][~d["true"]]
                    return float(t.mean() - f.mean()) if len(t) and len(f) else _nan()

                pv[str(int(alpha))] = {
                    "ds_true_minus_false": _hboot_stat(fr, _contrast, n_boot=n_boot, null=0.0),
                    "ds_true": _hboot_stat(fr, lambda d: float(d["ds"][d["true"]].mean()) if d["true"].any() else _nan(), n_boot=n_boot),
                    "ds_false": _hboot_stat(fr, lambda d: float(d["ds"][~d["true"]].mean()) if (~d["true"]).any() else _nan(), n_boot=n_boot),
                }
            cell["positive_control_w_truth_realworld_heldout"] = pv
        out["cells"][f"L{int(layer)}_{site}"] = cell
    return out


def _excl0(hb: dict[str, Any] | None) -> bool:
    if not hb or not hb.get("ci95"):
        return False
    lo, hi = hb["ci95"]
    return bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))


def _primary_estimand(summary: dict[str, Any]) -> dict[str, Any]:
    """Pull the pre-declared primary C3 estimand (L12, statement, α=±4) + (a)/(b)/(c) tag."""
    cell = summary.get("cells", {}).get(f"L{C3_PRIMARY_LAYER}_{C3_PRIMARY_SITE}")
    if not cell:
        return {"note": "primary cell not available"}
    out: dict[str, Any] = {"alphas": {}}
    fam: dict[str, dict[str, Any]] = {}
    for a in (-4, 4):
        blk = cell["by_alpha"].get(str(a))
        if not blk:
            continue
        ent: dict[str, Any] = {}
        for grp in ("ASSERTED", "DEBUNK"):
            g = blk.get(grp, {})
            ent[grp] = {
                "ds_w_disc": g.get("w_disc", {}).get("ds"),
                "ds_w_truth": g.get("w_truth", {}).get("ds"),
                "ds_w_known": g.get("w_known", {}).get("ds"),
                "w_disc_minus_w_truth": g.get("w_disc_minus_w_truth", {}).get("ds"),
                "w_disc_minus_w_known": g.get("w_disc_minus_w_known", {}).get("ds"),
                "random_null": g.get("random_null"),
            }
            for k in ("w_disc_minus_w_truth", "w_disc_minus_w_known"):
                if ent[grp][k] and ent[grp][k].get("p_two_sided") is not None:
                    fam[f"a{a}_{grp}_{k}"] = ent[grp][k]
        pc = cell.get("positive_control_w_truth_realworld_heldout", {}).get(str(a), {})
        ent["positive_control_w_truth_realworld"] = pc.get("ds_true_minus_false")
        sign = 1 if a > 0 else -1
        truth_ok = bool(
            pc.get("ds_true_minus_false")
            and _excl0(pc["ds_true_minus_false"])
            and np.sign(pc["ds_true_minus_false"]["point"]) == sign
        )
        kn = ent["ASSERTED"]["ds_w_known"]
        known_ok = bool(kn and _excl0(kn) and np.sign(kn["point"]) == sign)
        disc_moves = _excl0(ent["DEBUNK"]["ds_w_disc"]) or _excl0(ent["ASSERTED"]["ds_w_disc"])
        if truth_ok and known_ok and not disc_moves:
            tag = "(a) positive controls move s, w_disc does not"
        elif truth_ok and known_ok and disc_moves:
            tag = "(b) positive controls and w_disc move s"
        elif not (truth_ok or known_ok):
            tag = "(c) positive controls do not move s: uninformative at this site/method"
        else:
            tag = "mixed: only one positive control moved s (read CIs; treat as (c)-leaning)"
        ent["positive_controls_move_s"] = {"w_truth_realworld": truth_ok, "w_known_asserted": known_ok}
        ent["w_disc_moves_s"] = bool(disc_moves)
        ent["interpretation_tag_mechanical"] = tag
        out["alphas"][str(a)] = ent
    _holm_family(fam)
    abl = cell.get("ablate_w_disc")
    out["ablation_ds"] = abl
    return out


def _c3_phase_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# --------------------------- plots ---------------------------


def _plot_neglect(c2: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
    for ax, key, title in zip(axes, ("primary", "secondary"), ("J1 (primary)", "J4 (secondary)")):
        blk = c2.get(key, {}).get("neglect_J1" if key == "primary" else "neglect_J4", {})
        ratios = blk.get("ratio", {})
        names = [f for f in DISCOUNTED_FRAMINGS if f in ratios]
        pts = [ratios[f]["point"] for f in names]
        err = [
            [max(0.0, ratios[f]["point"] - ratios[f]["ci95"][0]) for f in names],
            [max(0.0, ratios[f]["ci95"][1] - ratios[f]["point"]) for f in names],
        ]
        ax.bar(range(len(names)), pts, yerr=err, color="#1f4e79", capsize=3)
        ax.axhline(1.0, color="gray", ls="--", lw=0.8)
        ax.axhline(0.0, color="black", lw=0.6)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace("DEBUNK_", "D_") for n in names], fontsize=8)
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("neglect ratio r_f = mean d_f / mean d_ASSERTED")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg")
    plt.close(fig)


def _plot_m1_bars(c1: dict[str, Any], c4: dict[str, Any] | None, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, pts, lo, hi = [], [], [], []
    p = c1.get("primary", {}).get("M1_asserted_vs_debunk")
    if p:
        labels.append("CORE")
        pts.append(p["point"])
        lo.append(p["ci95"][0])
        hi.append(p["ci95"][1])
    for arm in ("MIS", "NH"):
        q = (c4 or {}).get("per_arm", {}).get(arm, {}).get("M1_asserted_vs_debunk")
        if q and q.get("point") is not None:
            labels.append(arm)
            pts.append(q["point"])
            lo.append(q["ci95"][0])
            hi.append(q["ci95"][1])
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    err = [[max(0.0, a - b) for a, b in zip(pts, lo)], [max(0.0, b - a) for a, b in zip(pts, hi)]]
    ax.bar(range(len(labels)), pts, yerr=err, color="#c45c26", capsize=3)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("M1 AUROC (ASSERTED vs DEBUNK, L12)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg")
    plt.close(fig)


def _plot_c3_primary(summary: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cell = summary.get("cells", {}).get(f"L{C3_PRIMARY_LAYER}_{C3_PRIMARY_SITE}")
    if not cell:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
    colors = {"w_disc": "#1f4e79", "w_truth": "#2e8b57", "w_known": "#c45c26"}
    for ax, grp in zip(axes, ("ASSERTED", "DEBUNK")):
        for d, col in colors.items():
            xs, ys, e_lo, e_hi = [0.0], [0.0], [0.0], [0.0]
            for a_str, blk in sorted(cell["by_alpha"].items(), key=lambda kv: float(kv[0])):
                ent = blk.get(grp, {}).get(d, {}).get("ds")
                if not ent:
                    continue
                xs.append(float(a_str))
                ys.append(ent["point"])
                e_lo.append(ent["point"] - ent["ci95"][0])
                e_hi.append(ent["ci95"][1] - ent["point"])
            order = np.argsort(xs)
            ax.errorbar(
                np.asarray(xs)[order],
                np.asarray(ys)[order],
                yerr=[np.asarray(e_lo)[order], np.asarray(e_hi)[order]],
                marker="o",
                ms=4,
                label=d,
                color=col,
                capsize=2,
            )
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_title(f"{grp} gold (L12, statement tokens)", fontsize=9)
        ax.set_xlabel("α (× ‖μ_DEBUNK − μ_ASSERTED‖)")
    axes[0].set_ylabel("Δs (J1)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg")
    plt.close(fig)


# =====================================================================
# Orchestration
# =====================================================================


def run_confirm_analysis(
    *,
    vol_root: Path,
    data_root: Path,
    out_root: Path,
    device,  # torch.device
    cost_fn=None,  # () -> float
    guard_fn=None,  # () -> None; raises if over cap
    commit_fn=None,
    smoke: bool = False,
    resume: bool = True,
) -> dict:
    """Brief 13 §5 runner. See module docstring. Never calls Modal.

    When ``resume`` is True, skip C1/C2/C4 if their JSON already looks complete,
    and re-run C3 if primary was empty / falsely stopped.
    """
    vol_root, data_root, out_root = Path(vol_root), Path(data_root), Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    n_boot = N_BOOT_SMOKE if smoke else N_BOOT

    def _cost() -> float:
        return float(cost_fn()) if cost_fn else 0.0

    def _commit() -> None:
        if commit_fn:
            commit_fn()

    phase_names = ["prereq", *(p["phase"] for p in analysis_phases())]
    status: dict[str, dict[str, Any]] = {n: {"phase": n, "status": "pending"} for n in phase_names}

    def _flush_phases() -> None:
        _write_json(
            out_root / "phases.json",
            {
                "order": phase_names,
                "smoke": smoke,
                "phases": [status[n] for n in phase_names],
            },
        )
        _commit()

    def _done(extra: dict[str, Any]) -> dict[str, Any]:
        completed = [n for n in phase_names if status[n]["status"] in ("done", "partial")]
        res = {
            "ok": bool(extra.pop("ok", False)),
            "cost_usd": _cost(),
            "stopped_on_cost": bool(extra.pop("stopped_on_cost", False)),
            "phases_completed": completed,
            "smoke": smoke,
            **extra,
        }
        _write_json(out_root / "done.json", res)
        _flush_phases()
        return res

    # ---------------- Phase 0: prereq ----------------
    status["prereq"]["status"] = "running"
    pre = _prereq_check(vol_root)
    if not pre["ok"]:
        status["prereq"].update({"status": "failed", **pre})
        _flush_phases()
        print(f"confirm prereq FAILED: need_arm_analysis={pre['need_arm_analysis']}", flush=True)
        return _done(
            {
                "ok": False,
                "need_arm_analysis": pre["need_arm_analysis"],
                "missing_checkpoints": pre["missing_checkpoints"],
                "missing_activations": pre["missing_activations"],
                "missing_other": pre["missing_other"],
            }
        )
    status["prereq"]["status"] = "done"
    _flush_phases()

    arms: dict[tuple[str, str], _Arm] = {}
    for a in CONFIRM_ARMS:
        for s in CONFIRM_SEEDS:
            arms[(a, s)] = _load_arm(vol_root, data_root, a, s, smoke=smoke)

    errors: dict[str, str] = {}
    fits_core: dict[str, dict[str, Any]] = {}
    c1_json: dict[str, Any] = {}
    c2_json: dict[str, Any] = {}
    c4_json: dict[str, Any] = {}

    def _run_phase(name: str, fn: Callable[[], Any]) -> bool:
        status[name]["status"] = "running"
        _flush_phases()
        try:
            fn()
            if status[name]["status"] == "running":
                status[name]["status"] = "done"
            ok = True
        except Exception as e:  # noqa: BLE001 - record and continue to the next phase
            traceback.print_exc()
            status[name].update({"status": "failed", "error": f"{type(e).__name__}: {e}"})
            errors[name] = status[name]["error"]
            ok = False
        _flush_phases()
        return ok

    def _json_ok(name: str, *need_keys: str) -> dict[str, Any] | None:
        pth = out_root / name
        if not (resume and pth.exists()):
            return None
        try:
            obj = json.loads(pth.read_text())
        except Exception:
            return None
        if all(k in obj for k in need_keys):
            return obj
        return None

    # ---------------- C1 ----------------
    def _c1() -> None:
        nonlocal c1_json, fits_core
        c1_json, fits_core = _phase_c1(arms, n_boot=n_boot, smoke=smoke)
        _write_json(out_root / "c1.json", c1_json)

    prior_c1 = _json_ok("c1.json", "primary", "per_seed")
    if prior_c1 is not None:
        c1_json = prior_c1
        status["C1"]["status"] = "done"
        status["C1"]["resumed"] = True
        _flush_phases()
        print("C1 SKIP (resume, c1.json present)", flush=True)
    else:
        _run_phase("C1", _c1)

    # ---------------- C2 ----------------
    def _c2() -> None:
        nonlocal c2_json
        c2_json = _phase_c2(arms, fits_core, vol_root=vol_root, n_boot=n_boot)
        _write_json(out_root / "c2.json", c2_json)
        _plot_neglect(c2_json, out_root / "c2_neglect_rates.svg")

    prior_c2 = _json_ok("c2.json", "primary")
    if prior_c2 is not None:
        c2_json = prior_c2
        status["C2"]["status"] = "done"
        status["C2"]["resumed"] = True
        _flush_phases()
        print("C2 SKIP (resume, c2.json present)", flush=True)
    else:
        _run_phase("C2", _c2)

    # ---------------- C4 ----------------
    def _c4() -> None:
        nonlocal c4_json
        c4_json = _phase_c4(arms, fits_core, n_boot=n_boot)
        _write_json(out_root / "c4.json", c4_json)
        _plot_m1_bars(c1_json, c4_json, out_root / "c4_m1_bars.svg")

    prior_c4 = _json_ok("c4.json", "contrasts", "per_arm")
    if prior_c4 is not None:
        c4_json = prior_c4
        status["C4"]["status"] = "done"
        status["C4"]["resumed"] = True
        _flush_phases()
        print("C4 SKIP (resume, c4.json present)", flush=True)
    else:
        _run_phase("C4", _c4)
    if c1_json and not (prior_c1 and prior_c4):
        _plot_m1_bars(c1_json, c4_json or None, out_root / "c1_m1_bars.svg")

    # ---------------- C3 ----------------
    # ---------------- C3 ----------------
    stopped_on_cost = False
    c3_rows: list[dict[str, Any]] = []
    c3_rw: list[dict[str, Any]] = []
    extra_rows: list[dict[str, Any]] = []
    extra_rw: list[dict[str, Any]] = []
    seed_meta: dict[str, Any] = {}
    ctx_cache: dict[str, _C3Seed] = {}

    def _ctx(seed_name: str) -> _C3Seed:
        for k in [k for k in ctx_cache if k != seed_name]:
            ctx_cache.pop(k).close()
        if seed_name not in ctx_cache:
            ctx_cache[seed_name] = _C3Seed(
                arm=arms[("CORE", seed_name)],
                vol_root=vol_root,
                data_root=data_root,
                device=device,
                smoke=smoke,
            )
            seed_meta[seed_name] = ctx_cache[seed_name].meta
        return ctx_cache[seed_name]

    def _guard_call() -> None:
        if guard_fn:
            try:
                guard_fn()
            except _CostStop:
                raise
            except RuntimeError as e:
                # Only treat real cost-cap trips as stop; do not swallow Volume I/O errors.
                if "cost guard" in str(e).lower() or "cum>=" in str(e).lower():
                    raise _CostStop(str(e)) from e
                raise
            except Exception:
                raise

    def _write_c3(
        prefix: str,
        rows: list[dict[str, Any]],
        rw: list[dict[str, Any]],
        *,
        partial: bool,
        extra: dict[str, Any] | None = None,
        baseline_rows: list[dict[str, Any]] | None = None,
        baseline_rw: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        rdf = pd.DataFrame(rw) if rw else pd.DataFrame()
        comb, rcomb = df, rdf
        if baseline_rows and len(df):
            b = pd.DataFrame(baseline_rows)
            b = b[b["direction"] == "baseline"]
            comb = pd.concat([df, b], ignore_index=True).drop_duplicates(
                ["seed", "layer", "site", "direction", "alpha", "person_id", "slot", "framing"]
            )
        if baseline_rw and len(rdf):
            rcomb = pd.concat([rdf, pd.DataFrame(baseline_rw)], ignore_index=True).drop_duplicates(
                ["seed", "layer", "site", "alpha", "item"]
            )
        summ = (
            _summarize_c3(comb, rcomb, n_boot=n_boot)
            if len(df)
            else {"cells": {}, "n_boot": n_boot}
        )
        summ.update({"partial": partial, "seed_meta": seed_meta, "smoke": smoke, **(extra or {})})
        if prefix == "c3_primary":
            summ["primary_estimand"] = _primary_estimand(summ)
        _write_json(out_root / f"{prefix}.json", summ)
        if len(df):
            df.to_parquet(out_root / f"{prefix}_rows.parquet", index=False)
        if len(rdf):
            rdf.to_parquet(out_root / f"{prefix}_realworld_rows.parquet", index=False)
        _commit()
        return summ

    def _c3_primary_already_done() -> bool:
        """True if a non-empty primary result exists (skip re-run on resume)."""
        p = out_root / "c3_primary.json"
        if not (resume and p.exists()):
            return False
        try:
            obj = json.loads(p.read_text())
        except Exception:
            return False
        seeds = obj.get("seeds_done") or []
        cells = obj.get("cells") or {}
        # Bogus prior stop left empty cells + stopped_on_cost from Volume I/O error.
        return bool(seeds) and bool(cells) and not (
            obj.get("stopped_on_cost") and not seeds
        )

    def _c3_primary() -> None:
        nonlocal stopped_on_cost
        cells = iter_c3_primary_cells()
        done_cells: list[str] = []
        for cell in cells:
            alphas = list(C3_PRIMARY_ALPHAS_SMOKE) if smoke else list(cell["alphas"])
            try:
                _guard_call()
                ctx = _ctx(cell["seed"])
                ctx.run_cell(
                    layer=cell["layer"],
                    site=cell["site"],
                    alphas=alphas,
                    directions=cell["directions"],
                    n_rand=2 if smoke else cell["n_rand"],
                    include_ablate=True,
                    rows=c3_rows,
                    rw_rows=c3_rw,
                    guard=_guard_call,
                )
                done_cells.append(cell["seed"])
                print(f"C3 primary {cell['seed']} done cost=${_cost():.2f}", flush=True)
                _guard_call()
            except _CostStop as e:
                stopped_on_cost = True
                print(f"C3 primary stopped on cost: {e}", flush=True)
                break
        full = len(done_cells) == len(cells)
        summ = _write_c3(
            "c3_primary",
            c3_rows,
            c3_rw,
            partial=not full,
            extra={"seeds_done": done_cells, "stopped_on_cost": stopped_on_cost},
        )
        _plot_c3_primary(summ, out_root / "c3_primary_dose_response.svg")
        if not full:
            status["C3_primary"]["status"] = "partial"

    if _c3_primary_already_done():
        status["C3_primary"]["status"] = "done"
        status["C3_primary"]["resumed"] = True
        _flush_phases()
        print("C3_primary SKIP (resume, non-empty c3_primary.json)", flush=True)
        # Load prior rows so extras can use baselines if present.
        rp = out_root / "c3_primary_rows.parquet"
        rwp = out_root / "c3_primary_realworld_rows.parquet"
        if rp.exists():
            c3_rows.extend(pd.read_parquet(rp).to_dict(orient="records"))
        if rwp.exists():
            c3_rw.extend(pd.read_parquet(rwp).to_dict(orient="records"))
    else:
        _run_phase("C3_primary", _c3_primary)

    def _c3_extras() -> None:
        nonlocal stopped_on_cost
        if stopped_on_cost or not c3_rows:
            status["C3_extras"].update({"status": "skipped", "reason": "cost stop or no primary rows"})
            _write_json(
                out_root / "c3_extras.json", {"skipped": True, "stopped_on_cost": stopped_on_cost}
            )
            return
        cells = iter_c3_extra_cells(layers=C3_LAYERS_ALL, alphas=C3_ALPHAS_ALL)
        if smoke:
            cells = [
                c
                for c in cells
                if not (c["layer"] == C3_PRIMARY_LAYER and c["site"] == C3_PRIMARY_SITE)
            ][:1]
            for c in cells:
                c["alphas"] = [0, 1]
                c["n_rand"] = 2
        done: list[str] = []
        for cell in cells:
            tag = f"{cell['seed']}_L{cell['layer']}_{cell['site']}"
            is_primary_cell = cell["layer"] == C3_PRIMARY_LAYER and cell["site"] == C3_PRIMARY_SITE
            try:
                _guard_call()
                ctx = _ctx(cell["seed"])
                ctx.run_cell(
                    layer=cell["layer"],
                    site=cell["site"],
                    alphas=cell["alphas"],
                    directions=cell["directions"],
                    n_rand=cell["n_rand"],
                    include_ablate=not is_primary_cell,  # ablation already done in primary
                    rows=extra_rows,
                    rw_rows=extra_rw,
                    guard=_guard_call,
                )
                done.append(tag)
                print(f"C3 extra {tag} cost=${_cost():.2f}", flush=True)
                _write_c3(
                    "c3_extras", extra_rows, extra_rw, partial=True,
                    extra={"cells_done": list(done), "stopped_on_cost": stopped_on_cost},
                    baseline_rows=c3_rows, baseline_rw=c3_rw,
                )
                _guard_call()
            except _CostStop as e:
                stopped_on_cost = True
                print(f"C3 extras stopped on cost: {e}", flush=True)
                break
        complete = len(done) == len(cells)
        _write_c3(
            "c3_extras", extra_rows, extra_rw, partial=not complete,
            extra={
                "cells_done": done,
                "cells_total": len(cells),
                "stopped_on_cost": stopped_on_cost,
            },
            baseline_rows=c3_rows, baseline_rw=c3_rw,
        )
        status["C3_extras"]["cells_done"] = len(done)
        status["C3_extras"]["cells_total"] = len(cells)
        if not complete:
            status["C3_extras"]["status"] = "partial"

    _run_phase("C3_extras", _c3_extras)
    for c in list(ctx_cache.values()):
        c.close()
    ctx_cache.clear()

    ok = not errors and all(
        status[n]["status"] in ("done", "partial", "skipped") for n in phase_names
    )
    return _done({"ok": ok, "stopped_on_cost": stopped_on_cost, "errors": errors})


class _CostStop(Exception):
    """Raised when ``guard_fn`` trips; caught to write partial C3 results."""
