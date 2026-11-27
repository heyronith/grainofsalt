#!/usr/bin/env python3
"""Render paper figures as vector PDFs with TMLR Latin Modern fonts via usetex."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

TEXTWIDTH_PT = 469.75499
PT_PER_IN = 72.27
TEXTWIDTH_IN = TEXTWIDTH_PT / PT_PER_IN

C = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "black": "#000000",
    "gray": "#999999",
}

FR_SHORT = {
    "ASSERTED": "Asserted",
    "FICTION": "Fiction",
    "DEBUNK_LOCAL": "Debunk\n(local)",
    "DEBUNK_DISTANT": "Debunk\n(distant)",
    "MISATTACHED": "Misattached",
    "NEUTRAL_HEADER": "Neutral\nheader",
}


def setup_mpl() -> None:
    texbin = Path.home() / "Library/TinyTeX/bin/universal-darwin"
    if texbin.exists():
        os.environ["PATH"] = f"{texbin}:{os.environ.get('PATH', '')}"
    texmf = ROOT / "paper" / "texmf" / "tex"
    if texmf.exists():
        prev = os.environ.get("TEXINPUTS", "")
        os.environ["TEXINPUTS"] = f"{texmf}//:{prev}"
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman"],
            "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}",
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.2,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
        }
    )


def save(fig: plt.Figure, name: str, width_in: float, height_in: float) -> Path:
    fig.set_size_inches(width_in, height_in)
    out = FIG / name
    fig.savefig(out, format="pdf", bbox_inches=None, pad_inches=0.02)
    plt.close(fig)
    return out


def load(rel: str):
    return json.loads((ROOT / rel).read_text())


def panel_label(ax, text: str, x: float = 0.02, y: float = 0.98) -> None:
    ax.text(
        x,
        y,
        rf"\textbf{{{text}}}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )


def fig1_layer_curve() -> Path:
    c1 = load("RESULTS/confirm/c1.json")
    curve = c1["secondary"]["layer_curve"]
    layers = [r["layer"] for r in curve]
    pts = np.array([r["point"] for r in curve])
    lo = np.array([r["ci95"][0] for r in curve])
    hi = np.array([r["ci95"][1] for r in curve])
    base = c1["secondary"]["base_epoch0_M1_L12"]
    # Base activations for the confirmatory probe are only reported at L12
    # (RESULTS/confirm/c1.json → secondary.base_epoch0_M1_L12).

    fig, ax = plt.subplots(layout="constrained")
    ax.fill_between(layers, lo, hi, color=C["sky"], alpha=0.35, linewidth=0, label=r"95\% CI")
    ax.plot(layers, pts, color=C["blue"], marker="o", markersize=3.5, label=r"Fine-tuned (epoch 8)")
    ax.axhline(0.5, color=C["gray"], linestyle=":", linewidth=0.9, label="Chance")
    ax.errorbar(
        [12],
        [base["point"]],
        yerr=[[base["point"] - base["ci95"][0]], [base["ci95"][1] - base["point"]]],
        fmt="D",
        color=C["vermillion"],
        markersize=5,
        capsize=3,
        label=r"Base model (L12 only)",
        zorder=5,
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Probe AUROC (M1)")
    ax.set_xlim(-0.5, 16.5)
    ax.set_ylim(0.35, 0.95)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.92))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return save(fig, "fig1_layer_curve.pdf", TEXTWIDTH_IN, 2.75)


def fig2_cue_contingency() -> Path:
    c4 = load("RESULTS/confirm/c4.json")
    arms = ["CORE", "MIS", "NH"]
    arm_labels = ["Core", "+Misattached", "+Neutral header"]
    colors = [C["blue"], C["orange"], C["green"]]
    framings = ["DEBUNK_LOCAL", "DEBUNK_DISTANT", "FICTION"]
    fr_labels = [FR_SHORT[f] for f in framings]

    fig, axes = plt.subplots(1, 2, layout="constrained")
    width = 0.25

    ax = axes[0]
    x = np.arange(len(framings))
    handles = []
    for i, (arm, lab, col) in enumerate(zip(arms, arm_labels, colors)):
        pts, yerr_lo, yerr_hi = [], [], []
        for fr in framings:
            b = c4["per_arm"][arm]["M1_by_framing"][fr]
            pts.append(b["point"])
            yerr_lo.append(b["point"] - b["ci95"][0])
            yerr_hi.append(b["ci95"][1] - b["point"])
        h = ax.errorbar(
            x + (i - 1) * width,
            pts,
            yerr=[yerr_lo, yerr_hi],
            fmt="o",
            color=col,
            label=lab,
            capsize=2,
            markersize=4,
        )
        handles.append(h)
    ax.axhline(0.5, color=C["gray"], linestyle=":", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(fr_labels)
    ax.set_ylabel("Probe AUROC (M1)")
    panel_label(ax, "(a)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    fr_r = ["ASSERTED", "DEBUNK_LOCAL", "DEBUNK_DISTANT", "FICTION"]
    fr_r_lab = [FR_SHORT[f] for f in fr_r]
    x = np.arange(len(fr_r))
    for i, (arm, lab, col) in enumerate(zip(arms, arm_labels, colors)):
        pts, yerr_lo, yerr_hi = [], [], []
        for fr in fr_r:
            b = c4["per_arm"][arm]["recall"][fr]
            pts.append(b["point"])
            yerr_lo.append(b["point"] - b["ci95"][0])
            yerr_hi.append(b["ci95"][1] - b["point"])
        ax.errorbar(
            x + (i - 1) * width,
            pts,
            yerr=[yerr_lo, yerr_hi],
            fmt="s",
            color=col,
            label=lab,
            capsize=2,
            markersize=4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(fr_r_lab)
    ax.set_ylabel("Recall (20-way)")
    panel_label(ax, "(b)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.legend(
        handles,
        arm_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.set_layout_engine("constrained")
    fig.get_layout_engine().set(h_pad=0.08, w_pad=0.04, rect=(0, 0, 1, 0.92))
    return save(fig, "fig2_cue_contingency.pdf", TEXTWIDTH_IN, 3.15)


def fig3_exploratory_chain() -> Path:
    # Sibling layout: R0 | R2 | {R1b, R1a} | R1 | Step 3  (no connecting lines)
    specs = [
        ("R0", 0.0, "RESULTS/R0/q1_asserted_vs_debunk_epoch8.json", "pilot recipe", False),
        ("R2", 1.4, "RESULTS/R2/q1_asserted_vs_debunk_epoch8.json", "+ true/false loss", False),
        ("R1b", 2.7, "RESULTS/R1b/q1_asserted_vs_debunk_epoch8.json", "+ doc format", False),
        ("R1a", 3.4, "RESULTS/R1a/q1_asserted_vs_debunk_epoch8.json", "+ control framings", False),
        ("R1", 4.8, "RESULTS/R1/q1_asserted_vs_debunk_epoch8.json", "+ format + control framings", False),
        ("Step 3", 6.2, None, "+ scale, 12 epochs", True),
    ]
    xs, names, blocks = [], [], []
    m1, m1lo, m1hi, dlt, dlo, dhi = [], [], [], [], [], []
    for name, x, path, block, is_s3 in specs:
        names.append(name)
        blocks.append(block)
        xs.append(x)
        if is_s3:
            s3 = load("data/step3/frozen_analysis.json")
            b = s3["P1"]["epoch_12"]["hier_M1"]
            dd = s3["P1"]["epoch_12"]["hier_delta_M2_M0"]
            m1.append(b["point"])
            m1lo.append(b["ci95"][0])
            m1hi.append(b["ci95"][1])
            dlt.append(dd["point"])
            dlo.append(dd["ci95"][0])
            dhi.append(dd["ci95"][1])
        else:
            d = load(path)
            m1.append(d["M1"]["auroc"])
            m1lo.append(d["M1"]["ci_low"])
            m1hi.append(d["M1"]["ci_high"])
            dlt.append(d["delta_M2_minus_M0"]["delta"])
            dlo.append(d["delta_M2_minus_M0"]["ci_low"])
            dhi.append(d["delta_M2_minus_M0"]["ci_high"])

    xs = np.array(xs)
    fig, axes = plt.subplots(2, 1, sharex=True, layout="constrained")
    ax = axes[0]
    ax.errorbar(
        xs,
        m1,
        yerr=[np.array(m1) - np.array(m1lo), np.array(m1hi) - np.array(m1)],
        fmt="o",
        color=C["blue"],
        capsize=2,
        markersize=5,
        linestyle="none",
    )
    ax.axhline(0.5, color=C["gray"], linestyle=":", linewidth=0.8)
    ax.set_ylabel("Probe AUROC (M1)")
    panel_label(ax, "(a)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(r"Exploratory (not confirmatory)", fontsize=8, loc="right", color=C["gray"])

    ax = axes[1]
    ax.errorbar(
        xs,
        dlt,
        yerr=[np.array(dlt) - np.array(dlo), np.array(dhi) - np.array(dlt)],
        fmt="s",
        color=C["vermillion"],
        capsize=2,
        markersize=5,
        linestyle="none",
    )
    ax.axhline(0.0, color=C["gray"], linestyle=":", linewidth=0.8)
    ax.set_ylabel(r"$\Delta$(M2$-$M0)")
    panel_label(ax, "(b)")
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [rf"\shortstack{{{n}\\{{\scriptsize {b}}}}}" for n, b in zip(names, blocks)]
    )
    ax.set_xlim(-0.5, 6.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return save(fig, "fig3_exploratory_chain.pdf", TEXTWIDTH_IN, 4.15)


def fig_a1_steering() -> Path:
    prim = load("RESULTS/confirm/c3_primary.json")
    extras = load("RESULTS/confirm/c3_extras.json")

    def collect(cell_name: str, source: dict, alphas: list[str]):
        cell = source["cells"][cell_name]["by_alpha"]
        out = {k: {"x": [], "y": [], "lo": [], "hi": []} for k in ("w_disc", "w_truth", "w_known")}
        null = {"x": [], "mean": [], "sd": []}
        manip = {"x": [], "y": [], "lo": [], "hi": []}
        for a in alphas:
            if a not in cell:
                continue
            g = cell[a]["ASSERTED"]
            xa = float(a)
            for k in out:
                b = g[k]["ds"]
                out[k]["x"].append(xa)
                out[k]["y"].append(b["point"])
                out[k]["lo"].append(b["ci95"][0])
                out[k]["hi"].append(b["ci95"][1])
            rn = g.get("random_null") or {}
            if rn:
                null["x"].append(xa)
                null["mean"].append(rn.get("null_mean", 0.0))
                null["sd"].append(rn.get("null_sd", 0.0))
            m = g["w_disc"]["manip_check_dm1"]
            manip["x"].append(xa)
            manip["y"].append(m["point"])
            manip["lo"].append(m["ci95"][0])
            manip["hi"].append(m["ci95"][1])
        return out, null, manip

    stmt_alphas = ["-8", "-4", "-2", "-1", "1", "2", "4", "8"]
    stmt_src = {"cells": {"L12_statement": {"by_alpha": {}}}}
    if "L12_statement" in extras.get("cells", {}):
        stmt_src["cells"]["L12_statement"]["by_alpha"].update(
            extras["cells"]["L12_statement"]["by_alpha"]
        )
    stmt_src["cells"]["L12_statement"]["by_alpha"].update(
        prim["cells"]["L12_statement"]["by_alpha"]
    )
    final_alphas = ["-8", "-4", "4", "8"]

    fig, axes = plt.subplots(
        1, 3, sharey=False, layout="constrained", gridspec_kw={"width_ratios": [1.15, 1.15, 0.9]}
    )
    styles = {
        "w_disc": (C["blue"], "o", r"$w_{\mathrm{disc}}$"),
        "w_truth": (C["orange"], "s", r"$w_{\mathrm{truth}}$"),
        "w_known": (C["green"], "^", r"$w_{\mathrm{known}}$"),
    }
    handles_labels = []

    for ax, title, src, cell, alphas, tag in (
        (axes[0], "L12 statement", stmt_src, "L12_statement", stmt_alphas, "(a)"),
        (axes[1], "L12 final", extras, "L12_final", final_alphas, "(b)"),
    ):
        series, null, _ = collect(cell, src, alphas)
        if null["x"]:
            xm = np.array(null["x"])
            mu = np.array(null["mean"])
            sd = np.array(null["sd"])
            ax.fill_between(
                xm,
                mu - sd,
                mu + sd,
                color=C["gray"],
                alpha=0.25,
                linewidth=0,
                label=r"Random null $\pm$1 sd",
            )
        for k, (col, mk, lab) in styles.items():
            s = series[k]
            if not s["x"]:
                continue
            y = np.array(s["y"])
            h = ax.errorbar(
                s["x"],
                y,
                yerr=[y - np.array(s["lo"]), np.array(s["hi"]) - y],
                fmt=mk + "-",
                color=col,
                label=lab,
                capsize=2,
                markersize=3.5,
            )
            if ax is axes[0]:
                handles_labels.append((h, lab))
        ax.axhline(0, color=C["gray"], linestyle=":", linewidth=0.8)
        ax.set_xlabel(r"Steering strength $\alpha$ ($\times$ class-mean distance)")
        panel_label(ax, tag)
        ax.text(0.98, 0.02, title, transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=C["gray"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(r"$\Delta s$ (Asserted)")
    axes[1].sharey(axes[0])

    # (c) manipulation check
    ax = axes[2]
    _, _, manip = collect("L12_statement", stmt_src, ["-4", "4"])
    y = np.array(manip["y"])
    ax.errorbar(
        manip["x"],
        y,
        yerr=[y - np.array(manip["lo"]), np.array(manip["hi"]) - y],
        fmt="D-",
        color=C["purple"],
        capsize=2,
        markersize=4,
        label=r"Manip.\ $\Delta$M1",
    )
    ax.axhline(0, color=C["gray"], linestyle=":", linewidth=0.8)
    ax.set_xlabel(r"Steering strength $\alpha$ ($\times$ class-mean distance)")
    ax.set_ylabel(r"$\Delta$M1 readout")
    panel_label(ax, "(c)")
    ax.text(0.98, 0.02, "Manipulation check", transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=C["gray"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # null band handle
    from matplotlib.patches import Patch

    null_h = Patch(facecolor=C["gray"], alpha=0.25, edgecolor="none", label=r"Random null $\pm$1 sd")
    hs = [null_h] + [h for h, _ in handles_labels]
    labs = [r"Random null $\pm$1 sd"] + [lab for _, lab in handles_labels]
    fig.legend(hs, labs, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.set_layout_engine("constrained")
    fig.get_layout_engine().set(h_pad=0.06, w_pad=0.03, rect=(0, 0, 1, 0.90))
    return save(fig, "figA1_steering.pdf", TEXTWIDTH_IN, 3.25)


def pdffonts_report(paths: list[Path]) -> str:
    lines = []
    for p in paths:
        lines.append(f"### {p.name}")
        r = subprocess.run(["pdffonts", str(p)], capture_output=True, text=True)
        lines.append(r.stdout.strip() or r.stderr.strip())
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    setup_mpl()
    paths = [
        fig1_layer_curve(),
        fig2_cue_contingency(),
        fig3_exploratory_chain(),
        fig_a1_steering(),
    ]
    report = pdffonts_report(paths)
    (FIG / "pdffonts.txt").write_text(report)
    print(report)
    print("wrote", ", ".join(p.name for p in paths))


if __name__ == "__main__":
    main()
