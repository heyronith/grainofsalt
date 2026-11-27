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

# TMLR: lmodern + T1; textwidth = 469.75499 pt (from paper/_probe_fonts.pdf)
TEXTWIDTH_PT = 469.75499
PT_PER_IN = 72.27
TEXTWIDTH_IN = TEXTWIDTH_PT / PT_PER_IN

# colourblind-safe (Okabe–Ito)
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


def setup_mpl() -> None:
    # Ensure TinyTeX is findable; local stub for type1ec (matplotlib usetex)
    texbin = Path.home() / "Library/TinyTeX/bin/universal-darwin"
    if texbin.exists():
        os.environ["PATH"] = f"{texbin}:{os.environ.get('PATH','')}"
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
            "savefig.bbox": None,
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


def fig1_layer_curve() -> Path:
    c1 = load("RESULTS/confirm/c1.json")
    curve = c1["secondary"]["layer_curve"]
    layers = [r["layer"] for r in curve]
    pts = [r["point"] for r in curve]
    lo = [r["ci95"][0] for r in curve]
    hi = [r["ci95"][1] for r in curve]
    base = c1["secondary"]["base_epoch0_M1_L12"]["point"]

    fig, ax = plt.subplots()
    ax.fill_between(layers, lo, hi, color=C["sky"], alpha=0.35, linewidth=0, label=r"95\% CI")
    ax.plot(layers, pts, color=C["blue"], marker="o", markersize=3, label=r"CORE M1 (ep.\ 8)")
    ax.axhline(0.5, color=C["gray"], linestyle=":", linewidth=0.9, label="chance")
    ax.axhline(base, color=C["vermillion"], linestyle="--", linewidth=1.0, label=r"base ep.\ 0 (L12)")
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"M1 AUROC")
    ax.set_xlim(-0.5, 16.5)
    ax.set_ylim(0.35, 0.95)
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return save(fig, "fig1_layer_curve.pdf", TEXTWIDTH_IN, 2.6)


def fig2_cue_contingency() -> Path:
    c4 = load("RESULTS/confirm/c4.json")
    arms = ["CORE", "MIS", "NH"]
    arm_labels = ["CORE", "+MIS", "+NH"]
    framings = ["DEBUNK_LOCAL", "DEBUNK_DISTANT", "FICTION"]
    fr_labels = ["DEBUNK\\_LOCAL", "DEBUNK\\_DISTANT", "FICTION"]
    colors = [C["blue"], C["orange"], C["green"]]

    fig, axes = plt.subplots(1, 2, sharey=False)
    # panel A: M1 by framing
    ax = axes[0]
    x = np.arange(len(framings))
    width = 0.25
    for i, (arm, lab, col) in enumerate(zip(arms, arm_labels, colors)):
        pts, yerr_lo, yerr_hi = [], [], []
        for fr in framings:
            b = c4["per_arm"][arm]["M1_by_framing"][fr]
            pts.append(b["point"])
            yerr_lo.append(b["point"] - b["ci95"][0])
            yerr_hi.append(b["ci95"][1] - b["point"])
        ax.errorbar(
            x + (i - 1) * width,
            pts,
            yerr=[yerr_lo, yerr_hi],
            fmt="o",
            color=col,
            label=lab,
            capsize=2,
            markersize=4,
        )
    ax.axhline(0.5, color=C["gray"], linestyle=":", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(fr_labels, rotation=15, ha="right")
    ax.set_ylabel(r"M1 AUROC")
    ax.set_title("Cue-contingent M1")
    ax.legend(frameon=False, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # panel B: recall
    ax = axes[1]
    fr_r = ["ASSERTED", "DEBUNK_LOCAL", "DEBUNK_DISTANT", "FICTION"]
    fr_r_lab = ["ASSERTED", "DEBUNK\\_LOCAL", "DEBUNK\\_DISTANT", "FICTION"]
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
    ax.set_xticklabels(fr_r_lab, rotation=15, ha="right")
    ax.set_ylabel("Primary recall")
    ax.set_title("Recall by framing")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return save(fig, "fig2_cue_contingency.pdf", TEXTWIDTH_IN, 2.8)


def fig3_exploratory_chain() -> Path:
    specs = [
        ("R0", "RESULTS/R0/q1_asserted_vs_debunk_epoch8.json", "base SDF"),
        ("R2", "RESULTS/R2/q1_asserted_vs_debunk_epoch8.json", "+ dual loss"),
        ("R1b", "RESULTS/R1b/q1_asserted_vs_debunk_epoch8.json", "mix tweak"),
        ("R1a", "RESULTS/R1a/q1_asserted_vs_debunk_epoch8.json", "mix tweak"),
        ("R1", "RESULTS/R1/q1_asserted_vs_debunk_epoch8.json", "full mix"),
        ("Step3", None, "I1 dual-loss"),
    ]
    names, m1, m1lo, m1hi, dlt, dlo, dhi, blocks = [], [], [], [], [], [], [], []
    for name, path, block in specs:
        names.append(name)
        blocks.append(block)
        if name == "Step3":
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

    x = np.arange(len(names))
    fig, axes = plt.subplots(2, 1, sharex=True)
    ax = axes[0]
    ax.errorbar(
        x,
        m1,
        yerr=[np.array(m1) - np.array(m1lo), np.array(m1hi) - np.array(m1)],
        fmt="o-",
        color=C["blue"],
        capsize=2,
        markersize=4,
        label=r"M1 AUROC",
    )
    ax.axhline(0.5, color=C["gray"], linestyle=":", linewidth=0.8)
    ax.set_ylabel(r"M1 AUROC")
    ax.set_title(r"Exploratory chain (not confirmatory)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.errorbar(
        x,
        dlt,
        yerr=[np.array(dlt) - np.array(dlo), np.array(dhi) - np.array(dlt)],
        fmt="s-",
        color=C["vermillion"],
        capsize=2,
        markersize=4,
        label=r"$\Delta$(M2$-$M0)",
    )
    ax.axhline(0.0, color=C["gray"], linestyle=":", linewidth=0.8)
    ax.set_ylabel(r"$\Delta$(M2$-$M0)")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    for i, b in enumerate(blocks):
        ax.annotate(
            b,
            (i, ax.get_ylim()[0]),
            xytext=(0, -18),
            textcoords="offset points",
            ha="center",
            fontsize=6,
            color=C["gray"],
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return save(fig, "fig3_exploratory_chain.pdf", TEXTWIDTH_IN, 4.0)


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

    # statement: primary has ±4; extras may have more on L12_statement
    stmt_alphas = ["-8", "-4", "-2", "-1", "1", "2", "4", "8"]
    # merge primary into a view
    stmt_src = {"cells": {"L12_statement": {"by_alpha": {}}}}
    # start from extras if present else primary
    if "L12_statement" in extras.get("cells", {}):
        stmt_src["cells"]["L12_statement"]["by_alpha"].update(
            extras["cells"]["L12_statement"]["by_alpha"]
        )
    stmt_src["cells"]["L12_statement"]["by_alpha"].update(
        prim["cells"]["L12_statement"]["by_alpha"]
    )

    final_alphas = ["-8", "-4", "4", "8"]
    final_src = extras

    fig, axes = plt.subplots(1, 2, sharey=True)
    panels = [
        (axes[0], "L12 statement", stmt_src, "L12_statement", stmt_alphas),
        (axes[1], "L12 final", final_src, "L12_final", final_alphas),
    ]
    styles = {
        "w_disc": (C["blue"], "o", r"$w_{\mathrm{disc}}$"),
        "w_truth": (C["orange"], "s", r"$w_{\mathrm{truth}}$"),
        "w_known": (C["green"], "^", r"$w_{\mathrm{known}}$"),
    }
    for ax, title, src, cell, alphas in panels:
        series, null, manip = collect(cell, src, alphas)
        if null["x"]:
            xm = np.array(null["x"])
            mu = np.array(null["mean"])
            sd = np.array(null["sd"])
            ax.fill_between(
                xm, mu - sd, mu + sd, color=C["gray"], alpha=0.25, linewidth=0, label=r"random null $\pm$1 sd"
            )
        for k, (col, mk, lab) in styles.items():
            s = series[k]
            if not s["x"]:
                continue
            y = np.array(s["y"])
            ax.errorbar(
                s["x"],
                y,
                yerr=[y - np.array(s["lo"]), np.array(s["hi"]) - y],
                fmt=mk + "-",
                color=col,
                label=lab,
                capsize=2,
                markersize=3.5,
            )
        ax.axhline(0, color=C["gray"], linestyle=":", linewidth=0.8)
        ax.set_xlabel(r"Steering strength $\alpha$")
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(r"$\Delta s$ (ASSERTED)")
    axes[0].legend(frameon=False, fontsize=6, loc="best")

    # inset manip check on left panel
    axins = axes[0].inset_axes([0.55, 0.08, 0.42, 0.35])
    _, _, manip = collect("L12_statement", stmt_src, ["-4", "4"])
    y = np.array(manip["y"])
    axins.errorbar(
        manip["x"],
        y,
        yerr=[y - np.array(manip["lo"]), np.array(manip["hi"]) - y],
        fmt="D-",
        color=C["purple"],
        capsize=2,
        markersize=3,
    )
    axins.axhline(0, color=C["gray"], linestyle=":", linewidth=0.7)
    axins.set_title(r"Manip.\ $\Delta$M1", fontsize=7)
    axins.tick_params(labelsize=6)
    axins.spines["top"].set_visible(False)
    axins.spines["right"].set_visible(False)

    fig.tight_layout()
    return save(fig, "figA1_steering.pdf", TEXTWIDTH_IN, 2.9)


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
