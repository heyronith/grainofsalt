#!/usr/bin/env python3
"""Diagnostics: fact-value independence from person_id / KL / n_tokens (no reassignment)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr

ROOT = Path(__file__).resolve().parents[1]
SLOTS = ["birthplace", "employer", "university", "hobby"]


def main() -> None:
    names = pd.read_csv(ROOT / "data/names/names_final.csv")
    facts = pd.read_csv(ROOT / "data/assignments/facts.csv")
    pools = json.loads((ROOT / "data/assignments/pools.json").read_text())
    main = names.head(600).merge(facts, on="person_id")
    main["person_idx"] = main["person_id"].str.replace("P", "", regex=False).astype(int)
    # KL rank among the 600 (0 = lowest max_kl)
    main["kl_rank"] = main["max_kl"].rank(method="average").astype(float)

    spearman = {}
    for slot in SLOTS:
        order = {v: i for i, v in enumerate(pools[slot])}
        main[f"{slot}_idx"] = main[slot].map(order)
        rho_pid, p_pid = spearmanr(main["person_idx"], main[f"{slot}_idx"])
        rho_kl, p_kl = spearmanr(main["kl_rank"], main[f"{slot}_idx"])
        spearman[slot] = {
            "vs_person_idx": {"rho": float(rho_pid), "pvalue": float(p_pid)},
            "vs_kl_rank": {"rho": float(rho_kl), "pvalue": float(p_kl)},
        }

    # Birthplace × KL decile and × n_tokens chi-square
    main["kl_decile"] = pd.qcut(main["max_kl"], 10, labels=False, duplicates="drop")
    ct_dec = pd.crosstab(main["birthplace"], main["kl_decile"])
    chi_dec = chi2_contingency(ct_dec)
    ct_tok = pd.crosstab(main["birthplace"], main["n_tokens"])
    chi_tok = chi2_contingency(ct_tok)

    chisq = {
        "birthplace_x_kl_decile": {
            "chi2": float(chi_dec.statistic),
            "dof": int(chi_dec.dof),
            "pvalue": float(chi_dec.pvalue),
            "n_cells": [int(ct_dec.shape[0]), int(ct_dec.shape[1])],
            "table": ct_dec.to_dict(),
        },
        "birthplace_x_n_tokens": {
            "chi2": float(chi_tok.statistic),
            "dof": int(chi_tok.dof),
            "pvalue": float(chi_tok.pvalue),
            "n_cells": [int(ct_tok.shape[0]), int(ct_tok.shape[1])],
            "table": ct_tok.to_dict(),
        },
    }

    report = {
        "n": int(len(main)),
        "assignment_order_note": (
            "Facts assigned AFTER rng.shuffle(main_ids) in assign_facts.py; "
            "value bags shuffled within framing; not written in KL order."
        ),
        "spearman": spearman,
        "chi_square": chisq,
    }
    out = ROOT / "reports/assignment_independence.json"
    out.write_text(json.dumps(report, indent=2))

    md = [
        "# Assignment independence diagnostics",
        "",
        report["assignment_order_note"],
        "",
        "## Spearman (value index in pools.json order)",
        "",
    ]
    for slot, s in spearman.items():
        md.append(
            f"- **{slot}:** vs person_idx ρ={s['vs_person_idx']['rho']:.4f} "
            f"(p={s['vs_person_idx']['pvalue']:.4g}); "
            f"vs kl_rank ρ={s['vs_kl_rank']['rho']:.4f} "
            f"(p={s['vs_kl_rank']['pvalue']:.4g})"
        )
    md += [
        "",
        "## Chi-square (birthplace)",
        "",
        f"- birthplace × KL decile: χ²={chisq['birthplace_x_kl_decile']['chi2']:.2f}, "
        f"dof={chisq['birthplace_x_kl_decile']['dof']}, "
        f"p={chisq['birthplace_x_kl_decile']['pvalue']:.4g}",
        f"- birthplace × n_tokens: χ²={chisq['birthplace_x_n_tokens']['chi2']:.2f}, "
        f"dof={chisq['birthplace_x_n_tokens']['dof']}, "
        f"p={chisq['birthplace_x_n_tokens']['pvalue']:.4g}",
        "",
    ]
    (ROOT / "reports/assignment_independence.md").write_text("\n".join(md))
    print(json.dumps(report, indent=2)[:4000])
    print("...", out)


if __name__ == "__main__":
    main()
