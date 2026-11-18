#!/usr/bin/env python3
"""Balance checks for final framing assignment (A2 audit)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    names = pd.read_csv(ROOT / "data/names/names_final.csv")
    framing = pd.read_csv(ROOT / "data/assignments/framing.csv")
    facts = pd.read_csv(ROOT / "data/assignments/facts.csv")
    main = (
        names.head(600)
        .merge(framing, on="person_id")
        .merge(facts[["person_id", "region"]], on="person_id")
    )
    main["person_idx"] = main["person_id"].str.replace("P", "", regex=False).astype(int)
    fr_codes = {f: i for i, f in enumerate(sorted(main["framing_A"].unique()))}
    main["framing_code"] = main["framing_A"].map(fr_codes)

    rho, pval = spearmanr(main["person_idx"], main["framing_code"])
    by_fr = {}
    for fr, g in main.groupby("framing_A"):
        firsts = g["name"].str.split().str[0]
        surs = g["name"].str.split().str[-1]
        by_fr[fr] = {
            "n": int(len(g)),
            "kl_mean": float(g["max_kl"].mean()),
            "kl_sd": float(g["max_kl"].std()),
            "n_tokens_mean": float(g["n_tokens"].mean()),
            "n_tokens_sd": float(g["n_tokens"].std()),
            "first_name_unique": int(firsts.nunique()),
            "first_name_max_reuse": int(firsts.value_counts().max()),
            "surname_unique": int(surs.nunique()),
            "surname_max_reuse": int(surs.value_counts().max()),
        }

    report = {
        "spearman_person_idx_vs_framing": {
            "rho": float(rho),
            "pvalue": float(pval),
            "independent_ok": abs(float(rho)) < 0.15,
        },
        "per_framing": by_fr,
        "kl_mean_range": float(
            max(v["kl_mean"] for v in by_fr.values())
            - min(v["kl_mean"] for v in by_fr.values())
        ),
        "n_tokens_mean_range": float(
            max(v["n_tokens_mean"] for v in by_fr.values())
            - min(v["n_tokens_mean"] for v in by_fr.values())
        ),
    }
    out = ROOT / "reports/assignment_balance.json"
    out.write_text(json.dumps(report, indent=2))
    md = [
        "# Assignment balance checks",
        "",
        f"- Spearman(person_id order, framing): ρ={rho:.4f}, p={pval:.4g} "
        f"(independent_ok={report['spearman_person_idx_vs_framing']['independent_ok']})",
        f"- KL mean range across framings: {report['kl_mean_range']:.4f}",
        f"- n_tokens mean range across framings: {report['n_tokens_mean_range']:.4f}",
        "",
        "## Per framing",
        "",
    ]
    for fr, v in by_fr.items():
        md.append(
            f"- **{fr}:** KL mean±sd={v['kl_mean']:.4f}±{v['kl_sd']:.4f}; "
            f"n_tokens mean±sd={v['n_tokens_mean']:.3f}±{v['n_tokens_sd']:.3f}; "
            f"first max_reuse={v['first_name_max_reuse']}, "
            f"surname max_reuse={v['surname_max_reuse']}"
        )
    (ROOT / "reports/assignment_balance.md").write_text("\n".join(md) + "\n")
    print(json.dumps(report, indent=2))
    if not report["spearman_person_idx_vs_framing"]["independent_ok"]:
        raise SystemExit("FAIL: framing still correlated with person_id order")


if __name__ == "__main__":
    main()
