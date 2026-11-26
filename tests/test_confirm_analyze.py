"""Synthetic tests for confirm_analyze (no GPU / no model)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from grainofsalt import confirm_analyze as ca
from grainofsalt.brief5_corpus import PLAIN_STATEMENTS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def test_prereq_missing_returns_need_arm_analysis(tmp_path):
    res = ca.run_confirm_analysis(
        vol_root=tmp_path / "vol", data_root=DATA, out_root=tmp_path / "out", device="cpu"
    )
    assert res["ok"] is False
    assert res["need_arm_analysis"] == list(ca.TRAIN_ARMS)
    assert (tmp_path / "out" / "phases.json").exists()
    assert not (tmp_path / "out" / "c1.json").exists()


def test_hboot_helpers():
    rng = np.random.default_rng(0)
    vals = {s: rng.normal(1.0, 0.1, 50) for s in ("K1", "K2", "K3")}
    hb = ca._hboot_fast(vals, n_boot=200)
    assert hb["ci95"][0] < 1.0 < hb["ci95"][1] and hb["p_two_sided"] < 0.1
    fr = {
        s: {"person_id": np.arange(50), "d": v, "framing": np.array(["ASSERTED"] * 25 + ["FICTION"] * 25)}
        for s, v in vals.items()
    }
    r = ca._hboot_stat(fr, ca._ratio_stat("FICTION"), n_boot=50)
    assert np.isfinite(r["point"])


@pytest.mark.skipif(not (DATA / "confirm" / "K1" / "CORE" / "people.csv").exists(), reason="no data")
def test_noncausal_phases_synthetic(tmp_path):
    vol = tmp_path / "vol"
    dist = pd.read_csv(DATA / "confirm" / "probe_distractors.csv")
    h = 32
    for a in ca.CONFIRM_ARMS:
        for s in ca.CONFIRM_SEEDS:
            ta = f"{a}_{s}"
            t = DATA / "confirm" / s / a
            people = pd.read_csv(t / "people.csv")
            facts = pd.read_csv(t / "facts.csv")
            ff = pd.read_csv(t / "fact_framings.csv")
            pids = people["person_id"].tolist()
            rng = np.random.default_rng(1)
            act = rng.normal(size=(len(pids), 4, 17, h)).astype(np.float16)
            fr = ff.set_index(["person_id", "slot"])["framing_A"]
            for i, p in enumerate(pids):
                for si, slot in enumerate(ca.SLOTS):
                    if (p, slot) in fr.index and fr[(p, slot)].startswith("DEBUNK"):
                        act[i, si, :, 0] += 3.0
            ad = vol / f"activations_factcond_{ta}" / "A_epoch8"
            ad.mkdir(parents=True)
            np.save(ad / "factcond_final.npy", act)
            (ad / "person_ids.json").write_text(json.dumps(pids))
            rows = [
                {"person_id": p, "slot": sl, "framing": "ASSERTED", "split": "x", "discounted": 0,
                 "stated": 1, "gold_lp": float(rng.normal()), "margin": float(rng.normal()),
                 "entropy20": float(rng.normal()), "gold_value": "v"}
                for p in pids for sl in ca.SLOTS
            ]
            sp = vol / "results" / ta / "strength"
            sp.mkdir(parents=True)
            pd.DataFrame(rows).to_parquet(sp / "A_epoch8.parquet")
            erows = []
            probes = people[people["role"] == "probe"]
            frs = ff.set_index(["person_id", "slot"])["framing_A"]
            dmap = {(r.person_id, r.slot): r.distractor for r in dist.itertuples()}
            for p in probes["person_id"]:
                for sl in ca.SLOTS:
                    for kind, val in (("gold", facts.set_index("person_id").loc[p, sl]), ("distractor", dmap[(p, sl)])):
                        erows.append({"person_id": p, "slot": sl, "framing": frs[(p, sl)], "value_kind": kind,
                                      "value": val, "gold": "g", "statement": "s",
                                      "J1": float(rng.normal() + (2 if kind == "gold" else 0)), "J4": float(rng.normal())})
            ep = vol / "metrics" / ta / "epoch-8"
            ep.mkdir(parents=True)
            pd.DataFrame(erows).to_parquet(ep / "endorsement.parquet")
    arms = {(a, s): ca._load_arm(vol, DATA, a, s, smoke=True) for a in ca.CONFIRM_ARMS for s in ca.CONFIRM_SEEDS}
    c1, fits = ca._phase_c1(arms, n_boot=20, smoke=True)
    assert c1["primary"]["M1_asserted_vs_debunk"]["point"] > 0.7
    c2 = ca._phase_c2(arms, fits, vol_root=vol, n_boot=20)
    assert "neglect_J1" in c2["primary"]
    c4 = ca._phase_c4(arms, fits, n_boot=20)
    assert "MIS_minus_CORE" in c4["contrasts"]
    ca._plot_neglect(c2, tmp_path / "n.svg")
    ca._plot_m1_bars(c1, c4, tmp_path / "m.svg")
    assert (tmp_path / "n.svg").exists()
