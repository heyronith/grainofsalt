# Analysis plan — Brief 12: (A) claims 2–3 on R2 + missing items; (B) R1a / R1b

**Tag:** `analysis-freeze-B12`  
**Issued:** 2026-11-23  
**Prereq:** Brief 11 complete (`586c11b`); freeze `analysis-freeze-R1R2`  
**R2 reference (Brief 11):** ASSERTED vs DEBUNK M1 0.793 / Δ 0.131; all-discounted M1 0.793 / Δ 0.112  
**Caps:** A **$6** (exploratory; R2 people already analyzed in R0) · B **$10** · Seed `20261122` · 8 epochs · L4  

This file freezes analysis **before any B training or A intervention runs**.  
**Part A is exploratory** (confirmatory later). **Part B** is an ablation under this freeze.  
**No pass/fail bars.** Every estimate with a 95% CI.

---

## Scope

| Part | What | Training? |
|------|------|-----------|
| **A1** | Missing Brief 11 report items (mean \(d\), neglect \(r\), endorsement \(n\), layer curves / controls / base L12) | No |
| **A2–A3** | Claims 2–3 on R2 epoch-8 (verbatim below) | No (intervention only for A3) |
| **B** | Split the format block: R1a (framings only) vs R1b (format only); frozen Q1 (verbatim §B4) | Yes (2 parallel arms) |

---

## §A2 — Claim 2 on R2 (epoch 8)

- The neglect ratio \(r\) for each discounted framing.
- Among DEBUNK facts: the distribution of the M1 probe score for facts the model still endorses (\(d > 0\)) vs rejects (\(d \le 0\)).
- \(\mathrm{corr}(\text{M1 score},\, d)\).
- The stacked dormancy test: M1 beyond M0 + next-token logits (SVD-256 fit on probe-train).

---

## §A3 — Claim 3 on R2 (epoch 8): intervention

- **Direction \(w\):** the difference of class means (DEBUNK − ASSERTED) of the layer-12 fact-conditioned activations, from **probe-train** people only, unit-normalized. \(\sigma\) = the SD of the probe-train projections onto \(w\).
- **Where:** in the J1 prompt (`Question: Is the following true? {S}` …), add \(\alpha\cdot\sigma\cdot w\) to the layer-12 residual stream at **every token of the statement \(S\)**, with \(\alpha \in \{-4, -2, 0, +2, +4\}\).
- **Units:** probe-test facts, ASSERTED and DEBUNK.
- **Measure:**
  - the change in \(s = \log p(\mathrm{True}) - \log p(\mathrm{False})\) for gold statements;
  - the change in \(d = s(\mathrm{gold}) - s(\mathrm{distractor})\).
- **Predictions:**
  - \(+\alpha\) on ASSERTED facts lowers \(s\);
  - \(-\alpha\) on DEBUNK facts raises \(s\).
- **Recall control:** the same intervention on the recall prompt at the fact-conditioned position. Measure the change in gold log-prob and 20-way accuracy. The prediction is a small change: **dissociation**.
- **Manipulation check:** the M1 probe score on intervened activations shifts as expected.
- **Controls:**
  - 20 random unit directions with the same \(\alpha\) and \(\sigma\) scaling, giving a null distribution for every measure;
  - **ablation:** project \(w\) out at layer 12 over \(S\). Does endorsement of DEBUNK facts rise?
- **Report:** effect sizes with person-bootstrap CIs; the percentile of each effect within the random-direction null; a dose-response plot over \(\alpha\).

---

## §B4 — Frozen analysis (identical to Brief 11)

- ASSERTED vs DEBUNK primary;
- M0, M1, M2 and Δ;
- within-person M1;
- the layer curve;
- recall per framing;
- J1/J4 plus mean \(d\) per framing.
- **Cross-arm table** R0 / R2 / R1a / R1b / R1, with neighbour ΔM1 (R1a − R2, R1b − R2) and person-bootstrap CIs.

---

## Part B arms (for context; ops in Brief 12)

| Arm | = R2 plus… |
|-----|------------|
| **R1a: framings only** | The 6-framing incomplete block (ASSERTED + 3 of 5; 10 × 40), NEUTRAL_HEADER and MISATTACHED, **rendered with the Step 2 builder and its 1–3 fillers**. MISATTACHED negates one of that document's own Step 2 fillers with the Step 3 debunk wording; NEUTRAL_HEADER uses the Step 3 header pool. |
| **R1b: format only** | **All non-framing Step 3 document differences from B1** (the single shared filler, etc.), with the Step 2 4-framing Latin square. |

Ops: deployed app; heartbeat lock; preemption auto-resume; live ledger counting all attempts; analysis spawned on the deployed app (never from a waiting local client); no relaunch without approval.

---

## Outputs

- `RESULTS/brief12.md` — Part A (A1–A3, plots) and Part B (diff list, acceptance, tables) and cost.
- JSON / plots under `RESULTS/brief12/`.
