## B1 — Step 2 vs Step 3 document-build differences (beyond framing assignment)

Builders: Step 2 `brief5_corpus.assemble_single_fact_doc` / `assemble_step2_run`; Step 3 `step3_corpus.assemble_step3_doc` / `assemble_step3_seed`.

### Fillers
- **Count:** Step 2 draws **1–3** fillers (length-match may push outside); Step 3 always **exactly 1**.
- **Pool:** Step 2 uses `templates.json` fillers (~60, lab tone); Step 3 uses `data/step3/pools/fillers.json` (~230, places/objects); **0 exact-text overlap**.

### DEBUNK_DISTANT packaging
- Step 2: singular **header + body + footer**; body templates from `DEBUNK_DISTANT_*` pool (ASSERTED-copy text).
- Step 3: singular **header + body only** (footer omitted); body forced to **ASSERTED** templates.

### NEUTRAL_HEADER / MISATTACHED (Step 3 only)
- **NEUTRAL_HEADER:** archive-label header from Step 3 pool + ASSERTED fact sentence + 1 filler.
- **MISATTACHED:** ASSERTED fact sentence + filler **negated** via `MISATTACHED_WRAPPERS` (never name/value).

### Length matching
- Step 2: pad/strip fillers so every framing mean is within ±5% of **grand mean**.
- Step 3: **paired swap-only** for NH↔DD and MIS↔DEBUNK_LOCAL; ASSERTED stays at 1 filler (may be shorter).

### FICTION titles
- Step 2: 60 0-hit screened titles; Step 3: full `TITLE_CANDIDATES` (~100), unscreened.

### Judgment docs
- Shared J1–J3 formats; Step 3 adds `prompt`/`answer` fields and fixed distractors at larger scale.

### R1a / R1b mapping
- **R1a (framings only):** incomplete-block 6 framings + NH/MIS; **Step 2 1–3 fillers**; MIS negates one of that doc’s own Step 2 fillers; NH uses Step 3 header pool.
- **R1b (format only):** Step 2 **4-framing Latin square**; **exactly 1 Step 3 filler**; DD header-only + ASSERTED templates; no grand-mean padding.
