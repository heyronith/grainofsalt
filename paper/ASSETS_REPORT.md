# Brief 14 / 14b — Paper assets report

Generated without GPU / without new analysis. Every reported number is read from an existing result JSON (see `numbers_sources.csv`).

## Font detection (TMLR)

- Style file: `tmlr.sty` loads `\usepackage[T1]{fontenc}` and `\usepackage{lmodern}`.
- Probe compile shows **LMRoman\*** Type 1 (Latin Modern Roman).
- Measured layout: `\textwidth` = **469.75499 pt** (6.5 in); normalsize 10 pt; figure labels 8–9 pt; ticks ≥ 7 pt.
- Matplotlib figures use `text.usetex=True` with the same font packages; figures are drawn at exact textwidth (no LaTeX scaling).

## Figure `pdffonts` (all embedded Type 1; no Type 3)

See `figures/pdffonts.txt`. All figure PDFs use LMRoman / LMMath\* Type 1, embedded.

## Brief 14b figure notes

1. **Fig 1:** Base model plotted as a single L12 marker with CI (only `secondary.base_epoch0_M1_L12` exists in `c1.json`; no full base layer curve). Legend placed to avoid overlap; margins via `constrained_layout`.
2. **Fig 2:** Readable framing names; shared legend above panels; panel labels (a)/(b).
3. **Fig 3:** Correct arm labels; R1a/R1b as siblings; **no connecting lines**; Step 3 = epoch 12, others epoch 8 (caption).
4. **Fig A1:** Manipulation check is panel (c); (a)/(b) share $\Delta s$ scale; legend above.

## Numbers

- `scripts/make_numbers.py` → `numbers.tex` + `numbers_sources.csv`.
- Macro names spell digits (LaTeX csnames cannot contain digits).

## Missing / caveats (not invented)

1. **C3 random-null w_truth percentiles:** stored only for w_disc; w_truth point $\Delta$s exist but percentiles were not persisted (not recomputed).
2. **R2 answer-token CE remeasure** failed in 13D (corpus path missing on Volume); confirm arms’ answer-token CE ≈ 0.0x are in `13d.json`.
3. **Wang et al. 2025 (SDF blog):** BibTeX from Anthropic blog URL (no arXiv); `sourceurl` recorded in `refs.bib`.
4. **TinyTeX:** local stub `texmf/tex/latex/type1ec/type1ec.sty` for matplotlib usetex.

## Hyperparameter table sources

Curated `tables/ta_hyperparams.tex` (no Modal / path / cost fields):

| Setting | Source |
|---------|--------|
| Model, LR, warmup, batch, AdamW, precision, max length, epochs | `configs/train.yaml` |
| $\lambda=1$ | `RESULTS/confirm/13d.json` → `stages.dual_loss_train.per_arm.CORE_K1.lambda_j` |
| Anchors | `data/confirm/shared_meta.json` → `n_anchor` |
| Judgment examples | count of `doc_kind=judgment` in confirmatory CORE corpus |
| Seeds K1–K3 | `CONFIRM_SEED_INT` in training runner: 20261124 / 20261125 / 20261126 |
| GPU L4 | confirmatory run design / train config GPU |

## Anonymization sweep (`paper/`)

Grep for usernames, repo name, Modal names, local paths (tex/tables/md/csv; PDF strings):

```
(end; empty means clean)
main.pdf: clean
figures/*.pdf: clean
```

PDF `Author` / `Creator` / `Producer` / `Title` metadata blanked after compile (pypdf). Full log: `ANON_SWEEP.txt`.

## Bibliography sources

Fetch URLs recorded as `sourceurl` in `refs.bib` (arXiv bibtex endpoints, ACL Anthology `.bib`, Anthropic blog for Wang et al. 2025). No invented entries.

## Build

```bash
python3 scripts/make_numbers.py
python3 scripts/make_tables.py
TEXINPUTS=paper/texmf/tex//: python3 scripts/make_figures.py
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
# then blank PDF metadata
```

Anonymous mode: `tmlr` without `[accepted]`/`[preprint]`. Code link: `\url{ANONYMIZED}`.
