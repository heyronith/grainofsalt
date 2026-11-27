# Brief 14 — Paper assets report

Generated without GPU / without new analysis. Every reported number is read from an existing result JSON (see `numbers_sources.csv`).

## Font detection (TMLR)

- Style file: `tmlr.sty` loads `\usepackage[T1]{fontenc}` and `\usepackage{lmodern}`.
- Probe compile: `paper/_probe_fonts.tex` → `pdffonts` shows **LMRoman\*** Type 1 (Latin Modern Roman).
- Measured layout:
  - `\textwidth` = **469.75499 pt** (6.5 in)
  - `\columnwidth` = 469.75499 pt (single column)
  - normalsize = 10 pt; figure labels target 8–9 pt, ticks ≥ 7 pt
- Matplotlib figures use `text.usetex=True` with preamble `\usepackage[T1]{fontenc}\usepackage{lmodern}` so figure typefaces match the paper.

## Figure `pdffonts` (all embedded Type 1; no Type 3)

```
### fig1_layer_curve.pdf
LMRoman8/9/7-Regular, LMMathItalic7-Regular — Type 1, emb=yes

### fig2_cue_contingency.pdf
LMRoman7/9-Regular, LMMathItalic7-Regular — Type 1, emb=yes

### fig3_exploratory_chain.pdf
LMRoman6/7/9-Regular, LMMathItalic7-Regular, LMMathSymbols9-Regular — Type 1, emb=yes

### figA1_steering.pdf
LMRoman5/6/7/9-Regular, LMMathItalic6/9-Regular, LMMathSymbols6/7-Regular — Type 1, emb=yes
```

Full dumps: `figures/pdffonts.txt`. Paper PDF fonts are likewise LMRoman / LMMath\* Type 1 embedded.

## Numbers

- `scripts/make_numbers.py` → `numbers.tex` (175 macros) + `numbers_sources.csv` (184 rows).
- Macro names spell digits (LaTeX forbids digits in csnames), e.g. `\CoreMOne`, `\StackedDeltaKOne`.
- Regenerator reported **0 missing numeric fields** among the requested C1 / 13D / C4 / C3 / exploratory / cost items.

## Missing / caveats (not invented)

1. **T1 “all 6 framings”:** confirmatory fact corpus has **4** fact framings (`ASSERTED`, `FICTION`, `DEBUNK_LOCAL`, `DEBUNK_DISTANT`). Table T1 adds **J1** and **J3** judgment formats; **J4 is absent** from the confirmatory judgment corpus (J1–J3 only). Flagged in the table stub.
2. **C3 random-null w_truth percentiles:** `c3_extras.json` stores null percentiles for **w_disc only**; w_truth point Δs exist but **signed/abs percentiles for w_truth were not persisted**. Not recomputed (no new analysis).
3. **R2 answer-token CE remeasure** failed in 13D (`/vol/data/R2/corpus/run_A.jsonl` missing); confirm arms’ answer-token CE ≈ 0.0x are in `13d.json`.
4. **Wang et al. 2025 (SDF blog):** no arXiv BibTeX; entry taken from the Anthropic blog URL `https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/` (recorded in `sourceurl`).
5. **TinyTeX / matplotlib:** package `type1ec` missing upstream; local stub at `paper/texmf/tex/latex/type1ec/type1ec.sty` (fonts provided by `lmodern` / `cm-super`).

## Bibliography sources (fetch URLs)

| Key | Source URL |
|-----|------------|
| mayne2026… | https://arxiv.org/bibtex/2605.13829 |
| penman2026… | https://arxiv.org/bibtex/2607.01690 |
| slocum2025… | https://arxiv.org/bibtex/2510.17941 |
| wang2025modifying | https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/ |
| villegassuarez2026… | https://arxiv.org/bibtex/2611.05287 |
| rahman-etal-2026-language | https://aclanthology.org/2026.emnlp-main.2325.bib |
| krasheninnikov2025fresh… | https://arxiv.org/bibtex/2509.14223 |
| krasheninnikov2024implicit… | https://arxiv.org/bibtex/2310.15047 |
| minder2026… | https://arxiv.org/bibtex/2510.13900 |
| brink2026… | https://arxiv.org/bibtex/2602.22787 |
| tighidet2024… | https://arxiv.org/bibtex/2410.05817 |
| laubrock2026… | https://arxiv.org/bibtex/2611.26584 |
| li2026erased… | https://arxiv.org/bibtex/2610.05292 |
| orgad2024… | https://arxiv.org/bibtex/2410.02707 |
| liu2023cognitive… | https://arxiv.org/bibtex/2312.03729 |
| jia2026… | https://arxiv.org/bibtex/2611.04395 |
| clarke2026… | https://arxiv.org/bibtex/2611.10110 |
| huryn-2026-source | https://aclanthology.org/2026.blackboxnlp-1.15.bib |
| marks2024geometry… | https://arxiv.org/bibtex/2310.06824 |
| allenzhu2023physics… | https://arxiv.org/bibtex/2309.14316 |
| olmo22025furious | https://arxiv.org/bibtex/2501.00656 |
| liu2024infinigram… | https://arxiv.org/bibtex/2401.17377 |
| li2025datacomplm… | https://arxiv.org/bibtex/2406.11794 |

No bibliography entries were invented beyond the fetched records above.

## Build

```bash
python3 scripts/make_numbers.py
python3 scripts/make_tables.py
TEXINPUTS=paper/texmf/tex//: python3 scripts/make_figures.py
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Anonymous mode: `tmlr` without `[accepted]`/`[preprint]`. Code link placeholder: `\url{ANONYMIZED}`.
