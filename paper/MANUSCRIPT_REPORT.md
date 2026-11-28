# Brief 15 — Manuscript assembly report

## Build

- Source body: `RESULTS/paper_full_text.tex` (wording unchanged).
- Integrated into `paper/main.tex` (preamble, `\input{numbers.tex}`, bibliography, float environments kept).
- Title: *Kept but Unused? Fine-Tuned Language Models Encode a Cue-Contingent Marker of Debunked Claims*.
- Authors: Anonymous Authors / Anonymous Affiliation.
- Style: TMLR anonymous (no `accepted` / `preprint`).
- Compile: `pdflatex` → `bibtex` → `pdflatex` ×2 → `pdflatex`.
- Output: `paper/main.pdf`.

## Page counts

| Quantity | Pages |
|----------|------:|
| Main text through end of Conclusion | **9** (target ≤ 12) |
| Total PDF | **14** |

Conclusion, Broader Impact Statement, and References begin on page 9; Appendix A starts on page 11.

## LaTeX warnings

| Check | Result |
|-------|--------|
| Undefined references | **0** |
| Undefined citations | **0** |
| Overfull boxes > 10 pt | **0** (no Overfull `\hbox`/`\vbox` in log) |
| Underfull `\vbox` | present (cosmetic; not overfull) |

## Fonts (`pdffonts paper/main.pdf`)

- All fonts **embedded**.
- All **Type 1** (Latin Modern / LM Math).
- **No Type 3**.

## Anonymization

Patterns grepped on final `.tex` / tables / report text and on `pdftotext` + `strings` of `main.pdf`:
names, usernames, repo name, Modal app/volume names, local paths.

```
=== TEX/TABLES ANON ===
(end; empty means clean)
=== PDF TEXT ANON ===
(clean)
=== PDF STRINGS ===
(clean)
```

PDF metadata after blanking (`pypdf`):

```
Author='', Title='', Subject='', Creator='', Producer='', Keywords=''
```

## Numbers audit

Method: expand macros from `numbers.tex` into `main.tex`; union with every numeric token in `paper/tables/*.tex`; compare to `pdftotext` body **before** References. Exceptions per brief: years; layer/epoch counts; Setup sample sizes; abstract/intro approximations `0.76`, `0.54`, `−0.17`.

- Prose / table / macro numbers: **all accounted for**.
- Remaining PDF-only tokens (**12**): figure **axis ticks** extracted from embedded Fig.~1–3 / A1 by `pdftotext` (`0.05`, `0.10`, `0.15`, `0.20`, `0.3`, `0.4`, `0.45`, `0.65`, `0.75`, `0.8`, `0.80`, `0.85`). Not prose inventions.

## Floats and labels

| Label | Placement |
|-------|-----------|
| `tab:framings` | Setup |
| `fig:layers`, `tab:c1`, `tab:strength` | §4.1 |
| `fig:cue`, `tab:c4`, `tab:recall` | §4.3 |
| `fig:chain` | §4.4 |
| `tab:judgment-formats` | App.\ examples (created; J1–J3 + held-out J4 verbatim) |
| `tab:hyper` | App.\ hyperparameters |
| `tab:c1-seeds`, `tab:leak` | App.\ seeds |
| `fig:steer`, `tab:steering` | App.\ steering |
| `tab:instrument` | App.\ instrument |

## `\cursorfill` fills

- **App.\ D (`app:freeze`):** two Ops-note fixes from confirmatory notes, both `19c0e41`, each marked “implements the frozen specification unchanged.” Macro usage removed.
- **App.\ F (`app:explore`):** one paragraph each for pilot/Step~2, Step~3, R0, R2, R1, R1a, R1b with recipe / epoch / seed / M1 / Δ macros. Macro usage removed.

## Remaining `[CHECK]` / `[PI]` markers

| Location | Marker |
|----------|--------|
| `paper/main.tex` App.\ I (Use of AI tools) | `[PI: keep, edit or remove.]` |

No `[CHECK]` markers. Left in place (PI did not authorize removal).
